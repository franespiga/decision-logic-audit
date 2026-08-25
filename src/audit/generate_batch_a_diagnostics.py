#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate validation-period diagnostic inputs for the Batch A audit pipeline.

Refactors the post-``1. Feature Importance`` workflow of
``notebooks/1_ablation_walkthrough_all_models.ipynb`` into a resumable,
non-visual, all-model analytical pipeline covering 1,728 individual models
across 18 families on the validation window 2016-01-01..2024-04-30.

Holdout outcomes (May 2024--June 2026) are never loaded.

Typical invocation::

    python -m audit.generate_batch_a_diagnostics \\
      --notebook notebooks/1_ablation_walkthrough_all_models.ipynb \\
      --model-assessment-pickle results/model_assessment.pkl \\
      --output-dir results/mendeley/dt_metrics/audit_inputs \\
      --validation-start 2016-01-01 \\
      --validation-end 2024-04-30 \\
      --sanity-sample-size 64 \\
      --n-jobs 1 \\
      --seed 42 \\
      --resume
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Support both `python -m audit.generate_batch_a_diagnostics` and direct script execution.
if __package__ in {None, ""}:
    _REPO = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(_REPO / "src"))
    from audit import compute_core as core  # type: ignore
    from audit import discovery  # type: ignore
else:
    from . import compute_core as core
    from . import discovery

LOGGER = logging.getLogger("batch_a_diagnostics")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NOTEBOOK = REPO_ROOT / "notebooks" / "1_ablation_walkthrough_all_models.ipynb"
DEFAULT_METRICS_DIR = REPO_ROOT / "results" / "mendeley" / "dt_metrics"
DEFAULT_OUTPUT_DIR = DEFAULT_METRICS_DIR / "audit_inputs"
DEFAULT_MODELS_PATH = Path(
    os.environ.get("DECISION_LOGIC_AUDIT_MODELS_PATH", str(REPO_ROOT / "models"))
).expanduser()
DEFAULT_MODEL_CODE_ROOT = Path(
    os.environ.get("DECISION_LOGIC_AUDIT_MODEL_CODE_ROOT", str(REPO_ROOT))
).expanduser()
DEFAULT_ASSESSMENT = Path(
    os.environ.get(
        "DECISION_LOGIC_AUDIT_MODEL_ASSESSMENT",
        str(REPO_ROOT / "results" / "model_assessment.pkl"),
    )
).expanduser()
EXPECTED_N_MODELS = 1728
EXPECTED_N_FAMILIES = 18

METRIC_MANIFEST: List[Dict[str, Any]] = [
    {
        "metric_id": "contribution_feature_rank_stability",
        "diagnostic_dimension": "contribution",
        "preferred_direction": "higher_is_better",
        "core_or_optional": "core_score",
        "validity_gate_or_score": "score",
        "source_function": "compute_contribution_diagnostics",
        "aggregation_method": "median_iqr_across_models",
        "definition": (
            "Mean pairwise Kendall-tau concordance of absolute feature-family "
            "attribution rankings across deterministic chronological windows."
        ),
        "architecture_scope": "differentiable_policy",
    },
    {
        "metric_id": "contribution_baseline_concordance",
        "diagnostic_dimension": "contribution",
        "preferred_direction": "higher_is_better",
        "core_or_optional": "core_score",
        "validity_gate_or_score": "score",
        "source_function": "compute_contribution_diagnostics",
        "aggregation_method": "median_iqr_across_models",
        "definition": (
            "Mean pairwise Kendall-tau concordance of feature-family rankings "
            "across zero, empirical-mean, and sampled-background baselines."
        ),
        "architecture_scope": "differentiable_policy",
    },
    {
        "metric_id": "contribution_signed_positive_mass",
        "diagnostic_dimension": "contribution",
        "preferred_direction": "descriptive",
        "core_or_optional": "descriptive",
        "validity_gate_or_score": "descriptive",
        "source_function": "compute_contribution_diagnostics",
        "aggregation_method": "median_iqr_across_models",
        "definition": "Share of absolute attribution mass from positive signed attributions.",
        "architecture_scope": "differentiable_policy",
    },
    {
        "metric_id": "contribution_signed_negative_mass",
        "diagnostic_dimension": "contribution",
        "preferred_direction": "descriptive",
        "core_or_optional": "descriptive",
        "validity_gate_or_score": "descriptive",
        "source_function": "compute_contribution_diagnostics",
        "aggregation_method": "median_iqr_across_models",
        "definition": "Share of absolute attribution mass from negative signed attributions.",
        "architecture_scope": "differentiable_policy",
    },
    {
        "metric_id": "contribution_net_attribution",
        "diagnostic_dimension": "contribution",
        "preferred_direction": "descriptive",
        "core_or_optional": "descriptive",
        "validity_gate_or_score": "descriptive",
        "source_function": "compute_contribution_diagnostics",
        "aggregation_method": "median_iqr_across_models",
        "definition": "Net signed attribution sum across feature families.",
        "architecture_scope": "differentiable_policy",
    },
    {
        "metric_id": "necessity_guided_deletion_advantage",
        "diagnostic_dimension": "necessity",
        "preferred_direction": "higher_is_better",
        "core_or_optional": "core_score",
        "validity_gate_or_score": "score",
        "source_function": "compute_deletion_diagnostics",
        "aggregation_method": "median_iqr_across_models",
        "definition": (
            "Ratio of mean action degradation under attribution-guided feature "
            "deletion relative to matched random deletion."
        ),
        "architecture_scope": "sequential_policy",
    },
    {
        "metric_id": "necessity_temporal_profile_stability",
        "diagnostic_dimension": "necessity",
        "preferred_direction": "higher_is_better",
        "core_or_optional": "core_score",
        "validity_gate_or_score": "score",
        "source_function": "compute_temporal_diagnostics",
        "aggregation_method": "median_iqr_across_models",
        "definition": (
            "Kendall-tau concordance of normalized temporal-occlusion profiles "
            "across chronological windows."
        ),
        "architecture_scope": "history_conditioned_policy",
    },
    {
        "metric_id": "necessity_recency_share",
        "diagnostic_dimension": "necessity",
        "preferred_direction": "descriptive",
        "core_or_optional": "descriptive",
        "validity_gate_or_score": "descriptive",
        "source_function": "compute_temporal_diagnostics",
        "aggregation_method": "median_iqr_across_models",
        "definition": "Share of temporal-occlusion mass in the most recent lags.",
        "architecture_scope": "history_conditioned_policy",
    },
    {
        "metric_id": "necessity_temporal_half_life",
        "diagnostic_dimension": "necessity",
        "preferred_direction": "descriptive",
        "core_or_optional": "descriptive",
        "validity_gate_or_score": "descriptive",
        "source_function": "compute_temporal_diagnostics",
        "aggregation_method": "median_iqr_across_models",
        "definition": "Lag index at which cumulative normalized occlusion mass reaches 0.5.",
        "architecture_scope": "history_conditioned_policy",
    },
    {
        "metric_id": "necessity_peak_lag",
        "diagnostic_dimension": "necessity",
        "preferred_direction": "descriptive",
        "core_or_optional": "descriptive",
        "validity_gate_or_score": "descriptive",
        "source_function": "compute_temporal_diagnostics",
        "aggregation_method": "median_iqr_across_models",
        "definition": "Lag of maximum mean normalized occlusion mass.",
        "architecture_scope": "history_conditioned_policy",
    },
    {
        "metric_id": "necessity_profile_concentration",
        "diagnostic_dimension": "necessity",
        "preferred_direction": "descriptive",
        "core_or_optional": "descriptive",
        "validity_gate_or_score": "descriptive",
        "source_function": "compute_temporal_diagnostics",
        "aggregation_method": "median_iqr_across_models",
        "definition": "Herfindahl-Hirschman index of the mean normalized occlusion profile.",
        "architecture_scope": "history_conditioned_policy",
    },
    {
        "metric_id": "structure_latent_neighbourhood_persistence",
        "diagnostic_dimension": "structure",
        "preferred_direction": "higher_is_better",
        "core_or_optional": "core_score",
        "validity_gate_or_score": "score",
        "source_function": "compute_structure_diagnostics",
        "aggregation_method": "median_iqr_across_models",
        "definition": (
            "Mean Jaccard overlap of latent-space nearest neighbours across "
            "adjacent layers and bootstrap subsets."
        ),
        "architecture_scope": "representation_exposing_policy",
    },
    {
        "metric_id": "structure_conductance_concentration",
        "diagnostic_dimension": "structure",
        "preferred_direction": "lower_is_better",
        "core_or_optional": "descriptive",
        "validity_gate_or_score": "descriptive",
        "source_function": "compute_structure_diagnostics",
        "aggregation_method": "median_iqr_across_models",
        "definition": "Herfindahl-Hirschman index of absolute layer-conductance shares.",
        "architecture_scope": "layered_differentiable_policy",
    },
    {
        "metric_id": "structure_conductance_inverse_concentration",
        "diagnostic_dimension": "structure",
        "preferred_direction": "higher_is_better",
        "core_or_optional": "core_score",
        "validity_gate_or_score": "score",
        "source_function": "compute_structure_diagnostics",
        "aggregation_method": "median_iqr_across_models",
        "definition": (
            "One minus the HHI of absolute layer-conductance shares; higher "
            "values indicate less pathological concentration."
        ),
        "architecture_scope": "layered_differentiable_policy",
    },
]

CORE_METRIC_IDS = {
    e["metric_id"] for e in METRIC_MANIFEST if e["core_or_optional"] == "core_score"
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Batch A contribution-necessity-structure diagnostics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--model-assessment-pickle", type=Path, default=DEFAULT_ASSESSMENT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--validation-start", type=str, default="2016-01-01")
    parser.add_argument("--validation-end", type=str, default="2024-04-30")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sanity-sample-size", type=int, default=64)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--analysis-repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--model-code-root", type=Path, default=DEFAULT_MODEL_CODE_ROOT)
    parser.add_argument("--models-path", type=Path, default=DEFAULT_MODELS_PATH)
    parser.add_argument("--metrics-dir", type=Path, default=DEFAULT_METRICS_DIR)
    parser.add_argument("--profile", choices=sorted(core.PROFILES), default="standard")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-models", type=int, default=None)
    parser.add_argument("--model-id", action="append", default=None)
    parser.add_argument("--skip-assessment-pickle", action="store_true")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser.parse_args(argv)


def configure_logging(output_dir: Path, level: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "generate_batch_a_diagnostics.log"
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )


def profile_from_args(args: argparse.Namespace) -> core.ProfileConfig:
    base = core.PROFILES[args.profile]
    n = max(8, int(args.sanity_sample_size))
    return core.ProfileConfig(
        attribution_samples=n,
        temporal_samples=max(8, n // 2),
        structure_samples=n,
        deletion_samples=max(8, n // 2),
        ig_steps=base.ig_steps,
        sanity_ig_steps=base.sanity_ig_steps,
        chronological_windows=base.chronological_windows,
        random_deletion_repeats=base.random_deletion_repeats,
        bootstrap_repeats=base.bootstrap_repeats,
        deletion_fraction=base.deletion_fraction,
        recent_lags=base.recent_lags,
        knn_k=base.knn_k,
    )


def config_hash(args: argparse.Namespace, profile: core.ProfileConfig) -> str:
    payload = {
        "validation_start": str(args.validation_start),
        "validation_end": str(args.validation_end),
        "profile": asdict(profile),
        "seed": args.seed,
        "sanity_sample_size": args.sanity_sample_size,
        "metric_manifest": METRIC_MANIFEST,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def preflight_paths(args: argparse.Namespace) -> List[Dict[str, Any]]:
    unresolved: List[Dict[str, Any]] = []
    checks = [
        (args.notebook, "notebook", "Notebook source", "heading / Feature Importance"),
        (args.models_path, "models_path", "Family checkpoints root", "Configuration / Model loading"),
        (
            args.metrics_dir / "holdout_metrics_individual.csv",
            "metrics_catalog",
            "1,728-model catalog",
            "Batch A metrics",
        ),
        (args.analysis_repo_root / "src", "analysis_src", "Ablation helpers", "imports"),
        (args.model_code_root, "model_code_root", "TrainableDT / data loaders", "imports"),
    ]
    if not args.skip_assessment_pickle:
        checks.append(
            (
                args.model_assessment_pickle,
                "model_assessment_pickle",
                "Optional per-model hyperparameter metadata",
                "Load portfolio weights",
            )
        )
    for path, obj, expected, cell in checks:
        if path is None or not Path(path).exists():
            unresolved.append(
                {
                    "path_or_object": str(path),
                    "object": obj,
                    "expected": expected,
                    "notebook_cell": cell,
                    "reason": "path does not exist",
                }
            )
    return unresolved


def write_missing_paths_report(output_dir: Path, unresolved: List[Dict[str, Any]]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "BATCH_A_GENERATION_MISSING_PATHS.md"
    lines = [
        "# Batch A generation — missing paths",
        "",
        "Computation stopped before the expensive diagnostic loop.",
        "",
    ]
    for item in unresolved:
        lines.extend(
            [
                f"## `{item.get('object', item.get('path_or_object'))}`",
                "",
                f"- **Unresolved path / object:** `{item.get('path_or_object')}`",
                f"- **Expected:** {item.get('expected')}",
                f"- **Notebook reference:** {item.get('notebook_cell')}",
                f"- **Reason:** {item.get('reason')}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def to_model_record(rec: discovery.IndividualModelRecord) -> core.ModelRecord:
    return core.ModelRecord(
        model_id=rec.model_id,
        model_family=rec.model_family,
        model_path=rec.model_path,
        metadata_path=rec.metadata_path,
        metadata=rec.metadata,
        mapping_method=rec.mapping_method,
    )


def attribution_non_degenerate_gate(empirical_attribution: Any, n_samples: int) -> core.ValidityRow:
    array = np.asarray(
        empirical_attribution.detach().cpu().numpy()
        if torch.is_tensor(empirical_attribution)
        else empirical_attribution,
        dtype=float,
    ).reshape(-1)
    finite = bool(np.isfinite(array).all()) if array.size else False
    nonzero = bool(np.nanmax(np.abs(array)) > 1e-12) if array.size else False
    nonconstant = bool(np.nanstd(array) > 1e-12) if array.size else False
    passed = finite and nonzero and nonconstant
    return core.ValidityRow(
        model_id="",
        model_family="",
        gate_id="validity_attribution_non_degenerate_pass",
        value=float(passed),
        suggested_pass=passed,
        status="computed",
        n_samples=n_samples,
        ci_low=None,
        ci_high=None,
        notes=(
            f"finite={finite}; nonzero={nonzero}; nonconstant={nonconstant}. "
            "Pass requires finite, non-zero, and non-constant attribution outputs."
        ),
    )


def gate_status_from_validity(row: core.ValidityRow) -> str:
    if row.status in {"failed_to_compute", "error"}:
        return "error"
    if row.status in {"disabled", "not_computable"}:
        return "not_evaluated"
    if row.suggested_pass is True:
        return "passed"
    if row.suggested_pass is False:
        return "failed"
    if row.gate_id == "validity_input_permutation_pass" and row.value is not None:
        return "passed" if row.value > 0 else "failed"
    return "not_evaluated"


def process_one_model(
    api: core.ProjectAPI,
    record: core.ModelRecord,
    args: argparse.Namespace,
    profile: core.ProfileConfig,
    attribution_device: torch.device,
    run_hash: str,
) -> Dict[str, Any]:
    model_dir = args.output_dir / "per_model" / core.safe_id(record.model_id)
    model_dir.mkdir(parents=True, exist_ok=True)
    summary_path = model_dir / "summary.json"
    if args.resume and not args.overwrite and summary_path.exists():
        try:
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
            if existing.get("run_hash") == run_hash and existing.get("status") == "completed":
                LOGGER.info("Skipping completed model %s", record.model_id)
                return existing
        except Exception:
            pass

    model_seed = core.stable_seed(args.seed, record.model_id)
    core.set_global_seed(model_seed)
    started = time.time()
    metric_rows = []
    validity_rows: List[core.ValidityRow] = []
    detail_rows: List[Dict[str, Any]] = []
    stage_metadata: Dict[str, Any] = {}

    model = api.TrainableDT.from_pretrained(str(record.model_path)).to(attribution_device).eval()
    batch, context = core.prepare_model_batch(api, record, model, attribution_device)
    n_obs = len(batch["dates"])
    stage_metadata["validation_observation_count"] = n_obs

    contribution_rows, contribution_extras, contribution_meta = core.compute_contribution_diagnostics(
        api, model, batch, context, profile, model_seed, model_dir, save_details=True
    )
    metric_rows.extend(contribution_rows)
    stage_metadata.update(contribution_meta)
    validity_rows.append(
        attribution_non_degenerate_gate(
            contribution_extras["empirical_attribution"],
            len(contribution_extras["attribution_indices"]),
        )
    )

    deletion_rows, deletion_validity, deletion_meta = core.compute_deletion_diagnostics(
        api,
        model,
        batch,
        contribution_extras["empirical_attribution"],
        profile,
        model_seed,
        model_dir,
        save_details=True,
    )
    metric_rows.extend(deletion_rows)
    validity_rows.extend(deletion_validity)
    stage_metadata.update(deletion_meta)

    validity_rows.append(
        core.compute_input_permutation_validity(
            api, model, contribution_extras["sampled_batch"], model_seed
        )
    )
    validity_rows.append(
        core.compute_parameter_randomization_validity(
            api,
            record,
            contribution_extras["sampled_batch"],
            context,
            contribution_extras["empirical_profile"],
            profile,
            model_seed,
        )
    )

    model = model.to("cpu").eval()
    batch_cpu = core.move_batch_to_device(batch, torch.device("cpu"))
    temporal_rows, temporal_meta = core.compute_temporal_diagnostics(
        api, model, batch_cpu, context, profile, model_seed, model_dir, save_details=True
    )
    metric_rows.extend(temporal_rows)
    stage_metadata.update(temporal_meta)

    structure_rows, structure_meta = core.compute_structure_diagnostics(
        api, model, batch_cpu, profile, model_seed, model_dir, save_details=True
    )
    metric_rows.extend(structure_rows)
    stage_metadata.update(structure_meta)

    core.attach_identifiers(metric_rows, record.model_id, record.model_family)
    core.attach_identifiers(validity_rows, record.model_id, record.model_family)

    long_rows = []
    for row in metric_rows:
        long_rows.append(
            {
                "model_id": record.model_id,
                "model_family": record.model_family,
                "diagnostic_dimension": row.dimension,
                "metric_id": row.metric_id,
                "raw_value": row.metric_value,
                "preferred_direction": row.preferred_direction,
                "validation_start": str(pd.Timestamp(args.validation_start).date()),
                "validation_end": str(pd.Timestamp(args.validation_end).date()),
                "n_observations": n_obs,
                "seed": model_seed,
                "computation_status": "ok" if row.status == "computed" else row.status,
                "error_message": row.notes if row.status != "computed" else "",
            }
        )
    long_df = pd.DataFrame(long_rows)
    core.write_dataframe(
        long_df,
        model_dir / "diagnostics_individual_long.csv",
        model_dir / "diagnostics_individual_long.parquet",
    )

    gate_rows = []
    for row in validity_rows:
        gate_rows.append(
            {
                "model_id": record.model_id,
                "model_family": record.model_family,
                "validity_gate": row.gate_id,
                "gate_status": gate_status_from_validity(row),
                "raw_value": row.value,
                "threshold_or_rule": (row.notes.split(".")[0] if row.notes else ""),
                "notes": row.notes,
            }
        )
    pd.DataFrame(gate_rows).to_csv(model_dir / "validity_gates.csv", index=False)

    contrib_path = model_dir / "contribution_profiles.csv"
    if contrib_path.exists():
        contrib = pd.read_csv(contrib_path)
        for _, r in contrib.iterrows():
            detail_rows.append(
                {
                    "model_id": record.model_id,
                    "model_family": record.model_family,
                    "diagnostic_dimension": "contribution",
                    "metric_id": "contribution_feature_family_profile",
                    "feature_family": r.get("feature_family"),
                    "baseline": r.get("baseline"),
                    "layer": None,
                    "lag": None,
                    "window_id": r.get("window"),
                    "signed_value": r.get("signed_attribution"),
                    "absolute_value": r.get("absolute_attribution"),
                    "raw_value": r.get("absolute_attribution"),
                }
            )
    if detail_rows:
        core.write_dataframe(
            pd.DataFrame(detail_rows),
            model_dir / "diagnostics_details.csv",
            model_dir / "diagnostics_details.parquet",
        )

    summary = {
        "run_hash": run_hash,
        "status": "completed",
        "model_id": record.model_id,
        "model_family": record.model_family,
        "mapping_method": record.mapping_method,
        "model_path": str(record.model_path),
        "validation_start": str(pd.Timestamp(args.validation_start).date()),
        "validation_end": str(pd.Timestamp(args.validation_end).date()),
        "n_observations": n_obs,
        "seed": model_seed,
        "metric_count": len(metric_rows),
        "core_metric_ids": sorted(CORE_METRIC_IDS),
        "stage_metadata": stage_metadata,
        "duration_seconds": time.time() - started,
    }
    core.write_json(summary_path, summary)

    del model, batch, batch_cpu
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def iqr(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    return float(clean.quantile(0.75) - clean.quantile(0.25))


def write_generation_report(
    output_dir: Path,
    args: argparse.Namespace,
    qc: Mapping[str, Any],
    failures: pd.DataFrame,
) -> None:
    cmd = (
        f'python src\\audit\\generate_batch_a_diagnostics.py `\n'
        f'  --notebook "{args.notebook}" `\n'
        f'  --model-assessment-pickle "{args.model_assessment_pickle}" `\n'
        f'  --output-dir "{args.output_dir}" `\n'
        f'  --validation-start {args.validation_start} `\n'
        f'  --validation-end {args.validation_end} `\n'
        f'  --sanity-sample-size {args.sanity_sample_size} `\n'
        f'  --n-jobs {args.n_jobs} `\n'
        f'  --seed {args.seed} `\n'
        f'  --resume'
    )
    lines = [
        "# Batch A diagnostic generation report",
        "",
        "## Scope",
        "",
        f"- Validation: `{qc['validation_start']}` to `{qc['validation_end']}` (inclusive)",
        "- Holdout used: **No**",
        "- Plots generated: **No**",
        "",
        "## Notebook sections reused",
        "",
        "- Configuration / imports / AnalysisConfig (cells before Feature Importance)",
        "- Model and metadata loading conventions",
        "- `prepare_ablation_inputs` / `postprocess_attributions`",
        "- `feature_importance_per_indicator` (Integrated Gradients)",
        "- Temporal occlusion (`compute_temporal_occlusion_curves`)",
        "- Layer representations / conductance",
        "",
        "## Functions refactored",
        "",
        "- `src/audit/discovery.py`",
        "- `src/audit/compute_core.py`",
        "- `src/audit/generate_batch_a_diagnostics.py`",
        "",
        "## Source paths",
        "",
        f"- Notebook: `{args.notebook}`",
        f"- Models path: `{args.models_path}`",
        f"- Metrics dir: `{args.metrics_dir}`",
        f"- Assessment pickle: `{args.model_assessment_pickle}`",
        f"- Output dir: `{args.output_dir}`",
        "",
        "## Coverage",
        "",
        f"- Models completed: **{qc['completed_n_models']}** (expected {qc['expected_n_models']})",
        f"- Families completed: **{qc['completed_n_families']}** (expected {qc['expected_n_families']})",
        f"- Failed models: **{qc['failed_n_models']}**",
        f"- Successful core metric rows: **{qc['successful_core_metric_rows']}**",
        f"- Every family has every core metric: **{qc['every_family_has_every_core_metric']}**",
        "",
        "## Core metrics (higher_is_better)",
        "",
    ]
    for mid in sorted(CORE_METRIC_IDS):
        lines.append(f"- `{mid}`")
    lines.extend(
        [
            "",
            "## Aggregation hierarchy",
            "",
            "1. Individual model × metric",
            "2. Family median and IQR = P75 − P25",
            "3. Validity gates",
            "",
            f"## Sanity-check sampling (`--sanity-sample-size={args.sanity_sample_size}`, seed={args.seed})",
            "",
            "## Failed or missing models",
            "",
        ]
    )
    if failures.empty:
        lines.append("- None recorded in this run.")
    else:
        for _, row in failures.head(50).iterrows():
            lines.append(
                f"- `{row.get('model_id')}` ({row.get('model_family')}): {row.get('error_message')}"
            )
    lines.extend(["", "## Exact reproduction command", "", "```powershell", cmd, "```", ""])
    (output_dir / "BATCH_A_DIAGNOSTIC_GENERATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def aggregate_outputs(output_dir: Path, run_hash: str, args: argparse.Namespace) -> Dict[str, Any]:
    long_frames: List[pd.DataFrame] = []
    detail_frames: List[pd.DataFrame] = []
    gate_frames: List[pd.DataFrame] = []
    failures: List[Dict[str, Any]] = []
    for model_dir in sorted((output_dir / "per_model").glob("*")):
        if not model_dir.is_dir():
            continue
        summary_path = model_dir / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") == "failed":
            failures.append(summary)
            continue
        if summary.get("run_hash") != run_hash or summary.get("status") != "completed":
            continue
        lp = model_dir / "diagnostics_individual_long.csv"
        if lp.exists():
            long_frames.append(pd.read_csv(lp))
        dp = model_dir / "diagnostics_details.csv"
        if dp.exists():
            detail_frames.append(pd.read_csv(dp))
        gp = model_dir / "validity_gates.csv"
        if gp.exists():
            gate_frames.append(pd.read_csv(gp))

    individual = pd.concat(long_frames, ignore_index=True) if long_frames else pd.DataFrame()
    details = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    gates = pd.concat(gate_frames, ignore_index=True) if gate_frames else pd.DataFrame()

    if not individual.empty and "raw_value" in individual.columns:
        individual["raw_value"] = pd.to_numeric(individual["raw_value"], errors="coerce")
        vals = individual["raw_value"].to_numpy(dtype=float)
        individual.loc[~np.isfinite(vals), "raw_value"] = np.nan

    family_rows = []
    if not individual.empty:
        for (family, dim, metric_id, direction), group in individual.groupby(
            ["model_family", "diagnostic_dimension", "metric_id", "preferred_direction"],
            dropna=False,
            sort=True,
        ):
            values = pd.to_numeric(group["raw_value"], errors="coerce")
            arr = values.to_numpy(dtype=float)
            ok = values[np.isfinite(arr)]
            family_rows.append(
                {
                    "model_family": family,
                    "diagnostic_dimension": dim,
                    "metric_id": metric_id,
                    "median_value": float(ok.median()) if len(ok) else np.nan,
                    "iqr_value": iqr(ok) if len(ok) else np.nan,
                    "n_models": int(group["model_id"].nunique()),
                    "n_successful_models": int(ok.notna().sum()),
                    "preferred_direction": direction,
                    "validation_start": str(pd.Timestamp(args.validation_start).date()),
                    "validation_end": str(pd.Timestamp(args.validation_end).date()),
                }
            )
    family = pd.DataFrame(family_rows)

    core.write_dataframe(
        individual,
        output_dir / "batch_a_diagnostics_individual_long.csv",
        output_dir / "batch_a_diagnostics_individual_long.parquet",
    )
    core.write_dataframe(
        details,
        output_dir / "batch_a_diagnostics_details.csv",
        output_dir / "batch_a_diagnostics_details.parquet",
    )
    core.write_dataframe(
        family,
        output_dir / "batch_a_diagnostics_family_long.csv",
        output_dir / "batch_a_diagnostics_family_long.parquet",
    )
    gates.to_csv(output_dir / "batch_a_validity_gates.csv", index=False)

    failure_df = pd.DataFrame(failures)
    if failure_df.empty:
        failure_df = pd.DataFrame(
            columns=["model_id", "model_family", "model_path", "error_type", "error_message", "traceback"]
        )
    failure_df.to_csv(output_dir / "batch_a_diagnostic_failures.csv", index=False)
    core.write_yaml_compatible(output_dir / "batch_a_diagnostic_manifest.yaml", METRIC_MANIFEST)

    n_models = int(individual["model_id"].nunique()) if not individual.empty else 0
    n_families = int(individual["model_family"].nunique()) if not individual.empty else 0
    core_ok = 0
    if not individual.empty:
        core_df = individual[individual["metric_id"].isin(CORE_METRIC_IDS)]
        core_ok = int((core_df["computation_status"] == "ok").sum())

    qc: Dict[str, Any] = {
        "expected_n_models": EXPECTED_N_MODELS,
        "expected_n_families": EXPECTED_N_FAMILIES,
        "completed_n_models": n_models,
        "completed_n_families": n_families,
        "failed_n_models": int(len(failure_df)),
        "successful_core_metric_rows": core_ok,
        "validation_start": str(pd.Timestamp(args.validation_start).date()),
        "validation_end": str(pd.Timestamp(args.validation_end).date()),
        "holdout_used": False,
        "plots_generated": False,
        "run_hash": run_hash,
        "seed": args.seed,
        "sanity_sample_size": args.sanity_sample_size,
        "model_count_matches_expected": n_models == EXPECTED_N_MODELS,
        "family_count_matches_expected": n_families == EXPECTED_N_FAMILIES,
        "core_metrics": sorted(CORE_METRIC_IDS),
        "every_family_has_every_core_metric": False,
    }
    if not family.empty:
        pivot = family[family["metric_id"].isin(CORE_METRIC_IDS)].pivot_table(
            index="model_family", columns="metric_id", values="median_value", aggfunc="first"
        )
        qc["every_family_has_every_core_metric"] = bool(
            pivot.shape[0] == n_families and set(pivot.columns) >= CORE_METRIC_IDS and pivot.notna().all().all()
        )

    core.write_json(output_dir / "batch_a_diagnostic_qc.json", qc)
    write_generation_report(output_dir, args, qc, failure_df)
    return {"individual": individual, "family": family, "gates": gates, "failures": failure_df, "qc": qc}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.output_dir, args.log_level)
    core.set_validation_window(args.validation_start, args.validation_end)
    core.register_metrics(METRIC_MANIFEST)
    profile = profile_from_args(args)
    run_hash = config_hash(args, profile)
    core.set_global_seed(args.seed)
    device = core.resolve_device(args.device)

    LOGGER.info("Run hash: %s", run_hash)
    LOGGER.info("Validation: %s -> %s", args.validation_start, args.validation_end)
    LOGGER.info("Holdout files are not loaded")

    unresolved = preflight_paths(args)
    if unresolved:
        report = write_missing_paths_report(args.output_dir, unresolved)
        LOGGER.error("Missing paths detected. See %s", report)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "per_model").mkdir(parents=True, exist_ok=True)
    core.write_yaml_compatible(args.output_dir / "batch_a_diagnostic_manifest.yaml", METRIC_MANIFEST)

    api = core.load_project_api(args.analysis_repo_root, args.model_code_root)
    assessment = None if args.skip_assessment_pickle else args.model_assessment_pickle
    records, mapping_report, disc_unresolved = discovery.build_individual_records(
        args.metrics_dir,
        args.models_path,
        assessment,
    )
    core.write_dataframe(mapping_report, args.output_dir / "family_mapping_report.csv")
    if disc_unresolved:
        write_missing_paths_report(args.output_dir, disc_unresolved)
        LOGGER.error("Unresolved family checkpoints; see BATCH_A_GENERATION_MISSING_PATHS.md")
        return 2

    if args.model_id:
        requested = set(args.model_id)
        records = [r for r in records if r.model_id in requested]
    if args.max_models is not None:
        records = records[: args.max_models]

    n_models = len(records)
    n_families = len({r.model_family for r in records})
    if n_models != EXPECTED_N_MODELS and args.max_models is None and not args.model_id:
        LOGGER.warning("Expected %d models, discovered %d", EXPECTED_N_MODELS, n_models)
    if n_families != EXPECTED_N_FAMILIES and args.max_models is None and not args.model_id:
        LOGGER.warning("Expected %d families, discovered %d", EXPECTED_N_FAMILIES, n_families)

    summaries: List[Dict[str, Any]] = []

    def _run(rec: discovery.IndividualModelRecord) -> Dict[str, Any]:
        return process_one_model(api, to_model_record(rec), args, profile, device, run_hash)

    if args.n_jobs and args.n_jobs > 1:
        with ThreadPoolExecutor(max_workers=args.n_jobs) as pool:
            futures = {pool.submit(_run, rec): rec for rec in records}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Batch A diagnostics"):
                rec = futures[fut]
                try:
                    summaries.append(fut.result())
                except Exception as exc:
                    LOGGER.error("Model %s failed: %s", rec.model_id, exc)
                    failure = {
                        "model_id": rec.model_id,
                        "model_family": rec.model_family,
                        "model_path": str(rec.model_path),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "traceback": traceback.format_exc(),
                        "status": "failed",
                        "run_hash": run_hash,
                    }
                    fail_dir = args.output_dir / "per_model" / core.safe_id(rec.model_id)
                    fail_dir.mkdir(parents=True, exist_ok=True)
                    core.write_json(fail_dir / "summary.json", failure)
    else:
        for rec in tqdm(records, desc="Batch A diagnostics", unit="model"):
            LOGGER.info("Processing %s (%s)", rec.model_id, rec.model_family)
            try:
                summaries.append(_run(rec))
            except Exception as exc:
                LOGGER.error("Model %s failed: %s", rec.model_id, exc)
                failure = {
                    "model_id": rec.model_id,
                    "model_family": rec.model_family,
                    "model_path": str(rec.model_path),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                    "status": "failed",
                    "run_hash": run_hash,
                }
                fail_dir = args.output_dir / "per_model" / core.safe_id(rec.model_id)
                fail_dir.mkdir(parents=True, exist_ok=True)
                core.write_json(fail_dir / "summary.json", failure)
                gc.collect()

    result = aggregate_outputs(args.output_dir, run_hash, args)
    qc = result["qc"]

    print("\n=== Batch A diagnostic generation summary ===")
    print(f"models processed: {qc['completed_n_models']}")
    print(f"families processed: {qc['completed_n_families']}")
    print(f"successful core metric rows: {qc['successful_core_metric_rows']}")
    print(f"failed models: {qc['failed_n_models']}")
    if not result["gates"].empty:
        print(
            "validity checks completed:",
            int(result["gates"]["gate_status"].isin(["passed", "failed"]).sum()),
        )
    else:
        print("validity checks completed: 0")
    print(f"output paths: {args.output_dir}")
    print("\nExact PowerShell rerun command:")
    print(
        f'python src\\audit\\generate_batch_a_diagnostics.py `\n'
        f'  --notebook "{args.notebook}" `\n'
        f'  --model-assessment-pickle "{args.model_assessment_pickle}" `\n'
        f'  --output-dir "{args.output_dir}" `\n'
        f'  --validation-start {args.validation_start} `\n'
        f'  --validation-end {args.validation_end} `\n'
        f'  --sanity-sample-size {args.sanity_sample_size} `\n'
        f'  --n-jobs {args.n_jobs} `\n'
        f'  --seed {args.seed} `\n'
        f'  --resume'
    )

    if qc["completed_n_models"] == 0:
        return 2
    if not qc.get("every_family_has_every_core_metric", False) and args.max_models is None and not args.model_id:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
