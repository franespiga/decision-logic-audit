#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate validation-period diagnostic inputs for the Batch A audit pipeline.

This script refactors the post-``get_top_k`` portion of the original notebook into
an all-model, resumable, non-visual analytical pipeline.  It processes each
frozen Decision Transformer checkpoint sequentially and writes model- and
family-level diagnostics for the contribution, necessity, and structure audit
axes.

The script deliberately does NOT:
  * inspect or use May-2024--June-2026 holdout outcomes;
  * select models from performance;
  * compare only the best and worst models;
  * generate t-SNE/UMAP figures, attention visualisations, or prospective-return
    analyses;
  * claim that diagnostics predict future Sharpe ratios.

Default paths are repository-relative and overridable from the CLI or
environment variables.

Typical invocation
------------------
python scripts/generate_batch_a_diagnostics.py \\
  --analysis-repo-root . \\
  --model-code-root . \\
  --models-path models \\
  --metrics-dir results/mendeley/dt_metrics \\
  --model-assessment-pickle results/model_assessment.pkl \\
  --profile publication \\
  --device auto \\
  --resume
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import importlib
import inspect
import json
import logging
import math
import os
import random
import re
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

try:
    import cloudpickle
except ImportError:  # pragma: no cover - required only for optional mapping fallback
    cloudpickle = None


LOGGER = logging.getLogger("batch_a_diagnostics")
_SCRIPT_ROOT = Path(__file__).resolve().parents[1]


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else default

VALIDATION_START = pd.Timestamp("2016-01-01")
VALIDATION_END = pd.Timestamp("2024-04-30")

REGIME_BOUNDARIES: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]] = {
    "bull_market": (pd.Timestamp("2016-01-01"), pd.Timestamp("2020-02-15")),
    "crash_period": (pd.Timestamp("2020-02-16"), pd.Timestamp("2020-05-31")),
    "recovery_growth": (pd.Timestamp("2020-06-01"), pd.Timestamp("2021-12-31")),
    "sideways_uncertain": (pd.Timestamp("2022-01-01"), pd.Timestamp("2024-04-30")),
}

METRIC_REGISTRY: List[Dict[str, Any]] = [
    {
        "metric_id": "contribution_feature_rank_stability",
        "human_readable_name": "Feature-family rank stability across chronological windows",
        "dimension": "contribution",
        "preferred_direction": "higher_is_better",
        "role": "core_score",
        "architecture_scope": "differentiable_policy",
        "justification": "Stable feature-family rankings provide more reproducible contribution evidence.",
    },
    {
        "metric_id": "contribution_baseline_concordance",
        "human_readable_name": "Attribution rank concordance across alternative baselines",
        "dimension": "contribution",
        "preferred_direction": "higher_is_better",
        "role": "core_score",
        "architecture_scope": "differentiable_policy",
        "justification": "Baseline-sensitive rankings should not be treated as stable contribution evidence.",
    },
    {
        "metric_id": "contribution_signed_rank_stability",
        "human_readable_name": "Signed feature-family rank stability across chronological windows",
        "dimension": "contribution",
        "preferred_direction": "higher_is_better",
        "role": "optional_score",
        "architecture_scope": "differentiable_policy",
        "justification": "Signed stability checks whether directional influence is consistent over time.",
    },
    {
        "metric_id": "contribution_attribution_evenness",
        "human_readable_name": "Normalized entropy of the absolute feature-family attribution profile",
        "dimension": "contribution",
        "preferred_direction": "higher_is_better",
        "role": "descriptive_only",
        "architecture_scope": "differentiable_policy",
        "justification": "Evenness is descriptive; diffuse attribution is not assumed to be universally superior.",
    },
    {
        "metric_id": "necessity_deletion_lift_vs_random",
        "human_readable_name": "Attribution-guided deletion effect relative to matched random deletion",
        "dimension": "necessity",
        "preferred_direction": "higher_is_better",
        "role": "core_score",
        "architecture_scope": "sequential_policy",
        "justification": "Inputs ranked as influential should change the policy more than matched random inputs.",
    },
    {
        "metric_id": "necessity_temporal_profile_stability",
        "human_readable_name": "Temporal-occlusion profile stability across chronological windows",
        "dimension": "necessity",
        "preferred_direction": "higher_is_better",
        "role": "core_score",
        "architecture_scope": "history_conditioned_policy",
        "justification": "Stable temporal-necessity profiles provide stronger evidence than isolated lag effects.",
    },
    {
        "metric_id": "necessity_temporal_half_life_cv",
        "human_readable_name": "Coefficient of variation of temporal half-life across chronological windows",
        "dimension": "necessity",
        "preferred_direction": "lower_is_better",
        "role": "optional_score",
        "architecture_scope": "history_conditioned_policy",
        "justification": "Lower dispersion indicates a more stable temporal dependence pattern.",
    },
    {
        "metric_id": "necessity_recency_index",
        "human_readable_name": "Share of temporal-occlusion mass in the most recent lags",
        "dimension": "necessity",
        "preferred_direction": "higher_is_better",
        "role": "descriptive_only",
        "architecture_scope": "history_conditioned_policy",
        "justification": "Recency is reported descriptively and is not assumed to be universally preferable.",
    },
    {
        "metric_id": "structure_neighbourhood_persistence",
        "human_readable_name": "Latent-neighbourhood persistence across adjacent layers and bootstrap subsets",
        "dimension": "structure",
        "preferred_direction": "higher_is_better",
        "role": "core_score",
        "architecture_scope": "representation_exposing_policy",
        "justification": "Persistent local neighbourhoods indicate a less erratic internal representation geometry.",
    },
    {
        "metric_id": "structure_conductance_evenness",
        "human_readable_name": "Normalized entropy of absolute layer-conductance shares",
        "dimension": "structure",
        "preferred_direction": "higher_is_better",
        "role": "core_score",
        "architecture_scope": "layered_differentiable_policy",
        "justification": "Higher evenness indicates less extreme concentration of influence in one component.",
    },
    {
        "metric_id": "structure_effective_rank",
        "human_readable_name": "Normalized effective rank of the deepest latent representation",
        "dimension": "structure",
        "preferred_direction": "higher_is_better",
        "role": "optional_score",
        "architecture_scope": "representation_exposing_policy",
        "justification": "A higher effective rank is evidence against severe representational collapse.",
    },
]

VALIDITY_REGISTRY: List[Dict[str, Any]] = [
    {
        "gate_id": "baseline_concordance_available",
        "human_readable_name": "Alternative-baseline concordance is computable",
        "preferred_direction": "boolean_pass",
        "suggested_threshold": None,
        "mandatory_candidate": True,
    },
    {
        "gate_id": "guided_deletion_exceeds_random",
        "human_readable_name": "Attribution-guided deletion exceeds matched random deletion",
        "preferred_direction": "boolean_pass",
        "suggested_threshold": "bootstrap_ci_low_difference_gt_0",
        "mandatory_candidate": True,
    },
    {
        "gate_id": "parameter_randomization_changes_attribution",
        "human_readable_name": "Attribution ranking changes after parameter randomization",
        "preferred_direction": "higher_is_better",
        "suggested_threshold": 0.20,
        "mandatory_candidate": False,
    },
    {
        "gate_id": "input_permutation_changes_action",
        "human_readable_name": "Policy output changes after feature-wise input permutation",
        "preferred_direction": "higher_is_better",
        "suggested_threshold": None,
        "mandatory_candidate": False,
    },
]


@dataclass(frozen=True)
class ProfileConfig:
    attribution_samples: int
    temporal_samples: int
    structure_samples: int
    deletion_samples: int
    ig_steps: int
    sanity_ig_steps: int
    chronological_windows: int
    random_deletion_repeats: int
    bootstrap_repeats: int
    deletion_fraction: float
    recent_lags: int
    knn_k: int


PROFILES: Dict[str, ProfileConfig] = {
    "smoke": ProfileConfig(
        attribution_samples=16,
        temporal_samples=12,
        structure_samples=16,
        deletion_samples=12,
        ig_steps=8,
        sanity_ig_steps=6,
        chronological_windows=3,
        random_deletion_repeats=3,
        bootstrap_repeats=10,
        deletion_fraction=0.10,
        recent_lags=3,
        knn_k=3,
    ),
    "standard": ProfileConfig(
        attribution_samples=64,
        temporal_samples=48,
        structure_samples=64,
        deletion_samples=48,
        ig_steps=32,
        sanity_ig_steps=16,
        chronological_windows=4,
        random_deletion_repeats=20,
        bootstrap_repeats=50,
        deletion_fraction=0.10,
        recent_lags=3,
        knn_k=5,
    ),
    "publication": ProfileConfig(
        attribution_samples=128,
        temporal_samples=96,
        structure_samples=128,
        deletion_samples=64,
        ig_steps=50,
        sanity_ig_steps=24,
        chronological_windows=4,
        random_deletion_repeats=30,
        bootstrap_repeats=100,
        deletion_fraction=0.10,
        recent_lags=3,
        knn_k=7,
    ),
}


@dataclass
class ProjectAPI:
    find_directories_by_patterns: Callable[..., Any]
    load_datasets: Callable[..., Any]
    prepare_ablation_inputs: Callable[..., Any]
    postprocess_attributions: Callable[..., Any]
    feature_importance_per_indicator: Callable[..., Any]
    compute_temporal_occlusion_curves: Callable[..., Any]
    extract_layer_representations: Callable[..., Any]
    compute_layer_conductance: Callable[..., Any]
    TrainableDT: Any
    forward_action_helper: Optional[Callable[..., Any]] = None


@dataclass
class ModelRecord:
    model_id: str
    model_family: str
    model_path: Path
    metadata_path: Path
    metadata: Dict[str, Any]
    mapping_method: str


@dataclass
class MetricRow:
    model_id: str
    model_family: str
    dimension: str
    metric_id: str
    metric_value: Optional[float]
    preferred_direction: str
    role: str
    architecture_scope: str
    status: str
    n_samples: Optional[int]
    ci_low: Optional[float]
    ci_high: Optional[float]
    source_method: str
    notes: str


@dataclass
class ValidityRow:
    model_id: str
    model_family: str
    gate_id: str
    value: Optional[float]
    suggested_pass: Optional[bool]
    status: str
    n_samples: Optional[int]
    ci_low: Optional[float]
    ci_high: Optional[float]
    notes: str


class DiagnosticError(RuntimeError):
    """Raised for an expected, model-local diagnostic failure."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate all-model validation diagnostics required by Batch A.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--analysis-repo-root",
        type=Path,
        default=_env_path("DECISION_LOGIC_AUDIT_REPO_ROOT", _SCRIPT_ROOT),
        help="Repository containing the manuscript analysis and result directories.",
    )
    parser.add_argument(
        "--model-code-root",
        type=Path,
        default=_env_path("DECISION_LOGIC_AUDIT_MODEL_CODE_ROOT", _SCRIPT_ROOT),
        help="Repository containing models, data, ablation, and representation modules.",
    )
    parser.add_argument(
        "--models-path",
        type=Path,
        default=_env_path("DECISION_LOGIC_AUDIT_MODELS_PATH", _SCRIPT_ROOT / "models"),
        help="Root directory containing trained checkpoint folders.",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=_env_path(
            "DECISION_LOGIC_AUDIT_METRICS_DIR",
            _SCRIPT_ROOT / "results" / "mendeley" / "dt_metrics",
        ),
        help="Directory containing model_cost_sensitivity and related performance tables.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_env_path(
            "DECISION_LOGIC_AUDIT_DIAGNOSTICS_DIR",
            _SCRIPT_ROOT / "results" / "mendeley" / "dt_diagnostics",
        ),
        help="Output directory for Batch A diagnostic inputs.",
    )
    parser.add_argument(
        "--model-assessment-pickle",
        type=Path,
        default=_env_path(
            "DECISION_LOGIC_AUDIT_MODEL_ASSESSMENT",
            _SCRIPT_ROOT / "results" / "model_assessment.pkl",
        ),
        help="Optional fallback source for identifier reconciliation; portfolio outcomes are not analysed.",
    )
    parser.add_argument(
        "--family-map-csv",
        type=Path,
        default=None,
        help="Optional explicit two-column model_id/model_family mapping, preferred over inference.",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="publication",
        help="Computational profile controlling deterministic sample sizes and repetitions.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device used for Integrated Gradients. Temporal and representation diagnostics use CPU for compatibility.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-models",
        type=int,
        default=None,
        help="Optional limit for smoke testing. Omit for the publication run.",
    )
    parser.add_argument(
        "--model-id",
        action="append",
        default=None,
        help="Process only the specified model ID. Repeat the flag for multiple models.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip models with a completed per-model summary from the same configuration hash.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite per-model outputs and consolidated tables.",
    )
    parser.add_argument(
        "--save-details",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save aggregated attribution, temporal, and layer profiles for each model.",
    )
    parser.add_argument(
        "--run-parameter-randomization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run a sampled parameter-randomization attribution sanity check.",
    )
    parser.add_argument(
        "--strict-family-mapping",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail before computation if a checkpoint cannot be mapped to the performance-table family identifier.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return parser.parse_args()


def configure_logging(output_dir: Path, level: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "generate_batch_a_diagnostics.log"
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def stable_seed(base_seed: int, text: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{text}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def safe_id(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return clean[:180] or "unnamed"


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=json_default, allow_nan=False)


def write_yaml_compatible(path: Path, payload: Any) -> None:
    """Write YAML when PyYAML is installed; JSON is valid YAML otherwise."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml  # type: ignore

        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
    except ImportError:
        write_json(path, payload)


def write_dataframe(df: pd.DataFrame, csv_path: Path, parquet_path: Optional[Path] = None) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    if parquet_path is not None:
        try:
            df.to_parquet(parquet_path, index=False)
        except Exception as exc:  # pragma: no cover - parquet engine may be absent
            LOGGER.warning("Could not write %s: %s", parquet_path, exc)


def add_import_paths(*roots: Path) -> None:
    candidates: List[Path] = []
    for root in roots:
        candidates.extend([root, root / "src", root / "notebooks"])
    for candidate in candidates:
        if candidate.exists():
            text = str(candidate.resolve())
            if text not in sys.path:
                sys.path.insert(0, text)


def import_attribute(candidates: Sequence[Tuple[str, str]], required: bool = True) -> Any:
    errors: List[str] = []
    for module_name, attribute_name in candidates:
        try:
            module = importlib.import_module(module_name)
            return getattr(module, attribute_name)
        except Exception as exc:
            errors.append(f"{module_name}.{attribute_name}: {exc}")
    if required:
        raise ImportError("Unable to import required project attribute:\n" + "\n".join(errors))
    LOGGER.warning("Optional project helper unavailable: %s", " | ".join(errors))
    return None


def load_project_api(analysis_repo_root: Path, model_code_root: Path) -> ProjectAPI:
    add_import_paths(analysis_repo_root, model_code_root)
    return ProjectAPI(
        find_directories_by_patterns=import_attribute(
            [("src", "find_directories_by_patterns"), ("portfolio_optimization.src", "find_directories_by_patterns")]
        ),
        load_datasets=import_attribute([("data", "load_datasets"), ("src.data", "load_datasets")]),
        prepare_ablation_inputs=import_attribute(
            [("ablation.preprocessing", "prepare_ablation_inputs"), ("src.ablation.preprocessing", "prepare_ablation_inputs")]
        ),
        postprocess_attributions=import_attribute(
            [("ablation.postprocessing", "postprocess_attributions"), ("src.ablation.postprocessing", "postprocess_attributions")]
        ),
        feature_importance_per_indicator=import_attribute(
            [("ablation", "feature_importance_per_indicator"), ("src.ablation", "feature_importance_per_indicator")]
        ),
        compute_temporal_occlusion_curves=import_attribute(
            [
                ("ablation.temporal_analysis", "compute_temporal_occlusion_curves"),
                ("src.ablation.temporal_analysis", "compute_temporal_occlusion_curves"),
            ]
        ),
        extract_layer_representations=import_attribute(
            [
                ("regime_representation.transformations", "extract_layer_representations"),
                ("src.regime_representation.transformations", "extract_layer_representations"),
            ]
        ),
        compute_layer_conductance=import_attribute(
            [
                ("regime_representation.transformations", "compute_layer_conductance"),
                ("src.regime_representation.transformations", "compute_layer_conductance"),
            ]
        ),
        TrainableDT=import_attribute([("models.model", "TrainableDT"), ("src.models.model", "TrainableDT")]),
        forward_action_helper=import_attribute(
            [
                ("ablation", "_forward_action"),
                ("src.ablation", "_forward_action"),
                ("utils.model_helpers", "_forward_action"),
                ("src.utils.model_helpers", "_forward_action"),
            ],
            required=False,
        ),
    )


def discover_table_by_stem(directory: Path, stem: str) -> Path:
    matches = sorted(p for p in directory.glob(f"{stem}.*") if p.is_file())
    if not matches:
        exact = directory / stem
        if exact.exists() and exact.is_file():
            return exact
        raise FileNotFoundError(f"No table with stem '{stem}' found under {directory}")
    preferred_suffixes = [".parquet", ".csv", ".pkl", ".pickle", ".xlsx", ".json"]
    for suffix in preferred_suffixes:
        for match in matches:
            if match.suffix.lower() == suffix:
                return match
    return matches[0]


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".pkl", ".pickle"}:
        value = pd.read_pickle(path)
        if isinstance(value, pd.DataFrame):
            return value
        raise TypeError(f"Pickle {path} does not contain a pandas DataFrame")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported table format: {path}")


def normalize_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def build_family_mapping(
    metrics_dir: Path,
    explicit_map_csv: Optional[Path],
) -> Tuple[Dict[str, str], pd.DataFrame, Path]:
    if explicit_map_csv is not None:
        mapping_path = explicit_map_csv
        df = pd.read_csv(mapping_path)
    else:
        mapping_path = discover_table_by_stem(metrics_dir, "model_cost_sensitivity")
        df = load_table(mapping_path)

    required = {"model_id", "model_family"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Family mapping source {mapping_path} is missing columns {sorted(missing)}")

    pairs = df.loc[:, ["model_id", "model_family"]].dropna().drop_duplicates()
    counts = pairs.groupby("model_id")["model_family"].nunique()
    ambiguous = counts[counts > 1]
    if not ambiguous.empty:
        raise ValueError(
            "Some model_id values map to multiple model families: " + ", ".join(ambiguous.index.astype(str)[:20])
        )
    mapping = dict(zip(pairs["model_id"].astype(str), pairs["model_family"].astype(str)))
    return mapping, pairs, mapping_path


def nested_get(mapping: Mapping[str, Any], paths: Sequence[Sequence[str]], default: Any = None) -> Any:
    for path in paths:
        current: Any = mapping
        ok = True
        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                ok = False
                break
            current = current[key]
        if ok and current is not None:
            return current
    return default


def derive_family_signature(metadata: Mapping[str, Any]) -> str:
    args = metadata.get("experiment_args", {}) if isinstance(metadata.get("experiment_args", {}), Mapping) else {}
    lookback = nested_get(metadata, [["lookback"], ["experiment_args", "lookback"]], "NA")
    rebalance = nested_get(
        metadata,
        [
            ["rebalancing_frequency"],
            ["rebalance_every"],
            ["experiment_args", "rebalancing_frequency"],
            ["experiment_args", "rebalance_every"],
            ["experiment_args", "rebalance_frequency"],
        ],
        "NA",
    )
    technical = nested_get(
        metadata,
        [
            ["technical_indicators"],
            ["use_technical_indicators"],
            ["experiment_args", "technical_indicators"],
            ["experiment_args", "use_technical_indicators"],
        ],
        "NA",
    )
    conditioning = nested_get(
        metadata,
        [
            ["conditioning"],
            ["objective"],
            ["experiment_args", "conditioning"],
            ["experiment_args", "objective"],
            ["experiment_args", "reward_type"],
        ],
        args.get("trajectory_generator", "NA"),
    )
    return f"DT_lb{lookback}_re{rebalance}_ti{technical}_cond{conditioning}"


def resolve_model_family(
    model_id: str,
    metadata: Mapping[str, Any],
    mapping: Mapping[str, str],
) -> Tuple[str, str]:
    if model_id in mapping:
        return mapping[model_id], "exact_model_id"

    model_path_text = str(metadata.get("model_path", ""))
    candidates: List[Tuple[str, str]] = []
    for candidate_id, family in mapping.items():
        if candidate_id and (candidate_id in model_id or model_id in candidate_id):
            candidates.append((candidate_id, family))
        elif candidate_id and candidate_id in model_path_text:
            candidates.append((candidate_id, family))
    unique_families = sorted({family for _, family in candidates})
    if len(unique_families) == 1:
        return unique_families[0], "substring_model_id"

    normalized_model = normalize_text(model_id)
    normalized_matches = {
        family for candidate_id, family in mapping.items() if normalize_text(candidate_id) == normalized_model
    }
    if len(normalized_matches) == 1:
        return next(iter(normalized_matches)), "normalized_model_id"

    for key_path in [
        ["model_family"],
        ["family"],
        ["experiment_args", "model_family"],
        ["experiment_args", "family"],
    ]:
        value = nested_get(metadata, [key_path])
        if value is not None:
            family_text = str(value)
            if family_text in set(mapping.values()):
                return family_text, "metadata_family"

    signature = derive_family_signature(metadata)
    signature_norm = normalize_text(signature)
    family_matches = [family for family in set(mapping.values()) if normalize_text(family) == signature_norm]
    if len(family_matches) == 1:
        return family_matches[0], "derived_exact_family"

    return signature, "derived_unmatched_family"


def discover_models(
    api: ProjectAPI,
    models_path: Path,
    family_mapping: Mapping[str, str],
    strict_family_mapping: bool,
) -> Tuple[List[ModelRecord], pd.DataFrame]:
    raw_paths = api.find_directories_by_patterns(str(models_path), patterns=["out", "model$"])
    model_paths = sorted({Path(p) for p in raw_paths})
    records: List[ModelRecord] = []
    mapping_rows: List[Dict[str, Any]] = []
    known_families = set(family_mapping.values())

    for model_path in model_paths:
        # The historical checkpoint layout is not fully uniform. Find the nearest
        # metadata.json by walking upward instead of assuming model_path.parent.
        metadata_path: Optional[Path] = None
        search_nodes = [model_path] + list(model_path.parents)
        for node in search_nodes:
            candidate_dir = node if node.is_dir() else node.parent
            candidate = candidate_dir / "metadata.json"
            if candidate.exists():
                metadata_path = candidate
                break
            if candidate_dir == models_path or models_path not in candidate_dir.parents:
                # Stop after leaving the checkpoint root.
                if candidate_dir != model_path:
                    break
        if metadata_path is None:
            model_id = model_path.parent.name
            LOGGER.warning("Skipping %s because no ancestor metadata.json was found", model_path)
            mapping_rows.append(
                {
                    "model_id": model_id,
                    "model_path": str(model_path),
                    "model_family": None,
                    "mapping_method": "missing_metadata",
                    "mapping_resolved": False,
                }
            )
            continue
        experiment_dir = metadata_path.parent
        model_id = experiment_dir.name
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        metadata["model_path"] = str(model_path)
        # Match the original notebook convention: the data.h5 directory is two
        # levels above the experiment directory unless metadata provides it.
        metadata.setdefault("data_path", str(experiment_dir.parent.parent))
        family, method = resolve_model_family(model_id, metadata, family_mapping)
        resolved = family in known_families
        mapping_rows.append(
            {
                "model_id": model_id,
                "model_path": str(model_path),
                "model_family": family,
                "mapping_method": method,
                "mapping_resolved": resolved,
            }
        )
        records.append(
            ModelRecord(
                model_id=model_id,
                model_family=family,
                model_path=model_path,
                metadata_path=metadata_path,
                metadata=metadata,
                mapping_method=method,
            )
        )

    mapping_report = pd.DataFrame(mapping_rows)
    unresolved = mapping_report[~mapping_report["mapping_resolved"].fillna(False)]
    if strict_family_mapping and not unresolved.empty:
        unresolved_ids = ", ".join(unresolved["model_id"].astype(str).tolist())
        raise ValueError(
            "Checkpoint-to-family mapping is unresolved for: "
            f"{unresolved_ids}. Provide --family-map-csv with exact mappings or disable strict mapping for inspection."
        )
    return records, mapping_report


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def call_with_supported_kwargs(func: Callable[..., Any], **kwargs: Any) -> Any:
    signature = inspect.signature(func)
    accepts_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
    if accepts_var_kwargs:
        return func(**kwargs)
    filtered = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return func(**filtered)


def extract_rebalance_every(metadata: Mapping[str, Any]) -> int:
    value = nested_get(
        metadata,
        [
            ["rebalancing_frequency"],
            ["rebalance_every"],
            ["experiment_args", "rebalancing_frequency"],
            ["experiment_args", "rebalance_every"],
            ["experiment_args", "rebalance_frequency"],
        ],
        10,
    )
    try:
        return int(value)
    except (TypeError, ValueError):
        LOGGER.warning("Could not parse rebalancing frequency %r; using 10", value)
        return 10


def prepare_model_batch(
    api: ProjectAPI,
    record: ModelRecord,
    model: torch.nn.Module,
    device: torch.device,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    metadata = record.metadata
    cfg = metadata["experiment_args"]
    data_path = Path(str(metadata["data_path"])) / "data.h5"
    if not data_path.exists():
        alternative = record.model_path.parent.parent.parent / "data.h5"
        if alternative.exists():
            data_path = alternative
    data, indicators, observations = api.load_datasets(str(data_path), cfg)
    instruments = metadata.get("instruments")
    env_state = metadata.get("env_state")
    if env_state is None:
        raise DiagnosticError(f"metadata for {record.model_id} does not contain env_state")

    eval_data = data.merge(indicators, right_index=True, left_index=True).loc[:, env_state].dropna()
    eval_data.index = pd.to_datetime(eval_data.index)
    eval_data = eval_data.loc[VALIDATION_START:VALIDATION_END]
    if eval_data.empty:
        raise DiagnosticError("No validation observations after 2016-01-01--2024-04-30 filtering")

    if instruments is None:
        raise DiagnosticError(f"metadata for {record.model_id} does not contain instruments")
    ohlc_data = data.loc[:, instruments].ffill().dropna()
    ohlc_data.index = pd.to_datetime(ohlc_data.index)
    ohlc_data = ohlc_data.loc[:VALIDATION_END]

    ref_date = pd.Timestamp(cfg.get("reference_date", VALIDATION_START))
    effective_ref_date = max(ref_date, VALIDATION_START)
    lookback = int(metadata["lookback"])
    annualized_return = float(metadata.get("target_return", cfg.get("target_return", 0.0)))
    rebalance_every = extract_rebalance_every(metadata)

    batch = call_with_supported_kwargs(
        api.prepare_ablation_inputs,
        model=model,
        backtest_data=eval_data,
        ref_date=str(effective_ref_date.date()),
        ohlc_data=ohlc_data,
        lookback=lookback,
        rebalance_every=rebalance_every,
        annualized_return=annualized_return,
        use_trajectory_return_objective=False,
        use_incremental_timesteps=True,
        ppy=252,
        device=device,
    )
    dates = pd.to_datetime(pd.Index(batch["dates"]))
    mask = (dates >= VALIDATION_START) & (dates <= VALIDATION_END)
    if not mask.all():
        batch = subset_batch(batch, np.flatnonzero(mask))
        dates = pd.to_datetime(pd.Index(batch["dates"]))
    if len(dates) == 0:
        raise DiagnosticError("Prepared batch contains no validation-period dates")

    context = {
        "data": data,
        "indicators": indicators,
        "observations": observations,
        "instruments": instruments,
        "env_state": env_state,
        "eval_data": eval_data,
        "lookback": lookback,
        "rebalance_every": rebalance_every,
        "effective_ref_date": effective_ref_date,
    }
    return batch, context


def subset_batch(batch: Mapping[str, Any], indices: Sequence[int]) -> Dict[str, Any]:
    idx_np = np.asarray(indices, dtype=int)
    result: Dict[str, Any] = {}
    batch_length = len(batch["dates"])
    for key, value in batch.items():
        if torch.is_tensor(value) and value.ndim >= 1 and value.shape[0] == batch_length:
            index_tensor = torch.as_tensor(idx_np, device=value.device, dtype=torch.long)
            result[key] = value.index_select(0, index_tensor)
        elif isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == batch_length:
            result[key] = value[idx_np]
        elif isinstance(value, (pd.Series, pd.DataFrame)) and len(value) == batch_length:
            result[key] = value.iloc[idx_np]
        elif isinstance(value, (list, tuple)) and len(value) == batch_length:
            selected = [value[i] for i in idx_np]
            result[key] = type(value)(selected) if isinstance(value, tuple) else selected
        else:
            result[key] = value
    return result


def assign_regimes(dates: Sequence[Any]) -> np.ndarray:
    date_index = pd.to_datetime(pd.Index(dates))
    labels: List[str] = []
    for date in date_index:
        found = "Unknown"
        for label, (start, end) in REGIME_BOUNDARIES.items():
            if start <= date <= end:
                found = label
                break
        labels.append(found)
    return np.asarray(labels, dtype=object)


def stratified_chronological_indices(
    dates: Sequence[Any],
    max_samples: int,
    seed: int,
) -> np.ndarray:
    dates_idx = pd.to_datetime(pd.Index(dates))
    n = len(dates_idx)
    if n <= max_samples:
        return np.arange(n, dtype=int)
    labels = assign_regimes(dates_idx)
    rng = np.random.default_rng(seed)
    known_labels = [label for label in REGIME_BOUNDARIES if np.any(labels == label)]
    if not known_labels:
        return np.linspace(0, n - 1, max_samples, dtype=int)

    base = max_samples // len(known_labels)
    remainder = max_samples % len(known_labels)
    selected: List[int] = []
    for position, label in enumerate(known_labels):
        candidates = np.flatnonzero(labels == label)
        quota = min(len(candidates), base + (1 if position < remainder else 0))
        if quota == 0:
            continue
        if quota == len(candidates):
            chosen = candidates
        else:
            # Quantile positions preserve chronological coverage; a deterministic
            # jitter avoids identical samples when many positions round together.
            positions = np.linspace(0, len(candidates) - 1, quota)
            jitter = rng.uniform(-0.20, 0.20, size=quota)
            positions = np.clip(np.rint(positions + jitter), 0, len(candidates) - 1).astype(int)
            chosen = np.unique(candidates[positions])
            if len(chosen) < quota:
                remaining = np.setdiff1d(candidates, chosen, assume_unique=False)
                add = rng.choice(remaining, size=quota - len(chosen), replace=False)
                chosen = np.sort(np.concatenate([chosen, add]))
        selected.extend(chosen.tolist())

    if len(selected) < max_samples:
        remaining = np.setdiff1d(np.arange(n), np.asarray(selected, dtype=int), assume_unique=False)
        extra_count = min(max_samples - len(selected), len(remaining))
        if extra_count:
            extra_positions = np.linspace(0, len(remaining) - 1, extra_count, dtype=int)
            selected.extend(remaining[extra_positions].tolist())
    return np.asarray(sorted(set(selected)))[:max_samples]


def make_chronological_windows(n_rows: int, n_windows: int) -> List[np.ndarray]:
    if n_rows <= 1:
        return [np.arange(n_rows)]
    n_windows = max(2, min(n_windows, n_rows))
    return [arr for arr in np.array_split(np.arange(n_rows), n_windows) if len(arr) > 0]


def tensor_to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def finite_float(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def pairwise_rank_concordance(profiles: Sequence[np.ndarray], absolute: bool = False) -> float:
    values: List[float] = []
    for i in range(len(profiles)):
        for j in range(i + 1, len(profiles)):
            x = np.asarray(profiles[i], dtype=float)
            y = np.asarray(profiles[j], dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 3:
                continue
            tau = stats.kendalltau(x[mask], y[mask], variant="b").statistic
            if np.isfinite(tau):
                values.append(abs(float(tau)) if absolute else float(tau))
    return float(np.mean(values)) if values else float("nan")


def normalized_entropy(values: np.ndarray) -> float:
    x = np.abs(np.asarray(values, dtype=float))
    x = x[np.isfinite(x)]
    if len(x) <= 1 or x.sum() <= 0:
        return float("nan")
    p = x / x.sum()
    entropy = -np.sum(p[p > 0] * np.log(p[p > 0]))
    return float(entropy / np.log(len(p)))


def bootstrap_mean_ci(values: Sequence[float], repeats: int, seed: int, alpha: float = 0.05) -> Tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(array) == 1 or repeats <= 1:
        value = float(array.mean())
        return value, value, value
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(repeats, len(array)), replace=True).mean(axis=1)
    return float(array.mean()), float(np.quantile(samples, alpha / 2)), float(np.quantile(samples, 1 - alpha / 2))


def parse_indicator_from_label(label: Any) -> str:
    if isinstance(label, (tuple, list)):
        if len(label) >= 2:
            return str(label[1])
        return "|".join(map(str, label))
    text = str(label)
    # The existing post-processing convention is instrument_indicator_lag.
    parts = text.rsplit("_", 1)[0].split("_")
    if len(parts) >= 2:
        return "_".join(parts[1:])
    return text


def flatten_feature_names(feature_names: Any, expected_columns: int) -> np.ndarray:
    names = np.asarray(feature_names, dtype=object).reshape(-1)
    if len(names) == expected_columns:
        return names.astype(str)
    if len(names) > expected_columns:
        return names[:expected_columns].astype(str)
    fallback = np.asarray([f"feature_{i}" for i in range(expected_columns)], dtype=object)
    fallback[: len(names)] = names
    return fallback.astype(str)


def ensure_2d_attribution_matrix(attributions: Any) -> np.ndarray:
    array = tensor_to_numpy(attributions)
    if array.ndim == 0:
        return array.reshape(1, 1)
    if array.ndim == 1:
        return array.reshape(1, -1)
    return array.reshape(array.shape[0], -1)


def aggregate_feature_family_profile(
    attribution_matrix: np.ndarray,
    feature_names: np.ndarray,
    signed: bool,
) -> pd.Series:
    matrix = np.asarray(attribution_matrix, dtype=float)
    values = np.nanmean(matrix if signed else np.abs(matrix), axis=0)
    df = pd.DataFrame(
        {
            "feature": feature_names,
            "indicator": [parse_indicator_from_label(name) for name in feature_names],
            "value": values,
        }
    )
    profile = df.groupby("indicator", sort=True)["value"].mean()
    return profile


def align_profiles(profiles: Mapping[str, pd.Series]) -> Tuple[List[str], Dict[str, np.ndarray]]:
    labels = sorted(set().union(*(set(series.index.astype(str)) for series in profiles.values())))
    aligned: Dict[str, np.ndarray] = {}
    for name, series in profiles.items():
        aligned[name] = series.reindex(labels).fillna(0.0).to_numpy(dtype=float)
    return labels, aligned


def raw_feature_importance(attribution: Any, state_feature_count: int) -> np.ndarray:
    array = tensor_to_numpy(attribution)
    matching_axes = [axis for axis, size in enumerate(array.shape) if size == state_feature_count]
    if not matching_axes:
        flattened = np.abs(array).reshape(-1)
        if len(flattened) < state_feature_count:
            raise DiagnosticError(
                f"Cannot infer state-feature axis from attribution shape {array.shape}; expected {state_feature_count} features"
            )
        return flattened[:state_feature_count]
    feature_axis = matching_axes[-1]
    moved = np.moveaxis(array, feature_axis, -1)
    reduce_axes = tuple(range(moved.ndim - 1))
    return np.nanmean(np.abs(moved), axis=reduce_axes)


def build_baselines(states: torch.Tensor, seed: int) -> Dict[str, torch.Tensor]:
    mean_baseline = states.mean(dim=0, keepdim=True).expand_as(states).clone()
    zero_baseline = torch.zeros_like(states)
    generator = torch.Generator(device=states.device)
    generator.manual_seed(seed)
    permutation = torch.randperm(states.shape[0], generator=generator, device=states.device)
    sampled_background = states.index_select(0, permutation).detach().clone()
    return {
        "zero": zero_baseline,
        "empirical_mean": mean_baseline,
        "sampled_background": sampled_background,
    }


def compute_ig(
    api: ProjectAPI,
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    baseline: torch.Tensor,
    ig_steps: int,
) -> Any:
    kwargs = {
        "model": model,
        "states": batch["states"],
        "actions": batch["actions"],
        "returns_to_go": batch["returns_to_go"],
        "timesteps": batch["timesteps"],
        "attention_mask": batch.get("attention_mask"),
        "baseline": baseline,
        "target_action_idx": None,
        "n_steps": ig_steps,
        "ig_steps": ig_steps,
    }
    return call_with_supported_kwargs(api.feature_importance_per_indicator, **kwargs)


def postprocess_attribution(
    api: ProjectAPI,
    attribution: Any,
    context: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    env_state = context["env_state"]
    instruments = context["instruments"]
    num_indicators = len({item[1] for item in env_state}) if env_state and isinstance(env_state[0], (list, tuple)) else len(env_state)
    non_cash_instruments = set(instruments) - {"CASH"}
    processed, names = call_with_supported_kwargs(
        api.postprocess_attributions,
        attributions=attribution,
        feature_names=env_state,
        lookback=int(context["lookback"]) + 1,
        num_indicators=num_indicators,
        num_instruments=len(non_cash_instruments),
    )
    matrix = ensure_2d_attribution_matrix(processed)
    feature_names = flatten_feature_names(names, matrix.shape[1])
    return matrix, feature_names


def row_indices_for_date_window(n_rows: int, n_dates: int, date_indices: np.ndarray) -> np.ndarray:
    if n_dates <= 0:
        return np.arange(n_rows)
    if n_rows == n_dates:
        return date_indices
    if n_rows % n_dates == 0:
        repeat = n_rows // n_dates
        rows: List[int] = []
        for index in date_indices:
            rows.extend(range(index * repeat, (index + 1) * repeat))
        return np.asarray(rows, dtype=int)
    # Fallback proportional mapping when project post-processing has a different layout.
    boundaries = np.linspace(0, n_rows, n_dates + 1, dtype=int)
    rows = []
    for index in date_indices:
        rows.extend(range(boundaries[index], boundaries[index + 1]))
    return np.asarray(rows, dtype=int)


def compute_contribution_diagnostics(
    api: ProjectAPI,
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    context: Mapping[str, Any],
    profile: ProfileConfig,
    seed: int,
    model_dir: Path,
    save_details: bool,
) -> Tuple[List[MetricRow], Dict[str, Any], Dict[str, Any]]:
    attribution_indices = stratified_chronological_indices(batch["dates"], profile.attribution_samples, seed)
    sampled = subset_batch(batch, attribution_indices)
    states = sampled["states"]
    baselines = build_baselines(states, seed)
    baseline_matrices: Dict[str, np.ndarray] = {}
    baseline_profiles: Dict[str, pd.Series] = {}
    raw_attributions: Dict[str, Any] = {}
    feature_names: Optional[np.ndarray] = None

    for baseline_name, baseline in baselines.items():
        LOGGER.info("Computing IG baseline=%s on %d samples", baseline_name, len(attribution_indices))
        attribution = compute_ig(api, model, sampled, baseline, profile.ig_steps)
        matrix, names = postprocess_attribution(api, attribution, context)
        raw_attributions[baseline_name] = attribution
        baseline_matrices[baseline_name] = matrix
        baseline_profiles[baseline_name] = aggregate_feature_family_profile(matrix, names, signed=False)
        if feature_names is None:
            feature_names = names

    labels, aligned_baseline = align_profiles(baseline_profiles)
    baseline_concordance = pairwise_rank_concordance(list(aligned_baseline.values()))

    empirical_matrix = baseline_matrices["empirical_mean"]
    assert feature_names is not None
    date_windows = make_chronological_windows(len(attribution_indices), profile.chronological_windows)
    abs_window_profiles: List[pd.Series] = []
    signed_window_profiles: List[pd.Series] = []
    for date_window in date_windows:
        rows = row_indices_for_date_window(empirical_matrix.shape[0], len(attribution_indices), date_window)
        abs_window_profiles.append(
            aggregate_feature_family_profile(empirical_matrix[rows], feature_names, signed=False)
        )
        signed_window_profiles.append(
            aggregate_feature_family_profile(empirical_matrix[rows], feature_names, signed=True)
        )
    _, aligned_abs = align_profiles({str(i): p for i, p in enumerate(abs_window_profiles)})
    _, aligned_signed = align_profiles({str(i): p for i, p in enumerate(signed_window_profiles)})
    feature_rank_stability = pairwise_rank_concordance(list(aligned_abs.values()))
    signed_rank_stability = pairwise_rank_concordance(list(aligned_signed.values()))
    attribution_evenness = normalized_entropy(aligned_baseline["empirical_mean"])

    metric_rows = [
        make_metric_row(
            "contribution_feature_rank_stability",
            feature_rank_stability,
            len(attribution_indices),
            "Integrated Gradients; empirical-mean baseline; chronological windows",
        ),
        make_metric_row(
            "contribution_baseline_concordance",
            baseline_concordance,
            len(attribution_indices),
            "Integrated Gradients; zero, empirical-mean, and sampled-background baselines",
        ),
        make_metric_row(
            "contribution_signed_rank_stability",
            signed_rank_stability,
            len(attribution_indices),
            "Signed Integrated Gradients; chronological windows",
        ),
        make_metric_row(
            "contribution_attribution_evenness",
            attribution_evenness,
            len(attribution_indices),
            "Normalized entropy of empirical-mean-baseline attribution shares",
        ),
    ]

    if save_details:
        profile_rows: List[Dict[str, Any]] = []
        for baseline_name, series in baseline_profiles.items():
            for feature_family, value in series.items():
                profile_rows.append(
                    {
                        "baseline": baseline_name,
                        "feature_family": feature_family,
                        "absolute_attribution": float(value),
                    }
                )
        for window_index, (abs_profile, signed_profile) in enumerate(zip(abs_window_profiles, signed_window_profiles)):
            aligned_features = sorted(set(abs_profile.index) | set(signed_profile.index))
            for feature_family in aligned_features:
                profile_rows.append(
                    {
                        "baseline": "empirical_mean",
                        "window": window_index,
                        "feature_family": feature_family,
                        "absolute_attribution": finite_float(abs_profile.get(feature_family, np.nan)),
                        "signed_attribution": finite_float(signed_profile.get(feature_family, np.nan)),
                    }
                )
        write_dataframe(
            pd.DataFrame(profile_rows),
            model_dir / "contribution_profiles.csv",
            model_dir / "contribution_profiles.parquet",
        )
        pd.DataFrame(
            {
                "sample_index": attribution_indices,
                "date": pd.to_datetime(pd.Index(batch["dates"])[attribution_indices]),
                "regime": assign_regimes(pd.Index(batch["dates"])[attribution_indices]),
            }
        ).to_csv(model_dir / "attribution_sample_dates.csv", index=False)

    extras = {
        "attribution_indices": attribution_indices,
        "sampled_batch": sampled,
        "empirical_attribution": raw_attributions["empirical_mean"],
        "empirical_profile": baseline_profiles["empirical_mean"],
        "baseline_concordance": baseline_concordance,
    }
    metadata = {
        "feature_family_count": len(labels),
        "attribution_sample_count": len(attribution_indices),
        "chronological_windows": len(date_windows),
    }
    return metric_rows, extras, metadata


def get_registry_entry(metric_id: str) -> Dict[str, Any]:
    for entry in METRIC_REGISTRY:
        if entry["metric_id"] == metric_id:
            return entry
    raise KeyError(metric_id)


def make_metric_row(
    metric_id: str,
    value: Any,
    n_samples: Optional[int],
    source_method: str,
    status: Optional[str] = None,
    ci_low: Optional[float] = None,
    ci_high: Optional[float] = None,
    notes: str = "",
) -> MetricRow:
    registry = get_registry_entry(metric_id)
    numeric = finite_float(value)
    row_status = status or ("computed" if numeric is not None else "not_computable")
    return MetricRow(
        model_id="",
        model_family="",
        dimension=registry["dimension"],
        metric_id=metric_id,
        metric_value=numeric,
        preferred_direction=registry["preferred_direction"],
        role=registry["role"],
        architecture_scope=registry["architecture_scope"],
        status=row_status,
        n_samples=n_samples,
        ci_low=finite_float(ci_low),
        ci_high=finite_float(ci_high),
        source_method=source_method,
        notes=notes,
    )


def extract_action_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        tensor = output
    elif isinstance(output, Mapping):
        keys = ["action_preds", "actions", "action_predictions", "predicted_actions", "action"]
        tensor = next((output[key] for key in keys if key in output and torch.is_tensor(output[key])), None)
        if tensor is None:
            tensors = [value for value in output.values() if torch.is_tensor(value)]
            if not tensors:
                raise DiagnosticError("Model output mapping contains no tensor action prediction")
            tensor = tensors[-1]
    elif hasattr(output, "action_preds"):
        tensor = output.action_preds
    elif hasattr(output, "actions") and torch.is_tensor(output.actions):
        tensor = output.actions
    elif isinstance(output, (tuple, list)):
        tensors = [value for value in output if torch.is_tensor(value)]
        if not tensors:
            raise DiagnosticError("Model output tuple contains no tensor")
        # Decision Transformer implementations commonly return state, action, return.
        tensor = tensors[1] if len(tensors) >= 2 else tensors[0]
    else:
        raise DiagnosticError(f"Unsupported model output type: {type(output)!r}")

    if tensor.ndim >= 3:
        tensor = tensor[:, -1, ...]
    return tensor.reshape(tensor.shape[0], -1)


def forward_action(
    api: ProjectAPI,
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    states_override: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    states = states_override if states_override is not None else batch["states"]
    kwargs = {
        "model": model,
        "states": states,
        "actions": batch["actions"],
        "returns_to_go": batch["returns_to_go"],
        "timesteps": batch["timesteps"],
        "attention_mask": batch.get("attention_mask"),
    }
    with torch.no_grad():
        if api.forward_action_helper is not None:
            try:
                output = call_with_supported_kwargs(api.forward_action_helper, **kwargs)
                return extract_action_tensor(output)
            except Exception as exc:
                LOGGER.debug("Project _forward_action helper failed; trying direct model call: %s", exc)

        model_kwargs = {key: value for key, value in kwargs.items() if key != "model" and value is not None}
        try:
            output = call_with_supported_kwargs(model.forward, **model_kwargs)
        except Exception:
            try:
                output = model(**model_kwargs)
            except Exception as exc:
                raise DiagnosticError(f"Unable to obtain action predictions from the model: {exc}") from exc
    return extract_action_tensor(output)


def action_distance(reference: torch.Tensor, perturbed: torch.Tensor) -> np.ndarray:
    if reference.shape != perturbed.shape:
        raise DiagnosticError(f"Action tensors have incompatible shapes {reference.shape} and {perturbed.shape}")
    return torch.mean(torch.abs(reference - perturbed), dim=1).detach().cpu().numpy()


def compute_deletion_diagnostics(
    api: ProjectAPI,
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    empirical_attribution: Any,
    profile: ProfileConfig,
    seed: int,
    model_dir: Path,
    save_details: bool,
) -> Tuple[List[MetricRow], List[ValidityRow], Dict[str, Any]]:
    indices = stratified_chronological_indices(batch["dates"], profile.deletion_samples, seed + 101)
    sampled = subset_batch(batch, indices)
    states = sampled["states"]
    n_features = states.shape[-1]
    importance = raw_feature_importance(empirical_attribution, n_features)
    delete_count = max(1, int(math.ceil(n_features * profile.deletion_fraction)))
    top_indices = np.argsort(importance)[-delete_count:]
    baseline = states.mean(dim=0, keepdim=True).expand_as(states)

    reference_action = forward_action(api, model, sampled)
    guided_states = states.clone()
    guided_states[..., top_indices] = baseline[..., top_indices]
    guided_action = forward_action(api, model, sampled, guided_states)
    guided_distance = action_distance(reference_action, guided_action)

    rng = np.random.default_rng(seed + 102)
    random_distances: List[np.ndarray] = []
    random_feature_sets: List[List[int]] = []
    for _ in range(profile.random_deletion_repeats):
        random_indices = np.sort(rng.choice(n_features, size=delete_count, replace=False))
        random_states = states.clone()
        random_states[..., random_indices] = baseline[..., random_indices]
        random_action = forward_action(api, model, sampled, random_states)
        random_distances.append(action_distance(reference_action, random_action))
        random_feature_sets.append(random_indices.tolist())

    random_matrix = np.vstack(random_distances)
    random_per_sample = np.median(random_matrix, axis=0)
    eps = 1e-12
    lift = float(np.mean(guided_distance) / (np.mean(random_per_sample) + eps))
    paired_difference = guided_distance - random_per_sample
    mean_diff, ci_low, ci_high = bootstrap_mean_ci(
        paired_difference,
        profile.bootstrap_repeats,
        seed + 103,
    )
    suggested_pass = bool(np.isfinite(ci_low) and ci_low > 0)

    metric_rows = [
        make_metric_row(
            "necessity_deletion_lift_vs_random",
            lift,
            len(indices),
            "Top-ranked state-feature deletion versus matched random deletion",
            ci_low=ci_low,
            ci_high=ci_high,
            notes=f"Deleted {delete_count}/{n_features} state features; CI columns refer to guided-minus-random action distance.",
        )
    ]
    validity_rows = [
        ValidityRow(
            model_id="",
            model_family="",
            gate_id="guided_deletion_exceeds_random",
            value=mean_diff,
            suggested_pass=suggested_pass,
            status="computed",
            n_samples=len(indices),
            ci_low=ci_low,
            ci_high=ci_high,
            notes="Suggested pass requires the bootstrap lower CI of guided-minus-random action distance to exceed zero.",
        )
    ]

    if save_details:
        detail = pd.DataFrame(
            {
                "sample_index": indices,
                "date": pd.to_datetime(pd.Index(batch["dates"])[indices]),
                "guided_action_distance": guided_distance,
                "median_random_action_distance": random_per_sample,
                "paired_difference": paired_difference,
            }
        )
        write_dataframe(detail, model_dir / "deletion_sample_results.csv", model_dir / "deletion_sample_results.parquet")
        write_json(
            model_dir / "deletion_configuration.json",
            {
                "n_state_features": n_features,
                "deleted_feature_count": delete_count,
                "deleted_feature_indices": top_indices.tolist(),
                "random_feature_sets": random_feature_sets,
                "deletion_fraction": profile.deletion_fraction,
            },
        )

    return metric_rows, validity_rows, {
        "guided_mean_action_distance": float(np.mean(guided_distance)),
        "random_median_mean_action_distance": float(np.mean(random_per_sample)),
        "guided_minus_random_mean": mean_diff,
        "guided_minus_random_ci_low": ci_low,
        "guided_minus_random_ci_high": ci_high,
        "deleted_feature_count": delete_count,
    }


def summarize_temporal_curves(
    curves: np.ndarray,
    lag_positions: np.ndarray,
    recent_lags: int,
) -> Dict[str, Any]:
    array = np.abs(np.asarray(curves, dtype=float))
    if array.ndim != 3:
        raise DiagnosticError(f"Expected temporal occlusion curves with shape [sample, action, lag], got {array.shape}")
    n_samples, n_actions, n_lags = array.shape
    if len(lag_positions) != n_lags:
        lag_positions = np.arange(n_lags)

    mass = array.sum(axis=2)
    normalized = np.divide(array, mass[..., None], out=np.zeros_like(array), where=mass[..., None] > 0)
    mean_profile = normalized.mean(axis=(0, 1))

    half_lives = np.zeros((n_samples, n_actions), dtype=float)
    for sample_idx in range(n_samples):
        for action_idx in range(n_actions):
            profile = normalized[sample_idx, action_idx]
            cumulative = np.cumsum(profile)
            half_lives[sample_idx, action_idx] = float(np.searchsorted(cumulative, 0.5, side="left"))

    recent_count = min(recent_lags, n_lags)
    recency = normalized[..., :recent_count].sum(axis=2)
    return {
        "normalized_curves": normalized,
        "mean_profile": mean_profile,
        "half_lives": half_lives,
        "recency": recency,
        "mean_recency": float(np.mean(recency)),
        "mean_half_life": float(np.mean(half_lives)),
        "n_samples": n_samples,
        "n_actions": n_actions,
        "n_lags": n_lags,
    }


def compute_temporal_diagnostics(
    api: ProjectAPI,
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    context: Mapping[str, Any],
    profile: ProfileConfig,
    seed: int,
    model_dir: Path,
    save_details: bool,
) -> Tuple[List[MetricRow], Dict[str, Any]]:
    indices = stratified_chronological_indices(batch["dates"], profile.temporal_samples, seed + 201)
    sampled = subset_batch(batch, indices)
    reshaped_states, _ = call_with_supported_kwargs(
        api.postprocess_attributions,
        attributions=sampled["states"],
        feature_names=context["env_state"],
        lookback=int(context["lookback"]) + 1,
        num_indicators=len({item[1] for item in context["env_state"]})
        if context["env_state"] and isinstance(context["env_state"][0], (list, tuple))
        else len(context["env_state"]),
        num_instruments=len(set(context["instruments"]) - {"CASH"}),
    )
    reshaped_states_tensor = torch.as_tensor(reshaped_states, dtype=sampled["states"].dtype, device="cpu")
    cpu_batch = move_batch_to_device(sampled, torch.device("cpu"))
    model_cpu = model.to("cpu").eval()
    regime_labels = assign_regimes(cpu_batch["dates"])
    kwargs = {
        "model": model_cpu,
        "states": cpu_batch["states"],
        "reshaped_states": reshaped_states_tensor,
        "actions": cpu_batch["actions"],
        "returns_to_go": cpu_batch["returns_to_go"],
        "timesteps": cpu_batch["timesteps"],
        "regime_labels": regime_labels,
        "market_labels": regime_labels,
    }
    occlusion = call_with_supported_kwargs(api.compute_temporal_occlusion_curves, **kwargs)
    curves = tensor_to_numpy(occlusion["occlusion_curves"])
    lag_positions = tensor_to_numpy(occlusion.get("lag_positions", np.arange(curves.shape[-1]))).reshape(-1)
    summary = summarize_temporal_curves(curves, lag_positions, profile.recent_lags)

    windows = make_chronological_windows(len(indices), profile.chronological_windows)
    window_profiles: List[np.ndarray] = []
    window_half_lives: List[float] = []
    profile_rows: List[Dict[str, Any]] = []
    for window_index, window in enumerate(windows):
        window_curves = summary["normalized_curves"][window]
        window_profile = window_curves.mean(axis=(0, 1))
        window_profiles.append(window_profile)
        half_life = float(summary["half_lives"][window].mean())
        window_half_lives.append(half_life)
        for lag_idx, value in enumerate(window_profile):
            profile_rows.append(
                {
                    "window": window_index,
                    "lag_index": lag_idx,
                    "lag_position": finite_float(lag_positions[lag_idx]),
                    "normalized_occlusion_mass": float(value),
                    "mean_half_life": half_life,
                }
            )

    temporal_stability = pairwise_rank_concordance(window_profiles)
    half_life_mean = float(np.mean(window_half_lives)) if window_half_lives else float("nan")
    half_life_cv = (
        float(np.std(window_half_lives, ddof=1) / (abs(half_life_mean) + 1e-12))
        if len(window_half_lives) > 1
        else 0.0
    )

    metric_rows = [
        make_metric_row(
            "necessity_temporal_profile_stability",
            temporal_stability,
            len(indices),
            "Temporal occlusion; rank concordance across chronological windows",
        ),
        make_metric_row(
            "necessity_temporal_half_life_cv",
            half_life_cv,
            len(indices),
            "Temporal occlusion; half-life coefficient of variation across chronological windows",
        ),
        make_metric_row(
            "necessity_recency_index",
            summary["mean_recency"],
            len(indices),
            f"Temporal occlusion; mass in the {profile.recent_lags} most recent lags",
        ),
    ]

    if save_details:
        write_dataframe(
            pd.DataFrame(profile_rows),
            model_dir / "temporal_profiles.csv",
            model_dir / "temporal_profiles.parquet",
        )
        pd.DataFrame(
            {
                "sample_index": indices,
                "date": pd.to_datetime(pd.Index(batch["dates"])[indices]),
                "regime": regime_labels,
                "mean_half_life": summary["half_lives"].mean(axis=1),
                "mean_recency_index": summary["recency"].mean(axis=1),
            }
        ).to_csv(model_dir / "temporal_sample_summaries.csv", index=False)

    return metric_rows, {
        "temporal_sample_count": len(indices),
        "mean_half_life": summary["mean_half_life"],
        "mean_recency_index": summary["mean_recency"],
        "temporal_windows": len(windows),
    }


def move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in batch.items():
        result[key] = value.to(device) if torch.is_tensor(value) else value
    return result


def representation_to_matrix(value: Any) -> np.ndarray:
    array = tensor_to_numpy(value)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    if array.ndim == 2:
        return array
    if array.ndim >= 3:
        # The last sequence/token representation is the closest architecture-
        # neutral analogue to the final portfolio decision state.
        return array[:, -1, ...].reshape(array.shape[0], -1)
    raise DiagnosticError(f"Unsupported representation shape {array.shape}")


def standardize_matrix(matrix: np.ndarray) -> np.ndarray:
    x = np.asarray(matrix, dtype=float)
    means = np.nanmean(x, axis=0)
    stds = np.nanstd(x, axis=0)
    stds[stds < 1e-12] = 1.0
    standardized = (np.nan_to_num(x, nan=means) - means) / stds
    return standardized


def knn_sets(matrix: np.ndarray, k: int) -> List[set[int]]:
    n = len(matrix)
    if n < 3:
        return [set() for _ in range(n)]
    k_eff = max(1, min(k, n - 1))
    neighbors = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean").fit(matrix)
    indices = neighbors.kneighbors(return_distance=False)
    return [set(row[row != index][:k_eff].tolist()) for index, row in enumerate(indices)]


def mean_jaccard(first: Sequence[set[int]], second: Sequence[set[int]], indices: Optional[np.ndarray] = None) -> float:
    if indices is None:
        indices = np.arange(min(len(first), len(second)))
    values: List[float] = []
    for index in indices:
        a = first[int(index)]
        b = second[int(index)]
        union = a | b
        if union:
            values.append(len(a & b) / len(union))
    return float(np.mean(values)) if values else float("nan")


def neighbourhood_persistence(
    representations: Mapping[str, np.ndarray],
    k: int,
    bootstrap_repeats: int,
    seed: int,
) -> Tuple[float, Dict[str, Any]]:
    layers = list(representations)
    standardized = {name: standardize_matrix(representations[name]) for name in layers}
    knn = {name: knn_sets(matrix, k) for name, matrix in standardized.items()}

    adjacent_scores: List[float] = []
    for first, second in zip(layers[:-1], layers[1:]):
        adjacent_scores.append(mean_jaccard(knn[first], knn[second]))

    deepest = layers[-1]
    full_sets = knn[deepest]
    n = len(standardized[deepest])
    rng = np.random.default_rng(seed)
    bootstrap_scores: List[float] = []
    if n >= 6:
        subset_size = max(k + 2, int(round(0.80 * n)))
        subset_size = min(subset_size, n)
        for _ in range(bootstrap_repeats):
            subset = np.sort(rng.choice(n, size=subset_size, replace=False))
            sub_matrix = standardized[deepest][subset]
            sub_sets_local = knn_sets(sub_matrix, k)
            global_to_local = {global_index: local_index for local_index, global_index in enumerate(subset)}
            scores: List[float] = []
            subset_set = set(subset.tolist())
            for global_index in subset:
                local_index = global_to_local[int(global_index)]
                full_restricted = full_sets[int(global_index)] & subset_set
                sub_global = {int(subset[j]) for j in sub_sets_local[local_index]}
                union = full_restricted | sub_global
                if union:
                    scores.append(len(full_restricted & sub_global) / len(union))
            if scores:
                bootstrap_scores.append(float(np.mean(scores)))

    components = [score for score in adjacent_scores + bootstrap_scores if np.isfinite(score)]
    persistence = float(np.mean(components)) if components else float("nan")
    return persistence, {
        "adjacent_layer_scores": adjacent_scores,
        "bootstrap_scores": bootstrap_scores,
        "layers": layers,
        "k": k,
    }


def effective_rank(matrix: np.ndarray) -> float:
    x = standardize_matrix(matrix)
    x = x - x.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(x, full_matrices=False, compute_uv=False)
    energy = singular**2
    if energy.sum() <= 0 or len(energy) <= 1:
        return float("nan")
    p = energy / energy.sum()
    entropy = -np.sum(p[p > 0] * np.log(p[p > 0]))
    return float(np.exp(entropy) / len(energy))


def flatten_conductance(value: Any) -> pd.Series:
    if isinstance(value, pd.Series):
        return value.astype(float)
    if isinstance(value, pd.DataFrame):
        numeric = value.select_dtypes(include=[np.number])
        if numeric.empty:
            raise DiagnosticError("Layer-conductance DataFrame contains no numeric columns")
        return numeric.abs().mean(axis=0)
    if isinstance(value, Mapping):
        rows: Dict[str, float] = {}
        for key, item in value.items():
            array = tensor_to_numpy(item).astype(float)
            rows[str(key)] = float(np.nanmean(np.abs(array)))
        return pd.Series(rows, dtype=float)
    array = tensor_to_numpy(value).astype(float)
    if array.ndim == 0:
        return pd.Series({"layer_0": float(abs(array))})
    if array.ndim == 1:
        return pd.Series(array, index=[f"layer_{i}" for i in range(len(array))], dtype=float)
    flattened = np.nanmean(np.abs(array), axis=tuple(range(1, array.ndim)))
    return pd.Series(flattened, index=[f"layer_{i}" for i in range(len(flattened))], dtype=float)


def compute_structure_diagnostics(
    api: ProjectAPI,
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    profile: ProfileConfig,
    seed: int,
    model_dir: Path,
    save_details: bool,
) -> Tuple[List[MetricRow], Dict[str, Any]]:
    indices = stratified_chronological_indices(batch["dates"], profile.structure_samples, seed + 301)
    sampled = move_batch_to_device(subset_batch(batch, indices), torch.device("cpu"))
    model_cpu = model.to("cpu").eval()

    representations_raw = call_with_supported_kwargs(
        api.extract_layer_representations,
        model=model_cpu,
        states=sampled["states"],
        actions=sampled["actions"],
        returns_to_go=sampled["returns_to_go"],
        timesteps=sampled["timesteps"],
        attention_mask=sampled.get("attention_mask"),
    )
    if not isinstance(representations_raw, Mapping) or not representations_raw:
        raise DiagnosticError("extract_layer_representations returned no layer mapping")
    representations = {str(key): representation_to_matrix(value) for key, value in representations_raw.items()}
    persistence, persistence_detail = neighbourhood_persistence(
        representations,
        profile.knn_k,
        profile.bootstrap_repeats,
        seed + 302,
    )
    deepest_rank = effective_rank(representations[list(representations)[-1]])

    conductance_raw = call_with_supported_kwargs(
        api.compute_layer_conductance,
        model=model_cpu,
        states=sampled["states"],
        actions=sampled["actions"],
        returns_to_go=sampled["returns_to_go"],
        timesteps=sampled["timesteps"],
        attention_mask=sampled.get("attention_mask"),
    )
    conductance = flatten_conductance(conductance_raw).abs()
    conductance_evenness = normalized_entropy(conductance.to_numpy(dtype=float))
    shares = conductance / conductance.sum() if conductance.sum() > 0 else conductance * np.nan

    metric_rows = [
        make_metric_row(
            "structure_neighbourhood_persistence",
            persistence,
            len(indices),
            "Adjacent-layer and bootstrap k-nearest-neighbour overlap",
        ),
        make_metric_row(
            "structure_conductance_evenness",
            conductance_evenness,
            len(indices),
            "Normalized entropy of absolute layer-conductance shares",
        ),
        make_metric_row(
            "structure_effective_rank",
            deepest_rank,
            len(indices),
            "Normalized effective rank of the deepest extracted representation",
        ),
    ]

    if save_details:
        layer_rows: List[Dict[str, Any]] = []
        for layer_name, matrix in representations.items():
            layer_rows.append(
                {
                    "layer": layer_name,
                    "n_samples": matrix.shape[0],
                    "representation_dimension": matrix.shape[1],
                    "effective_rank": effective_rank(matrix),
                    "conductance_magnitude": finite_float(conductance.get(layer_name, np.nan)),
                    "conductance_share": finite_float(shares.get(layer_name, np.nan)),
                }
            )
        write_dataframe(pd.DataFrame(layer_rows), model_dir / "structure_layer_profiles.csv", model_dir / "structure_layer_profiles.parquet")
        write_json(model_dir / "neighbourhood_persistence_details.json", persistence_detail)

    return metric_rows, {
        "structure_sample_count": len(indices),
        "layer_count": len(representations),
        "conductance_layer_count": len(conductance),
        "neighbourhood_persistence": persistence,
        "conductance_evenness": conductance_evenness,
    }


def reset_last_parameterized_modules(model: torch.nn.Module, module_count: int = 3) -> List[str]:
    candidates: List[Tuple[str, torch.nn.Module]] = []
    for name, module in model.named_modules():
        if name and hasattr(module, "reset_parameters") and callable(module.reset_parameters):
            candidates.append((name, module))
    selected = candidates[-module_count:]
    for _, module in selected:
        module.reset_parameters()
    return [name for name, _ in selected]


def compute_parameter_randomization_validity(
    api: ProjectAPI,
    record: ModelRecord,
    sampled_batch: Mapping[str, Any],
    context: Mapping[str, Any],
    original_profile: pd.Series,
    profile: ProfileConfig,
    seed: int,
) -> ValidityRow:
    try:
        random_model = api.TrainableDT.from_pretrained(str(record.model_path)).to("cpu").eval()
        torch.manual_seed(seed + 401)
        reset_modules = reset_last_parameterized_modules(random_model, module_count=3)
        if not reset_modules:
            raise DiagnosticError("No resettable parameterized modules were found")
        cpu_batch = move_batch_to_device(sampled_batch, torch.device("cpu"))
        baseline = cpu_batch["states"].mean(dim=0, keepdim=True).expand_as(cpu_batch["states"]).clone()
        randomized_attr = compute_ig(api, random_model, cpu_batch, baseline, profile.sanity_ig_steps)
        randomized_matrix, randomized_names = postprocess_attribution(api, randomized_attr, context)
        randomized_profile = aggregate_feature_family_profile(randomized_matrix, randomized_names, signed=False)
        labels, aligned = align_profiles({"original": original_profile, "randomized": randomized_profile})
        tau = stats.kendalltau(aligned["original"], aligned["randomized"], variant="b").statistic
        sensitivity = float(1.0 - abs(tau)) if np.isfinite(tau) else float("nan")
        suggested_pass = bool(np.isfinite(sensitivity) and sensitivity >= 0.20)
        del random_model
        gc.collect()
        return ValidityRow(
            model_id="",
            model_family="",
            gate_id="parameter_randomization_changes_attribution",
            value=finite_float(sensitivity),
            suggested_pass=suggested_pass,
            status="computed" if np.isfinite(sensitivity) else "not_computable",
            n_samples=len(cpu_batch["dates"]),
            ci_low=None,
            ci_high=None,
            notes=(
                "Value is 1-|Kendall tau| between original and randomized feature-family rankings. "
                f"Reset modules: {reset_modules}. The 0.20 threshold is a documented candidate, not a universal rule."
            ),
        )
    except Exception as exc:
        return ValidityRow(
            model_id="",
            model_family="",
            gate_id="parameter_randomization_changes_attribution",
            value=None,
            suggested_pass=None,
            status="failed_to_compute",
            n_samples=len(sampled_batch["dates"]),
            ci_low=None,
            ci_high=None,
            notes=str(exc),
        )


def compute_input_permutation_validity(
    api: ProjectAPI,
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    seed: int,
) -> ValidityRow:
    states = batch["states"]
    if states.shape[0] < 2:
        return ValidityRow("", "", "input_permutation_changes_action", None, None, "not_computable", len(batch["dates"]), None, None, "Fewer than two samples")
    reference = forward_action(api, model, batch)
    permuted = states.clone()
    rng = np.random.default_rng(seed + 501)
    for feature_index in range(states.shape[-1]):
        order = torch.as_tensor(rng.permutation(states.shape[0]), device=states.device, dtype=torch.long)
        permuted[..., feature_index] = states.index_select(0, order)[..., feature_index]
    changed = forward_action(api, model, batch, permuted)
    distances = action_distance(reference, changed)
    mean_value, ci_low, ci_high = bootstrap_mean_ci(distances, 100, seed + 502)
    return ValidityRow(
        model_id="",
        model_family="",
        gate_id="input_permutation_changes_action",
        value=finite_float(mean_value),
        suggested_pass=None,
        status="computed",
        n_samples=len(batch["dates"]),
        ci_low=finite_float(ci_low),
        ci_high=finite_float(ci_high),
        notes="Mean absolute action change after independent feature-wise permutation across sampled dates; no universal threshold imposed.",
    )


def attach_identifiers(
    rows: Iterable[Any],
    model_id: str,
    model_family: str,
) -> None:
    for row in rows:
        row.model_id = model_id
        row.model_family = model_family


def config_hash(args: argparse.Namespace, profile: ProfileConfig) -> str:
    payload = {
        "validation_start": str(VALIDATION_START.date()),
        "validation_end": str(VALIDATION_END.date()),
        "profile": asdict(profile),
        "seed": args.seed,
        "run_parameter_randomization": args.run_parameter_randomization,
        "metric_registry": METRIC_REGISTRY,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def process_one_model(
    api: ProjectAPI,
    record: ModelRecord,
    args: argparse.Namespace,
    profile: ProfileConfig,
    attribution_device: torch.device,
    run_hash: str,
) -> Dict[str, Any]:
    model_dir = args.output_dir / "per_model" / safe_id(record.model_id)
    model_dir.mkdir(parents=True, exist_ok=True)
    summary_path = model_dir / "summary.json"
    if args.resume and not args.force and summary_path.exists():
        try:
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
            if existing.get("run_hash") == run_hash and existing.get("status") == "completed":
                LOGGER.info("Skipping completed model %s", record.model_id)
                return existing
        except Exception:
            pass

    model_seed = stable_seed(args.seed, record.model_id)
    set_global_seed(model_seed)
    started = time.time()
    metric_rows: List[MetricRow] = []
    validity_rows: List[ValidityRow] = []
    stage_metadata: Dict[str, Any] = {}

    model = api.TrainableDT.from_pretrained(str(record.model_path)).to(attribution_device).eval()
    batch, context = prepare_model_batch(api, record, model, attribution_device)
    stage_metadata["validation_observation_count"] = len(batch["dates"])
    stage_metadata["validation_effective_start"] = str(pd.to_datetime(batch["dates"]).min().date())
    stage_metadata["validation_effective_end"] = str(pd.to_datetime(batch["dates"]).max().date())
    stage_metadata["lookback"] = int(context["lookback"])
    stage_metadata["rebalance_every"] = int(context["rebalance_every"])

    contribution_rows, contribution_extras, contribution_meta = compute_contribution_diagnostics(
        api,
        model,
        batch,
        context,
        profile,
        model_seed,
        model_dir,
        args.save_details,
    )
    metric_rows.extend(contribution_rows)
    stage_metadata.update(contribution_meta)
    validity_rows.append(
        ValidityRow(
            model_id="",
            model_family="",
            gate_id="baseline_concordance_available",
            value=finite_float(contribution_extras["baseline_concordance"]),
            suggested_pass=bool(np.isfinite(contribution_extras["baseline_concordance"])),
            status="computed" if np.isfinite(contribution_extras["baseline_concordance"]) else "not_computable",
            n_samples=len(contribution_extras["attribution_indices"]),
            ci_low=None,
            ci_high=None,
            notes="The scalar value is the mean pairwise Kendall tau across three attribution baselines.",
        )
    )

    deletion_rows, deletion_validity, deletion_meta = compute_deletion_diagnostics(
        api,
        model,
        batch,
        contribution_extras["empirical_attribution"],
        profile,
        model_seed,
        model_dir,
        args.save_details,
    )
    metric_rows.extend(deletion_rows)
    validity_rows.extend(deletion_validity)
    stage_metadata.update(deletion_meta)

    permutation_batch = contribution_extras["sampled_batch"]
    validity_rows.append(compute_input_permutation_validity(api, model, permutation_batch, model_seed))

    if args.run_parameter_randomization:
        validity_rows.append(
            compute_parameter_randomization_validity(
                api,
                record,
                contribution_extras["sampled_batch"],
                context,
                contribution_extras["empirical_profile"],
                profile,
                model_seed,
            )
        )
    else:
        validity_rows.append(
            ValidityRow(
                model_id="",
                model_family="",
                gate_id="parameter_randomization_changes_attribution",
                value=None,
                suggested_pass=None,
                status="disabled",
                n_samples=None,
                ci_low=None,
                ci_high=None,
                notes="Disabled by CLI configuration.",
            )
        )

    # Temporal and structure utilities are most compatible with CPU in the existing repository.
    model = model.to("cpu").eval()
    batch_cpu = move_batch_to_device(batch, torch.device("cpu"))
    temporal_rows, temporal_meta = compute_temporal_diagnostics(
        api,
        model,
        batch_cpu,
        context,
        profile,
        model_seed,
        model_dir,
        args.save_details,
    )
    metric_rows.extend(temporal_rows)
    stage_metadata.update(temporal_meta)

    structure_rows, structure_meta = compute_structure_diagnostics(
        api,
        model,
        batch_cpu,
        profile,
        model_seed,
        model_dir,
        args.save_details,
    )
    metric_rows.extend(structure_rows)
    stage_metadata.update(structure_meta)

    attach_identifiers(metric_rows, record.model_id, record.model_family)
    attach_identifiers(validity_rows, record.model_id, record.model_family)
    metric_df = pd.DataFrame([asdict(row) for row in metric_rows])
    validity_df = pd.DataFrame([asdict(row) for row in validity_rows])
    write_dataframe(metric_df, model_dir / "diagnostics_model_long.csv", model_dir / "diagnostics_model_long.parquet")
    write_dataframe(validity_df, model_dir / "diagnostic_validity_model.csv", model_dir / "diagnostic_validity_model.parquet")

    summary = {
        "run_hash": run_hash,
        "status": "completed",
        "model_id": record.model_id,
        "model_family": record.model_family,
        "mapping_method": record.mapping_method,
        "model_path": str(record.model_path),
        "validation_start": str(VALIDATION_START.date()),
        "validation_end": str(VALIDATION_END.date()),
        "profile": asdict(profile),
        "metric_count": len(metric_rows),
        "validity_gate_count": len(validity_rows),
        "stage_metadata": stage_metadata,
        "duration_seconds": time.time() - started,
    }
    write_json(summary_path, summary)

    del model, batch, batch_cpu
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def read_per_model_outputs(output_dir: Path, run_hash: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_frames: List[pd.DataFrame] = []
    validity_frames: List[pd.DataFrame] = []
    summaries: List[Dict[str, Any]] = []
    for model_dir in sorted((output_dir / "per_model").glob("*")):
        if not model_dir.is_dir():
            continue
        summary_path = model_dir / "summary.json"
        metric_path = model_dir / "diagnostics_model_long.csv"
        validity_path = model_dir / "diagnostic_validity_model.csv"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("run_hash") != run_hash or summary.get("status") != "completed":
            continue
        summaries.append(summary)
        if metric_path.exists():
            metric_frames.append(pd.read_csv(metric_path))
        if validity_path.exists():
            validity_frames.append(pd.read_csv(validity_path))
    metrics = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    validity = pd.concat(validity_frames, ignore_index=True) if validity_frames else pd.DataFrame()
    summary_df = pd.json_normalize(summaries)
    return metrics, validity, summary_df


def iqr(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    return float(clean.quantile(0.75) - clean.quantile(0.25))


def aggregate_family_metrics(model_metrics: pd.DataFrame) -> pd.DataFrame:
    if model_metrics.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    group_columns = [
        "model_family",
        "dimension",
        "metric_id",
        "preferred_direction",
        "role",
        "architecture_scope",
        "source_method",
    ]
    for keys, group in model_metrics.groupby(group_columns, dropna=False, sort=True):
        values = pd.to_numeric(group["metric_value"], errors="coerce")
        valid = values.dropna()
        row = dict(zip(group_columns, keys))
        row.update(
            {
                "raw_family_value": float(valid.median()) if not valid.empty else np.nan,
                "family_iqr": iqr(valid),
                "n_individual_models": int(group["model_id"].nunique()),
                "n_valid_model_values": int(valid.notna().sum()),
                "n_total_model_values": int(len(group)),
                "status": "computed" if not valid.empty else "not_computable",
                "notes": "Median and IQR across loaded checkpoint models mapped to the family.",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_family_validity(model_validity: pd.DataFrame) -> pd.DataFrame:
    if model_validity.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for (family, gate_id), group in model_validity.groupby(["model_family", "gate_id"], sort=True):
        values = pd.to_numeric(group["value"], errors="coerce").dropna()
        suggested = group["suggested_pass"].dropna()
        rows.append(
            {
                "model_family": family,
                "gate_id": gate_id,
                "raw_family_value": float(values.median()) if not values.empty else np.nan,
                "family_iqr": iqr(values),
                "n_individual_models": int(group["model_id"].nunique()),
                "n_computed_values": int(len(values)),
                "suggested_pass_rate": float(pd.to_numeric(suggested, errors="coerce").mean()) if len(suggested) else np.nan,
                "all_suggested_pass": bool(pd.to_numeric(suggested, errors="coerce").all()) if len(suggested) else None,
                "status": "computed" if not values.empty or len(suggested) else "not_computable",
                "notes": "Family summary of model-level validity evidence; Batch A determines final gate policy.",
            }
        )
    return pd.DataFrame(rows)


def pivot_summary(long_df: pd.DataFrame, value_column: str, index_columns: Sequence[str]) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame()
    pivot = long_df.pivot_table(index=list(index_columns), columns="metric_id", values=value_column, aggfunc="first")
    pivot.columns.name = None
    return pivot.reset_index()


def generate_manifest(
    args: argparse.Namespace,
    profile: ProfileConfig,
    run_hash: str,
    model_metrics: pd.DataFrame,
    family_metrics: pd.DataFrame,
    model_validity: pd.DataFrame,
    mapping_report: pd.DataFrame,
    failures: pd.DataFrame,
    mapping_source: Path,
) -> Dict[str, Any]:
    coverage: Dict[str, Any] = {}
    for entry in METRIC_REGISTRY:
        metric_id = entry["metric_id"]
        subset = family_metrics[family_metrics["metric_id"] == metric_id] if not family_metrics.empty else pd.DataFrame()
        coverage[metric_id] = {
            "dimension": entry["dimension"],
            "role": entry["role"],
            "family_count": int(subset["model_family"].nunique()) if not subset.empty else 0,
            "computed_family_count": int(subset["raw_family_value"].notna().sum()) if not subset.empty else 0,
        }
    return {
        "run_hash": run_hash,
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "validation_start": str(VALIDATION_START.date()),
        "validation_end": str(VALIDATION_END.date()),
        "holdout_used": False,
        "profile_name": args.profile,
        "profile": asdict(profile),
        "seed": args.seed,
        "paths": {
            "analysis_repo_root": str(args.analysis_repo_root),
            "model_code_root": str(args.model_code_root),
            "models_path": str(args.models_path),
            "metrics_dir": str(args.metrics_dir),
            "output_dir": str(args.output_dir),
            "family_mapping_source": str(mapping_source),
            "model_assessment_pickle": str(args.model_assessment_pickle),
        },
        "counts": {
            "discovered_models": int(len(mapping_report)),
            "resolved_models": int(mapping_report["mapping_resolved"].fillna(False).sum()) if not mapping_report.empty else 0,
            "completed_models": int(model_metrics["model_id"].nunique()) if not model_metrics.empty else 0,
            "completed_families": int(family_metrics["model_family"].nunique()) if not family_metrics.empty else 0,
            "failed_models": int(len(failures)),
        },
        "metric_coverage": coverage,
        "metric_registry": METRIC_REGISTRY,
        "validity_registry": VALIDITY_REGISTRY,
        "methodological_notes": [
            "All diagnostics use validation-period data only.",
            "All model checkpoints are processed independently of performance ranking.",
            "Diagnostic metrics assess internal stability and faithfulness; they do not predict later Sharpe ratios.",
            "Family values are medians across loaded checkpoint models mapped to each family; IQR is retained.",
            "Attention and future-return latent analyses are not required for the architecture-neutral audit score and are intentionally omitted.",
        ],
    }


def write_readme(output_dir: Path, manifest: Mapping[str, Any]) -> None:
    counts = manifest["counts"]
    content = f"""# Batch A diagnostic inputs

This directory was generated by `generate_batch_a_diagnostics.py`.

## Temporal scope

- Training ends: 2015-12-31
- Validation and diagnostic calibration: 2016-01-01 to 2024-04-30
- Holdout outcomes used by this script: **No**

## Coverage

- Completed checkpoint models: {counts['completed_models']}
- Completed model families: {counts['completed_families']}
- Failed models: {counts['failed_models']}

## Principal files

- `diagnostics_model_long.csv` / `.parquet`: one row per checkpoint and diagnostic metric.
- `diagnostics_family_long.csv` / `.parquet`: family medians and IQRs in the exact long format consumed by Batch A.
- `diagnostics_model_summary.csv` / `.parquet`: wide checkpoint-level metric table.
- `diagnostics_family_summary.csv` / `.parquet`: wide family-level metric table.
- `diagnostic_validity_model.csv` / `.parquet`: model-level validity evidence.
- `diagnostic_validity_family.csv` / `.parquet`: family-level validity summaries.
- `diagnostic_metric_registry.yaml`: fixed metric roles and directions.
- `diagnostic_validity_registry.yaml`: candidate validity-gate definitions.
- `diagnostic_manifest.json`: configuration, coverage, and provenance.
- `diagnostic_failures.csv`: model-local failures without fabricated replacement values.
- `family_mapping_report.csv`: checkpoint-to-family reconciliation.

## Interpretation boundary

These outputs support a validation-stage audit of contribution, necessity, and structure. They do not constitute a forecast of later performance and must not be recalibrated using the May-2024--June-2026 holdout.
"""
    (output_dir / "README.md").write_text(content, encoding="utf-8")


def main() -> int:
    args = parse_args()
    configure_logging(args.output_dir, args.log_level)
    profile = PROFILES[args.profile]
    run_hash = config_hash(args, profile)
    set_global_seed(args.seed)
    attribution_device = resolve_device(args.device)

    LOGGER.info("Run hash: %s", run_hash)
    LOGGER.info("Attribution device: %s", attribution_device)
    LOGGER.info("Validation period: %s to %s", VALIDATION_START.date(), VALIDATION_END.date())
    LOGGER.info("Holdout files are not loaded by this script")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "per_model").mkdir(parents=True, exist_ok=True)
    write_yaml_compatible(args.output_dir / "diagnostic_metric_registry.yaml", METRIC_REGISTRY)
    write_yaml_compatible(args.output_dir / "diagnostic_validity_registry.yaml", VALIDITY_REGISTRY)
    write_json(
        args.output_dir / "diagnostic_run_config.json",
        {
            "run_hash": run_hash,
            "args": vars(args),
            "profile": asdict(profile),
            "validation_start": VALIDATION_START,
            "validation_end": VALIDATION_END,
            "regime_boundaries": REGIME_BOUNDARIES,
        },
    )

    api = load_project_api(args.analysis_repo_root, args.model_code_root)
    family_mapping, family_pairs, mapping_source = build_family_mapping(args.metrics_dir, args.family_map_csv)
    records, mapping_report = discover_models(
        api,
        args.models_path,
        family_mapping,
        args.strict_family_mapping,
    )
    write_dataframe(mapping_report, args.output_dir / "family_mapping_report.csv", args.output_dir / "family_mapping_report.parquet")

    if args.model_id:
        requested = set(args.model_id)
        records = [record for record in records if record.model_id in requested]
        missing_requested = requested - {record.model_id for record in records}
        if missing_requested:
            raise ValueError(f"Requested model IDs were not discovered: {sorted(missing_requested)}")
    if args.max_models is not None:
        records = records[: args.max_models]
    if not records:
        raise RuntimeError("No model checkpoints were selected")

    failures: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    for record in tqdm(records, desc="Batch A diagnostics", unit="model"):
        LOGGER.info("Processing model=%s family=%s", record.model_id, record.model_family)
        try:
            summary = process_one_model(api, record, args, profile, attribution_device, run_hash)
            summaries.append(summary)
        except Exception as exc:
            LOGGER.error("Model %s failed: %s", record.model_id, exc)
            LOGGER.debug("Traceback for %s:\n%s", record.model_id, traceback.format_exc())
            failure = {
                "model_id": record.model_id,
                "model_family": record.model_family,
                "model_path": str(record.model_path),
                "stage": "model_processing",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            failure_dir = args.output_dir / "per_model" / safe_id(record.model_id)
            failure_dir.mkdir(parents=True, exist_ok=True)
            write_json(
                failure_dir / "summary.json",
                {
                    "run_hash": run_hash,
                    "status": "failed",
                    **failure,
                },
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    failure_df = pd.DataFrame(failures)
    if failure_df.empty:
        failure_df = pd.DataFrame(
            columns=["model_id", "model_family", "model_path", "stage", "error_type", "error_message", "traceback"]
        )
    failure_df.to_csv(args.output_dir / "diagnostic_failures.csv", index=False)

    model_metrics, model_validity, summary_df = read_per_model_outputs(args.output_dir, run_hash)
    if model_metrics.empty:
        LOGGER.error("No model diagnostics were successfully produced")
        return 2

    family_metrics = aggregate_family_metrics(model_metrics)
    family_validity = aggregate_family_validity(model_validity)
    model_summary = pivot_summary(model_metrics, "metric_value", ["model_id", "model_family"])
    family_summary = pivot_summary(family_metrics, "raw_family_value", ["model_family"])

    write_dataframe(model_metrics, args.output_dir / "diagnostics_model_long.csv", args.output_dir / "diagnostics_model_long.parquet")
    write_dataframe(family_metrics, args.output_dir / "diagnostics_family_long.csv", args.output_dir / "diagnostics_family_long.parquet")
    write_dataframe(model_summary, args.output_dir / "diagnostics_model_summary.csv", args.output_dir / "diagnostics_model_summary.parquet")
    write_dataframe(family_summary, args.output_dir / "diagnostics_family_summary.csv", args.output_dir / "diagnostics_family_summary.parquet")
    write_dataframe(model_validity, args.output_dir / "diagnostic_validity_model.csv", args.output_dir / "diagnostic_validity_model.parquet")
    write_dataframe(family_validity, args.output_dir / "diagnostic_validity_family.csv", args.output_dir / "diagnostic_validity_family.parquet")
    write_dataframe(summary_df, args.output_dir / "model_processing_summary.csv", args.output_dir / "model_processing_summary.parquet")

    manifest = generate_manifest(
        args,
        profile,
        run_hash,
        model_metrics,
        family_metrics,
        model_validity,
        mapping_report,
        failure_df,
        mapping_source,
    )
    write_json(args.output_dir / "diagnostic_manifest.json", manifest)
    write_readme(args.output_dir, manifest)

    expected_core = {
        entry["metric_id"]
        for entry in METRIC_REGISTRY
        if entry["role"] == "core_score"
    }
    computed_core = set(
        family_metrics.loc[family_metrics["raw_family_value"].notna(), "metric_id"].astype(str)
    )
    missing_core = sorted(expected_core - computed_core)
    completed_families = family_metrics["model_family"].nunique()
    LOGGER.info("Completed models: %d", model_metrics["model_id"].nunique())
    LOGGER.info("Completed families: %d", completed_families)
    LOGGER.info("Core metrics available: %d/%d", len(expected_core) - len(missing_core), len(expected_core))
    if missing_core:
        LOGGER.warning("Missing core metrics: %s", missing_core)
    if completed_families != 18 and args.max_models is None and not args.model_id:
        LOGGER.warning("Expected 18 families for the publication run, obtained %d", completed_families)
    LOGGER.info("Outputs: %s", args.output_dir)
    return 0 if not missing_core else 3


if __name__ == "__main__":
    raise SystemExit(main())
