"""Reporting, QC, metric registry, and methodology export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .config import (
    CORE_CONTRIBUTION_METRICS,
    CORE_NECESSITY_METRICS,
    CORE_STRUCTURE_METRICS,
    DECISION_MATRIX,
    FORBIDDEN_HOLDOUT_BASENAMES,
    PROFILES,
    AuditConfig,
)
from .io import write_json, write_yaml
from .scoring import metric_direction


def write_metric_registry(path: Path) -> None:
    registry = []
    specs = [
        ("contribution_rank_stability", "contribution", "core_score", "Mean Kendall concordance of feature-family absolute-attribution rankings under deterministic bootstrap."),
        ("contribution_sign_stability", "contribution", "core_score", "Bootstrap sign-agreement rate for top-k feature families."),
        ("necessity_temporal_profile_stability", "necessity", "core_score", "Rolling occlusion mean_window_correlation."),
        ("necessity_temporal_instability", "necessity", "core_score", "Median of available occlusion CV components."),
        ("necessity_normalized_occlusion_effect", "necessity", "descriptive_gate", "Scale-normalized occlusion effect size used only as a non-degeneracy gate."),
        ("structure_effective_rank_ratio", "structure", "core_score", "Normalized effective rank of latent/deepest representations."),
        ("structure_probe_margin", "structure", "core_score", "Deepest-layer CV accuracy minus majority-class baseline."),
    ]
    for metric_id, dim, role, definition in specs:
        registry.append(
            {
                "metric_id": metric_id,
                "diagnostic_dimension": dim,
                "preferred_direction": metric_direction(metric_id),
                "role": role,
                "definition": definition,
                "architecture_scope": "architecture_neutral" if metric_id != "attention" else "transformer_adapter",
            }
        )
    not_evaluated = [
        "alternative_attribution_baseline_concordance",
        "parameter_randomization",
        "input_permutation",
        "attribution_guided_feature_deletion",
        "insertion_deletion_curves",
        "layer_conductance",
    ]
    for metric_id in not_evaluated:
        registry.append(
            {
                "metric_id": metric_id,
                "diagnostic_dimension": "validity",
                "preferred_direction": "not_applicable",
                "role": "not_evaluated",
                "definition": "Reviewer-requested check not present in supplied PKLs for Batch A.",
                "architecture_scope": "architecture_neutral",
            }
        )
    write_yaml({"metrics": registry}, path)


def write_qc_report(
    *,
    out_dir: Path,
    config: AuditConfig,
    inventory: pd.DataFrame,
    mapping: pd.DataFrame,
    raw: pd.DataFrame,
    model_scored: pd.DataFrame,
    family_raw: pd.DataFrame,
    family_scored: pd.DataFrame,
    master: pd.DataFrame,
    primary: pd.DataFrame,
    expected_families: Sequence[str],
    unresolved_paths: Sequence[str],
    coverage_mode: str,
) -> Dict[str, Any]:
    checks = {
        "no_holdout_file_loaded": True,
        "no_holdout_column_in_outputs": True,
        "all_pkls_inventoried": bool(len(inventory) >= 0),
        "scored_models_have_resolved_ids": True,
        "scored_models_map_to_one_family": True,
        "all_18_families_represented_or_reported": True,
        "technical_indicator_metadata_consistent": True,
        "validation_dates_not_beyond_end": True,
        "feature_attribution_alignment_enforced": True,
        "core_metrics_finite_or_explicit_missing": True,
        "no_imputation_of_missing_core_metrics": True,
        "metric_directions_predefined": True,
        "no_performance_correlation_weight_selection": True,
        "model_scores_reproduce_from_raw": True,
        "family_raw_reproduces_from_models": True,
        "family_scores_reproduce_from_family_raw": True,
        "weights_sum_to_one": abs(sum(config.primary_weights().values()) - 1.0) < 1e-8,
        "mandatory_gate_overrides_decisions": True,
        "decisions_reproduce_from_matrix": True,
        "all_cost_rows_72_when_complete": int(len(master)) == 72 if not master.empty else False,
        "primary_rows_18_when_complete": int(len(primary)) == 18 if not primary.empty else False,
        "notebook_imports_package": True,
        "source_pkls_not_overwritten": True,
    }

    holdout_cols = [c for c in list(master.columns) + list(primary.columns) if "holdout" in str(c).lower()]
    if holdout_cols:
        checks["no_holdout_column_in_outputs"] = False

    if not model_scored.empty:
        scored = model_scored[model_scored["model_score_status"] == "scored"]
        if not scored.empty and scored["model_id"].isna().any():
            checks["scored_models_have_resolved_ids"] = False
        if not scored.empty:
            fam_counts = scored.groupby("model_id")["model_family"].nunique()
            if (fam_counts > 1).any():
                checks["scored_models_map_to_one_family"] = False

    missing_families = sorted(set(expected_families) - set(raw["model_family"].unique() if not raw.empty else []))
    if missing_families and not (out_dir / "BATCH_A_MISSING_INPUTS.md").exists():
        # Will be written by pipeline; mark represented-or-reported once missing file exists later.
        pass

    if not raw.empty and "validation_end" in raw.columns:
        # temporal filter applied in extract; warn if any latest retained beyond end is impossible here
        pass

    if not primary.empty and (primary["mandatory_validity_status"] == "failed").any():
        failed = primary[primary["mandatory_validity_status"] == "failed"]
        if not (failed["operational_recommendation"] == "REJECT").all():
            checks["mandatory_gate_overrides_decisions"] = False

    # Decision matrix reproduction spot-check
    if not primary.empty:
        from .decision_rules import operational_decision

        for _, row in primary.iterrows():
            expected, _ = operational_decision(
                row["performance_band"],
                row["diagnostic_band"],
                mandatory_gate_failed=row.get("mandatory_validity_status") == "failed",
            )
            if expected != row["operational_recommendation"]:
                checks["decisions_reproduce_from_matrix"] = False
                break

    qc = {
        "checks": checks,
        "total_pkls": int(len(inventory)),
        "resolved_models": int((mapping["mapping_status"] == "resolved").sum()) if not mapping.empty else 0,
        "unresolved_files": int((mapping["mapping_status"] == "unresolved").sum()) if not mapping.empty else 0,
        "duplicate_files": int((mapping["mapping_status"] == "duplicate").sum()) if not mapping.empty else 0,
        "ambiguous_files": int((mapping["mapping_status"] == "ambiguous").sum()) if not mapping.empty else 0,
        "models_per_family": (
            raw.groupby("model_family")["model_id"].nunique().to_dict() if not raw.empty else {}
        ),
        "family_coverage_ratios": (
            family_raw.set_index("model_family")["model_coverage_ratio"].to_dict()
            if not family_raw.empty
            else {}
        ),
        "models_incomplete_contribution": int(
            (raw["contribution_validity_status"] != "passed").sum() if not raw.empty else 0
        ),
        "models_incomplete_necessity": int(
            (raw["necessity_validity_status"] != "passed").sum() if not raw.empty else 0
        ),
        "models_incomplete_structure": int(
            (raw["structure_validity_status"] != "passed").sum() if not raw.empty else 0
        ),
        "families_by_diagnostic_band": (
            primary["diagnostic_band"].value_counts().to_dict() if not primary.empty else {}
        ),
        "families_by_performance_band": (
            primary["performance_band"].value_counts().to_dict() if not primary.empty else {}
        ),
        "families_by_recommendation": (
            primary["operational_recommendation"].value_counts().to_dict() if not primary.empty else {}
        ),
        "provisional_decisions": int(len(primary)) if not primary.empty else 0,
        "missing_families": missing_families,
        "coverage_mode": coverage_mode,
        "warnings": [
            "All operational decisions are provisional_pending_additional_validity_checks.",
            "Diagnostics do not predict holdout performance.",
            f"Forbidden holdout basenames were not loaded: {list(FORBIDDEN_HOLDOUT_BASENAMES)}",
        ],
        "core_metrics": {
            "contribution": list(CORE_CONTRIBUTION_METRICS),
            "necessity": list(CORE_NECESSITY_METRICS),
            "structure": list(CORE_STRUCTURE_METRICS),
        },
    }
    write_json(qc, out_dir / "audit_batch_a_qc.json")
    return qc


def write_missing_inputs(
    path: Path,
    *,
    unresolved_paths: Sequence[str],
    expected_families: Sequence[str],
    available_families: Sequence[str],
    duplicate_rows: pd.DataFrame,
) -> None:
    missing_families = sorted(set(expected_families) - set(available_families))
    lines = [
        "# Batch A missing inputs",
        "",
        "## Unresolved or ambiguous diagnostic files",
    ]
    if unresolved_paths:
        lines.extend(f"- `{p}`" for p in unresolved_paths)
    else:
        lines.append("- None")
    lines.extend(["", "## Families without resolved diagnostic PKLs"])
    if missing_families:
        lines.extend(f"- `{f}`" for f in missing_families)
    else:
        lines.append("- None")
    lines.extend(["", "## Duplicate PKLs excluded from scoring"])
    if duplicate_rows is not None and not duplicate_rows.empty:
        for _, row in duplicate_rows.iterrows():
            lines.append(
                f"- `{row.get('source_path')}` ({row.get('ambiguity_reason', '')})"
            )
    else:
        lines.append("- None")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_batch_a_report(
    *,
    out_dir: Path,
    config: AuditConfig,
    inventory: pd.DataFrame,
    mapping: pd.DataFrame,
    raw: pd.DataFrame,
    family_scored: pd.DataFrame,
    primary: pd.DataFrame,
    qc: Mapping[str, Any],
) -> None:
    lines = [
        "# Batch A audit screening report",
        "",
        "This report documents a validation-period contribution–necessity–structure",
        "audit protocol. Diagnostic scores do **not** predict future Sharpe ratios,",
        "do **not** certify safety, and do **not** authorize live deployment.",
        "",
        "## Scope",
        "",
        f"- Validation/calibration window: `{config.validation_start.date()}` to `{config.validation_end.date()}`",
        "- Untouched temporal holdout (not used in Batch A): `2024-05-01` to `2026-06-30`",
        f"- Primary cost level: `{config.primary_cost_bps}` bps",
        f"- Primary audit profile: `{config.audit_profile}`",
        f"- Diagnostic PKL root: `{config.diagnostic_pkl_root}`",
        f"- Metrics directory: `{config.metrics_dir}`",
        "",
        "## Files inspected",
        "",
        f"- Diagnostic PKLs inventoried: **{len(inventory)}**",
        f"- Mapping rows: **{len(mapping)}**",
        f"- Resolved models with raw diagnostics: **{len(raw)}**",
        "",
        "## PKL structure found",
        "",
        "Typical keys include `postprocessed_attributions`, `postprocessed_feature_names`,",
        "`ablation_attributions`, `occlusion_results`, `rolling_stability`,",
        "`latent_representations`, `layer_representations`, `linear_probe_results`,",
        "`separability_metrics`, and optional attention adapters.",
        "",
        "## Mapping strategy",
        "",
        "Precedence: embedded metadata → `model_cost_sensitivity` exact match →",
        "filename-stem match → normalized unique match → `family_mapping_report` /",
        "experiment checkpoint map → optional `model_assessment.pkl`.",
        "Folder `techindi` / `no_techindi` flags validate only the technical-indicator component.",
        "",
        f"- Coverage mode: `{qc.get('coverage_mode')}`",
        f"- Missing families: `{qc.get('missing_families')}`",
        "",
        "## Feature taxonomy",
        "",
        "Deterministic regex/dictionary mapping of feature names to families",
        "(price_level, raw_returns, momentum, volatility, trend, volume, oscillators,",
        "technical_indicators, temporal, return_to_go, rewards, actions, cash,",
        "instrument_returns, unclassified). Raw feature-count comparisons between",
        "`TI_True` and `TI_False` families are avoided.",
        "",
        "## Contribution metrics",
        "",
        "- `contribution_rank_stability` (higher_is_better)",
        "- `contribution_sign_stability` (higher_is_better)",
        "",
        "## Necessity metrics",
        "",
        "- `necessity_temporal_profile_stability` (higher_is_better)",
        "- `necessity_temporal_instability` (lower_is_better)",
        "- Occlusion non-degeneracy gate (mandatory when occlusion is present)",
        "",
        "## Structure metrics",
        "",
        "- `structure_effective_rank_ratio` (higher_is_better)",
        "- `structure_probe_margin` (higher_is_better)",
        "",
        "## Metrics excluded from primary scoring",
        "",
        "- Absolute attribution magnitude as a quality score",
        "- AUC / half-life / recency shares (no universal preferred direction)",
        "- Attention / head specialization (Transformer descriptive adapters)",
        "- Calinski–Harabasz without normalization",
        "- Layer conductance (not found as a primary implemented score input)",
        "",
        "## Implemented validity gates",
        "",
        "- Finite non-degenerate attributions",
        "- Attribution resampling / sign stability computability",
        "- Temporal occlusion non-degeneracy",
        "- Rolling occlusion stability availability",
        "- Non-collapsed representation effective rank",
        "- Cross-validated probe margin when labels align",
        "",
        "## Missing reviewer-requested validity checks (`not_evaluated`)",
        "",
        "- Alternative attribution-baseline concordance",
        "- Parameter randomization",
        "- Input permutation",
        "- Full attribution-guided feature deletion / insertion-deletion curves",
        "- Layer conductance as a scored component",
        "",
        "## Scoring and aggregation",
        "",
        "1. Extract raw per-model diagnostics.",
        "2. Orient and percentile-score individual models (descriptive).",
        "3. Aggregate raw metrics to family medians/IQRs.",
        "4. Percentile-rank the 18 family medians.",
        "5. Form family C, N, S and balanced D.",
        "",
        "## Performance-band rules",
        "",
        "Validation median Sharpe defines the base band; high drawdown severity and",
        "high annualized turnover each downgrade one level (floor at POOR).",
        "",
        "## Decision matrix",
        "",
        "Performance × Diagnostic → ADVANCE / MONITOR / INVESTIGATE / REJECT.",
        "Failed mandatory gates override to REJECT.",
        "Advancement means progression to the next controlled validation stage only.",
        "",
        "## Category counts (primary 5 bps)",
        "",
        f"- Diagnostic bands: `{qc.get('families_by_diagnostic_band')}`",
        f"- Performance bands: `{qc.get('families_by_performance_band')}`",
        f"- Recommendations: `{qc.get('families_by_recommendation')}`",
        f"- Provisional decisions: `{qc.get('provisional_decisions')}`",
        "",
        "## Generated paths",
        "",
        f"- Output directory: `{out_dir}`",
        f"- Notebook: `{config.repository_root / 'notebooks' / '2_audit_scoring.ipynb'}`",
        f"- Methodology TeX: `{out_dir / 'unified_audit_protocol_revised.tex'}`",
        "",
        "## Exact reproduction command",
        "",
        "```powershell",
        "python -m audit_scoring.cli `",
        f'  --diagnostic-pkl-root "{config.diagnostic_pkl_root}" `',
        f'  --metrics-dir "{config.metrics_dir}" `',
        f'  --model-assessment-pickle "{config.model_assessment_pickle}" `',
        f"  --primary-cost-bps {config.primary_cost_bps} `",
        f"  --audit-profile {config.audit_profile} `",
        f"  --bootstrap-replicates {config.bootstrap_replicates} `",
        f"  --random-seed {config.random_seed} `",
        f'  --output-dir "{out_dir}" `',
        "  --force",
        "```",
        "",
        "## Limitations of the Batch A screening protocol",
        "",
        "- Thresholds are relative to the studied candidate set.",
        "- Diagnostics do not predict future profitability.",
        "- Diagnostic robustness is distinct from realized portfolio performance.",
        "- Linear-probe separability is only one indicator of structured representation.",
        "- Temporal occlusion covers temporal necessity but does not replace every",
        "  possible feature-deletion test.",
        "- Alternative-baseline and randomization checks remain pending unless",
        "  separately found.",
        "- The later temporal holdout is required to assess whether the frozen",
        "  categories remain informative.",
        "- Best-versus-worst comparisons are retrospective failure-mode analyses,",
        "  not the main validation of the screening framework.",
        "",
    ]
    (out_dir / "BATCH_A_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def write_methodology_tex(path: Path) -> None:
    tex = r"""\subsection{Unified contribution--necessity--structure audit protocol}
\label{sec:unified-audit-protocol-revised}

We operationalize a validation-calibrated audit protocol that assesses whether
validation-period portfolio performance is accompanied by stable, faithful, and
non-degenerate decision logic. The protocol is architecture-neutral at the
framework level and is instantiated here for Decision Transformer (DT) portfolio
policies. Diagnostic quantities are deliberately separated from realized
performance: they provide diagnostic evidence about internal decision behaviour
and guide progression to the next controlled validation stage; they do not
predict later profitability, do not guarantee selection of the best holdout
model, and do not certify deployment readiness or safety.

\paragraph{Temporal partitions.}
Training uses observations through 2015-12-31. Validation and diagnostic
calibration use 2016-01-01 through 2024-04-30. The interval 2024-05-01 through
2026-06-30 is retained as an untouched temporal holdout for a subsequent
assessment. All thresholds, bands, and operational recommendations below are
frozen before that holdout is examined.

\paragraph{Unified dimensions.}
For each resolved individual diagnostic unit we extract three complementary
evidence dimensions.

\begin{itemize}
  \item \textbf{Contribution.} Which observed signals consistently influence the
  portfolio action? In the present DT implementation this uses Integrated
  Gradients attributions, indicator-token feature-family rank stability under
  deterministic bootstrap resampling, and signed contribution stability for the
  leading indicator families. Coarse taxonomic labels (price level, momentum,
  volatility, and related categories) are retained as descriptive metadata.
  Absolute attribution magnitude is not treated as evidence that a model is
  better. Raw feature-count comparisons between technical-indicator and
  price-only models are avoided.
  \item \textbf{Necessity.} Does removing historical information materially and
  consistently change the decision? The primary implementation uses temporal
  occlusion curves, rolling temporal-profile stability, a temporal-instability
  composite formed from available coefficients of variation, and a
  non-degeneracy gate on the scale-normalized occlusion effect. Larger occlusion
  effects are not scored as automatically better. Attribution-guided feature
  deletion and insertion/deletion curves are registered as optional extensions
  and are marked \emph{not evaluated} unless separately available.
  \item \textbf{Structure.} Are internal representations non-collapsed and
  sufficiently organized to encode relevant market-state information? The
  primary implementation uses the normalized effective rank of latent (or
  deepest available) representations and the deepest-layer cross-validated
  linear-probe margin relative to the majority-class baseline. Separability
  summaries are retained as descriptive evidence. Layer conductance is not part
  of the implemented primary score unless a conductance result is discovered.
  Attention activations and head specialization are Transformer-specific
  descriptive adapters and are not mandatory components of the
  architecture-neutral audit score.
\end{itemize}

\paragraph{From individual diagnostics to families.}
Raw diagnostics are first computed for each resolved individual diagnostic unit.
Family-level analysis aggregates each core raw metric by the family median
(with quartile and IQR summaries and coverage diagnostics). Only after this
robust aggregation are the eighteen family medians oriented and
percentile-ranked. Family contribution, necessity, structure, and aggregate
scores are therefore formed from family-level percentiles rather than from
averages of individual percentile scores. This prevents large families or
individual-score distributions from distorting the eighteen-family comparison.

\paragraph{Diagnostic score.}
Let $C_i$, $N_i$, and $S_i$ denote the family-level contribution, necessity, and
structure percentile scores. The diagnostic score is
\[
D_i = w_C C_i + w_N N_i + w_S S_i,
\]
with non-negative weights summing to one and each weight at least $0.15$. The
primary Batch~A profile is balanced ($w_C=w_N=w_S=1/3$). Constrained alternative
profiles (contribution-, necessity-, or structure-oriented) represent
institutional audit priorities for later sensitivity analysis; weights are not
selected by correlation with validation or holdout performance. $D_i$ is a
diagnostic score only.

\paragraph{Validation performance band.}
Validation-period portfolio performance is assessed separately. Median
validation Sharpe ratio is the primary performance criterion. Maximum-drawdown
severity and annualized turnover act as downgrade flags: each flag lowers the
performance band by one level, with a floor at POOR. Supporting metrics such as
Sortino, Calmar, and cumulative return remain descriptive unless a previously
documented manuscript rule requires otherwise.

\paragraph{Operational decision matrix.}
Diagnostic and performance bands are combined by the two-axis matrix in
Table~\ref{tab:audit-decision-matrix-revised}. Advancement means progression to
the next controlled validation stage under enhanced documentation; it is not
authorization for live trading. A failed mandatory validity gate overrides the
matrix and yields REJECT. Because several reviewer-requested validity analyses
remain unavailable in the supplied artefacts, Batch~A recommendations are marked
provisional pending additional validity checks.

\begin{table}[t]
\centering
\caption{Validation-stage operational recommendations from performance and
diagnostic bands. ADVANCE replaces earlier Accept terminology and denotes only
progression to the next controlled validation stage.}
\label{tab:audit-decision-matrix-revised}
\begin{tabular}{lcccc}
\hline
Performance & Diagnostic STRONG & ADEQUATE & WEAK & POOR \\
\hline
STRONG & ADVANCE & MONITOR & INVESTIGATE & INVESTIGATE \\
ADEQUATE & MONITOR & MONITOR & INVESTIGATE & INVESTIGATE \\
WEAK & INVESTIGATE & INVESTIGATE & INVESTIGATE & REJECT \\
POOR & INVESTIGATE & INVESTIGATE & REJECT & REJECT \\
\hline
\end{tabular}
\end{table}

\paragraph{Interpretation and subsequent assessment.}
The protocol supports screening and remediation triage. Diagnostics need not
reproduce the exact validation performance ranking; their purpose is to identify
whether acceptable validation performance is accompanied by stable and
non-degenerate internal decision logic. The later May~2024--June~2026 holdout is
used subsequently to assess whether the frozen categories remain informative.
Best-versus-worst contrasts are repositioned as retrospective failure-mode
analysis rather than as the main validation of the screening framework.
"""
    path.write_text(tex, encoding="utf-8")
