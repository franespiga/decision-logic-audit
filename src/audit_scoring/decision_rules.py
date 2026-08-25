"""Diagnostic bands, validation performance bands, and operational decisions."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import (
    BAND_ORDER,
    COST_LEVELS_BPS,
    DECISION_MATRIX,
    DECISION_ORDER,
    PRIMARY_COST_BPS,
)
from .io import discover_metric_file, read_table


def _band_from_sharpe(sharpe: float, p25: float, p50: float, p75: float) -> str:
    if not np.isfinite(sharpe):
        return "POOR"
    if sharpe >= p75:
        return "STRONG"
    if sharpe >= p50:
        return "ADEQUATE"
    if sharpe >= p25:
        return "WEAK"
    return "POOR"


def downgrade_band(band: str, n_steps: int) -> str:
    order = list(BAND_ORDER)
    idx = order.index(band) if band in order else order.index("POOR")
    idx = min(idx + max(n_steps, 0), len(order) - 1)
    return order[idx]


def assign_diagnostic_band(
    row: Mapping[str, Any],
    thresholds: Mapping[str, float],
    *,
    coverage_ratio_threshold: float = 0.80,
) -> Tuple[str, str]:
    """Return (band, reason) with deterministic precedence."""
    failed_gate = str(row.get("mandatory_validity_status", "passed")) == "failed"
    if failed_gate:
        return "POOR", "failed_mandatory_validity_gate"

    missing_core = str(row.get("family_score_status", "")) == "not_scored_missing_core_dimension"
    if missing_core or not np.isfinite(row.get("diagnostic_score", np.nan)):
        return "POOR", "missing_core_dimension"

    coverage = float(row.get("model_coverage_ratio", np.nan))
    if np.isfinite(coverage) and coverage < coverage_ratio_threshold:
        return "POOR", f"inadequate_coverage<{coverage_ratio_threshold}"

    d = float(row["diagnostic_score"])
    c = float(row.get("contribution_score", np.nan))
    n = float(row.get("necessity_score", np.nan))
    s = float(row.get("structure_score", np.nan))
    p25, p50, p75 = thresholds["D_p25"], thresholds["D_p50"], thresholds["D_p75"]
    high_disp = bool(row.get("high_within_family_dispersion", False))

    dims = np.asarray([c, n, s], dtype=float)
    n_below_p50 = int(np.sum(dims < p50))
    n_below_p25 = int(np.sum(dims < p25))

    # STRONG
    if (
        d >= p75
        and c >= p50
        and n >= p50
        and s >= p50
        and not high_disp
        and coverage >= coverage_ratio_threshold
    ):
        return "STRONG", "D>=P75_and_all_dimensions>=P50_coverage_ok"

    # ADEQUATE
    if d >= p50 and n_below_p50 <= 1 and n_below_p25 == 0:
        return "ADEQUATE", "D>=P50_at_most_one_dim_below_P50_none_below_P25"

    # WEAK
    if (d >= p25 or n_below_p25 == 1 or high_disp) and not failed_gate:
        reasons = []
        if d >= p25:
            reasons.append("D>=P25")
        if n_below_p25 == 1:
            reasons.append("one_dim_below_P25")
        if high_disp:
            reasons.append("high_within_family_dispersion")
        return "WEAK", "+".join(reasons)

    return "POOR", "default_poor"


def compute_diagnostic_thresholds(scored_families: pd.DataFrame) -> Dict[str, float]:
    d = pd.to_numeric(scored_families["diagnostic_score"], errors="coerce").dropna()
    if d.empty:
        return {"D_p25": np.nan, "D_p50": np.nan, "D_p75": np.nan}
    return {
        "D_p25": float(d.quantile(0.25)),
        "D_p50": float(d.quantile(0.50)),
        "D_p75": float(d.quantile(0.75)),
    }


def load_validation_performance(metrics_dir, cost_levels: Sequence[int] = COST_LEVELS_BPS) -> pd.DataFrame:
    """Load family validation performance; never loads forbidden holdout files."""
    fam_path = discover_metric_file(metrics_dir, "family_cost_sensitivity")
    fam = read_table(fam_path)
    # Optional turnover enrichment
    try:
        turn_path = discover_metric_file(metrics_dir, "family_level_turnover")
        turn = read_table(turn_path)
    except FileNotFoundError:
        turn = pd.DataFrame()

    required = {"model_family", "cost_bps", "median_sharpe"}
    missing = required - set(fam.columns)
    if missing:
        raise ValueError(f"{fam_path} missing columns {sorted(missing)}")

    df = fam.copy()
    df = df[df["cost_bps"].isin(list(cost_levels))].copy()
    if not turn.empty and "median_annualized_turnover" in turn.columns:
        # Prefer cost-sensitive turnover from family_cost_sensitivity when present.
        pass
    return df


def assign_performance_bands(family_perf: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """Assign performance bands per cost level with drawdown/turnover downgrades."""
    rows = []
    thresholds: Dict[str, Dict[str, float]] = {}
    for cost, sub in family_perf.groupby("cost_bps"):
        sharpe = pd.to_numeric(sub["median_sharpe"], errors="coerce")
        dd = pd.to_numeric(sub.get("median_max_drawdown"), errors="coerce")
        # severity = absolute magnitude of drawdown
        dd_severity = (-dd).where(dd < 0, dd.abs())
        turnover = pd.to_numeric(sub.get("median_annualized_turnover"), errors="coerce")
        thr = {
            "sharpe_p25": float(sharpe.quantile(0.25)),
            "sharpe_p50": float(sharpe.quantile(0.50)),
            "sharpe_p75": float(sharpe.quantile(0.75)),
            "drawdown_severity_p75": float(dd_severity.quantile(0.75)),
            "turnover_p75": float(turnover.quantile(0.75)),
        }
        thresholds[str(int(cost))] = thr
        for _, row in sub.iterrows():
            base = _band_from_sharpe(
                float(row["median_sharpe"]) if np.isfinite(row["median_sharpe"]) else np.nan,
                thr["sharpe_p25"],
                thr["sharpe_p50"],
                thr["sharpe_p75"],
            )
            sev = float(dd_severity.loc[row.name]) if row.name in dd_severity.index else np.nan
            tov = float(turnover.loc[row.name]) if row.name in turnover.index else np.nan
            high_dd = bool(np.isfinite(sev) and sev >= thr["drawdown_severity_p75"])
            high_to = bool(np.isfinite(tov) and tov >= thr["turnover_p75"])
            n_down = int(high_dd) + int(high_to)
            final = downgrade_band(base, n_down)
            rows.append(
                {
                    **{k: row.get(k) for k in sub.columns},
                    "performance_band_base": base,
                    "performance_band": final,
                    "high_drawdown_flag": high_dd,
                    "high_turnover_flag": high_to,
                    "drawdown_severity": sev,
                }
            )
    return pd.DataFrame(rows), thresholds


def operational_decision(
    performance_band: str,
    diagnostic_band: str,
    *,
    mandatory_gate_failed: bool = False,
) -> Tuple[str, str]:
    if mandatory_gate_failed:
        return "REJECT", "mandatory_validity_gate_override"
    key = (str(performance_band), str(diagnostic_band))
    decision = DECISION_MATRIX.get(key, "REJECT")
    return decision, f"matrix[{performance_band},{diagnostic_band}]"


def build_validation_master(
    family_scored: pd.DataFrame,
    family_perf_banded: pd.DataFrame,
    diagnostic_thresholds: Mapping[str, float],
    *,
    coverage_ratio_threshold: float = 0.80,
    primary_cost_bps: int = PRIMARY_COST_BPS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Combine diagnostics and performance into master and primary decision tables."""
    # Diagnostic bands once (family-level, cost-invariant)
    diag_rows = []
    for _, row in family_scored.iterrows():
        band, reason = assign_diagnostic_band(
            row,
            diagnostic_thresholds,
            coverage_ratio_threshold=coverage_ratio_threshold,
        )
        diag_rows.append(
            {
                "model_family": row["model_family"],
                "diagnostic_band": band,
                "diagnostic_band_reason": reason,
                "diagnostic_score": row.get("diagnostic_score"),
                "contribution_score": row.get("contribution_score"),
                "necessity_score": row.get("necessity_score"),
                "structure_score": row.get("structure_score"),
                "model_coverage_ratio": row.get("model_coverage_ratio"),
                "n_models_available": row.get("n_models_available"),
                "n_models_expected": row.get("n_models_expected"),
                "high_within_family_dispersion": row.get("high_within_family_dispersion"),
                "family_score_status": row.get("family_score_status"),
                "mandatory_validity_status": row.get("mandatory_validity_status", "passed"),
                "provisional": True,  # pending reviewer-requested validity checks
            }
        )
    diag = pd.DataFrame(diag_rows)

    merged = family_perf_banded.merge(diag, on="model_family", how="left")
    decisions = []
    for _, row in merged.iterrows():
        gate_failed = str(row.get("mandatory_validity_status", "passed")) == "failed"
        decision, reason = operational_decision(
            str(row.get("performance_band", "POOR")),
            str(row.get("diagnostic_band", "POOR")),
            mandatory_gate_failed=gate_failed,
        )
        # Provisional label when reviewer checks pending
        operational_status = decision
        note = reason
        if bool(row.get("provisional", True)):
            note = f"{reason}|provisional_pending_additional_validity_checks"
        decisions.append(
            {
                **row.to_dict(),
                "operational_recommendation": operational_status,
                "decision_reason": note,
            }
        )
    master = pd.DataFrame(decisions)

    # Deterministic sort helpers
    master["decision_rank"] = master["operational_recommendation"].map(
        {d: i for i, d in enumerate(DECISION_ORDER)}
    )
    master = master.sort_values(
        ["cost_bps", "decision_rank", "diagnostic_score", "median_sharpe"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)

    primary = master[master["cost_bps"] == primary_cost_bps].copy().reset_index(drop=True)
    return master, primary
