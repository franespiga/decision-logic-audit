"""Transparent family-level case classification for Batch B."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .holdout_assessment import INCOMPLETE_EVIDENCE_FAMILY, drawdown_severity
from .enrichment import PROGRESSION_COMPATIBLE, PROGRESSION_RESTRICTED


def _rank_series(s: pd.Series, *, ascending: bool) -> pd.Series:
    return s.rank(method="average", ascending=ascending)


def classify_cases(master_5bps: pd.DataFrame) -> pd.DataFrame:
    df = master_5bps.copy()
    df["holdout_drawdown_severity"] = drawdown_severity(df["holdout_median_max_drawdown"])
    n = len(df)
    df["holdout_sharpe_rank"] = _rank_series(df["holdout_median_sharpe"], ascending=False)
    df["holdout_sortino_rank"] = _rank_series(df["holdout_median_sortino"], ascending=False)
    df["holdout_calmar_rank"] = _rank_series(df["holdout_median_calmar"], ascending=False)
    df["holdout_drawdown_rank"] = _rank_series(df["holdout_drawdown_severity"], ascending=True)

    sharpe_q = df["holdout_median_sharpe"].quantile([0.25, 0.50, 0.75])
    dd_q = df["holdout_drawdown_severity"].quantile([0.25, 0.50, 0.75])
    to_q = df["holdout_median_annualized_turnover"].quantile([0.25, 0.50, 0.75])
    sortino_q = df["holdout_median_sortino"].quantile([0.75])
    calmar_q = df["holdout_median_calmar"].quantile([0.75])

    rows: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        fam = str(row["model_family"])
        decision = str(row["frozen_screening_decision"])
        diag = str(row["diagnostic_band"])
        perf = str(row["validation_performance_band"])
        rejection_basis = str(row.get("rejection_basis", "") or "")
        hs = float(row["holdout_median_sharpe"])
        dd = float(row["holdout_drawdown_severity"])
        to = float(row["holdout_median_annualized_turnover"])
        hsort = float(row["holdout_median_sortino"])
        hcal = float(row["holdout_median_calmar"])

        label = "mixed_or_inconclusive"
        reason = "no_clear_case_rule_matched"
        review = False

        if fam == INCOMPLETE_EVIDENCE_FAMILY or rejection_basis == "incomplete_diagnostic_evidence":
            label = "evidence_incomplete_exclusion"
            reason = "frozen_reject_due_to_incomplete_diagnostic_evidence"
            review = True
        elif decision == "MONITOR" and hs >= sharpe_q.loc[0.50] and dd <= dd_q.loc[0.75] and to <= to_q.loc[0.75]:
            label = "concordant_conditional_progression"
            reason = "monitor_with_holdout_sharpe_top_half_and_nonextreme_risk_flags"
        elif decision in PROGRESSION_COMPATIBLE and (
            hs <= sharpe_q.loc[0.25] or dd >= dd_q.loc[0.75]
        ):
            label = "adverse_holdout_among_progression_compatible"
            reason = "advance_or_monitor_with_weak_holdout_sharpe_or_severe_drawdown"
            review = True
        elif decision in PROGRESSION_RESTRICTED and (
            hs >= sharpe_q.loc[0.75] or hsort >= sortino_q.loc[0.75] or hcal >= calmar_q.loc[0.75]
        ):
            label = "conservative_screening_case"
            reason = "investigate_or_reject_with_top_quartile_holdout_return_metric"
            review = True
        elif perf in {"STRONG", "ADEQUATE"} and diag in {"WEAK", "POOR"} and (
            hs < sharpe_q.loc[0.50] or hcal < df["holdout_median_calmar"].median()
        ):
            label = "diagnostic_warning_aligned_with_holdout"
            reason = "strong_validation_performance_but_weak_diagnostics_and_below_median_holdout"
        elif diag in {"STRONG", "ADEQUATE"} and perf in {"WEAK", "POOR"} and (
            hs < sharpe_q.loc[0.50] or hcal < df["holdout_median_calmar"].median()
        ):
            label = "performance_limitation_aligned_with_holdout"
            reason = "adequate_diagnostics_but_weak_validation_performance_and_below_median_holdout"

        rows.append(
            {
                "model_family": fam,
                "frozen_screening_decision": decision,
                "diagnostic_band": diag,
                "validation_performance_band": perf,
                "rejection_basis": rejection_basis,
                "holdout_sharpe_rank": float(row["holdout_sharpe_rank"]),
                "holdout_sortino_rank": float(row["holdout_sortino_rank"]),
                "holdout_calmar_rank": float(row["holdout_calmar_rank"]),
                "holdout_drawdown_rank": float(row["holdout_drawdown_rank"]),
                "case_label": label,
                "case_reason": reason,
                "provisional_interpretation": (
                    "Descriptive case label under frozen provisional screening; "
                    "not a predictive classification."
                ),
                "requires_manual_review": bool(review),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["case_label", "model_family"]).reset_index(drop=True)
