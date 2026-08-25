"""Descriptive rank concordance between frozen diagnostics and holdout outcomes."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .holdout_assessment import INCOMPLETE_EVIDENCE_FAMILY, drawdown_severity


INTERPRETATION_LABELS = (
    "positive_concordance",
    "negative_concordance",
    "weak_or_uncertain_concordance",
    "no_clear_concordance",
)


def _label_concordance(rho: float, ci_low: float, ci_high: float) -> str:
    if not np.isfinite(rho):
        return "no_clear_concordance"
    # Uncertainty: CI crosses zero or |rho| small
    if ci_low <= 0 <= ci_high or abs(rho) < 0.20:
        if abs(rho) < 0.10:
            return "no_clear_concordance"
        return "weak_or_uncertain_concordance"
    if rho > 0:
        return "positive_concordance"
    return "negative_concordance"


def bootstrap_rank_ci(
    x: np.ndarray,
    y: np.ndarray,
    *,
    method: str,
    n_boot: int,
    seed: int,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    stats_list = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb, yb = x[idx], y[idx]
        if method == "spearman":
            coef, _ = stats.spearmanr(xb, yb)
        else:
            coef, _ = stats.kendalltau(xb, yb, variant="b")
        if np.isfinite(coef):
            stats_list.append(float(coef))
    if len(stats_list) < 10:
        return float("nan"), float("nan")
    return float(np.quantile(stats_list, 0.025)), float(np.quantile(stats_list, 0.975))


def pairwise_concordance(
    df: pd.DataFrame,
    source_col: str,
    outcome_col: str,
    *,
    method: str,
    n_boot: int = 10000,
    seed: int = 42,
    exclude_families: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    exclude = set(exclude_families or [])
    work = df[~df["model_family"].isin(exclude)].copy()
    work = work[np.isfinite(pd.to_numeric(work[source_col], errors="coerce"))].copy()
    work = work[np.isfinite(pd.to_numeric(work[outcome_col], errors="coerce"))].copy()
    excluded = sorted(set(df["model_family"]) - set(work["model_family"]))
    x = pd.to_numeric(work[source_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(work[outcome_col], errors="coerce").to_numpy(dtype=float)
    n = int(len(work))
    if n < 3:
        return {
            "source_variable": source_col,
            "holdout_outcome": outcome_col,
            "method": method,
            "coefficient": np.nan,
            "p_value": np.nan,
            "bootstrap_ci_lower": np.nan,
            "bootstrap_ci_upper": np.nan,
            "n_families": n,
            "excluded_families": "|".join(excluded),
            "interpretation_label": "no_clear_concordance",
            "notes": "insufficient_complete_cases",
        }
    if method == "spearman":
        coef, p = stats.spearmanr(x, y)
    else:
        coef, p = stats.kendalltau(x, y, variant="b")
    lo, hi = bootstrap_rank_ci(x, y, method=method, n_boot=n_boot, seed=seed)
    return {
        "source_variable": source_col,
        "holdout_outcome": outcome_col,
        "method": method,
        "coefficient": float(coef) if np.isfinite(coef) else np.nan,
        "p_value": float(p) if np.isfinite(p) else np.nan,
        "bootstrap_ci_lower": lo,
        "bootstrap_ci_upper": hi,
        "n_families": n,
        "excluded_families": "|".join(excluded),
        "interpretation_label": _label_concordance(float(coef) if np.isfinite(coef) else np.nan, lo, hi),
        "notes": "descriptive_rank_association_not_predictive",
    }


def compute_rank_concordance_table(
    master_5bps: pd.DataFrame,
    *,
    n_boot: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    df = master_5bps.copy()
    df["holdout_drawdown_severity"] = drawdown_severity(df["holdout_median_max_drawdown"])
    df["neg_holdout_drawdown_severity"] = -df["holdout_drawdown_severity"]
    df["neg_holdout_annualized_turnover"] = -pd.to_numeric(
        df["holdout_median_annualized_turnover"], errors="coerce"
    )
    if "validation_median_max_drawdown" in df.columns:
        df["validation_drawdown_severity"] = drawdown_severity(df["validation_median_max_drawdown"])
        df["neg_validation_drawdown_severity"] = -df["validation_drawdown_severity"]
    if "validation_median_annualized_turnover" in df.columns:
        df["neg_validation_annualized_turnover"] = -pd.to_numeric(
            df["validation_median_annualized_turnover"], errors="coerce"
        )

    source_vars = [
        "contribution_score",
        "necessity_score",
        "structure_score",
        "diagnostic_score",
    ]
    # For context: validation vs holdout performance concordance
    context_sources = [
        ("validation_median_sharpe", "holdout_median_sharpe"),
        ("validation_median_sortino", "holdout_median_sortino"),
        ("validation_median_calmar", "holdout_median_calmar"),
        ("neg_validation_drawdown_severity", "neg_holdout_drawdown_severity"),
        ("neg_validation_annualized_turnover", "neg_holdout_annualized_turnover"),
    ]
    outcomes = [
        "holdout_median_sharpe",
        "holdout_median_sortino",
        "holdout_median_calmar",
        "holdout_median_annualized_return",
        "neg_holdout_drawdown_severity",
        "neg_holdout_annualized_turnover",
    ]

    rows: List[Dict[str, object]] = []
    for src in source_vars:
        exclude = [INCOMPLETE_EVIDENCE_FAMILY]  # always exclude incomplete for score analyses
        for outcome in outcomes:
            for method in ("spearman", "kendall"):
                rows.append(
                    pairwise_concordance(
                        df,
                        src,
                        outcome,
                        method=method,
                        n_boot=n_boot,
                        seed=seed,
                        exclude_families=exclude,
                    )
                )

    for src, outcome in context_sources:
        if src not in df.columns or outcome not in df.columns:
            continue
        for method in ("spearman", "kendall"):
            row = pairwise_concordance(
                df,
                src,
                outcome,
                method=method,
                n_boot=n_boot,
                seed=seed,
                exclude_families=[],
            )
            row["notes"] = "validation_vs_holdout_context_not_diagnostic"
            rows.append(row)

    out = pd.DataFrame(rows)
    return out.sort_values(
        ["source_variable", "holdout_outcome", "method"]
    ).reset_index(drop=True)
