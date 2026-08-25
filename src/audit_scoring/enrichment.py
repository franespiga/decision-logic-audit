"""Category enrichment, operational groups, top/bottom coverage, cost sensitivity."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .holdout_assessment import INCOMPLETE_EVIDENCE_FAMILY, drawdown_severity

PROGRESSION_COMPATIBLE = ("ADVANCE", "MONITOR")
PROGRESSION_RESTRICTED = ("INVESTIGATE", "REJECT")
DECISION_ORDER = ("ADVANCE", "MONITOR", "INVESTIGATE", "REJECT")


def _bootstrap_median_ci(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    if len(values) < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    meds = []
    for _ in range(n_boot):
        draw = rng.choice(values, size=len(values), replace=True)
        meds.append(float(np.median(draw)))
    return float(np.quantile(meds, 0.025)), float(np.quantile(meds, 0.975))


def category_summary(
    master: pd.DataFrame,
    *,
    decision_col: str = "frozen_screening_decision",
    n_boot: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    df = master.copy()
    df["holdout_drawdown_severity"] = drawdown_severity(df["holdout_median_max_drawdown"])
    rows = []
    for decision in DECISION_ORDER:
        sub = df[df[decision_col] == decision]
        n = int(len(sub))
        row: Dict[str, object] = {
            "frozen_screening_decision": decision,
            "n_families": n,
            "operational_group": (
                "progression_compatible"
                if decision in PROGRESSION_COMPATIBLE
                else "progression_restricted"
            ),
        }
        metrics = {
            "holdout_median_sharpe": "median_holdout_sharpe",
            "holdout_median_sortino": "median_holdout_sortino",
            "holdout_median_calmar": "median_holdout_calmar",
            "holdout_median_annualized_return": "median_holdout_annualized_return",
            "holdout_drawdown_severity": "median_holdout_drawdown_severity",
            "holdout_median_annualized_turnover": "median_holdout_annualized_turnover",
        }
        if n == 0:
            for _, out_name in metrics.items():
                row[out_name] = np.nan
                row[f"iqr_{out_name}"] = np.nan
                row[f"ci_low_{out_name}"] = np.nan
                row[f"ci_high_{out_name}"] = np.nan
            row["notes"] = "category_empty_statistics_not_estimable"
        else:
            for src, out_name in metrics.items():
                vals = pd.to_numeric(sub[src], errors="coerce").dropna().to_numpy(dtype=float)
                row[out_name] = float(np.median(vals)) if len(vals) else np.nan
                if len(vals):
                    row[f"iqr_{out_name}"] = float(np.quantile(vals, 0.75) - np.quantile(vals, 0.25))
                else:
                    row[f"iqr_{out_name}"] = np.nan
                if n >= 3 and out_name == "median_holdout_sharpe":
                    lo, hi = _bootstrap_median_ci(vals, n_boot=n_boot, seed=seed)
                    row["ci_low_median_holdout_sharpe"] = lo
                    row["ci_high_median_holdout_sharpe"] = hi
                elif out_name == "median_holdout_sharpe":
                    row["ci_low_median_holdout_sharpe"] = np.nan
                    row["ci_high_median_holdout_sharpe"] = np.nan
            row["notes"] = "descriptive_category_summary"
            row["family_ids"] = "|".join(sorted(sub["model_family"].astype(str)))
        rows.append(row)
    return pd.DataFrame(rows)


def operational_group_summary(
    master: pd.DataFrame,
    *,
    n_boot: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    df = master.copy()
    df["holdout_drawdown_severity"] = drawdown_severity(df["holdout_median_max_drawdown"])
    df["operational_group"] = np.where(
        df["frozen_screening_decision"].isin(PROGRESSION_COMPATIBLE),
        "progression_compatible",
        "progression_restricted",
    )
    rows = []
    for group in ("progression_compatible", "progression_restricted"):
        sub = df[df["operational_group"] == group]
        vals = pd.to_numeric(sub["holdout_median_sharpe"], errors="coerce").dropna().to_numpy()
        dd = pd.to_numeric(sub["holdout_drawdown_severity"], errors="coerce").dropna().to_numpy()
        lo, hi = _bootstrap_median_ci(vals, n_boot=n_boot, seed=seed) if len(sub) >= 3 else (np.nan, np.nan)
        rows.append(
            {
                "operational_group": group,
                "n_families": int(len(sub)),
                "median_holdout_sharpe": float(np.median(vals)) if len(vals) else np.nan,
                "iqr_holdout_sharpe": float(np.quantile(vals, 0.75) - np.quantile(vals, 0.25))
                if len(vals)
                else np.nan,
                "ci_low_median_holdout_sharpe": lo,
                "ci_high_median_holdout_sharpe": hi,
                "median_holdout_drawdown_severity": float(np.median(dd)) if len(dd) else np.nan,
                "family_ids": "|".join(sorted(sub["model_family"].astype(str))),
                "notes": "progression labels refer only to frozen stage progression",
            }
        )
    return pd.DataFrame(rows)


def top_bottom_coverage(master_5bps: pd.DataFrame, ks: Sequence[int] = (3, 5)) -> pd.DataFrame:
    df = master_5bps.copy()
    df["holdout_drawdown_severity"] = drawdown_severity(df["holdout_median_max_drawdown"])
    ranking_specs = [
        ("holdout_median_sharpe", True, "sharpe"),
        ("holdout_median_sortino", True, "sortino"),
        ("holdout_median_calmar", True, "calmar"),
        ("holdout_median_annualized_return", True, "annualized_return"),
        ("holdout_drawdown_severity", False, "drawdown_severity"),
    ]
    rows = []
    for col, higher_better, label in ranking_specs:
        ranked = df.sort_values(col, ascending=not higher_better, kind="mergesort").reset_index(drop=True)
        for k in ks:
            top = ranked.head(k)
            bottom = ranked.tail(k)
            top_prog = top[top["frozen_screening_decision"].isin(PROGRESSION_COMPATIBLE)]
            bot_rest = bottom[bottom["frozen_screening_decision"].isin(PROGRESSION_RESTRICTED)]
            incomplete_in_top = INCOMPLETE_EVIDENCE_FAMILY in set(top["model_family"])
            incomplete_in_bottom = INCOMPLETE_EVIDENCE_FAMILY in set(bottom["model_family"])
            rows.append(
                {
                    "ranking_metric": label,
                    "k": k,
                    "set_type": "top",
                    "family_ids": "|".join(top["model_family"].astype(str)),
                    "frozen_decisions": "|".join(top["frozen_screening_decision"].astype(str)),
                    "progression_compatible_count": int(len(top_prog)),
                    "progression_compatible_share": float(len(top_prog) / k),
                    "progression_restricted_count": int(
                        top["frozen_screening_decision"].isin(PROGRESSION_RESTRICTED).sum()
                    ),
                    "incomplete_evidence_family_present": bool(incomplete_in_top),
                    "all_covered_are_monitor": bool(
                        len(top_prog) > 0
                        and set(top_prog["frozen_screening_decision"]) == {"MONITOR"}
                    ),
                    "notes": "top-k screening coverage; not predictive accuracy",
                }
            )
            rows.append(
                {
                    "ranking_metric": label,
                    "k": k,
                    "set_type": "bottom",
                    "family_ids": "|".join(bottom["model_family"].astype(str)),
                    "frozen_decisions": "|".join(bottom["frozen_screening_decision"].astype(str)),
                    "progression_compatible_count": int(
                        bottom["frozen_screening_decision"].isin(PROGRESSION_COMPATIBLE).sum()
                    ),
                    "progression_compatible_share": float(
                        bottom["frozen_screening_decision"].isin(PROGRESSION_COMPATIBLE).mean()
                    ),
                    "progression_restricted_count": int(len(bot_rest)),
                    "progression_restricted_share": float(len(bot_rest) / k),
                    "incomplete_evidence_family_present": bool(incomplete_in_bottom),
                    "all_covered_are_monitor": False,
                    "notes": "bottom-k progression-restricted coverage; not predictive accuracy",
                }
            )
    return pd.DataFrame(rows)


def incomplete_evidence_sensitivity(master_5bps: pd.DataFrame) -> pd.DataFrame:
    df = master_5bps.copy()
    rows = []
    # Category median Sharpe with/without incomplete family
    for name, population, subset in [
        (
            "category_median_holdout_sharpe",
            "all_18_families",
            df,
        ),
        (
            "category_median_holdout_sharpe",
            "evidence_complete_17",
            df[df["model_family"] != INCOMPLETE_EVIDENCE_FAMILY],
        ),
    ]:
        for decision in DECISION_ORDER:
            sub = subset[subset["frozen_screening_decision"] == decision]
            vals = pd.to_numeric(sub["holdout_median_sharpe"], errors="coerce").dropna()
            rows.append(
                {
                    "analysis_name": name,
                    "outcome_metric": "holdout_median_sharpe",
                    "analysis_population": f"{population}|{decision}",
                    "included_family_count": int(len(sub)),
                    "included_families": "|".join(sorted(sub["model_family"].astype(str))),
                    "excluded_families": (
                        INCOMPLETE_EVIDENCE_FAMILY
                        if population.startswith("evidence_complete")
                        else ""
                    ),
                    "estimate": float(vals.median()) if len(vals) else np.nan,
                    "notes": "empty_category" if len(sub) == 0 else "descriptive_sensitivity",
                }
            )
    # Score-correlation sample sizes
    complete = df[df["diagnostic_score_available"] == True]  # noqa: E712
    rows.append(
        {
            "analysis_name": "diagnostic_score_complete_cases",
            "outcome_metric": "n_families",
            "analysis_population": "evidence_complete",
            "included_family_count": int(len(complete)),
            "included_families": "|".join(sorted(complete["model_family"].astype(str))),
            "excluded_families": INCOMPLETE_EVIDENCE_FAMILY,
            "estimate": float(len(complete)),
            "notes": "score-based analyses exclude incomplete evidence family",
        }
    )
    # Frozen decision preserved
    inc = df[df["model_family"] == INCOMPLETE_EVIDENCE_FAMILY]
    rows.append(
        {
            "analysis_name": "incomplete_family_frozen_decision",
            "outcome_metric": "frozen_screening_decision",
            "analysis_population": "all_18_families",
            "included_family_count": int(len(inc)),
            "included_families": INCOMPLETE_EVIDENCE_FAMILY,
            "excluded_families": "",
            "estimate": np.nan,
            "notes": (
                f"decision={inc.iloc[0]['frozen_screening_decision']};"
                f"rejection_basis={inc.iloc[0].get('rejection_basis')}"
                if len(inc)
                else "missing"
            ),
        }
    )
    return pd.DataFrame(rows)


def variant_distribution(individual: pd.DataFrame, cost_bps: Optional[int] = None) -> pd.DataFrame:
    df = individual.copy()
    if cost_bps is not None:
        df = df[df["cost_bps"] == cost_bps]
    rows = []
    # global quartiles within cost
    for cost, sub_cost in df.groupby("cost_bps"):
        sharpe_all = pd.to_numeric(sub_cost["sharpe"], errors="coerce")
        q25, q75 = sharpe_all.quantile(0.25), sharpe_all.quantile(0.75)
        for family, sub in sub_cost.groupby("model_family"):
            s = pd.to_numeric(sub["sharpe"], errors="coerce").dropna()
            dd = drawdown_severity(sub["max_drawdown"]).dropna()
            to = pd.to_numeric(sub["annualized_turnover"], errors="coerce").dropna()
            iqr = float(s.quantile(0.75) - s.quantile(0.25)) if len(s) else np.nan
            med = float(s.median()) if len(s) else np.nan
            dispersion = "broadly_shared"
            if np.isfinite(iqr) and np.isfinite(med):
                if abs(med) > 1e-12 and iqr / abs(med) > 1.0:
                    dispersion = "moderately_dispersed"
                if (s > q75).mean() > 0.35 or (s < q25).mean() > 0.35:
                    dispersion = "tail_influenced"
            rows.append(
                {
                    "model_family": family,
                    "cost_bps": int(cost),
                    "n_ex_post_variants": int(sub["model_id"].nunique()),
                    "median_holdout_sharpe": med,
                    "iqr_holdout_sharpe": iqr,
                    "min_holdout_sharpe": float(s.min()) if len(s) else np.nan,
                    "max_holdout_sharpe": float(s.max()) if len(s) else np.nan,
                    "proportion_positive_sharpe": float((s > 0).mean()) if len(s) else np.nan,
                    "proportion_global_top_quartile": float((s >= q75).mean()) if len(s) else np.nan,
                    "proportion_global_bottom_quartile": float((s <= q25).mean()) if len(s) else np.nan,
                    "median_drawdown_severity": float(dd.median()) if len(dd) else np.nan,
                    "iqr_drawdown_severity": float(dd.quantile(0.75) - dd.quantile(0.25))
                    if len(dd)
                    else np.nan,
                    "median_annualized_turnover": float(to.median()) if len(to) else np.nan,
                    "iqr_annualized_turnover": float(to.quantile(0.75) - to.quantile(0.25))
                    if len(to)
                    else np.nan,
                    "within_family_dispersion_label": dispersion,
                    "notes": "ex-post evaluation variants of one trained checkpoint; not independent policies",
                }
            )
    return pd.DataFrame(rows).sort_values(["cost_bps", "model_family"]).reset_index(drop=True)
