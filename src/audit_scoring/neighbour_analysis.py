"""Deterministic nearest-neighbour family pair selection (outcome-independent)."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .holdout_assessment import (
    INCOMPLETE_EVIDENCE_FAMILY,
    drawdown_severity,
    parse_family_design_factors,
)

FACTOR_KEYS = (
    "technical_indicators",
    "lookback",
    "rebalancing_frequency",
    "objective_or_conditioning",
)


def hamming_distance(fam_a: str, fam_b: str) -> Tuple[int, List[str], List[str]]:
    fa = parse_family_design_factors(fam_a)
    fb = parse_family_design_factors(fam_b)
    shared = []
    differing = []
    dist = 0
    for key in FACTOR_KEYS:
        if fa[key] == fb[key]:
            shared.append(f"{key}={fa[key]}")
        else:
            dist += 1
            differing.append(f"{key}:{fa[key]}!={fb[key]}")
    return dist, shared, differing


def build_neighbour_candidates(master_5bps: pd.DataFrame) -> pd.DataFrame:
    df = master_5bps.copy()
    complete = df[
        (df["diagnostic_score_available"] == True)  # noqa: E712
        & (df["model_family"] != INCOMPLETE_EVIDENCE_FAMILY)
        & (df.get("evidence_complete_for_primary_score", True) == True)  # noqa: E712
    ].copy()
    families = sorted(complete["model_family"].astype(str).unique())
    indexed = complete.set_index("model_family")
    rows = []
    for i, a in enumerate(families):
        for b in families[i + 1 :]:
            dist, shared, differing = hamming_distance(a, b)
            ra, rb = indexed.loc[a], indexed.loc[b]
            rows.append(
                {
                    "family_a": a,
                    "family_b": b,
                    "ex_ante_hamming_distance": dist,
                    "ex_ante_shared": "|".join(shared),
                    "ex_ante_differing": "|".join(differing),
                    "abs_validation_sharpe_diff": abs(
                        float(ra["validation_median_sharpe"]) - float(rb["validation_median_sharpe"])
                    ),
                    "abs_validation_calmar_diff": abs(
                        float(ra["validation_median_calmar"]) - float(rb["validation_median_calmar"])
                    ),
                    "abs_diagnostic_score_diff": abs(
                        float(ra["diagnostic_score"]) - float(rb["diagnostic_score"])
                    ),
                    "contribution_score_diff": float(ra["contribution_score"]) - float(rb["contribution_score"]),
                    "necessity_score_diff": float(ra["necessity_score"]) - float(rb["necessity_score"]),
                    "structure_score_diff": float(ra["structure_score"]) - float(rb["structure_score"]),
                    "holdout_sharpe_diff": float(ra["holdout_median_sharpe"]) - float(rb["holdout_median_sharpe"]),
                    "holdout_calmar_diff": float(ra["holdout_median_calmar"]) - float(rb["holdout_median_calmar"]),
                    "holdout_drawdown_diff": float(
                        drawdown_severity(pd.Series([ra["holdout_median_max_drawdown"]])).iloc[0]
                        - drawdown_severity(pd.Series([rb["holdout_median_max_drawdown"]])).iloc[0]
                    ),
                    "holdout_turnover_diff": float(ra["holdout_median_annualized_turnover"])
                    - float(rb["holdout_median_annualized_turnover"]),
                    "validation_sharpe_a": float(ra["validation_median_sharpe"]),
                    "validation_sharpe_b": float(rb["validation_median_sharpe"]),
                    "diagnostic_score_a": float(ra["diagnostic_score"]),
                    "diagnostic_score_b": float(rb["diagnostic_score"]),
                    "contribution_a": float(ra["contribution_score"]),
                    "contribution_b": float(rb["contribution_score"]),
                    "necessity_a": float(ra["necessity_score"]),
                    "necessity_b": float(rb["necessity_score"]),
                    "structure_a": float(ra["structure_score"]),
                    "structure_b": float(rb["structure_score"]),
                    "holdout_sharpe_a": float(ra["holdout_median_sharpe"]),
                    "holdout_sharpe_b": float(rb["holdout_median_sharpe"]),
                    "holdout_drawdown_a": float(
                        drawdown_severity(pd.Series([ra["holdout_median_max_drawdown"]])).iloc[0]
                    ),
                    "holdout_drawdown_b": float(
                        drawdown_severity(pd.Series([rb["holdout_median_max_drawdown"]])).iloc[0]
                    ),
                    "holdout_turnover_a": float(ra["holdout_median_annualized_turnover"]),
                    "holdout_turnover_b": float(rb["holdout_median_annualized_turnover"]),
                }
            )
    return pd.DataFrame(rows)


def select_neighbour_pairs(
    candidates: pd.DataFrame,
    *,
    n_pairs: int = 3,
) -> pd.DataFrame:
    """Select pairs using the documented deterministic rule (no holdout-favorable criterion)."""
    if candidates.empty:
        return candidates.copy()

    # Prefer Hamming distance 1, else allow 2
    pool = candidates[candidates["ex_ante_hamming_distance"] == 1]
    if len(pool) < n_pairs:
        pool = candidates[candidates["ex_ante_hamming_distance"].isin([1, 2])]

    # Lowest quartile of abs validation Sharpe difference
    q25 = pool["abs_validation_sharpe_diff"].quantile(0.25)
    near = pool[pool["abs_validation_sharpe_diff"] <= q25].copy()
    if near.empty:
        near = pool.copy()

    # Prefer largest diagnostic-score difference
    near = near.sort_values(
        ["abs_diagnostic_score_diff", "abs_validation_sharpe_diff", "family_a", "family_b"],
        ascending=[False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    selected = []
    family_counts: Dict[str, int] = {}
    for _, row in near.iterrows():
        a, b = str(row["family_a"]), str(row["family_b"])
        if family_counts.get(a, 0) >= 2 or family_counts.get(b, 0) >= 2:
            continue
        selected.append(row)
        family_counts[a] = family_counts.get(a, 0) + 1
        family_counts[b] = family_counts.get(b, 0) + 1
        if len(selected) >= n_pairs:
            break

    # If still short, relax family reuse limit to 3
    if len(selected) < n_pairs:
        for _, row in near.iterrows():
            if any(
                str(row["family_a"]) == str(s["family_a"]) and str(row["family_b"]) == str(s["family_b"])
                for s in selected
            ):
                continue
            selected.append(row)
            if len(selected) >= n_pairs:
                break

    out = pd.DataFrame(selected).reset_index(drop=True)
    if out.empty:
        return out

    # Dimension illustration notes and holdout support labels (descriptive only)
    rationales = []
    for i, row in out.iterrows():
        dims = []
        if abs(row["contribution_score_diff"]) >= abs(row["necessity_score_diff"]) and abs(
            row["contribution_score_diff"]
        ) >= abs(row["structure_score_diff"]):
            dims.append("contribution")
        if abs(row["necessity_score_diff"]) >= abs(row["contribution_score_diff"]) and abs(
            row["necessity_score_diff"]
        ) >= abs(row["structure_score_diff"]):
            dims.append("necessity")
        if abs(row["structure_score_diff"]) >= abs(row["contribution_score_diff"]) and abs(
            row["structure_score_diff"]
        ) >= abs(row["necessity_score_diff"]):
            dims.append("structure")
        # Holdout support for diagnostic contrast: if higher D family has higher holdout Sharpe
        higher_d_is_a = row["diagnostic_score_a"] >= row["diagnostic_score_b"]
        higher_h_is_a = row["holdout_sharpe_a"] >= row["holdout_sharpe_b"]
        if abs(row["holdout_sharpe_diff"]) < 0.05:
            holdout_view = "inconclusive"
        elif higher_d_is_a == higher_h_is_a:
            holdout_view = "supports"
        else:
            holdout_view = "contradicts"
        rationales.append(
            {
                "pair_id": i + 1,
                "scientific_selection_rationale": (
                    f"Hamming={int(row['ex_ante_hamming_distance'])}; "
                    f"validation Sharpe proximity (abs diff={row['abs_validation_sharpe_diff']:.4f}); "
                    f"maximized abs diagnostic contrast ({row['abs_diagnostic_score_diff']:.4f}); "
                    "selected without using favorable holdout direction."
                ),
                "primary_diagnostic_dimension_illustrated": "|".join(sorted(set(dims))) or "aggregate",
                "holdout_contrast_view": holdout_view,
            }
        )
    rat = pd.DataFrame(rationales)
    out = pd.concat([out.reset_index(drop=True), rat], axis=1)
    return out
