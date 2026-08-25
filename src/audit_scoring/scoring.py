"""Percentile scoring for model- and family-level diagnostics."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import (
    CORE_CONTRIBUTION_METRICS,
    CORE_NECESSITY_METRICS,
    CORE_STRUCTURE_METRICS,
    HIGHER_IS_BETTER_METRICS,
    LOWER_IS_BETTER_METRICS,
    PROFILES,
    validate_weights,
)


def average_rank(values: Sequence[float], *, higher_is_better: bool) -> np.ndarray:
    """Return 1-based average ranks for ties."""
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    mask = np.isfinite(arr)
    if mask.sum() == 0:
        return out
    x = arr[mask]
    # denser ranking via argsort of oriented values
    oriented = x if higher_is_better else -x
    order = np.argsort(oriented, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and oriented[order[j + 1]] == oriented[order[i]]:
            j += 1
        # average rank of tied group (1-based)
        avg = 0.5 * ((i + 1) + (j + 1))
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    out[mask] = ranks
    return out


def percentile_scores(values: Sequence[float], *, higher_is_better: bool) -> np.ndarray:
    """Oriented percentile scores using average ranks for ties."""
    arr = np.asarray(values, dtype=float)
    ranks = average_rank(arr, higher_is_better=higher_is_better)
    out = np.full(arr.shape, np.nan, dtype=float)
    mask = np.isfinite(ranks)
    n = int(mask.sum())
    if n <= 1:
        out[mask] = 50.0 if n == 1 else np.nan
        return out
    out[mask] = 100.0 * (ranks[mask] - 1.0) / (n - 1.0)
    return out


def metric_direction(metric: str) -> str:
    if metric in LOWER_IS_BETTER_METRICS:
        return "lower_is_better"
    if metric in HIGHER_IS_BETTER_METRICS:
        return "higher_is_better"
    return "descriptive"


def score_model_metrics(raw: pd.DataFrame) -> pd.DataFrame:
    """Create per-model oriented percentile scores and dimension aggregates."""
    df = raw.copy()
    scored = df[
        [
            c
            for c in [
                "model_id",
                "model_family",
                "model_type",
                "technical_indicators",
                "contribution_validity_status",
                "necessity_validity_status",
                "structure_validity_status",
                "overall_raw_validity_status",
                "warnings",
            ]
            if c in df.columns
        ]
    ].copy()

    for metric in list(CORE_CONTRIBUTION_METRICS) + list(CORE_NECESSITY_METRICS) + list(
        CORE_STRUCTURE_METRICS
    ):
        if metric not in df.columns:
            scored[f"pct_{metric}"] = np.nan
            continue
        higher = metric_direction(metric) == "higher_is_better"
        scored[f"pct_{metric}"] = percentile_scores(df[metric].to_numpy(), higher_is_better=higher)

    def _mean_cols(cols: Sequence[str]) -> pd.Series:
        block = scored.loc[:, cols]
        return block.mean(axis=1, skipna=False)

    c_cols = [f"pct_{m}" for m in CORE_CONTRIBUTION_METRICS]
    n_cols = [f"pct_{m}" for m in CORE_NECESSITY_METRICS]
    s_cols = [f"pct_{m}" for m in CORE_STRUCTURE_METRICS]

    scored["contribution_score_model"] = _mean_cols(c_cols)
    scored["necessity_score_model"] = _mean_cols(n_cols)
    scored["structure_score_model"] = _mean_cols(s_cols)
    scored["contribution_complete"] = scored[c_cols].notna().all(axis=1)
    scored["necessity_complete"] = scored[n_cols].notna().all(axis=1)
    scored["structure_complete"] = scored[s_cols].notna().all(axis=1)
    complete = (
        scored["contribution_complete"]
        & scored["necessity_complete"]
        & scored["structure_complete"]
    )
    scored["diagnostic_score_model"] = np.where(
        complete,
        (
            scored["contribution_score_model"]
            + scored["necessity_score_model"]
            + scored["structure_score_model"]
        )
        / 3.0,
        np.nan,
    )
    scored["model_score_status"] = np.where(
        complete, "scored", "not_scored_missing_core_dimension"
    )
    scored["validity_status"] = df.get(
        "overall_raw_validity_status", pd.Series(["incomplete"] * len(df))
    )
    return scored


def score_family_medians(
    family_raw: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Percentile-rank family medians then form C/N/S/D."""
    weights = dict(weights or PROFILES["balanced"])
    validate_weights(weights)
    df = family_raw.copy()
    out = df[
        [
            c
            for c in [
                "model_family",
                "n_models_expected",
                "n_models_available",
                "model_coverage_ratio",
                "contribution_iqr_summary",
                "necessity_iqr_summary",
                "structure_iqr_summary",
                "high_within_family_dispersion",
                "validity_status",
                "warnings",
            ]
            if c in df.columns
        ]
    ].copy()

    metric_map = {
        "contribution_rank_stability": "median_contribution_rank_stability",
        "contribution_sign_stability": "median_contribution_sign_stability",
        "necessity_temporal_profile_stability": "median_necessity_temporal_profile_stability",
        "necessity_temporal_instability": "median_necessity_temporal_instability",
        "structure_effective_rank_ratio": "median_structure_effective_rank_ratio",
        "structure_probe_margin": "median_structure_probe_margin",
    }

    pct_cols: Dict[str, str] = {}
    for metric, col in metric_map.items():
        if col not in df.columns:
            out[f"pct_{metric}"] = np.nan
            pct_cols[metric] = f"pct_{metric}"
            continue
        higher = metric_direction(metric) == "higher_is_better"
        out[f"pct_{metric}"] = percentile_scores(df[col].to_numpy(), higher_is_better=higher)
        pct_cols[metric] = f"pct_{metric}"

    out["contribution_score"] = out[
        [pct_cols[m] for m in CORE_CONTRIBUTION_METRICS]
    ].mean(axis=1, skipna=False)
    out["necessity_score"] = out[
        [pct_cols[m] for m in CORE_NECESSITY_METRICS]
    ].mean(axis=1, skipna=False)
    out["structure_score"] = out[
        [pct_cols[m] for m in CORE_STRUCTURE_METRICS]
    ].mean(axis=1, skipna=False)

    out["contribution_metric_count"] = out[
        [pct_cols[m] for m in CORE_CONTRIBUTION_METRICS]
    ].notna().sum(axis=1)
    out["necessity_metric_count"] = out[
        [pct_cols[m] for m in CORE_NECESSITY_METRICS]
    ].notna().sum(axis=1)
    out["structure_metric_count"] = out[
        [pct_cols[m] for m in CORE_STRUCTURE_METRICS]
    ].notna().sum(axis=1)

    dim_complete = (
        (out["contribution_metric_count"] == len(CORE_CONTRIBUTION_METRICS))
        & (out["necessity_metric_count"] == len(CORE_NECESSITY_METRICS))
        & (out["structure_metric_count"] == len(CORE_STRUCTURE_METRICS))
    )
    out["diagnostic_score"] = np.where(
        dim_complete,
        weights["contribution"] * out["contribution_score"]
        + weights["necessity"] * out["necessity_score"]
        + weights["structure"] * out["structure_score"],
        np.nan,
    )
    out["family_score_status"] = np.where(
        dim_complete, "scored", "not_scored_missing_core_dimension"
    )
    return out
