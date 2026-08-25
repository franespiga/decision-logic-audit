"""Perturbed-state plausibility diagnostics (Batch C2)."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from .cardinality_controls import family_raw_cardinality, guided_raw_count, sample_cardinality_matched_families
from .baseline_validity import build_empirical_median_baseline, ranking_from_abs_profile
from .faithfulness_validity import replace_families_with_baseline


def _flatten_states(states: torch.Tensor) -> np.ndarray:
    arr = states.detach().cpu().numpy()
    return arr.reshape(arr.shape[0], -1)


def nearest_observed_distance(query: np.ndarray, pool: np.ndarray) -> np.ndarray:
    # query [N,D], pool [M,D] — compute min L2 to pool (excluding exact self if present)
    # For speed use chunked computation
    n = query.shape[0]
    out = np.empty(n, dtype=float)
    chunk = 64
    for start in range(0, n, chunk):
        q = query[start : start + chunk]
        # distances to all pool
        # (Q,1,D)-(1,M,D)
        d2 = ((q[:, None, :] - pool[None, :, :]) ** 2).sum(axis=-1)
        out[start : start + chunk] = np.sqrt(d2.min(axis=1))
    return out


def standardized_euclidean(query: np.ndarray, reference: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mu = reference.mean(axis=0)
    sd = reference.std(axis=0)
    sd = np.where(sd < eps, 1.0, sd)
    zq = (query - mu) / sd
    zr = (reference - mu) / sd
    # distance of each query to its own original counterpart when shapes match
    if query.shape == reference.shape:
        return np.sqrt(((zq - zr) ** 2).sum(axis=1))
    return np.sqrt((zq ** 2).sum(axis=1))


def mahalanobis_distance(query: np.ndarray, reference: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    mu = reference.mean(axis=0)
    # diagonal covariance for stability in high-D
    var = reference.var(axis=0) + eps
    d = (query - mu) / np.sqrt(var)
    return np.sqrt((d ** 2).sum(axis=1))


def range_violation_rate(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    lo = reference.min(axis=0)
    hi = reference.max(axis=0)
    below = query < lo
    above = query > hi
    return (below | above).mean(axis=1)


def assess_plausibility(
    original_states: torch.Tensor,
    guided_states: torch.Tensor,
    random_states: torch.Tensor,
    pool_states: torch.Tensor,
    *,
    model_family: str,
    depth: int,
) -> Dict[str, float]:
    orig = _flatten_states(original_states)
    guid = _flatten_states(guided_states)
    rand = _flatten_states(random_states)
    pool = _flatten_states(pool_states)

    g_nn = nearest_observed_distance(guid, pool)
    r_nn = nearest_observed_distance(rand, pool)
    g_se = standardized_euclidean(guid, orig)
    r_se = standardized_euclidean(rand, orig)
    g_mah = mahalanobis_distance(guid, pool)
    r_mah = mahalanobis_distance(rand, pool)
    g_out = range_violation_rate(guid, pool)
    r_out = range_violation_rate(rand, pool)

    guided_more_off = float(np.nanmean(g_nn > r_nn))
    status = (
        "UNSUPPORTED"
        if guided_more_off > 0.6 and float(np.nanmedian(g_nn - r_nn)) > 0
        else ("SUPPORTED" if guided_more_off < 0.4 else "MIXED")
    )
    return {
        "model_family": model_family,
        "depth": depth,
        "median_guided_nn_distance": float(np.nanmedian(g_nn)),
        "median_random_nn_distance": float(np.nanmedian(r_nn)),
        "median_guided_std_euclidean": float(np.nanmedian(g_se)),
        "median_random_std_euclidean": float(np.nanmedian(r_se)),
        "median_guided_mahalanobis": float(np.nanmedian(g_mah)),
        "median_random_mahalanobis": float(np.nanmedian(r_mah)),
        "mean_guided_range_violation": float(np.nanmean(g_out)),
        "mean_random_range_violation": float(np.nanmean(r_out)),
        "fraction_guided_farther_than_random": guided_more_off,
        "perturbed_state_plausibility_status": status,
    }


def run_plausibility_for_family(
    batch_states: torch.Tensor,
    pool_states: torch.Tensor,
    family_ids: np.ndarray,
    abs_profile: pd.Series,
    *,
    model_family: str,
    max_depth: int = 3,
    random_seed: int = 42,
) -> pd.DataFrame:
    ranked = ranking_from_abs_profile(abs_profile)
    card = family_raw_cardinality(family_ids)
    median_baseline = build_empirical_median_baseline(pool_states, batch_states)
    rng = np.random.default_rng(random_seed)
    rows = []
    for depth in range(1, min(max_depth, len(ranked)) + 1):
        families = ranked[:depth]
        target_raw = guided_raw_count(family_ids, families)
        guided = replace_families_with_baseline(batch_states, median_baseline, family_ids, families)
        choice, _, _ = sample_cardinality_matched_families(
            ranked,
            card,
            guided_families=families,
            n_families=depth,
            target_raw_count=target_raw,
            rng=rng,
        )
        random = replace_families_with_baseline(batch_states, median_baseline, family_ids, choice)
        rows.append(
            assess_plausibility(
                batch_states, guided, random, pool_states, model_family=model_family, depth=depth
            )
        )
    return pd.DataFrame(rows)
