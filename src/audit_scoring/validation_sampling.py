"""Deterministic validation-period sampling for Batch C."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import REGIME_BOUNDARIES, regime_label_for_date
from .validity_config import PRIMARY_REGIMES


def assign_regimes(dates: Sequence[Any]) -> np.ndarray:
    labels = []
    for d in pd.to_datetime(pd.Index(dates)):
        labels.append(regime_label_for_date(pd.Timestamp(d)))
    return np.asarray(labels, dtype=object)


def _quota_plan(n_regimes: int, total: int, per_regime: int) -> List[int]:
    if per_regime * n_regimes == total:
        return [per_regime] * n_regimes
    base = total // n_regimes
    rem = total % n_regimes
    return [base + (1 if i < rem else 0) for i in range(n_regimes)]


def stratified_sample_indices(
    dates: Sequence[Any],
    *,
    n_samples: int,
    per_regime: int,
    random_seed: int,
    regimes: Sequence[str] = PRIMARY_REGIMES,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Regime-stratified deterministic sampling with quota redistribution.

    Returns (selected_indices, regime_labels_for_all, selection_rule).
    """
    dates_idx = pd.to_datetime(pd.Index(dates))
    n = len(dates_idx)
    labels = assign_regimes(dates_idx)
    if n == 0:
        return np.asarray([], dtype=int), labels, "empty"
    if n <= n_samples:
        return np.arange(n, dtype=int), labels, "all_available"

    rng = np.random.default_rng(int(random_seed))
    regime_list = [r for r in regimes if np.any(labels == r)]
    if not regime_list:
        # Fall back to evenly spaced chronological sample.
        idx = np.linspace(0, n - 1, n_samples, dtype=int)
        return np.unique(idx), labels, "chronological_fallback"

    quotas = _quota_plan(len(regime_list), n_samples, per_regime)
    selected: List[int] = []
    leftover_budget = 0
    available_pool: List[int] = []

    for i, regime in enumerate(regime_list):
        candidates = np.flatnonzero(labels == regime)
        quota = quotas[i]
        take = min(len(candidates), quota)
        leftover_budget += max(0, quota - take)
        if take == 0:
            continue
        if take == len(candidates):
            chosen = candidates
        else:
            positions = np.linspace(0, len(candidates) - 1, take)
            jitter = rng.uniform(-0.15, 0.15, size=take)
            pos = np.clip(np.rint(positions + jitter), 0, len(candidates) - 1).astype(int)
            chosen = np.unique(candidates[pos])
            if len(chosen) < take:
                remaining = np.setdiff1d(candidates, chosen, assume_unique=False)
                add = rng.choice(remaining, size=take - len(chosen), replace=False)
                chosen = np.sort(np.concatenate([chosen, add]))
        selected.extend(chosen.tolist())
        remaining = np.setdiff1d(candidates, chosen, assume_unique=False)
        available_pool.extend(remaining.tolist())

    selected_arr = np.unique(np.asarray(selected, dtype=int))
    need = n_samples - len(selected_arr)
    if need > 0:
        pool = np.setdiff1d(np.arange(n), selected_arr, assume_unique=False)
        if len(pool) > 0:
            # Prefer leftover within-regime remainders first.
            prefer = np.intersect1d(pool, np.asarray(available_pool, dtype=int), assume_unique=False)
            order = prefer if len(prefer) else pool
            add_n = min(need, len(order))
            # Deterministic: sort then take evenly spaced.
            order = np.sort(order)
            picks = order[np.linspace(0, len(order) - 1, add_n, dtype=int)] if add_n else np.asarray([], dtype=int)
            if len(np.unique(picks)) < add_n:
                picks = rng.choice(order, size=add_n, replace=False)
            selected_arr = np.sort(np.unique(np.concatenate([selected_arr, picks])))

    if len(selected_arr) > n_samples:
        selected_arr = np.sort(rng.choice(selected_arr, size=n_samples, replace=False))

    rule = (
        f"stratified_regimes seed={random_seed} target={n_samples} "
        f"per_regime_pref={per_regime} regimes={list(regime_list)}"
    )
    return selected_arr.astype(int), labels, rule


def build_sample_manifest(
    model_family: str,
    dates: Sequence[Any],
    primary_idx: np.ndarray,
    randomization_idx: np.ndarray,
    background_idx: np.ndarray,
    *,
    labels: np.ndarray,
    random_seed: int,
    primary_rule: str,
    randomization_rule: str,
    background_rule: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    date_index = pd.to_datetime(pd.Index(dates))

    def _add(indices: np.ndarray, role: str, rule: str) -> None:
        for local_i in indices:
            rows.append(
                {
                    "model_family": model_family,
                    "sample_role": role,
                    "observation_index": int(local_i),
                    "date": pd.Timestamp(date_index[int(local_i)]).isoformat(),
                    "regime": str(labels[int(local_i)]),
                    "random_seed": int(random_seed),
                    "selection_rule": rule,
                }
            )

    _add(primary_idx, "primary", primary_rule)
    _add(randomization_idx, "randomization", randomization_rule)
    _add(background_idx, "background", background_rule)
    return pd.DataFrame(rows)


def assert_validation_dates(dates: Sequence[Any], validation_end: pd.Timestamp) -> None:
    date_index = pd.to_datetime(pd.Index(dates))
    if len(date_index) and date_index.max() > validation_end:
        raise AssertionError(
            f"Sample contains dates after validation end {validation_end}: max={date_index.max()}"
        )
