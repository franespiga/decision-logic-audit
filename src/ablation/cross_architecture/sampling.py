"""
Market regimes, the common evaluation sample and regime-stratified sampling.

The four-regime configuration used here is the revised one in which
15--28 February 2020 belongs to the COVID crash period, so that the evaluation
window is partitioned without gaps and no ``Unknown`` label can arise.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Revised four-regime configuration (no gaps, therefore no "Unknown" label).
# The pre-revision configuration started the crash period on 2020-03-01, which
# left 17--28 February 2020 unlabelled.
REGIMES: Dict[str, Tuple[str, str]] = {
    "bull_market": ("2016-01-01", "2020-02-14"),
    "crash_period": ("2020-02-15", "2020-05-31"),
    "recovery_growth": ("2020-06-01", "2021-12-31"),
    "sideways_uncertain": ("2022-01-01", "2024-04-30"),
}

REGIME_ORDER: Tuple[str, ...] = tuple(REGIMES)

PERIODS: Dict[str, Tuple[Optional[str], Optional[str]]] = {
    "train": (None, "2015-12-31"),
    "eval": ("2016-01-01", "2024-04-30"),
    "holdout": ("2024-05-01", None),
}


def assign_regimes(dates: Sequence[pd.Timestamp], regimes: Optional[Dict] = None) -> np.ndarray:
    """Label each date with its regime. Raises if any date is unlabelled."""
    regimes = regimes or REGIMES
    bounds = {k: (pd.Timestamp(a), pd.Timestamp(b)) for k, (a, b) in regimes.items()}
    labels = []
    for dt in pd.DatetimeIndex(dates):
        hit = [name for name, (a, b) in bounds.items() if a <= dt <= b]
        if not hit:
            raise ValueError(
                f"date {dt.date()} falls outside the four-regime configuration; the "
                "configuration must partition the evaluation window without gaps"
            )
        if len(hit) > 1:
            raise ValueError(f"date {dt.date()} matches multiple regimes {hit}")
        labels.append(hit[0])
    return np.asarray(labels, dtype=object)


def period_dates(index: pd.DatetimeIndex, period: str) -> pd.DatetimeIndex:
    start, end = PERIODS[period]
    idx = pd.DatetimeIndex(index).sort_values()
    if start is not None:
        idx = idx[idx >= pd.Timestamp(start)]
    if end is not None:
        idx = idx[idx <= pd.Timestamp(end)]
    return idx


def common_dates(
    per_model_dates: Dict[str, Sequence[pd.Timestamp]],
) -> pd.DatetimeIndex:
    """Intersection of the dates at which every model has a valid observation."""
    sets = [set(pd.DatetimeIndex(v)) for v in per_model_dates.values()]
    if not sets:
        return pd.DatetimeIndex([])
    common = set.intersection(*sets)
    return pd.DatetimeIndex(sorted(common))


def stratified_sample(
    dates: pd.DatetimeIndex,
    n_target: int,
    seed: int = 42,
    regimes: Optional[Dict] = None,
) -> Tuple[pd.DatetimeIndex, pd.DataFrame]:
    """
    Draw a regime-balanced sample of ``n_target`` dates without replacement.

    Regimes are filled to equal quota where possible; a regime that cannot supply
    its quota contributes all of its dates and the shortfall is redistributed over
    the remaining regimes so that the requested total is still reached when the
    data allow it. The realised composition is returned alongside the sample.
    """
    regimes = regimes or REGIMES
    labels = assign_regimes(dates, regimes)
    rng = np.random.default_rng(seed)
    pools = {name: np.asarray(dates)[labels == name] for name in regimes if (labels == name).any()}
    if not pools:
        return pd.DatetimeIndex([]), pd.DataFrame()

    order = [name for name in REGIME_ORDER if name in pools]
    quota = {name: 0 for name in order}
    remaining = min(int(n_target), int(sum(len(p) for p in pools.values())))
    # Water-filling: repeatedly give one slot to every regime that still has dates.
    while remaining > 0:
        open_regimes = [name for name in order if quota[name] < len(pools[name])]
        if not open_regimes:
            break
        for name in open_regimes:
            if remaining == 0:
                break
            quota[name] += 1
            remaining -= 1

    picked: List[pd.Timestamp] = []
    for name in order:
        pool = np.sort(pools[name])
        k = quota[name]
        if k >= len(pool):
            chosen = pool
        else:
            chosen = pool[np.sort(rng.choice(len(pool), size=k, replace=False))]
        picked.extend(list(chosen))

    sample = pd.DatetimeIndex(sorted(pd.DatetimeIndex(picked)))
    if sample.has_duplicates:
        raise AssertionError("stratified sample contains duplicated dates")
    composition = pd.DataFrame(
        {
            "regime": order,
            "available": [len(pools[n]) for n in order],
            "sampled": [quota[n] for n in order],
        }
    )
    composition["share_of_sample"] = composition["sampled"] / max(composition["sampled"].sum(), 1)
    return sample, composition


def validate_sample(
    sample: pd.DatetimeIndex,
    holdout_start: str = "2024-05-01",
    regimes: Optional[Dict] = None,
) -> Dict[str, object]:
    """Assertions required by the design: no duplicates, no Unknown, no leakage."""
    if sample.has_duplicated if hasattr(sample, "has_duplicated") else sample.has_duplicates:
        raise AssertionError("diagnostic sample contains duplicated dates")
    labels = assign_regimes(sample, regimes)  # raises on any unlabelled date
    if "Unknown" in set(labels):
        raise AssertionError("diagnostic sample contains an Unknown regime label")
    leak = sample[sample >= pd.Timestamp(holdout_start)]
    if len(leak):
        raise AssertionError(f"{len(leak)} sampled dates leak into the temporal holdout")
    counts = pd.Series(labels).value_counts().to_dict()
    return {"n": int(len(sample)), "regime_counts": counts, "unknown": 0}


def sample_size_table(
    per_model_dates: Dict[str, Sequence[pd.Timestamp]],
    sample: pd.DatetimeIndex,
    regimes: Optional[Dict] = None,
) -> pd.DataFrame:
    """Per-model, per-regime observation counts for the rollout and the sample."""
    regimes = regimes or REGIMES
    rows = []
    sample_set = set(pd.DatetimeIndex(sample))
    for model_id, dates in per_model_dates.items():
        idx = pd.DatetimeIndex(dates)
        labels = assign_regimes(idx, regimes)
        in_sample = np.array([d in sample_set for d in idx])
        for name in REGIME_ORDER:
            mask = labels == name
            rows.append(
                {
                    "model_id": model_id,
                    "regime": name,
                    "n_evaluation_dates": int(mask.sum()),
                    "n_diagnostic_sample": int((mask & in_sample).sum()),
                }
            )
        rows.append(
            {
                "model_id": model_id,
                "regime": "all",
                "n_evaluation_dates": int(len(idx)),
                "n_diagnostic_sample": int(in_sample.sum()),
            }
        )
    return pd.DataFrame(rows)


def train_reference_statistics(
    frame: pd.DataFrame, train_end: str = "2015-12-31"
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Per-feature mean and standard deviation computed on the training period only.

    These are the replacement values used for deletion and occlusion baselines.
    Using the training window keeps every baseline free of evaluation-period
    information, i.e. no preprocessing statistic is refitted on evaluation data.
    """
    train = frame.loc[: pd.Timestamp(train_end)]
    if train.empty:
        raise ValueError("no training-period rows available for reference statistics")
    mean = train.mean(axis=0).values.astype(np.float64)
    std = train.std(axis=0).replace(0.0, 1.0).values.astype(np.float64)
    return mean, std, train
