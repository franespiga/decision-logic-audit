"""Paper regime schedule for Comment 3 (user-approved)."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd

PAPER_REGIMES: Dict[str, Tuple[str, str]] = {
    "bull_market": ("2016-01-01", "2020-02-15"),
    "crash_period": ("2020-02-16", "2020-05-31"),
    "recovery_growth": ("2020-06-01", "2021-12-31"),
    "sideways_uncertain": ("2022-01-01", "2024-05-30"),
}

REGIME_ORDER: Tuple[str, ...] = tuple(PAPER_REGIMES)


def assign_paper_regimes(
    dates: Sequence[pd.Timestamp],
    regimes: Dict[str, Tuple[str, str]] | None = None,
    *,
    allow_unknown: bool = False,
) -> np.ndarray:
    """Label each date. Raises on Unknown unless allow_unknown=True."""
    regimes = regimes or PAPER_REGIMES
    bounds = {k: (pd.Timestamp(a), pd.Timestamp(b)) for k, (a, b) in regimes.items()}
    labels = []
    for dt in pd.DatetimeIndex(dates):
        hit = [name for name, (a, b) in bounds.items() if a <= dt <= b]
        if not hit:
            if allow_unknown:
                labels.append("Unknown")
                continue
            raise ValueError(f"date {dt.date()} outside paper regime schedule")
        if len(hit) > 1:
            raise ValueError(f"date {dt.date()} matches multiple regimes {hit}")
        labels.append(hit[0])
    return np.asarray(labels, dtype=object)


def count_regime_episodes(labels: np.ndarray, regime: str) -> int:
    """Number of contiguous runs equal to ``regime``."""
    if len(labels) == 0:
        return 0
    mask = labels == regime
    if not mask.any():
        return 0
    # new episode when True after False (or at start)
    prev = np.concatenate([[False], mask[:-1]])
    return int(np.sum(mask & ~prev))
