"""Moving-block bootstrap for serially dependent allocation series."""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import numpy as np


def default_block_length(n: int) -> int:
    """Common rule of thumb: round(n**(1/3)), at least 1."""
    if n <= 1:
        return 1
    return max(1, int(round(n ** (1.0 / 3.0))))


def moving_block_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    """Circular moving-block bootstrap index vector of length n."""
    if n <= 0:
        return np.array([], dtype=int)
    L = max(1, min(int(block_length), n))
    n_blocks = int(np.ceil(n / L))
    starts = rng.integers(0, n, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + L) % n for s in starts])[:n]
    return idx.astype(int)


def block_bootstrap_ci(
    values: np.ndarray,
    stat_fn: Optional[Callable[[np.ndarray], float]] = None,
    *,
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    block_length: Optional[int] = None,
    seed: int = 42,
) -> Tuple[float, float, float, int, int]:
    """
    Returns (point_estimate, lower, upper, block_length, n_bootstrap).
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if stat_fn is None:
        stat_fn = lambda a: float(np.mean(a))
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 1, int(n_bootstrap)
    if n == 1:
        v = float(stat_fn(x))
        return v, v, v, 1, int(n_bootstrap)

    L = int(block_length) if block_length is not None else default_block_length(n)
    rng = np.random.default_rng(seed)
    point = float(stat_fn(x))
    draws = np.empty(int(n_bootstrap), dtype=float)
    for b in range(int(n_bootstrap)):
        idx = moving_block_indices(n, L, rng)
        draws[b] = float(stat_fn(x[idx]))
    alpha = 1.0 - ci
    lo, hi = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return point, float(lo), float(hi), L, int(n_bootstrap)


def paired_block_bootstrap_diff_ci(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    block_length: Optional[int] = None,
    seed: int = 42,
) -> Tuple[float, float, float, int, int]:
    """
    Paired difference mean(a - b) with a shared block resampling of dates.
    ``a`` and ``b`` must be aligned on the same dates.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError("paired series must have equal length")
    diff = a - b
    return block_bootstrap_ci(
        diff,
        stat_fn=lambda x: float(np.mean(x)),
        n_bootstrap=n_bootstrap,
        ci=ci,
        block_length=block_length,
        seed=seed,
    )
