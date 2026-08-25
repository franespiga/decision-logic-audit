"""Cardinality-matched random controls for deletion/insertion (Batch C2)."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


def family_raw_cardinality(family_ids: np.ndarray) -> Dict[str, int]:
    families = np.asarray(family_ids, dtype=object)
    out: Dict[str, int] = {}
    for fam in sorted(set(families.tolist())):
        out[str(fam)] = int(np.sum(families == fam))
    return out


def guided_raw_count(family_ids: np.ndarray, selected_families: Sequence[str]) -> int:
    fam_set = set(map(str, selected_families))
    return int(sum(1 for f in family_ids if str(f) in fam_set))


def sample_cardinality_matched_families(
    ranked_families: Sequence[str],
    family_cardinality: Dict[str, int],
    *,
    guided_families: Sequence[str],
    n_families: int,
    target_raw_count: int,
    rng: np.random.Generator,
    exclude_guided: bool = True,
) -> Tuple[List[str], int, int]:
    """Sample n_families minimizing |raw_count - target_raw_count|.

    Returns (families, raw_count, residual_abs_diff).
    """
    pool = [f for f in ranked_families if (f not in guided_families) or not exclude_guided]
    if len(pool) < n_families:
        pool = list(ranked_families)
    if len(pool) < n_families:
        raise ValueError("Not enough families for cardinality-matched control")

    best: List[str] = []
    best_diff = 10**9
    best_raw = -1
    # Exact / nearest search: try many random subsets; keep nearest cardinality.
    n_tries = min(2000, max(200, 50 * len(pool)))
    for _ in range(n_tries):
        choice = list(rng.choice(pool, size=n_families, replace=False))
        raw = int(sum(family_cardinality.get(f, 0) for f in choice))
        diff = abs(raw - int(target_raw_count))
        if diff < best_diff or (diff == best_diff and (not best or tuple(choice) < tuple(best))):
            best_diff = diff
            best = choice
            best_raw = raw
            if diff == 0:
                break
    return best, best_raw, int(best_diff)


def ti_availability_flag(family_name: str) -> bool:
    return str(family_name).endswith("_TI_True")
