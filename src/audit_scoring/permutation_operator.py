"""Regime-preserving donor permutation operator for Batch C3."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


def generate_derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    """Deterministic derangement of 0..n-1 (n >= 2)."""
    if n < 2:
        raise ValueError("Derangement requires n >= 2")
    idx = np.arange(n)
    for _ in range(1000):
        perm = rng.permutation(n)
        if not np.any(perm == idx):
            return perm.astype(int)
    # Fallback: cycle shift
    return np.roll(idx, 1).astype(int)


def build_regime_donor_map(
    regimes: Sequence[str],
    *,
    seed: int,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Map each local observation index to a same-regime donor (derangement).

    Returns donor_indices (len N; -1 where regime too small) and assignment table.
    """
    regimes_arr = np.asarray(regimes, dtype=object)
    n = len(regimes_arr)
    donors = np.full(n, -1, dtype=int)
    rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(int(seed))

    for regime in sorted(set(regimes_arr.tolist())):
        members = np.flatnonzero(regimes_arr == regime)
        if len(members) < 2:
            for m in members:
                rows.append(
                    {
                        "regime": str(regime),
                        "recipient_local_index": int(m),
                        "donor_local_index": -1,
                        "same_regime": True,
                        "is_derangement": False,
                        "notes": "regime_stratum_size_lt_2",
                    }
                )
            continue
        # Sub-rng per regime (stable across PYTHONHASHSEED)
        regime_digest = int.from_bytes(str(regime).encode("utf-8"), "little", signed=False) % 10_000
        sub = np.random.default_rng(int(seed) * 1_000_003 + regime_digest)
        perm = generate_derangement(len(members), sub)
        for i, recip in enumerate(members):
            donor = int(members[perm[i]])
            donors[recip] = donor
            rows.append(
                {
                    "regime": str(regime),
                    "recipient_local_index": int(recip),
                    "donor_local_index": donor,
                    "same_regime": True,
                    "is_derangement": donor != int(recip),
                    "notes": "",
                }
            )
    return donors, pd.DataFrame(rows)


def family_feature_mask(family_ids: np.ndarray, families: Sequence[str]) -> np.ndarray:
    fam_set = set(map(str, families))
    return np.asarray([str(f) in fam_set for f in family_ids], dtype=bool)


def permute_state_families(
    states: torch.Tensor,
    donor_indices: np.ndarray,
    family_ids: np.ndarray,
    families: Sequence[str],
) -> torch.Tensor:
    """Replace selected feature-family blocks using donor observation indices.

    states: [N, T, F]
    donor_indices: [N] with -1 meaning leave unchanged
    """
    out = states.clone()
    mask = family_feature_mask(family_ids, families)
    if not np.any(mask):
        return out
    mask_t = torch.as_tensor(mask, device=states.device)
    for i, donor in enumerate(donor_indices):
        if int(donor) < 0 or int(donor) == i:
            continue
        out[i, :, mask_t] = states[int(donor), :, mask_t]
    return out


def permute_all_state_families(
    states: torch.Tensor,
    donor_indices: np.ndarray,
) -> torch.Tensor:
    """Permute entire state tensor across donors (all feature families)."""
    out = states.clone()
    for i, donor in enumerate(donor_indices):
        if int(donor) < 0 or int(donor) == i:
            continue
        out[i] = states[int(donor)]
    return out


def valid_permutation_mask(donor_indices: np.ndarray) -> np.ndarray:
    return np.asarray([(int(d) >= 0 and int(d) != i) for i, d in enumerate(donor_indices)], dtype=bool)
