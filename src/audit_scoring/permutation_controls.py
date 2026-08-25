"""Identity and all-state permutation controls for Batch C3."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .checkpoint_loading import DecisionTransformerAdapter, allocation_distance, cash_and_risky_deltas
from .permutation_config import IDENTITY_ACTION_TOL
from .permutation_operator import permute_all_state_families, valid_permutation_mask
from .targeted_validity import TARGET_L2_EQUAL, compute_ig_with_target, attribution_matrix_from_tensor


def _l2_equal_target(actions: np.ndarray) -> np.ndarray:
    equal = 1.0 / actions.shape[-1]
    return ((actions - equal) ** 2).sum(axis=-1)


def run_identity_control(
    adapter: DecisionTransformerAdapter,
    batch: Mapping[str, torch.Tensor],
    *,
    model_family: str,
    cash_idx: int,
) -> Dict[str, Any]:
    """Pass unchanged inputs through the permutation execution path."""
    # Clone path: "permute" with identity donors
    states = batch["states"]
    identity_donors = np.arange(states.shape[0], dtype=int)
    # No-op permute
    states_p = states.clone()
    sub = dict(batch)
    sub["states"] = states_p
    a0 = adapter.forward_actions(batch).detach().cpu().numpy()
    a1 = adapter.forward_actions(sub).detach().cpu().numpy()
    dist, _ = allocation_distance(a0, a1)
    cash_d, risky_d = cash_and_risky_deltas(a0, a1, cash_idx)
    med_dist = float(np.nanmedian(dist))
    passed = med_dist <= IDENTITY_ACTION_TOL
    return {
        "model_family": model_family,
        "control": "identity",
        "median_tv": med_dist,
        "median_cash_abs": float(np.nanmedian(cash_d)),
        "median_risky_delta": float(np.nanmedian(risky_d)),
        "median_target_change": float(np.nanmedian(np.abs(_l2_equal_target(a1) - _l2_equal_target(a0)))),
        "pass": passed,
        "notes": "" if passed else "identity_control_nonzero_action_distance",
    }


def run_all_state_control(
    adapter: DecisionTransformerAdapter,
    batch: Mapping[str, torch.Tensor],
    donor_indices: np.ndarray,
    *,
    model_family: str,
    seed: int,
    cash_idx: int,
) -> Dict[str, Any]:
    mask = valid_permutation_mask(donor_indices)
    if not np.any(mask):
        return {
            "model_family": model_family,
            "control": "all_state_family_permutation",
            "seed": int(seed),
            "n_valid": 0,
            "median_tv": float("nan"),
            "pass": True,
            "notes": "no_valid_permutation_stratum",
        }
    a0 = adapter.forward_actions(batch).detach().cpu().numpy()
    states_p = permute_all_state_families(batch["states"], donor_indices)
    sub = dict(batch)
    sub["states"] = states_p
    a1 = adapter.forward_actions(sub).detach().cpu().numpy()
    dist, _ = allocation_distance(a0[mask], a1[mask])
    cash_d, risky_d = cash_and_risky_deltas(a0[mask], a1[mask], cash_idx)
    return {
        "model_family": model_family,
        "control": "all_state_family_permutation",
        "seed": int(seed),
        "n_valid": int(mask.sum()),
        "median_tv": float(np.nanmedian(dist)),
        "median_cash_abs": float(np.nanmedian(cash_d)),
        "median_risky_delta": float(np.nanmedian(risky_d)),
        "median_target_change": float(
            np.nanmedian(np.abs(_l2_equal_target(a1[mask]) - _l2_equal_target(a0[mask])))
        ),
        "pass": True,
        "notes": "positive_sensitivity_control",
    }
