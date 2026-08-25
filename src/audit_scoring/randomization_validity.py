"""Parameter-randomization sanity checks (Batch C Phase 5)."""

from __future__ import annotations

import gc
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau, spearmanr

from .baseline_validity import (
    build_empirical_median_baseline,
    compute_ig_family_profiles,
    ranking_from_abs_profile,
    signed_agreement,
    top_k_jaccard,
)
from .checkpoint_loading import (
    DecisionTransformerAdapter,
    allocation_distance,
    cash_and_risky_deltas,
)
from .validity_config import (
    ACTION_DISTANCE_TOL,
    RAND_MIXED_FULL_SIM,
    RAND_MIXED_GAP,
    RAND_SUPPORTED_FULL_SIM,
    RAND_SUPPORTED_GAP,
)

DEFAULT_LEVELS = ("output_head", "final_block_and_head", "full_model")


def _align_profiles(a: pd.Series, b: pd.Series) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    labels = sorted(set(a.index.astype(str)) | set(b.index.astype(str)))
    xa = a.reindex(labels).fillna(0.0).to_numpy(dtype=float)
    xb = b.reindex(labels).fillna(0.0).to_numpy(dtype=float)
    return xa, xb, labels


def compare_profiles(
    original_abs: pd.Series,
    original_signed: pd.Series,
    random_abs: pd.Series,
    random_signed: pd.Series,
    top_k: int,
) -> Dict[str, float]:
    xa, xb, labels = _align_profiles(original_abs, random_abs)
    tau = kendalltau(xa, xb, variant="b").statistic
    spear = spearmanr(xa, xb).correlation
    rank_o = ranking_from_abs_profile(original_abs.reindex(labels).fillna(0.0))
    rank_r = ranking_from_abs_profile(random_abs.reindex(labels).fillna(0.0))
    k = min(top_k, len(labels))
    return {
        "kendall_tau_b": float(tau) if np.isfinite(tau) else float("nan"),
        "spearman": float(spear) if np.isfinite(spear) else float("nan"),
        "top_k_jaccard": top_k_jaccard(rank_o, rank_r, k),
        "top_k_sign_agreement": signed_agreement(
            original_signed.reindex(labels).fillna(0.0),
            random_signed.reindex(labels).fillna(0.0),
            rank_o[:k],
        ),
    }


def classify_randomization_status(
    *,
    baseline_reference_concordance: float,
    full_randomization_similarity: float,
    median_action_distance: float,
) -> Tuple[str, Dict[str, float]]:
    if not np.isfinite(full_randomization_similarity) or not np.isfinite(baseline_reference_concordance):
        return "INCOMPLETE", {}
    gap = float(baseline_reference_concordance - max(0.0, full_randomization_similarity))
    metrics = {
        "baseline_reference_concordance": float(baseline_reference_concordance),
        "full_randomization_similarity": float(full_randomization_similarity),
        "randomization_gap": gap,
        "median_action_distance": float(median_action_distance),
    }
    identical = abs(full_randomization_similarity) > 0.95 and median_action_distance <= ACTION_DISTANCE_TOL
    if (
        full_randomization_similarity <= RAND_SUPPORTED_FULL_SIM
        and gap >= RAND_SUPPORTED_GAP
        and median_action_distance > ACTION_DISTANCE_TOL
        and not identical
    ):
        return "SUPPORTED", metrics
    if full_randomization_similarity > RAND_MIXED_FULL_SIM or gap < RAND_MIXED_GAP or identical:
        return "UNSUPPORTED", metrics
    if full_randomization_similarity <= RAND_MIXED_FULL_SIM and gap >= RAND_MIXED_GAP:
        return "MIXED", metrics
    return "UNSUPPORTED", metrics


def run_parameter_randomization(
    adapter: DecisionTransformerAdapter,
    checkpoint_path,
    batch_tensors: Mapping[str, torch.Tensor],
    pool_states: torch.Tensor,
    family_ids: np.ndarray,
    original_abs: pd.Series,
    original_signed: pd.Series,
    *,
    model_family: str,
    seeds: Sequence[int],
    levels: Sequence[str] = DEFAULT_LEVELS,
    n_steps: int,
    batch_size: int,
    top_k: int,
    cash_idx: int,
    baseline_reference_concordance: float,
) -> Dict[str, Any]:
    median_baseline = build_empirical_median_baseline(pool_states, batch_tensors["states"])
    seed_rows: List[Dict[str, Any]] = []
    progression_rows: List[Dict[str, Any]] = []
    available_levels: List[str] = []
    level_notes: List[str] = []

    # Original actions once
    adapter.load_fresh_copy(checkpoint_path)
    original_actions = adapter.forward_actions(batch_tensors).detach().cpu().numpy()

    for level in levels:
        level_taus: List[float] = []
        for seed in seeds:
            adapter.load_fresh_copy(checkpoint_path)
            try:
                reset_names = adapter.reinitialize_modules(level, int(seed))
                if level not in available_levels:
                    available_levels.append(level)
            except Exception as exc:
                level_notes.append(f"{level}: unavailable ({exc})")
                seed_rows.append(
                    {
                        "model_family": model_family,
                        "randomization_level": level,
                        "seed": int(seed),
                        "status": "INCOMPLETE",
                        "notes": str(exc),
                    }
                )
                adapter.release()
                continue

            rand_actions = adapter.forward_actions(batch_tensors).detach().cpu().numpy()
            dist, dist_name = allocation_distance(original_actions, rand_actions)
            cash_d, risky_d = cash_and_risky_deltas(original_actions, rand_actions, cash_idx)

            abs_p, signed_p, _ = compute_ig_family_profiles(
                adapter,
                batch_tensors,
                median_baseline,
                family_ids,
                n_steps=n_steps,
                batch_size=batch_size,
            )
            comps = compare_profiles(original_abs, original_signed, abs_p, signed_p, top_k)
            level_taus.append(comps["kendall_tau_b"])
            seed_rows.append(
                {
                    "model_family": model_family,
                    "randomization_level": level,
                    "seed": int(seed),
                    "status": "computed",
                    "reset_modules": "|".join(reset_names),
                    "distance_metric": dist_name,
                    "median_allocation_distance": float(np.nanmedian(dist)),
                    "median_cash_abs_change": float(np.nanmedian(cash_d)),
                    "median_risky_exposure_change": float(np.nanmedian(risky_d)),
                    **comps,
                }
            )
            adapter.release()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if level_taus:
            progression_rows.append(
                {
                    "model_family": model_family,
                    "randomization_level": level,
                    "median_kendall_tau_b": float(np.nanmedian(level_taus)),
                    "iqr_kendall_tau_b": float(
                        np.nanpercentile(level_taus, 75) - np.nanpercentile(level_taus, 25)
                    )
                    if len(level_taus) >= 2
                    else 0.0,
                    "n_seeds": len(level_taus),
                }
            )

    seed_df = pd.DataFrame(seed_rows)
    prog_df = pd.DataFrame(progression_rows)

    full_rows = seed_df[seed_df["randomization_level"] == "full_model"] if not seed_df.empty else seed_df
    if full_rows.empty or "kendall_tau_b" not in full_rows.columns:
        status, metrics = "INCOMPLETE", {}
    else:
        full_sim = float(np.nanmedian(full_rows["kendall_tau_b"].astype(float)))
        med_dist = float(np.nanmedian(full_rows["median_allocation_distance"].astype(float)))
        status, metrics = classify_randomization_status(
            baseline_reference_concordance=baseline_reference_concordance,
            full_randomization_similarity=full_sim,
            median_action_distance=med_dist,
        )

    # Progressive deterioration: similarity should not systematically increase
    violation = False
    if len(prog_df) >= 2:
        order = [l for l in DEFAULT_LEVELS if l in set(prog_df["randomization_level"])]
        vals = [
            float(prog_df.loc[prog_df.randomization_level == l, "median_kendall_tau_b"].iloc[0])
            for l in order
        ]
        for i in range(1, len(vals)):
            if np.isfinite(vals[i]) and np.isfinite(vals[i - 1]) and vals[i] > vals[i - 1] + 0.05:
                violation = True
                break

    per_family = pd.DataFrame(
        [
            {
                "model_family": model_family,
                "randomization_status": status,
                **metrics,
                "available_levels": "|".join(available_levels),
                "progression_violation": bool(violation),
                "notes": "; ".join(dict.fromkeys(level_notes)),
            }
        ]
    )
    return {
        "status": status,
        "metrics": metrics,
        "per_family": per_family,
        "by_level_seed": seed_df,
        "progression": prog_df,
    }
