"""Attribution-guided deletion and insertion faithfulness (Batch C Phases 6–8)."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from .baseline_validity import build_empirical_median_baseline, ranking_from_abs_profile
from .checkpoint_loading import (
    DecisionTransformerAdapter,
    allocation_distance,
    cash_and_risky_deltas,
)
from .validity_config import FAITHFULNESS_EFFECT_SIZE


def _family_feature_mask(family_ids: np.ndarray, families: Sequence[str]) -> np.ndarray:
    fam_set = set(map(str, families))
    return np.asarray([str(f) in fam_set for f in family_ids], dtype=bool)


def replace_families_with_baseline(
    states: torch.Tensor,
    baseline: torch.Tensor,
    family_ids: np.ndarray,
    families: Sequence[str],
) -> torch.Tensor:
    """Replace all features belonging to selected families across all sequence positions."""
    out = states.clone()
    mask = _family_feature_mask(family_ids, families)
    if not np.any(mask):
        return out
    # states/baseline: [B, T, F]
    mask_t = torch.as_tensor(mask, device=states.device)
    out[..., mask_t] = baseline[..., mask_t]
    return out


def restore_families_from_original(
    baseline_states: torch.Tensor,
    original_states: torch.Tensor,
    family_ids: np.ndarray,
    families: Sequence[str],
) -> torch.Tensor:
    out = baseline_states.clone()
    mask = _family_feature_mask(family_ids, families)
    if not np.any(mask):
        return out
    mask_t = torch.as_tensor(mask, device=original_states.device)
    out[..., mask_t] = original_states[..., mask_t]
    return out


def auc_trapz(depths: Sequence[int], values: Sequence[float]) -> float:
    if len(depths) == 0:
        return float("nan")
    if len(depths) == 1:
        return float(values[0])
    return float(np.trapz(values, depths))


def paired_bootstrap_ci(
    paired_gaps: np.ndarray,
    *,
    replicates: int,
    seed: int,
    alpha: float = 0.05,
) -> Tuple[float, float, float, float]:
    """Return mean, ci_low, ci_high, standardized paired effect size (mean/sd)."""
    x = np.asarray(paired_gaps, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if len(x) > 1 else float("nan")
    effect = float(mean / sd) if np.isfinite(sd) and sd > 0 else float("nan")
    if len(x) == 1 or replicates <= 1:
        return mean, mean, mean, effect
    rng = np.random.default_rng(seed)
    samples = rng.choice(x, size=(replicates, len(x)), replace=True).mean(axis=1)
    return mean, float(np.quantile(samples, alpha / 2)), float(np.quantile(samples, 1 - alpha / 2)), effect


def classify_faithfulness(gap: float, ci_low: float, effect: float) -> str:
    if not np.isfinite(gap):
        return "INCOMPLETE"
    if gap > 0 and np.isfinite(ci_low) and ci_low > 0 and np.isfinite(effect) and effect >= FAITHFULNESS_EFFECT_SIZE:
        return "SUPPORTED"
    if gap > 0:
        return "MIXED"
    return "UNSUPPORTED"


def _action_metrics(
    original: np.ndarray,
    perturbed: np.ndarray,
    cash_idx: int,
) -> Dict[str, np.ndarray]:
    dist, _ = allocation_distance(original, perturbed)
    cash_d, risky_d = cash_and_risky_deltas(original, perturbed, cash_idx)
    # cosine similarity per row
    o = original / np.maximum(np.linalg.norm(original, axis=-1, keepdims=True), 1e-12)
    p = perturbed / np.maximum(np.linalg.norm(perturbed, axis=-1, keepdims=True), 1e-12)
    cos = (o * p).sum(axis=-1)
    largest = np.max(np.abs(original - perturbed), axis=-1)
    return {
        "tv": dist,
        "cash_abs": cash_d,
        "risky_delta": risky_d,
        "cosine": cos,
        "largest_asset": largest,
    }


def run_deletion_insertion(
    adapter: DecisionTransformerAdapter,
    batch_tensors: Mapping[str, torch.Tensor],
    pool_states: torch.Tensor,
    family_ids: np.ndarray,
    abs_profile: pd.Series,
    *,
    model_family: str,
    max_depth: int,
    random_control_repeats: int,
    bootstrap_replicates: int,
    random_seed: int,
    cash_idx: int,
    inference_batch_size: int = 32,
) -> Dict[str, Any]:
    ranked = ranking_from_abs_profile(abs_profile)
    n_fam = len(ranked)
    depths = list(range(1, min(max_depth, n_fam) + 1))
    if not depths:
        empty = pd.DataFrame()
        return {
            "deletion_status": "INCOMPLETE",
            "insertion_status": "INCOMPLETE",
            "deletion_per_family": empty,
            "deletion_curves": empty,
            "deletion_random": empty,
            "insertion_per_family": empty,
            "insertion_curves": empty,
            "insertion_random": empty,
            "action_effects": empty,
            "metrics": {},
        }

    median_baseline = build_empirical_median_baseline(pool_states, batch_tensors["states"])
    original_actions = adapter.forward_actions(batch_tensors).detach().cpu().numpy()
    n_obs = original_actions.shape[0]
    rng = np.random.default_rng(int(random_seed))

    # --- Deletion ---
    del_curve_rows = []
    del_rand_rows = []
    guided_del_auc_per_obs = np.zeros(n_obs, dtype=float)
    random_del_auc_per_obs = np.zeros(n_obs, dtype=float)

    guided_del_by_depth: Dict[int, np.ndarray] = {}
    mean_rand_by_depth: Dict[int, np.ndarray] = {}
    for depth in depths:
        families = ranked[:depth]
        states_del = replace_families_with_baseline(
            batch_tensors["states"], median_baseline, family_ids, families
        )
        sub = dict(batch_tensors)
        sub["states"] = states_del
        actions = adapter.forward_actions(sub).detach().cpu().numpy()
        metrics = _action_metrics(original_actions, actions, cash_idx)
        guided_del_by_depth[depth] = metrics["tv"]
        del_curve_rows.append(
            {
                "model_family": model_family,
                "curve": "guided_deletion",
                "depth": depth,
                "mean_tv": float(np.nanmean(metrics["tv"])),
                "median_tv": float(np.nanmedian(metrics["tv"])),
                "median_cash_abs": float(np.nanmedian(metrics["cash_abs"])),
                "median_risky_delta": float(np.nanmedian(metrics["risky_delta"])),
                "median_cosine": float(np.nanmedian(metrics["cosine"])),
                "families": "|".join(families),
            }
        )

        rand_tvs = []
        for rep in range(random_control_repeats):
            pool = [f for f in ranked if f not in families]
            if len(pool) < depth:
                pool = ranked[:]
            choice = list(rng.choice(pool, size=depth, replace=False))
            states_r = replace_families_with_baseline(
                batch_tensors["states"], median_baseline, family_ids, choice
            )
            sub_r = dict(batch_tensors)
            sub_r["states"] = states_r
            act_r = adapter.forward_actions(sub_r).detach().cpu().numpy()
            m_r = _action_metrics(original_actions, act_r, cash_idx)
            rand_tvs.append(m_r["tv"])
            del_rand_rows.append(
                {
                    "model_family": model_family,
                    "depth": depth,
                    "repeat": rep,
                    "mean_tv": float(np.nanmean(m_r["tv"])),
                    "families": "|".join(choice),
                }
            )
        mean_rand_tv = np.stack(rand_tvs, axis=0).mean(axis=0)
        mean_rand_by_depth[depth] = mean_rand_tv
        del_curve_rows.append(
            {
                "model_family": model_family,
                "curve": "random_deletion_mean",
                "depth": depth,
                "mean_tv": float(np.nanmean(mean_rand_tv)),
                "median_tv": float(np.nanmedian(mean_rand_tv)),
                "median_cash_abs": float("nan"),
                "median_risky_delta": float("nan"),
                "median_cosine": float("nan"),
                "families": "",
            }
        )

    for obs in range(n_obs):
        g_vals = [float(guided_del_by_depth[d][obs]) for d in depths]
        r_vals = [float(mean_rand_by_depth[d][obs]) for d in depths]
        guided_del_auc_per_obs[obs] = auc_trapz(depths, g_vals)
        random_del_auc_per_obs[obs] = auc_trapz(depths, r_vals)

    deletion_gaps = guided_del_auc_per_obs - random_del_auc_per_obs
    del_mean, del_lo, del_hi, del_eff = paired_bootstrap_ci(
        deletion_gaps, replicates=bootstrap_replicates, seed=random_seed + 11
    )
    deletion_status = classify_faithfulness(del_mean, del_lo, del_eff)

    # --- Insertion ---
    ins_curve_rows = []
    ins_rand_rows = []
    baseline_only = dict(batch_tensors)
    baseline_only["states"] = median_baseline
    # distance to original when fully baseline
    base_actions = adapter.forward_actions(baseline_only).detach().cpu().numpy()

    guided_ins_by_depth: Dict[int, np.ndarray] = {}
    for depth in depths:
        families = ranked[:depth]
        states_ins = restore_families_from_original(
            median_baseline, batch_tensors["states"], family_ids, families
        )
        sub = dict(batch_tensors)
        sub["states"] = states_ins
        actions = adapter.forward_actions(sub).detach().cpu().numpy()
        metrics = _action_metrics(original_actions, actions, cash_idx)
        guided_ins_by_depth[depth] = metrics["tv"]
        ins_curve_rows.append(
            {
                "model_family": model_family,
                "curve": "guided_insertion",
                "depth": depth,
                "mean_tv": float(np.nanmean(metrics["tv"])),
                "median_tv": float(np.nanmedian(metrics["tv"])),
                "median_cash_abs": float(np.nanmedian(metrics["cash_abs"])),
                "median_risky_delta": float(np.nanmedian(metrics["risky_delta"])),
                "median_cosine": float(np.nanmedian(metrics["cosine"])),
                "families": "|".join(families),
            }
        )
        for rep in range(random_control_repeats):
            pool = [f for f in ranked if f not in families]
            if len(pool) < depth:
                pool = ranked[:]
            choice = list(rng.choice(pool, size=depth, replace=False))
            states_r = restore_families_from_original(
                median_baseline, batch_tensors["states"], family_ids, choice
            )
            sub_r = dict(batch_tensors)
            sub_r["states"] = states_r
            act_r = adapter.forward_actions(sub_r).detach().cpu().numpy()
            m_r = _action_metrics(original_actions, act_r, cash_idx)
            ins_rand_rows.append(
                {
                    "model_family": model_family,
                    "depth": depth,
                    "repeat": rep,
                    "mean_tv": float(np.nanmean(m_r["tv"])),
                    "families": "|".join(choice),
                    "tv_vector": m_r["tv"],
                }
            )

    mean_rand_ins_by_depth: Dict[int, np.ndarray] = {}
    for depth in depths:
        vectors = [row["tv_vector"] for row in ins_rand_rows if row["depth"] == depth]
        mean_rand_ins_by_depth[depth] = np.mean(np.stack(vectors, axis=0), axis=0)
        ins_curve_rows.append(
            {
                "model_family": model_family,
                "curve": "random_insertion_mean",
                "depth": depth,
                "mean_tv": float(np.nanmean(mean_rand_ins_by_depth[depth])),
                "median_tv": float(np.nanmedian(mean_rand_ins_by_depth[depth])),
                "median_cash_abs": float("nan"),
                "median_risky_delta": float("nan"),
                "median_cosine": float("nan"),
                "families": "",
            }
        )

    guided_ins_auc = np.zeros(n_obs)
    random_ins_auc = np.zeros(n_obs)
    for obs in range(n_obs):
        g_vals = [float(guided_ins_by_depth[d][obs]) for d in depths]
        r_vals = [float(mean_rand_ins_by_depth[d][obs]) for d in depths]
        guided_ins_auc[obs] = auc_trapz(depths, g_vals)
        random_ins_auc[obs] = auc_trapz(depths, r_vals)

    # insertion_gap = mean_random_distance_auc - guided_distance_auc
    insertion_gaps = random_ins_auc - guided_ins_auc
    ins_mean, ins_lo, ins_hi, ins_eff = paired_bootstrap_ci(
        insertion_gaps, replicates=bootstrap_replicates, seed=random_seed + 22
    )
    insertion_status = classify_faithfulness(ins_mean, ins_lo, ins_eff)

    # Strip tv_vector before exporting random controls
    ins_rand_export = [
        {k: v for k, v in row.items() if k != "tv_vector"} for row in ins_rand_rows
    ]

    # Portfolio action effects (guided deletion depth=min(3,max))
    focus_depth = min(3, max(depths))
    families_focus = ranked[:focus_depth]
    states_focus = replace_families_with_baseline(
        batch_tensors["states"], median_baseline, family_ids, families_focus
    )
    sub_f = dict(batch_tensors)
    sub_f["states"] = states_focus
    act_f = adapter.forward_actions(sub_f).detach().cpu().numpy()
    m_f = _action_metrics(original_actions, act_f, cash_idx)

    action_effects = pd.DataFrame(
        [
            {
                "model_family": model_family,
                "focus_depth": focus_depth,
                "median_abs_cash_shift": float(np.nanmedian(m_f["cash_abs"])),
                "median_risky_exposure_change": float(np.nanmedian(m_f["risky_delta"])),
                "median_largest_asset_weight_change": float(np.nanmedian(m_f["largest_asset"])),
                "median_allocation_tv": float(np.nanmedian(m_f["tv"])),
                "baseline_only_median_tv": float(
                    np.nanmedian(allocation_distance(original_actions, base_actions)[0])
                ),
            }
        ]
    )

    metrics = {
        "deletion_gap": del_mean,
        "deletion_ci_lower": del_lo,
        "deletion_ci_upper": del_hi,
        "deletion_effect_size": del_eff,
        "insertion_gap": ins_mean,
        "insertion_ci_lower": ins_lo,
        "insertion_ci_upper": ins_hi,
        "insertion_effect_size": ins_eff,
        "ranked_families": ranked,
    }

    summary = pd.DataFrame(
        [
            {
                "model_family": model_family,
                "deletion_status": deletion_status,
                "insertion_status": insertion_status,
                **{k: v for k, v in metrics.items() if k != "ranked_families"},
            }
        ]
    )

    return {
        "deletion_status": deletion_status,
        "insertion_status": insertion_status,
        "deletion_per_family": summary,
        "deletion_curves": pd.DataFrame(del_curve_rows),
        "deletion_random": pd.DataFrame(del_rand_rows),
        "insertion_per_family": summary.copy(),
        "insertion_curves": pd.DataFrame(ins_curve_rows),
        "insertion_random": pd.DataFrame(ins_rand_export),
        "action_effects": action_effects,
        "metrics": metrics,
    }
