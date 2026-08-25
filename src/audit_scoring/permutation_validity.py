"""Core Batch C3 permutation validity calculations."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau, spearmanr

from .baseline_validity import ranking_from_abs_profile, top_k_jaccard
from .cardinality_controls import (
    family_raw_cardinality,
    guided_raw_count,
    sample_cardinality_matched_families,
)
from .checkpoint_loading import (
    DecisionTransformerAdapter,
    aggregate_family_profile,
    allocation_distance,
    cash_and_risky_deltas,
)
from .faithfulness_validity import auc_trapz, classify_faithfulness, paired_bootstrap_ci
from .permutation_config import IE_SENSITIVITY, IE_SUPPORTED_SPEARMAN, IE_SUPPORTED_TOPK, PERM_EFFECT_SIZE
from .permutation_operator import (
    permute_all_state_families,
    permute_state_families,
    valid_permutation_mask,
)
from .targeted_validity import (
    TARGET_CASH,
    TARGET_L2_EQUAL,
    TARGET_VECTOR_L1,
    attribution_matrix_from_tensor,
    compare_rank_maps,
    compute_ig_with_target,
    profiles_from_attr,
)


def _l2_equal(actions: np.ndarray) -> np.ndarray:
    equal = 1.0 / actions.shape[-1]
    return ((actions - equal) ** 2).sum(axis=-1)


def _action_effects(a0: np.ndarray, a1: np.ndarray, cash_idx: int) -> Dict[str, np.ndarray]:
    tv, _ = allocation_distance(a0, a1)
    cash, risky = cash_and_risky_deltas(a0, a1, cash_idx)
    largest = np.max(np.abs(a0 - a1), axis=-1)
    o = a0 / np.maximum(np.linalg.norm(a0, axis=-1, keepdims=True), 1e-12)
    p = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-12)
    cos = (o * p).sum(axis=-1)
    tgt = np.abs(_l2_equal(a1) - _l2_equal(a0))
    return {
        "tv": tv,
        "cash_abs": cash,
        "risky_delta": risky,
        "largest_asset": largest,
        "cosine": cos,
        "target_change": tgt,
    }


def _ig_matrix(
    adapter: DecisionTransformerAdapter,
    batch: Mapping[str, torch.Tensor],
    baseline: torch.Tensor,
    *,
    target_name: str,
    n_steps: int,
    cash_idx: int,
    batch_size: int,
) -> np.ndarray:
    n = batch["states"].shape[0]
    mats = []
    for start in range(0, n, batch_size):
        sl = slice(start, min(n, start + batch_size))
        sub = {k: v[sl] for k, v in batch.items() if torch.is_tensor(v)}
        base = baseline[sl] if baseline.shape[0] == n else baseline
        attr, _ = compute_ig_with_target(
            adapter, sub, base, target_name=target_name, n_steps=n_steps, cash_idx=cash_idx
        )
        mats.append(attribution_matrix_from_tensor(attr))
    return np.concatenate(mats, axis=0)


def single_family_permutation_effects(
    adapter: DecisionTransformerAdapter,
    batch: Mapping[str, torch.Tensor],
    donor_indices: np.ndarray,
    family_ids: np.ndarray,
    feature_families: Sequence[str],
    original_abs: pd.Series,
    original_signed: pd.Series,
    median_baseline: torch.Tensor,
    *,
    model_family: str,
    seed: int,
    cash_idx: int,
    n_steps: int,
    batch_size: int,
    recompute_attribution: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    a0 = adapter.forward_actions(batch).detach().cpu().numpy()
    mask = valid_permutation_mask(donor_indices)
    effect_rows = []
    attr_rows = []
    for fam in feature_families:
        states_p = permute_state_families(batch["states"], donor_indices, family_ids, [fam])
        sub = dict(batch)
        sub["states"] = states_p
        a1 = adapter.forward_actions(sub).detach().cpu().numpy()
        if np.any(mask):
            eff = _action_effects(a0[mask], a1[mask], cash_idx)
            effect_rows.append(
                {
                    "model_family": model_family,
                    "seed": int(seed),
                    "feature_family": fam,
                    "n_valid": int(mask.sum()),
                    "median_tv": float(np.nanmedian(eff["tv"])),
                    "iqr_tv": float(np.nanpercentile(eff["tv"], 75) - np.nanpercentile(eff["tv"], 25)),
                    "median_cash_abs": float(np.nanmedian(eff["cash_abs"])),
                    "median_risky_delta": float(np.nanmedian(eff["risky_delta"])),
                    "median_largest_asset": float(np.nanmedian(eff["largest_asset"])),
                    "median_cosine": float(np.nanmedian(eff["cosine"])),
                    "median_target_change": float(np.nanmedian(eff["target_change"])),
                    "original_ig_abs": float(original_abs.get(fam, 0.0)),
                }
            )
        else:
            effect_rows.append(
                {
                    "model_family": model_family,
                    "seed": int(seed),
                    "feature_family": fam,
                    "n_valid": 0,
                    "median_tv": float("nan"),
                    "original_ig_abs": float(original_abs.get(fam, 0.0)),
                    "notes": "no_valid_permutation",
                }
            )
        if recompute_attribution and np.any(mask):
            mat = _ig_matrix(
                adapter,
                sub,
                median_baseline,
                target_name=TARGET_L2_EQUAL,
                n_steps=n_steps,
                cash_idx=cash_idx,
                batch_size=batch_size,
            )
            abs_p, signed_p = profiles_from_attr(mat, family_ids)
            comps = compare_rank_maps(original_abs, abs_p, original_signed, signed_p, top_k=5)
            attr_rows.append(
                {
                    "model_family": model_family,
                    "seed": int(seed),
                    "feature_family": fam,
                    **comps,
                    "attribution_norm_change": float(
                        abs(float(np.nanmean(np.abs(mat))) - float(original_abs.abs().mean()))
                    ),
                }
            )
    return pd.DataFrame(effect_rows), pd.DataFrame(attr_rows)


def importance_effect_concordance(
    effects: pd.DataFrame,
    *,
    model_family: str,
    top_k: int = 5,
    bootstrap_replicates: int = 1000,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Aggregate across seeds: median TV per feature family vs original IG abs."""
    if effects.empty or effects["median_tv"].isna().all():
        return {
            "model_family": model_family,
            "importance_effect_status": "INCOMPLETE",
            "spearman": float("nan"),
            "kendall_tau_b": float("nan"),
            "topk_overlap": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
        }
    agg = (
        effects.groupby("feature_family", as_index=False)
        .agg(median_tv=("median_tv", "median"), original_ig_abs=("original_ig_abs", "first"))
        .dropna()
    )
    if len(agg) < 3:
        return {
            "model_family": model_family,
            "importance_effect_status": "INCOMPLETE",
            "spearman": float("nan"),
            "kendall_tau_b": float("nan"),
            "topk_overlap": float("nan"),
            "n_families": int(len(agg)),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
        }
    x = agg["original_ig_abs"].to_numpy(dtype=float)
    y = agg["median_tv"].to_numpy(dtype=float)
    spear = spearmanr(x, y).correlation
    tau = kendalltau(x, y, variant="b").statistic
    rank_ig = ranking_from_abs_profile(pd.Series(x, index=agg["feature_family"]))
    rank_perm = ranking_from_abs_profile(pd.Series(y, index=agg["feature_family"]))
    k = min(top_k, len(agg))
    jacc = top_k_jaccard(rank_ig, rank_perm, k)
    # Bootstrap over families (conservative surrogate for obs-level when seed-aggregated)
    rng = np.random.default_rng(random_seed)
    boots = []
    n = len(agg)
    for _ in range(max(200, min(bootstrap_replicates, 2000))):
        idx = rng.integers(0, n, size=n)
        s = spearmanr(x[idx], y[idx]).correlation
        if np.isfinite(s):
            boots.append(float(s))
    ci_lo = float(np.quantile(boots, 0.025)) if boots else float("nan")
    ci_hi = float(np.quantile(boots, 0.975)) if boots else float("nan")
    # permutation p-value
    null = []
    for _ in range(999):
        yp = rng.permutation(y)
        null.append(float(spearmanr(x, yp).correlation))
    pval = float((1 + sum(abs(n) >= abs(float(spear)) for n in null if np.isfinite(n))) / (len(null) + 1))

    spear_f = float(spear) if np.isfinite(spear) else float("nan")
    if spear_f >= IE_SUPPORTED_SPEARMAN and np.isfinite(ci_lo) and ci_lo > 0 and jacc >= IE_SUPPORTED_TOPK:
        status = "SUPPORTED"
    elif spear_f > 0:
        status = "MIXED"
    elif not np.isfinite(spear_f):
        status = "INCOMPLETE"
    else:
        status = "UNSUPPORTED"

    top_ig = rank_ig[0] if rank_ig else ""
    top_ig_rank_in_perm = (rank_perm.index(top_ig) + 1) if top_ig in rank_perm else -1

    return {
        "model_family": model_family,
        "importance_effect_status": status,
        "spearman": spear_f,
        "kendall_tau_b": float(tau) if np.isfinite(tau) else float("nan"),
        "topk_overlap": float(jacc),
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "permutation_pvalue": pval,
        "n_families": int(len(agg)),
        "top_ig_family": top_ig,
        "top_ig_rank_in_permutation_impact": int(top_ig_rank_in_perm),
    }


def importance_effect_sensitivity(row: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    spear = float(row.get("spearman", float("nan")))
    jacc = float(row.get("topk_overlap", float("nan")))
    ci_lo = float(row.get("ci_lower", float("nan")))
    for cut in IE_SENSITIVITY:
        if spear >= cut and np.isfinite(ci_lo) and ci_lo > 0 and jacc >= IE_SUPPORTED_TOPK:
            st = "SUPPORTED"
        elif spear > 0:
            st = "MIXED"
        elif not np.isfinite(spear):
            st = "INCOMPLETE"
        else:
            st = "UNSUPPORTED"
        rows.append(
            {
                "model_family": row.get("model_family"),
                "spearman_cutoff": cut,
                "status": st,
                "spearman": spear,
                "ci_lower": ci_lo,
                "topk_overlap": jacc,
            }
        )
    return pd.DataFrame(rows)


def guided_vs_random_permutation(
    adapter: DecisionTransformerAdapter,
    batch: Mapping[str, torch.Tensor],
    donor_by_seed: Mapping[int, np.ndarray],
    family_ids: np.ndarray,
    ranked_families: Sequence[str],
    *,
    model_family: str,
    max_depth: int,
    random_control_repeats: int,
    bootstrap_replicates: int,
    random_seed: int,
    cash_idx: int,
) -> Dict[str, Any]:
    card = family_raw_cardinality(family_ids)
    depths = list(range(1, min(max_depth, len(ranked_families)) + 1))
    if not depths:
        empty = pd.DataFrame()
        return {
            "guided_permutation_status": "INCOMPLETE",
            "curves": empty,
            "controls": empty,
            "matching": empty,
            "summary": empty,
            "metrics": {},
        }

    a0 = adapter.forward_actions(batch).detach().cpu().numpy()
    n_obs = a0.shape[0]
    seeds = list(donor_by_seed.keys())
    rng = np.random.default_rng(int(random_seed))

    guided_by_depth: Dict[int, List[np.ndarray]] = {d: [] for d in depths}
    random_by_depth: Dict[int, List[np.ndarray]] = {d: [] for d in depths}
    curve_rows = []
    control_rows = []
    match_rows = []

    for seed, donors in donor_by_seed.items():
        mask = valid_permutation_mask(donors)
        for depth in depths:
            families = list(ranked_families[:depth])
            target_raw = guided_raw_count(family_ids, families)
            states_g = permute_state_families(batch["states"], donors, family_ids, families)
            sub_g = dict(batch)
            sub_g["states"] = states_g
            a_g = adapter.forward_actions(sub_g).detach().cpu().numpy()
            if np.any(mask):
                tv_g = allocation_distance(a0[mask], a_g[mask])[0]
            else:
                tv_g = np.full(n_obs, np.nan)
            # store full-length with nan for invalid
            full_g = np.full(n_obs, np.nan)
            if np.any(mask):
                full_g[mask] = tv_g
            guided_by_depth[depth].append(full_g)

            curve_rows.append(
                {
                    "model_family": model_family,
                    "seed": int(seed),
                    "curve": "guided",
                    "depth": depth,
                    "mean_tv": float(np.nanmean(full_g)),
                    "median_tv": float(np.nanmedian(full_g)),
                    "families": "|".join(families),
                    "raw_feature_count": target_raw,
                }
            )

            rand_stack = []
            for rep in range(random_control_repeats):
                choice, raw_c, residual = sample_cardinality_matched_families(
                    ranked_families,
                    card,
                    guided_families=families,
                    n_families=depth,
                    target_raw_count=target_raw,
                    rng=rng,
                    exclude_guided=True,
                )
                if set(choice) == set(families):
                    # redraw once
                    choice, raw_c, residual = sample_cardinality_matched_families(
                        ranked_families,
                        card,
                        guided_families=families,
                        n_families=depth,
                        target_raw_count=target_raw,
                        rng=rng,
                        exclude_guided=True,
                    )
                states_r = permute_state_families(batch["states"], donors, family_ids, choice)
                sub_r = dict(batch)
                sub_r["states"] = states_r
                a_r = adapter.forward_actions(sub_r).detach().cpu().numpy()
                full_r = np.full(n_obs, np.nan)
                if np.any(mask):
                    full_r[mask] = allocation_distance(a0[mask], a_r[mask])[0]
                rand_stack.append(full_r)
                control_rows.append(
                    {
                        "model_family": model_family,
                        "seed": int(seed),
                        "depth": depth,
                        "repeat": rep,
                        "mean_tv": float(np.nanmean(full_r)),
                        "families": "|".join(choice),
                        "guided_raw_count": target_raw,
                        "control_raw_count": raw_c,
                        "cardinality_residual": residual,
                    }
                )
                match_rows.append(
                    {
                        "model_family": model_family,
                        "seed": int(seed),
                        "depth": depth,
                        "repeat": rep,
                        "guided_raw_count": target_raw,
                        "control_raw_count": raw_c,
                        "cardinality_residual": residual,
                        "n_families": depth,
                    }
                )
            mean_rand = np.nanmean(np.stack(rand_stack, axis=0), axis=0)
            random_by_depth[depth].append(mean_rand)
            curve_rows.append(
                {
                    "model_family": model_family,
                    "seed": int(seed),
                    "curve": "matched_random",
                    "depth": depth,
                    "mean_tv": float(np.nanmean(mean_rand)),
                    "median_tv": float(np.nanmedian(mean_rand)),
                    "families": "",
                    "raw_feature_count": target_raw,
                }
            )

    # Aggregate across seeds: mean per obs then AUC
    guided_auc = np.zeros(n_obs)
    random_auc = np.zeros(n_obs)
    for i in range(n_obs):
        g_vals = [float(np.nanmean([arr[i] for arr in guided_by_depth[d]])) for d in depths]
        r_vals = [float(np.nanmean([arr[i] for arr in random_by_depth[d]])) for d in depths]
        guided_auc[i] = auc_trapz(depths, g_vals)
        random_auc[i] = auc_trapz(depths, r_vals)
    gaps = guided_auc - random_auc
    finite = gaps[np.isfinite(gaps)]
    gmean, glo, ghi, geff = paired_bootstrap_ci(
        finite if len(finite) else gaps,
        replicates=bootstrap_replicates,
        seed=random_seed + 17,
    )
    status = classify_faithfulness(gmean, glo, geff)
    # Override effect size threshold name already in classify_faithfulness via FAITHFULNESS_EFFECT_SIZE=0.20

    # Action effects at focus depth
    focus = min(3, max(depths))
    cash_vals = []
    tv_vals = []
    for seed, donors in donor_by_seed.items():
        mask = valid_permutation_mask(donors)
        families = list(ranked_families[:focus])
        states_g = permute_state_families(batch["states"], donors, family_ids, families)
        sub = dict(batch)
        sub["states"] = states_g
        a_g = adapter.forward_actions(sub).detach().cpu().numpy()
        if np.any(mask):
            eff = _action_effects(a0[mask], a_g[mask], cash_idx)
            cash_vals.append(float(np.nanmedian(eff["cash_abs"])))
            tv_vals.append(float(np.nanmedian(eff["tv"])))

    summary = pd.DataFrame(
        [
            {
                "model_family": model_family,
                "guided_permutation_status": status,
                "guided_permutation_gap": gmean,
                "guided_permutation_ci_lower": glo,
                "guided_permutation_ci_upper": ghi,
                "guided_permutation_effect_size": geff,
                "median_guided_tv_focus": float(np.nanmedian(tv_vals)) if tv_vals else float("nan"),
                "median_guided_cash_focus": float(np.nanmedian(cash_vals)) if cash_vals else float("nan"),
                "n_seeds": len(seeds),
                "random_control_repeats": random_control_repeats,
            }
        ]
    )
    return {
        "guided_permutation_status": status,
        "curves": pd.DataFrame(curve_rows),
        "controls": pd.DataFrame(control_rows),
        "matching": pd.DataFrame(match_rows),
        "summary": summary,
        "metrics": {
            "guided_permutation_gap": gmean,
            "guided_permutation_ci_lower": glo,
            "guided_permutation_effect_size": geff,
        },
    }


def global_permutation_attribution(
    adapter: DecisionTransformerAdapter,
    batch: Mapping[str, torch.Tensor],
    donor_indices: np.ndarray,
    family_ids: np.ndarray,
    original_abs: pd.Series,
    original_signed: pd.Series,
    median_baseline: torch.Tensor,
    *,
    model_family: str,
    seed: int,
    cash_idx: int,
    n_steps: int,
    batch_size: int,
) -> Dict[str, Any]:
    a0 = adapter.forward_actions(batch).detach().cpu().numpy()
    states_p = permute_all_state_families(batch["states"], donor_indices)
    sub = dict(batch)
    sub["states"] = states_p
    a1 = adapter.forward_actions(sub).detach().cpu().numpy()
    mask = valid_permutation_mask(donor_indices)
    if not np.any(mask):
        return {
            "model_family": model_family,
            "seed": int(seed),
            "kendall_tau_b": float("nan"),
            "notes": "no_valid",
        }
    mat = _ig_matrix(
        adapter,
        sub,
        median_baseline,
        target_name=TARGET_L2_EQUAL,
        n_steps=n_steps,
        cash_idx=cash_idx,
        batch_size=batch_size,
    )
    abs_p, signed_p = profiles_from_attr(mat, family_ids)
    comps = compare_rank_maps(original_abs, abs_p, original_signed, signed_p, top_k=5)
    dist, _ = allocation_distance(a0[mask], a1[mask])
    return {
        "model_family": model_family,
        "seed": int(seed),
        **comps,
        "median_tv": float(np.nanmedian(dist)),
        "attribution_norm": float(np.nanmean(np.abs(mat))),
    }


def combined_permutation_status(
    *,
    importance_effect_status: str,
    guided_permutation_status: str,
    identity_pass: bool,
) -> Tuple[str, str]:
    if not identity_pass:
        return "UNSUPPORTED", "Identity control failed."
    if importance_effect_status == "INCOMPLETE" or guided_permutation_status == "INCOMPLETE":
        return "INCOMPLETE", "Required permutation evidence unavailable."
    if importance_effect_status == "UNSUPPORTED" or guided_permutation_status == "UNSUPPORTED":
        return "UNSUPPORTED", "Primary importance-effect or guided-permutation check unsupported."
    if importance_effect_status == "SUPPORTED" and guided_permutation_status == "SUPPORTED":
        return "SUPPORTED", "Importance-effect and guided-permutation supported; identity control passed."
    return "MIXED", "Mixed permutation evidence under primary operational rules."
