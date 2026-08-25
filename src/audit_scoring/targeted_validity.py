"""Multi-target IG, randomization, and orchestration helpers for Batch C2."""

from __future__ import annotations

import gc
import hashlib
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import kendalltau, spearmanr

from .baseline_validity import (
    build_empirical_median_baseline,
    build_sampled_background_states,
    build_zero_baseline,
    classify_baseline_status,
    pairwise_concordance,
    ranking_from_abs_profile,
    signed_agreement,
    top_k_jaccard,
)
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
from .faithfulness_validity import (
    auc_trapz,
    classify_faithfulness,
    paired_bootstrap_ci,
    replace_families_with_baseline,
    restore_families_from_original,
    _action_metrics,
)
from .holdout_assessment import sha256_file


TARGET_L2_EQUAL = "l2_equal"
TARGET_CASH = "cash"
TARGET_VECTOR_L1 = "vector_output_l1"
TARGET_LARGEST_RISKY = "largest_risky"
TARGET_SUM_WEIGHTS = "sum_weights_negative_control"


def config_hash(payload: Mapping[str, Any]) -> str:
    import json

    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def selection_list_hash(families: Sequence[str]) -> str:
    text = "\n".join(sorted(map(str, families)))
    return hashlib.sha256(text.encode()).hexdigest()


def _prepare_rtg(rtg: torch.Tensor) -> torch.Tensor:
    return rtg.unsqueeze(-1) if rtg.ndim == 2 else rtg


def compute_ig_with_target(
    adapter: DecisionTransformerAdapter,
    batch: Mapping[str, torch.Tensor],
    baseline: torch.Tensor,
    *,
    target_name: str,
    n_steps: int,
    cash_idx: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Integrated Gradients under a named allocation functional."""
    from captum.attr import IntegratedGradients

    states = batch["states"].detach().clone().requires_grad_(True)
    actions = batch["actions"]
    rtg = _prepare_rtg(batch["returns_to_go"])
    timesteps = batch["timesteps"]
    attention_mask = batch.get("attention_mask")
    baseline = baseline.to(states.device).detach()
    if baseline.shape != states.shape:
        if baseline.shape[0] == 1:
            baseline = baseline.expand_as(states)
        else:
            raise ValueError("baseline shape mismatch")

    diagnostics: Dict[str, float] = {}

    def preds_fn(input_states: torch.Tensor) -> torch.Tensor:
        return adapter._forward_action(
            adapter.model,
            input_states,
            actions,
            rtg,
            timesteps,
            attention_mask=attention_mask,
            target_action_idx=None,
        )

    with torch.no_grad():
        ref = preds_fn(states.detach())
        equal = 1.0 / float(ref.shape[-1])
        diagnostics["target_variance_l2_equal"] = float(((ref - equal) ** 2).sum(-1).var().cpu())
        diagnostics["target_variance_cash"] = float(ref[:, cash_idx].var().cpu())
        diagnostics["target_variance_sum"] = float(ref.sum(-1).var().cpu())
        diagnostics["mean_action_sum"] = float(ref.sum(-1).mean().cpu())

    if target_name == TARGET_L2_EQUAL:

        def forward_wrapper(input_states: torch.Tensor) -> torch.Tensor:
            preds = preds_fn(input_states)
            equal = 1.0 / float(preds.shape[-1])
            return ((preds - equal) ** 2).sum(dim=-1)

        ig = IntegratedGradients(forward_wrapper)
        attr = ig.attribute(states, baselines=baseline, n_steps=n_steps)
    elif target_name == TARGET_CASH:

        def forward_wrapper(input_states: torch.Tensor) -> torch.Tensor:
            return preds_fn(input_states)[:, cash_idx]

        ig = IntegratedGradients(forward_wrapper)
        attr = ig.attribute(states, baselines=baseline, n_steps=n_steps)
    elif target_name == TARGET_SUM_WEIGHTS:

        def forward_wrapper(input_states: torch.Tensor) -> torch.Tensor:
            return preds_fn(input_states).sum(dim=-1)

        ig = IntegratedGradients(forward_wrapper)
        attr = ig.attribute(states, baselines=baseline, n_steps=n_steps)
    elif target_name == TARGET_VECTOR_L1:
        act_dim = int(actions.shape[-1])
        accumulated = None
        for t in range(act_dim):

            def forward_wrapper(input_states: torch.Tensor, target=t) -> torch.Tensor:
                return preds_fn(input_states)[:, target]

            ig = IntegratedGradients(forward_wrapper)
            a = ig.attribute(states, baselines=baseline, n_steps=n_steps)
            accumulated = a.abs() if accumulated is None else accumulated + a.abs()
        attr = accumulated
    elif target_name == TARGET_LARGEST_RISKY:
        with torch.no_grad():
            ref = preds_fn(states.detach())
            risky = ref.clone()
            risky[:, cash_idx] = -1e9
            top_idx = risky.argmax(dim=-1)  # [B]

        def forward_wrapper(input_states: torch.Tensor) -> torch.Tensor:
            preds = preds_fn(input_states)
            # gather observation-specific largest risky component using frozen indices
            return preds.gather(1, top_idx.view(-1, 1).to(preds.device)).squeeze(1)

        ig = IntegratedGradients(forward_wrapper)
        attr = ig.attribute(states, baselines=baseline, n_steps=n_steps)
    else:
        raise ValueError(f"Unknown target {target_name}")

    diagnostics["attribution_norm"] = float(attr.detach().abs().mean().cpu())
    return attr.detach(), diagnostics


def attribution_matrix_from_tensor(attr: torch.Tensor) -> np.ndarray:
    mat = attr.detach().cpu().numpy()
    if mat.ndim == 3:
        mat = mat.sum(axis=1)
    return mat


def profiles_from_attr(matrix: np.ndarray, family_ids: np.ndarray) -> Tuple[pd.Series, pd.Series]:
    return (
        aggregate_family_profile(matrix, family_ids, signed=False),
        aggregate_family_profile(matrix, family_ids, signed=True),
    )


def raw_feature_ranks(matrix: np.ndarray, feature_names: np.ndarray) -> pd.DataFrame:
    values = np.nanmean(np.abs(matrix), axis=0)
    signed = np.nanmean(matrix, axis=0)
    order = np.argsort(-values)
    rows = []
    for rank, idx in enumerate(order, start=1):
        rows.append(
            {
                "feature_name": str(feature_names[idx]),
                "raw_rank": rank,
                "abs_attribution": float(values[idx]),
                "signed_attribution": float(signed[idx]),
            }
        )
    return pd.DataFrame(rows)


def compare_rank_maps(
    abs_a: pd.Series,
    abs_b: pd.Series,
    signed_a: pd.Series,
    signed_b: pd.Series,
    top_k: int,
) -> Dict[str, float]:
    labels = sorted(set(abs_a.index.astype(str)) | set(abs_b.index.astype(str)))
    xa = abs_a.reindex(labels).fillna(0.0).to_numpy(dtype=float)
    xb = abs_b.reindex(labels).fillna(0.0).to_numpy(dtype=float)
    tau = kendalltau(xa, xb, variant="b").statistic
    spear = spearmanr(xa, xb).correlation
    ra = ranking_from_abs_profile(abs_a.reindex(labels).fillna(0.0))
    rb = ranking_from_abs_profile(abs_b.reindex(labels).fillna(0.0))
    k = min(top_k, len(labels))
    return {
        "kendall_tau_b": float(tau) if np.isfinite(tau) else float("nan"),
        "spearman": float(spear) if np.isfinite(spear) else float("nan"),
        "top_k_jaccard": top_k_jaccard(ra, rb, k),
        "top_k_sign_agreement": signed_agreement(
            signed_a.reindex(labels).fillna(0.0),
            signed_b.reindex(labels).fillna(0.0),
            ra[:k],
        ),
    }


def run_cardinality_matched_deletion_insertion(
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
    do_insertion: bool = True,
) -> Dict[str, Any]:
    ranked = ranking_from_abs_profile(abs_profile)
    card = family_raw_cardinality(family_ids)
    depths = list(range(1, min(max_depth, len(ranked)) + 1))
    if not depths:
        empty = pd.DataFrame()
        return {
            "deletion_status": "INCOMPLETE",
            "insertion_status": "INCOMPLETE",
            "deletion": empty,
            "insertion": empty,
            "controls": empty,
        }

    median_baseline = build_empirical_median_baseline(pool_states, batch_tensors["states"])
    original_actions = adapter.forward_actions(batch_tensors).detach().cpu().numpy()
    n_obs = original_actions.shape[0]
    rng = np.random.default_rng(int(random_seed))

    guided_del: Dict[int, np.ndarray] = {}
    mean_rand_del: Dict[int, np.ndarray] = {}
    control_rows: List[Dict[str, Any]] = []
    del_rows: List[Dict[str, Any]] = []

    for depth in depths:
        families = ranked[:depth]
        target_raw = guided_raw_count(family_ids, families)
        states_del = replace_families_with_baseline(
            batch_tensors["states"], median_baseline, family_ids, families
        )
        sub = dict(batch_tensors)
        sub["states"] = states_del
        actions = adapter.forward_actions(sub).detach().cpu().numpy()
        metrics = _action_metrics(original_actions, actions, cash_idx)
        guided_del[depth] = metrics["tv"]
        del_rows.append(
            {
                "model_family": model_family,
                "curve": "guided_deletion",
                "depth": depth,
                "n_families": depth,
                "raw_feature_count": target_raw,
                "mean_tv": float(np.nanmean(metrics["tv"])),
                "median_tv": float(np.nanmedian(metrics["tv"])),
                "median_cash_abs": float(np.nanmedian(metrics["cash_abs"])),
                "median_risky_delta": float(np.nanmedian(metrics["risky_delta"])),
                "families": "|".join(families),
            }
        )

        rand_tvs = []
        for rep in range(random_control_repeats):
            choice, raw_c, residual = sample_cardinality_matched_families(
                ranked,
                card,
                guided_families=families,
                n_families=depth,
                target_raw_count=target_raw,
                rng=rng,
                exclude_guided=True,
            )
            states_r = replace_families_with_baseline(
                batch_tensors["states"], median_baseline, family_ids, choice
            )
            sub_r = dict(batch_tensors)
            sub_r["states"] = states_r
            act_r = adapter.forward_actions(sub_r).detach().cpu().numpy()
            m_r = _action_metrics(original_actions, act_r, cash_idx)
            rand_tvs.append(m_r["tv"])
            control_rows.append(
                {
                    "model_family": model_family,
                    "task": "deletion",
                    "depth": depth,
                    "repeat": rep,
                    "guided_raw_count": target_raw,
                    "control_raw_count": raw_c,
                    "cardinality_residual": residual,
                    "n_families": depth,
                    "mean_tv": float(np.nanmean(m_r["tv"])),
                    "families": "|".join(choice),
                }
            )
        mean_rand_del[depth] = np.mean(np.stack(rand_tvs, axis=0), axis=0)
        del_rows.append(
            {
                "model_family": model_family,
                "curve": "cardinality_matched_random_deletion",
                "depth": depth,
                "n_families": depth,
                "raw_feature_count": target_raw,
                "mean_tv": float(np.nanmean(mean_rand_del[depth])),
                "median_tv": float(np.nanmedian(mean_rand_del[depth])),
                "median_cash_abs": float("nan"),
                "median_risky_delta": float("nan"),
                "families": "",
            }
        )

    guided_auc = np.zeros(n_obs)
    rand_auc = np.zeros(n_obs)
    for i in range(n_obs):
        guided_auc[i] = auc_trapz(depths, [float(guided_del[d][i]) for d in depths])
        rand_auc[i] = auc_trapz(depths, [float(mean_rand_del[d][i]) for d in depths])
    gaps = guided_auc - rand_auc
    gmean, glo, ghi, geff = paired_bootstrap_ci(gaps, replicates=bootstrap_replicates, seed=random_seed + 7)
    deletion_status = classify_faithfulness(gmean, glo, geff)

    insertion_status = "INCOMPLETE"
    ins_rows: List[Dict[str, Any]] = []
    imean = ilo = ihi = ieff = float("nan")
    if do_insertion:
        guided_ins: Dict[int, np.ndarray] = {}
        mean_rand_ins: Dict[int, np.ndarray] = {}
        for depth in depths:
            families = ranked[:depth]
            target_raw = guided_raw_count(family_ids, families)
            states_ins = restore_families_from_original(
                median_baseline, batch_tensors["states"], family_ids, families
            )
            sub = dict(batch_tensors)
            sub["states"] = states_ins
            actions = adapter.forward_actions(sub).detach().cpu().numpy()
            metrics = _action_metrics(original_actions, actions, cash_idx)
            guided_ins[depth] = metrics["tv"]
            ins_rows.append(
                {
                    "model_family": model_family,
                    "curve": "guided_insertion",
                    "depth": depth,
                    "n_families": depth,
                    "raw_feature_count": target_raw,
                    "mean_tv": float(np.nanmean(metrics["tv"])),
                    "median_tv": float(np.nanmedian(metrics["tv"])),
                    "median_cash_abs": float(np.nanmedian(metrics["cash_abs"])),
                    "median_risky_delta": float(np.nanmedian(metrics["risky_delta"])),
                    "families": "|".join(families),
                }
            )
            rand_tvs = []
            for rep in range(random_control_repeats):
                choice, raw_c, residual = sample_cardinality_matched_families(
                    ranked,
                    card,
                    guided_families=families,
                    n_families=depth,
                    target_raw_count=target_raw,
                    rng=rng,
                    exclude_guided=True,
                )
                states_r = restore_families_from_original(
                    median_baseline, batch_tensors["states"], family_ids, choice
                )
                sub_r = dict(batch_tensors)
                sub_r["states"] = states_r
                act_r = adapter.forward_actions(sub_r).detach().cpu().numpy()
                m_r = _action_metrics(original_actions, act_r, cash_idx)
                rand_tvs.append(m_r["tv"])
                control_rows.append(
                    {
                        "model_family": model_family,
                        "task": "insertion",
                        "depth": depth,
                        "repeat": rep,
                        "guided_raw_count": target_raw,
                        "control_raw_count": raw_c,
                        "cardinality_residual": residual,
                        "n_families": depth,
                        "mean_tv": float(np.nanmean(m_r["tv"])),
                        "families": "|".join(choice),
                    }
                )
            mean_rand_ins[depth] = np.mean(np.stack(rand_tvs, axis=0), axis=0)
            ins_rows.append(
                {
                    "model_family": model_family,
                    "curve": "cardinality_matched_random_insertion",
                    "depth": depth,
                    "n_families": depth,
                    "raw_feature_count": target_raw,
                    "mean_tv": float(np.nanmean(mean_rand_ins[depth])),
                    "median_tv": float(np.nanmedian(mean_rand_ins[depth])),
                    "median_cash_abs": float("nan"),
                    "median_risky_delta": float("nan"),
                    "families": "",
                }
            )
        g_ins = np.zeros(n_obs)
        r_ins = np.zeros(n_obs)
        for i in range(n_obs):
            g_ins[i] = auc_trapz(depths, [float(guided_ins[d][i]) for d in depths])
            r_ins[i] = auc_trapz(depths, [float(mean_rand_ins[d][i]) for d in depths])
        igaps = r_ins - g_ins
        imean, ilo, ihi, ieff = paired_bootstrap_ci(
            igaps, replicates=bootstrap_replicates, seed=random_seed + 9
        )
        insertion_status = classify_faithfulness(imean, ilo, ieff)

    summary = pd.DataFrame(
        [
            {
                "model_family": model_family,
                "deletion_status": deletion_status,
                "deletion_gap": gmean,
                "deletion_ci_lower": glo,
                "deletion_ci_upper": ghi,
                "deletion_effect_size": geff,
                "insertion_status": insertion_status,
                "insertion_gap": imean,
                "insertion_ci_lower": ilo,
                "insertion_ci_upper": ihi,
                "insertion_effect_size": ieff,
                "control_type": "cardinality_matched",
                "random_control_repeats": random_control_repeats,
            }
        ]
    )
    return {
        "deletion_status": deletion_status,
        "insertion_status": insertion_status,
        "deletion": pd.DataFrame(del_rows),
        "insertion": pd.DataFrame(ins_rows),
        "controls": pd.DataFrame(control_rows),
        "summary": summary,
        "metrics": {
            "deletion_gap": gmean,
            "deletion_ci_lower": glo,
            "insertion_gap": imean,
            "insertion_ci_lower": ilo,
        },
    }


def parameter_change_stats(before: nn.Module, after: nn.Module) -> Dict[str, float]:
    n_changed = 0
    n_total = 0
    norm_before = 0.0
    norm_after = 0.0
    delta_norm = 0.0
    for (nb, pb), (na, pa) in zip(before.named_parameters(), after.named_parameters()):
        if pb.requires_grad:
            n_total += pb.numel()
            b = pb.detach().float().cpu().reshape(-1)
            a = pa.detach().float().cpu().reshape(-1)
            n_changed += int((b != a).sum().item())
            norm_before += float(torch.norm(b).item() ** 2)
            norm_after += float(torch.norm(a).item() ** 2)
            delta_norm += float(torch.norm(a - b).item() ** 2)
    return {
        "fraction_parameters_changed": float(n_changed / max(n_total, 1)),
        "parameter_norm_before": float(np.sqrt(norm_before)),
        "parameter_norm_after": float(np.sqrt(norm_after)),
        "parameter_norm_change": float(np.sqrt(delta_norm)),
    }
