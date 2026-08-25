"""Alternative-baseline attribution concordance (Batch C Phase 3–4)."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau, spearmanr

from .checkpoint_loading import (
    DecisionTransformerAdapter,
    aggregate_family_profile,
    allocation_distance,
)
from .validity_config import (
    BASELINE_MIXED_MEDIAN_TAU,
    BASELINE_SENSITIVITY_TAUS,
    BASELINE_SUPPORTED_MEDIAN_TAU,
    BASELINE_SUPPORTED_MIN_JACCARD,
    BASELINE_SUPPORTED_SIGN_AGREE,
)


def _stable_hash_array(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(arr).tobytes())
    h.update(str(arr.shape).encode())
    return h.hexdigest()


def build_zero_baseline(states: torch.Tensor) -> torch.Tensor:
    """Exact zero baseline on the model-ready (already preprocessed) state tensor."""
    return torch.zeros_like(states)


def build_empirical_median_baseline(pool_states: torch.Tensor, target_states: torch.Tensor) -> torch.Tensor:
    """Per-feature median over validation pool, repeated across sequence length."""
    # pool_states: [N, T, F]
    med = pool_states.reshape(-1, pool_states.shape[-1]).median(dim=0).values
    baseline = med.view(1, 1, -1).expand_as(target_states).clone()
    return baseline


def build_sampled_background_states(
    pool_states: torch.Tensor,
    background_indices: Sequence[int],
) -> torch.Tensor:
    idx = torch.as_tensor(list(background_indices), device=pool_states.device, dtype=torch.long)
    return pool_states.index_select(0, idx).detach().clone()


def baselines_payload(
    *,
    zero: torch.Tensor,
    empirical_median: torch.Tensor,
    background_states: torch.Tensor,
    background_indices: Sequence[int],
) -> Dict[str, Any]:
    return {
        "zero_hash": _stable_hash_array(zero.detach().cpu().numpy()),
        "empirical_median_hash": _stable_hash_array(empirical_median.detach().cpu().numpy()),
        "background_indices": [int(i) for i in background_indices],
        "background_hash": _stable_hash_array(background_states.detach().cpu().numpy()),
        "definitions": {
            "zero": "torch.zeros_like(states) on preprocessed validation states",
            "empirical_median": "feature-wise median over validation pool, broadcast to sequence",
            "sampled_background": "regime-stratified validation background; IG averaged over backgrounds",
        },
    }


def top_k_jaccard(rank_a: Sequence[str], rank_b: Sequence[str], k: int) -> float:
    a = list(rank_a)[:k]
    b = list(rank_b)[:k]
    if k <= 0:
        return float("nan")
    inter = len(set(a) & set(b))
    union = len(set(a) | set(b))
    return float(inter / union) if union else float("nan")


def signed_agreement(signed_a: pd.Series, signed_b: pd.Series, top_names: Sequence[str]) -> float:
    if not len(top_names):
        return float("nan")
    agrees = []
    for name in top_names:
        if name not in signed_a.index or name not in signed_b.index:
            continue
        sa = float(signed_a.loc[name])
        sb = float(signed_b.loc[name])
        if sa == 0 or sb == 0:
            agrees.append(float(np.sign(sa) == np.sign(sb)))
        else:
            agrees.append(float(np.sign(sa) == np.sign(sb)))
    return float(np.mean(agrees)) if agrees else float("nan")


def ranking_from_abs_profile(profile: pd.Series) -> List[str]:
    return profile.abs().sort_values(ascending=False).index.astype(str).tolist()


def pairwise_concordance(
    abs_profiles: Mapping[str, pd.Series],
    signed_profiles: Mapping[str, pd.Series],
    top_k: int,
) -> pd.DataFrame:
    names = list(abs_profiles.keys())
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            labels = sorted(set(abs_profiles[a].index) | set(abs_profiles[b].index))
            xa = abs_profiles[a].reindex(labels).fillna(0.0).to_numpy(dtype=float)
            xb = abs_profiles[b].reindex(labels).fillna(0.0).to_numpy(dtype=float)
            tau = kendalltau(xa, xb, variant="b").statistic
            spear = spearmanr(xa, xb).correlation
            rank_a = ranking_from_abs_profile(abs_profiles[a].reindex(labels).fillna(0.0))
            rank_b = ranking_from_abs_profile(abs_profiles[b].reindex(labels).fillna(0.0))
            k = min(top_k, len(labels))
            jacc = top_k_jaccard(rank_a, rank_b, k)
            top = rank_a[:k]
            sign_agree = signed_agreement(
                signed_profiles[a].reindex(labels).fillna(0.0),
                signed_profiles[b].reindex(labels).fillna(0.0),
                top,
            )
            rows.append(
                {
                    "baseline_a": a,
                    "baseline_b": b,
                    "kendall_tau_b": float(tau) if np.isfinite(tau) else float("nan"),
                    "spearman": float(spear) if np.isfinite(spear) else float("nan"),
                    "top_k": int(k),
                    "top_k_jaccard": jacc,
                    "top_k_sign_agreement": sign_agree,
                }
            )
    return pd.DataFrame(rows)


def classify_baseline_status(pairwise: pd.DataFrame) -> Tuple[str, Dict[str, float]]:
    if pairwise.empty or pairwise["kendall_tau_b"].isna().all():
        return "INCOMPLETE", {}
    taus = pairwise["kendall_tau_b"].astype(float)
    jaccs = pairwise["top_k_jaccard"].astype(float)
    signs = pairwise["top_k_sign_agreement"].astype(float)
    median_tau = float(np.nanmedian(taus))
    min_tau = float(np.nanmin(taus))
    min_jaccard = float(np.nanmin(jaccs))
    mean_sign = float(np.nanmean(signs))
    metrics = {
        "median_kendall_tau_b": median_tau,
        "min_kendall_tau_b": min_tau,
        "min_top_k_jaccard": min_jaccard,
        "mean_top_k_sign_agreement": mean_sign,
    }
    if (
        median_tau >= BASELINE_SUPPORTED_MEDIAN_TAU
        and min_jaccard >= BASELINE_SUPPORTED_MIN_JACCARD
        and mean_sign >= BASELINE_SUPPORTED_SIGN_AGREE
    ):
        return "SUPPORTED", metrics
    if median_tau < BASELINE_MIXED_MEDIAN_TAU or min_tau < 0:
        return "UNSUPPORTED", metrics
    if median_tau >= BASELINE_MIXED_MEDIAN_TAU and min_tau >= 0:
        return "MIXED", metrics
    return "UNSUPPORTED", metrics


def compute_ig_family_profiles(
    adapter: DecisionTransformerAdapter,
    batch_tensors: Mapping[str, torch.Tensor],
    baseline: torch.Tensor,
    family_ids: np.ndarray,
    *,
    n_steps: int,
    batch_size: int,
) -> Tuple[pd.Series, pd.Series, np.ndarray]:
    """Return (abs_profile, signed_profile, attribution_matrix [N,F])."""
    n = batch_tensors["states"].shape[0]
    attrs = []
    for start in range(0, n, batch_size):
        sl = slice(start, min(n, start + batch_size))
        sub = {k: v[sl] for k, v in batch_tensors.items() if torch.is_tensor(v)}
        base = baseline[sl] if baseline.shape[0] == n else baseline
        attr = adapter.compute_integrated_gradients(sub, base, n_steps=n_steps)
        # attr: [B, T, F] -> [B, F]
        mat = attr.detach().cpu().numpy()
        if mat.ndim == 3:
            mat = mat.sum(axis=1)
        attrs.append(mat)
    matrix = np.concatenate(attrs, axis=0)
    abs_prof = aggregate_family_profile(matrix, family_ids, signed=False)
    signed_prof = aggregate_family_profile(matrix, family_ids, signed=True)
    return abs_prof, signed_prof, matrix


def compute_background_averaged_profiles(
    adapter: DecisionTransformerAdapter,
    batch_tensors: Mapping[str, torch.Tensor],
    background_states: torch.Tensor,
    family_ids: np.ndarray,
    *,
    n_steps: int,
    batch_size: int,
) -> Tuple[pd.Series, pd.Series, np.ndarray]:
    """Average IG attributions across each background observation as baseline."""
    n = batch_tensors["states"].shape[0]
    accum = None
    for b_idx in range(background_states.shape[0]):
        base_vec = background_states[b_idx : b_idx + 1]
        baseline = base_vec.expand_as(batch_tensors["states"]).clone()
        _, _, matrix = compute_ig_family_profiles(
            adapter,
            batch_tensors,
            baseline,
            family_ids,
            n_steps=n_steps,
            batch_size=batch_size,
        )
        accum = matrix if accum is None else accum + matrix
    assert accum is not None
    matrix = accum / float(background_states.shape[0])
    abs_prof = aggregate_family_profile(matrix, family_ids, signed=False)
    signed_prof = aggregate_family_profile(matrix, family_ids, signed=True)
    return abs_prof, signed_prof, matrix


def run_baseline_concordance(
    adapter: DecisionTransformerAdapter,
    batch_tensors: Mapping[str, torch.Tensor],
    pool_states: torch.Tensor,
    background_states: torch.Tensor,
    family_ids: np.ndarray,
    *,
    model_family: str,
    n_steps: int,
    batch_size: int,
    top_k: int,
) -> Dict[str, Any]:
    zero = build_zero_baseline(batch_tensors["states"])
    median = build_empirical_median_baseline(pool_states, batch_tensors["states"])

    abs_profiles: Dict[str, pd.Series] = {}
    signed_profiles: Dict[str, pd.Series] = {}
    matrices: Dict[str, np.ndarray] = {}

    for name, baseline in (("zero", zero), ("empirical_median", median)):
        abs_p, signed_p, mat = compute_ig_family_profiles(
            adapter, batch_tensors, baseline, family_ids, n_steps=n_steps, batch_size=batch_size
        )
        abs_profiles[name] = abs_p
        signed_profiles[name] = signed_p
        matrices[name] = mat

    abs_p, signed_p, mat = compute_background_averaged_profiles(
        adapter,
        batch_tensors,
        background_states,
        family_ids,
        n_steps=n_steps,
        batch_size=batch_size,
    )
    abs_profiles["sampled_background"] = abs_p
    signed_profiles["sampled_background"] = signed_p
    matrices["sampled_background"] = mat

    pairwise = pairwise_concordance(abs_profiles, signed_profiles, top_k)
    pairwise.insert(0, "model_family", model_family)
    status, metrics = classify_baseline_status(pairwise)

    per_family_rows = []
    labels = sorted(set().union(*(set(s.index) for s in abs_profiles.values())))
    for fam in labels:
        row = {"model_family": model_family, "feature_family": fam}
        for bname in abs_profiles:
            row[f"abs_{bname}"] = float(abs_profiles[bname].get(fam, 0.0))
            row[f"signed_{bname}"] = float(signed_profiles[bname].get(fam, 0.0))
            row[f"rank_abs_{bname}"] = int(
                ranking_from_abs_profile(abs_profiles[bname]).index(fam) + 1
                if fam in abs_profiles[bname].index
                else -1
            )
        per_family_rows.append(row)
    per_family = pd.DataFrame(per_family_rows)

    sens_rows = []
    for cutoff in BASELINE_SENSITIVITY_TAUS:
        if metrics.get("median_kendall_tau_b", float("nan")) >= cutoff and metrics.get(
            "min_kendall_tau_b", float("nan")
        ) >= 0:
            label = "SUPPORTED_like" if cutoff <= metrics.get("median_kendall_tau_b", -1) else "MIXED_like"
        else:
            label = "UNSUPPORTED_like"
        # Re-evaluate with cutoff replacing 0.50
        med = metrics.get("median_kendall_tau_b", float("nan"))
        mn = metrics.get("min_kendall_tau_b", float("nan"))
        jacc = metrics.get("min_top_k_jaccard", float("nan"))
        sign = metrics.get("mean_top_k_sign_agreement", float("nan"))
        if med >= cutoff and jacc >= BASELINE_SUPPORTED_MIN_JACCARD and sign >= BASELINE_SUPPORTED_SIGN_AGREE:
            label = "SUPPORTED"
        elif med >= BASELINE_MIXED_MEDIAN_TAU and mn >= 0:
            label = "MIXED"
        elif not np.isfinite(med):
            label = "INCOMPLETE"
        else:
            label = "UNSUPPORTED"
        sens_rows.append(
            {
                "model_family": model_family,
                "median_tau_cutoff": cutoff,
                "status": label,
                "median_kendall_tau_b": med,
                "min_kendall_tau_b": mn,
            }
        )

    return {
        "status": status,
        "metrics": metrics,
        "pairwise": pairwise,
        "per_family": per_family,
        "sensitivity": pd.DataFrame(sens_rows),
        "abs_profiles": abs_profiles,
        "signed_profiles": signed_profiles,
        "matrices": matrices,
        "empirical_median_baseline": median,
        "zero_baseline": zero,
    }
