"""
Necessity diagnostics: does removing the identified information change the decision?

Two experiments are run for every model.

**Attribution-guided deletion.** Feature groups -- complete (instrument, indicator)
series, not isolated flattened coordinates -- are ranked by the contribution
analysis and the top 5 / 10 / 20 / 30% are replaced by an on-manifold reference
drawn from the training period. The same experiment is repeated with matched
random feature groups so that the excess degradation of guided over random
deletion can be estimated with uncertainty.

**Temporal occlusion.** Complete historical timesteps are occluded, giving a lag
profile per model. Because the models have different lookbacks, the raw lag axis
is retained and a normalised relative-lag axis in [0, 1] is added for comparison;
interpolation is used only for the common-grid summary.

Degradation is measured on the decision itself: total-variation distance between
the original and perturbed allocations, selected-action flip rate for discrete
policies, absolute change of the CASH weight, Jensen--Shannon divergence between
allocations, and the normalised change of the decision target.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats

from .targets import PolicyAdapter, Rollout

DELETION_FRACTIONS: Tuple[float, ...] = (0.05, 0.10, 0.20, 0.30)
EPS = 1e-12


# ---------------------------------------------------------------------------
# Decision-change measures
# ---------------------------------------------------------------------------


def total_variation(w0: np.ndarray, w1: np.ndarray) -> np.ndarray:
    """Total-variation distance between allocation rows, in [0, 1]."""
    return 0.5 * np.abs(w1 - w0).sum(axis=1)


def jensen_shannon(w0: np.ndarray, w1: np.ndarray) -> np.ndarray:
    """Jensen--Shannon divergence between allocation rows (base 2), in [0, 1]."""
    p = np.clip(w0, EPS, None)
    q = np.clip(w1, EPS, None)
    p = p / p.sum(axis=1, keepdims=True)
    q = q / q.sum(axis=1, keepdims=True)
    m = 0.5 * (p + q)

    def kl(a, b):
        return (a * (np.log2(a) - np.log2(b))).sum(axis=1)

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def decision_change(
    adapter: PolicyAdapter,
    sample: Rollout,
    perturbed: torch.Tensor,
    w0: np.ndarray,
    sel0: Optional[np.ndarray],
    tau0: np.ndarray,
    tau_scale: float,
    variant: str,
    reference: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """Aggregate decision degradation caused by one perturbation."""
    w1 = adapter.portfolio_weights(perturbed, sample.aux)
    with torch.no_grad():
        tau1 = (
            adapter.decision_target(perturbed.to(adapter.device), sample.aux, reference, variant=variant)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
    out = {
        "weight_l1": float(np.abs(w1 - w0).sum(axis=1).mean()),
        "weight_tv": float(total_variation(w0, w1).mean()),
        "jensen_shannon": float(jensen_shannon(w0, w1).mean()),
        "cash_change": float(np.abs(w1[:, -1] - w0[:, -1]).mean()),
        "target_change_normalised": float(np.abs(tau1 - tau0).mean() / max(tau_scale, EPS)),
    }
    if sel0 is not None:
        sel1 = adapter.selected_actions(perturbed, sample.aux)
        out["action_flip_rate"] = float((sel1 != sel0).mean())
    else:
        out["action_flip_rate"] = float("nan")
    return out


# ---------------------------------------------------------------------------
# Attribution-guided deletion
# ---------------------------------------------------------------------------


def feature_group_ranking(
    cube: np.ndarray, feature_labels: Sequence[Tuple[str, str]]
) -> np.ndarray:
    """Indices of the observation features ordered by decreasing attribution."""
    score = np.abs(cube).sum(axis=2).mean(axis=0)  # (n_features,)
    return np.argsort(-score, kind="stable")


def _apply_group_deletion(
    states: torch.Tensor, replacement: torch.Tensor, features: Sequence[int]
) -> torch.Tensor:
    out = states.clone()
    idx = torch.as_tensor(list(features), dtype=torch.long)
    if idx.numel():
        out[:, :, idx] = replacement[:, :, idx]
    return out


def run_deletion(
    adapter: PolicyAdapter,
    sample: Rollout,
    cube: np.ndarray,
    replacement: torch.Tensor,
    variant: str,
    *,
    fractions: Sequence[float] = DELETION_FRACTIONS,
    n_random: int = 20,
    seed: int = 42,
) -> Dict[str, object]:
    """Guided versus matched-random feature-group deletion curves."""
    w0 = sample.weights.values.astype(np.float64)
    sel0 = adapter.selected_actions(sample.states, sample.aux)
    reference = adapter.reference(sample.states, sample.aux)
    with torch.no_grad():
        tau0 = (
            adapter.decision_target(
                sample.states.to(adapter.device), sample.aux, reference, variant=variant
            )
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
    tau_scale = float(np.std(tau0)) if np.std(tau0) > 0 else float(np.mean(np.abs(tau0)))
    order = feature_group_ranking(cube, adapter.feature_labels)
    n_features = adapter.n_features
    rng = np.random.default_rng(seed)
    degenerate_attribution = float(np.abs(cube).max()) <= 1e-9

    guided_rows, random_rows = [], []
    for frac in fractions:
        k = max(1, int(round(frac * n_features)))
        guided = order[:k]
        rec = decision_change(
            adapter,
            sample,
            _apply_group_deletion(sample.states, replacement, guided),
            w0,
            sel0,
            tau0,
            tau_scale,
            variant,
            reference,
        )
        rec.update({"fraction": frac, "n_features_deleted": k, "strategy": "attribution_guided"})
        guided_rows.append(rec)

        reps = []
        for r in range(int(n_random)):
            choice = rng.choice(n_features, size=k, replace=False)
            rr = decision_change(
                adapter,
                sample,
                _apply_group_deletion(sample.states, replacement, choice),
                w0,
                sel0,
                tau0,
                tau_scale,
                variant,
                reference,
            )
            rr.update({"fraction": frac, "n_features_deleted": k, "strategy": "random", "repeat": r})
            reps.append(rr)
        random_rows.extend(reps)

    guided_df = pd.DataFrame(guided_rows)
    random_df = pd.DataFrame(random_rows)
    metric_cols = [
        "weight_l1",
        "weight_tv",
        "jensen_shannon",
        "cash_change",
        "target_change_normalised",
        "action_flip_rate",
    ]
    random_summary = (
        random_df.groupby("fraction")[metric_cols].agg(["mean", "std"]).reset_index()
    )
    random_summary.columns = [
        "fraction" if a == "fraction" else f"random_{a}_{b}" for a, b in random_summary.columns
    ]

    merged = guided_df.merge(random_summary, on="fraction", how="left")
    for col in metric_cols:
        merged[f"excess_{col}"] = merged[col] - merged[f"random_{col}_mean"]
        # Relative excess: the guided minus random difference divided by their sum.
        # This is the dimensionless form required for cross-architecture comparison,
        # because the absolute size of a decision change depends on how much a given
        # policy moves its allocation at all. It lies in [-1, 1]; zero means guided
        # deletion is exactly as damaging as random deletion.
        denom = (merged[col] + merged[f"random_{col}_mean"]).values.astype(np.float64)
        numer = merged[f"excess_{col}"].values.astype(np.float64)
        relative = np.full(denom.shape, np.nan, dtype=np.float64)
        usable = np.isfinite(denom) & (denom > EPS)
        np.divide(numer, denom, out=relative, where=usable)
        relative[~usable] = np.nan
        merged[f"relative_excess_{col}"] = relative

    excess_tv = float(np.nanmean(merged["excess_weight_tv"].values))
    excess_target = float(np.nanmean(merged["excess_target_change_normalised"].values))

    def pooled_relative_excess(col: str) -> float:
        """
        Relative excess degradation pooled over the deletion grid.

        The ratio is taken between the grid means rather than averaged over
        per-fraction ratios, because an individual fraction can have a denominator
        close to zero and would then dominate an average of ratios.
        """
        g = float(np.nanmean(merged[col].values.astype(np.float64)))
        r = float(np.nanmean(merged[f"random_{col}_mean"].values.astype(np.float64)))
        return (g - r) / (g + r) if np.isfinite(g + r) and (g + r) > EPS else float("nan")

    relative_tv = pooled_relative_excess("weight_tv")
    relative_target = pooled_relative_excess("target_change_normalised")

    # Primary faithfulness: the pooled relative excess allocation change, clipped
    # at zero so that "guided deletion is no more damaging than random" scores zero.
    if degenerate_attribution:
        # No attribution ranking exists, so guided deletion is not defined.
        faithfulness = float("nan")
    elif not np.isfinite(relative_tv):
        # Neither guided nor random deletion moves the allocation at all: the
        # information identified as influential is demonstrably not necessary.
        faithfulness = 0.0
    else:
        faithfulness = float(np.clip(relative_tv, 0.0, 1.0))

    return {
        "guided": guided_df,
        "random": random_df,
        "curve": merged,
        "metrics": {
            "deletion_excess_weight_tv": excess_tv,
            "deletion_excess_target_change": excess_target,
            "deletion_relative_excess_weight_tv": relative_tv,
            "deletion_relative_excess_target_change": relative_target,
            "deletion_faithfulness": faithfulness,
            "deletion_guided_weight_tv_at_30pct": float(
                merged.loc[merged["fraction"] == max(fractions), "weight_tv"].iloc[0]
            ),
            "deletion_guided_flip_rate_at_30pct": float(
                merged.loc[merged["fraction"] == max(fractions), "action_flip_rate"].iloc[0]
            ),
        },
    }


# ---------------------------------------------------------------------------
# Temporal occlusion
# ---------------------------------------------------------------------------


def temporal_occlusion_profiles(
    adapter: PolicyAdapter,
    sample: Rollout,
    replacement: torch.Tensor,
    variant: str,
) -> Dict[str, np.ndarray]:
    """
    Occlude one complete historical timestep at a time.

    Returns per-sample profiles over lags with index 0 the most recent row, for
    the decision-target change and for the allocation total-variation change.
    """
    w0 = sample.weights.values.astype(np.float64)
    reference = adapter.reference(sample.states, sample.aux)
    with torch.no_grad():
        tau0 = (
            adapter.decision_target(
                sample.states.to(adapter.device), sample.aux, reference, variant=variant
            )
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
    n, window = sample.states.shape[0], sample.states.shape[1]
    tau_profile = np.zeros((n, window), dtype=np.float64)
    tv_profile = np.zeros((n, window), dtype=np.float64)
    for row in range(window):
        pert = sample.states.clone()
        pert[:, row, :] = replacement[:, row, :]
        with torch.no_grad():
            tau = (
                adapter.decision_target(pert.to(adapter.device), sample.aux, reference, variant=variant)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
        w1 = adapter.portfolio_weights(pert, sample.aux)
        lag = window - 1 - row
        tau_profile[:, lag] = np.abs(tau - tau0)
        tv_profile[:, lag] = total_variation(w0, w1)
    return {"target": tau_profile, "weight_tv": tv_profile}


def _normalise_rows(profile: np.ndarray) -> np.ndarray:
    total = profile.sum(axis=1, keepdims=True)
    out = np.zeros_like(profile)
    ok = (total > EPS).ravel()
    out[ok] = profile[ok] / total[ok]
    return out


def temporal_metrics(profile: np.ndarray) -> Dict[str, float]:
    """Shape descriptors of a normalised lag profile (index 0 = most recent)."""
    p = _normalise_rows(profile)
    valid = p.sum(axis=1) > 0
    if not valid.any():
        return {
            k: float("nan")
            for k in (
                "peak_lag",
                "half_life",
                "recency_index",
                "entropy",
                "hhi",
                "normalised_auc",
            )
        }
    p = p[valid]
    lags = np.arange(p.shape[1])
    cdf = np.cumsum(p, axis=1)
    half = (cdf >= 0.5).argmax(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.where(p > 0, p * np.log(p), 0.0).sum(axis=1) / np.log(p.shape[1])
    mean_curve = p.mean(axis=0)
    denom = max(p.shape[1] - 1, 1)
    return {
        "peak_lag": float(np.median(p.argmax(axis=1))),
        "half_life": float(np.median(half)),
        "recency_index": float(((p * lags).sum(axis=1) / denom).mean()),
        "entropy": float(ent.mean()),
        "hhi": float((p**2).sum(axis=1).mean()),
        "normalised_auc": float(np.trapz(mean_curve, dx=1.0 / denom)),
    }


def profile_stability(
    profile: np.ndarray, regimes: np.ndarray, n_bootstrap: int = 200, seed: int = 42
) -> Dict[str, float]:
    """Stability of the normalised lag profile across stratified bootstrap resamples."""
    p = _normalise_rows(profile)
    if p.sum() <= EPS or p.shape[1] < 3:
        return {"median_spearman": float("nan"), "min_spearman": float("nan"), "n_pairs": 0}
    rng = np.random.default_rng(seed)
    idx_by_regime = [np.where(regimes == r)[0] for r in np.unique(regimes)]
    reps = []
    for _ in range(int(n_bootstrap)):
        rows = np.concatenate([rng.choice(v, size=len(v), replace=True) for v in idx_by_regime])
        reps.append(p[rows].mean(axis=0))
    matrix = np.stack(reps)
    corrs = []
    cap = min(60, matrix.shape[0])
    for i in range(cap):
        for j in range(i + 1, cap):
            a, b = matrix[i], matrix[j]
            if np.allclose(a, a[0]) or np.allclose(b, b[0]):
                continue
            c = stats.spearmanr(a, b).correlation
            if np.isfinite(c):
                corrs.append(float(c))
    if not corrs:
        return {"median_spearman": float("nan"), "min_spearman": float("nan"), "n_pairs": 0}
    return {
        "median_spearman": float(np.median(corrs)),
        "min_spearman": float(np.min(corrs)),
        "n_pairs": len(corrs),
    }


def relative_lag_grid(profile: np.ndarray, n_grid: int = 33) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate a mean lag profile onto a common relative-lag axis in [0, 1].

    Used only for cross-model visualisation and summary; the raw lag values are
    retained by the caller.
    """
    p = _normalise_rows(profile).mean(axis=0)
    if p.size < 2:
        return np.linspace(0, 1, n_grid), np.full(n_grid, np.nan)
    src = np.linspace(0.0, 1.0, p.size)
    grid = np.linspace(0.0, 1.0, n_grid)
    return grid, np.interp(grid, src, p)


def run_necessity(
    adapter: PolicyAdapter,
    sample: Rollout,
    regimes: np.ndarray,
    cube: np.ndarray,
    replacement: torch.Tensor,
    variant: str,
    *,
    n_random: int = 20,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> Dict[str, object]:
    """Compute every necessity artefact for one model."""
    deletion = run_deletion(
        adapter, sample, cube, replacement, variant, n_random=n_random, seed=seed
    )
    profiles = temporal_occlusion_profiles(adapter, sample, replacement, variant)
    primary = profiles["target"]
    metrics_t = temporal_metrics(primary)
    metrics_w = temporal_metrics(profiles["weight_tv"])
    stability = profile_stability(primary, regimes, n_bootstrap=n_bootstrap, seed=seed)

    regime_profiles = {}
    for r in np.unique(regimes):
        rows = np.where(regimes == r)[0]
        regime_profiles[r] = _normalise_rows(primary[rows]).mean(axis=0)

    grid, interpolated = relative_lag_grid(primary)

    metrics = {
        **deletion["metrics"],
        "temporal_profile_stability": stability["median_spearman"],
        "temporal_profile_stability_worst": stability["min_spearman"],
        **{f"temporal_{k}": v for k, v in metrics_t.items()},
        **{f"temporal_weight_{k}": v for k, v in metrics_w.items()},
    }
    return {
        "model_id": adapter.record.model_id,
        "deletion": deletion,
        "profiles_raw": profiles,
        "profile_normalised": _normalise_rows(primary).mean(axis=0),
        "profile_weight_normalised": _normalise_rows(profiles["weight_tv"]).mean(axis=0),
        "regime_profiles": regime_profiles,
        "relative_lag_grid": grid,
        "relative_lag_profile": interpolated,
        "temporal_metrics_target": metrics_t,
        "temporal_metrics_weight": metrics_w,
        "profile_stability": stability,
        "metrics": metrics,
    }
