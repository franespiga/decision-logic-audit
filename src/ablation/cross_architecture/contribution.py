"""
Contribution diagnostics: which inputs influence the allocation decision?

Two attribution methods are computed for every model:

* **group occlusion** -- a perturbation method that replaces one
  (feature, lag) coordinate with a reference value and records the absolute
  change of the decision target. It requires no gradient and is therefore
  defined for every architecture in this study.
* **integrated gradients** -- the gradient-based reference method, computed with
  the same three baselines.

Both are reported for every model. Which one supplies the primary ranking is a
per-model technical decision recorded in ``attribution_method`` and driven only by
whether the policy admits usable gradients (see ``gradient_health``); it is never
chosen by inspecting the diagnostic outcome.

All cross-model quantities derived here are normalised shares or ranks.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats

from .targets import PolicyAdapter, Rollout, tile_context

BASELINE_NAMES = ("zero", "train_mean", "train_sample")
DEGENERACY_TOL = 1e-9


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def build_baselines(
    adapter: PolicyAdapter,
    n_samples: int,
    train_frame: pd.DataFrame,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """
    Reference tensors of shape ``(n_samples, window, n_features)``.

    ``zero``          the naive zero reference;
    ``train_mean``    the per-feature training-period mean;
    ``train_sample``  a data-supported empirical background: for every sample row
                      one complete training-period observation window is drawn, so
                      the reference stays on the data manifold and preserves the
                      cross-feature correlation structure.
    """
    window, n_features = adapter.window, adapter.n_features
    values = train_frame.loc[:, adapter.feature_labels].values.astype(np.float32)
    if values.shape[0] < window:
        raise ValueError(f"{adapter.record.model_id}: training frame shorter than the window")
    mean = values.mean(axis=0)

    rng = np.random.default_rng(seed)
    starts = rng.integers(0, values.shape[0] - window + 1, size=n_samples)
    background = np.stack([values[s : s + window] for s in starts]).astype(np.float32)

    return {
        "zero": torch.zeros(n_samples, window, n_features, dtype=torch.float32),
        "train_mean": torch.from_numpy(
            np.broadcast_to(mean, (n_samples, window, n_features)).copy()
        ),
        "train_sample": torch.from_numpy(background),
    }


# ---------------------------------------------------------------------------
# Decision-target evaluation
# ---------------------------------------------------------------------------


def evaluate_target(
    adapter: PolicyAdapter,
    states: torch.Tensor,
    aux: Dict[str, torch.Tensor],
    reference: Dict[str, torch.Tensor],
    variant: str = "primary",
) -> np.ndarray:
    with torch.no_grad():
        tau = adapter.decision_target(states, aux, reference, variant=variant)
    return tau.detach().cpu().numpy().astype(np.float64)


def gradient_health(
    adapter: PolicyAdapter, sample: Rollout, variant: str = "primary"
) -> Dict[str, float]:
    """
    Measure whether the policy admits usable gradients at the sampled points.

    A saturating squashing non-linearity makes the decision target locally
    constant, so every gradient-based attribution is identically zero even though
    the policy is not constant globally. This is a property of the frozen policy,
    not of the diagnostic.
    """
    states = sample.states.clone().to(adapter.device).requires_grad_(True)
    reference = adapter.reference(sample.states, sample.aux)
    tau = adapter.decision_target(states, sample.aux, reference, variant=variant)
    grad = torch.autograd.grad(tau.sum(), states, allow_unused=True)[0]
    if grad is None:
        return {"grad_abs_mean": 0.0, "grad_nonzero_fraction": 0.0, "target_std": 0.0}
    with torch.no_grad():
        tau_v = adapter.decision_target(sample.states, sample.aux, reference, variant=variant)
    return {
        "grad_abs_mean": float(grad.abs().mean().item()),
        "grad_nonzero_fraction": float((grad != 0).float().mean().item()),
        "target_std": float(tau_v.detach().float().std(unbiased=False).item()),
    }


# ---------------------------------------------------------------------------
# Attribution cubes
# ---------------------------------------------------------------------------


def occlusion_attribution_cube(
    adapter: PolicyAdapter,
    sample: Rollout,
    baseline: torch.Tensor,
    variant: str = "primary",
    coord_chunk: int = 16,
) -> np.ndarray:
    """
    Absolute decision-target change from occluding each (feature, lag) coordinate.

    Returns ``(N, n_features, window)`` with lag axis ordered so that index 0 is
    the most recent observation row.
    """
    device = adapter.device
    states = sample.states.to(device)
    base = baseline.to(device)
    reference = adapter.reference(sample.states, sample.aux)
    n, window, n_features = states.shape
    tau0 = adapter.decision_target(states, sample.aux, reference, variant=variant).detach()

    out = np.zeros((n, n_features, window), dtype=np.float64)
    coords = [(f, r) for f in range(n_features) for r in range(window)]
    with torch.no_grad():
        for start in range(0, len(coords), coord_chunk):
            chunk = coords[start : start + coord_chunk]
            reps = len(chunk)
            big = states.repeat(reps, 1, 1)
            for j, (f, r) in enumerate(chunk):
                big[j * n : (j + 1) * n, r, f] = base[:, r, f]
            tau = adapter.decision_target(
                big, tile_context(sample.aux, reps), tile_context(reference, reps), variant=variant
            )
            delta = (tau.view(reps, n) - tau0.view(1, n)).abs().detach().cpu().numpy()
            for j, (f, r) in enumerate(chunk):
                out[:, f, window - 1 - r] = delta[j]
    return out


def integrated_gradients_cube(
    adapter: PolicyAdapter,
    sample: Rollout,
    baseline: torch.Tensor,
    variant: str = "primary",
    n_steps: int = 32,
    batch_size: int = 32,
) -> Tuple[np.ndarray, float]:
    """
    Integrated gradients of the decision target, plus the convergence delta.

    Returns ``(N, n_features, window)`` signed attributions with lag index 0 the
    most recent row, and the mean absolute convergence delta.
    """
    from captum.attr import IntegratedGradients

    device = adapter.device
    attributions: List[np.ndarray] = []
    deltas: List[float] = []
    n = len(sample)
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        sub = sample.select(range(start, stop))
        states = sub.states.to(device)
        base = baseline[start:stop].to(device)
        reference = adapter.reference(sub.states, sub.aux)
        aux = sub.aux
        batch = stop - start

        def forward(x: torch.Tensor) -> torch.Tensor:
            # Captum expands the batch along the integration path in alpha-major
            # order, so tiling the policy context by the expansion factor keeps
            # every row paired with its own observation.
            reps = max(1, x.shape[0] // batch)
            return adapter.decision_target(
                x, tile_context(aux, reps), tile_context(reference, reps), variant=variant
            )

        ig = IntegratedGradients(forward)
        attr, delta = ig.attribute(
            states, baselines=base, n_steps=n_steps, return_convergence_delta=True
        )
        attributions.append(attr.detach().cpu().numpy())
        deltas.append(float(delta.abs().mean().item()))
    cube = np.concatenate(attributions, axis=0)  # (N, window, n_features)
    cube = np.transpose(cube, (0, 2, 1))[:, :, ::-1]  # -> (N, n_features, lag) lag0 = recent
    return np.ascontiguousarray(cube), float(np.mean(deltas))


# ---------------------------------------------------------------------------
# Aggregation into dimensionless shares
# ---------------------------------------------------------------------------


def _shares(values: np.ndarray) -> np.ndarray:
    total = float(np.sum(np.abs(values)))
    if total <= 0:
        return np.zeros_like(values, dtype=np.float64)
    return np.abs(values) / total


def normalised_entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    if p.size <= 1:
        return 0.0
    return float(-(p * np.log(p)).sum() / np.log(p.size))


def hhi(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    return float((p**2).sum())


def effective_number(p: np.ndarray) -> float:
    h = hhi(p)
    return float(1.0 / h) if h > 0 else 0.0


def aggregate_cube(
    cube: np.ndarray,
    feature_labels: Sequence[Tuple[str, str]],
    window: int,
) -> Dict[str, pd.Series]:
    """
    Collapse an attribution cube into normalised shares along each semantic axis.

    ``cube`` is ``(N, n_features, window)``; the mean over the sample of the
    absolute attribution is normalised to sum to one within each axis.
    """
    mean_abs = np.abs(cube).mean(axis=0)  # (n_features, window)
    instruments = [f[0] for f in feature_labels]
    indicators = [f[1] for f in feature_labels]
    per_feature = mean_abs.sum(axis=1)
    per_lag = mean_abs.sum(axis=0)

    feature_names = [f"{i}|{j}" for i, j in feature_labels]
    out = {
        "feature": pd.Series(_shares(per_feature), index=feature_names).sort_index(),
        "indicator": pd.Series(per_feature, index=indicators).groupby(level=0).sum(),
        "instrument": pd.Series(per_feature, index=instruments).groupby(level=0).sum(),
        "lag": pd.Series(per_lag, index=[f"lag_{k}" for k in range(window)]),
    }
    out["indicator"] = pd.Series(_shares(out["indicator"].values), index=out["indicator"].index)
    out["instrument"] = pd.Series(_shares(out["instrument"].values), index=out["instrument"].index)
    out["lag"] = pd.Series(_shares(out["lag"].values), index=out["lag"].index)
    return out


def signed_feature_summary(
    cube: np.ndarray, feature_labels: Sequence[Tuple[str, str]]
) -> pd.DataFrame:
    """Signed and absolute per-feature attribution (signed only meaningful for IG)."""
    signed = cube.sum(axis=2).mean(axis=0)
    absolute = np.abs(cube).sum(axis=2).mean(axis=0)
    return pd.DataFrame(
        {
            "instrument": [f[0] for f in feature_labels],
            "indicator": [f[1] for f in feature_labels],
            "mean_signed": signed,
            "mean_absolute": absolute,
            "absolute_share": _shares(absolute),
        }
    ).sort_values("absolute_share", ascending=False)


def concentration_summary(shares: Dict[str, pd.Series]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for axis, s in shares.items():
        p = s.values.astype(np.float64)
        out[f"{axis}_entropy"] = normalised_entropy(p)
        out[f"{axis}_hhi"] = hhi(p)
        out[f"{axis}_effective_number"] = effective_number(p)
    return out


# ---------------------------------------------------------------------------
# Rank stability and baseline concordance
# ---------------------------------------------------------------------------


def _group_labels(
    feature_labels: Sequence[Tuple[str, str]], level: str = "feature"
) -> List[str]:
    """Semantic group label per observation feature at the requested granularity."""
    if level == "indicator":
        return [f[1] for f in feature_labels]
    if level == "instrument":
        return [f[0] for f in feature_labels]
    return [f"{f[0]}|{f[1]}" for f in feature_labels]


def _group_scores(cube: np.ndarray, groups: Sequence[str], rows: np.ndarray) -> np.ndarray:
    """Mean absolute attribution per semantic group over the given sample rows."""
    per_feature = np.abs(cube[rows]).sum(axis=2).mean(axis=0)
    frame = pd.Series(per_feature, index=list(groups)).groupby(level=0).sum()
    return frame.sort_index().values


def _median_pairwise_rank_correlation(
    matrix: np.ndarray, method: str = "spearman"
) -> Tuple[float, float, int]:
    """Median and minimum pairwise rank correlation between the rows of ``matrix``."""
    n = matrix.shape[0]
    corrs: List[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = matrix[i], matrix[j]
            if np.allclose(a, a[0]) or np.allclose(b, b[0]):
                continue
            if method == "kendall":
                c = stats.kendalltau(a, b).correlation
            else:
                c = stats.spearmanr(a, b).correlation
            if np.isfinite(c):
                corrs.append(float(c))
    if not corrs:
        return float("nan"), float("nan"), 0
    return float(np.median(corrs)), float(np.min(corrs)), len(corrs)


def bootstrap_rank_stability(
    cube: np.ndarray,
    feature_labels: Sequence[Tuple[str, str]],
    regimes: np.ndarray,
    n_bootstrap: int = 200,
    seed: int = 42,
    level: str = "feature",
) -> Dict[str, object]:
    """
    Stability of the feature-group ranking across regime-stratified bootstrap
    resamples of the diagnostic sample.

    ``level`` selects the semantic granularity: ``feature`` for
    (instrument, indicator) pairs, ``indicator`` for indicator families.
    """
    groups = _group_labels(feature_labels, level)
    names = sorted(set(groups))

    if float(np.abs(cube).max()) <= DEGENERACY_TOL:
        return {
            "median_spearman": float("nan"),
            "min_spearman": float("nan"),
            "median_kendall": float("nan"),
            "n_pairs": 0,
            "degenerate": True,
            "replicate_scores": None,
            "groups": names,
        }

    rng = np.random.default_rng(seed)
    n = cube.shape[0]
    idx_by_regime = {r: np.where(regimes == r)[0] for r in np.unique(regimes)}
    replicates = []
    for _ in range(int(n_bootstrap)):
        rows = np.concatenate(
            [rng.choice(v, size=len(v), replace=True) for v in idx_by_regime.values()]
        )
        replicates.append(_group_scores(cube, groups, rows))
    matrix = np.stack(replicates)
    med_s, min_s, n_pairs = _median_pairwise_rank_correlation(matrix, "spearman")
    # Kendall on a subset of replicate pairs: it is O(k log k) but the pair count
    # grows quadratically, so cap the number of replicates used.
    cap = min(40, matrix.shape[0])
    med_k, _, _ = _median_pairwise_rank_correlation(matrix[:cap], "kendall")
    return {
        "median_spearman": med_s,
        "min_spearman": min_s,
        "median_kendall": med_k,
        "n_pairs": n_pairs,
        "degenerate": False,
        "replicate_scores": matrix,
        "groups": names,
    }


def rolling_rank_stability(
    cube: np.ndarray,
    feature_labels: Sequence[Tuple[str, str]],
    n_windows: int = 8,
    level: str = "feature",
) -> Dict[str, float]:
    """Stability of the feature-group ranking across contiguous temporal windows."""
    groups = _group_labels(feature_labels, level)
    if float(np.abs(cube).max()) <= DEGENERACY_TOL:
        return {"median_spearman": float("nan"), "min_spearman": float("nan"), "n_windows": 0}
    n = cube.shape[0]
    bounds = np.linspace(0, n, n_windows + 1).astype(int)
    blocks = [np.arange(bounds[i], bounds[i + 1]) for i in range(n_windows) if bounds[i + 1] > bounds[i]]
    if len(blocks) < 2:
        return {"median_spearman": float("nan"), "min_spearman": float("nan"), "n_windows": len(blocks)}
    matrix = np.stack([_group_scores(cube, groups, b) for b in blocks])
    med, mn, _ = _median_pairwise_rank_correlation(matrix, "spearman")
    return {"median_spearman": med, "min_spearman": mn, "n_windows": len(blocks)}


def baseline_concordance(
    cubes: Dict[str, np.ndarray],
    feature_labels: Sequence[Tuple[str, str]],
    level: str = "feature",
) -> Dict[str, object]:
    """Rank concordance of the feature-group ranking between attribution baselines."""
    groups = _group_labels(feature_labels, level)
    names = [k for k in cubes if float(np.abs(cubes[k]).max()) > DEGENERACY_TOL]
    if len(names) < 2:
        return {
            "median_spearman": float("nan"),
            "min_spearman": float("nan"),
            "pairwise": pd.DataFrame(),
            "degenerate": True,
        }
    rows = np.arange(next(iter(cubes.values())).shape[0])
    scores = {k: _group_scores(cubes[k], groups, rows) for k in names}
    recs = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            s = stats.spearmanr(scores[a], scores[b]).correlation
            k = stats.kendalltau(scores[a], scores[b]).correlation
            recs.append({"baseline_a": a, "baseline_b": b, "spearman": float(s), "kendall": float(k)})
    pairwise = pd.DataFrame(recs)
    return {
        "median_spearman": float(pairwise["spearman"].median()),
        "min_spearman": float(pairwise["spearman"].min()),
        "pairwise": pairwise,
        "degenerate": False,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_contribution(
    adapter: PolicyAdapter,
    sample: Rollout,
    regimes: np.ndarray,
    train_frame: pd.DataFrame,
    *,
    n_bootstrap: int = 200,
    seed: int = 42,
    ig_steps: int = 32,
    coord_chunk: int = 16,
) -> Dict[str, object]:
    """Compute every contribution artefact for one model."""
    model_id = adapter.record.model_id
    baselines = build_baselines(adapter, len(sample), train_frame, seed=seed)

    health = gradient_health(adapter, sample, "primary")
    variant = "primary"
    notes: List[str] = []
    if adapter.has_complementary_target():
        if health["target_std"] <= DEGENERACY_TOL:
            variant = "complementary"
            notes.append(
                "primary decision target is constant on the diagnostic sample; the "
                "documented complementary pre-squash target is used for attribution"
            )
        elif health["grad_nonzero_fraction"] <= 0.0:
            variant = "complementary"
            notes.append(
                "the primary decision target has zero gradient everywhere on the diagnostic "
                "sample because the squashing non-linearity is saturated; the documented "
                "complementary pre-squash target is used for attribution and for layer influence"
            )
    health_used = gradient_health(adapter, sample, variant)

    occ_cubes: Dict[str, np.ndarray] = {}
    ig_cubes: Dict[str, np.ndarray] = {}
    ig_deltas: Dict[str, float] = {}
    for name, base in baselines.items():
        occ_cubes[name] = occlusion_attribution_cube(
            adapter, sample, base, variant=variant, coord_chunk=coord_chunk
        )
        cube, delta = integrated_gradients_cube(
            adapter, sample, base, variant=variant, n_steps=ig_steps
        )
        ig_cubes[name] = cube
        ig_deltas[name] = delta

    ig_usable = health_used["grad_nonzero_fraction"] > 0.0 and any(
        float(np.abs(c).max()) > DEGENERACY_TOL for c in ig_cubes.values()
    )
    primary_method = "integrated_gradients" if ig_usable else "group_occlusion"
    if not ig_usable:
        notes.append(
            "integrated gradients vanish because the policy's activations are "
            "saturated at the sampled observations; the perturbation-based group "
            "occlusion attribution supplies the primary ranking"
        )
    primary_cubes = ig_cubes if primary_method == "integrated_gradients" else occ_cubes
    reference_baseline = "train_sample"
    primary_cube = primary_cubes[reference_baseline]

    shares = aggregate_cube(primary_cube, adapter.feature_labels, adapter.window)
    concentration = concentration_summary(shares)
    signed = signed_feature_summary(ig_cubes[reference_baseline], adapter.feature_labels)

    boot = bootstrap_rank_stability(
        primary_cube, adapter.feature_labels, regimes, n_bootstrap=n_bootstrap, seed=seed
    )
    boot_ind = bootstrap_rank_stability(
        primary_cube,
        adapter.feature_labels,
        regimes,
        n_bootstrap=n_bootstrap,
        seed=seed,
        level="indicator",
    )
    rolling = rolling_rank_stability(primary_cube, adapter.feature_labels)
    concord = baseline_concordance(primary_cubes, adapter.feature_labels)

    return {
        "model_id": model_id,
        "attribution_variant": variant,
        "attribution_method": primary_method,
        "reference_baseline": reference_baseline,
        "gradient_health": health,
        "gradient_health_used": health_used,
        "ig_convergence_delta": ig_deltas,
        "occlusion_cubes": occ_cubes,
        "ig_cubes": ig_cubes,
        "primary_cube": primary_cube,
        "shares": shares,
        "concentration": concentration,
        "signed_features": signed,
        "rank_stability_feature": boot,
        "rank_stability_indicator": boot_ind,
        "rank_stability_rolling": rolling,
        "baseline_concordance": concord,
        "degenerate": bool(boot.get("degenerate")),
        "notes": notes,
        "metrics": {
            "contribution_rank_stability": boot["median_spearman"],
            "contribution_rank_stability_indicator": boot_ind["median_spearman"],
            "contribution_rank_stability_rolling": rolling["median_spearman"],
            "contribution_baseline_concordance": concord["median_spearman"],
            "contribution_baseline_concordance_worst": concord["min_spearman"],
            "attribution_entropy_feature": concentration["feature_entropy"],
            "attribution_hhi_feature": concentration["feature_hhi"],
            "attribution_effective_features": concentration["feature_effective_number"],
        },
    }
