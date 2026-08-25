"""
Structure diagnostics: are the internal representations and the distribution of
decision influence stable and non-collapsed?

Latent neighbourhood stability is computed in the original high-dimensional
latent space -- never in a two-dimensional projection -- as the k-nearest
neighbour overlap between the unperturbed representation and the representation
obtained under three controlled perturbations: stratified subsample resampling,
small standardised input noise, and masking of low-importance feature groups.

Layer-influence concentration measures how the decision influence is distributed
over the comparable decision-processing layers of each network. Layer
conductance is attempted first; where the policy admits no usable gradient a
documented controlled layer-ablation alternative is used instead, and the method
is recorded per model. Raw influence values are never compared across
architectures -- only the normalised shares and the concentration indices are.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .contribution import effective_number, hhi, normalised_entropy
from .targets import PolicyAdapter, Rollout, tile_context

K_VALUES: Tuple[int, ...] = (5, 10, 20)
PRIMARY_K = 10
EPS = 1e-12


# ---------------------------------------------------------------------------
# Latent neighbourhood stability
# ---------------------------------------------------------------------------


def standardise(latent: np.ndarray) -> np.ndarray:
    """Zero-mean unit-variance per dimension; constant dimensions are dropped."""
    x = np.asarray(latent, dtype=np.float64)
    std = x.std(axis=0)
    keep = std > EPS
    if not keep.any():
        return np.zeros((x.shape[0], 1))
    x = x[:, keep]
    return (x - x.mean(axis=0)) / x.std(axis=0)


def knn_indices(x: np.ndarray, k: int, subset: Optional[np.ndarray] = None) -> np.ndarray:
    """Indices of the k nearest neighbours of every row, excluding the row itself."""
    rows = np.arange(x.shape[0]) if subset is None else np.asarray(subset)
    d = ((x[rows][:, None, :] - x[rows][None, :, :]) ** 2).sum(axis=-1)
    np.fill_diagonal(d, np.inf)
    kk = min(k, max(len(rows) - 1, 1))
    return np.argsort(d, axis=1, kind="stable")[:, :kk]


def neighbourhood_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Mean Jaccard similarity between two aligned neighbour-index matrices."""
    if a.shape != b.shape or a.size == 0:
        return float("nan")
    vals = []
    for i in range(a.shape[0]):
        sa, sb = set(a[i].tolist()), set(b[i].tolist())
        union = sa | sb
        vals.append(len(sa & sb) / len(union) if union else np.nan)
    return float(np.nanmean(vals))


def stability_under_resampling(
    latent: np.ndarray,
    regimes: np.ndarray,
    k_values: Sequence[int] = K_VALUES,
    n_replicates: int = 30,
    keep_fraction: float = 0.8,
    seed: int = 42,
) -> Dict[int, float]:
    """
    Neighbourhood agreement between regime-stratified subsamples.

    Two subsamples are drawn per replicate; the neighbourhoods of the observations
    they share are compared, so the metric reflects how much the latent geometry
    depends on which observations happen to be present.
    """
    x = standardise(latent)
    rng = np.random.default_rng(seed)
    by_regime = [np.where(regimes == r)[0] for r in np.unique(regimes)]
    out: Dict[int, List[float]] = {k: [] for k in k_values}
    for _ in range(int(n_replicates)):
        sets = []
        for _ in range(2):
            rows = np.concatenate(
                [
                    rng.choice(v, size=max(2, int(round(keep_fraction * len(v)))), replace=False)
                    for v in by_regime
                ]
            )
            sets.append(np.sort(rows))
        common = np.intersect1d(sets[0], sets[1])
        if len(common) < max(k_values) + 2:
            continue
        for k in k_values:
            neigh = []
            for rows in sets:
                idx = knn_indices(x[rows], k)
                # Map neighbour positions back to global row identifiers, then keep
                # only neighbours that exist in both subsamples.
                glob = rows[idx]
                pos = {r: i for i, r in enumerate(rows)}
                sel = np.array([pos[c] for c in common])
                neigh.append(glob[sel])
            a, b = neigh
            vals = []
            for i in range(len(common)):
                sa = set(a[i].tolist()) & set(common.tolist())
                sb = set(b[i].tolist()) & set(common.tolist())
                union = sa | sb
                if union:
                    vals.append(len(sa & sb) / len(union))
            if vals:
                out[k].append(float(np.mean(vals)))
    return {k: (float(np.mean(v)) if v else float("nan")) for k, v in out.items()}


def stability_under_input_noise(
    adapter: PolicyAdapter,
    sample: Rollout,
    feature_std: np.ndarray,
    k_values: Sequence[int] = K_VALUES,
    epsilon: float = 0.05,
    n_replicates: int = 5,
    seed: int = 42,
) -> Dict[int, float]:
    """Neighbourhood agreement under small standardised additive input noise."""
    base = standardise(adapter.latent(sample.states, sample.aux))
    scale = torch.tensor(feature_std, dtype=torch.float32).view(1, 1, -1)
    rng = torch.Generator().manual_seed(seed)
    out: Dict[int, List[float]] = {k: [] for k in k_values}
    ref = {k: knn_indices(base, k) for k in k_values}
    for _ in range(int(n_replicates)):
        noise = torch.randn(sample.states.shape, generator=rng) * scale * float(epsilon)
        perturbed = standardise(adapter.latent(sample.states + noise, sample.aux))
        for k in k_values:
            out[k].append(neighbourhood_overlap(ref[k], knn_indices(perturbed, k)))
    return {k: float(np.nanmean(v)) for k, v in out.items()}


def stability_under_masking(
    adapter: PolicyAdapter,
    sample: Rollout,
    cube: np.ndarray,
    replacement: torch.Tensor,
    k_values: Sequence[int] = K_VALUES,
    mask_fraction: float = 0.20,
) -> Dict[int, float]:
    """Neighbourhood agreement when low-importance feature groups are masked."""
    base = standardise(adapter.latent(sample.states, sample.aux))
    score = np.abs(cube).sum(axis=2).mean(axis=0)
    order = np.argsort(score, kind="stable")  # ascending: least important first
    k_mask = max(1, int(round(mask_fraction * adapter.n_features)))
    masked = sample.states.clone()
    idx = torch.as_tensor(order[:k_mask].copy(), dtype=torch.long)
    masked[:, :, idx] = replacement[:, :, idx]
    pert = standardise(adapter.latent(masked, sample.aux))
    return {
        k: neighbourhood_overlap(knn_indices(base, k), knn_indices(pert, k)) for k in k_values
    }


# ---------------------------------------------------------------------------
# Layer-influence concentration
# ---------------------------------------------------------------------------


def _layer_conductance(
    adapter: PolicyAdapter,
    sample: Rollout,
    baseline: torch.Tensor,
    variant: str,
    n_steps: int = 16,
) -> Optional[Dict[str, float]]:
    from captum.attr import LayerConductance

    reference = adapter.reference(sample.states, sample.aux)
    aux = sample.aux
    batch = len(sample)

    def forward(x: torch.Tensor) -> torch.Tensor:
        reps = max(1, x.shape[0] // batch)
        return adapter.decision_target(
            x, tile_context(aux, reps), tile_context(reference, reps), variant=variant
        )

    values: Dict[str, float] = {}
    for name, module in adapter.layer_modules().items():
        try:
            lc = LayerConductance(forward, module)
            attr = lc.attribute(
                sample.states.to(adapter.device),
                baselines=baseline.to(adapter.device),
                n_steps=n_steps,
                attribute_to_layer_input=True,
            )
            if isinstance(attr, (tuple, list)):
                attr = attr[0]
            values[name] = float(attr.abs().sum().item())
        except Exception:  # noqa: BLE001 - unsupported layer type for this method
            return None
    if not values or sum(values.values()) <= EPS:
        return None
    return values


def _gradient_times_activation(
    adapter: PolicyAdapter, sample: Rollout, variant: str
) -> Optional[Dict[str, float]]:
    reference = adapter.reference(sample.states, sample.aux)
    activations: Dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(name: str):
        def hook(_module, _inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            t.retain_grad()
            activations[name] = t
            return out

        return hook

    modules = adapter.layer_modules()
    for name, module in modules.items():
        handles.append(module.register_forward_hook(make_hook(name)))
    try:
        states = sample.states.clone().to(adapter.device).requires_grad_(True)
        tau = adapter.decision_target(states, sample.aux, reference, variant=variant)
        tau.sum().backward()
        values = {}
        for name in modules:
            act = activations.get(name)
            if act is None or act.grad is None:
                return None
            values[name] = float((act.grad * act).abs().sum().item())
    finally:
        for h in handles:
            h.remove()
    if sum(values.values()) <= EPS:
        return None
    return values


def _layer_ablation(
    adapter: PolicyAdapter, sample: Rollout, variant: str
) -> Dict[str, float]:
    """
    Controlled layer ablation: replace each layer's output by its sample mean and
    record the mean absolute change of the decision target.
    """
    reference = adapter.reference(sample.states, sample.aux)
    with torch.no_grad():
        tau0 = adapter.decision_target(
            sample.states.to(adapter.device), sample.aux, reference, variant=variant
        ).detach()

    values: Dict[str, float] = {}
    for name, module in adapter.layer_modules().items():
        captured: List[torch.Tensor] = []

        def capture(_m, _i, out):
            captured.append((out[0] if isinstance(out, (tuple, list)) else out).detach())
            return out

        h = module.register_forward_hook(capture)
        with torch.no_grad():
            adapter.decision_target(
                sample.states.to(adapter.device), sample.aux, reference, variant=variant
            )
        h.remove()
        if not captured:
            values[name] = 0.0
            continue
        mean_act = captured[0].mean(dim=0, keepdim=True)

        def replace(_m, _i, out):
            if isinstance(out, (tuple, list)):
                return (mean_act.expand_as(out[0]),) + tuple(out[1:])
            return mean_act.expand_as(out)

        h = module.register_forward_hook(replace)
        with torch.no_grad():
            tau = adapter.decision_target(
                sample.states.to(adapter.device), sample.aux, reference, variant=variant
            ).detach()
        h.remove()
        values[name] = float((tau - tau0).abs().mean().item())
    return values


def layer_influence(
    adapter: PolicyAdapter,
    sample: Rollout,
    baseline: torch.Tensor,
    variant: str,
    gradients_usable: bool,
) -> Dict[str, object]:
    """Normalised layer-influence shares and their concentration indices."""
    method = None
    values: Optional[Dict[str, float]] = None
    if gradients_usable:
        values = _layer_conductance(adapter, sample, baseline, variant)
        method = "captum_layer_conductance" if values else None
        if values is None:
            values = _gradient_times_activation(adapter, sample, variant)
            method = "gradient_times_activation" if values else None
    if values is None:
        values = _layer_ablation(adapter, sample, variant)
        method = "controlled_layer_ablation"

    names = list(values)
    raw = np.array([values[n] for n in names], dtype=np.float64)
    total = raw.sum()
    n_layers = len(names)
    # A total influence of zero carries no evidence about how influence is
    # distributed over depth. Reporting uniform shares here would award a perfect
    # dispersion score to a policy whose decision no layer affects, so the metric
    # is left undefined instead.
    measurable = bool(total > EPS) and n_layers > 1
    shares = raw / total if total > EPS else np.full(raw.size, np.nan)
    if measurable:
        raw_hhi = hhi(shares)
        floor = 1.0 / n_layers
        normalised_hhi = float((raw_hhi - floor) / (1.0 - floor))
        entropy = normalised_entropy(shares)
        effective = effective_number(shares)
        max_share = float(shares.max())
        dispersion = float(1.0 - normalised_hhi)
    else:
        raw_hhi = normalised_hhi = entropy = effective = max_share = dispersion = float("nan")
    frame = pd.DataFrame(
        {"layer": names, "raw_influence": raw, "share": shares, "method": method}
    )
    return {
        "method": method,
        "measurable": measurable,
        "frame": frame,
        "shares": shares,
        "metrics": {
            "layer_influence_hhi": raw_hhi,
            "layer_influence_normalised_hhi": normalised_hhi,
            "layer_influence_entropy": entropy,
            "layer_influence_effective_layers": effective,
            "layer_influence_max_share": max_share,
            "non_concentrated_layer_influence": dispersion,
            "layer_influence_total_raw": float(total),
            "n_layers": n_layers,
        },
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_structure(
    adapter: PolicyAdapter,
    sample: Rollout,
    regimes: np.ndarray,
    cube: np.ndarray,
    replacement: torch.Tensor,
    baseline: torch.Tensor,
    variant: str,
    feature_std: np.ndarray,
    gradients_usable: bool,
    *,
    seed: int = 42,
    n_replicates: int = 30,
) -> Dict[str, object]:
    """Compute every structure artefact for one model."""
    latent = adapter.latent(sample.states, sample.aux)
    resample = stability_under_resampling(latent, regimes, seed=seed, n_replicates=n_replicates)
    noise = stability_under_input_noise(adapter, sample, feature_std, seed=seed)
    masking = stability_under_masking(adapter, sample, cube, replacement)

    rows = []
    for k in K_VALUES:
        rows.append(
            {
                "k": k,
                "resampling": resample.get(k, float("nan")),
                "input_noise": noise.get(k, float("nan")),
                "feature_masking": masking.get(k, float("nan")),
            }
        )
    sensitivity = pd.DataFrame(rows)
    sensitivity["mean"] = sensitivity[["resampling", "input_noise", "feature_masking"]].mean(axis=1)

    primary = float(sensitivity.loc[sensitivity["k"] == PRIMARY_K, "mean"].iloc[0])
    layers = layer_influence(adapter, sample, baseline, variant, gradients_usable)

    lat_std = standardise(latent)
    metrics = {
        "latent_neighbourhood_stability": primary,
        "latent_stability_resampling_k10": resample.get(PRIMARY_K, float("nan")),
        "latent_stability_noise_k10": noise.get(PRIMARY_K, float("nan")),
        "latent_stability_masking_k10": masking.get(PRIMARY_K, float("nan")),
        "latent_dimension": int(latent.shape[1]),
        "latent_active_dimension": int(lat_std.shape[1]),
        **layers["metrics"],
    }
    return {
        "model_id": adapter.record.model_id,
        "latent": latent,
        "sensitivity": sensitivity,
        "layer_influence": layers,
        "metrics": metrics,
    }
