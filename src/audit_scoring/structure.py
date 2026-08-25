"""Structure diagnostics: effective rank and linear-probe margin."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .io import scalar_float, to_numpy


@dataclass
class StructureResult:
    structure_effective_rank_ratio: float = float("nan")
    structure_probe_margin: float = float("nan")
    structure_probe_cv_std: float = float("nan")
    n_latent_samples: int = 0
    validity_status: str = "incomplete"
    warnings: List[str] = field(default_factory=list)
    descriptive: Dict[str, Any] = field(default_factory=dict)


def _layer_index(name: str) -> int:
    text = str(name)
    digits = "".join(ch if ch.isdigit() else " " for ch in text).split()
    if not digits:
        return -1
    return int(digits[-1])


def deepest_layer_key(mapping: Dict[str, Any]) -> Optional[str]:
    if not mapping:
        return None
    scored = sorted(mapping.keys(), key=lambda k: (_layer_index(k), str(k)))
    return scored[-1]


def effective_rank_ratio(representation: Any) -> Tuple[float, int]:
    """Normalized effective rank of a representation matrix."""
    mat = to_numpy(representation).astype(float)
    if mat.ndim > 2:
        # (n, ..., d) -> (n, d)
        mat = mat.reshape(mat.shape[0], -1)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    # Drop non-finite rows
    finite_rows = np.isfinite(mat).all(axis=1)
    mat = mat[finite_rows]
    n_samples = int(mat.shape[0])
    if n_samples < 2 or mat.shape[1] < 1:
        return float("nan"), n_samples
    mat = mat - np.nanmean(mat, axis=0, keepdims=True)
    # SVD on centered matrix
    try:
        singular = np.linalg.svd(mat, compute_uv=False)
    except np.linalg.LinAlgError:
        return float("nan"), n_samples
    singular = singular[np.isfinite(singular) & (singular > 0)]
    if singular.size == 0:
        return float("nan"), n_samples
    power = singular ** 2
    probs = power / power.sum()
    probs = probs[probs > 0]
    eff = float(np.exp(-np.sum(probs * np.log(probs))))
    denom = float(min(n_samples, mat.shape[1]))
    if denom <= 0:
        return float("nan"), n_samples
    return eff / denom, n_samples


def majority_class_rate(labels: Any) -> float:
    arr = to_numpy(labels).reshape(-1)
    if arr.size == 0:
        return float("nan")
    # Align possible repeated labels by taking unique? Use as-is frequencies.
    values, counts = np.unique(arr.astype(str), return_counts=True)
    if counts.size == 0:
        return float("nan")
    return float(counts.max() / counts.sum())


def compute_structure_diagnostics(
    latent_representations: Any = None,
    layer_representations: Any = None,
    linear_probe_results: Any = None,
    separability_metrics: Any = None,
    regime_labels: Any = None,
) -> StructureResult:
    result = StructureResult()

    hidden = None
    if isinstance(latent_representations, dict) and latent_representations.get("hidden") is not None:
        hidden = latent_representations.get("hidden")
        if regime_labels is None:
            regime_labels = latent_representations.get("labels")
    elif isinstance(layer_representations, dict) and layer_representations:
        key = deepest_layer_key(layer_representations)
        hidden = layer_representations.get(key) if key else None
        result.descriptive["representation_source"] = key
    else:
        result.warnings.append("No latent/layer representation available")

    if hidden is not None:
        ratio, n_samples = effective_rank_ratio(hidden)
        result.structure_effective_rank_ratio = ratio
        result.n_latent_samples = n_samples
    else:
        result.warnings.append("Effective-rank representation missing")

    probe = linear_probe_results if isinstance(linear_probe_results, dict) else {}
    if probe:
        key = deepest_layer_key(probe)
        deepest = probe.get(key, {}) if key else {}
        result.descriptive["probe_layer"] = key
        cv_mean = scalar_float(deepest.get("cv_accuracy_mean"))
        cv_std = scalar_float(deepest.get("cv_accuracy_std"))
        result.structure_probe_cv_std = cv_std
        if regime_labels is None:
            result.warnings.append(
                "Regime labels unavailable for majority baseline; probe margin marked missing"
            )
        else:
            # Align labels length to latent samples when possible
            labels = to_numpy(regime_labels).reshape(-1)
            if result.n_latent_samples > 0 and labels.size != result.n_latent_samples:
                if labels.size % result.n_latent_samples == 0:
                    n_actions = labels.size // result.n_latent_samples
                    labels = labels.reshape(result.n_latent_samples, n_actions)[:, 0]
                elif result.n_latent_samples % labels.size == 0:
                    # labels shorter: leave as-is for majority rate
                    pass
                else:
                    result.warnings.append("Probe labels could not be aligned; margin missing")
                    labels = np.asarray([])
            baseline = majority_class_rate(labels) if labels.size else float("nan")
            result.descriptive["majority_class_rate"] = baseline
            if np.isfinite(cv_mean) and np.isfinite(baseline):
                result.structure_probe_margin = float(cv_mean - baseline)
            else:
                result.warnings.append("Probe margin could not be computed")
    else:
        result.warnings.append("linear_probe_results unavailable")

    # Optional separability descriptive
    if isinstance(separability_metrics, dict) and separability_metrics:
        key = deepest_layer_key(separability_metrics)
        deepest_sep = separability_metrics.get(key, {}) if key else {}
        if isinstance(deepest_sep, dict):
            result.descriptive["silhouette_score"] = scalar_float(
                deepest_sep.get("silhouette_score")
            )
            result.descriptive["separation_ratio"] = scalar_float(
                deepest_sep.get("separation_ratio")
            )
            result.descriptive["calinski_harabasz_score"] = scalar_float(
                deepest_sep.get("calinski_harabasz_score")
            )

    if np.isfinite(result.structure_effective_rank_ratio) and np.isfinite(
        result.structure_probe_margin
    ):
        result.validity_status = "passed"
    elif np.isfinite(result.structure_effective_rank_ratio) or np.isfinite(
        result.structure_probe_margin
    ):
        result.validity_status = "incomplete"
    else:
        result.validity_status = "failed"
    return result
