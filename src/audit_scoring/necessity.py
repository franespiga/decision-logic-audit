"""Necessity diagnostics based on temporal occlusion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .io import scalar_float, to_numpy


@dataclass
class NecessityResult:
    necessity_temporal_profile_stability: float = float("nan")
    necessity_temporal_instability: float = float("nan")
    necessity_normalized_occlusion_effect: float = float("nan")
    necessity_non_degeneracy_pass: bool = False
    n_occlusion_samples: int = 0
    validity_status: str = "incomplete"
    warnings: List[str] = field(default_factory=list)
    descriptive: Dict[str, Any] = field(default_factory=dict)


def _cv_value(value: Any) -> float:
    x = scalar_float(value)
    if not np.isfinite(x) or x < 0:
        return float("nan")
    return float(x)


def compute_normalized_occlusion_effect(
    occlusion_curves: Any,
    baseline_output: Any,
    eps: float = 1e-8,
) -> float:
    """Median over samples and lags of ||occluded - baseline|| / (||baseline|| + eps)."""
    curves = to_numpy(occlusion_curves).astype(float)
    baseline = to_numpy(baseline_output).astype(float)
    if curves.size == 0 or baseline.size == 0:
        return float("nan")

    # Expected shapes: curves (n, assets, lags) or (n, lags, assets); baseline (n, assets)
    if curves.ndim == 2:
        # (n, lags) with scalar/1d action — treat as (n, 1, lags)
        curves = curves[:, None, :]
    if baseline.ndim == 1:
        baseline = baseline[:, None]

    if curves.ndim != 3:
        curves = curves.reshape(curves.shape[0], -1, curves.shape[-1])

    n = min(curves.shape[0], baseline.shape[0])
    curves = curves[:n]
    baseline = baseline[:n]

    # Align asset axis: prefer matching baseline dim 1 with one of curves dims.
    if curves.shape[1] != baseline.shape[1] and curves.shape[2] == baseline.shape[1]:
        curves = np.transpose(curves, (0, 2, 1))

    base_norm = np.linalg.norm(baseline, axis=1)  # (n,)
    # curves: (n, assets, lags)
    diffs = curves - baseline[:, :, None]
    diff_norm = np.linalg.norm(diffs, axis=1)  # (n, lags)
    denom = base_norm[:, None] + float(eps)
    ratios = diff_norm / denom
    ratios = ratios[np.isfinite(ratios)]
    if ratios.size == 0:
        return float("nan")
    return float(np.median(ratios))


def compute_necessity_diagnostics(
    occlusion_results: Any,
    rolling_stability: Any = None,
    *,
    eps: float = 1e-8,
    nondegeneracy_tol: float = 1e-6,
) -> NecessityResult:
    result = NecessityResult()
    occ = occlusion_results if isinstance(occlusion_results, dict) else {}
    roll = rolling_stability if isinstance(rolling_stability, dict) else {}

    # Prefer rolling_stability nested inside occlusion_results when present.
    if not roll and isinstance(occ.get("rolling_stability"), dict):
        roll = occ["rolling_stability"]

    stability_metrics = roll.get("stability_metrics", {}) if isinstance(roll, dict) else {}
    if isinstance(stability_metrics, dict):
        result.necessity_temporal_profile_stability = _cv_value(
            stability_metrics.get("mean_window_correlation")
        )
        cvs = [
            _cv_value(stability_metrics.get("curve_coefficient_of_variation")),
            _cv_value(stability_metrics.get("half_life_coefficient_of_variation")),
            _cv_value(stability_metrics.get("auc_coefficient_of_variation")),
        ]
        valid_cvs = [c for c in cvs if np.isfinite(c)]
        if len(valid_cvs) >= 2:
            result.necessity_temporal_instability = float(np.median(valid_cvs))
        else:
            result.warnings.append(
                "Fewer than two valid CV components for temporal instability"
            )
            result.descriptive["cv_components"] = cvs
    else:
        result.warnings.append("rolling_stability.stability_metrics unavailable")

    curves = occ.get("occlusion_curves")
    baseline = occ.get("baseline_output")
    if curves is not None and baseline is not None:
        curves_np = to_numpy(curves)
        baseline_np = to_numpy(baseline)
        result.n_occlusion_samples = int(curves_np.shape[0]) if curves_np.ndim >= 1 else 0
        effect = compute_normalized_occlusion_effect(curves_np, baseline_np, eps=eps)
        result.necessity_normalized_occlusion_effect = effect
        identical = False
        try:
            # Compare a reduced slice for identity check without huge memory blowups.
            if curves_np.ndim == 3 and baseline_np.ndim == 2:
                n = min(curves_np.shape[0], baseline_np.shape[0], 64)
                if curves_np.shape[1] == baseline_np.shape[1]:
                    sample = curves_np[:n] - baseline_np[:n, :, None]
                elif curves_np.shape[2] == baseline_np.shape[1]:
                    sample = curves_np[:n] - baseline_np[:n, None, :]
                else:
                    sample = curves_np[:n] - np.nan
                identical = bool(np.nanmax(np.abs(sample)) <= nondegeneracy_tol)
            else:
                identical = bool(
                    np.allclose(
                        curves_np.reshape(curves_np.shape[0], -1)[:8],
                        np.broadcast_to(
                            baseline_np.reshape(baseline_np.shape[0], -1)[:8, :1],
                            (min(8, curves_np.shape[0]), 1),
                        ),
                        atol=nondegeneracy_tol,
                        equal_nan=True,
                    )
                )
        except Exception:
            identical = False

        result.necessity_non_degeneracy_pass = bool(
            np.isfinite(effect) and effect > nondegeneracy_tol and not identical and result.n_occlusion_samples > 0
        )
        if not result.necessity_non_degeneracy_pass:
            result.warnings.append("Occlusion non-degeneracy gate failed or effect too small")
    else:
        result.warnings.append("occlusion_curves/baseline_output unavailable")

    # Optional descriptive AUC / half-life summaries
    auc = occ.get("auc_results")
    if isinstance(auc, dict):
        result.descriptive["auc_results_keys"] = list(auc.keys())
    if isinstance(stability_metrics, dict):
        result.descriptive["mean_auc_across_windows"] = scalar_float(
            stability_metrics.get("mean_auc_across_windows")
        )
        result.descriptive["mean_half_life_across_windows"] = scalar_float(
            stability_metrics.get("mean_half_life_across_windows")
        )

    core_ok = (
        np.isfinite(result.necessity_temporal_profile_stability)
        and np.isfinite(result.necessity_temporal_instability)
        and result.necessity_non_degeneracy_pass
    )
    if core_ok:
        result.validity_status = "passed"
    elif (
        np.isfinite(result.necessity_temporal_profile_stability)
        or np.isfinite(result.necessity_temporal_instability)
    ):
        result.validity_status = "incomplete"
    else:
        result.validity_status = "failed"
    return result
