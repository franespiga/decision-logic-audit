"""Family-level aggregation of per-model raw diagnostics."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import (
    CORE_CONTRIBUTION_METRICS,
    CORE_NECESSITY_METRICS,
    CORE_STRUCTURE_METRICS,
)


CORE_METRICS = (
    list(CORE_CONTRIBUTION_METRICS)
    + list(CORE_NECESSITY_METRICS)
    + list(CORE_STRUCTURE_METRICS)
)


def _iqr(series: pd.Series) -> float:
    x = pd.to_numeric(series, errors="coerce").dropna()
    if x.empty:
        return float("nan")
    return float(x.quantile(0.75) - x.quantile(0.25))


def _median(series: pd.Series) -> float:
    x = pd.to_numeric(series, errors="coerce").dropna()
    if x.empty:
        return float("nan")
    return float(x.median())


def _q(series: pd.Series, q: float) -> float:
    x = pd.to_numeric(series, errors="coerce").dropna()
    if x.empty:
        return float("nan")
    return float(x.quantile(q))


def aggregate_family_raw(
    model_raw: pd.DataFrame,
    expected_families: Sequence[str],
    expected_counts: Mapping[str, int],
    *,
    dispersion_rel_iqr_threshold: float = 0.75,
) -> pd.DataFrame:
    """Aggregate raw per-model metrics to family medians and IQRs."""
    rows = []
    for family in expected_families:
        sub = model_raw[model_raw["model_family"].astype(str) == family]
        n_available = int(sub["model_id"].nunique()) if not sub.empty else 0
        n_expected = int(expected_counts.get(family, 1))
        coverage = float(n_available / n_expected) if n_expected > 0 else float("nan")
        row: Dict[str, object] = {
            "model_family": family,
            "n_models_expected": n_expected,
            "n_models_available": n_available,
            "model_coverage_ratio": coverage,
            "missing_model_count": max(n_expected - n_available, 0),
        }
        warnings = []
        for metric in CORE_METRICS:
            if metric in sub.columns and not sub.empty:
                series = sub[metric]
                med = _median(series)
                q1 = _q(series, 0.25)
                q3 = _q(series, 0.75)
                iqr = _iqr(series)
                n_valid = int(pd.to_numeric(series, errors="coerce").notna().sum())
            else:
                med = q1 = q3 = iqr = float("nan")
                n_valid = 0
            row[f"median_{metric}"] = med
            row[f"q1_{metric}"] = q1
            row[f"q3_{metric}"] = q3
            row[f"iqr_{metric}"] = iqr
            row[f"n_valid_{metric}"] = n_valid

        # Dimension IQR summaries (mean of component IQRs)
        def _dim_iqr(metrics: Sequence[str]) -> float:
            vals = [row[f"iqr_{m}"] for m in metrics]
            vals = [float(v) for v in vals if np.isfinite(v)]
            return float(np.mean(vals)) if vals else float("nan")

        c_iqr = _dim_iqr(CORE_CONTRIBUTION_METRICS)
        n_iqr = _dim_iqr(CORE_NECESSITY_METRICS)
        s_iqr = _dim_iqr(CORE_STRUCTURE_METRICS)
        row["contribution_iqr_summary"] = c_iqr
        row["necessity_iqr_summary"] = n_iqr
        row["structure_iqr_summary"] = s_iqr

        # High dispersion: relative IQR large vs |median| for any core metric with n>1
        high_disp = False
        if n_available > 1:
            for metric in CORE_METRICS:
                med = row[f"median_{metric}"]
                iqr = row[f"iqr_{metric}"]
                if np.isfinite(med) and np.isfinite(iqr) and abs(float(med)) > 1e-12:
                    if float(iqr) / abs(float(med)) >= dispersion_rel_iqr_threshold:
                        high_disp = True
                        warnings.append(f"high_dispersion:{metric}")
                        break
                elif np.isfinite(iqr) and float(iqr) >= dispersion_rel_iqr_threshold:
                    # for metrics near zero, use absolute IQR threshold on [0,1]-ish metrics
                    if float(iqr) >= 0.25:
                        high_disp = True
                        warnings.append(f"high_dispersion:{metric}")
                        break
        row["high_within_family_dispersion"] = bool(high_disp)

        validity = "passed"
        if n_available == 0:
            validity = "failed"
            warnings.append("no_models_available")
        elif any(
            not np.isfinite(row[f"median_{m}"]) for m in CORE_METRICS
        ):
            validity = "incomplete"
            warnings.append("missing_core_family_median")
        row["validity_status"] = validity
        row["warnings"] = "; ".join(warnings)
        rows.append(row)
    return pd.DataFrame(rows)
