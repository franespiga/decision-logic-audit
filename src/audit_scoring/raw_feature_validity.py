"""Raw-feature and asset-output validity summaries (Batch C2)."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from .baseline_validity import ranking_from_abs_profile
from .targeted_validity import compare_rank_maps, raw_feature_ranks


def asset_token_from_feature(name: str) -> str:
    text = str(name)
    if text.startswith("^") or text.split("_")[0].isupper():
        return text.split("_")[0]
    return "unknown"


def summarize_asset_output_attributions(
    matrix: np.ndarray,
    feature_names: np.ndarray,
    *,
    model_family: str,
    target_name: str,
) -> pd.DataFrame:
    """Aggregate abs attributions by asset ticker prefix."""
    values = np.nanmean(np.abs(matrix), axis=0)
    signed = np.nanmean(matrix, axis=0)
    rows = []
    assets = [asset_token_from_feature(n) for n in feature_names]
    for asset in sorted(set(assets)):
        mask = np.asarray([a == asset for a in assets])
        rows.append(
            {
                "model_family": model_family,
                "target_name": target_name,
                "asset": asset,
                "mean_abs_attribution": float(np.nanmean(values[mask])),
                "mean_signed_attribution": float(np.nanmean(signed[mask])),
                "n_features": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_abs_attribution", ascending=False)


def resolution_comparison(
    *,
    model_family: str,
    family_abs: pd.Series,
    family_abs_rand: pd.Series,
    raw_abs_ranks: pd.DataFrame,
    raw_abs_ranks_rand: pd.DataFrame,
    top_k: int = 20,
) -> Dict[str, Any]:
    fam_comp = compare_rank_maps(family_abs, family_abs_rand, family_abs, family_abs_rand, top_k=5)
    # raw: align by feature name
    a = raw_abs_ranks.set_index("feature_name")["abs_attribution"]
    b = raw_abs_ranks_rand.set_index("feature_name")["abs_attribution"]
    labels = sorted(set(a.index) | set(b.index))
    xa = a.reindex(labels).fillna(0.0).to_numpy()
    xb = b.reindex(labels).fillna(0.0).to_numpy()
    tau = kendalltau(xa, xb, variant="b").statistic
    top_a = set(raw_abs_ranks.sort_values("abs_attribution", ascending=False).head(top_k)["feature_name"])
    top_b = set(raw_abs_ranks_rand.sort_values("abs_attribution", ascending=False).head(top_k)["feature_name"])
    jacc = len(top_a & top_b) / max(len(top_a | top_b), 1)
    stickiness_family = float(fam_comp["kendall_tau_b"]) if np.isfinite(fam_comp["kendall_tau_b"]) else float("nan")
    stickiness_raw = float(tau) if np.isfinite(tau) else float("nan")
    status = "UNSUPPORTED" if (stickiness_family > 0.4 and stickiness_raw > 0.4) else (
        "SUPPORTED" if (stickiness_family <= 0.2 or stickiness_raw <= 0.2) else "MIXED"
    )
    return {
        "model_family": model_family,
        "family_randomization_kendall": stickiness_family,
        "raw_randomization_kendall": stickiness_raw,
        "raw_top_k_jaccard": float(jacc),
        "raw_feature_randomization_status": status,
        "notes": "Descriptive multi-resolution comparison; not a multiple-testing corrected claim.",
    }
