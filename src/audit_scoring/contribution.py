"""Contribution diagnostics: feature-family rank and sign stability."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from .config import REGIME_BOUNDARIES, regime_label_for_date
from .feature_groups import family_labels_for_names, parse_indicator_token
from .io import scalar_float, to_numpy


@dataclass
class ContributionResult:
    contribution_rank_stability: float = float("nan")
    contribution_sign_stability: float = float("nan")
    n_attribution_observations: int = 0
    n_feature_families: int = 0
    validity_status: str = "incomplete"
    warnings: List[str] = field(default_factory=list)
    descriptive: Dict[str, Any] = field(default_factory=dict)


def _finite_matrix(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    arr[~np.isfinite(arr)] = np.nan
    return arr


def align_attribution_matrix(
    attributions: Any,
    feature_names: Any,
    dates: Optional[Any] = None,
    labels: Optional[Any] = None,
    validation_start: Optional[pd.Timestamp] = None,
    validation_end: Optional[pd.Timestamp] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[pd.DatetimeIndex], Optional[np.ndarray], Dict[str, Any]]:
    """Return (n_obs, n_features) matrix, names, filtered dates, regime labels, meta."""
    meta: Dict[str, Any] = {
        "earliest_date": None,
        "latest_date": None,
        "n_before_validation": 0,
        "n_after_validation": 0,
        "temporal_status": "unavailable",
    }
    attr = to_numpy(attributions)
    names_raw = to_numpy(feature_names)
    names_flat = names_raw.reshape(-1).astype(str)

    if attr.size == 0:
        return np.zeros((0, 0)), np.asarray([], dtype=str), None, None, meta

    # Common layout: (n_obs * n_actions, *feature_dims)
    feature_volume = int(np.prod(names_raw.shape)) if names_raw.size else int(np.prod(attr.shape[1:]))
    if feature_volume <= 0:
        feature_volume = int(np.prod(attr.shape[1:])) if attr.ndim > 1 else attr.shape[0]

    if attr.ndim == 1:
        matrix = attr.reshape(1, -1)
    elif attr.ndim == 2:
        matrix = attr
    else:
        # Collapse trailing feature dims.
        leading = attr.shape[0]
        matrix = attr.reshape(leading, -1)

    if matrix.shape[1] != len(names_flat):
        if matrix.shape[1] > len(names_flat) and matrix.shape[1] % max(len(names_flat), 1) == 0:
            # Unexpected; truncate.
            matrix = matrix[:, : len(names_flat)]
        elif len(names_flat) > matrix.shape[1]:
            names_flat = names_flat[: matrix.shape[1]]
        else:
            # pad names
            pad = [f"feature_{i}" for i in range(matrix.shape[1] - len(names_flat))]
            names_flat = np.concatenate([names_flat, np.asarray(pad, dtype=str)])

    date_index: Optional[pd.DatetimeIndex] = None
    regime = None
    if dates is not None:
        date_index = pd.DatetimeIndex(pd.to_datetime(to_numpy(dates).reshape(-1)))
        meta["earliest_date"] = str(date_index.min().date()) if len(date_index) else None
        meta["latest_date"] = str(date_index.max().date()) if len(date_index) else None
        meta["temporal_status"] = "verified"

        n_obs_dates = len(date_index)
        if matrix.shape[0] == n_obs_dates:
            pass
        elif n_obs_dates > 0 and matrix.shape[0] % n_obs_dates == 0:
            n_actions = matrix.shape[0] // n_obs_dates
            matrix = matrix.reshape(n_obs_dates, n_actions, matrix.shape[1]).mean(axis=1)
            if labels is not None:
                lab = to_numpy(labels).reshape(-1)
                if lab.shape[0] == n_obs_dates * n_actions:
                    lab = lab.reshape(n_obs_dates, n_actions)[:, 0]
                regime = lab.astype(str)
        else:
            meta["temporal_status"] = "misaligned"
            meta["warnings"] = ["Attribution rows could not be aligned to dates"]

        if validation_start is not None and validation_end is not None and matrix.shape[0] == len(date_index):
            before = date_index < validation_start
            after = date_index > validation_end
            meta["n_before_validation"] = int(before.sum())
            meta["n_after_validation"] = int(after.sum())
            keep = (~before) & (~after)
            if int((~keep).sum()) > 0:
                matrix = matrix[keep]
                date_index = date_index[keep]
                if regime is not None and len(regime) == len(keep):
                    regime = np.asarray(regime)[keep]
            elif meta["n_after_validation"] > 0:
                meta["temporal_status"] = "invalid_unfilterable"

    if regime is None and date_index is not None and len(date_index) == matrix.shape[0]:
        regime = np.asarray([regime_label_for_date(ts) for ts in date_index], dtype=str)

    if labels is not None and regime is None:
        lab = to_numpy(labels).reshape(-1).astype(str)
        if lab.shape[0] == matrix.shape[0]:
            regime = lab
        elif matrix.shape[0] > 0 and lab.shape[0] % matrix.shape[0] == 0:
            n_actions = lab.shape[0] // matrix.shape[0]
            regime = lab.reshape(matrix.shape[0], n_actions)[:, 0]

    return _finite_matrix(matrix), names_flat.astype(str), date_index, regime, meta


def _family_profiles(
    matrix: np.ndarray,
    feature_names: Sequence[str],
    signed: bool,
) -> pd.Series:
    families = family_labels_for_names(list(feature_names))
    values = np.nanmean(matrix if signed else np.abs(matrix), axis=0)
    df = pd.DataFrame({"family": families, "value": values})
    return df.groupby("family", sort=True)["value"].mean()


def _kendall_concordance(rank_a: Sequence[str], rank_b: Sequence[str]) -> float:
    labels = list(dict.fromkeys(list(rank_a) + list(rank_b)))
    if len(labels) < 2:
        return float("nan")
    pos_a = {k: i for i, k in enumerate(rank_a)}
    pos_b = {k: i for i, k in enumerate(rank_b)}
    # Use presence ranks; missing get large rank
    ra = np.asarray([pos_a.get(k, len(labels)) for k in labels], dtype=float)
    rb = np.asarray([pos_b.get(k, len(labels)) for k in labels], dtype=float)
    if np.allclose(ra, ra[0]) or np.allclose(rb, rb[0]):
        return float("nan")
    tau, _ = kendalltau(ra, rb)
    return float(tau) if np.isfinite(tau) else float("nan")


def _rank_families(profile: pd.Series) -> List[str]:
    return list(profile.sort_values(ascending=False).index.astype(str))


def _family_profiles_from_index(
    matrix: np.ndarray,
    family_ids: np.ndarray,
    unique_families: Sequence[str],
    signed: bool,
) -> pd.Series:
    values = np.nanmean(matrix if signed else np.abs(matrix), axis=0)
    out = {}
    for i, fam in enumerate(unique_families):
        mask = family_ids == i
        if not np.any(mask):
            out[fam] = 0.0
        else:
            out[fam] = float(np.nanmean(values[mask]))
    return pd.Series(out)


def compute_contribution_diagnostics(
    attributions: Any,
    feature_names: Any,
    *,
    dates: Any = None,
    labels: Any = None,
    bootstrap_replicates: int = 100,
    random_seed: int = 42,
    top_k: int = 5,
    min_observations: int = 30,
    min_feature_families: int = 2,
    validation_start: Optional[pd.Timestamp] = None,
    validation_end: Optional[pd.Timestamp] = None,
) -> ContributionResult:
    result = ContributionResult()
    matrix, names, date_index, regime, meta = align_attribution_matrix(
        attributions,
        feature_names,
        dates=dates,
        labels=labels,
        validation_start=validation_start,
        validation_end=validation_end,
    )
    result.descriptive["temporal_meta"] = meta
    if meta.get("warnings"):
        result.warnings.extend(meta["warnings"])

    if matrix.size == 0 or matrix.shape[0] == 0:
        result.validity_status = "failed"
        result.warnings.append("Empty attribution matrix")
        return result

    if not np.isfinite(matrix).any():
        result.validity_status = "failed"
        result.warnings.append("No finite attribution values")
        return result

    # Drop non-finite rows
    row_ok = np.isfinite(matrix).any(axis=1)
    matrix = matrix[row_ok]
    if regime is not None and len(regime) == row_ok.shape[0]:
        regime = regime[row_ok]
    if date_index is not None and len(date_index) == row_ok.shape[0]:
        date_index = date_index[row_ok]

    result.n_attribution_observations = int(matrix.shape[0])
    if result.n_attribution_observations < min_observations:
        result.validity_status = "failed"
        result.warnings.append(
            f"Insufficient attribution observations ({result.n_attribution_observations})"
        )
        return result

    var = np.nanvar(matrix)
    if not np.isfinite(var) or var <= 0:
        result.validity_status = "failed"
        result.warnings.append("Zero attribution variance")
        return result

    # Contribution ranking uses indicator tokens (Open/Close/RSI/...), which remain
    # comparable within TI and no-TI models without equating raw feature counts.
    family_names = [parse_indicator_token(n) for n in names]
    unique_families = sorted(set(family_names))
    # Keep coarse taxonomy only as descriptive evidence.
    coarse = family_labels_for_names(list(names))
    result.descriptive["coarse_taxonomy_family_count"] = len(set(coarse))
    family_ids = np.asarray([unique_families.index(f) for f in family_names], dtype=int)

    abs_full = _family_profiles_from_index(matrix, family_ids, unique_families, signed=False)
    signed_full = _family_profiles_from_index(matrix, family_ids, unique_families, signed=True)
    result.n_feature_families = int(len(abs_full))
    if result.n_feature_families < min_feature_families:
        result.validity_status = "failed"
        result.warnings.append("Insufficient feature families")
        return result

    full_rank = _rank_families(abs_full)
    k = min(top_k, len(signed_full))
    top_families = list(signed_full.abs().sort_values(ascending=False).index[:k].astype(str))
    full_signs = {fam: float(np.sign(signed_full.loc[fam])) for fam in top_families}

    rng = np.random.default_rng(random_seed)
    n = matrix.shape[0]
    use_stratified = regime is not None and len(np.unique(regime)) >= 2
    if not use_stratified:
        result.warnings.append("Regime labels unavailable; using ordinary bootstrap")

    rank_scores: List[float] = []
    sign_scores: List[float] = []

    strata: Dict[str, np.ndarray] = {}
    if use_stratified:
        for label in np.unique(regime):
            strata[str(label)] = np.flatnonzero(regime == label)

    for _ in range(bootstrap_replicates):
        if use_stratified:
            parts = []
            for idxs in strata.values():
                if len(idxs) == 0:
                    continue
                draw = rng.choice(idxs, size=len(idxs), replace=True)
                parts.append(draw)
            sample_idx = np.concatenate(parts) if parts else rng.integers(0, n, size=n)
        else:
            sample_idx = rng.integers(0, n, size=n)
        sample = matrix[sample_idx]
        abs_prof = _family_profiles_from_index(sample, family_ids, unique_families, signed=False)
        signed_prof = _family_profiles_from_index(sample, family_ids, unique_families, signed=True)
        boot_rank = _rank_families(abs_prof)
        tau = _kendall_concordance(full_rank, boot_rank)
        if np.isfinite(tau):
            rank_scores.append(tau)

        if use_stratified:
            agreements = []
            weights = []
            for label, idxs in strata.items():
                if len(idxs) < 2:
                    continue
                sampled_in = sample_idx[np.isin(sample_idx, idxs)]
                if sampled_in.size == 0:
                    continue
                sub_signed = _family_profiles_from_index(
                    matrix[sampled_in], family_ids, unique_families, signed=True
                )
                local = []
                for fam in top_families:
                    if fam not in sub_signed.index:
                        continue
                    full_sign = full_signs[fam]
                    if full_sign == 0:
                        continue
                    local.append(float(np.sign(sub_signed.loc[fam]) == full_sign))
                if local:
                    agreements.append(float(np.mean(local)))
                    weights.append(float(len(idxs)))
            if agreements:
                sign_scores.append(float(np.average(agreements, weights=weights)))
        else:
            local = []
            for fam in top_families:
                full_sign = full_signs[fam]
                if full_sign == 0:
                    continue
                local.append(float(np.sign(signed_prof.loc[fam]) == full_sign))
            if local:
                sign_scores.append(float(np.mean(local)))

    result.contribution_rank_stability = float(np.mean(rank_scores)) if rank_scores else float("nan")
    result.contribution_sign_stability = float(np.mean(sign_scores)) if sign_scores else float("nan")

    total_abs = float(np.nansum(np.abs(matrix)))
    pos = float(np.nansum(matrix[matrix > 0])) if np.any(matrix > 0) else 0.0
    neg = float(np.nansum(-matrix[matrix < 0])) if np.any(matrix < 0) else 0.0
    result.descriptive.update(
        {
            "top_feature_families": top_families,
            "absolute_attribution_concentration": float(
                abs_full.max() / abs_full.sum() if abs_full.sum() else np.nan
            ),
            "positive_attribution_share": pos / total_abs if total_abs else np.nan,
            "negative_attribution_share": neg / total_abs if total_abs else np.nan,
            "family_abs_profile": abs_full.to_dict(),
            "family_signed_profile": signed_full.to_dict(),
        }
    )

    if np.isfinite(result.contribution_rank_stability) and np.isfinite(
        result.contribution_sign_stability
    ):
        result.validity_status = "passed"
    else:
        result.validity_status = "incomplete"
        result.warnings.append("One or more contribution core metrics are missing")
    return result
