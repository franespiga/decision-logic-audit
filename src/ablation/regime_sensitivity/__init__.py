"""
Regime-definition sensitivity analysis helpers for the ESWA major revision.

Used by notebooks/1_ablation_walkthrough_regime_sensitivity.ipynb.
Does not alter the baseline scientific pipeline; only provides alternative
labelling specifications and comparative metrics.
"""
from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split

try:
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
except Exception as exc:  # pragma: no cover
    MarkovRegression = None
    _MARKOV_IMPORT_ERROR = exc
else:
    _MARKOV_IMPORT_ERROR = None

EVENT_REGIME_ORDER = (
    "bull_market",
    "crash_period",
    "recovery_growth",
    "sideways_uncertain",
)

# Match the notebook AnalysisConfig / PAPER_REGIMES schedule.
BASELINE_BOUNDARIES: Dict[str, Tuple[str, str]] = {
    "bull_market": ("2016-01-01", "2020-02-15"),
    "crash_period": ("2020-02-16", "2020-05-31"),
    "recovery_growth": ("2020-06-01", "2021-12-31"),
    "sideways_uncertain": ("2022-01-01", "2024-05-30"),
}

MARKOV_LABELS = ("high_volatility", "low_volatility")

SPEC_DISPLAY = {
    "baseline_event": "Baseline event",
    "boundary_minus_20td": "Boundaries $-$20 td",
    "boundary_plus_20td": "Boundaries $+$20 td",
    "conlon_markov_2state": "Conlon-inspired Markov (2-state)",
}

EVENT_COLORS = {
    "bull_market": "#2ca02c",
    "crash_period": "#d62728",
    "recovery_growth": "#1f77b4",
    "sideways_uncertain": "#ff7f0e",
    "high_volatility": "#d62728",
    "low_volatility": "#2ca02c",
    "Unknown": "#7f7f7f",
}


def _as_dtindex(trading_index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(trading_index))
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    return idx.sort_values().unique()


def define_event_regimes(
    trading_index,
    boundaries: Mapping[str, Tuple[str, str]] = BASELINE_BOUNDARIES,
    shift_trading_days: int = 0,
) -> pd.Series:
    """
    Assign mutually exclusive event-anchored regime labels on a trading calendar.

    Internal transition boundaries (bull→crash, crash→recovery, recovery→sideways)
    are shifted by ``shift_trading_days`` trading observations. The overall
    diagnostic window start/end remain fixed to the first/last dates in
    ``trading_index``.
    """
    idx = _as_dtindex(trading_index)
    if len(idx) == 0:
        return pd.Series(dtype=object)

    names = list(EVENT_REGIME_ORDER)
    # Anchor internal ends to the last trading day on/before the calendar end.
    raw_ends = [pd.Timestamp(boundaries[n][1]) for n in names[:-1]]
    positions = []
    for end in raw_ends:
        pos = int(idx.searchsorted(end, side="right") - 1)
        positions.append(pos)

    shifted = []
    for pos in positions:
        new_pos = int(pos + shift_trading_days)
        new_pos = max(0, min(len(idx) - 2, new_pos))
        shifted.append(new_pos)

    # Enforce strictly increasing cut points so periods remain contiguous.
    for i in range(1, len(shifted)):
        if shifted[i] <= shifted[i - 1]:
            shifted[i] = min(len(idx) - 2, shifted[i - 1] + 1)
    if shifted[-1] >= len(idx) - 1:
        # leave at least one observation for the final regime
        overflow = shifted[-1] - (len(idx) - 2)
        for i in range(len(shifted)):
            shifted[i] = max(0, shifted[i] - overflow)

    cuts = [-1] + shifted + [len(idx) - 1]
    labels = np.empty(len(idx), dtype=object)
    for r, name in enumerate(names):
        start_i = cuts[r] + 1
        end_i = cuts[r + 1]
        labels[start_i : end_i + 1] = name

    series = pd.Series(labels, index=idx, name="regime")
    assert series.isna().sum() == 0, "event regimes left NaN labels"
    assert set(series.unique()) <= set(names), set(series.unique())
    # mutual exclusivity / exhaustiveness on the provided index
    assert len(series) == len(idx)
    return series


def assert_event_partition(labels: pd.Series, expected: Sequence[str] = EVENT_REGIME_ORDER) -> None:
    expected = list(expected)
    assert labels.isna().sum() == 0
    assert (labels == "Unknown").sum() == 0, "Unknown labels remain among eligible observations"
    vals = set(labels.astype(str))
    assert vals <= set(expected), f"unexpected labels: {vals - set(expected)}"
    assert len(labels) == labels.index.nunique(), "duplicate dates in regime series"
    # no overlapping intervals: by construction one label per date
    vc = labels.value_counts()
    assert vc.sum() == len(labels)


def count_contiguous_episodes(labels: pd.Series) -> Dict[str, int]:
    out: Dict[str, int] = {}
    arr = labels.astype(str).to_numpy()
    for regime in pd.unique(arr):
        mask = arr == regime
        if not mask.any():
            out[regime] = 0
            continue
        prev = np.concatenate([[False], mask[:-1]])
        out[regime] = int(np.sum(mask & ~prev))
    return out


def max_drawdown(returns: pd.Series) -> float:
    r = returns.dropna().astype(float)
    if len(r) == 0:
        return np.nan
    wealth = (1.0 + r).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    return float(dd.min()) if len(dd) else np.nan


def annualised_return(returns: pd.Series, ppy: int = 252) -> float:
    r = returns.dropna().astype(float)
    if len(r) == 0:
        return np.nan
    growth = float((1.0 + r).prod())
    return growth ** (ppy / len(r)) - 1.0


def annualised_vol(returns: pd.Series, ppy: int = 252) -> float:
    r = returns.dropna().astype(float)
    if len(r) < 2:
        return np.nan
    return float(r.std(ddof=1) * np.sqrt(ppy))


def load_vix_series(path: Path | str) -> pd.Series:
    path = Path(path)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    col = "vix" if "vix" in df.columns else df.columns[0]
    s = df[col].astype(float)
    s.index = pd.to_datetime(s.index)
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    return s.sort_index()


def sp500_log_returns_from_ohlc(data: pd.DataFrame) -> pd.Series:
    """Extract daily S&P 500 close-to-close log returns from notebook market data."""
    if isinstance(data.columns, pd.MultiIndex):
        close = data[("^GSPC", "Close")] if ("^GSPC", "Close") in data.columns else data["^GSPC"]["Close"]
    else:
        close = data["^GSPC"]
    close = close.astype(float).dropna()
    close.index = pd.to_datetime(close.index)
    if getattr(close.index, "tz", None) is not None:
        close.index = close.index.tz_localize(None)
    logret = np.log(close / close.shift(1)).dropna()
    logret.name = "sp500_logret"
    return logret


@dataclass
class MarkovFitResult:
    result: Any
    labels: pd.Series
    filtered_prob_low_vol: pd.Series
    low_vol_state: int
    high_vol_state: int
    summary: Dict[str, Any]


def fit_conlon_markov_2state(
    log_returns: pd.Series,
    random_seed: int = 42,
    search_reps: int = 25,
    maxiter: int = 2000,
) -> MarkovFitResult:
    """
    Two-state Markov switching mean/variance model (Conlon-inspired sensitivity).

    Labels use *filtered* probabilities. The low-volatility state is identified
    from the fitted state variances (not by assuming state index 0).
    """
    if MarkovRegression is None:
        raise ImportError(f"statsmodels MarkovRegression unavailable: {_MARKOV_IMPORT_ERROR}")

    y = log_returns.dropna().astype(float).copy()
    y.index = pd.to_datetime(y.index)
    if getattr(y.index, "tz", None) is not None:
        y.index = y.index.tz_localize(None)

    np.random.seed(random_seed)
    model = MarkovRegression(y, k_regimes=2, trend="c", switching_variance=True)
    # search_reps / random starts where supported
    try:
        result = model.fit(maxiter=maxiter, disp=False, search_reps=search_reps)
    except TypeError:
        result = model.fit(maxiter=maxiter, disp=False)

    if not bool(getattr(result, "mle_retvals", {}).get("converged", True)):
        # statsmodels stores convergence in mle_retvals for some optimizers
        converged = getattr(result, "mle_retvals", None)
        if isinstance(converged, dict) and converged.get("converged") is False:
            raise RuntimeError("MarkovRegression failed to converge")

    # State variances: prefer regime_variance if present
    if hasattr(result, "regime_variance"):
        variances = np.asarray(result.regime_variance, dtype=float).ravel()
    else:
        # fallback: params named sigma2[0], sigma2[1]
        variances = np.array(
            [float(result.params[f"sigma2[{i}]"]) for i in range(2)], dtype=float
        )
    low_vol_state = int(np.argmin(variances))
    high_vol_state = 1 - low_vol_state

    filtered = np.asarray(result.filtered_marginal_probabilities)[:, low_vol_state]
    if filtered.ndim > 1:
        filtered = filtered.ravel()
    # Align length: filtered probs typically match endog length
    if len(filtered) != len(y):
        # some versions drop initial obs
        n = min(len(filtered), len(y))
        filtered = filtered[-n:]
        y = y.iloc[-n:]

    filtered = np.clip(filtered, 0.0, 1.0)
    labels_arr = np.where(filtered < 0.5, "high_volatility", "low_volatility")
    labels = pd.Series(labels_arr, index=y.index, name="regime")
    prob_low = pd.Series(filtered, index=y.index, name="filtered_prob_low_vol")

    # Transition matrix
    try:
        P = np.asarray(result.regime_transition, dtype=float)
    except Exception:
        P = np.asarray(getattr(result, "transition_matrix", np.nan))

    means = []
    for s in range(2):
        key = f"const[{s}]"
        if key in result.params.index:
            means.append(float(result.params[key]))
        else:
            means.append(np.nan)

    summary = {
        "converged": True,
        "loglikelihood": float(result.llf),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "low_vol_state": low_vol_state,
        "high_vol_state": high_vol_state,
        "state_variances": {int(i): float(variances[i]) for i in range(2)},
        "state_means": {int(i): float(means[i]) for i in range(2)},
        "transition_matrix": P.tolist() if hasattr(P, "tolist") else P,
        "nobs": int(result.nobs),
        "random_seed": random_seed,
        "search_reps": search_reps,
        "maxiter": maxiter,
    }
    # Explicit convergence check via mle_retvals when available
    mle = getattr(result, "mle_retvals", None)
    if isinstance(mle, dict) and "converged" in mle:
        summary["converged"] = bool(mle["converged"])
        if not summary["converged"]:
            raise RuntimeError("MarkovRegression reported non-convergence")

    return MarkovFitResult(
        result=result,
        labels=labels,
        filtered_prob_low_vol=prob_low,
        low_vol_state=low_vol_state,
        high_vol_state=high_vol_state,
        summary=summary,
    )


def build_regime_registry(
    common_index: pd.DatetimeIndex,
    sp500_simple_returns: pd.Series,
    vix: pd.Series,
    markov: MarkovFitResult,
    boundaries: Mapping[str, Tuple[str, str]] = BASELINE_BOUNDARIES,
) -> Tuple[Dict[str, pd.Series], pd.DataFrame]:
    """Build four labelling specifications aligned to the common calendar."""
    common_index = _as_dtindex(common_index)

    baseline = define_event_regimes(common_index, boundaries, shift_trading_days=0)
    minus20 = define_event_regimes(common_index, boundaries, shift_trading_days=-20)
    plus20 = define_event_regimes(common_index, boundaries, shift_trading_days=20)

    # Validate ±20 shift magnitudes on internal boundaries where history allows
    _validate_boundary_shift(baseline, minus20, expected_shift=-20)
    _validate_boundary_shift(baseline, plus20, expected_shift=20)

    assert_event_partition(baseline)
    assert_event_partition(minus20)
    assert_event_partition(plus20)

    markov_labels = markov.labels.reindex(common_index)
    # Drop dates without Markov coverage from markov series only; registry keeps NaN flagged
    markov_on_common = markov_labels.dropna().astype(str)
    assert set(markov_on_common.unique()) <= set(MARKOV_LABELS)

    specs = {
        "baseline_event": baseline,
        "boundary_minus_20td": minus20,
        "boundary_plus_20td": plus20,
        "conlon_markov_2state": markov_labels.astype(object),
    }

    rows = []
    for spec_name, series in specs.items():
        for dt, lab in series.items():
            eligible = pd.notna(lab) and str(lab) != "Unknown"
            rows.append(
                {
                    "date": pd.Timestamp(dt),
                    "regime_specification": spec_name,
                    "regime": lab if pd.notna(lab) else np.nan,
                    "sp500_return": float(sp500_simple_returns.get(dt, np.nan))
                    if dt in sp500_simple_returns.index
                    else np.nan,
                    "vix": float(vix.get(dt, np.nan)) if dt in vix.index else np.nan,
                    "eligible_observation": bool(eligible),
                }
            )
    long_df = pd.DataFrame(rows)
    return specs, long_df


def _transition_dates(labels: pd.Series) -> List[pd.Timestamp]:
    """Return dates where the label changes relative to the previous day."""
    arr = labels.astype(str)
    changes = arr != arr.shift(1)
    changes.iloc[0] = False
    return list(arr.index[changes])


def _validate_boundary_shift(base: pd.Series, alt: pd.Series, expected_shift: int) -> None:
    """Check that each internal transition moved by approximately expected_shift trading days."""
    b_trans = _transition_dates(base)
    a_trans = _transition_dates(alt)
    # Expect three internal transitions for four contiguous regimes
    if len(b_trans) < 3 or len(a_trans) < 3:
        warnings.warn(
            f"Expected 3 internal transitions; got base={len(b_trans)} alt={len(a_trans)}"
        )
        return
    idx = base.index
    for bt, at in zip(b_trans[:3], a_trans[:3]):
        ib = int(idx.get_indexer([bt])[0])
        ia = int(idx.get_indexer([at])[0])
        delta = ia - ib
        if abs(delta - expected_shift) > 0:
            # Allow exact match only; warn if history forced clipping
            if abs(delta) < abs(expected_shift):
                warnings.warn(
                    f"Boundary shift clipped near edge: {bt.date()} -> {at.date()} "
                    f"(delta={delta}, expected={expected_shift})"
                )
            else:
                assert delta == expected_shift, (
                    f"Boundary shift {bt.date()} -> {at.date()} delta={delta}, "
                    f"expected {expected_shift}"
                )


def regime_characteristics_table(
    specs: Mapping[str, pd.Series],
    sp500_simple_returns: pd.Series,
    vix: pd.Series,
    ppy: int = 252,
) -> pd.DataFrame:
    rows = []
    for spec_name, labels in specs.items():
        labs = labels.dropna()
        labs = labs[labs.astype(str) != "Unknown"]
        for regime, sub in labs.groupby(labs):
            dates = sub.index
            rets = sp500_simple_returns.reindex(dates).dropna()
            vx = vix.reindex(dates).dropna()
            episodes = count_contiguous_episodes(labs)[regime]
            rows.append(
                {
                    "regime_specification": spec_name,
                    "regime": regime,
                    "first_date": str(pd.Timestamp(dates.min()).date()),
                    "last_date": str(pd.Timestamp(dates.max()).date()),
                    "n_trading_days": int(len(dates)),
                    "n_contiguous_episodes": int(episodes),
                    "sp500_cumulative_return": float((1.0 + rets).prod() - 1.0)
                    if len(rets)
                    else np.nan,
                    "mean_daily_return": float(rets.mean()) if len(rets) else np.nan,
                    "annualised_return": annualised_return(rets, ppy=ppy),
                    "annualised_volatility": annualised_vol(rets, ppy=ppy),
                    "max_drawdown": max_drawdown(rets),
                    "mean_vix": float(vx.mean()) if len(vx) else np.nan,
                    "median_vix": float(vx.median()) if len(vx) else np.nan,
                    "vix_iqr": float(vx.quantile(0.75) - vx.quantile(0.25))
                    if len(vx)
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def dataframe_to_latex(df: pd.DataFrame, float_format: str = "%.3f") -> str:
    return df.to_latex(index=False, float_format=lambda x: float_format % x if pd.notna(x) else "")


def jaccard(a: Sequence, b: Sequence) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def rank_stability(
    ranking_a: Sequence[str],
    ranking_b: Sequence[str],
    top_k: Tuple[int, ...] = (10, 20),
) -> Dict[str, float]:
    """Compare two ordered feature rankings."""
    # Align on union; use ranks (1=best). Missing -> worst rank+1
    universe = list(dict.fromkeys(list(ranking_a) + list(ranking_b)))
    ra = {f: i + 1 for i, f in enumerate(ranking_a)}
    rb = {f: i + 1 for i, f in enumerate(ranking_b)}
    worst = max(len(ranking_a), len(ranking_b)) + 1
    xa = np.array([ra.get(f, worst) for f in universe], dtype=float)
    xb = np.array([rb.get(f, worst) for f in universe], dtype=float)
    out = {}
    if len(universe) >= 2:
        out["kendall_tau"] = float(stats.kendalltau(xa, xb).correlation or np.nan)
        out["spearman_rho"] = float(stats.spearmanr(xa, xb).correlation or np.nan)
    else:
        out["kendall_tau"] = np.nan
        out["spearman_rho"] = np.nan
    for k in top_k:
        out[f"jaccard_top{k}"] = jaccard(list(ranking_a)[:k], list(ranking_b)[:k])
    return out


def top_features_by_mean_abs(
    attributions,
    feature_names,
    regime_labels_date,
    n_instruments: int,
    top_k: int = 20,
) -> Dict[str, List[str]]:
    """
    Per-regime ranking by mean absolute attribution (signed array preserved for other uses).

    ``attributions`` is expected as in the notebook postprocessed tensor
    with rows = dates × instruments (instrument-major after date via np.repeat).
    """
    A = np.asarray(attributions, dtype=float)
    if hasattr(feature_names, "reshape"):
        feats = np.asarray(feature_names).reshape(-1)
    else:
        feats = np.asarray(feature_names).reshape(-1)

    # Expand date labels to attribution rows if needed
    reg = np.asarray(regime_labels_date)
    if len(reg) * n_instruments == A.shape[0]:
        reg_exp = np.repeat(reg, n_instruments)
    elif len(reg) == A.shape[0]:
        reg_exp = reg
    else:
        # Infer expansion factor from shapes (postprocessed attrs are typically date × n_inst)
        if A.shape[0] % len(reg) == 0:
            factor = A.shape[0] // len(reg)
            reg_exp = np.repeat(reg, factor)
        else:
            raise ValueError(
                f"Cannot align labels ({len(reg)}) with attributions ({A.shape[0]}) "
                f"using n_instruments={n_instruments}"
            )

    B = np.abs(A.reshape(A.shape[0], -1))
    out = {}
    for regime in pd.unique(reg_exp):
        if regime is None or (isinstance(regime, float) and np.isnan(regime)):
            continue
        if str(regime) == "Unknown":
            continue
        mask = reg_exp == regime
        if mask.sum() == 0:
            continue
        mean_abs = np.nanmean(B[mask], axis=0)
        order = np.argsort(-mean_abs)
        out[str(regime)] = [str(feats[i]) for i in order[:top_k]]
    return out


def signed_mean_attributions(
    attributions,
    feature_names,
    regime_labels_date,
    n_instruments: int,
) -> pd.DataFrame:
    """Mean signed attribution by regime × feature."""
    A = np.asarray(attributions, dtype=float)
    feats = np.asarray(feature_names).reshape(-1)
    reg = np.asarray(regime_labels_date)
    if len(reg) * n_instruments == A.shape[0]:
        reg_exp = np.repeat(reg, n_instruments)
    elif len(reg) == A.shape[0]:
        reg_exp = reg
    elif A.shape[0] % len(reg) == 0:
        reg_exp = np.repeat(reg, A.shape[0] // len(reg))
    else:
        raise ValueError(
            f"Cannot align labels ({len(reg)}) with attributions ({A.shape[0]})"
        )

    B = A.reshape(A.shape[0], -1)
    rows = []
    for regime in pd.unique(reg_exp):
        if str(regime) == "Unknown" or regime is None:
            continue
        mask = reg_exp == regime
        means = np.nanmean(B[mask], axis=0)
        for f, m in zip(feats, means):
            rows.append({"regime": str(regime), "feature": str(f), "mean_signed": float(m)})
    return pd.DataFrame(rows)


def align_labels_to_dates(spec_series: pd.Series, dates) -> np.ndarray:
    dates = pd.DatetimeIndex(pd.to_datetime(dates))
    if getattr(dates, "tz", None) is not None:
        dates = dates.tz_localize(None)
    labs = spec_series.reindex(dates)
    return labs.to_numpy()


def stress_reference_contrast(
    values_by_regime: Mapping[str, float],
    spec_name: str,
) -> Dict[str, Any]:
    """
    Principal stress-vs-reference contrast.
    Event specs: crash_period - bull_market.
    Markov: high_volatility - low_volatility.
    """
    if spec_name == "conlon_markov_2state":
        stress, ref = "high_volatility", "low_volatility"
    else:
        stress, ref = "crash_period", "bull_market"
    if stress not in values_by_regime or ref not in values_by_regime:
        return {
            "stress": stress,
            "reference": ref,
            "stress_value": np.nan,
            "reference_value": np.nan,
            "contrast": np.nan,
        }
    sv = float(values_by_regime[stress])
    rv = float(values_by_regime[ref])
    return {
        "stress": stress,
        "reference": ref,
        "stress_value": sv,
        "reference_value": rv,
        "contrast": sv - rv,
    }


def moving_block_bootstrap_mean_ci(
    x: np.ndarray,
    block_length: int = 20,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    n = len(x)
    block_length = max(1, min(block_length, n))
    n_blocks = int(np.ceil(n / block_length))
    means = []
    for _ in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        samples = []
        for s in starts:
            samples.append(x[s : min(s + block_length, n)])
            # wrap if needed
            rem = block_length - len(samples[-1])
            if rem > 0:
                samples.append(x[:rem])
        boot = np.concatenate(samples)[:n]
        means.append(boot.mean())
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def latent_separability_metrics(
    hidden: np.ndarray,
    labels: Sequence[str],
    random_state: int = 42,
) -> Dict[str, Any]:
    hidden = np.asarray(hidden, dtype=float)
    labels = np.asarray(labels)
    mask = pd.notna(labels) & (labels.astype(str) != "Unknown")
    X = hidden[mask]
    y = labels[mask].astype(str)
    out: Dict[str, Any] = {"n": int(len(y)), "n_regimes": int(len(np.unique(y)))}
    if out["n_regimes"] < 2 or out["n"] < 10:
        out["silhouette"] = np.nan
        return out
    try:
        out["silhouette"] = float(silhouette_score(X, y, metric="euclidean"))
    except Exception:
        out["silhouette"] = np.nan

    # Centroid pairwise distances
    cents = {}
    for r in np.unique(y):
        cents[r] = X[y == r].mean(axis=0)
    regs = list(cents.keys())
    dists = []
    for i in range(len(regs)):
        for j in range(i + 1, len(regs)):
            dists.append(float(np.linalg.norm(cents[regs[i]] - cents[regs[j]])))
    out["mean_centroid_distance"] = float(np.mean(dists)) if dists else np.nan
    return out


def linear_probe_summary(
    layer_reps: Mapping[str, Any],
    labels: Sequence[str],
    random_state: int = 121187,
) -> pd.DataFrame:
    """Lightweight linear-probe CV accuracy by layer (mirrors notebook Section 3.2)."""
    y_raw = np.asarray(labels)
    mask = pd.notna(y_raw) & (np.asarray(y_raw).astype(str) != "Unknown")
    y_raw = y_raw[mask]
    le = LabelEncoder()
    y = le.fit_transform(y_raw.astype(str))
    rows = []
    for layer_name, rep in layer_reps.items():
        if hasattr(rep, "detach"):
            X = rep.detach().cpu().numpy()
        else:
            X = np.asarray(rep)
        X = X[mask]
        if len(np.unique(y)) < 2 or X.shape[0] < 20:
            continue
        # need >=5 per class
        _, counts = np.unique(y, return_counts=True)
        if counts.min() < 5:
            continue
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        clf = LogisticRegression(random_state=random_state, max_iter=1000)
        try:
            cv = cross_val_score(clf, Xs, y, cv=5, scoring="accuracy")
            rows.append(
                {
                    "layer": layer_name,
                    "cv_accuracy_mean": float(cv.mean()),
                    "cv_accuracy_std": float(cv.std()),
                    "n_classes": int(len(np.unique(y))),
                    "n": int(len(y)),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "layer": layer_name,
                    "cv_accuracy_mean": np.nan,
                    "cv_accuracy_std": np.nan,
                    "n_classes": int(len(np.unique(y))),
                    "n": int(len(y)),
                    "error": str(exc),
                }
            )
    return pd.DataFrame(rows)


def cramers_v(confusion: pd.DataFrame) -> float:
    table = confusion.values.astype(float)
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.sum()
    if n == 0:
        return np.nan
    r, k = table.shape
    return float(np.sqrt(chi2 / (n * (min(r - 1, k - 1)))))


def event_markov_crosstab(event_labels: pd.Series, markov_labels: pd.Series) -> Dict[str, Any]:
    df = pd.DataFrame({"event": event_labels, "markov": markov_labels}).dropna()
    df = df[(df["event"].astype(str) != "Unknown") & (df["markov"].astype(str) != "Unknown")]
    ct = pd.crosstab(df["event"], df["markov"])
    row_pct = ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0) * 100.0
    col_pct = ct.div(ct.sum(axis=0).replace(0, np.nan), axis=1) * 100.0
    # Encode for ARI
    le_e = LabelEncoder().fit(df["event"].astype(str))
    le_m = LabelEncoder().fit(df["markov"].astype(str))
    ari = float(
        adjusted_rand_score(
            le_e.transform(df["event"].astype(str)),
            le_m.transform(df["markov"].astype(str)),
        )
    )
    return {
        "counts": ct,
        "row_pct": row_pct,
        "col_pct": col_pct,
        "cramers_v": cramers_v(ct),
        "adjusted_rand_index": ari,
        "n": int(len(df)),
    }


def temporal_curve_metrics(mean_curve: np.ndarray) -> Dict[str, float]:
    """mean_curve: lag relevance, index 0 = most recent."""
    c = np.asarray(mean_curve, dtype=float).ravel()
    c = np.nan_to_num(c, nan=0.0)
    s = c.sum()
    p = c / s if s > 0 else np.ones_like(c) / max(len(c), 1)
    lags = np.arange(len(p))
    cdf = np.cumsum(p)
    half = float(np.argmax(cdf >= 0.5)) if len(cdf) else np.nan
    return {
        "peak_lag": float(np.argmax(p)) if len(p) else np.nan,
        "half_life": half,
        # NumPy 2+: trapezoid; NumPy 1.x: trapz
        "auc": float(getattr(np, "trapezoid", np.trapz)(p, lags)) if len(p) > 1 else np.nan,
        "recency_index": float((p * lags).sum()),
        "curve": p,
    }


def compare_temporal_profiles(curve_a: np.ndarray, curve_b: np.ndarray) -> Dict[str, float]:
    ma = temporal_curve_metrics(curve_a)
    mb = temporal_curve_metrics(curve_b)
    pa, pb = ma["curve"], mb["curve"]
    n = min(len(pa), len(pb))
    pa, pb = pa[:n], pb[:n]
    corr = float(np.corrcoef(pa, pb)[0, 1]) if n > 1 and pa.std() > 0 and pb.std() > 0 else np.nan
    return {
        "corr": corr,
        "l1": float(np.abs(pa - pb).sum()),
        "l2": float(np.sqrt(((pa - pb) ** 2).sum())),
        "peak_lag_diff": ma["peak_lag"] - mb["peak_lag"],
        "half_life_diff": ma["half_life"] - mb["half_life"],
        "auc_diff": ma["auc"] - mb["auc"],
        "recency_index_diff": ma["recency_index"] - mb["recency_index"],
        "recency_a": ma["recency_index"],
        "recency_b": mb["recency_index"],
    }


def ensure_output_dirs(root: Path | str) -> Dict[str, Path]:
    root = Path(root)
    dirs = {
        "root": root,
        "data": root / "data",
        "figures": root / "figures",
        "tables": root / "tables",
        "latex": root / "latex",
        "logs": root / "logs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def save_figure(fig, path_stem: Path, dpi: int = 300) -> None:
    path_stem = Path(path_stem)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{path_stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{path_stem}.png", dpi=dpi, bbox_inches="tight")


def fmt_num(x, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "NA"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{float(x):.{digits}f}"
