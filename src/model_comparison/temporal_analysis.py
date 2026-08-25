"""
Temporal attribution analysis: per-model metrics and best-vs-worst comparison.

Inputs you already have:
- Attribution tensor per model: attr: torch.Tensor of shape (N, Indicators, InstrumentsMinus1, Lookback)
  where N = samples * actions
- Occlusion results per model: result of your compute_temporal_occlusion_curves(...)
  (dict containing 'occlusion_curves' of shape (batch_size, action_dim, lags))

Optional:
- Your compute_lag_relevance_auc_halflife(...) function (we re-implement key parts below for safety)

Outputs:
- Per-model summary dicts and DataFrames
- A comparison DataFrame with stats and effect sizes
"""

from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd

# If SciPy is available, we'll use it; otherwise we'll fallback where possible.
try:
    from scipy.stats import ttest_ind, mannwhitneyu, rankdata
except Exception:
    ttest_ind = None
    mannwhitneyu = None
    rankdata = None

EPS = 1e-12

# ---------- Helpers: math for metrics ----------

def _normalize_rows(x: np.ndarray, axis: int = -1) -> np.ndarray:
    s = x.sum(axis=axis, keepdims=True)
    s = np.where(s == 0.0, 1.0, s)
    return x / s

def entropy(p: np.ndarray, axis: int = -1) -> np.ndarray:
    """Shannon entropy with natural log, finite-safe."""
    p_safe = np.clip(p, EPS, 1.0)
    return -(p_safe * np.log(p_safe)).sum(axis=axis)

def hhi(p: np.ndarray, axis: int = -1) -> np.ndarray:
    """Herfindahl–Hirschman Index."""
    return (p ** 2).sum(axis=axis)

def recency_index(p: np.ndarray, recent_first: bool = True) -> np.ndarray:
    """
    Weighted-average lag index.
    If recent_first=True, lag 0 is most recent, increasing to older.
    p: (..., L)
    Returns: (...,)
    """
    L = p.shape[-1]
    lags = np.arange(L) if recent_first else np.arange(L-1, -1, -1)
    return (p * lags).sum(axis=-1)

def peak_lag(p: np.ndarray, recent_first: bool = True) -> np.ndarray:
    """
    Argmax lag index (0 = most recent).
    """
    idx = np.argmax(p, axis=-1)
    if recent_first:
        return idx
    else:
        L = p.shape[-1]
        return (L - 1) - idx

def half_life_from_curve(p: np.ndarray, recent_first: bool = True) -> np.ndarray:
    """
    Half-life (lag where cumulative mass reaches 50%).
    p: array (..., L) normalized over last axis.
    """
    if recent_first:
        # cumulative from most recent to oldest
        cdf = np.cumsum(p, axis=-1)
        half = 0.5
        # first index where CDF >= 0.5
        return (cdf >= half).argmax(axis=-1)
    else:
        cdf = np.cumsum(p[..., ::-1], axis=-1)
        idx = (cdf >= 0.5).argmax(axis=-1)
        L = p.shape[-1]
        return idx  # these already count from recent=0 in reversed order

def mean_variance_over_lags(p: np.ndarray) -> np.ndarray:
    """
    Variance of the per-lag probabilities for each sample-action; averaged across lags.
    Higher means less uniform temporal use (more instability).
    """
    return p.var(axis=-1).mean(axis=-1) if p.ndim > 2 else p.var(axis=-1).mean()

def roughness_msd(p: np.ndarray) -> np.ndarray:
    """
    Roughness via mean squared first differences along lag axis.
    Lower means smoother temporal decay.
    """
    diff = np.diff(p, axis=-1)
    return (diff ** 2).mean(axis=-1)

def bootstrap_ci(x: np.ndarray, iters: int = 2000, ci: float = 0.95, seed: int = 42) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    if n == 0:
        return (np.nan, np.nan)
    idx = rng.integers(0, n, size=(iters, n))
    samples = x[idx].mean(axis=1)
    low = np.percentile(samples, (1-ci)/2*100.0)
    high = np.percentile(samples, (1+(ci))/2*100.0)
    return float(low), float(high)

def hedges_g(x: np.ndarray, y: np.ndarray) -> float:
    """
    Hedges' g with small-sample correction.
    """
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    sx2 = x.var(ddof=1)
    sy2 = y.var(ddof=1)
    sp = np.sqrt(((nx-1)*sx2 + (ny-1)*sy2) / (nx + ny - 2))
    g = (x.mean() - y.mean()) / (sp + EPS)
    # correction
    J = 1.0 - (3.0 / (4.0*(nx + ny) - 9.0))
    return g * J

def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """
    Cliff's delta (non-parametric effect size).
    """
    if rankdata is None:
        # Fallback approximate using sampling to avoid SciPy dependency
        rng = np.random.default_rng(123)
        if len(x) == 0 or len(y) == 0:
            return np.nan
        xs = rng.choice(x, size=min(500, len(x)), replace=True)
        ys = rng.choice(y, size=min(500, len(y)), replace=True)
        # P(X>Y) - P(X<Y)
        comp = xs[:, None] - ys[None, :]
        return (np.mean(comp > 0) - np.mean(comp < 0))
    # Exact using ranks (efficient)
    nx, ny = len(x), len(y)
    # pairwise count method (O(nm)); okay for moderate sizes
    greater = 0
    less = 0
    for xi in x:
        greater += np.sum(xi > y)
        less += np.sum(xi < y)
    return (greater - less) / (nx * ny + EPS)

def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """
    FDR (BH) correction. Returns adjusted p-values.
    """
    m = len(pvals)
    order = np.argsort(pvals)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, m+1)
    adj = pvals * m / ranks
    # monotone
    adj_sorted = np.minimum.accumulate(adj[order][::-1])[::-1]
    out = np.empty_like(adj_sorted)
    out[order] = adj_sorted
    return np.minimum(out, 1.0)

# ---------- From inputs to per-model metrics ----------

def _per_sample_action_p_from_attributions(attr: np.ndarray, recent_first: bool = True) -> np.ndarray:
    """
    attr shape: (N, Indicators, InstrumentsMinus1, L)
    Returns per-sample-action probability over lags: (N, L)
    """
    # aggregate across feature/instrument axes
    agg = attr.sum(axis=(1, 2))  # (N, L)
    agg = np.abs(agg)  # total relevance mass
    p = _normalize_rows(agg, axis=-1)  # (N, L)
    if not recent_first:
        p = p[:, ::-1]
    return p

def _per_sample_action_p_from_occlusion(occlusion_curves: np.ndarray, recent_first: bool = True) -> np.ndarray:
    """
    occlusion_curves shape: (batch_size, action_dim, L)
    Returns per-sample-action probability over lags: (N, L) where N = batch_size * action_dim
    """
    curves = np.abs(occlusion_curves)  # (B, A, L)
    B, A, L = curves.shape
    curves = curves.reshape(B*A, L)
    p = _normalize_rows(curves, axis=-1)
    if not recent_first:
        p = p[:, ::-1]
    return p

def compute_temporal_metrics_matrix(p_lags: np.ndarray, label: str) -> Tuple[pd.DataFrame, Dict]:
    """
    p_lags: (N, L) normalized per-sample probability over lags (0 = most recent)
    Returns:
      - df with per-sample metrics
      - summary dict with aggregates and mean curves
    """
    N, L = p_lags.shape

    ent = entropy(p_lags, axis=1)
    conc_hhi = hhi(p_lags, axis=1)
    rec = recency_index(p_lags, recent_first=True)
    peak = peak_lag(p_lags, recent_first=True)
    hl = half_life_from_curve(p_lags, recent_first=True)
    stab = p_lags.var(axis=1).mean(axis=0)  # scalar? careful
    # We want stability per sample-action: average variance across lags
    stability = p_lags.var(axis=1)  # per-row variance across lags
    rough = roughness_msd(p_lags)

    df = pd.DataFrame({
        "source": label,
        "entropy": ent,
        "hhi": conc_hhi,
        "recency_index": rec,
        "peak_lag": peak,
        "half_life": hl,
        "stability_var": stability,
        "roughness_msd": rough
    })

    # Aggregate stats
    agg = {}
    for col in ["entropy", "hhi", "recency_index", "peak_lag", "half_life", "stability_var", "roughness_msd"]:
        vals = df[col].to_numpy()
        agg[f"{col}_mean"] = float(np.mean(vals))
        agg[f"{col}_std"]  = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
        agg[f"{col}_median"] = float(np.median(vals))
        q1, q3 = np.percentile(vals, [25, 75])
        agg[f"{col}_iqr"] = float(q3 - q1)
        ci_low, ci_high = bootstrap_ci(vals, iters=2000, ci=0.95, seed=123)
        agg[f"{col}_ci95_low"] = ci_low
        agg[f"{col}_ci95_high"] = ci_high

    mean_curve = p_lags.mean(axis=0)  # (L,)
    # bootstrap CI for the mean curve per lag
    L = mean_curve.shape[0]
    low_curve = np.zeros(L)
    high_curve = np.zeros(L)
    rng = np.random.default_rng(123)
    iters = 2000
    for t in range(L):
        vals = p_lags[:, t]
        idx = rng.integers(0, len(vals), size=(iters, len(vals)))
        samples = vals[idx].mean(axis=1)
        low_curve[t] = np.percentile(samples, 2.5)
        high_curve[t] = np.percentile(samples, 97.5)

    summary = {
        "agg": agg,
        "mean_curve": mean_curve,
        "mean_curve_ci_low": low_curve,
        "mean_curve_ci_high": high_curve
    }
    return df, summary

def summarize_model_temporal(
    attr_tensor: np.ndarray,
    occlusion_results: Dict,
    label_prefix: str
) -> Dict:
    """
    Computes per-model temporal metrics from both attributions and occlusions.
    Returns a dict with per-sample DataFrames and aggregate summaries.
    """
    # --- Build p(lag) from attributions ---
    p_attr = _per_sample_action_p_from_attributions(attr_tensor, recent_first=True)
    df_attr, sum_attr = compute_temporal_metrics_matrix(p_attr, label=f"{label_prefix}_attr")

    # --- Build p(lag) from occlusions ---
    occ_curves = occlusion_results["occlusion_curves"]  # torch.Tensor in your function; ensure numpy
    if hasattr(occ_curves, "detach"):
        occ_curves = occ_curves.detach().cpu().numpy()
    p_occ = _per_sample_action_p_from_occlusion(occ_curves, recent_first=True)
    df_occ, sum_occ = compute_temporal_metrics_matrix(p_occ, label=f"{label_prefix}_occlusion")

    # Curve-level distances (L2 between sources, useful sanity check)
    l2_attr_occ = float(np.linalg.norm(sum_attr["mean_curve"] - sum_occ["mean_curve"]))

    return {
        "per_sample_attr": df_attr,
        "per_sample_occ": df_occ,
        "summary_attr": sum_attr,
        "summary_occ": sum_occ,
        "attr_occ_mean_curve_L2": l2_attr_occ
    }

# ---------- Best vs Worst comparison ----------

def _two_sample_tests(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Returns: (p_value, p_value_mwu, hedges_g, cliffs_delta)
    """
    # Welch's t-test
    if ttest_ind is not None and len(x) > 1 and len(y) > 1:
        t_res = ttest_ind(x, y, equal_var=False, alternative='two-sided', nan_policy='omit')
        p_t = float(t_res.pvalue)
    else:
        p_t = np.nan

    # Mann-Whitney U
    if mannwhitneyu is not None and len(x) > 0 and len(y) > 0:
        try:
            u_res = mannwhitneyu(x, y, alternative='two-sided')
            p_u = float(u_res.pvalue)
        except Exception:
            p_u = np.nan
    else:
        p_u = np.nan

    g = hedges_g(x, y)
    cd = cliffs_delta(x, y)

    # Prefer parametric p if available; otherwise non-parametric
    p_main = p_t if not np.isnan(p_t) else p_u
    return p_main, p_u, g, cd

def compare_models_temporal(best: Dict, worst: Dict, source: str = "occlusion") -> pd.DataFrame:
    """
    best/worst: outputs of summarize_model_temporal(...)
    source: "occlusion" or "attr" to pick which set to compare
    Returns a DataFrame with stats, effect sizes, and BH-adjusted p-values.
    """
    assert source in ("occlusion", "attr")
    df_b = best[f"per_sample_{'occ' if source=='occlusion' else 'attr'}"].copy()
    df_w = worst[f"per_sample_{'occ' if source=='occlusion' else 'attr'}"].copy()

    metrics = ["entropy", "hhi", "recency_index", "peak_lag", "half_life", "stability_var", "roughness_msd"]

    rows = []
    pvals = []
    for m in metrics:
        xb = df_b[m].to_numpy()
        xw = df_w[m].to_numpy()
        p_main, p_u, g, cd = _two_sample_tests(xb, xw)
        delta = float(np.mean(xb) - np.mean(xw))
        row = {
            "metric": m,
            "source": source,
            "best_mean": float(np.mean(xb)),
            "worst_mean": float(np.mean(xw)),
            "delta_best_minus_worst": delta,
            "hedges_g": g,
            "cliffs_delta": cd,
            "p_value": p_main,
            "p_value_mannwhitney": p_u
        }
        rows.append(row)
        pvals.append(p_main if not np.isnan(p_main) else 1.0)

    comp = pd.DataFrame(rows)
    comp["p_adj_bh"] = benjamini_hochberg(np.array(pvals))
    return comp

def curve_distance_summary(best: Dict, worst: Dict) -> pd.DataFrame:
    """
    Summarize L2 distances between mean curves across sources.
    """
    rows = []
    for src in ["attr", "occ"]:
        mb = best[f"summary_{src}"]["mean_curve"]
        mw = worst[f"summary_{src}"]["mean_curve"]
        rows.append({
            "source": src,
            "mean_curve_L2": float(np.linalg.norm(mb - mw))
        })
    return pd.DataFrame(rows)


# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------- Utility to ensure output dir ----------
def _ensure_dir(path: str):
    if path and len(path) > 0:
        os.makedirs(path, exist_ok=True)

# --------- 1) Mean curve with 95% CI (best vs worst) ----------
def plot_temporal_mean_curves(best_metrics: dict,
                              worst_metrics: dict,
                              source: str = "occlusion",
                              savepath: str = None,
                              filename: str = "F_temporal_mean_curves.png",
                              title: str = "Temporal attribution profiles (mean ± 95% CI)"):
    """
    Plots mean p(lag) curves with 95% CI for best vs. worst model.
    source: 'occlusion' or 'attr'
    """
    assert source in ("occlusion", "attr")
    b = best_metrics[f"summary_{'occ' if source=='occlusion' else 'attr'}"]
    w = worst_metrics[f"summary_{'occ' if source=='occlusion' else 'attr'}"]

    mean_b, low_b, high_b = b["mean_curve"], b["mean_curve_ci_low"], b["mean_curve_ci_high"]
    mean_w, low_w, high_w = w["mean_curve"], w["mean_curve_ci_low"], w["mean_curve_ci_high"]

    L = len(mean_b)
    lags = np.arange(L)  # 0 = most recent

    plt.figure(figsize=(8, 5))
    # Best
    plt.plot(lags, mean_b, label="Best (mean)")
    plt.fill_between(lags, low_b, high_b, alpha=0.2, label="Best (95% CI)")
    # Worst
    plt.plot(lags, mean_w, label="Worst (mean)")
    plt.fill_between(lags, low_w, high_w, alpha=0.2, label="Worst (95% CI)")

    plt.title(title)
    plt.xlabel("Lag (0 = most recent)")
    plt.ylabel("Probability mass over lags")
    plt.legend()
    plt.tight_layout()

    if savepath:
        _ensure_dir(savepath)
        out = os.path.join(savepath, filename)
        plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.show()

# --------- 2) Peak-lag distribution (violin or histogram) ----------
def plot_peak_lag_violin(best_metrics: dict,
                         worst_metrics: dict,
                         source: str = "occlusion",
                         savepath: str = None,
                         filename: str = "F_peak_lag_violin.png",
                         title: str = "Peak-lag distributions"):
    """
    Violin plot of peak-lag distributions for best vs. worst.
    """
    assert source in ("occlusion", "attr")
    df_b = best_metrics[f"per_sample_{'occ' if source=='occlusion' else 'attr'}"]
    df_w = worst_metrics[f"per_sample_{'occ' if source=='occlusion' else 'attr'}"]

    data = [df_b["peak_lag"].to_numpy(), df_w["peak_lag"].to_numpy()]

    plt.figure(figsize=(7, 5))
    parts = plt.violinplot(data, showmeans=True, showextrema=True, showmedians=True)
    # X tick labels
    plt.xticks([1, 2], ["Best", "Worst"])
    plt.title(title)
    plt.ylabel("Lag index of peak (0 = most recent)")
    plt.tight_layout()

    if savepath:
        _ensure_dir(savepath)
        out = os.path.join(savepath, filename)
        plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.show()

def plot_peak_lag_hist(best_metrics: dict,
                       worst_metrics: dict,
                       source: str = "occlusion",
                       bins: int = 17,
                       savepath: str = None,
                       filename: str = "F_peak_lag_hist.png",
                       title: str = "Peak-lag histograms"):
    """
    Alternative: overlaid histograms of peak-lag (best vs. worst).
    """
    assert source in ("occlusion", "attr")
    df_b = best_metrics[f"per_sample_{'occ' if source=='occlusion' else 'attr'}"]
    df_w = worst_metrics[f"per_sample_{'occ' if source=='occlusion' else 'attr'}"]

    plt.figure(figsize=(8, 5))
    plt.hist(df_b["peak_lag"].to_numpy(), bins=bins, alpha=0.5, density=True, label="Best")
    plt.hist(df_w["peak_lag"].to_numpy(), bins=bins, alpha=0.5, density=True, label="Worst")
    plt.title(title)
    plt.xlabel("Lag index (0 = most recent)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()

    if savepath:
        _ensure_dir(savepath)
        out = os.path.join(savepath, filename)
        plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.show()

# --------- 3) Metric comparison bars ----------
def plot_metric_bars(comp_df: pd.DataFrame,
                     savepath: str = None,
                     filename: str = "F_temporal_metric_bars.png",
                     title: str = "Temporal metric comparison (best vs. worst)",
                     metrics: list = None):
    """
    Bar chart with deltas (best - worst) for selected metrics.
    comp_df: output of compare_models_temporal(...), for a single source
    """
    if metrics is None:
        metrics = ["entropy", "hhi", "recency_index", "half_life", "stability_var", "roughness_msd"]

    plot_df = comp_df[comp_df["metric"].isin(metrics)].copy()
    plot_df = plot_df.set_index("metric").loc[metrics].reset_index()

    x = np.arange(len(metrics))
    y = plot_df["delta_best_minus_worst"].to_numpy()

    plt.figure(figsize=(9, 5))
    plt.bar(x, y)
    plt.axhline(0, linestyle="--")
    plt.xticks(x, metrics, rotation=30, ha="right")
    plt.title(title)
    plt.ylabel("Δ (Best - Worst)")
    plt.tight_layout()

    if savepath:
        _ensure_dir(savepath)
        out = os.path.join(savepath, filename)
        plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.show()

# --------- 4) Curve distance summary (attr vs occ) ----------
def plot_curve_L2(curves_df: pd.DataFrame,
                  savepath: str = None,
                  filename: str = "F_curve_L2.png",
                  title: str = "Mean-curve L2 distance between models"):
    """
    Bar plot of mean-curve L2 distances for 'attr' and 'occ'.
    curves_df: output of curve_distance_summary(best_metrics, worst_metrics)
    """
    x = np.arange(len(curves_df))
    y = curves_df["mean_curve_L2"].to_numpy()

    plt.figure(figsize=(6, 4))
    plt.bar(x, y)
    plt.xticks(x, curves_df["source"].to_list())
    plt.title(title)
    plt.ylabel("L2 distance")
    plt.tight_layout()

    if savepath:
        _ensure_dir(savepath)
        out = os.path.join(savepath, filename)
        plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.show()

# --------- 5) Convenience: one-call figure generation ----------
def make_temporal_figures(best_metrics: dict,
                          worst_metrics: dict,
                          comp_occ: pd.DataFrame,
                          comp_attr: pd.DataFrame = None,
                          curves_df: pd.DataFrame = None,
                          outdir: str = "figures"):
    """
    Generates a standard figure set for the paper.
    """
    _ensure_dir(outdir)
    # Occlusion-based (primary)
    plot_temporal_mean_curves(best_metrics, worst_metrics, source="occlusion",
                              savepath=outdir, filename="F4_1_temporal_mean_curves_occ.png",
                              title="Temporal attribution profiles (occlusion, mean ± 95% CI)")
    plot_peak_lag_violin(best_metrics, worst_metrics, source="occlusion",
                         savepath=outdir, filename="F4_2_peak_lag_violin_occ.png",
                         title="Peak-lag distributions (occlusion)")
    plot_metric_bars(comp_occ, savepath=outdir, filename="F4_3_temporal_metric_bars_occ.png",
                     title="Temporal metrics (occlusion): Δ (Best − Worst)")

    # Attribution-based (optional secondary)
    if comp_attr is not None:
        plot_temporal_mean_curves(best_metrics, worst_metrics, source="attr",
                                  savepath=outdir, filename="F4_1b_temporal_mean_curves_attr.png",
                                  title="Temporal attribution profiles (attribution, mean ± 95% CI)")
        plot_peak_lag_violin(best_metrics, worst_metrics, source="attr",
                             savepath=outdir, filename="F4_2b_peak_lag_violin_attr.png",
                             title="Peak-lag distributions (attribution)")
        plot_metric_bars(comp_attr, savepath=outdir, filename="F4_3b_temporal_metric_bars_attr.png",
                         title="Temporal metrics (attribution): Δ (Best − Worst)")

    # Curve L2 (summary)
    if curves_df is not None:
        plot_curve_L2(curves_df, savepath=outdir, filename="F4_4_curve_L2.png",
                      title="Mean-curve L2 distance between models (attr vs. occ)")
