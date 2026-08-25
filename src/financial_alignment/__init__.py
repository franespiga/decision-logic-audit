# Cell 1: imports & helpers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from scipy.stats import wilcoxon, mannwhitneyu, chi2

# --- General utilities ---
def _to_2d_cash_attr(cash_attr: np.ndarray) -> np.ndarray:
    """
    Accepts (N,F) or (N,1,F) and returns (N,F).
    """
    X = np.asarray(cash_attr)
    if X.ndim == 3:
        assert X.shape[1] == 1, f"Expected (N,1,F) if 3D, got {X.shape}"
        return X[:, 0, :]
    elif X.ndim == 2:
        return X
    else:
        raise ValueError(f"cash_attr must be (N,F) or (N,1,F); got {X.shape}")

def _safe_wilcoxon_signed(x: np.ndarray) -> float:
    """
    One-sample Wilcoxon signed-rank test against 0 (median=0).
    Returns p-value; robust to zero or tiny variation vectors.
    """
    x = np.asarray(x, dtype=float)
    nz = np.count_nonzero(x)
    if nz < 10:
        return 1.0
    try:
        # zero_method='wilcox' drops zeros; if all zeros, it will error
        p = wilcoxon(x, zero_method='wilcox', correction=True, alternative='two-sided').pvalue
        return float(p) if np.isfinite(p) else 1.0
    except Exception:
        return 1.0

def benjamini_hochberg(pvals: List[float]) -> List[float]:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n+1)
    adj = p * n / ranks
    adj_sorted = np.minimum.accumulate(adj[order][::-1])[::-1]
    out = np.empty_like(adj_sorted)
    out[order] = np.clip(adj_sorted, 0, 1)
    return out.tolist()

# Cell 2: C1 — enhancer/suppressor stats (per feature)

def cash_feature_enhancer_suppressor_stats(
    cash_attr_best: np.ndarray,
    feature_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    For the best model: compute mean signed attribution per feature (CASH),
    Wilcoxon signed-rank p-value against 0, BH-adjusted p, and ranks.
    Returns a DataFrame with one row per feature.
    """
    A = _to_2d_cash_attr(cash_attr_best)      # (N,F)
    N, F = A.shape

    if feature_names is None:
        feature_names = [f"feat_{i}" for i in range(F)]

    means = A.mean(axis=0)
    medians = np.median(A, axis=0)
    pvals = [ _safe_wilcoxon_signed(A[:, j]) for j in range(F) ]
    pBH   = benjamini_hochberg(pvals)

    df = pd.DataFrame({
        "feature": feature_names,
        "mean_signed": means,
        "median_signed": medians,
        "p_wilcoxon": pvals,
        "p_BH": pBH
    })
    # Ranks: higher mean => stronger enhancer; lower (more negative) => suppressor
    df["rank_by_mean_desc"] = (-df["mean_signed"]).rank(method="average")
    df["rank_by_mean_asc"]  = ( df["mean_signed"]).rank(method="average")
    df = df.sort_values("mean_signed", ascending=False).reset_index(drop=True)
    return df

# Cell 3: C2 — risk-conditional feature roles (low vs high realized volatility)

def rolling_volatility(returns: np.ndarray, window: int = 20) -> np.ndarray:
    """
    Rolling std of returns. Returns array aligned to returns (NaN for first window-1 samples).
    """
    s = pd.Series(np.asarray(returns, dtype=float))
    vol = s.rolling(window=window, min_periods=window).std().to_numpy()
    return vol

def cash_feature_risk_delta(
    cash_attr_best: np.ndarray,
    returns: np.ndarray,
    feature_names: Optional[List[str]] = None,
    vol_window: int = 20,
    split: str = "median"  # or "quantiles:0.3,0.7"
) -> pd.DataFrame:
    """
    Split samples into low vs high risk using realized volatility of returns,
    then compute per-feature means in each group and the delta (high - low).
    Mann-Whitney U p-values per feature; BH-adjusted across features.
    """
    A = _to_2d_cash_attr(cash_attr_best)  # (N,F)
    N, F = A.shape
    if feature_names is None:
        feature_names = [f"feat_{i}" for i in range(F)]

    vol = rolling_volatility(returns, window=vol_window)
    mask = ~np.isnan(vol)
    A2 = A[mask]
    vol2 = vol[mask]

    # Split
    if split == "median":
        thr = np.nanmedian(vol2)
        low_idx  = vol2 <= thr
        high_idx = vol2 >  thr
    elif split.startswith("quantiles"):
        qs = split.split(":")[1]
        qlo, qhi = [float(x) for x in qs.split(",")]
        lo_thr = np.quantile(vol2, qlo)
        hi_thr = np.quantile(vol2, qhi)
        low_idx  = vol2 <= lo_thr
        high_idx = vol2 >= hi_thr
    else:
        raise ValueError("split must be 'median' or 'quantiles:qlo,qhi'")

    rows = []
    pvals = []
    for j in range(F):
        x_low  = A2[low_idx,  j]
        x_high = A2[high_idx, j]
        # MW-U between independent groups
        if len(x_low) < 5 or len(x_high) < 5:
            p = 1.0
        else:
            p = mannwhitneyu(x_high, x_low, alternative="two-sided").pvalue
        pvals.append(p)
        rows.append({
            "feature": feature_names[j],
            "mean_low":  float(np.mean(x_low))  if len(x_low)  > 0 else np.nan,
            "mean_high": float(np.mean(x_high)) if len(x_high) > 0 else np.nan,
            "delta_high_minus_low": float(np.mean(x_high) - np.mean(x_low))
        })
    df = pd.DataFrame(rows)
    df["p"] = pvals
    df["p_BH"] = benjamini_hochberg(df["p"].tolist())
    df = df.sort_values("delta_high_minus_low", ascending=False).reset_index(drop=True)
    return df

# Cell 4: C3 — regime stability of CASH feature ranks (Kendall's W)

def kendalls_W(rank_matrix: np.ndarray) -> Tuple[float, float]:
    """
    Kendall's coefficient of concordance W for an m x n rank matrix:
      m raters (regimes), n items (features).
    Returns (W, p_value_approx via chi-square).
    """
    R = np.asarray(rank_matrix, dtype=float)  # shape (m, n)
    m, n = R.shape
    # Sum of ranks per item
    Rj = R.sum(axis=0)
    Rbar = Rj.mean()
    S = np.sum((Rj - Rbar)**2)
    W = 12 * S / (m**2 * (n**3 - n)) if n > 1 else np.nan
    # Approximate chi-square with df = n-1
    chi2_stat = m * (n - 1) * W
    p = 1.0 - chi2.cdf(chi2_stat, df=n-1) if np.isfinite(W) else np.nan
    return float(W), float(p)

def cash_feature_rank_stability_by_regime(
    cash_attr_best: np.ndarray,
    regimes: np.ndarray,
    feature_names: Optional[List[str]] = None
) -> Dict[str, object]:
    """
    Compute per-regime mean signed CASH attribution per feature,
    derive ranks (descending), then Kendall's W across regimes.
    Returns dict with W, p, and a DataFrame of per-regime means and ranks.
    """
    A = _to_2d_cash_attr(cash_attr_best)  # (N,F)
    N, F = A.shape
    if feature_names is None:
        feature_names = [f"feat_{i}" for i in range(F)]

    df = pd.DataFrame(A, columns=feature_names)
    df["regime"] = regimes

    per_regime = (
        df.groupby("regime")[feature_names]
        .mean()
        .reset_index()
    )  # shape (m, F) with regimes as rows

    # Build rank matrix (m x n) descending (higher mean => rank 1)
    regime_labels = per_regime["regime"].tolist()
    M = per_regime[feature_names].to_numpy()
    # ranks per row
    ranks = np.argsort(np.argsort(-M, axis=1), axis=1) + 1  # 1..F
    W, p = kendalls_W(ranks)

    per_regime_ranks = per_regime.copy()
    per_regime_ranks[[f"{fn}_rank" for fn in feature_names]] = ranks

    return {
        "W": W, "p": p,
        "per_regime_means": per_regime,
        "per_regime_ranks": per_regime_ranks,
        "regime_labels": regime_labels
    }

# Cell 5: D — plots (best model only)

def plot_cash_top_enhancers(df_feat: pd.DataFrame, top_k: int = 10,
                            savepath: Optional[str] = None, filename: str = "cash_top_enhancers.png"):
    """
    Plot top-K positive mean signed features for CASH (enhancers).
    """
    d = df_feat.sort_values("mean_signed", ascending=False).head(top_k)
    plt.figure(figsize=(7, 4.8))
    plt.barh(d["feature"][::-1], d["mean_signed"][::-1], color='blue')
    plt.xlabel("Mean signed attribution to CASH")
    plt.title(f"CASH enhancers (top {top_k}) — Best model")
    plt.tight_layout()
    if savepath:
        import os; os.makedirs(savepath, exist_ok=True)
        plt.savefig(f"{savepath}/{filename}", dpi=800, bbox_inches="tight")
    plt.show()

def plot_cash_top_suppressors(df_feat: pd.DataFrame, top_k: int = 10,
                              savepath: Optional[str] = None, filename: str = "cash_top_suppressors.png"):
    """
    Plot top-K negative mean signed features for CASH (suppressors).
    """
    d = df_feat.sort_values("mean_signed", ascending=True).head(top_k)
    plt.figure(figsize=(7, 4.8))
    plt.barh(d["feature"][::-1], d["mean_signed"][::-1], color='red')
    plt.xlabel("Mean signed attribution to CASH")
    plt.title(f"CASH suppressors (top {top_k}) — Best model")
    plt.tight_layout()
    if savepath:
        import os; os.makedirs(savepath, exist_ok=True)
        plt.savefig(f"{savepath}/{filename}", dpi=800, bbox_inches="tight")
    plt.show()

def plot_cash_category_deltas(
    df_risk: pd.DataFrame,
    feature_names: List[str],
    feature_categories: Optional[Dict[str, List[str]]] = None,
    savepath: Optional[str] = None,
    filename: str = "cash_category_deltas.png"
):
    """
    Aggregate feature deltas (high - low) by category and plot bars.
    If no categories provided, skips gracefully.
    """
    if not feature_categories:
        print("No feature_categories provided — skipping category delta plot.")
        return

    # Build a mapping feature -> category (first matching pattern wins)
    def match_category(name: str) -> Optional[str]:
        for cat, patterns in feature_categories.items():
            for p in patterns:
                if p.lower() in name.lower():
                    return cat
        return None

    cats, deltas = [], []
    for _, row in df_risk.iterrows():
        cat = match_category(row["feature"])
        if cat is not None:
            cats.append(cat)
            deltas.append(row["delta_high_minus_low"])

    if not cats:
        print("No features matched any category patterns.")
        return

    # Aggregate by category (mean delta)
    agg = pd.DataFrame({"category": cats, "delta": deltas}).groupby("category")["delta"].mean().reset_index()
    agg = agg.sort_values("delta", ascending=False)

    plt.figure(figsize=(7, 4.5))
    plt.barh(agg["category"][::-1], agg["delta"][::-1])
    plt.xlabel("Δ mean CASH attribution (High vol − Low vol)")
    plt.title("Category-level CASH attribution shifts under risk — Best model")
    plt.tight_layout()
    if savepath:
        import os; os.makedirs(savepath, exist_ok=True)
        plt.savefig(f"{savepath}/{filename}", dpi=800, bbox_inches="tight")
    plt.show()

def plot_cash_by_regime_box(
    cash_attr_best: np.ndarray,
    regimes: np.ndarray,
    savepath: Optional[str] = None,
    filename: str = "cash_signed_by_regime.png"
):
    """
    Boxplot of signed total CASH attribution per regime.
    """
    A = _to_2d_cash_attr(cash_attr_best)
    s = A.sum(axis=1)  # signed total CASH attribution
    df = pd.DataFrame({"signed_cash": s, "regime": regimes})
    groups = [g["signed_cash"].to_numpy() for _, g in df.groupby("regime")]

    plt.figure(figsize=(7.5, 5.0))
    plt.boxplot(groups, labels=[str(r) for r in sorted(df["regime"].unique())], showfliers=False)
    plt.ylabel("Signed total CASH attribution")
    plt.title("CASH attribution by regime — Best model")
    plt.tight_layout()
    if savepath:
        import os; os.makedirs(savepath, exist_ok=True)
        plt.savefig(f"{savepath}/{filename}", dpi=800, bbox_inches="tight")
    plt.show()

# Cell 6: Optional D — volatility vs absolute CASH attribution (dual-axis)

def plot_vol_vs_abs_cash(
    cash_attr_best: np.ndarray,
    returns: np.ndarray,
    vol_window: int = 20,
    x_labels:str = None,
    savepath: Optional[str] = None,
    filename: str = "vol_vs_abs_cash.png"
):
    """
    Plot realized volatility (rolling std) vs absolute CASH attribution magnitude over time.
    """
    A = _to_2d_cash_attr(cash_attr_best)
    abs_cash = np.abs(A).sum(axis=1)  # strength of CASH signal
    vol = rolling_volatility(returns, window=vol_window)

    # Align to same length
    x = np.arange(len(abs_cash)) if x_labels is None else x_labels
    plt.figure(figsize=(9, 4.8))
    ax1 = plt.gca()
    ax1.plot(x, abs_cash, label="|CASH attribution| (sum over features)")
    ax1.set_xlabel("Time (sample index)")
    ax1.set_ylabel("|CASH attribution|")

    ax2 = ax1.twinx()
    ax2.plot(x, vol, label=f"Vol (rolling {vol_window})", linestyle="--")
    ax2.set_ylabel("Realized volatility")

    plt.title("Volatility vs absolute CASH attribution — Best model")
    ax1.legend(loc="upper left"); ax2.legend(loc="upper right")
    plt.tight_layout()
    if savepath:
        import os; os.makedirs(savepath, exist_ok=True)
        plt.savefig(f"{savepath}/{filename}", dpi=800, bbox_inches="tight")
    plt.show()

# Cell 7: run C & D for the best model

def run_instrument_alignment(
    cash_attr_best: np.ndarray,          # (N,F) or (N,1,F)
    returns: np.ndarray,                 # (N,)
    regimes: Optional[np.ndarray] = None,# (N,)
    feature_names: Optional[List[str]] = None,
    feature_categories: Optional[Dict[str, List[str]]] = None,
    vol_window: int = 20,
    figures_dir: Optional[str] = "figures_cash_best"
) -> Dict[str, object]:
    # C1: enhancers/suppressors
    df_feat = cash_feature_enhancer_suppressor_stats(cash_attr_best, feature_names)

    # C2: risk-conditional roles
    df_risk = cash_feature_risk_delta(
        cash_attr_best, returns,
        feature_names=feature_names, vol_window=vol_window, split="median"
    )

    # C3: stability across regimes (if provided)
    stability = None
    if regimes is not None:
        stability = cash_feature_rank_stability_by_regime(
            cash_attr_best, regimes, feature_names=feature_names
        )

    # D: plots
    plot_cash_top_enhancers(df_feat, top_k=10, savepath=figures_dir, filename="CASH_enhancers.png")
    plot_cash_top_suppressors(df_feat, top_k=10, savepath=figures_dir, filename="CASH_suppressors.png")
    plot_cash_category_deltas(df_risk, feature_names or df_feat["feature"].tolist(),
                              feature_categories=feature_categories,
                              savepath=figures_dir, filename="CASH_category_deltas.png")
    if regimes is not None:
        plot_cash_by_regime_box(cash_attr_best, regimes, savepath=figures_dir, filename="CASH_by_regime.png")
    plot_vol_vs_abs_cash(cash_attr_best, returns, vol_window=vol_window,
                         x_labels = returns.index, savepath=figures_dir, filename="Vol_vs_absCASH.png")

    return {
        "enhancers_suppressors": df_feat,
        "risk_delta": df_risk,
        "stability": stability
    }
