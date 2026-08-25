import numpy as np
import torch
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from scipy.stats import kruskal, mannwhitneyu, norm
import matplotlib.pyplot as plt

def _row_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-normalize nonnegative matrix X to sum=1 per row."""
    X = np.asarray(X, dtype=float)
    X[X < 0] = 0.0
    s = X.sum(axis=1, keepdims=True)
    s[s <= eps] = 1.0
    return X / s

def shannon_entropy(P: np.ndarray) -> np.ndarray:
    """Shannon entropy per row; P is row-normalized."""
    P = np.clip(P, 1e-12, 1.0)
    return -(P * np.log(P)).sum(axis=1)

def pielou_evenness(P: np.ndarray) -> np.ndarray:
    """Pielou J = H / log(K)."""
    H = shannon_entropy(P)
    K = P.shape[1]
    return H / np.log(K)

def hhi(P: np.ndarray) -> np.ndarray:
    """Herfindahl-Hirschman Index per row."""
    return (P * P).sum(axis=1)

def participation_ratio(P: np.ndarray) -> np.ndarray:
    """Effective # of active categories: 1 / sum p^2."""
    return 1.0 / hhi(P)

def gini_coeff(P: np.ndarray) -> np.ndarray:
    """Gini coefficient per row; rows sum to 1."""
    # Sort ascending per row
    P_sorted = np.sort(P, axis=1)
    n = P.shape[1]
    i = np.arange(1, n + 1, dtype=float)
    # Using formula: G = 1 + 1/n - 2 * sum((n+1-i)*p_i) / sum(p)
    # sum(p)=1 per row
    weighted = ((n + 1 - i) * P_sorted).sum(axis=1)
    return 1.0 + 1.0 / n - 2.0 * weighted

def topk_coverage(P: np.ndarray, Ks: List[int]) -> Dict[int, np.ndarray]:
    """Return dict K -> coverage per row (sum of top-K probabilities)."""
    P_sorted_desc = np.sort(P, axis=1)[:, ::-1]
    cumsums = np.cumsum(P_sorted_desc, axis=1)
    out = {}
    for K in Ks:
        K = int(K)
        K = min(max(K, 1), P.shape[1])
        out[K] = cumsums[:, K - 1]
    return out

def active_share(P: np.ndarray, tau: float = 0.01) -> np.ndarray:
    """Fraction of categories with p > tau (not normalized by K)."""
    return (P > tau).sum(axis=1) / P.shape[1]

def lorenz_curve(mean_p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Lorenz curve (x cumulative fraction, y cumulative share) for mean composition."""
    p = np.asarray(mean_p, dtype=float)
    p = p[p >= 0]
    s = p.sum()
    if s <= 0:
        p = np.ones_like(p) / len(p)
    else:
        p = p / s
    p_sorted = np.sort(p)  # ascending
    cum = np.cumsum(p_sorted)
    x = np.linspace(0, 1, len(p_sorted), endpoint=True)
    return x, cum


# Cell 3: stats helpers (effect sizes, CIs, BH)

def hedges_g(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    mx, my = x.mean(), y.mean()
    sx2 = x.var(ddof=1) if nx > 1 else 0.0
    sy2 = y.var(ddof=1) if ny > 1 else 0.0
    s_pooled = np.sqrt(((nx-1)*sx2 + (ny-1)*sy2) / max(nx+ny-2, 1))
    d = (mx - my) / max(s_pooled, eps)
    # small-sample bias correction
    J = 1.0 - (3.0 / (4.0*(nx+ny) - 9.0))
    return float(J * d)

def cliffs_delta_from_u(x: np.ndarray, y: np.ndarray) -> float:
    """Approximate Cliff's delta via Mann–Whitney U: delta = 2U/(nx*ny) - 1. (Ties may bias slightly.)"""
    nx, ny = len(x), len(y)
    U = mannwhitneyu(x, y, alternative='two-sided').statistic
    return float(2.0 * U / (nx * ny) - 1.0)

def bootstrap_mean_ci(x: np.ndarray, iters: int = 2000, ci: float = 0.95, seed: int = 42) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    idx = rng.integers(0, n, size=(iters, n))
    means = x[idx].mean(axis=1)
    lo = np.percentile(means, (1-ci)/2*100)
    hi = np.percentile(means, (1+ci)/2*100)
    return float(lo), float(hi)

def bootstrap_diff_ci(x: np.ndarray, y: np.ndarray, iters: int = 2000, ci: float = 0.95, seed: int = 42) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    nx, ny = len(x), len(y)
    idx_x = rng.integers(0, nx, size=(iters, nx))
    idx_y = rng.integers(0, ny, size=(iters, ny))
    diffs = x[idx_x].mean(axis=1) - y[idx_y].mean(axis=1)
    lo = np.percentile(diffs, (1-ci)/2*100)
    hi = np.percentile(diffs, (1+ci)/2*100)
    return float(lo), float(hi)

def benjamini_hochberg(pvals: List[float]) -> List[float]:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n+1)
    adj = p * n / ranks
    # enforce monotonicity
    adj_sorted = np.minimum.accumulate(adj[order][::-1])[::-1]
    out = np.empty_like(adj_sorted)
    out[order] = np.clip(adj_sorted, 0, 1)
    return out.tolist()

# Cell 4: build compositions p^(F), p^(J), p^(L), optional p^(FJ), and recent-window features

def build_compositions(
    A: np.ndarray,
    L_recent: Optional[int] = None,
    most_recent_at_end: bool = True,
    include_FJ: bool = False
) -> Dict[str, np.ndarray]:
    """
    A: (N, F, J, L) absolute attributions (will abs() inside).
    L_recent: if set (e.g., 3), compute recent-window p_F_recent using last L_recent lags.
    most_recent_at_end: if True, the last index along L is the most recent lag.
    include_FJ: if True, return p_FJ over F*J (sum over L).
    """
    X = np.abs(np.asarray(A, dtype=float))
    N, F, J, L = X.shape

    # Feature composition: sum over instruments & lags -> (N, F)
    PF = X.sum(axis=(2, 3))
    PF = _row_normalize(PF)

    # Instrument composition: sum over features & lags -> (N, J)
    PJ = X.sum(axis=(1, 3))
    PJ = _row_normalize(PJ)

    # Lag composition: sum over features & instruments -> (N, L)
    PL = X.sum(axis=(1, 2))
    PL = _row_normalize(PL)

    out = {"pF": PF, "pJ": PJ, "pL": PL}

    # Optional F×J composition: sum over lags -> (N, F*J)
    if include_FJ:
        PFJ = X.sum(axis=3).reshape(N, F*J)
        PFJ = _row_normalize(PFJ)
        out["pFJ"] = PFJ

    # Optional recent-window features
    if L_recent is not None and L_recent > 0:
        if L_recent > L:
            L_recent = L
        if most_recent_at_end:
            sl = slice(L - L_recent, L)
        else:
            sl = slice(0, L_recent)
        PF_recent = X[:, :, :, sl].sum(axis=(2, 3))
        PF_recent = _row_normalize(PF_recent)
        out["pF_recent"] = PF_recent

    return out

# Cell 5: metrics per composition -> DataFrame

def metrics_from_composition(P: np.ndarray,
                             prefix: str,
                             topk_list: List[int] = [1,2,3,5],
                             tau: float = 0.01) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
    [prefix]_J, [prefix]_HHI, [prefix]_PR, [prefix]_Gini, [prefix]_AS
    and Top-K coverage columns: [prefix]_TopK where K in topk_list.
    """
    P = _row_normalize(P)
    df = pd.DataFrame({
        f"{prefix}_J": pielou_evenness(P),
        f"{prefix}_HHI": hhi(P),
        f"{prefix}_PR": participation_ratio(P),
        f"{prefix}_Gini": gini_coeff(P),
        f"{prefix}_AS": active_share(P, tau=tau)
    })
    tk = topk_coverage(P, topk_list)
    for K, vals in tk.items():
        df[f"{prefix}_Top{K}"] = vals
    return df

# Cell 6: per-model metrics

def compute_diversification_metrics_for_model(
    A: np.ndarray,                     # (N,F,J,L)
    regimes: Optional[np.ndarray] = None,  # (N,) ints or labels
    actions: Optional[np.ndarray] = None,  # (N,) optional action ids
    L_recent: Optional[int] = 3,
    include_FJ: bool = False,
    topk_list: List[int] = [1,2,3,5],
    tau: float = 0.01,
    most_recent_at_end: bool = True
) -> Dict[str, pd.DataFrame]:
    comps = build_compositions(A, L_recent=L_recent, most_recent_at_end=most_recent_at_end, include_FJ=include_FJ)

    dfF   = metrics_from_composition(comps["pF"],         prefix="F",   topk_list=topk_list, tau=tau)
    dfJ   = metrics_from_composition(comps["pJ"],         prefix="J",   topk_list=topk_list, tau=tau)
    dfL   = metrics_from_composition(comps["pL"],         prefix="L",   topk_list=topk_list, tau=tau)
    dfr   = metrics_from_composition(comps["pF_recent"],  prefix="Frec",topk_list=topk_list, tau=tau) if "pF_recent" in comps else None
    dfFJ  = metrics_from_composition(comps["pFJ"],        prefix="FJ",  topk_list=topk_list, tau=tau) if "pFJ" in comps else None

    # Attach labels if provided
    for df in [dfF, dfJ, dfL, dfr, dfFJ]:
        if df is not None:
            if regimes is not None:
                df["regime"] = regimes
            if actions is not None:
                df["action"] = actions

    out = {"features": dfF, "instruments": dfJ, "lags": dfL, "features_recent": dfr, "feat_instrument": dfFJ}
    # Keep raw compositions too (useful for plots)
    out["compositions"] = comps
    return out

# Cell 7: global comparison between models

def compare_models_global(
    df_best: pd.DataFrame,
    df_worst: pd.DataFrame,
    metric_cols: List[str],
    ci: float = 0.95
) -> pd.DataFrame:
    rows = []
    pvals = []
    for col in metric_cols:
        xb = df_best[col].to_numpy()
        xw = df_worst[col].to_numpy()
        # MW-U p-value
        mw = mannwhitneyu(xb, xw, alternative="two-sided")
        p = float(mw.pvalue)
        pvals.append(p)
        # CIs
        lo_b, hi_b = bootstrap_mean_ci(xb, ci=ci)
        lo_w, hi_w = bootstrap_mean_ci(xw, ci=ci)
        lo_d, hi_d = bootstrap_diff_ci(xb, xw, ci=ci)
        # Effect sizes
        g = hedges_g(xb, xw)
        cd = cliffs_delta_from_u(xb, xw)
        rows.append({
            "metric": col,
            "best_mean": float(xb.mean()),
            "best_median": float(np.median(xb)),
            "best_CI_lo": lo_b, "best_CI_hi": hi_b,
            "worst_mean": float(xw.mean()),
            "worst_median": float(np.median(xw)),
            "worst_CI_lo": lo_w, "worst_CI_hi": hi_w,
            "delta_mean": float(xb.mean() - xw.mean()),
            "delta_CI_lo": lo_d, "delta_CI_hi": hi_d,
            "hedges_g": g,
            "cliffs_delta": cd,
            "p": p
        })
    df = pd.DataFrame(rows)
    df["p_BH"] = benjamini_hochberg(df["p"].tolist())
    return df.sort_values("metric")

# Cell 8: per-regime (and per-action) comparisons

def compare_models_by_group(
    df_best: pd.DataFrame,
    df_worst: pd.DataFrame,
    metric: str,
    group_col: str = "regime",
    ci: float = 0.95
) -> pd.DataFrame:
    groups = sorted(set(df_best[group_col].unique()).intersection(df_worst[group_col].unique()))
    rows = []
    pvals = []
    for g in groups:
        xb = df_best[df_best[group_col] == g][metric].to_numpy()
        xw = df_worst[df_worst[group_col] == g][metric].to_numpy()
        if len(xb) == 0 or len(xw) == 0:
            continue
        mw = mannwhitneyu(xb, xw, alternative="two-sided")
        p = float(mw.pvalue); pvals.append(p)
        lo_d, hi_d = bootstrap_diff_ci(xb, xw, ci=ci)
        rows.append({
            group_col: g,
            "best_mean": float(xb.mean()),
            "worst_mean": float(xw.mean()),
            "delta_mean": float(xb.mean() - xw.mean()),
            "hedges_g": hedges_g(xb, xw),
            "cliffs_delta": cliffs_delta_from_u(xb, xw),
            "p": p,
            "delta_CI_lo": lo_d, "delta_CI_hi": hi_d
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_BH"] = benjamini_hochberg(out["p"].tolist())
    return out.sort_values(group_col)

def kw_within_model(df: pd.DataFrame, metric: str, group_col: str = "regime") -> Tuple[float, float]:
    """Kruskal–Wallis across groups within one model."""
    groups = [g[metric].to_numpy() for _, g in df.groupby(group_col)]
    if len(groups) < 2:
        return np.nan, np.nan
    H, p = kruskal(*groups)
    return float(H), float(p)



# Cell 9: plots (matplotlib, single chart per figure, no fixed colors)

def plot_ecdf(values_best: np.ndarray, values_worst: np.ndarray,
              xlabel: str, title: str,
              savepath: Optional[str] = None, filename: str = "ecdf.png"):
    xb = np.sort(values_best); xw = np.sort(values_worst)
    yb = np.linspace(0, 1, len(xb), endpoint=True)
    yw = np.linspace(0, 1, len(xw), endpoint=True)
    plt.figure(figsize=(6.5, 4.5))
    plt.plot(xb, yb, label="Best")
    plt.plot(xw, yw, label="Worst")
    plt.xlabel(xlabel); plt.ylabel("ECDF")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if savepath:
        import os; os.makedirs(savepath, exist_ok=True)
        plt.savefig(f"{savepath}/{filename}", dpi=300, bbox_inches="tight")
    plt.show()

def plot_ecdf_by_regime(df_best: pd.DataFrame, df_worst: pd.DataFrame,
                        metric: str, regime_label,
                        savepath: Optional[str] = None, filename: str = "ecdf_regime.png"):
    xb = df_best[df_best["regime"] == regime_label][metric].to_numpy()
    xw = df_worst[df_worst["regime"] == regime_label][metric].to_numpy()
    plot_ecdf(xb, xw, xlabel=metric.replace("_", " "),
              title=f"ECDF of {metric} — Regime: {regime_label}",
              savepath=savepath, filename=filename)

def plot_topk_coverage_curves(PF_best: np.ndarray, PF_worst: np.ndarray,
                              Ks: List[int], title: str,
                              savepath: Optional[str] = None, filename: str = "topk_curves.png",
                              iters: int = 1000, ci: float = 0.95):
    tb = topk_coverage(PF_best, Ks); tw = topk_coverage(PF_worst, Ks)
    Ks_sorted = sorted(Ks)
    mb, lb, hb = [], [], []
    mw, lw, hw = [], [], []
    for K in Ks_sorted:
        vb = tb[K]; vw = tw[K]
        mb.append(vb.mean()); mw.append(vw.mean())
        lo_b, hi_b = bootstrap_mean_ci(vb, iters=iters, ci=ci); lb.append(lo_b); hb.append(hi_b)
        lo_w, hi_w = bootstrap_mean_ci(vw, iters=iters, ci=ci); lw.append(lo_w); hw.append(hi_w)

    x = np.arange(len(Ks_sorted))
    plt.figure(figsize=(6.8, 4.5))
    plt.plot(x, mb, marker="o", label="Best")
    plt.fill_between(x, lb, hb, alpha=0.2)
    plt.plot(x, mw, marker="o", label="Worst")
    plt.fill_between(x, lw, hw, alpha=0.2)
    plt.xticks(x, [str(k) for k in Ks_sorted])
    plt.xlabel("K (Top-K features)"); plt.ylabel("Mean coverage")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if savepath:
        import os; os.makedirs(savepath, exist_ok=True)
        plt.savefig(f"{savepath}/{filename}", dpi=300, bbox_inches="tight")
    plt.show()

def plot_lorenz_curves(PF_best: np.ndarray, PF_worst: np.ndarray,
                       title: str,
                       savepath: Optional[str] = None, filename: str = "lorenz.png"):
    mean_b = PF_best.mean(axis=0)
    mean_w = PF_worst.mean(axis=0)
    xb, yb = lorenz_curve(mean_b)
    xw, yw = lorenz_curve(mean_w)
    G_b = gini_coeff(_row_normalize(mean_b.reshape(1,-1)))[0]
    G_w = gini_coeff(_row_normalize(mean_w.reshape(1,-1)))[0]
    plt.figure(figsize=(6.2, 5.5))
    plt.plot(xb, yb, label=f"Best (Gini={G_b:.2f})")
    plt.plot(xw, yw, label=f"Worst (Gini={G_w:.2f})")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("Cumulative fraction of features")
    plt.ylabel("Cumulative attribution share")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if savepath:
        import os; os.makedirs(savepath, exist_ok=True)
        plt.savefig(f"{savepath}/{filename}", dpi=300, bbox_inches="tight")
    plt.show()

def plot_instrument_evenness_by_regime(df_best: pd.DataFrame, df_worst: pd.DataFrame,
                                       savepath: Optional[str] = None, filename: str = "instrument_evenness.png"):
    regs = sorted(set(df_best["regime"].unique()).intersection(df_worst["regime"].unique()))
    # Build side-by-side boxplot data
    data = []
    labels = []
    positions = []
    pos = 1
    width = 0.35
    for r in regs:
        data.append(df_best[df_best["regime"] == r]["J_J"].to_numpy())
        labels.append(f"{r} (Best)")
        positions.append(pos - width/2)
        data.append(df_worst[df_worst["regime"] == r]["J_J"].to_numpy())
        labels.append(f"{r} (Worst)")
        positions.append(pos + width/2)
        pos += 1

    plt.figure(figsize=(max(7.0, 0.6*len(labels)), 5.0))
    plt.boxplot(data, positions=np.arange(1, len(labels)+1), widths=0.6, showfliers=False)
    plt.xticks(np.arange(1, len(labels)+1), labels, rotation=45, ha="right")
    plt.ylabel("Instrument evenness (Pielou J_J)")
    plt.title("Instrument-level evenness by regime (Best vs Worst)")
    plt.tight_layout()
    if savepath:
        import os; os.makedirs(savepath, exist_ok=True)
        plt.savefig(f"{savepath}/{filename}", dpi=300, bbox_inches="tight")
    plt.show()


# Cell 10: orchestration

def run_diversification_section(
    A_best: np.ndarray,                  # (N,11,6,17)
    A_worst: np.ndarray,                 # (N,11,6,17)
    regimes: Optional[np.ndarray] = None,# (N,)
    actions: Optional[np.ndarray] = None,# (N,)
    L_recent: int = 3,
    topk_list: List[int] = [1,2,3,5],
    tau: float = 0.01,
    most_recent_at_end: bool = True,
    figures_dir: Optional[str] = "figures_diversification",
    group_col:str = "regime"
) -> Dict[str, Dict]:
    # Compute per-model
    best = compute_diversification_metrics_for_model(
        A_best, regimes=regimes, actions=actions, L_recent=L_recent,
        include_FJ=False, topk_list=topk_list, tau=tau, most_recent_at_end=most_recent_at_end
    )
    worst = compute_diversification_metrics_for_model(
        A_worst, regimes=regimes, actions=actions, L_recent=L_recent,
        include_FJ=False, topk_list=topk_list, tau=tau, most_recent_at_end=most_recent_at_end
    )

    # ---- Global comparison (FEATURES) ----
    feat_metrics = ["F_J", "F_HHI", "F_PR", "F_Gini"] + [f"F_Top{k}" for k in topk_list] + ["F_AS"]
    global_cmp_features = compare_models_global(best["features"], worst["features"], metric_cols=feat_metrics)

    # ---- Global comparison (INSTRUMENTS) ----
    inst_metrics = ["J_J", "J_HHI", "J_PR", "J_Gini"] + [f"J_Top{k}" for k in topk_list] + ["J_AS"]
    global_cmp_instruments = compare_models_global(best["instruments"], worst["instruments"], metric_cols=inst_metrics)

    # ---- Global comparison (LAGS) ----
    lag_metrics = ["L_J", "L_HHI", "L_PR", "L_Gini"] + [f"L_Top{k}" for k in topk_list] + ["L_AS"]
    global_cmp_lags = compare_models_global(best["lags"], worst["lags"], metric_cols=lag_metrics)

    # ---- Per-regime comparisons (headline: feature evenness) ----
    per_regime_FJ = compare_models_by_group(best["features"], worst["features"], metric="F_J", group_col=group_col)

    # ---- Within-model KW across regimes (feature evenness) ----
    H_best, p_best = kw_within_model(best["features"], metric="F_J", group_col=group_col) if regimes is not None else (np.nan, np.nan)
    H_worst, p_worst = kw_within_model(worst["features"], metric="F_J", group_col=group_col) if regimes is not None else (np.nan, np.nan)

    # ---- Figures ----
    # 1) ECDF of feature evenness (global)
    plot_ecdf(best["features"]["F_J"].to_numpy(),
              worst["features"]["F_J"].to_numpy(),
              xlabel="Feature evenness (Pielou J_F)",
              title="ECDF — Feature evenness (Best vs Worst)",
              savepath=figures_dir, filename="F55_1_ecdf_evenness.png")

    # 2) Top-K coverage curves (features)
    plot_topk_coverage_curves(
        best["compositions"]["pF"], worst["compositions"]["pF"],
        Ks=topk_list, title="Top-K coverage over features (Best vs Worst)",
        savepath=figures_dir, filename="F55_2_topk.png"
    )

    # 3) Lorenz curves (features)
    plot_lorenz_curves(
        best["compositions"]["pF"], worst["compositions"]["pF"],
        title="Lorenz curves over features (Best vs Worst)",
        savepath=figures_dir, filename="F55_3_lorenz.png"
    )

    # 4) Instrument evenness by regime (boxplots)
    if regimes is not None:
        plot_instrument_evenness_by_regime(
            best["instruments"], worst["instruments"],
            savepath=figures_dir, filename="F55_4_instrument_evenness.png"
        )

        # Optional ECDF per regime (repeat per regime label; one figure per regime)
        for r in sorted(best["instruments"][group_col].unique()):
            plot_ecdf_by_regime(best["features"], worst["features"],
                                metric="F_J", regime_label=r,
                                savepath=figures_dir, filename=f"F55_ecdf_evenness_{group_col}_{r}.png")

    return {
        "best": best,
        "worst": worst,
        "global": {
            "features": global_cmp_features,
            "instruments": global_cmp_instruments,
            "lags": global_cmp_lags
        },
        "per_regime": {
            "features_evenness": per_regime_FJ,
            "KW_best": {"H": H_best, "p": p_best},
            "KW_worst": {"H": H_worst, "p": p_worst}
        }
    }
