# ============================================
# Cell 0 — Setup & helpers (no external I/O)
# ============================================
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

# Optional stats; code gracefully degrades if SciPy is missing
try:
    from scipy.stats import kruskal, mannwhitneyu, norm
except Exception:
    kruskal = None
    mannwhitneyu = None
    norm = None

EPS = 1e-12

def _as_np(x):
    if x is None:
        return None
    if hasattr(x, "detach"):  # torch tensor
        return x.detach().cpu().numpy()
    return np.asarray(x)

def normalize_rows(x: np.ndarray) -> np.ndarray:
    s = x.sum(axis=1, keepdims=True)
    s = np.where(s == 0.0, 1.0, s)
    return x / s

def entropy(p: np.ndarray) -> np.ndarray:
    p_safe = np.clip(p, EPS, 1.0)
    return -(p_safe * np.log(p_safe)).sum(axis=1)

def hhi(p: np.ndarray) -> np.ndarray:
    return (p**2).sum(axis=1)

def recency_index(p: np.ndarray) -> np.ndarray:
    L = p.shape[1]
    lags = np.arange(L)  # 0 = most recent
    return (p * lags).sum(axis=1)

def peak_lag(p: np.ndarray) -> np.ndarray:
    return np.argmax(p, axis=1)

def half_life(p: np.ndarray) -> np.ndarray:
    cdf = np.cumsum(p, axis=1)  # from most recent to oldest
    return (cdf >= 0.5).argmax(axis=1)

def roughness_msd(p: np.ndarray) -> np.ndarray:
    d = np.diff(p, axis=1)
    return (d**2).mean(axis=1)

def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    m = len(pvals)
    order = np.argsort(pvals)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, m+1)
    adj = pvals * m / ranks
    adj_sorted = np.minimum.accumulate(adj[order][::-1])[::-1]
    out = np.empty_like(adj_sorted)
    out[order] = adj_sorted
    return np.minimum(out, 1.0)

def hedges_g(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    sx2 = x.var(ddof=1)
    sy2 = y.var(ddof=1)
    sp = np.sqrt(((nx-1)*sx2 + (ny-1)*sy2) / (nx + ny - 2) + EPS)
    g = (x.mean() - y.mean()) / sp
    J = 1.0 - (3.0 / (4.0*(nx + ny) - 9.0))
    return g * J

def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    # Efficient for moderate sizes
    greater = 0
    less = 0
    for xi in x:
        greater += np.sum(xi > y)
        less += np.sum(xi < y)
    return (greater - less) / (len(x) * len(y) + EPS)

def _norm_cdf(z):
    if norm is not None:
        return norm.cdf(z)
    # Fallback (approx)
    return 0.5 * (1.0 + np.math.erf(z / np.sqrt(2)))

def dunn_posthoc(values: List[np.ndarray]) -> pd.DataFrame:
    """
    Dunn test across multiple independent groups.
    values: list of 1D arrays (one per regime)
    Returns DataFrame with pairwise z, p (two-sided), and pairs (i, j).
    """
    # Pool data & compute ranks (average ties)
    arrays = [np.asarray(v).ravel() for v in values]
    n_i = np.array([len(v) for v in arrays], dtype=float)
    labels = np.concatenate([[i]*len(v) for i, v in enumerate(arrays)])
    data = np.concatenate(arrays).astype(float)
    if data.size == 0 or np.any(n_i == 0):
        return pd.DataFrame(columns=["i","j","z","p"])
    # Ranks with average ties
    order = np.argsort(data)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(data)+1)
    # tie correction factor C
    _, counts = np.unique(data, return_counts=True)
    T = np.sum(counts**3 - counts)
    C = 1.0 - T / (len(data)**3 - len(data) + EPS)
    # Mean ranks per group
    Rbar = []
    start = 0
    for n in n_i.astype(int):
        idx = np.arange(start, start+n)
        Rbar.append(ranks[idx].mean())
        start += n
    Rbar = np.array(Rbar, dtype=float)
    # Pairwise z
    const = (len(data)*(len(data)+1)/12.0) * C
    rows = []
    for i in range(len(n_i)):
        for j in range(i+1, len(n_i)):
            denom = np.sqrt(const * (1.0/n_i[i] + 1.0/n_i[j]) + EPS)
            z = (Rbar[i] - Rbar[j]) / denom
            p = 2.0 * (1.0 - _norm_cdf(abs(z)))
            rows.append({"i": i, "j": j, "z": float(z), "p": float(p)})
    return pd.DataFrame(rows)

def kendalls_w(rank_matrix: np.ndarray) -> float:
    """
    rank_matrix: shape (n_features, m_regimes), smaller rank = more important.
    Returns Kendall's W in [0,1]; ties not corrected (simple version).
    """
    R = np.asarray(rank_matrix, dtype=float)
    n, m = R.shape  # n items (features), m judges (regimes)
    Rbar = R.mean(axis=1, keepdims=True)
    S = np.sum((R - Rbar)**2)
    denom = m**2 * (n**3 - n) / 12.0 + EPS
    W = (12.0 * S) / denom
    return max(0.0, min(1.0, float(W)))

# =======================================================
# Cell 1 — Minimal input adapter (pass your precomputes)
# =======================================================
"""
You should provide:
- regimes: array-like of shape (N,) with regime label per row (shared across models)
- For temporal (primary): p_occ_best, p_occ_worst as (N, L) row-normalized or raw (we normalize)
- Optional temporal (secondary): p_attr_best, p_attr_worst as (N, L)
- For features: feat_best, feat_worst as (N, F) absolute relevance; optional feature_names list of length F
"""

def pack_inputs(
    regimes,
    p_occ_best: np.ndarray,
    p_occ_worst: np.ndarray,
    p_attr_best: Optional[np.ndarray] = None,
    p_attr_worst: Optional[np.ndarray] = None,
    feat_best: Optional[np.ndarray] = None,
    feat_worst: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None
) -> Dict:
    reg = np.asarray(regimes)
    assert reg.ndim == 1, "regimes must be 1D"
    N = len(reg)

    def _check(P, name):
        if P is None: return None
        X = _as_np(P)
        assert X.shape[0] == N, f"{name} first dim must match len(regimes)"
        X = np.asarray(X, dtype=float)
        # Normalize rows if needed (temporal distributions)
        if X.ndim == 2 and np.all(X >= 0):
            s = X.sum(axis=1, keepdims=True)
            if not np.allclose(s[s>0], 1.0, atol=1e-3):
                X = normalize_rows(X)
        return X

    p_occ_b = _check(p_occ_best, "p_occ_best")
    p_occ_w = _check(p_occ_worst, "p_occ_worst")
    p_attr_b = _check(p_attr_best, "p_attr_best") if p_attr_best is not None else None
    p_attr_w = _check(p_attr_worst, "p_attr_worst") if p_attr_worst is not None else None

    feat_b = None if feat_best is None else np.asarray(feat_best, dtype=float)
    feat_w = None if feat_worst is None else np.asarray(feat_worst, dtype=float)
    if feat_b is not None:
        assert feat_b.shape[0] == N
    if feat_w is not None:
        assert feat_w.shape[0] == N
        assert (feat_b is None) or (feat_w.shape[1] == feat_b.shape[1])

    if feature_names is None and feat_b is not None:
        F = feat_b.shape[1]
        feature_names = [f"F{j}" for j in range(F)]

    return {
        "regimes": reg,
        "p_occ_best": p_occ_b,
        "p_occ_worst": p_occ_w,
        "p_attr_best": p_attr_b,
        "p_attr_worst": p_attr_w,
        "feat_best": feat_b,
        "feat_worst": feat_w,
        "feature_names": feature_names
    }

# =======================================================
# Cell 2 — Per-sample temporal metrics → DataFrames
# =======================================================
def temporal_metrics_df(p: np.ndarray, regimes: np.ndarray, model_label: str, source: str) -> pd.DataFrame:
    """
    p: (N, L) normalized per-sample distributions over lags (0 = most recent)
    regimes: (N,)
    """
    p = normalize_rows(p)
    df = pd.DataFrame({
        "model": model_label,
        "source": source,
        "regime": regimes,
        "entropy": entropy(p),
        "hhi": hhi(p),
        "recency_index": recency_index(p),
        "peak_lag": peak_lag(p),
        "half_life": half_life(p),
        "roughness_msd": roughness_msd(p),
        "stability_var": p.var(axis=1)
    })
    return df

def build_all_temporal_dfs(inputs: Dict) -> Dict[str, pd.DataFrame]:
    out = {}
    reg = inputs["regimes"]
    if inputs["p_occ_best"] is not None:
        out["occ_best"] = temporal_metrics_df(inputs["p_occ_best"], reg, "best", "occ")
    if inputs["p_occ_worst"] is not None:
        out["occ_worst"] = temporal_metrics_df(inputs["p_occ_worst"], reg, "worst", "occ")
    if inputs["p_attr_best"] is not None:
        out["attr_best"] = temporal_metrics_df(inputs["p_attr_best"], reg, "best", "attr")
    if inputs["p_attr_worst"] is not None:
        out["attr_worst"] = temporal_metrics_df(inputs["p_attr_worst"], reg, "worst", "attr")
    return out

    # =======================================================
    # Cell 3 — Per-regime summaries for temporal metrics
    # =======================================================
def summarize_temporal(df: pd.DataFrame, agg: str = "mean") -> pd.DataFrame:
    """
    df: output of temporal_metrics_df
    Returns per (model, source, regime) summary stats (mean, sd, median, IQR).
    """
    metrics = ["entropy","hhi","recency_index","peak_lag","half_life","roughness_msd","stability_var"]
    grp = df.groupby(["model","source","regime"])
    rows = []
    for (m,s,r), sub in grp:
        row = {"model": m, "source": s, "regime": r, "n": len(sub)}
        for col in metrics:
            vals = sub[col].to_numpy()
            row[f"{col}_{agg}"] = float(getattr(np, agg)(vals))
            row[f"{col}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
            row[f"{col}_median"] = float(np.median(vals))
            q1, q3 = np.percentile(vals, [25,75])
            row[f"{col}_iqr"] = float(q3 - q1)
        rows.append(row)
    return pd.DataFrame(rows)


# =======================================================
# Cell 4 — Feature relevance by regime (+ ranks & Kendall’s W)
# =======================================================
def feature_regime_means(feat: np.ndarray, regimes: np.ndarray, feature_names: List[str], model_label: str, source: str) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    feat: (N, F) absolute relevance (or any nonnegative importance)
    Returns:
      - df_means: rows=(regime, feature) with mean and share (%)
      - df_ranks: rows=(regime) cols=feature ranks (1=most relevant)
      - kendall_W across regimes on ranks
    """
    assert feat.ndim == 2
    F = feat.shape[1]
    feat = np.clip(feat, 0.0, None)
    df = pd.DataFrame(feat, columns=feature_names)
    df["regime"] = regimes
    grp = df.groupby("regime")

    # Means per regime & feature
    means = grp[feature_names].mean()
    shares = means.div(means.sum(axis=1).replace(0, np.nan), axis=0)  # relative share per regime
    means = means.reset_index().melt(id_vars="regime", var_name="feature", value_name="mean_value")
    shares = shares.reset_index().melt(id_vars="regime", var_name="feature", value_name="rel_share")
    df_means = means.merge(shares, on=["regime","feature"], how="left")
    df_means["model"] = model_label
    df_means["source"] = source

    # Ranks (1=most relevant)
    means_wide = df_means.pivot(index="regime", columns="feature", values="mean_value")
    ranks_wide = means_wide.rank(axis=1, ascending=False, method="average")
    df_ranks = ranks_wide.reset_index()
    df_ranks["model"] = model_label
    df_ranks["source"] = source

    # Kendall's W across regimes
    rank_matrix = ranks_wide.values.T  # shape (F, R)
    W = kendalls_w(rank_matrix) if rank_matrix.size > 0 else np.nan
    return df_means, df_ranks, W

# =======================================================
# Cell 5 — Within-model across-regimes tests (KW + Dunn)
# =======================================================
def kw_dunn_temporal(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Kruskal–Wallis across regimes for each temporal metric in a single (model, source) slice.
    Returns:
      - kw_df with H, p
      - dunn_df with pairwise z, p, p_BH per metric
    """
    assert kruskal is not None, "SciPy is required for Kruskal–Wallis."
    metrics = ["entropy","hhi","recency_index","peak_lag","half_life","roughness_msd","stability_var"]
    kw_rows = []
    dunn_blocks = []
    for met in metrics:
        groups = [g[met].to_numpy() for _, g in df.groupby("regime")]
        # skip empty regimes
        groups = [g for g in groups if len(g) > 0]
        if len(groups) < 2:
            continue
        stat, p = kruskal(*groups)
        kw_rows.append({"metric": met, "H": float(stat), "p": float(p)})
        # Dunn post-hoc
        dunn = dunn_posthoc(groups)
        if not dunn.empty:
            dunn["metric"] = met
            dunn_blocks.append(dunn)
    kw_df = pd.DataFrame(kw_rows)
    if not kw_df.empty:
        kw_df["p_BH"] = benjamini_hochberg(kw_df["p"].values)
    dunn_df = pd.DataFrame() if len(dunn_blocks) == 0 else pd.concat(dunn_blocks, ignore_index=True)
    if not dunn_df.empty:
        dunn_df["p_BH"] = benjamini_hochberg(dunn_df["p"].values)
    return kw_df, dunn_df

def kw_dunn_features(feat: np.ndarray, regimes: np.ndarray, feature_names: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    KW + Dunn for per-feature relevance across regimes.
    """
    assert kruskal is not None, "SciPy is required for Kruskal–Wallis."
    rows = []
    dunn_all = []
    for j, fname in enumerate(feature_names):
        by_reg = []
        for r in np.unique(regimes):
            by_reg.append(feat[regimes == r, j])
        by_reg = [v for v in by_reg if len(v) > 0]
        if len(by_reg) < 2:
            continue
        stat, p = kruskal(*by_reg)
        rows.append({"feature": fname, "H": float(stat), "p": float(p)})
        dunn = dunn_posthoc(by_reg)
        if not dunn.empty:
            dunn["feature"] = fname
            dunn_all.append(dunn)
    kw_df = pd.DataFrame(rows)
    if not kw_df.empty:
        kw_df["p_BH"] = benjamini_hochberg(kw_df["p"].values)
    dunn_df = pd.DataFrame() if len(dunn_all) == 0 else pd.concat(dunn_all, ignore_index=True)
    if not dunn_df.empty:
        dunn_df["p_BH"] = benjamini_hochberg(dunn_df["p"].values)
    return kw_df, dunn_df

# =======================================================
# Cell 6 — Between-model tests within each regime
# =======================================================
def between_models_temporal(df_best: pd.DataFrame, df_worst: pd.DataFrame) -> pd.DataFrame:
    """
    Compares best vs worst per regime for temporal metrics using Mann–Whitney U (nonparam)
    Returns one DataFrame with metrics per regime and effect sizes.
    """
    assert mannwhitneyu is not None, "SciPy is required for Mann–Whitney U."
    metrics = ["entropy","hhi","recency_index","peak_lag","half_life","roughness_msd","stability_var"]
    rows = []
    for r in sorted(df_best["regime"].unique()):
        sb = df_best[df_best["regime"] == r]
        sw = df_worst[df_worst["regime"] == r]
        if sb.empty or sw.empty:
            continue
        for met in metrics:
            xb = sb[met].to_numpy()
            xw = sw[met].to_numpy()
            try:
                u = mannwhitneyu(xb, xw, alternative="two-sided")
                p = float(u.pvalue)
            except Exception:
                p = np.nan
            g = hedges_g(xb, xw)
            cd = cliffs_delta(xb, xw)
            rows.append({
                "regime": r, "metric": met,
                "best_mean": float(xb.mean()),
                "worst_mean": float(xw.mean()),
                "delta": float(xb.mean() - xw.mean()),
                "hedges_g": g, "cliffs_delta": cd, "p": p
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        # BH adjust within each regime
        df["p_BH"] = np.nan
        for r in df["regime"].unique():
            mask = df["regime"] == r
            df.loc[mask, "p_BH"] = benjamini_hochberg(df.loc[mask, "p"].fillna(1.0).values)
    return df

def between_models_features(
    feat_best: np.ndarray,
    feat_worst: np.ndarray,
    regimes: np.ndarray,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Compares per-feature relevance best vs worst within each regime.
    """
    assert mannwhitneyu is not None, "SciPy is required for Mann–Whitney U."
    rows = []
    uniq = np.unique(regimes)
    for r in uniq:
        idx = (regimes == r)
        if idx.sum() == 0:
            continue
        Xb = feat_best[idx, :]
        Xw = feat_worst[idx, :]
        for j, fname in enumerate(feature_names):
            xb = Xb[:, j]
            xw = Xw[:, j]
            try:
                u = mannwhitneyu(xb, xw, alternative="two-sided")
                p = float(u.pvalue)
            except Exception:
                p = np.nan
            g = hedges_g(xb, xw)
            cd = cliffs_delta(xb, xw)
            rows.append({
                "regime": r, "feature": fname,
                "best_mean": float(xb.mean()),
                "worst_mean": float(xw.mean()),
                "delta": float(xb.mean() - xw.mean()),
                "hedges_g": g, "cliffs_delta": cd, "p": p
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        for r in df["regime"].unique():
            mask = df["regime"] == r
            df.loc[mask, "p_BH"] = benjamini_hochberg(df.loc[mask, "p"].fillna(1.0).values)
    return df

import numpy as np
from typing import Dict, List, Optional, Tuple

EPS = 1e-12

def _as_np(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _normalize_rows(X: np.ndarray) -> np.ndarray:
    s = X.sum(axis=1, keepdims=True)
    s[s == 0.0] = 1.0
    return X / s

def _align_lag_axis(P: np.ndarray, lag_positions: Optional[np.ndarray]) -> np.ndarray:
    """
    Reorders the lag axis so that index 0 corresponds to the MOST RECENT lag,
    1 to next oldest, ..., L-1 oldest.
    If lag_positions is provided (e.g., [L-1, L-2, ..., 0]), we sort by it.
    """
    if lag_positions is None:
        return P  # assume already recent-first
    lp = np.asarray(lag_positions).ravel()
    assert P.shape[-1] == lp.shape[0], "lag_positions length must match P.shape[-1]"
    # Sort so that lag 0 (most recent) comes first
    order = np.argsort(lp)  # ascending: 0, 1, ..., L-1
    return P[..., order]

def _flatten_temporal(P: np.ndarray, regimes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Accepts P with shape (N, L) or (B, A, L).
    Returns (P_flat, regimes_flat) where P_flat is (N, L) and regimes_flat is (N,).
    - If regimes is (B,), it is repeated A times to (B*A,).
    - If regimes is already (N,), it is returned as-is.
    """
    P = np.asarray(P, dtype=float)
    if P.ndim == 3:
        B, A, L = P.shape
        P_flat = P.reshape(B * A, L)
        # Broadcast regimes
        reg = np.asarray(regimes)
        if reg.ndim != 1:
            raise ValueError("regimes must be 1D.")
        if reg.shape[0] == B:
            reg_flat = np.repeat(reg, A)             # (B*A,)
        elif reg.shape[0] == B * A:
            reg_flat = reg
        else:
            raise ValueError(f"regimes length {reg.shape[0]} incompatible with (B={B}, A={A}).")
        return P_flat, reg_flat
    elif P.ndim == 2:
        N, L = P.shape
        reg = np.asarray(regimes)
        if reg.shape[0] not in (N,):
            raise ValueError(f"regimes length {reg.shape[0]} must equal N={N} for 2D inputs.")
        return P, reg
    else:
        raise ValueError("Temporal input must be 2D (N, L) or 3D (B, A, L).")

def _flatten_features(FM: np.ndarray, regimes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Accepts FM with shape (N, F) or (B, A, F). Returns (FM_flat, regimes_flat).
    Broadcasts regimes similarly to _flatten_temporal.
    """
    FM = np.asarray(FM, dtype=float)
    if FM.ndim == 3:
        B, A, F = FM.shape
        FM_flat = FM.reshape(B * A, F)
        reg = np.asarray(regimes)
        if reg.shape[0] == B:
            reg_flat = np.repeat(reg, A)
        elif reg.shape[0] == B * A:
            reg_flat = reg
        else:
            raise ValueError(f"regimes length {reg.shape[0]} incompatible with (B={B}, A={A}).")
        return FM_flat, reg_flat
    elif FM.ndim == 2:
        N = FM.shape[0]
        reg = np.asarray(regimes)
        if reg.shape[0] != N:
            raise ValueError(f"regimes length {reg.shape[0]} must equal N={N} for 2D feature inputs.")
        return FM, reg
    else:
        raise ValueError("Feature matrix must be 2D (N, F) or 3D (B, A, F).")

def pack_inputs(
    regimes,  # (B,) or (N,)
    p_occ_best,                 # (B,A,L) or (N,L)  -- REQUIRED
    p_occ_worst,                # (B,A,L) or (N,L)  -- REQUIRED
    p_attr_best=None,           # (B,A,L) or (N,L)  -- OPTIONAL
    p_attr_worst=None,          # (B,A,L) or (N,L)  -- OPTIONAL
    feat_best=None,             # (B,A,F) or (N,F)  -- OPTIONAL (needed for feature analysis)
    feat_worst=None,            # (B,A,F) or (N,F)  -- OPTIONAL
    feature_names: Optional[List[str]] = None,
    # OPTIONAL: pass lag positions to align “0 = most recent”
    lag_positions_occ_best: Optional[np.ndarray] = None,
    lag_positions_occ_worst: Optional[np.ndarray] = None,
    lag_positions_attr_best: Optional[np.ndarray] = None,
    lag_positions_attr_worst: Optional[np.ndarray] = None,
) -> Dict:
    """
    Accepts 2D or 3D inputs. If 3D, flattens (B, A, ·) → (N=B*A, ·) and repeats regimes accordingly.
    If lag_positions_* are provided, reorders the lag axis so index 0 = most recent.
    All temporal matrices are row-normalized on return.
    """
    reg = _as_np(regimes).ravel()

    # ---- Occlusion (REQUIRED) ----
    P_ob = _as_np(p_occ_best)
    P_ow = _as_np(p_occ_worst)
    # Align lag axes if needed
    if P_ob.ndim == 3:
        P_ob = _align_lag_axis(P_ob, lag_positions_occ_best)
    else:
        if lag_positions_occ_best is not None:
            P_ob = _align_lag_axis(P_ob, lag_positions_occ_best)
    if P_ow.ndim == 3:
        P_ow = _align_lag_axis(P_ow, lag_positions_occ_worst)
    else:
        if lag_positions_occ_worst is not None:
            P_ow = _align_lag_axis(P_ow, lag_positions_occ_worst)
    # Flatten and broadcast regimes
    P_ob, reg_flat = _flatten_temporal(P_ob, reg)
    P_ow, reg_chk  = _flatten_temporal(P_ow, reg)
    if reg_flat.shape != reg_chk.shape or not np.all(reg_flat == reg_chk):
        raise ValueError("Regime broadcasting mismatch between p_occ_best and p_occ_worst.")
    # Normalize rows
    P_ob = _normalize_rows(np.abs(P_ob))
    P_ow = _normalize_rows(np.abs(P_ow))

    # ---- Attribution (OPTIONAL) ----
    P_ab = P_aw = None
    if p_attr_best is not None and p_attr_worst is not None:
        P_ab = _as_np(p_attr_best)
        P_aw = _as_np(p_attr_worst)
        if P_ab.ndim == 3:
            P_ab = _align_lag_axis(P_ab, lag_positions_attr_best)
        else:
            if lag_positions_attr_best is not None:
                P_ab = _align_lag_axis(P_ab, lag_positions_attr_best)
        if P_aw.ndim == 3:
            P_aw = _align_lag_axis(P_aw, lag_positions_attr_worst)
        else:
            if lag_positions_attr_worst is not None:
                P_aw = _align_lag_axis(P_aw, lag_positions_attr_worst)

        P_ab, reg_attr_b = _flatten_temporal(P_ab, reg)
        P_aw, reg_attr_w = _flatten_temporal(P_aw, reg)
        if reg_attr_b.shape != reg_flat.shape or not np.all(reg_attr_b == reg_flat):
            raise ValueError("Regime mismatch between occlusion and attribution inputs (best).")
        if reg_attr_w.shape != reg_flat.shape or not np.all(reg_attr_w == reg_flat):
            raise ValueError("Regime mismatch between occlusion and attribution inputs (worst).")
        P_ab = _normalize_rows(np.abs(P_ab))
        P_aw = _normalize_rows(np.abs(P_aw))

    # ---- Features (OPTIONAL) ----
    FB = FW = None
    if feat_best is not None and feat_worst is not None:
        FB = _as_np(feat_best)
        FW = _as_np(feat_worst)
        FB, reg_feat_b = _flatten_features(FB, reg)
        FW, reg_feat_w = _flatten_features(FW, reg)
        if reg_feat_b.shape != reg_flat.shape or not np.all(reg_feat_b == reg_flat):
            raise ValueError("Regime mismatch between occlusion and feat_best.")
        if reg_feat_w.shape != reg_flat.shape or not np.all(reg_feat_w == reg_flat):
            raise ValueError("Regime mismatch between occlusion and feat_worst.")

        # Feature names
        if feature_names is None:
            F = FB.shape[1]
            feature_names = [f"F{j}" for j in range(F)]
        else:
            assert len(feature_names) == FB.shape[1], "feature_names length must match features dimension"

    return {
        "regimes": reg_flat,             # (N,)
        "p_occ_best": P_ob,              # (N, L)
        "p_occ_worst": P_ow,             # (N, L)
        "p_attr_best": P_ab,             # (N, L) or None
        "p_attr_worst": P_aw,            # (N, L) or None
        "feat_best": FB,                 # (N, F) or None
        "feat_worst": FW,                # (N, F) or None
        "feature_names": feature_names   # list[str] or None
    }


def build_all_temporal_dfs(inputs: Dict) -> Dict[str, pd.DataFrame]:
    out = {}
    reg = inputs["regimes"]
    if inputs["p_occ_best"] is not None:
        out["occ_best"] = temporal_metrics_df(inputs["p_occ_best"], reg, "best", "occ")
    if inputs["p_occ_worst"] is not None:
        out["occ_worst"] = temporal_metrics_df(inputs["p_occ_worst"], reg, "worst", "occ")
    if inputs["p_attr_best"] is not None:
        out["attr_best"] = temporal_metrics_df(inputs["p_attr_best"], reg, "best", "attr")
    if inputs["p_attr_worst"] is not None:
        out["attr_worst"] = temporal_metrics_df(inputs["p_attr_worst"], reg, "worst", "attr")
    return out




# =======================================================
# Cell 7 — One-shot runner (assemble everything)
# =======================================================
def run_regime_sensitivity(inputs: Dict) -> Dict[str, pd.DataFrame]:
    out = {}
    reg = inputs["regimes"]
    # Temporal metrics: build per-sample DFs
    dfs = build_all_temporal_dfs(inputs)

    # Summaries per (model, source, regime)
    for key, df in dfs.items():
        out[f"summary_{key}"] = summarize_temporal(df)

    # Within-model across-regimes tests (temporal)
    # Do separately for best/worst & for occ/attr if present
    if "occ_best" in dfs and "occ_worst" in dfs:
        kw_b, dunn_b = kw_dunn_temporal(dfs["occ_best"])
    kw_w, dunn_w = kw_dunn_temporal(dfs["occ_worst"])
    kw_b["model"]= "best"; kw_b["source"]="occ"
    kw_w["model"]= "worst"; kw_w["source"]="occ"
    dunn_b["model"]="best"; dunn_b["source"]="occ"
    dunn_w["model"]="worst"; dunn_w["source"]="occ"
    out["kw_temporal_occ"]   = pd.concat([kw_b, kw_w], ignore_index=True)
    out["dunn_temporal_occ"] = pd.concat([dunn_b, dunn_w], ignore_index=True)

    if "attr_best" in dfs and "attr_worst" in dfs:
        kw_b, dunn_b = kw_dunn_temporal(dfs["attr_best"])
    kw_w, dunn_w = kw_dunn_temporal(dfs["attr_worst"])
    kw_b["model"]= "best"; kw_b["source"]="attr"
    kw_w["model"]= "worst"; kw_w["source"]="attr"
    dunn_b["model"]="best"; dunn_b["source"]="attr"
    dunn_w["model"]="worst"; dunn_w["source"]="attr"
    out["kw_temporal_attr"]   = pd.concat([kw_b, kw_w], ignore_index=True)
    out["dunn_temporal_attr"] = pd.concat([dunn_b, dunn_w], ignore_index=True)

    # Between-model temporal within each regime
    if "occ_best" in dfs and "occ_worst" in dfs:
        out["between_temporal_occ"] = between_models_temporal(dfs["occ_best"], dfs["occ_worst"])
    if "attr_best" in dfs and "attr_worst" in dfs:
        out["between_temporal_attr"] = between_models_temporal(dfs["attr_best"], dfs["attr_worst"])

    # Feature relevance blocks (if provided)
    if (inputs["feat_best"] is not None) and (inputs["feat_worst"] is not None):
        names = inputs["feature_names"]
    # Per-model regime means/ranks + Kendall's W
    means_b, ranks_b, W_b = feature_regime_means(inputs["feat_best"], reg, names, "best", "occ")
    means_w, ranks_w, W_w = feature_regime_means(inputs["feat_worst"], reg, names, "worst", "occ")
    out["feature_means_best"] = means_b
    out["feature_means_worst"] = means_w
    out["feature_ranks_best"] = ranks_b
    out["feature_ranks_worst"] = ranks_w
    out["kendallsW"] = pd.DataFrame({"model":["best","worst"], "W":[W_b, W_w]})

    # KW + Dunn across regimes (within-model)
    kw_b, dunn_b = kw_dunn_features(inputs["feat_best"], reg, names)
    kw_w, dunn_w = kw_dunn_features(inputs["feat_worst"], reg, names)
    kw_b["model"]="best"; kw_w["model"]="worst"
    dunn_b["model"]="best"; dunn_w["model"]="worst"
    out["kw_features"]   = pd.concat([kw_b, kw_w], ignore_index=True)
    out["dunn_features"] = pd.concat([dunn_b, dunn_w], ignore_index=True)

    # Between-models per regime (features)
    out["between_features"] = between_models_features(inputs["feat_best"], inputs["feat_worst"], reg, names)
    out['raw_dfs'] = dfs

    return out


def summarize_temporal(df: pd.DataFrame, agg: str = "mean") -> pd.DataFrame:
    """
    df: output of temporal_metrics_df
    Returns per (model, source, regime) summary stats (mean, sd, median, IQR).
    """
    metrics = ["entropy","hhi","recency_index","peak_lag","half_life","roughness_msd","stability_var"]
    grp = df.groupby(["model","source","regime"])
    rows = []
    for (m,s,r), sub in grp:
        row = {"model": m, "source": s, "regime": r, "n": len(sub)}
        for col in metrics:
            vals = sub[col].to_numpy()
            row[f"{col}_{agg}"] = float(getattr(np, agg)(vals))
            row[f"{col}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
            row[f"{col}_median"] = float(np.median(vals))
            q1, q3 = np.percentile(vals, [25,75])
            row[f"{col}_iqr"] = float(q3 - q1)
        rows.append(row)
    return pd.DataFrame(rows)

import numpy as np

def attributions_to_p_attr(attr):
    X = attr.detach().cpu().numpy() if hasattr(attr, "detach") else np.asarray(attr)
    if X.ndim == 4:  # (N, F, J, L)
        P = np.abs(X).sum(axis=(1, 2))          # -> (N, L)
    elif X.ndim == 5:  # (B, A, F, J, L)
        P = np.abs(X).sum(axis=(2, 3))          # -> (B, A, L)
    else:
        raise ValueError("Expect (N,F,J,L) or (B,A,F,J,L)")
    # row-normalize
    s = P.sum(axis=-1, keepdims=True)
    s[s == 0.0] = 1.0
    return P / s

def attributions_to_feat_matrix(attr):
    X = attr.detach().cpu().numpy() if hasattr(attr, "detach") else np.asarray(attr)
    if X.ndim == 4:  # (N, F, J, L)
        Fmat = np.abs(X).sum(axis=(2, 3))       # -> (N, F)
    elif X.ndim == 5:  # (B, A, F, J, L)
        Fmat = np.abs(X).sum(axis=(3, 4))       # -> (B, A, F)
    else:
        raise ValueError("Expect (N,F,J,L) or (B,A,F,J,L)")
    return Fmat

# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------- Small helpers ----------
def _ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)

def _bootstrap_ci(vals: np.ndarray, iters: int = 2000, ci: float = 0.95, seed: int = 123) -> tuple[float,float]:
    vals = np.asarray(vals, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(vals)
    if n == 0:
        return (np.nan, np.nan)
    idx = rng.integers(0, n, size=(iters, n))
    means = vals[idx].mean(axis=1)
    lo = np.percentile(means, (1-ci)/2*100)
    hi = np.percentile(means, (1+ci)/2*100)
    return float(lo), float(hi)

# ============================================
# 1) Feature × Regime heatmap with rank labels
# ============================================
def plot_feature_regime_heatmap(means_df: pd.DataFrame,
                                ranks_df: pd.DataFrame,
                                model_label: str,
                                value_col: str = "rel_share",
                                savepath: str = None,
                                filename: str = "F5_1_heatmap.png",
                                title: str = "Feature relevance by regime (relative share)"):
    """
    means_df: results["feature_means_best"/"feature_means_worst"]
              cols: regime, feature, mean_value, rel_share, model, source
    ranks_df: results["feature_ranks_best"/"feature_ranks_worst"] (wide with features as columns)
    """
    df = means_df[means_df["model"] == model_label]
    reg_order = df["regime"].drop_duplicates().tolist()
    feat_order = sorted(df["feature"].unique())  # keep stable order; change if needed

    mat = df.pivot(index="feature", columns="regime", values=value_col).reindex(index=feat_order, columns=reg_order)
    rank_wide = ranks_df[ranks_df["model"] == model_label].set_index("regime")[feat_order].T.reindex(columns=reg_order)

    plt.figure(figsize=(1.8 + 0.5*len(reg_order), 1.8 + 0.35*len(feat_order)))
    im = plt.imshow(mat.values, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(np.arange(len(reg_order)), reg_order, rotation=30, ha="right")
    plt.yticks(np.arange(len(feat_order)), feat_order)
    plt.title(title)

    # annotate with ranks
    for i in range(len(feat_order)):
        for j in range(len(reg_order)):
            r = rank_wide.iloc[i, j]
            try:
                txt = f"{int(round(r))}"
            except Exception:
                txt = ""
            plt.text(j, i, txt, ha="center", va="center")

    plt.tight_layout()
    if savepath:
        _ensure_dir(savepath)
        plt.savefig(os.path.join(savepath, filename), dpi=800, bbox_inches="tight")
    plt.show()

# ============================================
# 2) Rank stability slope chart (top-K features)
# ============================================
def plot_rank_slope_chart(ranks_df: pd.DataFrame,
                          model_label: str,
                          top_k: int = 12,
                          savepath: str = None,
                          filename: str = "F5_2_slope.png",
                          title: str = "Feature rank stability across regimes (lower is better)"):
    """
    ranks_df: results["feature_ranks_best"/"feature_ranks_worst"]
              rows: regime, columns: features, plus 'model','source'
    """
    df = ranks_df[ranks_df["model"] == model_label].drop(columns=["model","source"], errors="ignore")
    reg_order = df["regime"].tolist()
    wide = df.drop(columns=["regime"]).T  # rows=features, cols=regimes
    # pick top-K by average rank
    avg_rank = wide.mean(axis=1).sort_values()
    sel = avg_rank.index[:min(top_k, len(avg_rank))]

    plt.figure(figsize=(1.6 + 0.5*len(reg_order), 4.5))
    x = np.arange(len(reg_order))
    for feat in sel:
        y = wide.loc[feat, :].values
        plt.plot(x, y, marker="o", linewidth=1.5)
        # label ends
        plt.text(x[0]-0.05, y[0], feat, ha="right", va="center")
    plt.gca().invert_yaxis()  # rank 1 at top
    plt.xticks(x, reg_order, rotation=30, ha="right")
    plt.ylabel("Rank (1 = most relevant)")
    plt.title(title)
    plt.tight_layout()
    if savepath:
        _ensure_dir(savepath)
        plt.savefig(os.path.join(savepath, filename), dpi=800, bbox_inches="tight")
    plt.show()

# ====================================================
# 3) Temporal metric violins by regime (single metric)
# ====================================================
def plot_temporal_metric_violin(df_model: pd.DataFrame,
                                metric: str = "recency_index",
                                savepath: str = None,
                                filename: str = "F5_4_violin.png",
                                title: str = "Temporal metric by regime"):
    """
    df_model: e.g., results["occ_best"] or results["occ_worst"] from build_all_temporal_dfs(...)
              (if you didn't keep those around, recompute via temporal_metrics_df)
    """
    regs = sorted(df_model["regime"].unique().tolist())
    data = [df_model[df_model["regime"] == r][metric].to_numpy() for r in regs]

    plt.figure(figsize=(1.6 + 0.5*len(regs), 4.5))
    plt.violinplot(data, showmeans=True, showextrema=True, showmedians=True)
    plt.xticks(np.arange(1, len(regs)+1), regs, rotation=30, ha="right")
    plt.ylabel(metric.replace("_", " "))
    plt.title(title)
    plt.tight_layout()
    if savepath:
        _ensure_dir(savepath)
        plt.savefig(os.path.join(savepath, filename), dpi=800, bbox_inches="tight")
    plt.show()

# ======================================================
# 4) Model × Regime interaction (mean ± 95% CI lines)
# ======================================================
def plot_model_regime_interaction(df_best: pd.DataFrame,
                                  df_worst: pd.DataFrame,
                                  metric: str = "recency_index",
                                  savepath: str = None,
                                  filename: str = "F5_5_interaction.png",
                                  title: str = "Model × Regime interaction"):
    regs = sorted(set(df_best["regime"].unique()).intersection(df_worst["regime"].unique()))
    mb, lb, hb = [], [], []
    mw, lw, hw = [], [], []
    for r in regs:
        xb = df_best[df_best["regime"] == r][metric].to_numpy()
        xw = df_worst[df_worst["regime"] == r][metric].to_numpy()
        mb.append(xb.mean()); lb_i, hb_i = _bootstrap_ci(xb); lb.append(lb_i); hb.append(hb_i)
        mw.append(xw.mean()); lw_i, hw_i = _bootstrap_ci(xw); lw.append(lw_i); hw.append(hw_i)

    x = np.arange(len(regs))
    plt.figure(figsize=(1.6 + 0.5*len(regs), 4.5))
    # best
    plt.plot(x, mb, marker="o", label="Best")
    plt.fill_between(x, lb, hb, alpha=0.2)
    # worst
    plt.plot(x, mw, marker="o", label="Worst")
    plt.fill_between(x, lw, hw, alpha=0.2)
    plt.xticks(x, regs, rotation=30, ha="right")
    plt.ylabel(metric.replace("_"," "))
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if savepath:
        _ensure_dir(savepath)
        plt.savefig(os.path.join(savepath, filename), dpi=800, bbox_inches="tight")
    plt.show()

# ==========================================================================
# 5) Within-regime feature comparison (Best vs Worst) with significance marks
# ==========================================================================
def plot_between_model_features_bar(between_df: pd.DataFrame,
                                    regime: str,
                                    top_k: int = 10,
                                    sort_by: str = "abs_delta",  # {"abs_delta","p_BH"}
                                    savepath: str = None,
                                    filename: str = "F5_6_feat_compare.png",
                                    title: str = "Feature relevance (Best vs Worst)"):
    """
    between_df: results["between_features"] with columns:
      regime, feature, best_mean, worst_mean, delta, hedges_g, cliffs_delta, p, p_BH
    """
    sub = between_df[between_df["regime"] == regime].copy()
    if sub.empty:
        raise ValueError(f"No rows for regime={regime}")
    if sort_by == "abs_delta":
        sub["abs_delta"] = sub["delta"].abs()
        sub = sub.sort_values("abs_delta", ascending=False)
    elif sort_by == "p_BH" and "p_BH" in sub.columns:
        sub = sub.sort_values("p_BH", ascending=True)
    pick = sub.head(min(top_k, len(sub))).copy()
    feats = pick["feature"].tolist()
    x = np.arange(len(feats))
    w = 0.4

    plt.figure(figsize=(1.6 + 0.5*len(feats), 4.8))
    plt.bar(x - w/2, pick["best_mean"].to_numpy(), width=w, label="Best")
    plt.bar(x + w/2, pick["worst_mean"].to_numpy(), width=w, label="Worst")
    # significance markers (BH < 0.05)
    if "p_BH" in pick.columns:
        for i, (_, row) in enumerate(pick.iterrows()):
            if pd.notnull(row["p_BH"]) and row["p_BH"] < 0.05:
                ymax = max(row["best_mean"], row["worst_mean"])
                plt.text(i, ymax, "*", ha="center", va="bottom")

    plt.xticks(x, feats, rotation=45, ha="right")
    plt.ylabel("Mean relevance")
    plt.title(f"{title} — {regime}")
    plt.legend()
    plt.tight_layout()
    if savepath:
        _ensure_dir(savepath)
        plt.savefig(os.path.join(savepath, filename.replace(".png", f"_{regime}.png")), dpi=800, bbox_inches="tight")
    plt.show()
