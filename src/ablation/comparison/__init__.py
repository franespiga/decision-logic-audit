
# --- Imports
import os, json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import itertools
import torch
from scipy.stats import kruskal, rankdata
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.stats import kendalltau

# Optional: Dunn's posthoc (if available). Fallback to pairwise MW if not.
try:
    import scikit_posthocs as posthocs
except Exception:
    posthocs = None

# --- Utilities

def bh_fdr(pvals, alpha=0.05):
    p = np.asarray(pvals, float)
    n = p.size
    order = np.argsort(p)
    ranks = np.arange(1, n+1)
    q = np.empty_like(p)
    q[order] = np.minimum.accumulate((p[order] * n / ranks)[::-1])[::-1]
    return q, q < alpha

def merge_dataframes_by_index(data_dict):
    """
    Merges all DataFrames contained in a dictionary by their index.
    The name of the column in the resulting DataFrame will be the key from the dictionary.

    Args:
        data_dict (dict): Dictionary where keys are column names and values are DataFrames to merge.

    Returns:
        pd.DataFrame: A single DataFrame with all DataFrames merged by index.
    """
    merged_df = pd.DataFrame()  # Start with an empty DataFrame
    for key, df in data_dict.items():
        if not merged_df.empty:
            merged_df = merged_df.merge(df, right_index = True, left_index = True)
        else:
            merged_df = df  # Rename the initial DataFrame column to the key
    return merged_df

def aggregate_attributions_per_regime(A:torch.Tensor,
                                      feats:list[str],
                                      reg_labels:list[str]
                                      ):
    B = A.reshape(A.shape[0],-1)
    date_indices = np.tile(reg_labels, (1,7)).reshape(-1)
    df_b = pd.DataFrame(B, index = date_indices,
                        columns = feats.reshape(-1))
    pos_mask = df_b>=0
    neg_mask = df_b<=0

    df_pos = df_b.copy()
    df_pos[~pos_mask] = np.nan
    df_neg = df_b.copy()
    df_neg[~neg_mask] = np.nan
    df_abs = np.abs(df_b)

    M_pos = dict()
    M_neg = dict()
    M_abs = dict()

    for reg in set(reg_labels):
        M_neg[reg] = pd.DataFrame({reg:np.nanmean(df_neg.loc[reg,:], axis = 0)}, index = df_neg.columns)
        M_pos[reg] = pd.DataFrame({reg:np.nanmean(df_pos.loc[reg,:], axis = 0)}, index = df_pos.columns.tolist())
        M_abs[reg] = pd.DataFrame({reg:df_abs.loc[reg,:].mean(axis = 0)})

        Mabs = merge_dataframes_by_index(M_abs)
        Mpos = merge_dataframes_by_index(M_pos)
        Mneg = merge_dataframes_by_index(M_neg)

    return Mabs, Mpos, Mneg, df_b

def kruskal_by_feature(A, reg_labels, feats, alpha=0.05):
    # Build per-feature lists across regimes
    regs = [r for r in pd.unique(reg_labels) if r is not None]
    rows = []
    for j, f in enumerate(feats):
        groups = [A.loc[r, f] for r in regs if np.sum(np.array(reg_labels)==r)>0]
        # Ensure at least two non-empty groups
        if sum(len(g)>0 for g in groups) < 2:
            rows.append((f, np.nan, np.nan)); continue
        stat, p = kruskal(*groups, nan_policy="omit")
        rows.append((f, stat, p))

    df = pd.DataFrame(rows, columns=["Feature","KW_stat","p"])
    df["q"], df["sig"] = bh_fdr(df["p"].fillna(1.0).values, alpha=alpha)
    return df.sort_values("q")

def posthoc_dunn(A, reg_labels, feat_name, regs_order=None):
    """
    Returns DataFrame of pairwise p-values (BH-adjusted) for a single feature.
    Uses scikit-posthocs Dunn if available; else falls back to pairwise Mann-Whitney with BH-FDR.
    """
    regs_order = regs_order or [r for r in pd.unique(reg_labels) if r is not None]
    j = None

    # Feature index lookup
    # Expect caller to pass A as [N,D] and provide the column index externally if needed.
    # Here we assume feature columns named; build a wrapper to pass j.

# --- Effect size: Cliff's delta ----------------------------------------------
def cliffs_delta(a: np.ndarray, b: np.ndarray):
    """
    Nonparametric effect size in [-1, 1].
    Returns (delta, magnitude_str) where magnitude_str ∈ {'negligible','small','medium','large'}.
    """
    a = np.asarray(a); b = np.asarray(b)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan, "na"
    # Compute all pairwise comparisons efficiently
    # Sort and rank trick
    # Fallback simple method (robust for moderate N)
    n, m = len(a), len(b)
    gt = 0; lt = 0
    for x in a:
        gt += np.sum(x > b)
        lt += np.sum(x < b)
    d = (gt - lt) / (n * m)

    ad = abs(d)
    if ad < 0.147: mag = "negligible"
    elif ad < 0.33: mag = "small"
    elif ad < 0.474: mag = "medium"
    else: mag = "large"
    return d, mag

# --- Post-hoc with Cliff's delta ---------------------------------------------
# --- Post-hoc with Cliff's delta ---------------------------------------------
def dunn_for_significant(A, reg_labels, feats, sig_features, alpha=0.05):
    """
    Return dict: (feature, j) -> DataFrame[regA, regB, p_adj, delta, magnitude]
    Uses scikit-posthocs Dunn if available; else pairwise Mann–Whitney.
    Adds Cliff's delta per pair.
    """
    out = {}
    regs = [r for r in pd.unique(reg_labels) if r is not None]
    reg_idx = {r: np.where(reg_labels == r)[0] for r in regs}

    for f, j in sig_features:
        # Collect per-regime series for this feature
        series_by_reg = {r: A[idx, j] for r, idx in reg_idx.items()}

        # p-values via Dunn or MW fallback
        if posthocs is not None:
            df = pd.DataFrame({r: series_by_reg[r] for r in regs})
            ph = posthocs.posthoc_dunn(df, p_adjust='fdr_bh')  # square matrix
            rows = []
            for i in range(len(regs)):
                for k in range(i+1, len(regs)):
                    rA, rB = regs[i], regs[k]
                    p_adj = float(ph.loc[rA, rB])
                    d, mag = cliffs_delta(series_by_reg[rA], series_by_reg[rB])
                    rows.append((rA, rB, p_adj, d, mag))
            res = pd.DataFrame(rows, columns=["regA","regB","p_adj","delta","magnitude"]).sort_values("p_adj")
        else:
            # Pairwise Mann–Whitney + BH-FDR
            from scipy.stats import mannwhitneyu
            pairs, pvals, deltas, mags = [], [], [], []
            for i in range(len(regs)):
                for k in range(i+1, len(regs)):
                    rA, rB = regs[i], regs[k]
                    a = series_by_reg[rA]; b = series_by_reg[rB]
                    if len(a) > 0 and len(b) > 0:
                        _, p = mannwhitneyu(a, b, alternative="two-sided")
                    else:
                        p = 1.0
                    d, mag = cliffs_delta(a, b)
                    pairs.append((rA, rB)); pvals.append(p); deltas.append(d); mags.append(mag)
            q, _ = bh_fdr(np.array(pvals), alpha=alpha)
            rows = [(a,b,float(qq),float(dd),mm) for (a,b),qq,dd,mm in zip(pairs,q,deltas,mags)]
            res = pd.DataFrame(rows, columns=["regA","regB","p_adj","delta","magnitude"]).sort_values("p_adj")

        out[(f, j)] = res
    return out

def kendall_tau_regime_similarity(Mabs):
    regs = list(Mabs.columns)
    # ranks per regime
    ranks = {r: rankdata(-Mabs[r].fillna(-np.inf).values, method="average") for r in regs}
    K = np.zeros((len(regs), len(regs)))
    for i, ri in enumerate(regs):
        for j, rj in enumerate(regs):
            if i==j: K[i,j]=1.0
            else:
                tau,_ = kendalltau(ranks[ri], ranks[rj])
                K[i,j] = tau
    return pd.DataFrame(K, index=regs, columns=regs)



def delta_matrix_for_feature(dunn_dict, feature_name, regimes):
    """
    Build a symmetric matrix of Cliff's δ across all regime pairs for the selected feature.
    Positive δ means (row) > (col) in attribution distribution (as computed in posthoc).
    If the posthoc table lacks a pair, δ is set to np.nan.
    """
    # locate the key (feature_name, j)
    keys = [k for k in dunn_dict.keys() if k[0] == feature_name]
    if not keys:
        raise KeyError(f"Feature '{feature_name}' not found in dunn_dict keys.")
    key = keys[0]
    df = dunn_dict[key]

    reg_to_idx = {r:i for i, r in enumerate(regimes)}
    M = np.full((len(regimes), len(regimes)), np.nan, dtype=float)
    np.fill_diagonal(M, 0.0)

    # fill from available pairs (regA, regB, delta)
    for _, row in df.iterrows():
        rA, rB, delta = row["regA"], row["regB"], float(row["delta"])
        if rA in reg_to_idx and rB in reg_to_idx:
            i, j = reg_to_idx[rA], reg_to_idx[rB]
            M[i, j] = delta
            M[j, i] = -delta  # antisymmetric by definition

    return M, regimes

def clustermap_order(Mstd, metric="correlation", method="ward"):
    # Compute distance and linkage for rows (features) and columns (regimes)
    row_dist = pdist(Mstd.values, metric=metric)
    row_link = linkage(row_dist, method=method)
    col_dist = pdist(Mstd.values.T, metric=metric)
    col_link = linkage(col_dist, method=method)
    row_order = leaves_list(row_link)
    col_order = leaves_list(col_link)
    return row_order, col_order, row_link, col_link

