import torch
import numpy as np
import pandas as pd
from math import log
from typing import Tuple
from scipy.stats import wilcoxon

def flatten_feature_cube(F: torch.Tensor, indicators, instruments, lags) -> Tuple[np.ndarray, pd.Index]:
    """
    F: (I=11, M=6, L=17) -> (K=1122,) with a MultiIndex for (indicator, instrument, lag)
    """
    I, M, L = F.shape
    arr = F.detach().cpu().numpy().reshape(-1)
    mi = pd.MultiIndex.from_product([indicators, instruments, lags],
                                    names=["indicator","instrument","lag"])
    return arr, mi

def aggregate_categories(F: torch.Tensor):
    # F: (11,6,17)
    by_indicator = F.sum(dim=(1,2)).detach().cpu().numpy()  # (11,)
    by_instrument= F.sum(dim=(0,2)).detach().cpu().numpy()  # (6,)
    by_lag       = F.sum(dim=(0,1)).detach().cpu().numpy()  # (17,)
    return by_indicator, by_instrument, by_lag

def to_weights(x: np.ndarray) -> np.ndarray:
    s = x.sum()
    return x / s if s > 0 else x

def entropy_norm(p: np.ndarray) -> float:
    p = p[p > 0]
    K = len(p)
    H = -(p * np.log(p)).sum()
    return H / log(K) if K > 0 else 0.0   # normalized to [0,1]

def hhi(p: np.ndarray) -> float:
    # Herfindahl-Hirschman Index (higher = more concentrated)
    return float((p**2).sum())

def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    # Efficient approximation using rank statistics
    # For paired data with same length, Cliff’s δ behaves similarly using all pairs.
    # Here we unroll all pairwise comparisons; for speed use a vectorized method if needed.
    nx, ny = len(x), len(y)
    # If identical for speed:
    if np.allclose(x, y):
        return 0.0
    # Sample-based approximation (subsample if huge):
    xs, ys = x, y
    gt = 0; lt = 0
    for xi in xs:
        lt += np.sum(xi < ys)
        gt += np.sum(xi > ys)
    return (gt - lt) / (nx*ny)



def per_feature_tests(A_best: torch.Tensor, A_worst: torch.Tensor, indicators, instruments, lags):
    """
    A_*: (N_sa, 11, 6, 17). Returns DataFrame with Wilcoxon p, median deltas, Cliff's δ
    """
    N, I, M, L = A_best.shape
    res = []
    for i in range(I):
        for m in range(M):
            for l in range(L):
                xb = A_best[:, i, m, l].detach().cpu().numpy()
                xw = A_worst[:, i, m, l].detach().cpu().numpy()
                # Paired Wilcoxon: drop exact= if large N
                try:
                    stat, p = wilcoxon(xb, xw, zero_method="wilcox", correction=False)
                except ValueError:
                    # All zeros or identical -> p=1
                    p = 1.0
                delta = np.median(xb - xw)
                d_eff = cliffs_delta(xb, xw)
                res.append((indicators[i], instruments[m], l, p, delta, d_eff))
    out = pd.DataFrame(res, columns=["indicator","instrument","lag","p_wilcoxon","median_delta","cliffs_delta"])
    # BH–FDR
    p = out["p_wilcoxon"].values
    order = np.argsort(p)
    ranked = p[order]
    m = len(p)
    q = np.empty_like(ranked)
    min_q = 1.0
    for k in range(m-1, -1, -1):
        min_q = min(min_q, ranked[k] * m / (k+1))
        q[k] = min_q
    out["q_value"] = 0.0
    out.loc[order, "q_value"] = q
    return out
