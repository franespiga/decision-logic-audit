import pandas as pd
import torch
import numpy as np
from . import hhi, entropy_norm, to_weights, flatten_feature_cube, aggregate_categories, per_feature_tests



def compare_feature_importance(
    A_best: torch.Tensor,
    A_worst: torch.Tensor,
    indicators: list,
    instruments: list,
    lags: list,
    topk: int = 10

):
    # 1) Global importance per model
    F_best = A_best.mean(dim=0)   # (11,6,17)
    F_worst= A_worst.mean(dim=0)

    # Flatten for ranking & diversification metrics
    v_best, mi = flatten_feature_cube(F_best, indicators, instruments, lags)  # (1122,), MultiIndex
    v_worst,_  = flatten_feature_cube(F_worst, indicators, instruments, lags)
    w_best = to_weights(v_best.copy())
    w_worst= to_weights(v_worst.copy())

    metrics = {
        "entropy_best": entropy_norm(w_best),
        "entropy_worst": entropy_norm(w_worst),
        "hhi_best": hhi(w_best),
        "hhi_worst": hhi(w_worst)
    }

    # 2) Δ per feature (global)
    delta_vec = v_best - v_worst
    delta = pd.Series(delta_vec, index=mi, name="delta")

    # 3) Category aggregates
    ind_b, ins_b, lag_b = aggregate_categories(F_best)
    ind_w, ins_w, lag_w = aggregate_categories(F_worst)
    agg = {
        "by_indicator": pd.DataFrame({"best": ind_b, "worst": ind_w}, index=indicators),
        "by_instrument": pd.DataFrame({"best": ins_b, "worst": ins_w}, index=instruments),
        "by_lag": pd.DataFrame({"best": lag_b, "worst": lag_w}, index=lags),
    }

    # 4) Per-feature paired tests
    tests = per_feature_tests(A_best, A_worst, indicators, instruments, lags)

    # 5) Top-K lists (global)
    top_best  = pd.Series(v_best, index=mi).sort_values(ascending=False).head(topk)
    top_worst = pd.Series(v_worst, index=mi).sort_values(ascending=False).head(topk)
    top_delta = delta.sort_values(ascending=False).head(topk)

    return {
        "F_best": F_best, "F_worst": F_worst,
        "metrics": metrics,
        "delta_series": delta,
        "aggregates": agg,
        "tests": tests,
        "top_best": top_best, "top_worst": top_worst, "top_delta": top_delta
    }


def plot_top_delta(delta_series: pd.Series, K=20):
    top_pos = delta_series.sort_values(ascending=False).head(K)
    top_neg = delta_series.sort_values(ascending=True).head(K)
    top = pd.concat([top_pos, top_neg])
    ax = top.sort_values().plot(kind="barh", figsize=(8,10), title=f"Top ±{K} Δ feature importance (best - worst)")
    ax.set_xlabel("Δ importance")
    return ax

def plot_bar_by_category(df: pd.DataFrame, title: str):
    ax = df.plot(kind="bar", figsize=(8,4), title=title)
    ax.set_ylabel("Aggregate importance")
    return ax



