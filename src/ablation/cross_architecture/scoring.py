"""
Score construction for the contribution / necessity / structure framework.

Six raw metrics -- two per dimension -- are oriented so that larger always means
"better behaved", then averaged with equal weights:

    C_i = mean(contribution metrics)
    N_i = mean(necessity metrics)
    S_i = mean(structure metrics)
    D_i = mean(C_i, N_i, S_i)

Scores reported here are **oriented normalised statistics in [0, 1]**, not
percentiles. No calibration reference distribution over Decision Transformer runs
exists in this repository, so no percentile transform is applied and no threshold
is recalibrated using the reinforcement-learning results. Portfolio performance
is never used to choose a metric direction or a weight.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

SCORE_KIND = "oriented normalised statistic in [0,1] (not a percentile)"

#: raw metric -> (dimension, orientation transform)
METRIC_SPEC: Dict[str, Dict[str, str]] = {
    "contribution_rank_stability": {
        "dimension": "contribution",
        "orientation": "clip_rank_correlation",
        "description": "median pairwise Spearman correlation of feature-group rankings across "
        "regime-stratified bootstrap resamples",
    },
    "contribution_baseline_concordance": {
        "dimension": "contribution",
        "orientation": "clip_rank_correlation",
        "description": "median pairwise Spearman correlation of feature-group rankings across "
        "attribution baselines",
    },
    "deletion_faithfulness": {
        "dimension": "necessity",
        "orientation": "identity_unit",
        "description": "relative excess allocation change of attribution-guided over matched "
        "random feature deletion, (guided - random) / (guided + random) pooled over the deletion "
        "grid and clipped at zero; zero means guided deletion is no more damaging than random, "
        "or that no deletion changes the allocation at all",
    },
    "temporal_profile_stability": {
        "dimension": "necessity",
        "orientation": "clip_rank_correlation",
        "description": "median pairwise Spearman correlation of the temporal-occlusion profile "
        "across bootstrap resamples",
    },
    "latent_neighbourhood_stability": {
        "dimension": "structure",
        "orientation": "identity_unit",
        "description": "mean k-nearest-neighbour Jaccard overlap of the latent representation "
        "under resampling, input noise and low-importance masking (k = 10)",
    },
    "non_concentrated_layer_influence": {
        "dimension": "structure",
        "orientation": "identity_unit",
        "description": "one minus the normalised Herfindahl index of layer-influence shares; "
        "larger means influence is spread over more layers",
    },
}

DIMENSIONS = ("contribution", "necessity", "structure")


def orient(value: float, mode: str) -> float:
    if value is None or not np.isfinite(value):
        return float("nan")
    if mode == "clip_rank_correlation":
        return float(np.clip(value, 0.0, 1.0))
    if mode == "identity_unit":
        return float(np.clip(value, 0.0, 1.0))
    raise ValueError(f"unknown orientation '{mode}'")


def score_model(metrics: Dict[str, float]) -> Dict[str, float]:
    """Oriented metrics and dimension scores for one model."""
    oriented: Dict[str, float] = {}
    for name, spec in METRIC_SPEC.items():
        oriented[f"oriented_{name}"] = orient(metrics.get(name, float("nan")), spec["orientation"])

    out: Dict[str, float] = dict(oriented)
    available = 0
    dim_values: List[float] = []
    for dim in DIMENSIONS:
        names = [n for n, s in METRIC_SPEC.items() if s["dimension"] == dim]
        vals = [oriented[f"oriented_{n}"] for n in names]
        finite = [v for v in vals if np.isfinite(v)]
        score = float(np.mean(finite)) if finite else float("nan")
        out[f"{dim}_score"] = score
        out[f"{dim}_metrics_available"] = len(finite)
        available += len(finite)
        dim_values.append(score)
    finite_dims = [v for v in dim_values if np.isfinite(v)]
    out["aggregate_diagnostic_score"] = float(np.mean(finite_dims)) if len(finite_dims) == 3 else float("nan")
    out["aggregate_partial_score"] = float(np.mean(finite_dims)) if finite_dims else float("nan")
    out["n_raw_metrics_available"] = available
    out["score_kind"] = SCORE_KIND
    return out


def build_score_table(per_model_metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    rows = []
    for model_id, metrics in per_model_metrics.items():
        row = {"model_id": model_id}
        row.update({k: metrics.get(k, float("nan")) for k in METRIC_SPEC})
        row.update(score_model(metrics))
        rows.append(row)
    return pd.DataFrame(rows)


def metric_dictionary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "metric": name,
                "dimension": spec["dimension"],
                "orientation": spec["orientation"],
                "direction": "larger is better",
                "description": spec["description"],
            }
            for name, spec in METRIC_SPEC.items()
        ]
    )


# ---------------------------------------------------------------------------
# Descriptive and statistical comparison
# ---------------------------------------------------------------------------


def family_summary(table: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    rows = []
    for family, grp in table.groupby("family"):
        for col in columns:
            vals = pd.to_numeric(grp[col], errors="coerce").dropna()
            rows.append(
                {
                    "family": family,
                    "metric": col,
                    "n": int(len(vals)),
                    "mean": float(vals.mean()) if len(vals) else float("nan"),
                    "median": float(vals.median()) if len(vals) else float("nan"),
                    "min": float(vals.min()) if len(vals) else float("nan"),
                    "max": float(vals.max()) if len(vals) else float("nan"),
                    "std": float(vals.std(ddof=1)) if len(vals) > 1 else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def rank_correlations(
    table: pd.DataFrame,
    score_columns: Sequence[str],
    performance_column: str,
    groups: Optional[Dict[str, pd.Index]] = None,
    min_n: int = 4,
) -> pd.DataFrame:
    """Spearman and Kendall correlations between diagnostic scores and performance."""
    from scipy import stats

    groups = groups or {"all": table.index}
    rows = []
    for gname, idx in groups.items():
        sub = table.loc[idx]
        for col in score_columns:
            x = pd.to_numeric(sub[col], errors="coerce")
            y = pd.to_numeric(sub[performance_column], errors="coerce")
            ok = x.notna() & y.notna()
            n = int(ok.sum())
            if n < min_n:
                rows.append(
                    {
                        "group": gname,
                        "score": col,
                        "performance": performance_column,
                        "n": n,
                        "spearman": float("nan"),
                        "spearman_p": float("nan"),
                        "kendall": float("nan"),
                        "kendall_p": float("nan"),
                        "note": f"not computed: n={n} < {min_n}",
                    }
                )
                continue
            s = stats.spearmanr(x[ok], y[ok])
            k = stats.kendalltau(x[ok], y[ok])
            rows.append(
                {
                    "group": gname,
                    "score": col,
                    "performance": performance_column,
                    "n": n,
                    "spearman": float(s.correlation),
                    "spearman_p": float(s.pvalue),
                    "kendall": float(k.correlation),
                    "kendall_p": float(k.pvalue),
                    "note": "descriptive only; sample size does not support inference",
                }
            )
    return pd.DataFrame(rows)


def bootstrap_ci(
    values: Sequence[float], n_bootstrap: int = 1000, ci: float = 0.95, seed: int = 42
) -> Dict[str, float]:
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return {"mean": float("nan"), "lower": float("nan"), "upper": float("nan"), "n": 0}
    if v.size == 1:
        return {"mean": float(v[0]), "lower": float(v[0]), "upper": float(v[0]), "n": 1}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, v.size, size=(int(n_bootstrap), v.size))
    means = v[draws].mean(axis=1)
    lo = float(np.percentile(means, (1 - ci) / 2 * 100))
    hi = float(np.percentile(means, (1 + ci) / 2 * 100))
    return {"mean": float(v.mean()), "lower": lo, "upper": hi, "n": int(v.size)}


def bootstrap_table(
    table: pd.DataFrame, columns: Sequence[str], n_bootstrap: int = 1000, seed: int = 42
) -> pd.DataFrame:
    rows = []
    for family, grp in list(table.groupby("family")) + [("all", table)]:
        for col in columns:
            stat = bootstrap_ci(
                pd.to_numeric(grp[col], errors="coerce").tolist(), n_bootstrap=n_bootstrap, seed=seed
            )
            rows.append({"group": family, "metric": col, **stat})
    return pd.DataFrame(rows)


def matched_comparison(table: pd.DataFrame) -> pd.DataFrame:
    """
    Descriptive comparison restricted to models that share the design factors that
    can be matched here: asset universe, rebalancing frequency and lookback.
    """
    rows = []
    for key, grp in table.groupby(["lookback", "rebalancing_frequency"]):
        if grp["family"].nunique() < 2:
            continue
        for col in (
            "contribution_score",
            "necessity_score",
            "structure_score",
            "aggregate_diagnostic_score",
            "gross_sharpe_ratio",
        ):
            for family, sub in grp.groupby("family"):
                vals = pd.to_numeric(sub[col], errors="coerce").dropna()
                rows.append(
                    {
                        "lookback": key[0],
                        "rebalancing_frequency": key[1],
                        "family": family,
                        "metric": col,
                        "n": int(len(vals)),
                        "mean": float(vals.mean()) if len(vals) else float("nan"),
                        "models": ", ".join(sub["model_id"]),
                    }
                )
    return pd.DataFrame(rows)
