from typing import Dict, Optional, Tuple, List
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import silhouette_score, roc_auc_score, r2_score
import matplotlib.pyplot as plt
import sys
sys.path.append('../..')

# ---- Reuse your provided functions via import or same notebook cell ----
from regime_representation.transformations import (
    extract_layer_representations, create_latent_embeddings,
    compute_layer_conductance, compute_predictive_bins
)
from regime_representation.assessments import (
    evaluate_linear_probe_accuracy,
    assess_predictive_calibration, compute_regime_separability_metrics)


# ---------- Small helpers ----------
def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _std_layer_reps(layer_reps: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    out = {}
    for k, Z in layer_reps.items():
        X = _to_numpy(Z)
        scaler = StandardScaler()
        out[k] = scaler.fit_transform(X)
    return out

def _select_layers(layer_reps: Dict[str, np.ndarray], keep: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
    if keep is None:
        return layer_reps
    return {k: v for k, v in layer_reps.items() if k in keep}

def _best_layer_by_metric(layer_scores: Dict[str, float]) -> Optional[str]:
    if len(layer_scores) == 0:
        return None
    return max(layer_scores.items(), key=lambda kv: (kv[1] if np.isfinite(kv[1]) else -np.inf))[0]

# ---------- Generic linear probes ----------
def probe_classification_curves(
    layer_reps_std: Dict[str, np.ndarray],
    y: np.ndarray,
    cv_splits: int = 5,
    metric: str = "accuracy",
    random_state: int = 42
) -> pd.DataFrame:
    """
    Run a simple logistic regression probe on each layer representation.
    y can be regimes, future bins, or past bins (integer labels).
    metric ∈ {"accuracy","roc_auc"}; roc_auc uses OVR with macro average if #classes>2.
    """
    rows = []
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    for layer, X in layer_reps_std.items():
        clf = LogisticRegression(max_iter=1000, random_state=random_state)
        if metric == "accuracy":
            scores = cross_val_score(clf, X, y, cv=skf, scoring="accuracy")
            val = float(scores.mean()); std = float(scores.std())
        elif metric == "roc_auc":
            # roc_auc handling via cross_val is trickier; fallback: manual CV
            aucs = []
            for tr, te in skf.split(X, y):
                clf.fit(X[tr], y[tr])
                prob = clf.predict_proba(X[te])
                if prob.shape[1] == 2:
                    auc = roc_auc_score(y[te], prob[:,1])
                else:
                    # OVR macro
                    auc = roc_auc_score(pd.get_dummies(y[te]).values, prob, average="macro", multi_class="ovr")
                aucs.append(auc)
            val = float(np.mean(aucs)); std = float(np.std(aucs))
        else:
            raise ValueError("metric must be 'accuracy' or 'roc_auc'")

        rows.append({"layer": layer, "metric": metric, "mean": val, "std": std})
    df = pd.DataFrame(rows).sort_values("layer")
    return df

def probe_regression_curves(
    layer_reps_std: Dict[str, np.ndarray],
    y_cont: np.ndarray,
    cv_splits: int = 5,
    random_state: int = 42,
    scorer: str = "r2"
) -> pd.DataFrame:
    """
    Ridge regression probe; default scorer is R^2 via cross_val_score.
    """
    rows = []
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    for layer, X in layer_reps_std.items():
        mdl = Ridge(alpha=1.0, random_state=random_state)
        scores = cross_val_score(mdl, X, y_cont, cv=kf, scoring=scorer)
        rows.append({"layer": layer, "metric": scorer, "mean": float(scores.mean()), "std": float(scores.std())})
    return pd.DataFrame(rows).sort_values("layer")

# ---------- Projection + silhouette ----------
def umap_per_layer(layer_reps_std: Dict[str, np.ndarray], n_neighbors: int = 15, min_dist: float = 0.1, random_state: int = 42) -> Dict[str, np.ndarray]:
    import umap
    embeds = {}
    for k, X in layer_reps_std.items():
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
        embeds[k] = reducer.fit_transform(X)
    return embeds

def silhouette_per_layer(X_dict: Dict[str, np.ndarray], labels: np.ndarray, metric: str = "euclidean") -> Dict[str, float]:
    out = {}
    for k, X in X_dict.items():
        try:
            out[k] = float(silhouette_score(X, labels, metric=metric))
        except Exception:
            out[k] = np.nan
    return out


def compare_conductance(best_df: pd.DataFrame, worst_df: pd.DataFrame) -> pd.DataFrame:
    """
    Align layers by index and compute deltas; attach concentration metrics.
    """
    def _idx(s): return int(str(s).split("_")[-1])

    B = best_df.copy(); B["layer_idx"] = B["layer"].map(_idx)
    W = worst_df.copy(); W["layer_idx"] = W["layer"].map(_idx)
    M = pd.merge(B[["layer_idx","value","share"]], W[["layer_idx","value","share"]],
                 on="layer_idx", suffixes=("_best","_worst"), how="outer").sort_values("layer_idx")
    M["delta_value"] = M["value_best"] - M["value_worst"]
    M["delta_share"] = M["share_best"] - M["share_worst"]
    M.attrs["best_entropy"] = best_df.attrs.get("entropy", np.nan)
    M.attrs["worst_entropy"] = worst_df.attrs.get("entropy", np.nan)
    M.attrs["best_hhi"] = best_df.attrs.get("hhi", np.nan)
    M.attrs["worst_hhi"] = worst_df.attrs.get("hhi", np.nan)
    return M

def compare_silhouettes(best: Dict[str, float], worst: Dict[str, float]) -> pd.DataFrame:
    keys = sorted(set(best.keys()).union(worst.keys()))
    rows = []
    for k in keys:
        b = best.get(k, np.nan)
        w = worst.get(k, np.nan)
        rows.append({"layer": k, "best": b, "worst": w, "delta": (b - w) if (np.isfinite(b) and np.isfinite(w)) else np.nan})
    return pd.DataFrame(rows).sort_values("layer")

def diff_probe_curves(df_best: pd.DataFrame, df_worst: pd.DataFrame) -> pd.DataFrame:
    M = pd.merge(df_best, df_worst, on=["layer","metric"], suffixes=("_best","_worst"), how="inner")
    M["delta_mean"] = M["mean_best"] - M["mean_worst"]
    return M.sort_values("layer")



# ---------- Conductance normalization & concentration ----------
def normalize_conductance(cond: Dict[str, float]) -> pd.DataFrame:
    df = pd.DataFrame([{"layer": k, "value": float(v)} for k, v in cond.items()]).sort_values(
        key=lambda s: s.apply(lambda x: int(str(x).split("_")[-1])), by="layer"
    )
    total = df["value"].sum()
    df["share"] = df["value"] / (total if total > 0 else 1.0)
    # concentration
    p = df["share"].to_numpy()
    p = p / (p.sum() if p.sum() > 0 else 1.0)
    entropy = -np.sum(np.clip(p, 1e-12, 1.0) * np.log(np.clip(p, 1e-12, 1.0)))
    hhi = float(np.sum(p**2))
    df.attrs["entropy"] = float(entropy)
    df.attrs["hhi"] = hhi
    return df

# ---------- Per-model runner ----------
def run_latent_model(
    model,
    states: torch.Tensor,
    actions: torch.Tensor,
    returns_to_go: torch.Tensor,
    timesteps: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    regimes: np.ndarray,              # (B,)
    future_returns: np.ndarray,       # (B,)
    past_signals: Optional[np.ndarray] = None,  # (B, P) or (B,) optional
    projection_neighbors: int = 15,
    projection_min_dist: float = 0.1,
    random_state: int = 42
) -> Dict:
    """
    Runs: layer conductance, hidden states (last-token pooling), standardization,
    UMAP per layer, silhouette (regime/future bins/past bins), probes (classif/regress),
    and calibration on the last layer.
    """
    # 1) Hidden states per layer (your function)
    layer_reps_t = extract_layer_representations(
        model, states, actions, returns_to_go, timesteps, attention_mask=attention_mask, target_layers=None
    )  # dict layer -> torch (B, d)
    # Standardize
    layer_reps_std = _std_layer_reps(layer_reps_t)  # dict layer -> np (B, d)

    # 2) Conductance (your function)
    conductance_raw = compute_layer_conductance(
        model, states, actions, returns_to_go, timesteps, attention_mask=attention_mask,
        normalize_by_depth=True, target_class=None
    )  # dict layer -> float
    conductance_df = normalize_conductance(conductance_raw)  # with entropy/hhi in attrs

    # 3) Projections (UMAP) per layer & silhouettes
    embeds = umap_per_layer(layer_reps_std, n_neighbors=projection_neighbors, min_dist=projection_min_dist, random_state=random_state)

    # Binning labels
    # future bins (e.g., deciles)
    fut_bins, fut_edges, fut_centers = compute_predictive_bins(future_returns, next(iter(layer_reps_std.values())), n_bins=10, method='quantile')
    # past bins (if provided)
    if past_signals is not None:
        P = _to_numpy(past_signals)
        if P.ndim == 2 and P.shape[1] > 1:
            pc1 = PCA(n_components=1, random_state=random_state).fit_transform(P).ravel()
            past_scalar = pc1
        else:
            past_scalar = P.ravel()
        past_bins, past_edges, past_centers = compute_predictive_bins(past_scalar, next(iter(layer_reps_std.values())), n_bins=10, method='quantile')
    else:
        past_bins = None

    sil_regime_hd = silhouette_per_layer(layer_reps_std, regimes, metric="euclidean")
    sil_future_hd = silhouette_per_layer(layer_reps_std, fut_bins, metric="euclidean")
    sil_past_hd   = silhouette_per_layer(layer_reps_std, past_bins, metric="euclidean") if past_bins is not None else {}

    # silhouettes on 2D maps (optional, for figure panels)
    sil_regime_2d = silhouette_per_layer(embeds, regimes, metric="euclidean")
    sil_future_2d = silhouette_per_layer(embeds, fut_bins, metric="euclidean")
    sil_past_2d   = silhouette_per_layer(embeds, past_bins, metric="euclidean") if past_bins is not None else {}

    # 4) Probes
    # Classification probes
    probe_regime_acc = probe_classification_curves(layer_reps_std, regimes, metric="accuracy")
    probe_future_acc = probe_classification_curves(layer_reps_std, fut_bins, metric="accuracy")
    probe_past_acc   = probe_classification_curves(layer_reps_std, past_bins, metric="accuracy") if past_bins is not None else None
    # Regression probes (future returns continuous)
    probe_future_r2  = probe_regression_curves(layer_reps_std, future_returns, scorer="r2")

    # 5) Calibration on last layer (your calibration function expects NxD)
    last_layer = sorted(layer_reps_std.keys(), key=lambda s: int(s.split("_")[-1]))[-1] if layer_reps_std else None
    calib = assess_predictive_calibration(layer_reps_std[last_layer], future_returns, fut_bins, fut_centers) if last_layer is not None else {}

    # 6) Choose “best” layers for visual focus
    # (a) by regime silhouette in HD, (b) by future bin silhouette in HD
    best_by_regime = _best_layer_by_metric(sil_regime_hd)
    best_by_future = _best_layer_by_metric(sil_future_hd)

    return {
        "layer_reps_std": layer_reps_std,
        "embeds_2d": embeds,
        "conductance_df": conductance_df,
        "sil_hd": {
            "regime": sil_regime_hd,
            "future": sil_future_hd,
            "past": sil_past_hd
        },
        "sil_2d": {
            "regime": sil_regime_2d,
            "future": sil_future_2d,
            "past": sil_past_2d
        },
        "probes": {
            "regime_acc": probe_regime_acc,
            "future_acc": probe_future_acc,
            "past_acc": probe_past_acc,
            "future_r2": probe_future_r2
        },
        "calibration": calib,
        "bins": {
            "future": {"bins": fut_bins, "edges": fut_edges, "centers": fut_centers},
            "past":   {"bins": past_bins} if past_bins is not None else None
        },
        "focus_layers": {
            "last": last_layer,
            "best_by_regime": best_by_regime,
            "best_by_future": best_by_future
        }
    }

# ---------- Plot: Conductance profiles ----------
def plot_conductance_profiles(best_df: pd.DataFrame, worst_df: pd.DataFrame,
                              savepath: Optional[str] = None, filename: str = "F54_conductance.png",
                              title: str = "Layer conductance profiles (share)"):

    def _idx(s): return int(str(s).split("_")[-1])

    B = best_df.copy(); B["layer_idx"] = B["layer"].map(_idx)
    W = worst_df.copy(); W["layer_idx"] = W["layer"].map(_idx)

    xB, yB = B["layer_idx"].to_numpy(), B["share"].to_numpy()
    xW, yW = W["layer_idx"].to_numpy(), W["share"].to_numpy()

    plt.figure(figsize=(8,4.5))
    plt.plot(xB, yB, marker="o", label="Best")
    plt.plot(xW, yW, marker="o", label="Worst")
    plt.xlabel("Layer index (depth)")
    plt.ylabel("Conductance share")
    be, bh = best_df.attrs.get("entropy", np.nan), best_df.attrs.get("hhi", np.nan)
    we, wh = worst_df.attrs.get("entropy", np.nan), worst_df.attrs.get("hhi", np.nan)
    plt.title(f"{title}\nConcentration — Best: H={be:.2f}, HHI={bh:.2f} | Worst: H={we:.2f}, HHI={wh:.2f}")
    plt.legend()
    plt.tight_layout()
    if savepath:
        import os; os.makedirs(savepath, exist_ok=True)
        plt.savefig(f"{savepath}/{filename}", dpi=800, bbox_inches="tight")
    plt.show()

# ---------- Plot: UMAP scatter (2D) ----------
def plot_umap_scatter(emb: np.ndarray,
                      colors: np.ndarray,
                      color_map: dict,
                      title: str, savepath: Optional[str] = None, filename: str = "F54_umap.png"):
    plt.figure(figsize=(8, 6))

    for k,v in color_map.items():
        mask = colors == k

        plt.scatter(emb[mask,0],
                    emb[mask,1], s=10,
                    c=v,
                    label=k, alpha=0.8)
    plt.title(title)
    plt.xticks([]); plt.yticks([])
    plt.legend(markerscale=2, frameon=False, bbox_to_anchor=(1.02,1), loc="upper left")
    plt.tight_layout()
    if savepath:
        import os; os.makedirs(savepath, exist_ok=True)
        plt.savefig(f"{savepath}/{filename}", dpi=800, bbox_inches="tight")
    plt.show()




# ---------- Plot: Probe vs depth (future vs past vs regime) ----------
def plot_probe_vs_depth(best_curves: Dict[str, pd.DataFrame], worst_curves: Dict[str, pd.DataFrame],
                        metric_key_best: str = "future_acc", metric_key_worst: str = "future_acc",
                        ylabel: str = "Accuracy",
                        savepath: Optional[str] = None, filename: str = "F54_probes.png",
                        title: str = "Probe performance vs depth"):
    """
    metric_key_* ∈ {"regime_acc","future_acc","past_acc","future_r2"}; choose the same for best & worst.
    """
    B = best_curves[metric_key_best].copy()
    W = worst_curves[metric_key_worst].copy()
    def _idx(s): return int(str(s).split("_")[-1])
    B["idx"] = B["layer"].map(_idx); W["idx"] = W["layer"].map(_idx)

    # Allow either accuracy (0-1) or R^2 (0-1)
    yb = B["mean"].to_numpy(); yw = W["mean"].to_numpy()

    plt.figure(figsize=(8,4.5))
    plt.plot(B["idx"], yb, marker="o", label=f"Best ({metric_key_best})")
    plt.plot(W["idx"], yw, marker="o", label=f"Worst ({metric_key_worst})")
    plt.xlabel("Layer index (depth)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if savepath:
        import os; os.makedirs(savepath, exist_ok=True)
        plt.savefig(f"{savepath}/{filename}", dpi=800, bbox_inches="tight")
    plt.show()

# ---------- Plot: Silhouette bars at focus layer ----------
def plot_silhouette_bars(best_vals: Dict[str, float], worst_vals: Dict[str, float],
                         label_order: List[str], title: str,
                         savepath: Optional[str] = None, filename: str = "F54_silhouette.png"):
    x = np.arange(len(label_order))
    b = [best_vals.get(k, np.nan) for k in label_order]
    w = [worst_vals.get(k, np.nan) for k in label_order]
    width = 0.4

    plt.figure(figsize=(6.5,4.5))
    plt.bar(x - width/2, b, width=width, label="Best")
    plt.bar(x + width/2, w, width=width, label="Worst")
    plt.xticks(x, label_order, rotation=0)
    plt.ylabel("Silhouette")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if savepath:
        import os; os.makedirs(savepath, exist_ok=True)
        plt.savefig(f"{savepath}/{filename}", dpi=800, bbox_inches="tight")
    plt.show()

# ---------- Plot: Calibration curve ----------
def plot_calibration_curve(calib: Dict, title: str, savepath: Optional[str] = None, filename: str = "F54_calib.png"):
    centers = calib.get("bin_centers"); curve = calib.get("calibration_curve"); counts = calib.get("bin_counts")
    if centers is None or curve is None:
        print("Calibration dict missing keys.")
        return
    plt.figure(figsize=(6,4.5))
    plt.plot(centers, curve, marker="o")
    plt.plot([centers.min(), centers.max()], [centers.min(), centers.max()], linestyle="--")  # ideal line
    plt.xlabel("Expected future return (bin center)")
    plt.ylabel("Observed mean future return")
    mce = calib.get("mean_calibration_error", np.nan)
    wce = calib.get("weighted_calibration_error", np.nan)
    plt.title(f"{title}\nMCE={mce:.4g}, WMCE={wce:.4g}")
    plt.tight_layout()
    if savepath:
        import os; os.makedirs(savepath, exist_ok=True)
        plt.savefig(f"{savepath}/{filename}", dpi=800, bbox_inches="tight")
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict
import matplotlib.colors as mcolors


def generate_color_palette(unique_values, existing_color_map=None):
    """
    Generate a color palette for unique values.

    Parameters:
    -----------
    unique_values : array-like
        Unique values that need colors
    existing_color_map : dict, optional
        Existing color mapping to extend

    Returns:
    --------
    dict : Complete color mapping
    """
    if existing_color_map is None:
        existing_color_map = {}

    # Get values that don't have colors assigned
    missing_values = [val for val in unique_values if val not in existing_color_map]

    if not missing_values:
        return existing_color_map

    # Generate colors for missing values using a color palette
    # Use tab10 for up to 10 colors, then cycle through other palettes
    if len(missing_values) <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, len(missing_values)))
    elif len(missing_values) <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, len(missing_values)))
    else:
        # For more colors, use a continuous colormap
        colors = plt.cm.viridis(np.linspace(0, 1, len(missing_values)))

    # Create complete color map
    complete_color_map = existing_color_map.copy()
    for i, val in enumerate(missing_values):
        complete_color_map[val] = colors[i]

    return complete_color_map


def plot_umap_scatter_subplot(ax, emb: np.ndarray, colors: np.ndarray,
                              color_map: dict, title: str):
    """
    Plot UMAP scatter on a given axis (for subplots).

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axis to plot on
    emb : np.ndarray
        2D embedding coordinates
    colors : np.ndarray
        Color labels for each point
    color_map : dict
        Mapping from color labels to actual colors
    title : str
        Title for the subplot
    """
    for k, v in color_map.items():
        mask = colors == k
        if mask.sum() > 0:  # Only plot if there are points with this label
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       s=10, c=v, label=k, alpha=0.4)

    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_umap_grid(best_res: dict, worst_res: dict, regime_lb: np.ndarray,
                   color_map: dict, savepath: Optional[str] = None,
                   filename: str = "umap_grid.png"):
    """
    Create a 4x2 grid of UMAP plots with a single shared legend.

    Parameters:
    -----------
    best_res : dict
        Results dictionary for best model containing 'embeds_2d'
    worst_res : dict
        Results dictionary for worst model containing 'embeds_2d'
    regime_lb : np.ndarray
        Regime labels for coloring points
    color_map : dict
        Color mapping dictionary (value -> color)
    savepath : str, optional
        Path to save the figure
    filename : str
        Filename for saved figure
    """
    # Get layers from best_res
    layers = list(best_res["embeds_2d"].keys())

    # Ensure we have enough layers for 4 rows (if not, repeat the last layer)
    while len(layers) < 4:
        layers.append(layers[-1])

    # Take only first 4 layers for 4 rows
    layers = layers[:4]

    # Generate complete color palette if needed
    unique_colors = np.unique(regime_lb)
    complete_color_map = generate_color_palette(unique_colors, color_map)

    # Create figure with 4 rows and 2 columns
    fig, axes = plt.subplots(4, 2, figsize=(12, 16))

    # Plot each layer for both best and worst models
    for i, layer in enumerate(layers):
        # Best model (left column)
        plot_umap_scatter_subplot(
            axes[i, 0],
            best_res["embeds_2d"][layer],
            regime_lb,
            complete_color_map,
            f"Best — UMAP ({layer})"
        )

        # Worst model (right column)
        plot_umap_scatter_subplot(
            axes[i, 1],
            worst_res["embeds_2d"][layer],
            regime_lb,
            complete_color_map,
            f"Worst — UMAP ({layer})"
        )

    # Create a single legend for the entire figure
    # Get legend elements from the first subplot
    handles, labels = axes[0, 0].get_legend_handles_labels()

    # Create legend below the subplots
    fig.legend(handles, labels, loc='lower center',
               bbox_to_anchor=(0.5, -0.02), ncol=len(labels),
               markerscale=2, frameon=False)

    # Adjust layout to make room for legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)

    # Save if path provided
    if savepath:
        import os
        os.makedirs(savepath, exist_ok=True)
        plt.savefig(f"{savepath}/{filename}", dpi=800, bbox_inches="tight")

    plt.show()
