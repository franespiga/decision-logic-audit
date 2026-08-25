# --- Volcano data (feature-level) from Dunn/MW posthoc results ----------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def volcano_dataframe_from_dunn(dunn_dict, top_k_labels=15):
    """
    Collapse pairwise posthoc results into one row per feature:
    - p_min: minimum BH-adjusted p-value across regime pairs
    - delta_at_pmin: Cliff's δ for the pair achieving p_min
    - magnitude: magnitude bucket for δ at p_min
    Returns DataFrame with columns:
      Feature, p_min, neglog10p, delta, magnitude, regA, regB
    """
    rows = []
    for (feat, j), df in dunn_dict.items():
        if df.empty or not np.isfinite(df["p_adj"]).any():
            continue
        i_min = df["p_adj"].idxmin()
        pmin = float(df.loc[i_min, "p_adj"])
        delt = float(df.loc[i_min, "delta"])
        mag  = str(df.loc[i_min, "magnitude"])
        rA   = str(df.loc[i_min, "regA"])
        rB   = str(df.loc[i_min, "regB"])
        rows.append((feat, pmin, -np.log10(max(pmin, 1e-300)), delt, mag, rA, rB))
    vol = pd.DataFrame(rows, columns=["Feature","p_min","neglog10p","delta","magnitude","regA","regB"])
    vol.sort_values(["p_min", "Feature"], inplace=True)
    # mark top labels to annotate on the plot
    vol["label_flag"] = False
    vol.loc[vol.index[:min(top_k_labels, len(vol))], "label_flag"] = True
    return vol

def plot_volcano(vol_df, title="Volcano: strongest regime contrast per feature", out_png=None):
    """
    Scatter: x = Cliff's δ, y = -log10(p_min).
    Points at higher y and larger |δ| are the most regime-discriminative.
    """
    x = vol_df["delta"].values
    y = vol_df["neglog10p"].values

    plt.figure(figsize=(9, 5))
    plt.scatter(x, y, s=18)  # no explicit colors per your plotting constraints
    plt.axvline(0.0, linestyle="--", linewidth=1)
    # optional horizontal line for q threshold (rough heuristic: -log10(0.05) ~ 1.3)
    plt.axhline(-np.log10(0.05), linestyle="--", linewidth=1)

    # annotate a subset (label_flag=True)
    for _, r in vol_df[vol_df["label_flag"]].iterrows():
        plt.text(r["delta"], r["neglog10p"], r["Feature"], fontsize=8, ha="left", va="bottom")

    plt.xlabel("Cliff's δ at min q (direction & effect size)")
    plt.ylabel("-log10(min q-value) across regime pairs")
    plt.title(title)
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=800)
        print(f"[saved] {out_png}")
    plt.show()


def plot_delta_heatmap(M, regimes, title="Cliff's δ across regime pairs for feature", out_png=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,5))
    im = plt.imshow(M, aspect="equal", origin="upper")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(regimes)), regimes, rotation=45, ha="right")
    plt.yticks(range(len(regimes)), regimes)
    plt.title(title)
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=150)
        print(f"[saved] {out_png}")
    plt.show()
