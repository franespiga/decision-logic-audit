"""
Orchestration for regime-sensitivity re-analyses of notebook sections 1.2, 3.1.2, 3.2, 5.2, 5.3.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from ablation.comparison import aggregate_attributions_per_regime, kruskal_by_feature, kendall_tau_regime_similarity
from ablation.regime_sensitivity import (
    EVENT_COLORS,
    EVENT_REGIME_ORDER,
    SPEC_DISPLAY,
    align_labels_to_dates,
    compare_temporal_profiles,
    dataframe_to_latex,
    event_markov_crosstab,
    fmt_num,
    latent_separability_metrics,
    linear_probe_summary,
    rank_stability,
    save_figure,
    signed_mean_attributions,
    stress_reference_contrast,
    top_features_by_mean_abs,
)

try:
    from model_comparison.regime_sensitivity import (
        attributions_to_feat_matrix,
        attributions_to_p_attr,
        pack_inputs,
        run_regime_sensitivity,
    )
except Exception:
    from src.model_comparison.regime_sensitivity import (
        attributions_to_feat_matrix,
        attributions_to_p_attr,
        pack_inputs,
        run_regime_sensitivity,
    )


def _labels_for_model(spec_series: pd.Series, dates) -> np.ndarray:
    labs = align_labels_to_dates(spec_series, dates)
    # Replace NaN with a droppable tag for filters
    out = []
    for x in labs:
        if pd.isna(x):
            out.append("Unknown")
        else:
            out.append(str(x))
    return np.asarray(out, dtype=object)


def run_section_1_2_sensitivity(
    specs: Mapping[str, pd.Series],
    attributions,
    feature_names,
    dates,
    n_instruments: int,
    out_dir: Path,
) -> Dict[str, Any]:
    """Reproduce regime-conditioned feature rankings + KW tests under each labelling."""
    out_dir = Path(out_dir)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    results = {}
    ranking_rows = []
    kw_rows = []
    contrast_rows = []

    feats = np.asarray(feature_names).reshape(-1)
    A = attributions

    for spec_name, series in specs.items():
        labs = _labels_for_model(series, dates)
        keep = labs != "Unknown"
        if keep.sum() == 0:
            continue
        # For aggregate_attributions_per_regime, pass date-level labels (function tiles)
        date_labs = labs
        try:
            Mabs, Mpos, Mneg, all_attrs = aggregate_attributions_per_regime(
                torch.as_tensor(A) if not torch.is_tensor(A) else A,
                feats,
                date_labs.tolist(),
            )
        except Exception as exc:
            # Fallback without aggregate helper
            Mabs = Mpos = Mneg = None
            all_attrs = None
            agg_error = str(exc)
        else:
            agg_error = None

        rankings = top_features_by_mean_abs(A, feats, date_labs, n_instruments, top_k=20)
        signed = signed_mean_attributions(A, feats, date_labs, n_instruments)

        kw_df = None
        if all_attrs is not None:
            try:
                kw_df = kruskal_by_feature(
                    all_attrs, date_labs, all_attrs.columns.tolist(), alpha=0.05
                )
            except Exception:
                kw_df = None

        # Stress contrast on mean abs share of top volatility-like features if present
        # Use mean |attribution| mass in crash/high vs bull/low for top-1 feature in stress
        mean_abs_by_reg = {}
        if signed is not None and len(signed):
            for regime, g in signed.groupby("regime"):
                mean_abs_by_reg[regime] = float(np.nanmean(np.abs(g["mean_signed"].values)))
        contrast = stress_reference_contrast(mean_abs_by_reg, spec_name)

        for regime, feats_r in rankings.items():
            for rank, f in enumerate(feats_r, start=1):
                ranking_rows.append(
                    {
                        "specification": spec_name,
                        "regime": regime,
                        "rank": rank,
                        "feature": f,
                    }
                )
        if kw_df is not None:
            tmp = kw_df.copy()
            tmp["specification"] = spec_name
            kw_rows.append(tmp)
        contrast_rows.append({"specification": spec_name, **contrast, "agg_error": agg_error})

        results[spec_name] = {
            "rankings": rankings,
            "signed": signed,
            "kw": kw_df,
            "Mabs": Mabs,
            "contrast": contrast,
        }

    ranking_df = pd.DataFrame(ranking_rows)
    ranking_df.to_csv(out_dir / "data" / "section_1_2_rankings.csv", index=False)
    if kw_rows:
        pd.concat(kw_rows, ignore_index=True).to_csv(
            out_dir / "data" / "section_1_2_kruskal.csv", index=False
        )
    pd.DataFrame(contrast_rows).to_csv(out_dir / "data" / "section_1_2_contrasts.csv", index=False)

    # Stability vs baseline rankings (per overlapping regimes + overall top features pooled)
    stability_rows = []
    base = results.get("baseline_event", {}).get("rankings", {})
    for spec_name, res in results.items():
        if spec_name == "baseline_event":
            continue
        # Compare shared regimes
        for regime in set(base) & set(res["rankings"]):
            st = rank_stability(base[regime], res["rankings"][regime])
            stability_rows.append({"specification": spec_name, "regime": regime, **st})
        # Also overall ranking by concatenating top features across regimes (unique order)
        def pool(rank_map):
            seen = []
            for r in rank_map:
                for f in rank_map[r]:
                    if f not in seen:
                        seen.append(f)
            return seen

        st_all = rank_stability(pool(base), pool(res["rankings"]))
        stability_rows.append({"specification": spec_name, "regime": "ALL_POOLED", **st_all})

        # Sign agreement on shared features for shared regimes
        if spec_name != "conlon_markov_2state":
            b_signed = results["baseline_event"]["signed"]
            a_signed = res["signed"]
            if b_signed is not None and a_signed is not None and len(b_signed) and len(a_signed):
                merged = b_signed.merge(
                    a_signed, on=["regime", "feature"], suffixes=("_base", "_alt")
                )
                if len(merged):
                    agree = np.sign(merged["mean_signed_base"]) == np.sign(merged["mean_signed_alt"])
                    stability_rows.append(
                        {
                            "specification": spec_name,
                            "regime": "SIGNED_AGREEMENT",
                            "sign_agreement": float(agree.mean()),
                        }
                    )

    stab_df = pd.DataFrame(stability_rows)
    stab_df.to_csv(out_dir / "data" / "section_1_2_stability.csv", index=False)
    results["stability"] = stab_df
    results["contrasts"] = pd.DataFrame(contrast_rows)
    return results


def run_section_3_1_2_sensitivity(
    specs: Mapping[str, pd.Series],
    hidden: np.ndarray,
    dates,
    out_dir: Path,
    umap_2d: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    rows = []
    for spec_name, series in specs.items():
        labs = _labels_for_model(series, dates)
        metrics = latent_separability_metrics(hidden, labs)
        rows.append({"specification": spec_name, **metrics})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "data" / "section_3_1_2_separability.csv", index=False)

    # Figure: silhouette comparison
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(df))
    ax.bar(x, df["silhouette"].values, color="#4c72b0", edgecolor="black", width=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([SPEC_DISPLAY.get(s, s) for s in df["specification"]], rotation=20, ha="right")
    ax.set_ylabel("Silhouette score (latent space)")
    ax.set_title("Section 3.1.2 — latent separability under alternative regimes")
    ax.axhline(0, color="grey", lw=0.8)
    fig.tight_layout()
    save_figure(fig, out_dir / "figures" / "F_section_3_1_2_regime_sensitivity")
    plt.close(fig)
    return {"metrics": df}


def run_section_3_2_sensitivity(
    specs: Mapping[str, pd.Series],
    layer_reps: Mapping[str, Any],
    dates,
    out_dir: Path,
) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    detail_rows = []
    summaries = []
    for spec_name, series in specs.items():
        labs = _labels_for_model(series, dates)
        probe = linear_probe_summary(layer_reps, labs, random_state=121187)
        if len(probe):
            probe = probe.copy()
            probe["specification"] = spec_name
            detail_rows.append(probe)
            deepest = probe.iloc[-1]
            summaries.append(
                {
                    "specification": spec_name,
                    "deepest_layer": deepest["layer"],
                    "cv_accuracy_mean": deepest["cv_accuracy_mean"],
                    "cv_accuracy_std": deepest["cv_accuracy_std"],
                    "n_classes": deepest["n_classes"],
                }
            )
        else:
            summaries.append(
                {
                    "specification": spec_name,
                    "deepest_layer": None,
                    "cv_accuracy_mean": np.nan,
                    "cv_accuracy_std": np.nan,
                    "n_classes": np.nan,
                }
            )

    detail = pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame()
    summary = pd.DataFrame(summaries)
    detail.to_csv(out_dir / "data" / "section_3_2_probe_detail.csv", index=False)
    summary.to_csv(out_dir / "data" / "section_3_2_probe_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    if len(summary):
        ax.errorbar(
            np.arange(len(summary)),
            summary["cv_accuracy_mean"],
            yerr=summary["cv_accuracy_std"],
            fmt="o-",
            color="#4c72b0",
            ecolor="black",
            capsize=3,
            label="Deepest-layer CV accuracy",
        )
        ax.set_xticks(np.arange(len(summary)))
        ax.set_xticklabels(
            [SPEC_DISPLAY.get(s, s) for s in summary["specification"]], rotation=20, ha="right"
        )
    ax.set_ylabel("CV accuracy")
    ax.set_title("Section 3.2 — linear-probe accuracy under alternative regimes")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    save_figure(fig, out_dir / "figures" / "F_section_3_2_regime_sensitivity")
    plt.close(fig)
    return {"detail": detail, "summary": summary}


def _mean_abs_occlusion_curve(occlusion_curves, labels, regime: str) -> np.ndarray:
    curves = occlusion_curves
    if hasattr(curves, "detach"):
        curves = curves.detach().cpu().numpy()
    curves = np.asarray(curves, dtype=float)
    # (batch, action, lags) -> mean abs over actions
    if curves.ndim == 3:
        c = np.abs(curves).mean(axis=1)
    else:
        c = np.abs(curves)
    mask = np.asarray(labels) == regime
    if mask.sum() == 0:
        return np.full(c.shape[-1], np.nan)
    return c[mask].mean(axis=0)


def run_section_5_2_sensitivity(
    specs: Mapping[str, pd.Series],
    occlusions_best: dict,
    occlusions_worst: dict,
    dates_best,
    dates_worst,
    out_dir: Path,
) -> Dict[str, Any]:
    """
    Temporal profiles by regime for best vs worst.
    Uses already-computed occlusion curves; only re-aggregates by labels.
    """
    out_dir = Path(out_dir)
    rows = []
    profile_rows = []
    for spec_name, series in specs.items():
        labs_b = _labels_for_model(series, dates_best)
        labs_w = _labels_for_model(series, dates_worst)
        regimes = sorted(set(labs_b) | set(labs_w))
        regimes = [r for r in regimes if r != "Unknown"]
        for regime in regimes:
            cb = _mean_abs_occlusion_curve(occlusions_best["occlusion_curves"], labs_b, regime)
            cw = _mean_abs_occlusion_curve(occlusions_worst["occlusion_curves"], labs_w, regime)
            cmp_ = compare_temporal_profiles(cb, cw)
            rows.append(
                {
                    "specification": spec_name,
                    "regime": regime,
                    "best_recency": cmp_["recency_a"],
                    "worst_recency": cmp_["recency_b"],
                    "best_minus_worst_recency": cmp_["recency_a"] - cmp_["recency_b"],
                    "corr_best_worst_curves": cmp_["corr"],
                    "peak_lag_best": temporal_peak(cb),
                    "peak_lag_worst": temporal_peak(cw),
                }
            )
            for lag, (vb, vw) in enumerate(zip(np.nan_to_num(cb), np.nan_to_num(cw))):
                profile_rows.append(
                    {
                        "specification": spec_name,
                        "regime": regime,
                        "lag": lag,
                        "best": vb,
                        "worst": vw,
                    }
                )

        # Direction: best more recent than worst in stress regime?
        stress = "high_volatility" if spec_name == "conlon_markov_2state" else "crash_period"
        sub = [r for r in rows if r["specification"] == spec_name and r["regime"] == stress]
        if sub:
            # lower recency_index => more weight on recent lags
            direction_best_more_recent = sub[0]["best_recency"] < sub[0]["worst_recency"]
            rows.append(
                {
                    "specification": spec_name,
                    "regime": f"DIRECTION_{stress}",
                    "best_more_recent_than_worst": direction_best_more_recent,
                    "best_minus_worst_recency": sub[0]["best_minus_worst_recency"],
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "data" / "section_5_2_temporal_by_regime.csv", index=False)
    pd.DataFrame(profile_rows).to_csv(
        out_dir / "data" / "section_5_2_curves.csv", index=False
    )

    # Figure: best-worst recency contrast in stress / reference
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    plot_df = df[df["regime"].isin(list(EVENT_REGIME_ORDER) + list(["high_volatility", "low_volatility"]))].copy()
    if len(plot_df):
        # focus on stress regimes + bull/low
        focus = plot_df[
            plot_df["regime"].isin(
                ["crash_period", "bull_market", "high_volatility", "low_volatility"]
            )
        ]
        for spec, g in focus.groupby("specification"):
            ax.plot(
                g["regime"],
                g["best_minus_worst_recency"],
                marker="o",
                label=SPEC_DISPLAY.get(spec, spec),
            )
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_ylabel("Recency index (best $-$ worst)")
    ax.set_title("Section 5.2 — best vs worst recency contrast by regime")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    save_figure(fig, out_dir / "figures" / "F_section_5_2_regime_sensitivity")
    plt.close(fig)
    return {"table": df}


def temporal_peak(curve: np.ndarray) -> float:
    c = np.asarray(curve, dtype=float)
    if not np.isfinite(c).any():
        return np.nan
    c = np.nan_to_num(c, nan=0.0)
    return float(np.argmax(c))


def run_section_5_3_sensitivity(
    specs: Mapping[str, pd.Series],
    attributions_best,
    attributions_worst,
    feature_names_best,
    feature_names_worst,
    occlusions_best: dict,
    occlusions_worst: dict,
    dates_best,
    dates_worst,
    indicators,
    out_dir: Path,
    n_instruments: int,
) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    rows = []
    rank_rows = []
    for spec_name, series in specs.items():
        labs_b = _labels_for_model(series, dates_best)
        # Expanded labels for pack_inputs (dates × instruments in occlusion curves)
        n_inst = occlusions_best["occlusion_curves"].shape[1]
        expanded = np.repeat(labs_b, n_inst)
        occ_b = occlusions_best["occlusion_curves"]
        occ_w = occlusions_worst["occlusion_curves"]
        if hasattr(occ_b, "detach"):
            occ_b = occ_b.detach().cpu().numpy()
            occ_w = occ_w.detach().cpu().numpy()
        # reshape like notebook
        occ_b2 = np.asarray(occ_b).reshape(-1, occ_b.shape[-1])
        occ_w2 = np.asarray(occ_w).reshape(-1, occ_w.shape[-1])
        # filter Unknown
        keep = expanded != "Unknown"
        if keep.sum() < 50:
            continue
        try:
            inputs = pack_inputs(
                expanded[keep],
                occ_b2[keep],
                occ_w2[keep],
                attributions_to_p_attr(attributions_best)[keep]
                if attributions_to_p_attr(attributions_best).shape[0] == len(expanded)
                else None,
                attributions_to_p_attr(attributions_worst)[keep]
                if attributions_to_p_attr(attributions_worst).shape[0] == len(expanded)
                else None,
                attributions_to_feat_matrix(attributions_best)[keep]
                if attributions_to_feat_matrix(attributions_best).shape[0] == len(expanded)
                else None,
                attributions_to_feat_matrix(attributions_worst)[keep]
                if attributions_to_feat_matrix(attributions_worst).shape[0] == len(expanded)
                else None,
                feature_names=indicators,
            )
            # pack_inputs signature may differ - try without optional kwargs issues
        except TypeError:
            inputs = pack_inputs(
                expanded[keep],
                occ_b2[keep],
                occ_w2[keep],
                feature_names=list(indicators) if indicators is not None else None,
            )
        except Exception as exc:
            rows.append({"specification": spec_name, "error": str(exc)})
            continue

        try:
            res = run_regime_sensitivity(inputs)
        except Exception as exc:
            rows.append({"specification": spec_name, "error": str(exc)})
            continue

        # Feature rank stability within model across regimes: mean Kendall of ranks
        fr_best = res.get("feature_ranks_best")
        if fr_best is not None and len(fr_best.columns) >= 2:
            num = fr_best.select_dtypes(include=[np.number])
            cols = list(num.columns)
            taus = []
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    taus.append(stats_kendall(num[cols[i]], num[cols[j]]))
            mean_tau = float(np.nanmean(taus)) if taus else np.nan
        else:
            mean_tau = np.nan

        # Recency interaction: crash/high - bull/low for best and worst
        raw = res.get("raw_dfs", {})
        rec_contrast = {}
        for key, model in [("occ_best", "best"), ("occ_worst", "worst")]:
            if key not in raw:
                continue
            df = raw[key]
            means = df.groupby("regime")["recency_index"].mean().to_dict()
            rec_contrast[model] = stress_reference_contrast(means, spec_name)

        rows.append(
            {
                "specification": spec_name,
                "mean_kendall_feature_ranks_best": mean_tau,
                "best_stress_recency": rec_contrast.get("best", {}).get("stress_value", np.nan),
                "best_ref_recency": rec_contrast.get("best", {}).get("reference_value", np.nan),
                "best_recency_contrast": rec_contrast.get("best", {}).get("contrast", np.nan),
                "worst_stress_recency": rec_contrast.get("worst", {}).get("stress_value", np.nan),
                "worst_ref_recency": rec_contrast.get("worst", {}).get("reference_value", np.nan),
                "worst_recency_contrast": rec_contrast.get("worst", {}).get("contrast", np.nan),
            }
        )
        if fr_best is not None:
            tmp = fr_best.copy()
            tmp["specification"] = spec_name
            tmp["model"] = "best"
            rank_rows.append(tmp.reset_index())

        # Also KW-style for worst model attributions by regime (notebook cell 184)
        try:
            labs_w = _labels_for_model(series, dates_worst)
            Mabs, _, _, all_attrs = aggregate_attributions_per_regime(
                torch.as_tensor(attributions_worst)
                if not torch.is_tensor(attributions_worst)
                else attributions_worst,
                np.asarray(feature_names_worst).reshape(-1),
                labs_w.tolist(),
            )
            kw = kruskal_by_feature(all_attrs, labs_w, all_attrs.columns.tolist(), alpha=0.05)
            n_sig = int(((kw["sig"]) & (kw["q"] < 0.05)).sum())
            rows[-1]["n_sig_features_worst_kw"] = n_sig
        except Exception:
            rows[-1]["n_sig_features_worst_kw"] = np.nan

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "data" / "section_5_3_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    if len(df) and "best_recency_contrast" in df.columns:
        x = np.arange(len(df))
        w = 0.35
        ax.bar(
            x - w / 2,
            df["best_recency_contrast"],
            width=w,
            label="Best (stress$-$ref)",
            color="#4c72b0",
            edgecolor="black",
        )
        ax.bar(
            x + w / 2,
            df["worst_recency_contrast"],
            width=w,
            label="Worst (stress$-$ref)",
            color="#dd8452",
            edgecolor="black",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [SPEC_DISPLAY.get(s, s) for s in df["specification"]], rotation=20, ha="right"
        )
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_ylabel("Recency contrast (stress $-$ reference)")
    ax.set_title("Section 5.3 — regime sensitivity of temporal focus")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(fig, out_dir / "figures" / "F_section_5_3_regime_sensitivity")
    plt.close(fig)
    return {"summary": df}


def stats_kendall(a, b) -> float:
    from scipy.stats import kendalltau

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return np.nan
    return float(kendalltau(a[mask], b[mask]).correlation or np.nan)


def plot_section_1_2_figure(sec12: Dict[str, Any], out_dir: Path) -> None:
    out_dir = Path(out_dir)
    stab = sec12.get("stability")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    if stab is not None and len(stab):
        sub = stab[stab["regime"] == "ALL_POOLED"].copy()
        axes[0].bar(
            np.arange(len(sub)),
            sub["kendall_tau"],
            color="#4c72b0",
            edgecolor="black",
        )
        axes[0].set_xticks(np.arange(len(sub)))
        axes[0].set_xticklabels(
            [SPEC_DISPLAY.get(s, s) for s in sub["specification"]], rotation=20, ha="right"
        )
        axes[0].set_ylabel("Kendall $\\tau$ vs baseline rankings")
        axes[0].set_title("Pooled feature-rank stability")
        axes[0].set_ylim(-1, 1)

        sub2 = stab[stab["regime"] == "SIGNED_AGREEMENT"].copy()
        if len(sub2):
            axes[1].bar(
                np.arange(len(sub2)),
                sub2["sign_agreement"],
                color="#55a868",
                edgecolor="black",
            )
            axes[1].set_xticks(np.arange(len(sub2)))
            axes[1].set_xticklabels(
                [SPEC_DISPLAY.get(s, s) for s in sub2["specification"]],
                rotation=20,
                ha="right",
            )
            axes[1].set_ylim(0, 1.05)
            axes[1].set_ylabel("Sign agreement")
            axes[1].set_title("Signed-attribution agreement vs baseline")
        else:
            axes[1].axis("off")
    fig.suptitle("Section 1.2 — regime-definition sensitivity", y=1.02)
    fig.tight_layout()
    save_figure(fig, out_dir / "figures" / "F_section_1_2_regime_sensitivity")
    plt.close(fig)
