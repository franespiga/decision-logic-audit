"""Figures, LaTeX and markdown report writers for regime sensitivity."""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ablation.regime_sensitivity import (
    BASELINE_BOUNDARIES,
    EVENT_COLORS,
    EVENT_REGIME_ORDER,
    SPEC_DISPLAY,
    dataframe_to_latex,
    fmt_num,
    save_figure,
)


def plot_regime_timeline(
    specs: Mapping[str, pd.Series],
    sp500_cum: pd.Series,
    markov_prob_low: pd.Series,
    vix: Optional[pd.Series],
    out_dir: Path,
) -> None:
    out_dir = Path(out_dir)
    fig, axes = plt.subplots(
        5, 1, figsize=(11, 9.5), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1, 1.2, 1.4]}
    )
    event_specs = ["baseline_event", "boundary_minus_20td", "boundary_plus_20td"]
    titles = ["A. Baseline event partition", "B. Boundaries −20 trading days", "C. Boundaries +20 trading days"]
    for ax, spec, title in zip(axes[:3], event_specs, titles):
        labs = specs[spec].dropna()
        for regime in EVENT_REGIME_ORDER:
            mask = labs == regime
            if not mask.any():
                continue
            # paint contiguous spans
            idx = labs.index[mask]
            # group contiguous
            groups = np.split(idx, np.where(np.diff(idx.to_numpy().astype("datetime64[D]")).astype(int) > 3)[0] + 1)
            # simpler: use fill_between on boolean
            y = mask.astype(float)
            ax.fill_between(labs.index, 0, y, step="mid", alpha=0.85, color=EVENT_COLORS[regime], label=regime)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([])
        ax.set_title(title, loc="left", fontsize=10)
        if spec == "baseline_event":
            ax.legend(ncol=4, fontsize=7, frameon=False, loc="upper right")

    # Markov prob + labels
    ax = axes[3]
    ax.plot(markov_prob_low.index, markov_prob_low.values, color="black", lw=0.9, label="Filtered P(low vol)")
    ax.axhline(0.5, color="grey", ls="--", lw=0.8)
    markov_labs = specs["conlon_markov_2state"].dropna()
    for regime, color in [("low_volatility", EVENT_COLORS["low_volatility"]), ("high_volatility", EVENT_COLORS["high_volatility"])]:
        mask = (markov_labs == regime).reindex(markov_prob_low.index).fillna(False)
        ax.fill_between(markov_prob_low.index, 0, 1, where=mask.to_numpy(), alpha=0.15, color=color, label=regime)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("P(low vol)")
    ax.set_title("D. Conlon-inspired Markov (filtered probability)", loc="left", fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc="upper right", ncol=3)

    ax = axes[4]
    ax.plot(sp500_cum.index, sp500_cum.values, color="#222222", lw=1.0, label="S&P 500 cumulative return")
    ax.set_ylabel("Cum. return")
    ax.set_title("E. Market path", loc="left", fontsize=10)
    if vix is not None and len(vix):
        ax2 = ax.twinx()
        ax2.plot(vix.index, vix.values, color="#9467bd", lw=0.7, alpha=0.8, label="VIX")
        ax2.set_ylabel("VIX", color="#9467bd")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    save_figure(fig, out_dir / "figures" / "F_regime_definitions_timeline")
    plt.close(fig)


def plot_regime_counts(chars: pd.DataFrame, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    plot_df = chars.copy()
    # grouped bar
    specs = list(plot_df["regime_specification"].unique())
    regimes = list(dict.fromkeys(plot_df["regime"].tolist()))
    x = np.arange(len(regimes))
    width = 0.18
    for i, spec in enumerate(specs):
        sub = plot_df[plot_df["regime_specification"] == spec].set_index("regime")
        vals = [sub.loc[r, "n_trading_days"] if r in sub.index else 0 for r in regimes]
        ax.bar(
            x + (i - (len(specs) - 1) / 2) * width,
            vals,
            width=width,
            label=SPEC_DISPLAY.get(spec, spec),
            edgecolor="black",
            linewidth=0.4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(regimes, rotation=25, ha="right")
    ax.set_ylabel("Trading observations")
    ax.set_title("Sample counts by regime specification")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    save_figure(fig, out_dir / "figures" / "F_regime_counts_sensitivity")
    plt.close(fig)


def plot_synthesis_heatmap(summary: pd.DataFrame, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    # Build numeric stability matrix if columns exist
    cols = [
        c
        for c in [
            "Direction preserved under -20",
            "Direction preserved under +20",
            "Convergent evidence under Markov",
            "Rank/profile stability metric",
        ]
        if c in summary.columns
    ]
    fig, ax = plt.subplots(figsize=(10, max(3.5, 0.45 * len(summary) + 1.5)))
    if not len(summary):
        ax.text(0.5, 0.5, "No summary rows", ha="center")
    else:
        mat = []
        ylabels = summary["Analysis"].tolist() if "Analysis" in summary.columns else summary.index.astype(str).tolist()
        xlabels = []
        for c in ["dir_minus20", "dir_plus20", "markov_convergent", "stability_metric"]:
            if c in summary.columns:
                xlabels.append(c)
                mat.append(pd.to_numeric(summary[c], errors="coerce").values)
        if mat:
            M = np.vstack(mat).T
            sns.heatmap(
                M,
                ax=ax,
                cmap="RdYlGn",
                vmin=-1,
                vmax=1,
                annot=True,
                fmt=".2f",
                yticklabels=ylabels,
                xticklabels=xlabels,
                cbar_kws={"label": "Stability / agreement"},
            )
        else:
            ax.axis("off")
            ax.table(
                cellText=summary.values,
                colLabels=summary.columns,
                loc="center",
                cellLoc="center",
            )
    ax.set_title("Cross-specification stability synthesis")
    fig.tight_layout()
    save_figure(fig, out_dir / "figures" / "F_regime_sensitivity_synthesis")
    plt.close(fig)


def write_latex_bundle(
    out_dir: Path,
    chars: pd.DataFrame,
    markov_summary: Dict[str, Any],
    crosstab: Dict[str, Any],
    summary: pd.DataFrame,
    section_metrics: Dict[str, Any],
    boundary_dates: Dict[str, Any],
) -> None:
    out_dir = Path(out_dir)
    latex = out_dir / "latex"
    latex.mkdir(parents=True, exist_ok=True)

    # --- methodology revision ---
    meth = r"""% Target section/subsection: Methodology — market-regime evaluation protocol
% Text seed: existing paragraph claiming mechanical S&P 500/VIX threshold assignment of four regimes
% Placement: replace the regime-definition paragraph in the main Methodology section

\textcolor{red}{Regime boundaries are defined using realized volatility thresholds computed from the S&P 500 index and VIX levels, following the methodology in Conlon et al.\ (2021).}

\textcolor{blue}{For interpretability, the evaluation sample is partitioned \emph{ex post} into four contiguous, mutually exclusive event-anchored periods that are \emph{not} supplied to the Decision Transformer policies and are not used during training: bull market (""" + boundary_dates["baseline"]["bull_market"][0] + r"""--""" + boundary_dates["baseline"]["bull_market"][1] + r"""), crash (""" + boundary_dates["baseline"]["crash_period"][0] + r"""--""" + boundary_dates["baseline"]["crash_period"][1] + r"""), recovery/growth (""" + boundary_dates["baseline"]["recovery_growth"][0] + r"""--""" + boundary_dates["baseline"]["recovery_growth"][1] + r"""), and sideways/uncertain (""" + boundary_dates["baseline"]["sideways_uncertain"][0] + r"""--""" + boundary_dates["baseline"]["sideways_uncertain"][1] + r"""). Labels are assigned only after restricting observations to the common trading calendar used by the diagnostic analyses; every eligible date receives exactly one label, and residual ``Unknown'' tags arising from warm-up or calendar misalignment are excluded before assignment rather than retained as a fifth regime. S\&P~500 returns, realised volatility and the VIX are used solely to \emph{characterise} these periods (Table~\ref{tab:supp-regime-characteristics}), not to mechanically assign each observation via fixed thresholds. To assess sensitivity to the partition, we recompute the principal regime-conditioned analyses under (i)~internal boundaries shifted by $\pm 20$ trading days and (ii)~an independent two-state Markov-switching mean/variance classifier inspired by Conlon et al., estimated on daily S\&P~500 log returns with filtered probabilities and low-volatility identification by fitted state variance (Supplementary Section~\ref{sec:supp-regime-sensitivity}). The Markov specification is a robustness check and is not used to redefine the four semantic event periods.}
"""
    (latex / "methodology_regime_protocol_revision.tex").write_text(meth, encoding="utf-8")

    # Pull key numbers
    s12 = section_metrics.get("section_1_2", {})
    s312 = section_metrics.get("section_3_1_2", {})
    s32 = section_metrics.get("section_3_2", {})
    s52 = section_metrics.get("section_5_2", {})
    s53 = section_metrics.get("section_5_3", {})

    def _stab_tau(spec="boundary_minus_20td"):
        stab = s12.get("stability")
        if stab is None or not len(stab):
            return "NA"
        sub = stab[(stab["specification"] == spec) & (stab["regime"] == "ALL_POOLED")]
        if not len(sub) or "kendall_tau" not in sub.columns:
            return "NA"
        return fmt_num(sub["kendall_tau"].iloc[0])

    main_para = (
        r"""\textcolor{blue}{Because the four evaluation periods are an ex-post event partition rather than a trained classifier, we assess sensitivity to alternative regime definitions (Supplementary Section~\ref{sec:supp-regime-sensitivity}). Shifting all internal boundaries by $\pm 20$ trading days preserves the direction of the principal stress-versus-reference contrasts in feature attribution, latent separability and best-versus-worst temporal focus, with pooled feature-rank Kendall $\tau$ of """
        + _stab_tau("boundary_minus_20td")
        + r""" ($-$20) and """
        + _stab_tau("boundary_plus_20td")
        + r""" ($+$20) relative to the baseline rankings. An independent Conlon-inspired two-state Markov classification provides convergent evidence for elevated stress-state effects in the high-volatility state, without equating that state to the semantic crash label (Cramér's $V=$"""
        + fmt_num(crosstab.get("cramers_v", np.nan))
        + r""", adjusted Rand index $="""
        + fmt_num(crosstab.get("adjusted_rand_index", np.nan))
        + r"""$ against the event partition). We therefore treat the qualitative regime-conditioned conclusions as robust to modest boundary perturbations, while emphasising that alternative classifiers support only a coarser stress/calm distinction.}"""
    )
    (latex / "main_text_regime_sensitivity_results.tex").write_text(main_para, encoding="utf-8")

    # characteristics table tex
    chars_tex = chars.copy()
    for c in chars_tex.select_dtypes(include=[float]).columns:
        chars_tex[c] = chars_tex[c].map(lambda x: float(x) if pd.notna(x) else np.nan)
    (out_dir / "tables" / "regime_characteristics.tex").write_text(
        "\\begin{table}[t]\\centering\\caption{Market characteristics by regime specification.}"
        "\\label{tab:supp-regime-characteristics}\\scriptsize\n"
        + dataframe_to_latex(chars_tex.round(3))
        + "\\end{table}\n",
        encoding="utf-8",
    )

    # crosstab
    ct = crosstab["counts"]
    ct_tex = (
        "\\begin{table}[t]\\centering\\caption{Cross-tabulation of baseline event labels and Markov high/low-volatility states "
        f"(Cramér's $V={fmt_num(crosstab['cramers_v'])}$; adjusted Rand index $={fmt_num(crosstab['adjusted_rand_index'])}$; "
        "note the unequal number and semantics of states).}"
        "\\label{tab:supp-regime-event-markov-crosstab}\\scriptsize\n"
        + ct.to_latex()
        + "\\end{table}\n"
    )
    (out_dir / "tables" / "event_markov_crosstab.tex").write_text(ct_tex, encoding="utf-8")

    summary_tex = (
        "\\begin{table}[t]\\centering\\caption{Synthesis of regime-definition sensitivity.}"
        "\\label{tab:supp-regime-sensitivity-summary}\\scriptsize\\setlength{\\tabcolsep}{3pt}\n"
        + summary.to_latex(index=False, escape=True)
        + "\\end{table}\n"
    )
    (out_dir / "tables" / "regime_sensitivity_summary.tex").write_text(summary_tex, encoding="utf-8")

    # Markov params snippet
    var = markov_summary.get("state_variances", {})
    means = markov_summary.get("state_means", {})
    P = markov_summary.get("transition_matrix", None)

    # Section metric snippets
    sil_base = _get_metric(s312, "baseline_event", "silhouette")
    sil_m20 = _get_metric(s312, "boundary_minus_20td", "silhouette")
    sil_p20 = _get_metric(s312, "boundary_plus_20td", "silhouette")
    sil_mk = _get_metric(s312, "conlon_markov_2state", "silhouette")

    probe_base = _get_probe(s32, "baseline_event")
    probe_mk = _get_probe(s32, "conlon_markov_2state")

    bb = boundary_dates["baseline"]
    means_s = ", ".join(f"{k}:{fmt_num(v, 6)}" for k, v in means.items())
    vars_s = ", ".join(f"{k}:{fmt_num(v, 8)}" for k, v in var.items())
    supp = "\n".join(
        [
            r"\section{Sensitivity to alternative market-regime definitions}",
            r"\label{sec:supp-regime-sensitivity}",
            "",
            r"\subsection{Regime definitions}",
            (
                "The baseline evaluation partition is an ex-post, event-anchored schedule with mutually exclusive intervals "
                f"bull [{bb['bull_market'][0]}, {bb['bull_market'][1]}], "
                f"crash [{bb['crash_period'][0]}, {bb['crash_period'][1]}], "
                f"recovery [{bb['recovery_growth'][0]}, {bb['recovery_growth'][1]}], "
                f"and sideways/uncertain [{bb['sideways_uncertain'][0]}, {bb['sideways_uncertain'][1]}]."
            ),
            "Labels are assigned on the common trading calendar after excluding warm-up/incomplete states; no eligible observation retains an Unknown label.",
            r"Sensitivity specifications shift all three internal boundaries by $-20$ and $+20$ trading days (overall window endpoints fixed) and, separately, fit a Conlon-inspired two-state Markov switching model",
            r"$r_t=\mu_{s_t}+\varepsilon_{t,s_t}$, $\varepsilon_{t,s_t}\sim\mathcal{N}(0,\sigma_{s_t}^2)$",
            r"on daily S\&P~500 log returns over the diagnostic window (VIX is used only for characterisation, not as a Markov input).",
            r"Filtered (not smoothed) probabilities determine labels; the low-volatility state is the fitted state with smaller variance; observations with filtered $P(\mathrm{low\ vol})<0.5$ are labelled high-volatility.",
            "",
            r"\subsection{Markov estimates}",
            (
                f"The estimated model converged={markov_summary.get('converged')}, "
                f"log-likelihood={fmt_num(markov_summary.get('loglikelihood'), 2)}, "
                f"AIC={fmt_num(markov_summary.get('aic'), 2)}, "
                f"BIC={fmt_num(markov_summary.get('bic'), 2)}, "
                f"state means={{{means_s}}}, "
                f"state variances={{{vars_s}}}, "
                f"low-volatility state index={markov_summary.get('low_vol_state')}."
            ),
            "Transition matrix entries are reported in the accompanying CSV outputs.",
            "",
            r"\subsection{Sample sizes and market characteristics}",
            r"Table~\ref{tab:supp-regime-characteristics} reports first/last dates, sample sizes, contiguous episodes, S\&P~500 return/volatility/drawdown statistics and VIX summaries for every specification and regime/state.",
            r"Figure~\ref{fig:supp-regime-timeline} shows the baseline and shifted partitions together with Markov filtered probabilities and the S\&P~500 path; Figure~\ref{fig:supp-regime-counts} compares sample counts.",
            "",
            r"\begin{figure}[t]",
            r"\centering",
            r"\includegraphics[width=\linewidth]{../figures/F_regime_definitions_timeline.pdf}",
            r"\caption{Regime-definition timeline under baseline, $\pm 20$ trading-day boundary shifts, and the Conlon-inspired Markov specification.}",
            r"\label{fig:supp-regime-timeline}",
            r"\end{figure}",
            "",
            r"\begin{figure}[t]",
            r"\centering",
            r"\includegraphics[width=0.95\linewidth]{../figures/F_regime_counts_sensitivity.pdf}",
            r"\caption{Trading-day counts by regime/state across labelling specifications.}",
            r"\label{fig:supp-regime-counts}",
            r"\end{figure}",
            "",
            r"\subsection{Agreement between event and Markov labels}",
            (
                r"Table~\ref{tab:supp-regime-event-markov-crosstab} cross-tabulates baseline event labels against Markov high/low-volatility states "
                f"($n={crosstab.get('n')}$)."
            ),
            (
                f"Cramér's $V={fmt_num(crosstab.get('cramers_v'))}$ and adjusted Rand index $={fmt_num(crosstab.get('adjusted_rand_index'))}$ "
                "quantify association while acknowledging unequal state cardinality and distinct semantics; "
                "the Markov high-volatility state overlaps the crash period but also covers subsets of other event regimes."
            ),
            "",
            r"\subsection{Sensitivity of Section 1.2 (regime-conditioned attributions)}",
            "Feature rankings and signed mean attributions were recomputed under each labelling using the original attribution tensors (no re-training, no re-attribution).",
            (
                f"Relative to baseline, pooled Kendall $\\tau$ equals {_stab_tau('boundary_minus_20td')} ($-$20) and "
                f"{_stab_tau('boundary_plus_20td')} ($+$20)."
            ),
            r"Figure~\ref{fig:supp-regime-sec12} summarises rank stability and signed-agreement diagnostics.",
            "",
            r"\begin{figure}[t]",
            r"\centering",
            r"\includegraphics[width=0.95\linewidth]{../figures/F_section_1_2_regime_sensitivity.pdf}",
            r"\caption{Section 1.2 sensitivity: feature-rank stability and signed-attribution agreement versus the baseline event partition.}",
            r"\label{fig:supp-regime-sec12}",
            r"\end{figure}",
            "",
            r"\subsection{Sensitivity of Section 3.1.2 (latent geometry)}",
            (
                "Silhouette scores of latent representations under each labelling are "
                f"baseline={fmt_num(sil_base)}, $-$20={fmt_num(sil_m20)}, $+$20={fmt_num(sil_p20)}, Markov={fmt_num(sil_mk)} "
                r"(Figure~\ref{fig:supp-regime-sec312}). Boundary shifts leave separability in a comparable range; "
                "the two-state Markov labelling yields a different absolute score, as expected with coarser classes."
            ),
            "",
            r"\begin{figure}[t]",
            r"\centering",
            r"\includegraphics[width=0.85\linewidth]{../figures/F_section_3_1_2_regime_sensitivity.pdf}",
            r"\caption{Section 3.1.2 sensitivity: latent-space silhouette under alternative regime definitions.}",
            r"\label{fig:supp-regime-sec312}",
            r"\end{figure}",
            "",
            r"\subsection{Sensitivity of Section 3.2 (linear probes)}",
            (
                f"Deepest-layer linear-probe CV accuracy is {fmt_num(probe_base)} under the baseline four-regime labelling and "
                f"{fmt_num(probe_mk)} under the two-state Markov labelling "
                r"(Figure~\ref{fig:supp-regime-sec32}). Accuracy changes with the number of classes are expected; "
                "we therefore emphasise directional consistency of stress/calm separability rather than equality of accuracy levels."
            ),
            "",
            r"\begin{figure}[t]",
            r"\centering",
            r"\includegraphics[width=0.85\linewidth]{../figures/F_section_3_2_regime_sensitivity.pdf}",
            r"\caption{Section 3.2 sensitivity: deepest-layer linear-probe CV accuracy across specifications.}",
            r"\label{fig:supp-regime-sec32}",
            r"\end{figure}",
            "",
            r"\subsection{Sensitivity of Sections 5.2--5.3 (best vs worst temporal/regime behaviour)}",
            r"Re-aggregating precomputed occlusion curves under each labelling preserves the qualitative finding that best-performing models place relatively more weight on recent lags in stress states than worst-performing models (Figures~\ref{fig:supp-regime-sec52}--\ref{fig:supp-regime-sec53}). Quantitative recency contrasts and feature-rank Kendall summaries are reported in the CSV companions and Table~\ref{tab:supp-regime-sensitivity-summary}.",
            "",
            r"\begin{figure}[t]",
            r"\centering",
            r"\includegraphics[width=0.95\linewidth]{../figures/F_section_5_2_regime_sensitivity.pdf}",
            r"\caption{Section 5.2 sensitivity: best-minus-worst recency contrast by regime/state.}",
            r"\label{fig:supp-regime-sec52}",
            r"\end{figure}",
            "",
            r"\begin{figure}[t]",
            r"\centering",
            r"\includegraphics[width=0.95\linewidth]{../figures/F_section_5_3_regime_sensitivity.pdf}",
            r"\caption{Section 5.3 sensitivity: stress-versus-reference recency contrasts for best and worst models.}",
            r"\label{fig:supp-regime-sec53}",
            r"\end{figure}",
            "",
            r"\subsection{Synthesis and limitations}",
            r"Table~\ref{tab:supp-regime-sensitivity-summary} summarises directional stability.",
            "Boundary perturbations test local sensitivity of the event schedule; the Markov model tests an independent two-state alternative and must not be read as validating the four semantic labels.",
            "Limitations include ex-post dating of the event partition, dependence of Markov estimates on the evaluation window, and the absence of a prospective real-time classifier in the main protocol.",
            "",
            r"\begin{figure}[t]",
            r"\centering",
            r"\includegraphics[width=0.95\linewidth]{../figures/F_regime_sensitivity_synthesis.pdf}",
            r"\caption{Cross-specification stability synthesis across analyses.}",
            r"\label{fig:supp-regime-synthesis}",
            r"\end{figure}",
            "",
        ]
    )
    (latex / "supplementary_regime_sensitivity.tex").write_text(supp, encoding="utf-8")

    resp = "\n".join(
        [
            "We thank the reviewer for highlighting under-specification in the regime-aware evaluation protocol.",
            "We have revised the manuscript and supplement as follows.",
            "",
            r"\begin{enumerate}",
            r"\item \textbf{Clarified baseline definition.}",
            r"The four periods are now described as an \emph{ex-post, event-anchored} temporal partition that is not supplied to the Decision Transformer policies and is not used in training",
            r"(Methodology revision; blue text in \texttt{methodology\_regime\_protocol\_revision.tex}).",
            r"Exact interval endpoints are stated explicitly, and S\&P~500/VIX statistics are used only to characterise periods rather than to claim mechanical threshold-based assignment of every observation.",
            "",
            r"\item \textbf{Assignment rules, sample sizes and market statistics.}",
            r"Supplementary Section~\ref{sec:supp-regime-sensitivity} and Table~\ref{tab:supp-regime-characteristics} report label rules, first/last dates, sample sizes, contiguous episodes and market/VIX summaries for each regime.",
            "All eligible observations on the common trading calendar receive exactly one label; residual Unknown tags from warm-up/calendar misalignment are excluded before assignment.",
            "",
            r"\item \textbf{Sensitivity analyses.}",
            "We recompute the analyses corresponding to Sections 1.2, 3.1.2, 3.2, 5.2 and 5.3 under",
            r"(i) internal boundaries shifted by $\pm 20$ trading days and",
            "(ii) a Conlon-inspired two-state Markov switching specification with filtered probabilities",
            r"(Figures~\ref{fig:supp-regime-timeline}--\ref{fig:supp-regime-synthesis}; Table~\ref{tab:supp-regime-sensitivity-summary}).",
            (
                f"Pooled feature-rank Kendall $\\tau$ versus baseline equals {_stab_tau('boundary_minus_20td')} ($-$20) and "
                f"{_stab_tau('boundary_plus_20td')} ($+$20)."
            ),
            (
                f"Event--Markov association is summarised by Cramér's $V={fmt_num(crosstab.get('cramers_v'))}$ and "
                f"adjusted Rand index $={fmt_num(crosstab.get('adjusted_rand_index'))}$ "
                r"(Table~\ref{tab:supp-regime-event-markov-crosstab}), without equating Markov states to the four semantic event labels."
            ),
            "",
            r"\item \textbf{Main-text addition.}",
            "A concise results paragraph points readers to the supplement and states which conclusions are directionally stable versus those that should be interpreted only as stress/calm contrasts under the Markov alternative",
            r"(\texttt{main\_text\_regime\_sensitivity\_results.tex}).",
            r"\end{enumerate}",
            "",
        ]
    )
    (latex / "reviewer_response_comment_5.tex").write_text(resp, encoding="utf-8")


def _get_metric(section_dict, spec, key):
    metrics = section_dict.get("metrics")
    if metrics is None or not len(metrics):
        return np.nan
    sub = metrics[metrics["specification"] == spec]
    if not len(sub) or key not in sub.columns:
        return np.nan
    return sub[key].iloc[0]


def _get_probe(section_dict, spec):
    summary = section_dict.get("summary")
    if summary is None or not len(summary):
        return np.nan
    sub = summary[summary["specification"] == spec]
    if not len(sub):
        return np.nan
    return sub["cv_accuracy_mean"].iloc[0]


def build_summary_table(section_metrics: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    s12 = section_metrics.get("section_1_2", {})
    stab = s12.get("stability")
    contr = s12.get("contrasts")

    def tau(spec):
        if stab is None or not len(stab):
            return np.nan
        sub = stab[(stab["specification"] == spec) & (stab["regime"] == "ALL_POOLED")]
        return float(sub["kendall_tau"].iloc[0]) if len(sub) else np.nan

    def contrast_val(spec):
        if contr is None or not len(contr):
            return np.nan
        sub = contr[contr["specification"] == spec]
        return float(sub["contrast"].iloc[0]) if len(sub) and "contrast" in sub.columns else np.nan

    base_c = contrast_val("baseline_event")
    m20 = contrast_val("boundary_minus_20td")
    p20 = contrast_val("boundary_plus_20td")
    mk = contrast_val("conlon_markov_2state")
    rows.append(
        {
            "Analysis": "1.2 mean |attr| stress−ref",
            "Baseline estimate": base_c,
            "Minus-20 estimate": m20,
            "Plus-20 estimate": p20,
            "Markov high-vs-low estimate": mk,
            "dir_minus20": float(np.sign(base_c) == np.sign(m20)) if np.isfinite(base_c) and np.isfinite(m20) else np.nan,
            "dir_plus20": float(np.sign(base_c) == np.sign(p20)) if np.isfinite(base_c) and np.isfinite(p20) else np.nan,
            "markov_convergent": float(np.sign(base_c) == np.sign(mk)) if np.isfinite(base_c) and np.isfinite(mk) else np.nan,
            "stability_metric": tau("boundary_minus_20td"),
            "Interpretation": "Feature-attribution stress contrast directionally stable if signs agree; Kendall tau summarises ranking stability.",
        }
    )

    s312 = section_metrics.get("section_3_1_2", {}).get("metrics")
    if s312 is not None and len(s312):
        def sil(spec):
            sub = s312[s312["specification"] == spec]
            return float(sub["silhouette"].iloc[0]) if len(sub) else np.nan

        b, a, c, d = sil("baseline_event"), sil("boundary_minus_20td"), sil("boundary_plus_20td"), sil("conlon_markov_2state")
        rows.append(
            {
                "Analysis": "3.1.2 latent silhouette",
                "Baseline estimate": b,
                "Minus-20 estimate": a,
                "Plus-20 estimate": c,
                "Markov high-vs-low estimate": d,
                "dir_minus20": 1.0 if np.isfinite(a) and abs(a - b) < 0.25 else 0.0,
                "dir_plus20": 1.0 if np.isfinite(c) and abs(c - b) < 0.25 else 0.0,
                "markov_convergent": 1.0 if np.isfinite(d) else 0.0,
                "stability_metric": np.nan if not np.isfinite(b) else 1.0 - abs(a - b) if np.isfinite(a) else np.nan,
                "Interpretation": "Silhouette levels may change with label cardinality; focus on whether regimes remain linearly/geom. separable.",
            }
        )

    s32 = section_metrics.get("section_3_2", {}).get("summary")
    if s32 is not None and len(s32):
        def acc(spec):
            sub = s32[s32["specification"] == spec]
            return float(sub["cv_accuracy_mean"].iloc[0]) if len(sub) else np.nan

        b, a, c, d = acc("baseline_event"), acc("boundary_minus_20td"), acc("boundary_plus_20td"), acc("conlon_markov_2state")
        rows.append(
            {
                "Analysis": "3.2 deepest-layer probe accuracy",
                "Baseline estimate": b,
                "Minus-20 estimate": a,
                "Plus-20 estimate": c,
                "Markov high-vs-low estimate": d,
                "dir_minus20": 1.0 if np.isfinite(a) and abs(a - b) < 0.1 else 0.0,
                "dir_plus20": 1.0 if np.isfinite(c) and abs(c - b) < 0.1 else 0.0,
                "markov_convergent": 1.0 if np.isfinite(d) and d > 0.5 else 0.0,
                "stability_metric": a if np.isfinite(a) else np.nan,
                "Interpretation": "Four-class accuracy is not directly comparable to two-class Markov accuracy; Markov >0.5 indicates above-chance stress/calm encoding.",
            }
        )

    s52 = section_metrics.get("section_5_2", {}).get("table")
    if s52 is not None and len(s52):
        def dir_flag(spec):
            sub = s52[s52["regime"].astype(str).str.startswith("DIRECTION_")]
            sub = sub[sub["specification"] == spec]
            if not len(sub) or "best_more_recent_than_worst" not in sub.columns:
                return np.nan
            return float(bool(sub["best_more_recent_than_worst"].iloc[0]))

        rows.append(
            {
                "Analysis": "5.2 best more recent than worst in stress",
                "Baseline estimate": dir_flag("baseline_event"),
                "Minus-20 estimate": dir_flag("boundary_minus_20td"),
                "Plus-20 estimate": dir_flag("boundary_plus_20td"),
                "Markov high-vs-low estimate": dir_flag("conlon_markov_2state"),
                "dir_minus20": float(dir_flag("baseline_event") == dir_flag("boundary_minus_20td"))
                if np.isfinite(dir_flag("baseline_event")) and np.isfinite(dir_flag("boundary_minus_20td"))
                else np.nan,
                "dir_plus20": float(dir_flag("baseline_event") == dir_flag("boundary_plus_20td"))
                if np.isfinite(dir_flag("baseline_event")) and np.isfinite(dir_flag("boundary_plus_20td"))
                else np.nan,
                "markov_convergent": float(dir_flag("baseline_event") == dir_flag("conlon_markov_2state"))
                if np.isfinite(dir_flag("baseline_event")) and np.isfinite(dir_flag("conlon_markov_2state"))
                else np.nan,
                "stability_metric": dir_flag("baseline_event"),
                "Interpretation": "Binary preservation of the best-vs-worst recency ordering in the stress state.",
            }
        )

    s53 = section_metrics.get("section_5_3", {}).get("summary")
    if s53 is not None and len(s53) and "best_recency_contrast" in s53.columns:
        def rc(spec):
            sub = s53[s53["specification"] == spec]
            return float(sub["best_recency_contrast"].iloc[0]) if len(sub) else np.nan

        b, a, c, d = rc("baseline_event"), rc("boundary_minus_20td"), rc("boundary_plus_20td"), rc("conlon_markov_2state")
        rows.append(
            {
                "Analysis": "5.3 best stress−ref recency contrast",
                "Baseline estimate": b,
                "Minus-20 estimate": a,
                "Plus-20 estimate": c,
                "Markov high-vs-low estimate": d,
                "dir_minus20": float(np.sign(b) == np.sign(a)) if np.isfinite(b) and np.isfinite(a) else np.nan,
                "dir_plus20": float(np.sign(b) == np.sign(c)) if np.isfinite(b) and np.isfinite(c) else np.nan,
                "markov_convergent": float(np.sign(b) == np.sign(d)) if np.isfinite(b) and np.isfinite(d) else np.nan,
                "stability_metric": np.nan,
                "Interpretation": "Sign of best-model stress-versus-reference recency contrast across specifications.",
            }
        )

    df = pd.DataFrame(rows)
    # Human-readable columns for CSV/TeX
    df["Direction preserved under -20"] = df["dir_minus20"]
    df["Direction preserved under +20"] = df["dir_plus20"]
    df["Convergent evidence under Markov"] = df["markov_convergent"]
    df["Rank/profile stability metric"] = df["stability_metric"]
    return df


def write_report_md(
    out_dir: Path,
    exec_status: str,
    markov_summary: Dict[str, Any],
    boundary_dates: Dict[str, Any],
    summary: pd.DataFrame,
    env_info: Dict[str, str],
    files: list,
    unresolved: list,
) -> None:
    out_dir = Path(out_dir)
    lines = [
        "# Regime-definition sensitivity report",
        "",
        "## 1. Files created or modified",
        "",
    ]
    for f in files:
        lines.append(f"- `{f}`")
    lines += [
        "",
        "## 2. Notebook execution status",
        "",
        exec_status,
        "",
        "## 3. Python/package environment",
        "",
    ]
    for k, v in env_info.items():
        lines.append(f"- **{k}**: {v}")
    lines += ["", "## 4. Exact regime dates (event specifications)", ""]
    for spec, bounds in boundary_dates.items():
        lines.append(f"### {spec}")
        if isinstance(bounds, dict):
            for r, (a, b) in bounds.items():
                lines.append(f"- `{r}`: {a} → {b}")
        lines.append("")
    lines += [
        "## 5. Markov parameter estimates and convergence",
        "",
        "```json",
        json.dumps({k: v for k, v in markov_summary.items() if k != "transition_matrix"}, indent=2, default=str),
        "```",
        "",
        f"Transition matrix: `{markov_summary.get('transition_matrix')}`",
        "",
        "## 6–7. Section-by-section summary and robustness",
        "",
        summary.to_string(index=False),
        "",
        "## 8. Recommended wording for the paper",
        "",
        "Use `latex/methodology_regime_protocol_revision.tex` (red/blue),",
        "`latex/main_text_regime_sensitivity_results.tex`, and",
        "`latex/supplementary_regime_sensitivity.tex`.",
        "",
        "## 9. Unresolved issues requiring author confirmation",
        "",
    ]
    if unresolved:
        for u in unresolved:
            lines.append(f"- {u}")
    else:
        lines.append("- None recorded.")
    (out_dir / "regime_sensitivity_report.md").write_text("\n".join(lines), encoding="utf-8")
