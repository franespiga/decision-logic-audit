"""
Tables, LaTeX drop-ins, figures and supplementary output.

Every number written here comes from the computed result objects; no value is
hard-coded and no placeholder is emitted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .sampling import REGIME_ORDER
from .scoring import METRIC_SPEC, SCORE_KIND

FAMILY_LABEL = {"DT": "Decision Transformer", "RL": "Conventional deep RL"}

MAIN_TABLE_COLUMNS = [
    ("family_label", "Family"),
    ("policy_label", "Policy"),
    ("gross_sharpe_ratio", "Sharpe ratio"),
    ("contribution_score", "Contribution"),
    ("necessity_score", "Necessity"),
    ("structure_score", "Structure"),
    ("aggregate_diagnostic_score", "Aggregate"),
]

DETAIL_COLUMNS = [
    "model_id",
    "family",
    "algorithm",
    "action_space",
    "lookback",
    "observation_window_rows",
    "n_observation_features",
    "rebalancing_frequency",
    "attribution_method",
    "attribution_variant",
    "layer_influence_method",
    "contribution_rank_stability",
    "contribution_baseline_concordance",
    "deletion_faithfulness",
    "temporal_profile_stability",
    "latent_neighbourhood_stability",
    "non_concentrated_layer_influence",
    "layer_influence_normalised_hhi",
    "layer_influence_max_share",
    "layer_influence_effective_layers",
    "attribution_effective_features",
    "attribution_entropy_feature",
    "temporal_peak_lag",
    "temporal_half_life",
    "temporal_recency_index",
    "temporal_entropy",
    "temporal_normalised_auc",
    "deletion_excess_weight_tv",
    "deletion_guided_weight_tv_at_30pct",
    "deletion_guided_flip_rate_at_30pct",
    "contribution_score",
    "necessity_score",
    "structure_score",
    "aggregate_diagnostic_score",
    "gross_cumulative_return",
    "gross_annualised_return",
    "gross_annualised_volatility",
    "gross_sharpe_ratio",
    "gross_sortino_ratio",
    "gross_max_drawdown",
    "net_annualised_return",
    "net_sharpe_ratio",
    "annualised_turnover",
    "mean_cash_allocation",
    "unique_allocations",
    "n_diagnostic_samples",
    "n_evaluation_dates",
    "weights_source",
    "performance_source",
]


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value.replace("_", r"\_").replace("^", r"\^{}")
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(v):
        return "n/a"
    return f"{v:.{digits}f}"


def _tt(text: str) -> str:
    """Typeset a file or setting name in a monospaced font, LaTeX-safe."""
    escaped = str(text).replace("\\", r"\textbackslash{}").replace("_", r"\_")
    return rf"\texttt{{{escaped}}}"


def markdown_table(frame: pd.DataFrame, digits: int = 4, max_rows: Optional[int] = None) -> str:
    """Render a frame as a GitHub-flavoured markdown table without extra dependencies."""
    if frame is None or len(frame) == 0:
        return "_none_"
    df = frame if max_rows is None else frame.head(max_rows)

    def cell(value) -> str:
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return "n/a"
        if isinstance(value, (int, np.integer, bool, np.bool_)):
            return str(value)
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{digits}g}"
        return str(value).replace("|", r"\|").replace("\n", " ")

    header = [str(c).replace("|", r"\|") for c in df.columns]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(cell(v) for v in row.tolist()) + " |")
    if max_rows is not None and len(frame) > max_rows:
        lines.append(f"\n_Showing the first {max_rows} of {len(frame)} rows._")
    return "\n".join(lines)


def write_table(frame: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def build_model_summary(table: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in DETAIL_COLUMNS if c in table.columns]
    extra = [c for c in table.columns if c not in cols]
    return table.loc[:, cols + extra]


# ---------------------------------------------------------------------------
# Main manuscript table
# ---------------------------------------------------------------------------


def main_latex_table(table: pd.DataFrame, sharpe_kind: str = "gross") -> str:
    df = table.copy()
    df["family_label"] = df["family"].map(FAMILY_LABEL).fillna(df["family"])
    df["policy_label"] = df.apply(
        lambda r: f"{r['algorithm']} ({r['model_id']})" if r["family"] == "RL" else str(r["model_id"]),
        axis=1,
    )
    df = df.sort_values(["family", "model_id"])

    lines: List[str] = []
    lines.append(r"% Generated by scripts/run_cross_architecture_comparison.py -- do not edit by hand.")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Cross-architecture diagnostic comparison. The same contribution, necessity "
        r"and structure diagnostics are applied to Decision Transformer policies and to "
        r"conventional deep reinforcement-learning policies over a common set of "
        r"regime-stratified evaluation dates.}"
    )
    lines.append(r"\label{tab:cross-architecture-summary}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{llrrrrr}")
    lines.append(r"\toprule")
    lines.append(" & ".join(h for _, h in MAIN_TABLE_COLUMNS) + r" \\")
    lines.append(r"\midrule")
    for _, row in df.iterrows():
        cells = []
        for key, _ in MAIN_TABLE_COLUMNS:
            value = row.get(key)
            cells.append(_fmt(value, 2 if key == "gross_sharpe_ratio" else 3))
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    note = (
        r"\footnotesize Sharpe ratio is \textbf{%s} of transaction costs, computed from "
        r"allocations reconstructed from the frozen policies on the trading calendar of the "
        r"evaluation window. Contribution, necessity and structure are equally weighted means "
        r"of two oriented diagnostics each; the aggregate is their unweighted mean. All "
        r"diagnostic scores are %s, oriented so that \emph{larger indicates less pathological "
        r"behaviour}. Entries reported as n/a are diagnostics that are undefined for that "
        r"policy and the reason is stated in the supplementary material; they are not missing "
        r"computations."
    ) % ("gross" if sharpe_kind == "gross" else "net", SCORE_KIND)
    lines.append(r"\par\vspace{2pt}")
    lines.append(r"\begin{minipage}{\textwidth}" + note + r"\end{minipage}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def figure_diagnostics_heatmap(table: pd.DataFrame, out_base: Path) -> List[Path]:
    """Normalised heatmap of the six oriented diagnostics plus the three dimension scores."""
    plt = _setup_matplotlib()
    metrics = list(METRIC_SPEC)
    scores = ["contribution_score", "necessity_score", "structure_score", "aggregate_diagnostic_score"]
    cols = [f"oriented_{m}" for m in metrics] + scores
    labels = [
        "Contribution:\nrank stability",
        "Contribution:\nbaseline concordance",
        "Necessity:\ndeletion faithfulness",
        "Necessity:\ntemporal stability",
        "Structure:\nlatent stability",
        "Structure:\nlayer dispersion",
        "Contribution\nscore",
        "Necessity\nscore",
        "Structure\nscore",
        "Aggregate\nscore",
    ]
    df = table.sort_values(["family", "model_id"])
    data = df.loc[:, cols].apply(pd.to_numeric, errors="coerce").values.astype(float)
    rows = [f"[{r.family}] {r.model_id}" for r in df.itertuples()]

    fig, ax = plt.subplots(figsize=(9.0, 0.62 * len(rows) + 2.6))
    masked = np.ma.masked_invalid(data)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.85")
    im = ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    ax.axvline(len(metrics) - 0.5, color="white", linewidth=2.5)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isfinite(v):
                ax.text(
                    j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if v < 0.55 else "black",
                )
            else:
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=7, color="0.3")
    ax.set_title(
        "Cross-architecture diagnostics (oriented, larger indicates less pathological behaviour)\n"
        "grey cells are diagnostics undefined for that policy"
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Oriented score")
    fig.tight_layout()
    paths = []
    for ext in ("png", "pdf"):
        p = Path(f"{out_base}.{ext}")
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def figure_functional_comparison(
    necessity_results: Dict[str, Dict],
    table: pd.DataFrame,
    out_base: Path,
) -> List[Path]:
    """Normalised temporal-occlusion profiles and attribution-guided deletion curves."""
    plt = _setup_matplotlib()
    order = table.sort_values(["family", "model_id"])["model_id"].tolist()
    families = dict(zip(table["model_id"], table["family"]))
    cmap = plt.get_cmap("tab10")
    colours = {m: cmap(i % 10) for i, m in enumerate(order)}
    styles = {"DT": "-", "RL": "--"}

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))

    ax = axes[0]
    for model_id in order:
        res = necessity_results.get(model_id)
        if not res:
            continue
        grid = res["relative_lag_grid"]
        prof = res["relative_lag_profile"]
        if not np.isfinite(prof).any():
            continue
        ax.plot(
            grid, prof, styles.get(families.get(model_id), "-"),
            color=colours[model_id], label=f"{model_id}", linewidth=1.6,
        )
    ax.set_xlabel("Relative lag (0 = most recent, 1 = oldest in window)")
    ax.set_ylabel("Normalised occlusion influence")
    ax.set_title("(a) Temporal-occlusion profiles")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1]
    for model_id in order:
        res = necessity_results.get(model_id)
        if not res:
            continue
        curve = res["deletion"]["curve"]
        x = curve["fraction"].values * 100.0
        ax.plot(
            x, curve["weight_tv"].values, styles.get(families.get(model_id), "-"),
            color=colours[model_id], marker="o", markersize=3.5, linewidth=1.5,
            label=f"{model_id} (guided)",
        )
        ax.plot(
            x, curve["random_weight_tv_mean"].values, ":",
            color=colours[model_id], linewidth=1.1, alpha=0.75,
        )
    ax.set_xlabel("Share of feature groups deleted (%)")
    ax.set_ylabel("Mean allocation change (total variation)")
    ax.set_title("(b) Attribution-guided vs. random deletion")
    ax.legend(frameon=False, fontsize=7, title="dotted = matched random", title_fontsize=7)

    fig.suptitle(
        "Cross-architecture functional comparison (dimensionless, per-model normalisation)",
        fontsize=10,
    )
    fig.tight_layout()
    paths = []
    for ext in ("png", "pdf"):
        p = Path(f"{out_base}.{ext}")
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# Supplementary LaTeX
# ---------------------------------------------------------------------------


def _latex_longtable(
    frame: pd.DataFrame,
    caption: str,
    label: str,
    digits: int = 3,
    max_rows: int = 60,
) -> str:
    df = frame.head(max_rows).copy()
    ncol = len(df.columns)
    align = "l" * 1 + "r" * (ncol - 1) if ncol > 1 else "l"
    header = " & ".join(str(c).replace("_", " ") for c in df.columns)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
    ]
    # A column whose values are all whole numbers reads badly as "16.000".
    per_column = {}
    for column in df.columns:
        values = pd.to_numeric(df[column], errors="coerce")
        integral = values.notna().all() and bool((values == values.round()).all())
        per_column[column] = 0 if integral else digits
    for _, row in df.iterrows():
        lines.append(
            " & ".join(_fmt(row[c], per_column[c]) for c in df.columns) + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}}"]
    if len(frame) > max_rows:
        lines.append(
            rf"\par\footnotesize Showing the first {max_rows} of {len(frame)} rows; the complete "
            r"table is provided as a CSV file in the supplementary archive."
        )
    lines.append(r"\end{table}")
    return "\n".join(lines)


def supplementary_tex(
    summary: pd.DataFrame,
    sample_sizes: pd.DataFrame,
    registry: pd.DataFrame,
    decision_targets: pd.DataFrame,
    metric_dict: pd.DataFrame,
    correlations: pd.DataFrame,
    bootstrap: pd.DataFrame,
    cash_by_regime: pd.DataFrame,
    excluded: pd.DataFrame,
    settings: Dict[str, object],
    figure_names: Sequence[str],
    csv_index: Sequence[str],
) -> str:
    parts: List[str] = []
    parts.append(
        r"""% Generated by scripts/run_cross_architecture_comparison.py -- do not edit by hand.
\section{Supplementary material: cross-architecture diagnostic comparison}
\label{sec:supp-cross-architecture}

This supplement documents the non-Transformer benchmark. The same contribution,
necessity and structure diagnostics were applied to every included policy over a
common, regime-stratified set of evaluation dates. All cross-model quantities are
dimensionless: normalised attribution shares, rank correlations, neighbourhood
overlaps and concentration indices. Raw attribution magnitudes, raw layer
conductance values and raw latent activations are architecture dependent and are
never compared directly across families."""
    )

    parts.append(r"\subsection{Model registry and inclusion decisions}")
    reg_cols = [
        c
        for c in [
            "model_id", "family", "algorithm", "action_space", "lookback",
            "window_length", "n_state_features", "rebalancing_frequency",
            "inclusion_status",
        ]
        if c in registry.columns
    ]
    parts.append(
        _latex_longtable(
            registry.loc[:, reg_cols],
            "Resolved comparison manifest. Every candidate checkpoint discovered under the "
            "Decision Transformer and reinforcement-learning model directories is listed with "
            "its inclusion decision.",
            "tab:supp-registry",
        )
    )
    if len(excluded):
        parts.append(
            _latex_longtable(
                excluded,
                "Excluded models and the documented technical incompatibility that caused each "
                "exclusion. No model was excluded on the basis of its diagnostic or portfolio "
                "results.",
                "tab:supp-excluded",
            )
        )
    else:
        parts.append(
            r"No candidate checkpoint was excluded: every discovered policy satisfied the "
            r"pre-specified technical inclusion criteria."
        )

    parts.append(r"\subsection{Decision targets}")
    dt_cols = [
        c
        for c in ["model_id", "algorithm", "action_space", "primary_decision_target",
                  "action_to_weight_mapping", "latent_layer"]
        if c in decision_targets.columns
    ]
    parts.append(
        _latex_longtable(
            decision_targets.loc[:, dt_cols],
            "Decision target, action-to-weight mapping and latent layer used for each policy.",
            "tab:supp-decision-targets",
        )
    )

    parts.append(r"\subsection{Sample sizes}")
    parts.append(
        _latex_longtable(
            sample_sizes,
            "Number of evaluation dates and of diagnostic-sample observations per model and "
            "market regime. The four-regime configuration partitions the evaluation window "
            "without gaps, so no observation carries an Unknown label.",
            "tab:supp-sample-sizes",
            digits=0,
            max_rows=80,
        )
    )

    parts.append(r"\subsection{Diagnostic definitions}")
    parts.append(
        _latex_longtable(
            metric_dict,
            "The six raw diagnostics, their dimension and their orientation. Larger always "
            "indicates less pathological behaviour.",
            "tab:supp-metric-dictionary",
        )
    )

    parts.append(r"\subsection{Model-level diagnostics}")
    keep = [
        c
        for c in [
            "model_id", "family", "attribution_method", "layer_influence_method",
            "contribution_rank_stability", "contribution_baseline_concordance",
            "deletion_faithfulness", "temporal_profile_stability",
            "latent_neighbourhood_stability", "non_concentrated_layer_influence",
            "contribution_score", "necessity_score", "structure_score",
            "aggregate_diagnostic_score",
        ]
        if c in summary.columns
    ]
    parts.append(
        _latex_longtable(
            summary.loc[:, keep],
            "Raw diagnostics and dimension scores for every included policy.",
            "tab:supp-model-diagnostics",
        )
    )

    parts.append(r"\subsection{Portfolio outcomes}")
    perf = [
        c
        for c in [
            "model_id", "family", "gross_annualised_return", "gross_annualised_volatility",
            "gross_sharpe_ratio", "gross_sortino_ratio", "gross_max_drawdown",
            "net_sharpe_ratio", "annualised_turnover", "mean_cash_allocation",
            "unique_allocations", "weights_source",
        ]
        if c in summary.columns
    ]
    parts.append(
        _latex_longtable(
            summary.loc[:, perf],
            "Portfolio outcomes reconstructed from the frozen policies on the trading calendar "
            "of the evaluation window. Net figures charge 20 basis points on the one-way traded "
            "fraction, the transaction-cost assumption already used in this project.",
            "tab:supp-portfolio",
        )
    )
    parts.append(
        _latex_longtable(
            cash_by_regime,
            "Mean CASH allocation by market regime.",
            "tab:supp-cash-regime",
            max_rows=60,
        )
    )

    parts.append(r"\subsection{Association with performance}")
    parts.append(
        _latex_longtable(
            correlations,
            "Rank correlations between diagnostic scores and the gross Sharpe ratio. These are "
            "descriptive: the number of policies is far too small to support inference, and no "
            "correlation should be read as evidence of prospective prediction.",
            "tab:supp-correlations",
        )
    )
    parts.append(
        _latex_longtable(
            bootstrap,
            "Bootstrap means and 95\\% percentile intervals of the diagnostic scores by family.",
            "tab:supp-bootstrap",
        )
    )

    parts.append(r"\subsection{Figures}")
    for name in figure_names:
        base = Path(name).stem
        parts.append(
            "\n".join(
                [
                    r"\begin{figure}[htbp]",
                    r"\centering",
                    rf"\includegraphics[width=\textwidth]{{{base}}}",
                    rf"\caption{{See Section~\ref{{sec:supp-cross-architecture}}. Source file: "
                    rf"{_tt(base)}.}}",
                    rf"\label{{fig:supp-{base}}}",
                    r"\end{figure}",
                ]
            )
        )

    parts.append(r"\subsection{Computational settings}")
    rows = "\n".join(f"{_tt(k)} & {_fmt(v)} \\\\" for k, v in settings.items())
    parts.append(
        "\n".join(
            [
                r"\begin{table}[htbp]",
                r"\centering",
                r"\caption{Computational settings of the reported run.}",
                r"\label{tab:supp-settings}",
                r"\begin{tabular}{ll}",
                r"\toprule",
                r"Setting & Value \\",
                r"\midrule",
                rows,
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
            ]
        )
    )

    parts.append(r"\subsection{Supplementary data files}")
    parts.append(
        r"\begin{itemize}"
        + "\n".join(rf"\item {_tt(c)}" for c in csv_index)
        + r"\end{itemize}"
    )
    return "\n\n".join(parts) + "\n"
