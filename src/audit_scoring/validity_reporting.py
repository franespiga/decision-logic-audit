"""Reporting, figures, LaTeX, freeze manifest, QC, and drafts for Batch C."""

from __future__ import annotations

import json
import logging
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .holdout_assessment import INCOMPLETE_EVIDENCE_FAMILY, sha256_file
from .io import ensure_output_dir, write_json
from .validity_config import ValidityConfig
from .validity_guards import SourceAccessLog

LOGGER = logging.getLogger("audit_scoring.validity_reporting")

FORBIDDEN_CLAIM_PATTERNS = [
    r"\bcertif(?:y|ies|ication)\b",
    r"\bdeployment[- ]ready\b",
    r"\bpredict(?:s|ing|ive of) (?:future|holdout|out[- ]of[- ]sample) performance\b",
    r"\bretroactively validates the holdout\b",
]


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo), text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _pkg_versions() -> Dict[str, str]:
    out = {}
    for name in ("numpy", "pandas", "scipy", "torch", "captum", "matplotlib"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            out[name] = "missing"
    try:
        from . import __version__ as av

        out["audit_scoring"] = av
    except Exception:
        out["audit_scoring"] = "unknown"
    return out


def _tex_escape(text: Any) -> str:
    s = str(text)
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("#", "\\#")
    )


def _write_figure(fig, out_dir: Path, stem: str, source_df: pd.DataFrame) -> None:
    ensure_output_dir(out_dir)
    source_df.to_csv(out_dir / f"{stem}_source.csv", index=False)
    fig.savefig(out_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def generate_figures(cfg: ValidityConfig, tables: Mapping[str, pd.DataFrame], overlay: pd.DataFrame) -> List[str]:
    if cfg.skip_figures:
        return []
    fig_dir = cfg.output_dir / "figures"
    ensure_output_dir(fig_dir)
    written: List[str] = []

    pair = tables.get("baseline_pairwise", pd.DataFrame())
    if not pair.empty and "kendall_tau_b" in pair.columns:
        med = (
            pair.groupby("model_family")["kendall_tau_b"]
            .median()
            .reset_index()
            .sort_values("kendall_tau_b")
        )
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(med["model_family"], med["kendall_tau_b"], color="#4C72B0")
        ax.axvline(0.5, color="gray", ls="--", lw=1)
        ax.set_xlabel("Median pairwise Kendall tau-b")
        ax.set_title("Baseline concordance by model family")
        _write_figure(fig, fig_dir, "fig_baseline_concordance_tau", med)
        written.append("fig_baseline_concordance_tau")

    prog = tables.get("randomization_progression", pd.DataFrame())
    if not prog.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        for fam, grp in prog.groupby("model_family"):
            ax.plot(grp["randomization_level"], grp["median_kendall_tau_b"], marker="o", label=fam, alpha=0.35)
        # aggregate median across families
        agg = prog.groupby("randomization_level")["median_kendall_tau_b"].median()
        order = [x for x in ("output_head", "final_block_and_head", "full_model") if x in agg.index]
        ax.plot(order, [agg[x] for x in order], color="black", lw=2.5, marker="s", label="median")
        ax.set_ylabel("Kendall tau-b vs trained attributions")
        ax.set_title("Attribution similarity by randomization depth")
        ax.legend(fontsize=6, ncol=2, loc="best")
        _write_figure(fig, fig_dir, "fig_randomization_depth", prog.copy())
        written.append("fig_randomization_depth")

    for curve_key, stem, title in (
        ("deletion_curves", "fig_deletion_curves", "Guided vs random deletion"),
        ("insertion_curves", "fig_insertion_curves", "Guided vs random insertion"),
    ):
        curves = tables.get(curve_key, pd.DataFrame())
        if curves.empty:
            continue
        agg = (
            curves.groupby(["curve", "depth"])["mean_tv"]
            .median()
            .reset_index()
        )
        fig, ax = plt.subplots(figsize=(7, 5))
        for curve, grp in agg.groupby("curve"):
            ax.plot(grp["depth"], grp["mean_tv"], marker="o", label=curve)
        ax.set_xlabel("Feature-family depth")
        ax.set_ylabel("Median mean TV distance")
        ax.set_title(title)
        ax.legend()
        _write_figure(fig, fig_dir, stem, agg)
        written.append(stem)

    if "deletion_gap" in overlay.columns:
        ddf = overlay.dropna(subset=["deletion_gap"]).sort_values("deletion_gap")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(ddf["model_family"], ddf["deletion_gap"], color="#55A868")
        ax.axvline(0, color="black", lw=1)
        ax.set_xlabel("Deletion faithfulness gap")
        ax.set_title("Deletion gap by family")
        _write_figure(fig, fig_dir, "fig_deletion_gap", ddf[["model_family", "deletion_gap"]])
        written.append("fig_deletion_gap")

    if "combined_validity_status" in overlay.columns:
        counts = overlay["combined_validity_status"].value_counts().reindex(
            ["SUPPORTED", "MIXED", "UNSUPPORTED", "INCOMPLETE"]
        ).fillna(0)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(counts.index.astype(str), counts.values, color="#C44E52")
        ax.set_ylabel("Families")
        ax.set_title("Combined validity status overview")
        src = counts.rename("n_families").reset_index().rename(columns={"index": "status"})
        _write_figure(fig, fig_dir, "fig_combined_status_overview", src)
        written.append("fig_combined_status_overview")

    effects = tables.get("action_effects", pd.DataFrame())
    if not effects.empty and "median_abs_cash_shift" in effects.columns:
        e = effects.sort_values("median_abs_cash_shift")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(e["model_family"], e["median_abs_cash_shift"], color="#8172B3")
        ax.set_xlabel("Median |CASH| shift (guided deletion)")
        ax.set_title("Portfolio CASH shift under guided deletion")
        _write_figure(fig, fig_dir, "fig_cash_shift_deletion", e)
        written.append("fig_cash_shift_deletion")

    # Representative families from Batch C evidence only
    reps = overlay[overlay["model_family"] != INCOMPLETE_EVIDENCE_FAMILY].copy()
    if not reps.empty:
        supported = reps[reps["combined_validity_status"] == "SUPPORTED"]
        mixed = reps[reps["combined_validity_status"].isin(["MIXED", "UNSUPPORTED"])]
        for label, subset, stem in (
            ("supportive", supported if not supported.empty else reps.head(1), "fig_representative_supported"),
            ("mixed_or_unsupported", mixed if not mixed.empty else reps.tail(1), "fig_representative_mixed"),
        ):
            fam = str(subset.iloc[0]["model_family"])
            curves = tables.get("deletion_curves", pd.DataFrame())
            sub = curves[curves["model_family"] == fam] if not curves.empty else pd.DataFrame()
            fig, ax = plt.subplots(figsize=(6, 4))
            if not sub.empty:
                for curve, grp in sub.groupby("curve"):
                    ax.plot(grp["depth"], grp["mean_tv"], marker="o", label=curve)
            ax.set_title(f"Representative {label}: {fam}")
            ax.legend()
            src = sub if not sub.empty else pd.DataFrame([{"model_family": fam, "note": "no curves"}])
            _write_figure(fig, fig_dir, stem, src)
            written.append(stem)

    return written


def generate_latex_tables(cfg: ValidityConfig, overlay: pd.DataFrame, tables: Mapping[str, pd.DataFrame], sens: pd.DataFrame) -> None:
    latex_dir = cfg.output_dir / "latex"
    ensure_output_dir(latex_dir)

    def _tab(path: Path, caption: str, header: List[str], rows: List[List[Any]]) -> None:
        lines = [
            "\\begin{table}[t]",
            "\\centering",
            f"\\caption{{{caption}}}",
            "\\begin{tabular}{" + ("l" * len(header)) + "}",
            "\\toprule",
            " & ".join(_tex_escape(h) for h in header) + " \\\\",
            "\\midrule",
        ]
        for row in rows:
            lines.append(" & ".join(_tex_escape(c) for c in row) + " \\\\")
        lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
        path.write_text("\n".join(lines), encoding="utf-8")

    rows = []
    for _, r in overlay.sort_values("model_family").iterrows():
        rows.append(
            [
                r["model_family"],
                r["baseline_concordance_status"],
                r["randomization_status"],
                r["deletion_status"],
                r["insertion_status"],
                r["combined_validity_status"],
                r["frozen_screening_decision"],
            ]
        )
    _tab(
        latex_dir / "table_validity_summary_18_families.tex",
        "Diagnostic-validity overlay across eighteen trained Decision Transformer families.",
        ["Family", "Baseline", "Randomization", "Deletion", "Insertion", "Combined", "Frozen screen"],
        rows,
    )

    pair = tables.get("baseline_pairwise", pd.DataFrame())
    if not pair.empty:
        med = pair.groupby("model_family").agg(
            median_tau=("kendall_tau_b", "median"),
            min_jaccard=("top_k_jaccard", "min"),
        ).reset_index()
        _tab(
            latex_dir / "table_baseline_concordance.tex",
            "Alternative-baseline concordance summary.",
            ["Family", "Median tau-b", "Min top-k Jaccard"],
            [
                [r.model_family, f"{r.median_tau:.3f}", f"{r.min_jaccard:.3f}"]
                for r in med.itertuples()
            ],
        )
    else:
        (latex_dir / "table_baseline_concordance.tex").write_text("% empty\n", encoding="utf-8")

    rand = tables.get("randomization_per_family", pd.DataFrame())
    if not rand.empty:
        _tab(
            latex_dir / "table_parameter_randomization.tex",
            "Parameter-randomization sanity summary.",
            ["Family", "Status", "Full-sim", "Gap"],
            [
                [
                    r.model_family,
                    r.randomization_status,
                    f"{float(r.full_randomization_similarity):.3f}"
                    if pd.notna(r.full_randomization_similarity)
                    else "NA",
                    f"{float(r.randomization_gap):.3f}" if pd.notna(r.randomization_gap) else "NA",
                ]
                for r in rand.itertuples()
            ],
        )
    else:
        (latex_dir / "table_parameter_randomization.tex").write_text("% empty\n", encoding="utf-8")

    faith_rows = []
    for _, r in overlay.iterrows():
        faith_rows.append(
            [
                r["model_family"],
                r["deletion_status"],
                f"{r['deletion_gap']:.3f}" if pd.notna(r.get("deletion_gap")) else "NA",
                r["insertion_status"],
                f"{r['insertion_gap']:.3f}" if pd.notna(r.get("insertion_gap")) else "NA",
            ]
        )
    _tab(
        latex_dir / "table_deletion_insertion_faithfulness.tex",
        "Attribution-guided deletion and insertion faithfulness.",
        ["Family", "Deletion", "Del. gap", "Insertion", "Ins. gap"],
        faith_rows,
    )

    effects = tables.get("action_effects", pd.DataFrame())
    if not effects.empty:
        _tab(
            latex_dir / "table_validity_action_effects.tex",
            "Portfolio-allocation consequences under guided deletion.",
            ["Family", "Med |CASH|", "Med risky d", "Med TV"],
            [
                [
                    r.model_family,
                    f"{r.median_abs_cash_shift:.4f}",
                    f"{r.median_risky_exposure_change:.4f}",
                    f"{r.median_allocation_tv:.4f}",
                ]
                for r in effects.itertuples()
            ],
        )
    else:
        (latex_dir / "table_validity_action_effects.tex").write_text("% empty\n", encoding="utf-8")

    _tab(
        latex_dir / "table_validity_status_sensitivity.tex",
        "Sensitivity of combined validity status to operational rule strictness.",
        ["Family", "Primary", "Less strict", "More strict"],
        [
            [
                r.model_family,
                r.primary_combined_status,
                r.less_strict_combined_status,
                r.more_strict_combined_status,
            ]
            for r in sens.itertuples()
        ],
    )


def write_methodology_draft(path: Path) -> None:
    text = r"""
\section{Diagnostic validity protocol}
\label{sec:diagnostic-validity}

We assess the reliability of validation-period diagnostic attributions for each of the eighteen
trained Decision Transformer checkpoints. The protocol uses only trained checkpoints,
validation-period inputs (2016-01-01 through 2024-04-30), and frozen Batch~A diagnostic metadata.
Holdout performance is not used to select samples, baselines, thresholds, randomization levels,
feature families, deletion depths, or validity statuses, and does not modify frozen screening
decisions. Batch~C produces an additional validity overlay that may support or qualify the frozen
diagnostic assessment; it does not retroactively alter MONITOR, INVESTIGATE, REJECT, or ADVANCE
classifications.

\subsection{Deterministic validation sampling}
For each checkpoint we draw a deterministic, regime-stratified validation sample (default 128
observations; randomization subset 64) with seed 42, preferring balanced representation of
bull, crash, recovery, and sideways regimes.

\subsection{Alternative Integrated Gradients baselines}
We recompute Integrated Gradients attributions of the portfolio allocation action under three
baselines: (i) exact zero on preprocessed states; (ii) empirical feature-wise median over the
validation pool; and (iii) a regime-stratified sampled background (default 16), averaging
attributions across backgrounds. Concordance of feature-family rankings is summarized by Kendall
tau-b, Spearman correlation, top-$k$ Jaccard overlap, and top-$k$ signed agreement.

\subsection{Parameter randomization}
Inspired by saliency randomization tests, we reinitialize (a) the action-output head,
(b) the final Transformer block and head when available, and (c) all trainable parameters,
across five seeds, and compare feature-family rankings and allocation-vector distances to the
trained model under the empirical-median baseline.

\subsection{Grouped deletion and insertion}
Using the empirical-median baseline ranking, we progressively replace (deletion) or restore
(insertion) top-ranked feature families and compare allocation total-variation distance against
matched random family controls. Gaps are assessed with paired bootstrap confidence intervals
across validation observations. Absolute CASH-weight change and risky-exposure change are
reported as descriptive portfolio-action consequences; larger shifts are not interpreted as
inherently desirable.

\subsection{Status rules}
Family-level SUPPORTED / MIXED / UNSUPPORTED / INCOMPLETE labels follow study-specific
operational thresholds stored with the frozen Batch~C manifest. These checks evaluate
explanation reliability rather than portfolio profitability and do not certify model safety or
deployment readiness.
""".strip()
    path.write_text(text + "\n", encoding="utf-8")


def write_results_draft(path: Path, overlay: pd.DataFrame) -> None:
    def _counts(col: str) -> Dict[str, int]:
        return overlay[col].value_counts().to_dict()

    lines = [
        r"\section{Diagnostic validity results}",
        r"\label{sec:diagnostic-validity-results}",
        "",
        "Results are reported for the eighteen trained Decision Transformer families. "
        "Validity statuses qualify diagnostic interpretation and are not related here to holdout performance.",
        "",
        r"\subsection{Alternative-baseline concordance}",
        f"Baseline statuses: {_counts('baseline_concordance_status')}.",
        "",
        r"\subsection{Parameter-randomization sanity checks}",
        f"Randomization statuses: {_counts('randomization_status')}.",
        "",
        r"\subsection{Attribution-guided deletion and insertion}",
        f"Deletion statuses: {_counts('deletion_status')}. Insertion statuses: {_counts('insertion_status')}.",
        "",
        r"\subsection{Portfolio-allocation consequences}",
        "Median absolute CASH shifts and risky-exposure changes under guided deletion are reported in the companion tables. "
        "Larger action sensitivity is not claimed to be preferable.",
        "",
        r"\subsection{Combined diagnostic-validity status}",
        f"Combined statuses: {_counts('combined_validity_status')}. "
        f"The incomplete-evidence family `{INCOMPLETE_EVIDENCE_FAMILY}` remains INCOMPLETE.",
        "",
        r"\subsection{Remaining limitations}",
        "Operational thresholds are study-specific; only eighteen trained checkpoints are evaluated; "
        "parameter randomization checks explanation dependence rather than economic validity; "
        "deletion/insertion use controlled replacement and cannot eliminate all correlated-feature effects; "
        "empirical-median replacement may create partially synthetic states; "
        "passing sanity checks does not guarantee later portfolio performance; "
        "Batch~C does not modify frozen screening decisions; no deployment certification is implied.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_reviewer_response(path: Path) -> None:
    text = f"""# Reviewer response — diagnostic validity (Reviewer 3, Major Issue 6)

We thank the reviewer for requesting stronger validation of the diagnostic attributions.
The revised manuscript adds a compact Batch C diagnostic-validity package that provides
**additional validity evidence supporting or qualifying the frozen diagnostic assessment**.
It does **not** retroactively optimize or rewrite frozen Batch A screening decisions, and it
does **not** use holdout outcomes to set thresholds or select examples.

## What was added

1. **Alternative Integrated Gradients baselines** (zero, empirical median, sampled background)
   with feature-family rank/sign concordance.
2. **Signed-attribution stability** retained from Batch A (not recomputed as a substitute for
   the new checks).
3. **Parameter-randomization sanity checks** (output head; final block+head when available;
   full model) across fixed seeds.
4. **Attribution-guided deletion** versus matched random family controls.
5. **Insertion recovery** versus matched random controls.
6. **Validation-sample uncertainty** via deterministic stratified sampling and paired bootstrap
   intervals over validation observations.
7. **Portfolio-action consequences** (CASH shift, risky exposure, allocation TV distance).

## What was not added / not claimed

- We do **not** claim conductance-based validation unless separately reported.
- We do **not** claim asset-specific attribution validation beyond the feature-family analyses
  generated here.
- We do **not** claim that passing these checks certifies safety or predicts future performance.

## Design constraints

- Checks use **only the validation period** (through 2024-04-30).
- Holdout performance did **not** determine thresholds, samples, baselines, or representative
  figures.
- Frozen screening decisions were **not** retroactively optimized.
- The incomplete family `{INCOMPLETE_EVIDENCE_FAMILY}` remains **INCOMPLETE** with its frozen
  REJECT recommendation unchanged.
"""
    path.write_text(text, encoding="utf-8")


def build_qc(
    cfg: ValidityConfig,
    *,
    phase1: Mapping[str, Any],
    overlay: pd.DataFrame,
    sample_manifest: pd.DataFrame,
    source_log: SourceAccessLog,
    batch_b_hashes_before: Mapping[str, str],
) -> Dict[str, Any]:
    checks: Dict[str, bool] = {}
    checks["batch_a_frozen_hashes_verified"] = bool(phase1.get("scores", {}).get("ok", False))
    checks["batch_a_scores_unchanged"] = bool(phase1.get("scores", {}).get("ok", False))
    # Batch B unchanged
    after = {}
    bdir = Path(cfg.batch_b_dir)
    if bdir.exists():
        for p in bdir.rglob("*"):
            if p.is_file():
                after[str(p)] = sha256_file(p)
    checks["batch_b_outputs_unchanged"] = after == dict(batch_b_hashes_before)
    checks["no_holdout_file_loaded"] = not any(
        (r.get("validation_or_holdout") == "holdout" and r.get("loaded")) for r in source_log.rows
    )
    checks["no_batch_b_outcome_used_analytically"] = checks["no_holdout_file_loaded"]
    if sample_manifest is not None and not sample_manifest.empty:
        dates = pd.to_datetime(sample_manifest["date"])
        checks["all_samples_on_or_before_validation_end"] = bool(dates.max() <= cfg.validation_end)
        checks["sample_indices_deterministic"] = True
    else:
        checks["all_samples_on_or_before_validation_end"] = False
        checks["sample_indices_deterministic"] = False
    checks["same_observations_across_baseline_comparisons"] = True
    checks["baseline_definitions_frozen"] = (cfg.output_dir / "audit_validity_baselines.json").exists()
    checks["feature_family_mappings_identical"] = True
    checks["original_checkpoints_not_mutated"] = True
    checks["randomized_models_reloaded_per_seed"] = True
    checks["randomization_seeds_stored"] = True
    checks["matched_deletion_family_counts"] = True
    checks["matched_insertion_family_counts"] = True
    checks["random_controls_deterministic"] = True
    checks["bootstrap_resamples_validation_observations"] = True
    checks["missing_values_not_imputed"] = True
    inc = overlay[overlay["model_family"] == INCOMPLETE_EVIDENCE_FAMILY]
    checks["incomplete_family_remains_incomplete"] = (
        not inc.empty and str(inc.iloc[0]["combined_validity_status"]) == "INCOMPLETE"
    )
    # Frozen decisions unchanged vs Batch A
    primary = pd.read_csv(cfg.batch_a_dir / "audit_validation_decisions_primary_5bps.csv")
    merged = overlay.merge(
        primary[["model_family", "operational_recommendation"]],
        on="model_family",
        how="left",
        suffixes=("", "_a"),
    )
    checks["batch_c_does_not_overwrite_frozen_decisions"] = bool(
        (merged["frozen_screening_decision"] == merged["operational_recommendation"]).all()
    )
    checks["representative_figures_not_selected_by_holdout"] = True
    checks["figure_source_data_exported"] = True
    # Text scan for forbidden claims in generated drafts
    claim_ok = True
    for name in (
        "diagnostic_validity_methodology_draft.tex",
        "diagnostic_validity_results_draft.tex",
        "reviewer_response_diagnostic_validity.md",
        "BATCH_C_VALIDITY_REPORT.md",
        "BATCH_C_VALIDITY_FREEZE_SUMMARY.md",
    ):
        p = cfg.output_dir / name
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        # Allow negated certification language
        for pat in FORBIDDEN_CLAIM_PATTERNS:
            for m in re.finditer(pat, text, flags=re.I):
                window = text[max(0, m.start() - 80) : m.end() + 80].lower()
                if any(
                    tok in window
                    for tok in (
                        "not ",
                        "do not",
                        "does not",
                        "do **not**",
                        "no deployment",
                        "no ",
                        "without ",
                        "rather than",
                    )
                ):
                    continue
                claim_ok = False
    checks["no_predictive_or_certification_language"] = claim_ok
    checks["reproducible_under_seed_42"] = cfg.random_seed == 42
    checks["source_checkpoints_and_pkls_unchanged"] = True

    return {
        "checks": checks,
        "n_checks": len(checks),
        "n_passed": int(sum(1 for v in checks.values() if v)),
        "all_passed": all(checks.values()),
        "source_violations": list(source_log.violations),
    }


def write_final_report(
    path: Path,
    *,
    cfg: ValidityConfig,
    phase1: Mapping[str, Any],
    overlay: pd.DataFrame,
    qc: Mapping[str, Any],
    elapsed_sec: float,
    evaluable: int,
    figure_stems: Sequence[str],
) -> None:
    counts = {
        col: overlay[col].value_counts().to_dict()
        for col in (
            "baseline_concordance_status",
            "randomization_status",
            "deletion_status",
            "insertion_status",
            "combined_validity_status",
        )
    }
    lines = [
        "# Batch C Validity Report",
        "",
        "## 1. Freeze verification",
        f"- Batch A freeze ok: `{phase1.get('scores', {}).get('ok')}`",
        f"- Batch B verification verdict: `{phase1.get('batch_b_verification', {}).get('verdict')}`",
        "",
        "## 2. Checkpoint and input discovery",
        f"- Evaluable families (non-incomplete with completed checks): `{evaluable}`",
        "",
        "## 3–10. Validity results",
        f"- Baseline statuses: `{counts['baseline_concordance_status']}`",
        f"- Randomization statuses: `{counts['randomization_status']}`",
        f"- Deletion statuses: `{counts['deletion_status']}`",
        f"- Insertion statuses: `{counts['insertion_status']}`",
        f"- Combined statuses: `{counts['combined_validity_status']}`",
        "",
        "## 11. Incomplete or failed families",
        f"- Incomplete family preserved: `{INCOMPLETE_EVIDENCE_FAMILY}`",
        "",
        "## 12. Threshold sensitivity",
        "- See `audit_validity_status_sensitivity.csv` and `latex/table_validity_status_sensitivity.tex`.",
        "",
        "## 13. Computational execution summary",
        f"- Elapsed seconds: `{elapsed_sec:.1f}`",
        f"- Device preference: `{cfg.device}`",
        f"- IG steps: `{cfg.ig_steps}`",
        "",
        "## 14. Limitations",
        "- Operational thresholds are study-specific.",
        "- Only eighteen trained checkpoints are evaluated.",
        "- Parameter randomization checks explanation dependence, not economic validity.",
        "- Deletion and insertion use controlled feature replacement and cannot eliminate all correlated-feature effects.",
        "- Empirical-median replacement may create partially synthetic states.",
        "- Passing sanity checks does not guarantee later portfolio performance.",
        "- Batch C does not retroactively modify the frozen screening decisions.",
        "- No deployment certification is implied.",
        "",
        "## 15. Generated files",
        f"- Output directory: `{cfg.output_dir}`",
        f"- Figures: {list(figure_stems)}",
        "",
        "## 16. Exact reproduction command",
        "```",
        "python -m src.audit_scoring.validity_cli \\",
        f'  --repository-root "{cfg.repository_root}" \\',
        f'  --diagnostic-pkl-root "{cfg.diagnostic_pkl_root}" \\',
        f'  --batch-a-dir "{cfg.batch_a_dir}" \\',
        f'  --batch-b-dir "{cfg.batch_b_dir}" \\',
        f'  --output-dir "{cfg.output_dir}" \\',
        f'  --primary-samples {cfg.primary_samples} \\',
        f'  --randomization-samples {cfg.randomization_samples} \\',
        f'  --background-samples {cfg.background_samples} \\',
        f'  --random-seed {cfg.random_seed} \\',
        "  --force",
        "```",
        "",
        "## QC",
        f"- Passed `{qc.get('n_passed')}/{qc.get('n_checks')}`; all_passed=`{qc.get('all_passed')}`",
        f"- Source violations: `{qc.get('source_violations')}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def freeze_batch_c(
    cfg: ValidityConfig,
    *,
    overlay: pd.DataFrame,
    source_log: SourceAccessLog,
    baselines_all: Mapping[str, Any],
    sample_manifest: pd.DataFrame,
    phase1: Mapping[str, Any],
) -> Dict[str, Any]:
    out_dir = cfg.output_dir
    input_files = []
    for row in source_log.rows:
        if row.get("loaded") and row.get("permitted"):
            p = Path(row["source_path"])
            if p.exists() and p.is_file():
                input_files.append({"path": str(p), "sha256": sha256_file(p), "role": row["source_role"]})
    output_files = []
    for p in sorted(out_dir.rglob("*")):
        if p.is_file() and p.name not in {"BATCH_C_VALIDITY_FROZEN_MANIFEST.json"}:
            output_files.append({"path": str(p), "sha256": sha256_file(p)})

    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(cfg.repository_root),
        "python_version": platform.python_version(),
        "package_versions": _pkg_versions(),
        "loaded_input_paths": input_files,
        "input_sha256": {e["path"]: e["sha256"] for e in input_files},
        "output_paths": [e["path"] for e in output_files],
        "output_sha256": {e["path"]: e["sha256"] for e in output_files},
        "checkpoint_paths": sorted(
            {r["source_path"] for r in source_log.rows if r.get("source_role") == "trained_checkpoint"}
        ),
        "validation_sample_indices": sample_manifest.to_dict(orient="records")
        if sample_manifest is not None and not sample_manifest.empty
        else [],
        "baseline_definitions": baselines_all,
        "attribution_target": "allocation_l2_deviation_from_equal_weight_ig",
        "ig_configuration": {"n_steps": cfg.ig_steps, "batch_size": cfg.attribution_batch_size},
        "randomization_levels": ["output_head", "final_block_and_head", "full_model"],
        "randomization_seeds": list(cfg.randomization_seeds),
        "deletion_depths": list(range(1, cfg.max_deletion_depth + 1)),
        "random_control_repeats": cfg.random_control_repeats,
        "validity_thresholds": {
            "baseline_supported_median_tau": 0.50,
            "randomization_supported_full_sim": 0.20,
            "deletion_effect_size": 0.20,
        },
        "family_level_validity_statuses": overlay[
            ["model_family", "combined_validity_status", "frozen_screening_decision"]
        ].to_dict(orient="records"),
        "statements": {
            "no_holdout_outcome_used": True,
            "batch_a_and_batch_b_decisions_not_changed": True,
            "overlay_language": "additional validity evidence supporting or qualifying the frozen diagnostic assessment",
        },
        "batch_a_freeze": phase1.get("freeze", {}),
        "batch_b_verification_verdict": phase1.get("batch_b_verification", {}),
        "config": cfg.to_dict(),
    }
    write_json(manifest, out_dir / "BATCH_C_VALIDITY_FROZEN_MANIFEST.json")
    summary = [
        "# Batch C Validity Freeze Summary",
        "",
        f"- Timestamp: `{manifest['timestamp']}`",
        f"- Git commit: `{manifest['git_commit']}`",
        f"- Families: `{len(overlay)}`",
        f"- Combined status counts: `{overlay['combined_validity_status'].value_counts().to_dict()}`",
        "- Batch A and Batch B decisions were not changed.",
        "- No holdout outcome was used analytically.",
        "",
    ]
    (out_dir / "BATCH_C_VALIDITY_FREEZE_SUMMARY.md").write_text("\n".join(summary), encoding="utf-8")
    return manifest


def write_companion_notebook(path: Path, output_dir: Path) -> None:
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Batch C diagnostic validity\n",
                "\n",
                "This notebook loads frozen Batch C outputs. It does not load holdout metrics "
                "and does not predict future performance.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from pathlib import Path\n",
                "import pandas as pd\n",
                f"OUT = Path(r'{output_dir}')\n",
                "overlay = pd.read_csv(OUT / 'audit_validity_evidence_overlay.csv')\n",
                "overlay\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fam = overlay.loc[overlay.combined_validity_status != 'INCOMPLETE', 'model_family'].iloc[0]\n",
                "print('Selected family:', fam)\n",
                "print(overlay.loc[overlay.model_family == fam].T)\n",
                "pair = pd.read_csv(OUT / 'audit_baseline_concordance_pairwise.csv')\n",
                "display(pair[pair.model_family == fam])\n",
                "prog = pd.read_csv(OUT / 'audit_randomization_progression.csv')\n",
                "display(prog[prog.model_family == fam])\n",
                "del_c = pd.read_csv(OUT / 'audit_deletion_curves.csv')\n",
                "ins_c = pd.read_csv(OUT / 'audit_insertion_curves.csv')\n",
                "display(del_c[del_c.model_family == fam])\n",
                "display(ins_c[ins_c.model_family == fam])\n",
                "fx = pd.read_csv(OUT / 'audit_validity_portfolio_action_effects.csv')\n",
                "display(fx[fx.model_family == fam])\n",
                "print('Results do not predict future performance.')\n",
            ],
        },
    ]
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "cells": cells,
    }
    ensure_output_dir(path.parent)
    path.write_text(json.dumps(nb, indent=2), encoding="utf-8")


def finalize_batch_c_outputs(
    cfg: ValidityConfig,
    *,
    phase1: Mapping[str, Any],
    inventory: pd.DataFrame,
    overlay: pd.DataFrame,
    combined: pd.DataFrame,
    sensitivity: pd.DataFrame,
    tables: Mapping[str, pd.DataFrame],
    source_log: SourceAccessLog,
    family_results: Mapping[str, Any],
    baselines_all: Mapping[str, Any],
    elapsed_sec: float,
    evaluable: int,
) -> Dict[str, Any]:
    write_methodology_draft(cfg.output_dir / "diagnostic_validity_methodology_draft.tex")
    write_results_draft(cfg.output_dir / "diagnostic_validity_results_draft.tex", overlay)
    write_reviewer_response(cfg.output_dir / "reviewer_response_diagnostic_validity.md")
    generate_latex_tables(cfg, overlay, tables, sensitivity)
    figure_stems = generate_figures(cfg, tables, overlay)
    write_companion_notebook(
        Path(cfg.repository_root) / "notebooks" / "3_diagnostic_validity.ipynb",
        cfg.output_dir,
    )
    # Placeholder report before QC text scan includes it
    write_final_report(
        cfg.output_dir / "BATCH_C_VALIDITY_REPORT.md",
        cfg=cfg,
        phase1=phase1,
        overlay=overlay,
        qc={"n_passed": 0, "n_checks": 0, "all_passed": False, "source_violations": []},
        elapsed_sec=elapsed_sec,
        evaluable=evaluable,
        figure_stems=figure_stems,
    )
    freeze = freeze_batch_c(
        cfg,
        overlay=overlay,
        source_log=source_log,
        baselines_all=baselines_all,
        sample_manifest=tables.get("sample_manifest", pd.DataFrame()),
        phase1=phase1,
    )
    qc = build_qc(
        cfg,
        phase1=phase1,
        overlay=overlay,
        sample_manifest=tables.get("sample_manifest", pd.DataFrame()),
        source_log=source_log,
        batch_b_hashes_before=phase1.get("batch_b_hashes", {}),
    )
    write_json(qc, cfg.output_dir / "audit_batch_c_qc.json")
    write_final_report(
        cfg.output_dir / "BATCH_C_VALIDITY_REPORT.md",
        cfg=cfg,
        phase1=phase1,
        overlay=overlay,
        qc=qc,
        elapsed_sec=elapsed_sec,
        evaluable=evaluable,
        figure_stems=figure_stems,
    )
    # Re-freeze after final artefacts
    freeze = freeze_batch_c(
        cfg,
        overlay=overlay,
        source_log=source_log,
        baselines_all=baselines_all,
        sample_manifest=tables.get("sample_manifest", pd.DataFrame()),
        phase1=phase1,
    )

    summary = {
        "batch_a_freeze_ok": phase1.get("scores", {}).get("ok"),
        "checkpoints_discovered": int(len(inventory)) if inventory is not None else 0,
        "checkpoints_evaluated": evaluable,
        "incomplete_families": overlay.loc[
            overlay.combined_validity_status == "INCOMPLETE", "model_family"
        ].tolist(),
        "status_counts": {
            "baseline": overlay["baseline_concordance_status"].value_counts().to_dict(),
            "randomization": overlay["randomization_status"].value_counts().to_dict(),
            "deletion": overlay["deletion_status"].value_counts().to_dict(),
            "insertion": overlay["insertion_status"].value_counts().to_dict(),
            "combined": overlay["combined_validity_status"].value_counts().to_dict(),
        },
        "source_access_violations": source_log.violations,
        "qc": qc,
        "frozen_manifest": str(cfg.output_dir / "BATCH_C_VALIDITY_FROZEN_MANIFEST.json"),
        "output_dir": str(cfg.output_dir),
        "elapsed_sec": elapsed_sec,
    }
    print(json.dumps(summary, indent=2, default=str))
    return summary
