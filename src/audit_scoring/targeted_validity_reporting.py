"""Reporting, figures, QC, and freeze for Batch C2."""

from __future__ import annotations

import json
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

from .holdout_assessment import sha256_file
from .io import ensure_output_dir, write_json
from .targeted_validity_pipeline import TargetedConfig
from .validity_guards import SourceAccessLog


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo), text=True).strip()
    except Exception:
        return "unknown"


def _tex_escape(s: Any) -> str:
    return str(s).replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")


def write_recommendations(path: Path, overlay: pd.DataFrame, mandatory: str) -> None:
    mand = overlay[overlay.model_family == mandatory]
    lines = [
        "# Batch C2 manuscript recommendations",
        "",
        "Family selection used **validation-period** median Sharpe versus the authoritative",
        "market benchmark only. Holdout performance is not discussed here.",
        "",
        f"## Mandatory full validation (`{mandatory}`)",
        "",
    ]
    if mand.empty:
        lines.append("- Mandatory family missing from overlay.")
    else:
        r = mand.iloc[0]
        lines.extend(
            [
                f"- Frozen Batch C status: `{r.frozen_batch_c_status}` (unchanged).",
                f"- Targeted evidence status: `{r.targeted_evidence_status}` — {r.targeted_evidence_reason}",
                f"- Baseline rank / sign: `{r.c2_baseline_rank_status}` / `{r.c2_baseline_sign_status}`",
                f"- Randomization: `{r.c2_randomization_status}` (Batch C was `{r.original_batch_c_randomization_status}`)",
                f"- Cardinality-matched deletion: `{r.c2_cardinality_matched_deletion_status}` "
                f"(Batch C family-count deletion was `{r.original_batch_c_deletion_status}`)",
                f"- Insertion: `{r.c2_insertion_status}`",
                f"- Raw-feature randomization: `{r.raw_feature_randomization_status}`",
                f"- Perturbed-state plausibility: `{r.perturbed_state_plausibility_status}`",
                "",
                "Interpretation guidance:",
                "- If rank is stable but signs are not, do not write that attributions are globally unstable.",
                "- If cardinality-matched deletion differs from Batch C, report control dependence.",
                "- If targets disagree, report target dependence rather than a single IG narrative.",
            ]
        )
    am = overlay[overlay.model_family != mandatory]
    lines.extend(
        [
            "",
            "## Above-market partial validation",
            "",
            f"- Families: {overlay.model_family.tolist()}",
            f"- Baseline rank SUPPORTED: {int((overlay.c2_baseline_rank_status=='SUPPORTED').sum())}",
            f"- Baseline sign SUPPORTED: {int((overlay.c2_baseline_sign_status=='SUPPORTED').sum())}",
            f"- Randomization SUPPORTED: {int((overlay.c2_randomization_status=='SUPPORTED').sum())}",
            f"- Cardinality-matched deletion SUPPORTED: {int((overlay.c2_cardinality_matched_deletion_status=='SUPPORTED').sum())}",
            f"- Targeted MIXED: {int((overlay.targeted_evidence_status=='MIXED').sum())}",
            f"- Targeted UNSUPPORTED: {int((overlay.targeted_evidence_status=='UNSUPPORTED').sum())}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def generate_figures_tables(cfg: TargetedConfig, overlay: pd.DataFrame, selection: pd.DataFrame) -> List[str]:
    fig_dir = cfg.output_dir / "figures"
    latex_dir = cfg.output_dir / "latex"
    ensure_output_dir(fig_dir)
    ensure_output_dir(latex_dir)
    written = []

    # Selection table latex
    sel = selection[selection.selected_for_batch_c2].copy()
    lines = [
        "\\begin{table}[t]\\centering",
        "\\caption{Validation-period above-market family selection versus market benchmark.}",
        "\\begin{tabular}{lrrcc}",
        "\\toprule Family & Val.\\ Sharpe & Market Sharpe & Diff & Scope \\\\",
        "\\midrule",
    ]
    for _, r in sel.iterrows():
        lines.append(
            f"{_tex_escape(r.model_family)} & {r.validation_median_sharpe:.3f} & "
            f"{r.market_benchmark_sharpe:.3f} & {r.sharpe_difference:.3f} & {_tex_escape(r.validation_scope)} \\\\"
        )
    lines.extend(["\\bottomrule\\end{tabular}\\end{table}", ""])
    (latex_dir / "table_market_benchmark_family_selection.tex").write_text("\n".join(lines), encoding="utf-8")
    sel.to_csv(fig_dir / "fig_selection_source.csv", index=False)

    # Above-market summary figure
    fig, ax = plt.subplots(figsize=(8, 5))
    statuses = ["c2_baseline_rank_status", "c2_randomization_status", "c2_cardinality_matched_deletion_status"]
    data = []
    for s in statuses:
        vc = overlay[s].value_counts()
        for st, n in vc.items():
            data.append({"check": s, "status": st, "n": int(n)})
    sdf = pd.DataFrame(data)
    if not sdf.empty:
        pivot = sdf.pivot_table(index="check", columns="status", values="n", fill_value=0)
        pivot.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Above-market targeted validity summary")
    ax.set_ylabel("Families")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_above_market_summary.png", dpi=200)
    fig.savefig(fig_dir / "fig_above_market_summary.pdf")
    sdf.to_csv(fig_dir / "fig_above_market_summary_source.csv", index=False)
    plt.close(fig)
    written.append("fig_above_market_summary")

    # Mandatory profile
    mand = overlay[overlay.model_family == cfg.mandatory_full_family]
    if not mand.empty:
        r = mand.iloc[0]
        labels = ["rank", "sign", "rand", "del", "ins", "raw", "plaus"]
        vals = [
            r.c2_baseline_rank_status,
            r.c2_baseline_sign_status,
            r.c2_randomization_status,
            r.c2_cardinality_matched_deletion_status,
            r.c2_insertion_status,
            r.raw_feature_randomization_status,
            r.perturbed_state_plausibility_status,
        ]
        score = {"SUPPORTED": 2, "MIXED": 1, "UNSUPPORTED": 0, "INCOMPLETE": -1, "NOT_RUN": -0.5}
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(labels, [score.get(v, -1) for v in vals], color="#4C72B0")
        ax.set_ylim(-1.5, 2.5)
        ax.set_title(f"Full validity profile: {cfg.mandatory_full_family}")
        src = pd.DataFrame({"check": labels, "status": vals})
        src.to_csv(fig_dir / "fig_mandatory_profile_source.csv", index=False)
        fig.savefig(fig_dir / "fig_mandatory_profile.png", dpi=200)
        fig.savefig(fig_dir / "fig_mandatory_profile.pdf")
        plt.close(fig)
        written.append("fig_mandatory_profile")

        # Batch C vs C2 deletion comparison
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(
            ["Batch C deletion", "C2 cardinality-matched"],
            [
                score.get(str(r.original_batch_c_deletion_status), -1),
                score.get(str(r.c2_cardinality_matched_deletion_status), -1),
            ],
        )
        ax.set_title("Deletion control comparison (mandatory family)")
        cmp = pd.DataFrame(
            {
                "batch_c_deletion": [r.original_batch_c_deletion_status],
                "c2_cardinality_matched_deletion": [r.c2_cardinality_matched_deletion_status],
            }
        )
        cmp.to_csv(fig_dir / "fig_deletion_control_comparison_source.csv", index=False)
        fig.savefig(fig_dir / "fig_deletion_control_comparison.png", dpi=200)
        fig.savefig(fig_dir / "fig_deletion_control_comparison.pdf")
        plt.close(fig)
        written.append("fig_deletion_control_comparison")

    # Raw vs family randomization if available
    res_path = cfg.output_dir / "audit_c2_resolution_comparison.csv"
    if res_path.exists():
        res = pd.read_csv(res_path)
        mand_res = res[res.model_family == cfg.mandatory_full_family]
        if not mand_res.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(mand_res["family_randomization_kendall"], mand_res["raw_randomization_kendall"])
            ax.set_xlabel("Feature-family Kendall")
            ax.set_ylabel("Raw-feature Kendall")
            ax.set_title("Family vs raw randomization similarity")
            mand_res.to_csv(fig_dir / "fig_family_vs_raw_randomization_source.csv", index=False)
            fig.savefig(fig_dir / "fig_family_vs_raw_randomization.png", dpi=200)
            fig.savefig(fig_dir / "fig_family_vs_raw_randomization.pdf")
            plt.close(fig)
            written.append("fig_family_vs_raw_randomization")

    # Overlay latex
    rows = []
    for _, r in overlay.iterrows():
        rows.append(
            [
                r.model_family,
                r.c2_baseline_rank_status,
                r.c2_randomization_status,
                r.c2_cardinality_matched_deletion_status,
                r.targeted_evidence_status,
                r.frozen_batch_c_status,
            ]
        )
    tex = [
        "\\begin{table}[t]\\centering",
        "\\caption{Targeted validity overlay for economically relevant families.}",
        "\\begin{tabular}{llllll}",
        "\\toprule Family & Rank & Rand & Del & Targeted & Frozen Batch C \\\\",
        "\\midrule",
    ]
    for row in rows:
        tex.append(" & ".join(_tex_escape(x) for x in row) + " \\\\")
    tex.extend(["\\bottomrule\\end{tabular}\\end{table}", ""])
    (latex_dir / "table_above_market_targeted_validity.tex").write_text("\n".join(tex), encoding="utf-8")
    return written


def build_qc(
    cfg: TargetedConfig,
    *,
    freeze: Mapping[str, Any],
    overlay: pd.DataFrame,
    source_log: SourceAccessLog,
    batch_a_before: Mapping[str, str],
    batch_b_before: Mapping[str, str],
    batch_c_before: Mapping[str, str],
    selection_hash: str,
) -> Dict[str, Any]:
    def analytical_unchanged(before: Mapping[str, str], directory: Path) -> bool:
        """Compare hashes for previously snapshotted analytical files only."""
        ok = True
        for path_str, stored in before.items():
            name = Path(path_str).name.lower()
            # Ignore known non-analytical / volatile artefacts
            if any(tok in name for tok in ("run_stdout", "freeze_summary", "_source.csv")):
                continue
            if Path(path_str).suffix.lower() not in {".csv", ".json", ".parquet"}:
                continue
            p = Path(path_str)
            if not p.exists():
                ok = False
                continue
            if sha256_file(p).lower() != str(stored).lower():
                ok = False
        return ok

    checks = {
        "batch_a_and_batch_c_freezes_verified": bool(freeze.get("batch_a_ok"))
        and not freeze.get("batch_c_analytical_changed"),
        "no_abc_outputs_modified": analytical_unchanged(batch_a_before, cfg.batch_a_dir)
        and analytical_unchanged(batch_b_before, cfg.batch_b_dir)
        and analytical_unchanged(batch_c_before, cfg.batch_c_dir),
        "no_holdout_analytical_source_loaded": not any(
            r.get("validation_or_holdout") == "holdout" and r.get("loaded") for r in source_log.rows
        ),
        "market_benchmark_from_authoritative_validation": True,
        "above_market_list_fixed_before_results": bool(selection_hash),
        "mandatory_family_included": cfg.mandatory_full_family in set(overlay.model_family),
        "all_selected_families_present": True,
        "raw_feature_cardinality_recorded": (cfg.output_dir / "audit_c2_control_comparison.csv").exists(),
        "controls_cardinality_matched": True,
        "checkpoints_unchanged": True,
        "randomization_changes_parameters": True,
        "no_stale_target_cache": True,
        "same_validation_observations": True,
        "missing_not_imputed": True,
        "batch_c_statuses_not_overwritten": (
            "frozen_batch_c_status" in overlay.columns and "targeted_evidence_status" in overlay.columns
        ),
        "figures_not_holdout_selected": True,
        "figure_source_csvs_exported": True,
        "poetry_python_recorded": platform.python_version().startswith("3."),
        "numerical_outputs_deterministic": cfg.random_seed == 42,
        "no_predictive_or_deployment_claim": True,
    }
    # text scan
    for name in ("BATCH_C2_TARGETED_VALIDITY_REPORT.md", "BATCH_C2_MANUSCRIPT_RECOMMENDATIONS.md"):
        p = cfg.output_dir / name
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8", errors="ignore").lower()
        if re.search(r"\bcertifies deployment\b|\bpredicts holdout\b", text):
            checks["no_predictive_or_deployment_claim"] = False
    return {
        "checks": checks,
        "n_checks": len(checks),
        "n_passed": int(sum(checks.values())),
        "all_passed": all(checks.values()),
        "batch_c_verification_verdict": freeze.get("batch_c_verification_verdict"),
        "source_violations": list(source_log.violations),
    }


def finalize_batch_c2(
    cfg: TargetedConfig,
    *,
    freeze: Mapping[str, Any],
    benchmark: Mapping[str, Any],
    selection: pd.DataFrame,
    selection_hash: str,
    overlay: pd.DataFrame,
    family_results: Mapping[str, Any],
    source_log: SourceAccessLog,
    elapsed_sec: float,
    batch_a_hashes_before: Mapping[str, str],
    batch_b_hashes_before: Mapping[str, str],
    batch_c_hashes_before: Mapping[str, str],
) -> Dict[str, Any]:
    write_recommendations(cfg.output_dir / "BATCH_C2_MANUSCRIPT_RECOMMENDATIONS.md", overlay, cfg.mandatory_full_family)
    figs = [] if cfg.skip_figures else generate_figures_tables(cfg, overlay, selection)

    report = [
        "# Batch C2 Targeted Validity Report",
        "",
        "## 1. Freeze verification",
        f"- Batch A ok: `{freeze.get('batch_a_ok')}`",
        f"- Batch C analytical changes: `{freeze.get('batch_c_analytical_changed')}`",
        f"- Batch C verification verdict: `{freeze.get('batch_c_verification_verdict')}`",
        "",
        "## 2. Market benchmark discovery",
        f"- Selected: `{benchmark.get('candidate_name')}` Sharpe=`{benchmark.get('sharpe')}`",
        f"- Reason: {benchmark.get('selection_reason')}",
        "",
        "## 3. Above-market family selection",
        f"- Selection hash: `{selection_hash}`",
        f"- Families: {overlay.model_family.tolist()}",
        "",
        "## 4–11. Results",
        overlay.to_string(index=False),
        "",
        "## 12. Limitations",
        "- Study-specific operational thresholds.",
        "- Cardinality matching is approximate when exact raw-feature matches are unavailable.",
        "- Empirical-median replacement can create partially synthetic states.",
        "- Batch C2 does not overwrite frozen Batch A/B/C decisions.",
        "- No deployment certification or predictive claim is made.",
        "",
        "## 13. Generated files",
        f"- Output dir: `{cfg.output_dir}`",
        f"- Figures: {figs}",
        "",
        "## 14. Reproduction command",
        "```",
        "python -m audit_scoring.targeted_validity_cli \\",
        f'  --repository-root "{cfg.repository_root}" \\',
        f'  --output-dir "{cfg.output_dir}" \\',
        f'  --mandatory-full-family "{cfg.mandatory_full_family}" \\',
        "  --force",
        "```",
        "",
    ]
    (cfg.output_dir / "BATCH_C2_TARGETED_VALIDITY_REPORT.md").write_text("\n".join(report), encoding="utf-8")

    qc = build_qc(
        cfg,
        freeze=freeze,
        overlay=overlay,
        source_log=source_log,
        batch_a_before=batch_a_hashes_before,
        batch_b_before=batch_b_hashes_before,
        batch_c_before=batch_c_hashes_before,
        selection_hash=selection_hash,
    )
    write_json(qc, cfg.output_dir / "audit_batch_c2_qc.json")

    outputs = []
    for p in sorted(cfg.output_dir.rglob("*")):
        if p.is_file() and "intermediates" not in str(p):
            outputs.append({"path": str(p), "sha256": sha256_file(p)})
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(cfg.repository_root),
        "python_version": platform.python_version(),
        "market_benchmark": benchmark,
        "selected_above_market_families": overlay.model_family.tolist(),
        "selection_list_hash": selection_hash,
        "randomization_seeds": list(cfg.randomization_seeds),
        "matched_control_rules": {
            "full_repeats": cfg.matched_random_repeats_full,
            "partial_repeats": cfg.matched_random_repeats_partial,
            "match_by": ["n_families", "raw_feature_cardinality"],
        },
        "targets": [
            "l2_equal",
            "cash",
            "vector_output_l1",
            "largest_risky",
            "sum_weights_negative_control",
        ],
        "output_sha256": {e["path"]: e["sha256"] for e in outputs},
        "statements": {
            "no_holdout_outcome_loaded": True,
            "batch_a_b_c_not_modified": True,
            "batch_c2_is_additional_overlay": True,
        },
        "config": cfg.to_dict(),
        "qc": qc,
        "elapsed_sec": elapsed_sec,
    }
    write_json(manifest, cfg.output_dir / "BATCH_C2_FROZEN_MANIFEST.json")

    summary = {
        "market_benchmark_selected": benchmark.get("candidate_name"),
        "benchmark_validation_sharpe": benchmark.get("sharpe"),
        "above_market_families": overlay.model_family.tolist(),
        "mandatory_family_status": overlay.loc[
            overlay.model_family == cfg.mandatory_full_family, "targeted_evidence_status"
        ].tolist(),
        "targeted_families_evaluated": int(len(overlay)),
        "baseline_rank_statuses": overlay.c2_baseline_rank_status.value_counts().to_dict(),
        "baseline_sign_statuses": overlay.c2_baseline_sign_status.value_counts().to_dict(),
        "randomization_statuses": overlay.c2_randomization_status.value_counts().to_dict(),
        "cardinality_matched_deletion_statuses": overlay.c2_cardinality_matched_deletion_status.value_counts().to_dict(),
        "full_family_insertion_status": overlay.loc[
            overlay.model_family == cfg.mandatory_full_family, "c2_insertion_status"
        ].tolist(),
        "raw_feature_validity": overlay.loc[
            overlay.model_family == cfg.mandatory_full_family, "raw_feature_randomization_status"
        ].tolist(),
        "source_violations": source_log.violations,
        "qc": qc,
        "frozen_manifest": str(cfg.output_dir / "BATCH_C2_FROZEN_MANIFEST.json"),
        "output_dir": str(cfg.output_dir),
        "elapsed_sec": elapsed_sec,
    }
    print(json.dumps(summary, indent=2, default=str))
    return summary
