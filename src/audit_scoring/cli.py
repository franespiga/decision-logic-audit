"""Command-line interface for Batch A audit scoring."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import AuditConfig, DEFAULT_CONFIG, PRIMARY_COST_BPS
from .pipeline import inspect_inventory, run_batch_a
from .io import read_table


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.audit_scoring.cli",
        description=(
            "Batch A validation-calibrated contribution–necessity–structure "
            "audit scoring. Does not load holdout metrics and does not claim "
            "predictive or deployment-ready status."
        ),
    )
    p.add_argument(
        "--diagnostic-pkl-root",
        type=Path,
        default=DEFAULT_CONFIG.diagnostic_pkl_root,
    )
    p.add_argument("--metrics-dir", type=Path, default=DEFAULT_CONFIG.metrics_dir)
    p.add_argument(
        "--model-assessment-pickle",
        type=Path,
        default=DEFAULT_CONFIG.model_assessment_pickle,
    )
    p.add_argument("--output-dir", type=Path, default=DEFAULT_CONFIG.output_dir)
    p.add_argument("--primary-cost-bps", type=int, default=PRIMARY_COST_BPS)
    p.add_argument("--audit-profile", type=str, default="balanced")
    p.add_argument("--bootstrap-replicates", type=int, default=100)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--coverage-ratio-threshold", type=float, default=0.80)
    p.add_argument("--inspect-only", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--load-model-assessment", action="store_true")
    p.add_argument("--model-id", type=str, default=None)
    p.add_argument("--model-family", type=str, default=None)
    return p


def _print_model_slice(output_dir: Path, model_id: str) -> None:
    raw = read_table(output_dir / "audit_model_diagnostics_raw.csv")
    scored = read_table(output_dir / "audit_model_diagnostics_scored.csv")
    sub_raw = raw[raw["model_id"].astype(str) == model_id]
    sub_scored = scored[scored["model_id"].astype(str) == model_id]
    print(json.dumps({"raw": sub_raw.to_dict(orient="records"), "scored": sub_scored.to_dict(orient="records")}, indent=2, default=str))


def _print_family_slice(output_dir: Path, model_family: str, primary_cost_bps: int) -> None:
    fam = read_table(output_dir / "audit_family_diagnostics_scored.csv")
    primary = read_table(output_dir / f"audit_validation_decisions_primary_{primary_cost_bps}bps.csv")
    sub_f = fam[fam["model_family"].astype(str) == model_family]
    sub_p = primary[primary["model_family"].astype(str) == model_family]
    print(json.dumps({"family_scored": sub_f.to_dict(orient="records"), "primary_decision": sub_p.to_dict(orient="records")}, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = AuditConfig(
        diagnostic_pkl_root=args.diagnostic_pkl_root,
        metrics_dir=args.metrics_dir,
        model_assessment_pickle=args.model_assessment_pickle,
        output_dir=args.output_dir,
        primary_cost_bps=args.primary_cost_bps,
        audit_profile=args.audit_profile,
        bootstrap_replicates=args.bootstrap_replicates,
        random_seed=args.random_seed,
        coverage_ratio_threshold=args.coverage_ratio_threshold,
        force=args.force,
        inspect_only=args.inspect_only,
        load_model_assessment=args.load_model_assessment,
        model_id=args.model_id,
        model_family=args.model_family,
    )

    if args.model_id and (config.output_dir / "audit_model_diagnostics_scored.csv").exists() and not args.force and args.inspect_only is False:
        # Inspect frozen outputs for one model without recomputing unless forced full run
        if not (config.output_dir / "audit_validation_thresholds.json").exists():
            pass
        else:
            _print_model_slice(config.output_dir, args.model_id)
            if args.model_family:
                _print_family_slice(config.output_dir, args.model_family, args.primary_cost_bps)
            return 0

    if args.inspect_only:
        result = inspect_inventory(config)
        print(
            json.dumps(
                {
                    "pkls": int(len(result["inventory"])),
                    "resolved": int((result["mapping"]["mapping_status"] == "resolved").sum()),
                    "unresolved": int((result["mapping"]["mapping_status"] == "unresolved").sum()),
                    "duplicate": int((result["mapping"]["mapping_status"] == "duplicate").sum()),
                    "expected_families": result["expected_families"],
                },
                indent=2,
            )
        )
        return 0

    summary = run_batch_a(config)
    print("\n=== Batch A summary ===")
    print(f"PKLs discovered: {summary['pkls_discovered']}")
    print(f"Models resolved: {summary['models_resolved']}")
    print(f"Models scored: {summary['models_scored']}")
    print(f"Unresolved/ambiguous: {summary['unresolved_or_ambiguous']}")
    print(f"Families represented: {summary['families_represented']}")
    print(f"Primary cost level: {summary['primary_cost_bps']} bps")
    qc = summary.get("qc", {})
    print(f"Core metrics: {qc.get('core_metrics')}")
    print(f"Families by diagnostic band: {qc.get('families_by_diagnostic_band')}")
    print(f"Families by performance band: {qc.get('families_by_performance_band')}")
    print(f"Families by recommendation: {qc.get('families_by_recommendation')}")
    print(f"Provisional decisions: {qc.get('provisional_decisions')}")
    print(f"Warnings: {qc.get('warnings')}")
    print(f"Output dir: {summary['output_dir']}")
    print(f"Notebook path: {config.repository_root / 'notebooks' / '2_audit_scoring.ipynb'}")
    print(f"Package path: {config.repository_root / 'src' / 'audit_scoring'}")

    if args.model_id:
        _print_model_slice(config.output_dir, args.model_id)
    if args.model_family:
        _print_family_slice(config.output_dir, args.model_family, args.primary_cost_bps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
