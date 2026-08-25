"""CLI for Batch B temporally separated holdout assessment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .holdout_reporting import run_batch_b


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.audit_scoring.holdout_cli",
        description=(
            "Batch B: temporally separated assessment of frozen Batch A "
            "provisional classifications. Does not recalibrate Batch A."
        ),
    )
    p.add_argument(
        "--batch-a-dir",
        type=Path,
        default=Path(r"C:\git\tesis\DT_ABLATION_RISK\results\mendeley\dt_metrics\audit_screening"),
    )
    p.add_argument(
        "--batch-a-verification-dir",
        type=Path,
        default=Path(
            r"C:\git\tesis\DT_ABLATION_RISK\results\mendeley\dt_metrics\audit_screening_verification"
        ),
    )
    p.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path(r"C:\git\tesis\DT_ABLATION_RISK\results\mendeley\dt_metrics"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"C:\git\tesis\DT_ABLATION_RISK\results\mendeley\dt_metrics\audit_holdout"),
    )
    p.add_argument(
        "--repo-root",
        type=Path,
        default=Path(r"C:\git\tesis\DT_ABLATION_RISK"),
    )
    p.add_argument("--primary-cost-bps", type=int, default=5)
    p.add_argument("--expected-variants-per-family", type=int, default=96)
    p.add_argument("--bootstrap-replicates", type=int, default=10000)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--auto-select-neighbour-pairs", type=int, default=3)
    p.add_argument("--confirm-family-checkpoint-mode", action="store_true")
    p.add_argument("--confirm-provisional-decisions", action="store_true")
    p.add_argument("--verify-freeze-only", action="store_true")
    p.add_argument("--skip-figures", action="store_true")
    p.add_argument("--force", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.primary_cost_bps != 5:
        raise SystemExit("Primary cost must remain 5 bps; Batch B does not retune the primary cost.")
    summary = run_batch_b(
        batch_a_dir=args.batch_a_dir,
        metrics_dir=args.metrics_dir,
        output_dir=args.output_dir,
        repo_root=args.repo_root,
        primary_cost_bps=args.primary_cost_bps,
        expected_variants_per_family=args.expected_variants_per_family,
        bootstrap_replicates=args.bootstrap_replicates,
        random_seed=args.random_seed,
        n_neighbour_pairs=args.auto_select_neighbour_pairs,
        verify_freeze_only=args.verify_freeze_only,
        skip_figures=args.skip_figures,
        force=args.force,
        confirm_family_checkpoint_mode=args.confirm_family_checkpoint_mode
        or args.verify_freeze_only,
        confirm_provisional_decisions=args.confirm_provisional_decisions
        or args.verify_freeze_only,
    )
    print("\n=== Batch B summary ===")
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
