"""CLI for Batch C2 targeted diagnostic-validity extension."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .targeted_validity_pipeline import TargetedConfig, run_batch_c2


def build_parser() -> argparse.ArgumentParser:
    d = TargetedConfig()
    p = argparse.ArgumentParser(
        prog="poetry run python -m src.audit_scoring.targeted_validity_cli",
        description="Batch C2 targeted validity overlay. Does not modify Batch A/B/C artefacts.",
    )
    p.add_argument("--repository-root", type=Path, default=d.repository_root)
    p.add_argument("--diagnostic-pkl-root", type=Path, default=d.diagnostic_pkl_root)
    p.add_argument("--batch-a-dir", type=Path, default=d.batch_a_dir)
    p.add_argument("--batch-b-dir", type=Path, default=d.batch_b_dir)
    p.add_argument("--batch-c-dir", type=Path, default=d.batch_c_dir)
    p.add_argument("--batch-c-verification-dir", type=Path, default=d.batch_c_verification_dir)
    p.add_argument("--metrics-dir", type=Path, default=d.metrics_dir)
    p.add_argument("--output-dir", type=Path, default=d.output_dir)
    p.add_argument("--family-mapping-report", type=Path, default=d.family_mapping_report)
    p.add_argument("--model-code-root", type=Path, default=d.model_code_root)
    p.add_argument("--mandatory-full-family", type=str, default=d.mandatory_full_family)
    p.add_argument("--primary-cost-bps", type=int, default=d.primary_cost_bps)
    p.add_argument("--validation-start", type=str, default="2016-01-01")
    p.add_argument("--validation-end", type=str, default="2024-04-30")
    p.add_argument("--primary-samples", type=int, default=d.primary_samples)
    p.add_argument("--randomization-samples", type=int, default=d.randomization_samples)
    p.add_argument("--background-samples", type=int, default=d.background_samples)
    p.add_argument("--randomization-seeds", type=int, nargs="+", default=list(d.randomization_seeds))
    p.add_argument("--matched-random-repeats-full", type=int, default=d.matched_random_repeats_full)
    p.add_argument("--matched-random-repeats-partial", type=int, default=d.matched_random_repeats_partial)
    p.add_argument("--bootstrap-replicates", type=int, default=d.bootstrap_replicates)
    p.add_argument("--random-seed", type=int, default=d.random_seed)
    p.add_argument("--ig-steps", type=int, default=d.ig_steps)
    p.add_argument("--device", type=str, default=d.device)
    p.add_argument("--families", type=str, nargs="*", default=None)
    p.add_argument("--discover-benchmark-only", action="store_true")
    p.add_argument("--inspect-only", action="store_true")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--skip-figures", action="store_true")
    p.add_argument("--force", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = TargetedConfig(
        repository_root=Path(args.repository_root),
        diagnostic_pkl_root=Path(args.diagnostic_pkl_root),
        batch_a_dir=Path(args.batch_a_dir),
        batch_b_dir=Path(args.batch_b_dir),
        batch_c_dir=Path(args.batch_c_dir),
        batch_c_verification_dir=Path(args.batch_c_verification_dir),
        metrics_dir=Path(args.metrics_dir),
        output_dir=Path(args.output_dir),
        family_mapping_report=Path(args.family_mapping_report),
        model_code_root=Path(args.model_code_root),
        mandatory_full_family=str(args.mandatory_full_family),
        primary_cost_bps=int(args.primary_cost_bps),
        validation_start=pd.Timestamp(args.validation_start),
        validation_end=pd.Timestamp(args.validation_end),
        primary_samples=int(args.primary_samples),
        randomization_samples=int(args.randomization_samples),
        background_samples=int(args.background_samples),
        randomization_seeds=tuple(int(x) for x in args.randomization_seeds),
        matched_random_repeats_full=int(args.matched_random_repeats_full),
        matched_random_repeats_partial=int(args.matched_random_repeats_partial),
        bootstrap_replicates=int(args.bootstrap_replicates),
        random_seed=int(args.random_seed),
        ig_steps=int(args.ig_steps),
        device=str(args.device),
        families=tuple(args.families) if args.families else (),
        discover_benchmark_only=bool(args.discover_benchmark_only),
        inspect_only=bool(args.inspect_only),
        resume=not bool(args.no_resume),
        skip_figures=bool(args.skip_figures),
        force=bool(args.force),
    )
    result = run_batch_c2(cfg)
    print("Batch C2 finished.")
    if isinstance(result, dict):
        for k in (
            "market_benchmark_selected",
            "benchmark_validation_sharpe",
            "above_market_families",
            "mandatory_family_status",
            "frozen_manifest",
            "output_dir",
        ):
            if k in result:
                print(f"{k}: {result[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
