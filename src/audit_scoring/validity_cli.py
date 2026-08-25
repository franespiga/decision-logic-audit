"""CLI for Batch C diagnostic-validity package."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .validity_config import DEFAULT_VALIDITY_CONFIG, ValidityConfig
from .validity_pipeline import run_batch_c, run_inspect_only


def build_parser() -> argparse.ArgumentParser:
    d = DEFAULT_VALIDITY_CONFIG
    p = argparse.ArgumentParser(
        prog="python -m audit_scoring.validity_cli",
        description=(
            "Batch C: diagnostic-validity overlay for frozen Batch A assessments. "
            "Does not modify Batch A or Batch B decisions."
        ),
    )
    p.add_argument("--repository-root", type=Path, default=d.repository_root)
    p.add_argument("--diagnostic-pkl-root", type=Path, default=d.diagnostic_pkl_root)
    p.add_argument("--batch-a-dir", type=Path, default=d.batch_a_dir)
    p.add_argument("--batch-b-dir", type=Path, default=d.batch_b_dir)
    p.add_argument("--batch-b-verification-dir", type=Path, default=d.batch_b_verification_dir)
    p.add_argument("--model-assessment-pickle", type=Path, default=d.model_assessment_pickle)
    p.add_argument("--family-mapping-report", type=Path, default=d.family_mapping_report)
    p.add_argument("--model-code-root", type=Path, default=d.model_code_root)
    p.add_argument("--output-dir", type=Path, default=d.output_dir)
    p.add_argument("--validation-start", type=str, default=str(d.validation_start.date()))
    p.add_argument("--validation-end", type=str, default=str(d.validation_end.date()))
    p.add_argument("--primary-samples", type=int, default=d.primary_samples)
    p.add_argument("--randomization-samples", type=int, default=d.randomization_samples)
    p.add_argument("--background-samples", type=int, default=d.background_samples)
    p.add_argument("--randomization-seeds", type=int, nargs="+", default=list(d.randomization_seeds))
    p.add_argument("--random-control-repeats", type=int, default=d.random_control_repeats)
    p.add_argument("--bootstrap-replicates", type=int, default=d.bootstrap_replicates)
    p.add_argument("--random-seed", type=int, default=d.random_seed)
    p.add_argument("--ig-steps", type=int, default=d.ig_steps)
    p.add_argument("--attribution-batch-size", type=int, default=d.attribution_batch_size)
    p.add_argument("--device", type=str, default=d.device)
    p.add_argument("--families", type=str, nargs="*", default=None)
    p.add_argument("--inspect-only", action="store_true")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--skip-figures", action="store_true")
    p.add_argument("--force", action="store_true")
    return p


def config_from_args(args: argparse.Namespace) -> ValidityConfig:
    families = tuple(args.families) if args.families else ()
    return ValidityConfig(
        repository_root=Path(args.repository_root),
        diagnostic_pkl_root=Path(args.diagnostic_pkl_root),
        batch_a_dir=Path(args.batch_a_dir),
        batch_b_dir=Path(args.batch_b_dir),
        batch_b_verification_dir=Path(args.batch_b_verification_dir),
        model_assessment_pickle=Path(args.model_assessment_pickle),
        family_mapping_report=Path(args.family_mapping_report),
        model_code_root=Path(args.model_code_root),
        output_dir=Path(args.output_dir),
        validation_start=pd.Timestamp(args.validation_start),
        validation_end=pd.Timestamp(args.validation_end),
        primary_samples=int(args.primary_samples),
        randomization_samples=int(args.randomization_samples),
        background_samples=int(args.background_samples),
        randomization_seeds=tuple(int(x) for x in args.randomization_seeds),
        random_control_repeats=int(args.random_control_repeats),
        bootstrap_replicates=int(args.bootstrap_replicates),
        random_seed=int(args.random_seed),
        ig_steps=int(args.ig_steps),
        attribution_batch_size=int(args.attribution_batch_size),
        device=str(args.device),
        families=families,
        inspect_only=bool(args.inspect_only),
        resume=not bool(args.no_resume),
        skip_figures=bool(args.skip_figures),
        force=bool(args.force),
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    if cfg.inspect_only:
        result = run_inspect_only(cfg)
    else:
        result = run_batch_c(cfg)
    print("Batch C finished.")
    if isinstance(result, dict):
        for key in (
            "batch_a_freeze_ok",
            "checkpoints_discovered",
            "checkpoints_evaluated",
            "incomplete_families",
            "status_counts",
            "source_access_violations",
            "frozen_manifest",
            "output_dir",
        ):
            if key in result:
                print(f"{key}: {result[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
