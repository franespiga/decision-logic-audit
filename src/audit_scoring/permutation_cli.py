"""CLI for Batch C3 input-permutation validity overlay."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .permutation_config import PermutationConfig
from .permutation_pipeline import run_batch_c3


def build_parser() -> argparse.ArgumentParser:
    d = PermutationConfig()
    p = argparse.ArgumentParser(
        prog="poetry run python -m src.audit_scoring.permutation_cli",
        description="Batch C3 input-permutation validity overlay. Does not modify Batches A/B/C/C2.",
    )
    p.add_argument("--repository-root", type=Path, default=d.repository_root)
    p.add_argument("--diagnostic-pkl-root", type=Path, default=d.diagnostic_pkl_root)
    p.add_argument("--batch-a-dir", type=Path, default=d.batch_a_dir)
    p.add_argument("--batch-b-dir", type=Path, default=d.batch_b_dir)
    p.add_argument("--batch-c-dir", type=Path, default=d.batch_c_dir)
    p.add_argument("--batch-c-verification-dir", type=Path, default=d.batch_c_verification_dir)
    p.add_argument("--batch-c2-dir", type=Path, default=d.batch_c2_dir)
    p.add_argument("--batch-c2-verification-dir", type=Path, default=d.batch_c2_verification_dir)
    p.add_argument("--family-mapping-report", type=Path, default=d.family_mapping_report)
    p.add_argument("--model-code-root", type=Path, default=d.model_code_root)
    p.add_argument("--output-dir", type=Path, default=d.output_dir)
    p.add_argument("--validation-start", type=str, default="2016-01-01")
    p.add_argument("--validation-end", type=str, default="2024-04-30")
    p.add_argument("--primary-samples", type=int, default=d.primary_samples)
    p.add_argument("--permutation-seeds", type=int, nargs="+", default=list(d.permutation_seeds))
    p.add_argument("--random-control-repeats", type=int, default=d.random_control_repeats)
    p.add_argument("--max-depth", type=int, default=d.max_depth)
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
    p.add_argument("--skip-attribution-recompute", action="store_true")
    p.add_argument("--force", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = PermutationConfig(
        repository_root=Path(args.repository_root),
        diagnostic_pkl_root=Path(args.diagnostic_pkl_root),
        batch_a_dir=Path(args.batch_a_dir),
        batch_b_dir=Path(args.batch_b_dir),
        batch_c_dir=Path(args.batch_c_dir),
        batch_c_verification_dir=Path(args.batch_c_verification_dir),
        batch_c2_dir=Path(args.batch_c2_dir),
        batch_c2_verification_dir=Path(args.batch_c2_verification_dir),
        family_mapping_report=Path(args.family_mapping_report),
        model_code_root=Path(args.model_code_root),
        output_dir=Path(args.output_dir),
        validation_start=pd.Timestamp(args.validation_start),
        validation_end=pd.Timestamp(args.validation_end),
        primary_samples=int(args.primary_samples),
        permutation_seeds=tuple(int(x) for x in args.permutation_seeds),
        random_control_repeats=int(args.random_control_repeats),
        max_depth=int(args.max_depth),
        bootstrap_replicates=int(args.bootstrap_replicates),
        random_seed=int(args.random_seed),
        ig_steps=int(args.ig_steps),
        attribution_batch_size=int(args.attribution_batch_size),
        device=str(args.device),
        families=tuple(args.families) if args.families else (),
        inspect_only=bool(args.inspect_only),
        resume=not bool(args.no_resume),
        skip_figures=bool(args.skip_figures),
        skip_attribution_recompute=bool(args.skip_attribution_recompute),
        force=bool(args.force),
    )
    summary = run_batch_c3(cfg)
    print("=== Batch C3 complete ===")
    for k, v in summary.items():
        if k in {"qc", "priors"}:
            continue
        print(f"{k}: {v}")
    if isinstance(summary.get("qc"), dict):
        print(f"qc_all_pass: {summary['qc'].get('all_pass')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
