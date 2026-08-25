"""
Cross-architecture diagnostic comparison: Decision Transformers versus conventional
deep reinforcement-learning policies.

One command reproduces the whole experiment: model discovery and manifest
resolution, the common regime-stratified evaluation sample, the contribution,
necessity and structure diagnostics, portfolio outcomes, the score construction,
and every table, figure and LaTeX drop-in.

Example
-------
python scripts/run_cross_architecture_comparison.py \
    --dt-model-dir models/comparison/dt \
    --rl-model-dir models/comparison/rl \
    --config config/cross_architecture_comparison.yaml \
    --output-dir results/cross_architecture \
    --period eval --seed 42 --n-attribution-samples 256 --n-bootstrap 1000 \
    --device auto
"""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import platform
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ablation.cross_architecture import (  # noqa: E402
    audit,
    contribution,
    data as market,
    discovery,
    manuscript,
    necessity,
    outcomes,
    reporting,
    sampling,
    scoring,
    structure,
    targets,
)
from ablation.cross_architecture import SCHEMA_VERSION  # noqa: E402

DEFAULT_DATA = "models/comparison/dt/DT_16_10/data.h5"


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Cross-architecture diagnostic comparison (DT versus deep RL).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dt-model-dir", default="models/comparison/dt")
    p.add_argument("--rl-model-dir", default="models/comparison/rl")
    p.add_argument("--config", default="config/cross_architecture_comparison.yaml")
    p.add_argument("--output-dir", default="results/cross_architecture")
    p.add_argument("--data-path", default=DEFAULT_DATA, help="frozen market data (HDF5)")
    p.add_argument("--period", default="eval", choices=sorted(sampling.PERIODS))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-attribution-samples", type=int, default=256)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument(
        "--n-stability-bootstrap",
        type=int,
        default=200,
        help="resamples for the rank- and profile-stability estimates",
    )
    p.add_argument("--n-random-deletions", type=int, default=20)
    p.add_argument("--ig-steps", type=int, default=32)
    p.add_argument("--occlusion-chunk", type=int, default=16)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument(
        "--resume",
        action="store_true",
        help="reuse cached per-model diagnostics from a previous run of the same configuration",
    )
    p.add_argument(
        "--models",
        default=None,
        help="comma-separated model_id subset (debugging only; the reported run uses all)",
    )
    p.add_argument("--pilot", action="store_true", help="debug-only fast pass; never report")
    p.add_argument(
        "--skip-tests",
        dest="run_tests",
        action="store_false",
        help="do not run the test suite against the generated artefacts",
    )
    return p


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def gpu_note(device: torch.device) -> str:
    """Live and peak device allocation, then reset the peak counter."""
    if device.type != "cuda":
        return ""
    live = torch.cuda.memory_allocated() / 2**20
    peak = torch.cuda.max_memory_allocated() / 2**20
    torch.cuda.reset_peak_memory_stats()
    return f"  [gpu {live:.0f} MiB live, {peak:.0f} MiB peak]"


def release(device: torch.device) -> None:
    """Drop unreferenced device tensors between models."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Stage 1: discovery, loading and inclusion
# ---------------------------------------------------------------------------


def stage_discovery(args, md: market.MarketData, device: torch.device):
    records = discovery.discover_models(Path(args.dt_model_dir), Path(args.rl_model_dir))
    log(f"discovered {len(records)} candidate checkpoints")

    period = sampling.period_dates(md.merged.index, args.period)
    adapters: Dict[str, targets.PolicyAdapter] = {}
    load_rows: List[Dict[str, object]] = []
    coverage_rows: List[Dict[str, object]] = []

    for record in records:
        error: Optional[str] = None
        try:
            discovery.resolve_record(
                record,
                indicator_columns=md.indicator_columns,
                price_columns=md.price_columns,
                default_data_path=str(md.path),
            )
        except Exception as exc:  # noqa: BLE001
            error = f"metadata resolution failed: {type(exc).__name__}: {exc}"

        adapter = None
        coverage = None
        if error is None:
            try:
                adapter = targets.build_adapter(record, device)
                frame = adapter.observation_frame(md.merged)
                usable = adapter.buildable_dates(frame, period)
                coverage = len(usable) / max(len(period), 1)
                coverage_rows.append(
                    {
                        "model_id": record.model_id,
                        "family": record.family,
                        "period": args.period,
                        "period_dates": len(period),
                        "buildable_dates": len(usable),
                        "coverage_fraction": coverage,
                        "first_usable": str(usable[0].date()) if usable else None,
                        "last_usable": str(usable[-1].date()) if usable else None,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"

        discovery.apply_inclusion_rules(
            record,
            loadable=adapter is not None,
            load_error=error,
            available_columns=md.columns,
            coverage_fraction=coverage,
        )
        if record.inclusion_status == "included" and adapter is not None:
            adapters[record.model_id] = adapter
        row = {
            "model_id": record.model_id,
            "family": record.family,
            "algorithm": record.algorithm,
            "checkpoint_path": record.checkpoint_path,
            "metadata_path": record.metadata_path,
            "data_path": record.data_path,
            "feature_extractor": record.feature_extractor,
            "lookback": record.lookback,
            "window_length": record.window_length,
            "n_state_features": record.n_features,
            "action_space": record.action_space,
            "n_actions": record.n_actions,
            "state_dim": record.state_dim,
            "reference_date": record.reference_date,
            "rebalancing_frequency": record.rebalancing_frequency,
            "loadable": adapter is not None,
            "load_error": error,
            "coverage_fraction": coverage,
            "inclusion_status": record.inclusion_status,
            "exclusion_reason": record.exclusion_reason,
        }
        load_rows.append(row)
        log(
            f"  {record.model_id:<28} {record.algorithm:<20} "
            f"{'included' if record.inclusion_status == 'included' else 'EXCLUDED'}"
            + ("" if error is None else f" ({error})")
        )

    return records, adapters, pd.DataFrame(load_rows), pd.DataFrame(coverage_rows), period


# ---------------------------------------------------------------------------
# Stage 2: rollouts, common dates and the diagnostic sample
# ---------------------------------------------------------------------------


def stage_sample(args, md: market.MarketData, adapters, period: pd.DatetimeIndex):
    rollouts: Dict[str, targets.Rollout] = {}
    frames: Dict[str, pd.DataFrame] = {}
    for model_id, adapter in adapters.items():
        frame = adapter.observation_frame(md.merged)
        frames[model_id] = frame
        usable = adapter.buildable_dates(frame, period)
        t0 = time.time()
        rollouts[model_id] = adapter.rollout(frame, usable)
        log(
            f"  rollout {model_id:<28} {len(rollouts[model_id]):>5} dates "
            f"({time.time() - t0:.1f}s)"
        )

    per_model_dates = {k: v.dates for k, v in rollouts.items()}
    common = sampling.common_dates(per_model_dates)
    log(f"common evaluation dates: {len(common)}")

    n_target = 32 if args.pilot else args.n_attribution_samples
    sample_dates, composition = sampling.stratified_sample(common, n_target, seed=args.seed)
    checks = sampling.validate_sample(sample_dates)
    log(f"diagnostic sample: {len(sample_dates)} dates, regimes {checks['regime_counts']}")

    sizes = sampling.sample_size_table(per_model_dates, sample_dates)
    return rollouts, frames, common, sample_dates, composition, checks, sizes


# ---------------------------------------------------------------------------
# Stage 3: per-model diagnostics
# ---------------------------------------------------------------------------


def compare_with_legacy(
    reconstructed: pd.DataFrame, legacy: Optional[pd.DataFrame]
) -> Dict[str, float]:
    """
    Compare a reconstructed allocation series with a legacy weight file.

    The comparison is repeated with the legacy index displaced by one trading day
    in each direction, because a legacy file may label a weight by its execution
    date rather than by the date of the observation window that produced it. This
    diagnoses an index offset without adopting one.
    """
    out: Dict[str, float] = {
        "legacy_weight_common_dates": 0,
        "legacy_weight_action_agreement": float("nan"),
        "legacy_weight_action_agreement_lead1": float("nan"),
        "legacy_weight_action_agreement_lag1": float("nan"),
        "legacy_weight_mean_l1": float("nan"),
        "legacy_weight_mean_l1_order_invariant": float("nan"),
    }
    if legacy is None:
        return out
    common = reconstructed.index.intersection(legacy.index).sort_values()
    if len(common) < 2:
        return out
    a = reconstructed.loc[common].values
    b = legacy.loc[common].values
    out["legacy_weight_common_dates"] = int(len(common))
    out["legacy_weight_mean_l1"] = float(np.abs(a - b).sum(axis=1).mean())
    # Comparing the sorted weight vectors separates two different questions: whether
    # the policy output itself is reproduced (order-invariant distance near zero) and
    # whether the instrument labelling agrees (labelled distance near zero).
    out["legacy_weight_mean_l1_order_invariant"] = float(
        np.abs(np.sort(a, axis=1) - np.sort(b, axis=1)).sum(axis=1).mean()
    )
    for shift, key in ((0, ""), (1, "_lead1"), (-1, "_lag1")):
        if shift == 0:
            aa, bb = a, b
        elif shift > 0:
            aa, bb = a[:-shift], b[shift:]
        else:
            aa, bb = a[-shift:], b[:shift]
        out[f"legacy_weight_action_agreement{key}"] = float(
            (np.argmax(aa, axis=1) == np.argmax(bb, axis=1)).mean()
        )
    return out


def diagnose_model(
    args,
    adapter: targets.PolicyAdapter,
    rollout: targets.Rollout,
    frame: pd.DataFrame,
    sample_dates: pd.DatetimeIndex,
    md: market.MarketData,
) -> Dict[str, object]:
    positions = [i for i, d in enumerate(rollout.dates) if d in set(sample_dates)]
    sample = rollout.select(positions)
    regimes = sampling.assign_regimes(sample.dates)

    train_frame = frame.loc[: pd.Timestamp(sampling.PERIODS["train"][1])]
    mean, std, _ = sampling.train_reference_statistics(
        train_frame, train_end=sampling.PERIODS["train"][1]
    )

    model_id = adapter.record.model_id
    stage_t0 = time.time()

    def stage_done(name: str) -> None:
        nonlocal stage_t0
        log(f"    {model_id} {name} {time.time() - stage_t0:.1f}s{gpu_note(adapter.device)}")
        stage_t0 = time.time()

    n_boot = 40 if args.pilot else args.n_stability_bootstrap
    contrib = contribution.run_contribution(
        adapter,
        sample,
        regimes,
        train_frame,
        n_bootstrap=n_boot,
        seed=args.seed,
        ig_steps=8 if args.pilot else args.ig_steps,
        coord_chunk=args.occlusion_chunk,
    )
    stage_done("contribution")

    # Replacement values for deletion and occlusion: the training-period mean, i.e.
    # a reference that carries no evaluation-period information.
    replacement = torch.from_numpy(
        np.broadcast_to(
            mean.astype(np.float32), (len(sample), adapter.window, adapter.n_features)
        ).copy()
    )
    variant = contrib["attribution_variant"]
    cube = contrib["primary_cube"]

    nec = necessity.run_necessity(
        adapter,
        sample,
        regimes,
        cube,
        replacement,
        variant,
        n_random=5 if args.pilot else args.n_random_deletions,
        n_bootstrap=n_boot,
        seed=args.seed,
    )
    stage_done("necessity")

    baseline = contribution.build_baselines(adapter, len(sample), train_frame, seed=args.seed)[
        contrib["reference_baseline"]
    ]
    gradients_usable = contrib["attribution_method"] == "integrated_gradients"
    feature_std = std.astype(np.float64)
    struct = structure.run_structure(
        adapter,
        sample,
        regimes,
        cube,
        replacement,
        baseline,
        variant,
        feature_std,
        gradients_usable,
        seed=args.seed,
        n_replicates=10 if args.pilot else 30,
    )
    stage_done("structure")

    out = outcomes.evaluate_outcomes(
        adapter.record.model_id, rollout.weights, md.prices, provenance="reconstructed"
    )
    simplex = outcomes.check_simplex(rollout.weights)
    legacy_path = adapter.record.legacy_weights_path
    legacy = outcomes.load_legacy_weights(legacy_path, rollout.dates)
    legacy_stats = compare_with_legacy(rollout.weights, legacy)
    legacy_stats["legacy_column_order"] = (
        ", ".join(pd.read_csv(legacy_path, index_col=0, nrows=1).columns)
        if legacy_path and Path(legacy_path).exists()
        else ""
    )

    metrics: Dict[str, object] = {}
    metrics.update(contrib["metrics"])
    metrics.update(nec["metrics"])
    metrics.update(struct["metrics"])
    metrics.update(out["metrics"])
    metrics.update(
        {
            "model_id": adapter.record.model_id,
            "family": adapter.record.family,
            "algorithm": adapter.record.algorithm,
            "action_space": adapter.record.action_space,
            "lookback": adapter.record.lookback,
            "observation_window_rows": adapter.window,
            "n_observation_features": adapter.n_features,
            "rebalancing_frequency": adapter.record.rebalancing_frequency,
            "attribution_method": contrib["attribution_method"],
            "attribution_variant": variant,
            "layer_influence_method": struct["layer_influence"]["method"],
            "latent_layer": adapter.latent_layer_name(),
            "n_diagnostic_samples": len(sample),
            "n_evaluation_dates": len(rollout),
            "weight_unique_rows": out["metrics"]["unique_allocations"],
            "grad_nonzero_fraction_primary_target": contrib["gradient_health"][
                "grad_nonzero_fraction"
            ],
            "grad_nonzero_fraction": contrib["gradient_health_used"]["grad_nonzero_fraction"],
            "decision_target_std": contrib["gradient_health"]["target_std"],
            **legacy_stats,
            **{f"simplex_{k}": v for k, v in simplex.items()},
        }
    )

    return {
        "contribution": contrib,
        "necessity": nec,
        "structure": struct,
        "outcomes": out,
        "metrics": metrics,
        "sample_dates": sample.dates,
        "regimes": regimes,
        "notes": contrib["notes"],
    }


# ---------------------------------------------------------------------------
# Stage 4: writing artefacts
# ---------------------------------------------------------------------------


def write_supplementary_data(out_dir: Path, model_id: str, res: Dict[str, object]) -> List[str]:
    """Per-model supplementary CSV files; returns the relative paths written."""
    supp = out_dir / "supplementary"
    written: List[str] = []

    def put(frame: pd.DataFrame, name: str) -> None:
        path = supp / name
        reporting.write_table(frame, path)
        written.append(str(path.relative_to(out_dir)))

    contrib = res["contribution"]
    shares = contrib["shares"]
    for axis, series in shares.items():
        put(
            series.rename("share").rename_axis(axis).reset_index(),
            f"attribution_share_{axis}__{model_id}.csv",
        )
    put(contrib["signed_features"], f"attribution_signed_features__{model_id}.csv")
    concord = contrib["baseline_concordance"]["pairwise"]
    if len(concord):
        put(concord, f"baseline_concordance__{model_id}.csv")
    boot = contrib["rank_stability_feature"]
    replicates = boot.get("replicate_scores")
    if replicates is not None:
        arr = np.asarray(replicates, dtype=np.float64)
        shares = arr / np.clip(arr.sum(axis=1, keepdims=True), 1e-12, None)
        put(
            pd.DataFrame(
                {
                    "feature_group": boot["groups"],
                    "mean_share": shares.mean(axis=0),
                    "std_share": shares.std(axis=0, ddof=1) if arr.shape[0] > 1 else 0.0,
                    "mean_rank": (-arr).argsort(axis=1).argsort(axis=1).mean(axis=0) + 1.0,
                }
            ),
            f"bootstrap_rank_stability__{model_id}.csv",
        )

    nec = res["necessity"]
    put(nec["deletion"]["curve"], f"deletion_curve__{model_id}.csv")
    put(nec["deletion"]["random"], f"deletion_random_repeats__{model_id}.csv")
    put(
        pd.DataFrame(
            {
                "lag": np.arange(len(nec["profile_normalised"])),
                "target_profile": nec["profile_normalised"],
                "weight_tv_profile": nec["profile_weight_normalised"],
            }
        ),
        f"temporal_occlusion_profile__{model_id}.csv",
    )
    put(
        pd.DataFrame(nec["regime_profiles"]).rename_axis("lag").reset_index(),
        f"temporal_occlusion_by_regime__{model_id}.csv",
    )
    put(
        pd.DataFrame(
            {"relative_lag": nec["relative_lag_grid"], "profile": nec["relative_lag_profile"]}
        ),
        f"temporal_occlusion_relative_lag__{model_id}.csv",
    )

    struct = res["structure"]
    put(struct["sensitivity"], f"latent_neighbourhood_stability__{model_id}.csv")
    put(struct["layer_influence"]["frame"], f"layer_influence__{model_id}.csv")

    out = res["outcomes"]
    put(out["cash_by_regime"], f"cash_allocation_by_regime__{model_id}.csv")
    put(out["returns"].rename_axis("date").reset_index(), f"portfolio_returns__{model_id}.csv")
    return written


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    started = datetime.now(timezone.utc)
    set_seeds(args.seed)
    device = resolve_device(args.device)

    out_dir = Path(args.output_dir)
    for sub in ("audit", "data", "tables", "figures", "manuscript", "supplementary", "cache"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    log(f"device={device}  seed={args.seed}  period={args.period}")
    if args.pilot:
        log("PILOT MODE: reduced sample and bootstrap counts; not a reportable run")

    md = market.load_market_data(args.data_path)
    log(f"market data {md.merged.shape} from {md.path}")

    records, adapters, load_report, coverage, period = stage_discovery(args, md, device)
    if args.models:
        keep = {m.strip() for m in args.models.split(",")}
        adapters = {k: v for k, v in adapters.items() if k in keep}
        log(f"restricted to {sorted(adapters)}")
    if not adapters:
        raise SystemExit("no model satisfied the inclusion criteria; see the audit report")

    families = {a.record.family for a in adapters.values()}
    if not {"DT", "RL"} <= families:
        log(f"WARNING: only families {sorted(families)} are included; the comparison is incomplete")

    # ---- manifest -------------------------------------------------------
    manifest_meta = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": started.isoformat(),
        "generator": "scripts/run_cross_architecture_comparison.py",
        "dt_model_dir": str(args.dt_model_dir),
        "rl_model_dir": str(args.rl_model_dir),
        "data_path": str(md.path),
        "data_md5_first_8mb": market.file_md5(Path(md.path)),
        "period": args.period,
        "period_range": list(sampling.PERIODS[args.period]),
        "seed": args.seed,
        "regime_configuration": {k: list(v) for k, v in sampling.REGIMES.items()},
        "transaction_cost_one_way": outcomes.TRANSACTION_COST,
    }
    discovery.write_manifest(Path(args.config), records, manifest_meta)
    log(f"wrote manifest {args.config}")

    registry = discovery.records_to_frame(records)
    reporting.write_table(registry, out_dir / "supplementary" / "model_registry.csv")
    excluded = registry.loc[
        registry["inclusion_status"] != "included",
        ["model_id", "family", "algorithm", "exclusion_reason"],
    ]
    reporting.write_table(excluded, out_dir / "supplementary" / "excluded_models.csv")

    # ---- decision targets ----------------------------------------------
    dt_table = targets.decision_target_table(adapters)
    reporting.write_table(dt_table, out_dir / "audit" / "decision_targets.csv")

    # ---- sample ---------------------------------------------------------
    rollouts, frames, common, sample_dates, composition, checks, sizes = stage_sample(
        args, md, adapters, period
    )
    reporting.write_table(
        pd.DataFrame(
            {
                "date": sample_dates,
                "regime": sampling.assign_regimes(sample_dates),
            }
        ),
        out_dir / "data" / "common_diagnostic_dates.csv",
    )
    reporting.write_table(
        pd.DataFrame({"date": common, "regime": sampling.assign_regimes(common)}),
        out_dir / "data" / "common_evaluation_dates.csv",
    )
    reporting.write_table(composition, out_dir / "tables" / "diagnostic_sample_composition.csv")
    reporting.write_table(sizes, out_dir / "tables" / "sample_sizes_by_model_regime.csv")

    # ---- diagnostics ----------------------------------------------------
    cache_key = (
        f"{args.period}_seed{args.seed}_n{len(sample_dates)}_ig{args.ig_steps}"
        f"_b{args.n_stability_bootstrap}_r{args.n_random_deletions}"
    )
    results: Dict[str, Dict[str, object]] = {}
    supp_files: List[str] = []
    for model_id, adapter in adapters.items():
        cache = out_dir / "cache" / f"{model_id}__{cache_key}.pkl"
        if args.resume and cache.exists():
            with cache.open("rb") as fh:
                results[model_id] = pickle.load(fh)
            log(f"  resumed {model_id} from cache")
        else:
            t0 = time.time()
            results[model_id] = diagnose_model(
                args, adapter, rollouts[model_id], frames[model_id], sample_dates, md
            )
            with cache.open("wb") as fh:
                pickle.dump(results[model_id], fh, protocol=pickle.HIGHEST_PROTOCOL)
            log(f"  diagnosed {model_id} in {time.time() - t0:.1f}s")
        supp_files.extend(write_supplementary_data(out_dir, model_id, results[model_id]))
        # Only the diagnostics are needed from here on. Releasing the rollout, the
        # observation frame and the policy weights keeps the device footprint at
        # one model at a time, which matters on an 8 GiB card.
        rollouts.pop(model_id, None)
        frames.pop(model_id, None)
        adapter.release()
        release(device)

    # ---- scores and comparisons ----------------------------------------
    per_model_metrics = {k: v["metrics"] for k, v in results.items()}
    scores = scoring.build_score_table(per_model_metrics)
    carried = set(scores.columns) - {"model_id"}
    meta_cols = pd.DataFrame(
        [{k: v for k, v in m.items() if k not in carried} for m in per_model_metrics.values()]
    )
    table = meta_cols.merge(scores, on="model_id", how="left")

    score_cols = [
        "contribution_score",
        "necessity_score",
        "structure_score",
        "aggregate_diagnostic_score",
    ]
    groups = {"all": table.index}
    for fam, grp in table.groupby("family"):
        groups[fam] = grp.index
    correlations = scoring.rank_correlations(table, score_cols, "gross_sharpe_ratio", groups)
    family = scoring.family_summary(table, score_cols + list(scoring.METRIC_SPEC))
    boots = scoring.bootstrap_table(
        table, score_cols, n_bootstrap=100 if args.pilot else args.n_bootstrap, seed=args.seed
    )
    matched = scoring.matched_comparison(table)

    summary = reporting.build_model_summary(table)
    reporting.write_table(summary, out_dir / "tables" / "cross_architecture_model_summary.csv")
    reporting.write_table(correlations, out_dir / "tables" / "score_performance_correlations.csv")
    reporting.write_table(family, out_dir / "tables" / "family_level_summary.csv")
    reporting.write_table(boots, out_dir / "tables" / "bootstrap_confidence_intervals.csv")
    reporting.write_table(matched, out_dir / "tables" / "matched_comparison.csv")
    reporting.write_table(scoring.metric_dictionary(), out_dir / "tables" / "metric_dictionary.csv")

    cash = pd.concat(
        [
            res["outcomes"]["cash_by_regime"].assign(model_id=mid)
            for mid, res in results.items()
        ],
        ignore_index=True,
    )
    reporting.write_table(cash, out_dir / "tables" / "cash_allocation_by_regime.csv")

    benchmark = outcomes.benchmark_outcomes(md.prices, common)
    reporting.write_table(
        pd.DataFrame([{"model_id": "EqualWeight_benchmark", **benchmark["metrics"]}]),
        out_dir / "tables" / "benchmark_outcomes.csv",
    )

    # ---- manuscript output ---------------------------------------------
    tex = out_dir / "manuscript" / "cross_architecture_summary.tex"
    tex.write_text(reporting.main_latex_table(table, sharpe_kind="gross"), encoding="utf-8")

    figures = []
    figures += reporting.figure_diagnostics_heatmap(
        table, out_dir / "figures" / "F_cross_architecture_diagnostics_heatmap"
    )
    figures += reporting.figure_functional_comparison(
        {k: v["necessity"] for k, v in results.items()},
        table,
        out_dir / "figures" / "F_cross_architecture_functional_comparison",
    )
    log(f"wrote {len(figures)} figure files")

    settings = {
        "schema_version": SCHEMA_VERSION,
        "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "seed": args.seed,
        "period": args.period,
        "period_range": " to ".join(sampling.PERIODS[args.period]),
        "n_common_evaluation_dates": int(len(common)),
        "n_common_sample_dates": int(len(sample_dates)),
        "n_attribution_samples_requested": 32 if args.pilot else args.n_attribution_samples,
        "n_stability_bootstrap": 40 if args.pilot else args.n_stability_bootstrap,
        "n_outcome_bootstrap": 100 if args.pilot else args.n_bootstrap,
        "n_random_deletions": 5 if args.pilot else args.n_random_deletions,
        "integrated_gradient_steps": 8 if args.pilot else args.ig_steps,
        "deletion_fractions": ", ".join(str(f) for f in necessity.DELETION_FRACTIONS),
        "knn_values": ", ".join(str(k) for k in structure.K_VALUES),
        "transaction_cost_one_way": outcomes.TRANSACTION_COST,
        "score_kind": scoring.SCORE_KIND,
        "pilot_mode": args.pilot,
    }

    supp_tex = reporting.supplementary_tex(
        summary=summary,
        sample_sizes=sizes,
        registry=registry,
        decision_targets=dt_table,
        metric_dict=scoring.metric_dictionary(),
        correlations=correlations,
        bootstrap=boots,
        cash_by_regime=cash,
        excluded=excluded,
        settings=settings,
        figure_names=[p.name for p in figures if p.suffix == ".pdf"],
        csv_index=sorted(set(supp_files)),
    )
    (out_dir / "supplementary" / "cross_architecture_supplement.tex").write_text(
        supp_tex, encoding="utf-8"
    )

    (out_dir / "manuscript" / "non_transformer_benchmark_methods.tex").write_text(
        manuscript.methods_tex(table, sizes, dt_table, settings), encoding="utf-8"
    )
    (out_dir / "manuscript" / "cross_architecture_results.tex").write_text(
        manuscript.results_tex(table, correlations, boots, family, matched, settings),
        encoding="utf-8",
    )
    (out_dir / "manuscript" / "cross_architecture_discussion.tex").write_text(
        manuscript.discussion_tex(table, settings), encoding="utf-8"
    )

    # ---- audit and missing inputs --------------------------------------
    reconstruction = pd.DataFrame(
        [
            {
                "model_id": mid,
                "weights_source": res["metrics"]["weights_source"],
                "n_dates": res["metrics"]["n_evaluation_dates"],
                "distinct_allocations": res["metrics"]["unique_allocations"],
                "min_weight": res["metrics"]["simplex_min_weight"],
                "max_sum_deviation": res["metrics"]["simplex_max_abs_sum_deviation"],
                "legacy_file": bool(_legacy_path(adapters, mid)),
                "legacy_common_dates": res["metrics"]["legacy_weight_common_dates"],
                "legacy_action_agreement": res["metrics"]["legacy_weight_action_agreement"],
                "legacy_agreement_offset_plus1": res["metrics"][
                    "legacy_weight_action_agreement_lead1"
                ],
                "legacy_agreement_offset_minus1": res["metrics"][
                    "legacy_weight_action_agreement_lag1"
                ],
                "legacy_mean_l1": res["metrics"]["legacy_weight_mean_l1"],
                "legacy_mean_l1_order_invariant": res["metrics"][
                    "legacy_weight_mean_l1_order_invariant"
                ],
                "legacy_column_order": res["metrics"]["legacy_column_order"],
                "action_index_order": ", ".join(adapters[mid].action_instruments),
                "decision_target_std": res["metrics"]["decision_target_std"],
                "grad_nonzero_primary_target": res["metrics"][
                    "grad_nonzero_fraction_primary_target"
                ],
                "grad_nonzero_target_used": res["metrics"]["grad_nonzero_fraction"],
            }
            for mid, res in results.items()
        ]
    )
    reporting.write_table(reconstruction, out_dir / "audit" / "reconstruction_checks.csv")

    audit_md = audit.render_audit(
        records=records,
        load_report=load_report,
        coverage=coverage,
        data_files=market.identical_data_files([r.data_path for r in records] + [md.path]),
        reconstruction=reconstruction,
        decision_targets=dt_table,
        common_dates_n=len(common),
        settings=settings,
    )
    (out_dir / "audit" / "implementation_audit.md").write_text(audit_md, encoding="utf-8")

    missing = build_missing_inputs(results, adapters)
    (out_dir / "missing_inputs.md").write_text(
        audit.render_missing_inputs(missing), encoding="utf-8"
    )

    # ---- run report -----------------------------------------------------
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    report = render_run_report(
        args=args,
        settings=settings,
        table=table,
        registry=registry,
        sizes=sizes,
        checks=checks,
        elapsed=elapsed,
        figures=figures,
        out_dir=out_dir,
    )
    (out_dir / "run_report.md").write_text(report, encoding="utf-8")

    # The suite checks the artefacts that were just written, so it runs last and its
    # outcome is appended to the report the reviewer reads.
    if args.run_tests:
        log("running the test suite against the generated artefacts")
        validation, summary = run_test_suite(out_dir)
        (out_dir / "run_report.md").write_text(
            report.rstrip("\n") + "\n\n" + validation, encoding="utf-8"
        )
        log(summary)

    (out_dir / "audit" / "run_settings.json").write_text(
        json.dumps(settings, indent=2, default=str), encoding="utf-8"
    )
    log(f"done in {elapsed:.1f}s; artefacts under {out_dir}")
    return 0


def _legacy_path(adapters, model_id: str) -> Optional[str]:
    ad = adapters.get(model_id)
    return ad.record.legacy_weights_path if ad else None


def build_missing_inputs(results, adapters) -> List[Dict[str, str]]:
    """Report only inputs that are genuinely absent, with their exact schema."""
    items: List[Dict[str, str]] = []

    disagreeing = [
        mid
        for mid, res in results.items()
        if np.isfinite(res["metrics"]["legacy_weight_mean_l1"])
        and res["metrics"]["legacy_weight_mean_l1"] > 1e-6
    ]
    if disagreeing:
        items.append(
            {
                "title": "Recorded observation-column order and action-index order for the "
                "Stable-Baselines3 training runs",
                "affected_models": ", ".join(sorted(disagreeing)),
                "exact_missing_file": "an `env_state.json` (or equivalent field inside "
                "`metadata.json`) written by the SB3 training script, next to each "
                "`best_model.zip`",
                "required_schema": "JSON object with keys `observation_columns` (ordered array of "
                "`[instrument, field]` pairs, one per observation column) and `action_instruments` "
                "(ordered array of instrument names, one per action index, cash last)",
                "expected_columns": "77 observation columns for the included runs (42 technical "
                "indicators plus 35 price/volume columns) and 7 action instruments",
                "expected_date_range": "not applicable: this is a configuration artefact, not a "
                "time series",
                "expected_frequency": "not applicable",
                "expected_weight_convention": "long-only, fully invested, rows summing to one, "
                "cash as the final column",
                "why_necessary": "the observation ordering and the action-index ordering are "
                "currently reconstructed from the frozen data file and from "
                "`metadata.json:instruments`. The reconstruction reproduces the allocation values "
                "of the continuous policy and the greedy action of the value-based policy, but "
                "the legacy weight files are labelled in the instrument order of the data file "
                "rather than in the order declared by metadata, so for the policies listed above "
                "the instrument that each action index denotes cannot be confirmed from the "
                "repository alone. A recorded ordering would remove that residual ambiguity. It "
                "affects the labelling of per-instrument attribution shares and of the "
                "reconstructed allocations, not the existence or the internal consistency of any "
                "diagnostic.",
                "current_workaround": "the ordering declared in each run's own metadata is used, "
                "was fixed before any diagnostic was computed, is documented in the audit, and is "
                "applied identically to every policy; all reported outcomes are reconstructed "
                "from the frozen policy rather than read from the legacy files.",
            }
        )

    stochastic = [
        mid
        for mid, res in results.items()
        if res["metrics"]["algorithm"] in {"PPO", "A2C"}
        and np.isfinite(res["metrics"]["legacy_weight_action_agreement"])
        and res["metrics"]["legacy_weight_action_agreement"] < 0.5
    ]
    if stochastic:
        items.append(
            {
                "title": "Action-selection convention used in the legacy backtest of the "
                "stochastic policies",
                "affected_models": ", ".join(sorted(stochastic)),
                "exact_missing_file": "the backtest configuration that produced "
                "`results/weights_BEST.csv` (for example a `backtest_config.json` recording the "
                "`deterministic` flag and the random seed passed to `model.predict`)",
                "required_schema": "JSON object with keys `deterministic` (boolean) and `seed` "
                "(integer)",
                "expected_columns": "not applicable",
                "expected_date_range": "not applicable",
                "expected_frequency": "not applicable",
                "expected_weight_convention": "not applicable",
                "why_necessary": "for a categorical policy the recorded allocations can only be "
                "reproduced exactly if it is known whether the legacy rollout sampled from the "
                "action distribution or took its mode. The disagreement pattern between the "
                "reconstruction and the legacy file is diffuse rather than a fixed relabelling, "
                "which is the signature of sampling.",
                "current_workaround": "this pipeline always takes the greedy action, which is the "
                "convention that makes the decision target well defined and identical across "
                "evaluations; it is applied to every discrete policy and is recorded in the "
                "decision-target table.",
            }
        )
    return items


def run_test_suite(out_dir: Path) -> Tuple[str, str]:
    """Run the test suite against the artefacts just written; return a report section.

    The suite is executed as a subprocess so that it sees the files on disk and
    starts from a clean interpreter, and its full output is kept next to the audit.
    """
    cmd = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    elapsed = time.time() - t0
    text = proc.stdout + proc.stderr
    log_path = out_dir / "audit" / "test_results.txt"
    log_path.write_text(f"$ {' '.join(cmd)}\n\n{text}", encoding="utf-8")

    # Verbose unittest output prints "<test> (<module>.<Class>.<test>)" and then the
    # outcome, possibly after a docstring line, so the class is carried forward.
    per_class: Dict[str, Dict[str, int]] = {}
    current = "unknown"
    for line in text.splitlines():
        found = re.search(r"\((?:tests\.)?test_\w+\.(\w+)\.\w+\)", line)
        if found:
            current = found.group(1)
        outcome = re.search(r"\.\.\. (ok|FAIL|ERROR|skipped)", line)
        if outcome:
            bucket = per_class.setdefault(current, {"ok": 0, "failed": 0, "skipped": 0})
            status = outcome.group(1)
            key = "ok" if status == "ok" else "skipped" if status == "skipped" else "failed"
            bucket[key] += 1

    ran = re.search(r"^Ran (\d+) tests?", text, re.M)
    verdict = "passed" if proc.returncode == 0 else "FAILED"
    summary = (
        f"{ran.group(1) if ran else 'unknown number of'} tests, {verdict}, in {elapsed:.1f} s. "
        f"Full output: `audit/test_results.txt`."
    )
    lines = ["## Validation\n"]
    lines.append("```\n" + " ".join(["python", *cmd[1:]]) + "\n```\n")
    lines.append(summary + "\n")
    if per_class:
        frame = pd.DataFrame(
            [
                {"test group": name, "passed": v["ok"], "skipped": v["skipped"], "failed": v["failed"]}
                for name, v in sorted(per_class.items())
            ]
        )
        lines.append(reporting.markdown_table(frame) + "\n")
    if proc.returncode != 0:
        failures = re.findall(r"^(?:FAIL|ERROR): (.+)$", text, re.M)
        lines.append("Failing tests:\n")
        lines.extend(f"* `{f}`" for f in failures)
        lines.append("")
    return "\n".join(lines), summary


def render_run_report(
    args, settings, table, registry, sizes, checks, elapsed, figures, out_dir: Path
) -> str:
    inc = registry.loc[registry["inclusion_status"] == "included", "model_id"].tolist()
    exc = registry.loc[registry["inclusion_status"] != "included"]
    lines = ["# Run report: cross-architecture diagnostic comparison\n"]
    lines.append(f"Completed in {elapsed:.1f} s on {settings['device']}.\n")
    lines.append("## Command\n")
    lines.append("```\n" + str(settings["command"]) + "\n```\n")
    lines.append("## Models\n")
    lines.append(f"Included ({len(inc)}): {', '.join(inc)}\n")
    if len(exc):
        lines.append("Excluded:\n")
        lines.append(
            reporting.markdown_table(exc[["model_id", "exclusion_reason"]]) + "\n"
        )
    else:
        lines.append("Excluded: none.\n")
    lines.append("## Sample\n")
    lines.append(
        f"{checks['n']} common diagnostic dates; regime composition "
        f"{checks['regime_counts']}; Unknown labels: {checks['unknown']}.\n"
    )
    lines.append("## Results\n")
    keep = [
        c
        for c in [
            "model_id", "family", "algorithm", "attribution_method",
            "layer_influence_method", "contribution_score", "necessity_score",
            "structure_score", "aggregate_diagnostic_score", "gross_sharpe_ratio",
            "net_sharpe_ratio", "unique_allocations",
        ]
        if c in table.columns
    ]
    lines.append(reporting.markdown_table(table.loc[:, keep]) + "\n")
    lines.append("## Artefacts\n")
    supp_count = 0
    for path in sorted(out_dir.rglob("*")):
        if not path.is_file() or path.suffix == ".pkl":
            continue
        rel = path.relative_to(out_dir).as_posix()
        if rel.startswith("supplementary/") and path.suffix == ".csv":
            supp_count += 1
            continue
        lines.append(f"* `{rel}`")
    lines.append(f"* `supplementary/` additionally contains {supp_count} per-model CSV files")
    lines.append("")
    lines.append("## Settings\n")
    lines.append(
        reporting.markdown_table(
            pd.DataFrame([settings]).T.reset_index().rename(columns={"index": "setting", 0: "value"})
        )
        + "\n"
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
