"""
Headless runner for multifamily ablation over all RL models in the manifest.
"""

from __future__ import annotations

# Import torch before numpy/pandas on this Windows env (DLL init order)
import torch

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from ablation.multi_family import (  # noqa: E402
    DEFAULT_PERIODS,
    build_eval_state_frame,
    enrich_spec_from_metadata,
    evaluate_model_weights,
    feature_importance_multifamily,
    load_market_data,
    load_model,
    load_models_manifest,
    parse_models_dict,
    prepare_ablation_batch,
    register_rl_factory_shim,
    save_evaluation_outputs,
)
from ablation.postprocessing import postprocess_attributions  # noqa: E402
from src import compute_portfolio_returns  # noqa: E402


def _tupleize_env_state(env_state):
    out = []
    for item in env_state:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            out.append((item[0], item[1]))
        else:
            out.append(item)
    return out


def run_one(
    label: str,
    spec,
    device: torch.device,
    export_path: Path,
    analysis_period: str,
    periods: dict,
    skip_attribution: bool,
    attribution_max_samples: int | None,
    data_cache: dict,
):
    data_path = spec.data_path
    if not data_path or not Path(data_path).exists():
        raise FileNotFoundError(f"Missing data_path for {label}: {data_path}")

    if data_path not in data_cache:
        data_raw, indicators_raw, _ = load_market_data(data_path, cfg={"start_date": None})
        data_raw.index = pd.to_datetime(data_raw.index)
        indicators_raw.index = pd.to_datetime(indicators_raw.index)
        data_cache[data_path] = (data_raw, indicators_raw)
    data_raw, indicators_raw = data_cache[data_path]

    spec.env_state = _tupleize_env_state(spec.env_state)
    model, spec = load_model(spec, device=device)

    drop_cash = bool(getattr(spec, "drop_cash_from_obs", False) or spec.feature_extractor == "CNN")
    full_state = build_eval_state_frame(
        data_raw,
        indicators_raw,
        env_state=spec.env_state,
        drop_cash=drop_cash,
    )
    traded = [i for i in spec.instruments if i != "CASH"]
    ohlc_cols = [c for c in data_raw.columns if isinstance(c, tuple) and c[0] in traded]
    if not ohlc_cols:
        ohlc_cols = traded
    ohlc = data_raw.loc[:, ohlc_cols].ffill().dropna()

    ref = periods[analysis_period][0] or spec.reference_date
    batch = prepare_ablation_batch(
        model=model,
        spec=spec,
        eval_data=full_state,
        ohlc_data=ohlc,
        device=device,
        ref_date=ref,
    )
    rolled = evaluate_model_weights(
        model=model,
        spec=spec,
        eval_data=full_state,
        ohlc_data=ohlc,
        device=device,
        ref_date=ref,
        batch=batch,
    )

    out_dir = export_path / label
    save_evaluation_outputs(
        output_path=str(out_dir),
        model_key=label,
        weights=rolled["weights"],
        comparison=rolled["comparison"],
        metadata={
            "label": label,
            "type": spec.normalized_type(),
            "algorithm": spec.algorithm,
            "feature_extractor": spec.feature_extractor,
            "lookback": spec.lookback,
            "rebalance_every": spec.rebalance_every,
            "instruments": spec.instruments,
            "analysis_period": analysis_period,
            "data_path": data_path,
            "path": spec.path,
            "obs_cols": len(spec.env_state or []),
        },
    )

    dfw = rolled["weights"].copy()
    dfw.index = pd.to_datetime(dfw.index)
    expanded = dfw.reindex(pd.date_range(dfw.index.min(), dfw.index.max(), freq="D")).ffill()
    pret = compute_portfolio_returns(expanded, ohlc, label)
    pret.to_csv(out_dir / "results" / f"portfolio_returns_{label}.csv")

    attr_summary = None
    if not skip_attribution:
        states = batch["states"]
        if attribution_max_samples and states.shape[0] > attribution_max_samples:
            idx = torch.linspace(0, states.shape[0] - 1, attribution_max_samples).long()
            batch_attr = dict(batch)
            batch_attr["states"] = states[idx]
            batch_attr["actions"] = batch["actions"][idx]
            batch_attr["returns_to_go"] = batch["returns_to_go"][idx]
            batch_attr["timesteps"] = batch["timesteps"][idx]
        else:
            batch_attr = batch
            idx = None

        baseline = torch.tile(batch_attr["states"].mean(dim=0), (batch_attr["states"].shape[0], 1, 1))
        attr = feature_importance_multifamily(
            model=model,
            spec=spec,
            batch=batch_attr,
            baseline=baseline,
            target_action_idx=None,
        )
        mean_abs = attr.detach().abs().mean(dim=tuple(range(attr.ndim - 1))).cpu().numpy().reshape(-1)
        feat_names = [f"dim_{i}" for i in range(mean_abs.shape[0])]
        df_imp = pd.DataFrame(
            {"feature": feat_names, "mean_abs_attribution": mean_abs}
        ).sort_values("mean_abs_attribution", ascending=False)
        df_imp.to_csv(out_dir / "results" / f"feature_importance_{label}.csv", index=False)
        attr_summary = df_imp.head(10).to_dict(orient="records")
        torch.save(
            {"attributions": attr.detach().cpu(), "indices": None if idx is None else idx.cpu()},
            out_dir / "results" / f"attributions_{label}.pt",
        )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "label": label,
        "weights_shape": list(rolled["weights"].shape),
        "batch_states": list(batch["states"].shape),
        "top_features": attr_summary,
        "out_dir": str(out_dir),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(ROOT / "models" / "rl" / "models_manifest.json"))
    parser.add_argument("--export", default=str(ROOT / "results" / "ablation_multifamily_rl"))
    parser.add_argument("--period", default="eval", choices=["train", "eval", "holdout"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-attribution", action="store_true")
    parser.add_argument("--attribution-max-samples", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip models that already have weights_*.csv under the export dir",
    )
    args = parser.parse_args()

    register_rl_factory_shim()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    export_path = Path(args.export)
    export_path.mkdir(parents=True, exist_ok=True)
    (export_path / "results").mkdir(exist_ok=True)

    models_cfg = load_models_manifest(args.manifest)
    if args.only:
        # All substrings must match (AND). Example: --only SAC LB_16 experiment_1
        models_cfg = {
            k: v for k, v in models_cfg.items() if all(s in k for s in args.only)
        }
    labels = list(models_cfg.keys())
    if args.limit:
        labels = labels[: args.limit]

    specs = parse_models_dict({k: models_cfg[k] for k in labels})
    periods = dict(DEFAULT_PERIODS)
    data_cache = {}
    results = []
    failures = []

    print(f"Device={device} models={len(labels)} skip_attr={args.skip_attribution}")
    for label in tqdm(labels, desc="RL ablation"):
        try:
            if args.skip_existing:
                existing = export_path / label / "results" / f"weights_{label}.csv"
                if existing.exists():
                    print(f"SKIP {label}: existing {existing.name}")
                    continue
            spec = enrich_spec_from_metadata(specs[label])
            # preserve manifest drop_cash / data_path
            raw = models_cfg[label]
            if "drop_cash_from_obs" in raw:
                spec.drop_cash_from_obs = bool(raw["drop_cash_from_obs"])
            if raw.get("data_path"):
                spec.data_path = raw["data_path"]
            info = run_one(
                label=label,
                spec=spec,
                device=device,
                export_path=export_path,
                analysis_period=args.period,
                periods=periods,
                skip_attribution=args.skip_attribution,
                attribution_max_samples=args.attribution_max_samples,
                data_cache=data_cache,
            )
            results.append(info)
            print(f"OK {label}: weights={info['weights_shape']} states={info['batch_states']}")
        except Exception as e:
            failures.append({"label": label, "error": str(e), "trace": traceback.format_exc()})
            print(f"FAIL {label}: {e}")

    summary = {"n_ok": len(results), "n_fail": len(failures), "results": results, "failures": failures}
    summary_path = export_path / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"Done. ok={len(results)} fail={len(failures)} -> {summary_path}")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
