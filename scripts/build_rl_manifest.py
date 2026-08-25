"""Build MODELS manifest for all RL_AG_* runs under models/rl."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RL_ROOT = ROOT / "models" / "rl"
OUT_PATH = RL_ROOT / "models_manifest.json"

NAME_RE = re.compile(
    r"^RL_AG_(?P<algo>[A-Za-z0-9]+)_FT_(?P<fe>CNN|MLP)_LB_(?P<lb>\d+)_RE_(?P<re>\d+)(?:_refdate_.*)?$",
    re.IGNORECASE,
)

EXPERIMENT_TI_FALLBACK = {
    "experiment_5": {
        "cash_instrument": True,
        "instruments": ["^GSPC", "^GDAXI", "^IXIC", "^RUT", "^N225", "^FTSE"],
        "technical_indicators": ["ROCP", "ROC", "WCLPRICE", "NATR"],
    }
}

DATA_FALLBACK = {
    "experiment_1": RL_ROOT / "experiment_3" / "data.h5",
}

# Matches ECI sb3_trainer.agent_types
DISCRETE_ALGOS = {"DQN", "PPO"}


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _merge_meta(run_dir: Path, experiment_dir: Path) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    for src in (
        EXPERIMENT_TI_FALLBACK.get(experiment_dir.name, {}),
        _load_json(experiment_dir / "metadata.json"),
        _load_json(run_dir / "metadata.json"),
    ):
        meta.update({k: v for k, v in src.items() if v is not None})
    return meta


def _resolve_data_path(experiment_dir: Path) -> Optional[str]:
    direct = experiment_dir / "data.h5"
    if direct.exists():
        return str(direct.resolve())
    fb = DATA_FALLBACK.get(experiment_dir.name)
    if fb is not None and Path(fb).exists():
        return str(Path(fb).resolve())
    shared = ROOT / "data" / "experiment_2026" / "data.h5"
    if shared.exists():
        return str(shared.resolve())
    return None


def _find_weights(run_dir: Path) -> Optional[Path]:
    for c in (
        run_dir / "best_model.zip",
        run_dir / "models" / "final_model.zip",
        run_dir / "final_model.zip",
        run_dir / "model.zip",
    ):
        if c.exists():
            return c
    zips = sorted(run_dir.glob("**/*.zip"))
    return zips[0] if zips else None


def _build_env_state_from_data(data_path: str, drop_cash: bool) -> List[List[str]]:
    """Derive observation columns from HDF5: indicators merged with OHLC prices."""
    indicators = pd.read_hdf(data_path, "technical_indicators")
    data = pd.read_hdf(data_path, "instruments")
    merged = indicators.merge(data, right_index=True, left_index=True)
    if drop_cash:
        merged = merged.drop(columns=[c for c in merged.columns if c[0] == "CASH"])
    return [[a, b] for a, b in merged.columns]


def _action_instruments(data_path: str) -> List[str]:
    data = pd.read_hdf(data_path, "instruments")
    return [c[0] for c in data.columns if c[1] == "Close"]


def build_manifest() -> Dict[str, Dict[str, Any]]:
    models: Dict[str, Dict[str, Any]] = {}
    skipped: List[Tuple[str, str]] = []
    env_cache: Dict[Tuple[str, bool], List[List[str]]] = {}
    inst_cache: Dict[str, List[str]] = {}

    for experiment_dir in sorted(RL_ROOT.glob("experiment_*")):
        if not experiment_dir.is_dir():
            continue
        data_path = _resolve_data_path(experiment_dir)
        if data_path is None:
            skipped.append((experiment_dir.name, "no data.h5"))
            continue
        for run_dir in sorted(experiment_dir.iterdir()):
            if not (run_dir.is_dir() and run_dir.name.startswith("RL_AG_")):
                continue
            m = NAME_RE.match(run_dir.name)
            if not m:
                skipped.append((run_dir.name, "name mismatch"))
                continue
            weights = _find_weights(run_dir)
            if weights is None:
                skipped.append((run_dir.name, "no weights"))
                continue
            meta = _merge_meta(run_dir, experiment_dir)
            algo = str(meta.get("agent_algorithm") or m.group("algo")).upper()
            fe = str(meta.get("feature_extractor") or m.group("fe")).upper()
            lookback = int(m.group("lb"))
            meta_lb = meta.get("lookback")
            if meta_lb is not None and int(meta_lb) != lookback:
                # Folder name is authoritative; checkpoint obs_space confirms at load
                pass
            rebalance = int(m.group("re"))
            meta_re = meta.get("rebalance_every")
            if meta_re is not None:
                rebalance = int(meta_re)
            drop_cash = fe == "CNN"
            cache_key = (data_path, drop_cash)
            if cache_key not in env_cache:
                env_cache[cache_key] = _build_env_state_from_data(data_path, drop_cash)
            if data_path not in inst_cache:
                inst_cache[data_path] = _action_instruments(data_path)

            label = f"{experiment_dir.name}__{run_dir.name}"
            models[label] = {
                "type": "SB3",
                "path": str(weights.resolve()),
                "algorithm": algo,
                "feature_extractor": fe,
                "lookback": lookback,
                "rebalance_every": rebalance,
                "action_space": "discrete" if algo in DISCRETE_ALGOS else "continuous",
                "instruments": inst_cache[data_path],
                "env_state": env_cache[cache_key],
                "drop_cash_from_obs": drop_cash,
                "metadata_path": str((run_dir / "metadata.json").resolve())
                if (run_dir / "metadata.json").exists()
                else (
                    str((experiment_dir / "metadata.json").resolve())
                    if (experiment_dir / "metadata.json").exists()
                    else None
                ),
                "data_path": data_path,
                "experiment": experiment_dir.name,
                "run_dir": str(run_dir.resolve()),
                "reference_date": "2016-01-01",
                "cnn_layout": "instruments_x_time_x_features",
            }

    payload = {
        "n_models": len(models),
        "skipped": [{"name": n, "reason": r} for n, r in skipped],
        "models": models,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(models)} models -> {OUT_PATH}")
    if skipped:
        print(f"Skipped {len(skipped)}:")
        for n, r in skipped:
            print(f"  - {n}: {r}")
    return models


if __name__ == "__main__":
    build_manifest()
