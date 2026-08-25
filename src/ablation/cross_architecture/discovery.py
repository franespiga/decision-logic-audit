"""
Recursive discovery of Decision Transformer and Stable-Baselines3 checkpoints and
resolution of the comparison manifest.

Inclusion is decided by the pre-specified, purely technical rules in
``INCLUSION_CRITERIA``. No criterion refers to diagnostic or portfolio results.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

# Canonical column order used for every portfolio weight vector in this study.
# Individual policies index their action space differently; weights are always
# re-indexed onto this order before anything is compared.
CANONICAL_INSTRUMENTS: Tuple[str, ...] = (
    "^GSPC",
    "^GDAXI",
    "^IXIC",
    "^RUT",
    "^N225",
    "^FTSE",
    "CASH",
)

SB3_ALGORITHMS = ("PPO", "A2C", "DQN", "DDPG", "TD3", "SAC")

INCLUSION_CRITERIA: Tuple[str, ...] = (
    "loadable: the checkpoint can be deserialised with its declared framework",
    "metadata: a metadata.json declaring instruments and either env_state or "
    "technical_indicators is present next to the checkpoint",
    "universe: the action space maps onto the common seven-element instrument "
    "universe (six equity indices plus CASH)",
    "features: every declared observation column exists in the frozen market "
    "data file",
    "coverage: at least 95% of the common evaluation dates can be built into a "
    "complete observation window",
    "mapping: a deterministic action-to-portfolio-weight mapping is defined for "
    "the declared action space",
)


@dataclass
class ModelRecord:
    """Resolved specification of one candidate policy."""

    model_id: str
    family: str  # "DT" or "RL"
    algorithm: str
    checkpoint_path: str
    metadata_path: Optional[str] = None
    data_path: Optional[str] = None
    preprocessing_path: Optional[str] = None
    environment_path: Optional[str] = None
    feature_extractor: str = "MLP"
    state_features: List[Tuple[str, str]] = field(default_factory=list)
    lookback: Optional[int] = None
    window_length: Optional[int] = None  # observation rows actually consumed
    action_space: str = "continuous"  # "continuous" | "discrete"
    n_actions: Optional[int] = None
    instruments: List[str] = field(default_factory=list)
    reference_date: Optional[str] = None
    rebalancing_frequency: int = 1
    target_return: Optional[float] = None
    weights_source: str = "reconstructed"
    performance_source: str = "reconstructed"
    legacy_weights_path: Optional[str] = None
    state_dim: Optional[int] = None
    inclusion_status: str = "pending"
    exclusion_reason: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["state_features"] = [list(x) for x in self.state_features]
        return payload

    @property
    def n_features(self) -> int:
        return len(self.state_features)


def _md5(path: Path, limit: int = 8_000_000) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        h.update(fh.read(limit))
    return h.hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_dt_models(root: Path) -> List[ModelRecord]:
    """Find DT runs: any directory containing ``model/config.json``."""
    records: List[ModelRecord] = []
    root = Path(root)
    if not root.exists():
        return records
    for cfg in sorted(root.rglob("model/config.json")):
        model_dir = cfg.parent
        run_dir = model_dir.parent
        meta_path = run_dir / "metadata.json"
        # The experiment folder above ``out/`` carries the frozen data file.
        data_path = None
        for parent in [run_dir, *run_dir.parents]:
            candidate = parent / "data.h5"
            if candidate.exists():
                data_path = candidate
                break
            if parent == root:
                break
        # A stable, human-readable identifier: <experiment folder>/<strategy>
        exp_folder = run_dir
        while exp_folder.parent != root and exp_folder.parent != exp_folder:
            exp_folder = exp_folder.parent
        name = exp_folder.name
        model_id = name if name.upper().startswith("DT") else f"DT_{name}"
        cfg_json = _read_json(cfg)
        records.append(
            ModelRecord(
                model_id=model_id,
                family="DT",
                algorithm="DecisionTransformer",
                checkpoint_path=str(model_dir),
                metadata_path=str(meta_path) if meta_path.exists() else None,
                data_path=str(data_path) if data_path else None,
                feature_extractor="Transformer",
                action_space="continuous",
                n_actions=cfg_json.get("act_dim"),
                state_dim=cfg_json.get("state_dim"),
            )
        )
    return records


def discover_rl_models(root: Path) -> List[ModelRecord]:
    """Find SB3 runs: any directory containing ``best_model.zip``/``final_model.zip``."""
    records: List[ModelRecord] = []
    root = Path(root)
    if not root.exists():
        return records
    seen: set[Path] = set()
    for pattern in ("best_model.zip", "models/final_model.zip", "final_model.zip"):
        for ckpt in sorted(root.rglob(pattern)):
            run_dir = ckpt.parent if ckpt.name == "best_model.zip" else ckpt.parent.parent
            if run_dir in seen:
                continue
            seen.add(run_dir)
            meta_path = run_dir / "metadata.json"
            meta = _read_json(meta_path) if meta_path.exists() else {}
            algorithm = str(meta.get("agent_algorithm", "")).upper()
            if not algorithm:
                # fall back to the folder naming convention RL_AG_<ALGO>_...
                parts = run_dir.name.split("_")
                algorithm = next((p for p in parts if p.upper() in SB3_ALGORITHMS), "").upper()
            legacy = run_dir / "results" / "weights_BEST.csv"
            records.append(
                ModelRecord(
                    model_id=f"RL_{algorithm}_LB{meta.get('lookback', '?')}",
                    family="RL",
                    algorithm=algorithm or "UNKNOWN",
                    checkpoint_path=str(ckpt),
                    metadata_path=str(meta_path) if meta_path.exists() else None,
                    feature_extractor=str(meta.get("feature_extractor", "MLP")).upper(),
                    lookback=meta.get("lookback"),
                    rebalancing_frequency=int(meta.get("rebalance_every", 1) or 1),
                    legacy_weights_path=str(legacy) if legacy.exists() else None,
                )
            )
    return records


def discover_models(dt_root: Path, rl_root: Path) -> List[ModelRecord]:
    return discover_dt_models(dt_root) + discover_rl_models(rl_root)


# ---------------------------------------------------------------------------
# Metadata resolution
# ---------------------------------------------------------------------------


def _dt_state_features(meta: Dict[str, Any]) -> List[Tuple[str, str]]:
    return [tuple(x) for x in meta.get("env_state", [])]


def _rl_state_features(
    meta: Dict[str, Any],
    indicator_columns: Sequence[Tuple[str, str]],
    price_columns: Sequence[Tuple[str, str]],
) -> List[Tuple[str, str]]:
    """
    Reconstruct the SB3 observation column order.

    The training environment merged the technical-indicator frame with the price
    frame (``indicators.merge(data)``), so the observation columns are the
    indicator columns in file order followed by the price columns in file order.
    Only columns that exist in the frozen data file are used, which is why the
    ``technical_indicators`` list in metadata.json is not trusted verbatim (it
    names indicators, such as STOCHF, that the frozen file does not contain).
    """
    return [tuple(c) for c in indicator_columns] + [tuple(c) for c in price_columns]


def resolve_record(
    record: ModelRecord,
    indicator_columns: Optional[Sequence[Tuple[str, str]]] = None,
    price_columns: Optional[Sequence[Tuple[str, str]]] = None,
    default_data_path: Optional[str] = None,
    eval_start: str = "2016-01-01",
) -> ModelRecord:
    """Fill a record from its own metadata.json. Never guesses silently."""
    meta: Dict[str, Any] = {}
    if record.metadata_path and Path(record.metadata_path).exists():
        meta = _read_json(Path(record.metadata_path))
    exp = meta.get("experiment_args", {})

    record.data_path = record.data_path or default_data_path
    record.reference_date = (
        record.reference_date or exp.get("reference_date") or meta.get("reference_date") or eval_start
    )

    if record.family == "DT":
        record.state_features = _dt_state_features(meta)
        record.lookback = meta.get("lookback")
        # A DT consumes lookback + 1 observation rows (see prepare_ablation_inputs).
        record.window_length = None if record.lookback is None else int(record.lookback) + 1
        record.instruments = list(meta.get("instruments", []))
        record.target_return = meta.get("target_return")
        record.action_space = "continuous"
        record.notes.append(f"strategy={meta.get('strategy')}")
    else:
        if indicator_columns is None or price_columns is None:
            raise ValueError(
                f"{record.model_id}: RL observation layout requires the frozen data columns"
            )
        record.state_features = _rl_state_features(meta, indicator_columns, price_columns)
        record.lookback = meta.get("lookback", record.lookback)
        record.window_length = None if record.lookback is None else int(record.lookback)
        instruments = list(meta.get("instruments", []))
        if meta.get("cash_instrument") and "CASH" not in instruments:
            instruments = instruments + ["CASH"]
        record.instruments = instruments
        record.rebalancing_frequency = int(meta.get("rebalance_every", 1) or 1)
        record.notes.append(
            "action index order taken from metadata.instruments (+CASH); the legacy "
            "weights_BEST.csv column labels disagree with this order and are not "
            "treated as authoritative"
        )
    return record


# ---------------------------------------------------------------------------
# Inclusion rules
# ---------------------------------------------------------------------------


def apply_inclusion_rules(
    record: ModelRecord,
    *,
    loadable: bool,
    load_error: Optional[str],
    available_columns: Sequence[Tuple[str, str]],
    coverage_fraction: Optional[float],
    min_coverage: float = 0.95,
) -> ModelRecord:
    """Apply the pre-specified technical inclusion rules to a single record."""
    reasons: List[str] = []

    if not loadable:
        reasons.append(f"checkpoint could not be loaded: {load_error}")
    if not record.metadata_path:
        reasons.append("no metadata.json next to the checkpoint")
    if not record.state_features:
        reasons.append("observation column layout could not be resolved from metadata")
    if not record.instruments:
        reasons.append("instrument universe not declared in metadata")
    elif set(record.instruments) != set(CANONICAL_INSTRUMENTS):
        missing = sorted(set(CANONICAL_INSTRUMENTS) - set(record.instruments))
        extra = sorted(set(record.instruments) - set(CANONICAL_INSTRUMENTS))
        reasons.append(
            f"instrument universe differs from the common universe (missing={missing}, extra={extra})"
        )
    if record.state_features:
        available = set(map(tuple, available_columns))
        absent = [c for c in record.state_features if tuple(c) not in available]
        if absent:
            reasons.append(
                f"{len(absent)} declared observation columns absent from the frozen data file "
                f"(first: {absent[:3]})"
            )
    if record.window_length is None:
        reasons.append("observation window length (lookback) not declared")
    if record.action_space not in {"continuous", "discrete"}:
        reasons.append(f"unsupported action space '{record.action_space}'")
    if coverage_fraction is not None and coverage_fraction < min_coverage:
        reasons.append(
            f"only {coverage_fraction:.1%} of common evaluation dates yield a complete "
            f"observation window (< {min_coverage:.0%})"
        )

    record.inclusion_status = "included" if not reasons else "excluded"
    record.exclusion_reason = None if not reasons else "; ".join(reasons)
    return record


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------


def manifest_to_yaml(records: Sequence[ModelRecord], meta: Dict[str, Any]) -> str:
    """Render the resolved manifest as YAML without requiring a YAML writer."""
    import yaml

    payload = {
        "meta": meta,
        "inclusion_criteria": list(INCLUSION_CRITERIA),
        "canonical_instruments": list(CANONICAL_INSTRUMENTS),
        "models": [r.to_dict() for r in records],
    }
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=100)


def write_manifest(path: Path, records: Sequence[ModelRecord], meta: Dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest_to_yaml(records, meta), encoding="utf-8")
    return path


def read_manifest(path: Path) -> Tuple[List[ModelRecord], Dict[str, Any]]:
    import yaml

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    records = []
    for entry in payload.get("models", []):
        entry = dict(entry)
        entry["state_features"] = [tuple(x) for x in entry.get("state_features", [])]
        records.append(ModelRecord(**entry))
    return records, payload.get("meta", {})


def records_to_frame(records: Sequence[ModelRecord]) -> pd.DataFrame:
    rows = []
    for r in records:
        d = r.to_dict()
        d["n_state_features"] = r.n_features
        d.pop("state_features")
        d["notes"] = " | ".join(r.notes)
        rows.append(d)
    return pd.DataFrame(rows)
