"""Individual-model discovery for Batch A diagnostics.

Maps the 1,728 hyperparameter variants onto the 18 family checkpoints under
``model_checkpoints``, using performance tables and optional assessment metadata.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

LOGGER = logging.getLogger("batch_a_diagnostics.discovery")

BENCHMARK_FAMILIES = {"EWP", "^GSPC"}
MODEL_ID_PATTERN = re.compile(
    r"^(?P<family>.+)_ar_(?P<ar>-?\d+(?:\.\d+)?)_re_(?P<re>\d+)_it_(?P<it>True|False)"
    r"_tro_(?P<tro>True|False)_rd_(?P<rd>\d{4}-\d{2}-\d{2})$"
)


@dataclass
class IndividualModelRecord:
    model_id: str
    model_family: str
    model_path: Path
    metadata_path: Path
    metadata: Dict[str, Any]
    hyperparams: Dict[str, Any]
    mapping_method: str


def family_stem_and_ti(model_family: str) -> Tuple[str, bool]:
    """Split ``DT_16_1_TI_False`` into stem ``DT_16_1`` and TI flag."""
    if model_family.endswith("_TI_True"):
        return model_family[: -len("_TI_True")], True
    if model_family.endswith("_TI_False"):
        return model_family[: -len("_TI_False")], False
    raise ValueError(f"Cannot parse TI flag from model_family={model_family!r}")


def parse_model_id(model_id: str) -> Optional[Dict[str, Any]]:
    match = MODEL_ID_PATTERN.match(str(model_id))
    if not match:
        return None
    groups = match.groupdict()
    return {
        "model_family": groups["family"],
        "annualized_return": float(groups["ar"]),
        "rebalance_every": int(groups["re"]),
        "use_incremental_timesteps": groups["it"] == "True",
        "use_trajectory_return_objective": groups["tro"] == "True",
        "reference_date": groups["rd"],
    }


def load_individual_catalog(metrics_dir: Path) -> pd.DataFrame:
    """Load the 1,728-model / 18-family catalog from Batch A performance tables."""
    candidates = [
        metrics_dir / "holdout_metrics_individual.csv",
        metrics_dir / "holdout_metrics_individual.parquet",
        metrics_dir / "validation_holdout_metrics_individual.csv",
        metrics_dir / "model_cost_sensitivity.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(
            f"No individual metrics table found under {metrics_dir}. "
            "Expected holdout_metrics_individual.csv."
        )
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    required = {"model_id", "model_family"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns {sorted(missing)}")
    catalog = (
        df.loc[:, ["model_id", "model_family"]]
        .dropna()
        .astype({"model_id": str, "model_family": str})
        .drop_duplicates()
    )
    catalog = catalog[~catalog["model_family"].isin(BENCHMARK_FAMILIES)].copy()
    catalog = catalog.sort_values(["model_family", "model_id"]).reset_index(drop=True)
    LOGGER.info(
        "Loaded catalog from %s: %d models, %d families",
        path,
        catalog["model_id"].nunique(),
        catalog["model_family"].nunique(),
    )
    return catalog


def discover_family_checkpoints(models_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Discover the 18 family checkpoints under ``models_path``.

    Expected layout::
        models_path/{techindi|no_techindi}/{stem}/out/RL_AG_DT_*/metadata.json
        models_path/{techindi|no_techindi}/{stem}/out/RL_AG_DT_*/model/
    """
    mapping: Dict[str, Dict[str, Any]] = {}
    if not models_path.exists():
        return mapping

    for meta_path in sorted(models_path.rglob("metadata.json")):
        # Prefer */out/*/metadata.json
        if "out" not in meta_path.parts:
            continue
        experiment_dir = meta_path.parent
        model_dir = experiment_dir / "model"
        if not model_dir.exists():
            # Some layouts store the pretrained folder as experiment_dir itself
            model_dir = experiment_dir
        stem_dir = experiment_dir.parent.parent  # .../{stem}/out/{exp}
        ti_dir = stem_dir.parent.name.lower()
        stem = stem_dir.name
        if ti_dir == "techindi":
            family = f"{stem}_TI_True"
        elif ti_dir == "no_techindi":
            family = f"{stem}_TI_False"
        else:
            path_text = str(meta_path).lower().replace("\\", "/")
            family = (
                f"{stem}_TI_False"
                if "/no_techindi/" in f"/{path_text}/"
                else f"{stem}_TI_True"
            )

        with meta_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        metadata["model_path"] = str(model_dir)
        # Prefer the family-local data.h5 (matches notebook experiment layout).
        if (stem_dir / "data.h5").exists():
            metadata["data_path"] = str(stem_dir)
        elif (stem_dir.parent / "data.h5").exists():
            metadata["data_path"] = str(stem_dir.parent)
        else:
            metadata["data_path"] = str(stem_dir)

        mapping[family] = {
            "model_family": family,
            "model_path": model_dir,
            "metadata_path": meta_path,
            "metadata": metadata,
            "stem": stem,
            "ti_dir": ti_dir,
        }
    LOGGER.info("Discovered %d family checkpoints under %s", len(mapping), models_path)
    return mapping


def load_weight_metadata(assessment_pickle: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if assessment_pickle is None or not Path(assessment_pickle).exists():
        return {}
    if cloudpickle_load := _cloudpickle_load(assessment_pickle):
        raw = cloudpickle_load.get("weight_metadata", {})
        return {str(k): dict(v) if isinstance(v, Mapping) else {} for k, v in raw.items()}
    return {}


def _cloudpickle_load(path: Path) -> Optional[Dict[str, Any]]:
    try:
        import cloudpickle
    except ImportError:
        LOGGER.warning("cloudpickle unavailable; cannot read %s", path)
        return None
    with Path(path).open("rb") as handle:
        payload = cloudpickle.load(handle)
    return payload if isinstance(payload, dict) else None


def resolve_hyperparams(
    model_id: str,
    model_family: str,
    weight_metadata: Mapping[str, Mapping[str, Any]],
    family_metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    parsed = parse_model_id(model_id) or {}
    wm = dict(weight_metadata.get(model_id, {}))
    lookback = int(wm.get("lookback", family_metadata.get("lookback", 16)))
    return {
        "annualized_return": float(
            wm.get("annualized_return", parsed.get("annualized_return", family_metadata.get("target_return", 0.0)))
        ),
        "rebalance_every": int(wm.get("rebalance_every", parsed.get("rebalance_every", 1))),
        "use_incremental_timesteps": bool(
            wm.get("use_incremental_timesteps", parsed.get("use_incremental_timesteps", True))
        ),
        "use_trajectory_return_objective": bool(
            wm.get(
                "use_trajectory_return_objectives",
                wm.get(
                    "use_trajectory_return_objective",
                    parsed.get("use_trajectory_return_objective", False),
                ),
            )
        ),
        "reference_date": str(
            wm.get("reference_date", parsed.get("reference_date", family_metadata.get("experiment_args", {}).get("reference_date", "2016-01-01")))
        ),
        "lookback": lookback,
        "technical_indicators": bool(wm.get("technical_indicators", model_family.endswith("_TI_True"))),
        "model_family": model_family,
    }


def build_individual_records(
    metrics_dir: Path,
    models_path: Path,
    assessment_pickle: Optional[Path] = None,
) -> Tuple[List[IndividualModelRecord], pd.DataFrame, List[Dict[str, Any]]]:
    """
    Build one record per individual model.

    Returns
    -------
    records, mapping_report, unresolved_paths
    """
    catalog = load_individual_catalog(metrics_dir)
    family_ckpts = discover_family_checkpoints(models_path)
    weight_meta = load_weight_metadata(assessment_pickle)

    records: List[IndividualModelRecord] = []
    rows: List[Dict[str, Any]] = []
    unresolved: List[Dict[str, Any]] = []

    for _, row in catalog.iterrows():
        model_id = str(row["model_id"])
        model_family = str(row["model_family"])
        ckpt = family_ckpts.get(model_family)
        if ckpt is None:
            unresolved.append(
                {
                    "path_or_object": f"family_checkpoint[{model_family}]",
                    "expected": f"{models_path}/{{techindi|no_techindi}}/<stem>/out/*/metadata.json",
                    "notebook_cell": "Configuration / Model loading (cells 6-10)",
                    "model_id": model_id,
                    "model_family": model_family,
                    "reason": "No matching family checkpoint discovered",
                }
            )
            rows.append(
                {
                    "model_id": model_id,
                    "model_family": model_family,
                    "model_path": None,
                    "mapping_method": "missing_family_checkpoint",
                    "mapping_resolved": False,
                }
            )
            continue

        hyperparams = resolve_hyperparams(model_id, model_family, weight_meta, ckpt["metadata"])
        metadata = dict(ckpt["metadata"])
        metadata["target_return"] = hyperparams["annualized_return"]
        metadata["lookback"] = hyperparams["lookback"]
        exp_args = dict(metadata.get("experiment_args", {}))
        exp_args["reference_date"] = hyperparams["reference_date"]
        metadata["experiment_args"] = exp_args
        metadata["individual_hyperparams"] = hyperparams

        records.append(
            IndividualModelRecord(
                model_id=model_id,
                model_family=model_family,
                model_path=Path(ckpt["model_path"]),
                metadata_path=Path(ckpt["metadata_path"]),
                metadata=metadata,
                hyperparams=hyperparams,
                mapping_method="family_checkpoint_plus_individual_hyperparams",
            )
        )
        rows.append(
            {
                "model_id": model_id,
                "model_family": model_family,
                "model_path": str(ckpt["model_path"]),
                "mapping_method": "family_checkpoint_plus_individual_hyperparams",
                "mapping_resolved": True,
                "rebalance_every": hyperparams["rebalance_every"],
                "annualized_return": hyperparams["annualized_return"],
                "use_incremental_timesteps": hyperparams["use_incremental_timesteps"],
                "use_trajectory_return_objective": hyperparams["use_trajectory_return_objective"],
            }
        )

    # Family-level unresolved checkpoints (expected 18 families with no path)
    for family in sorted(set(catalog["model_family"]) - set(family_ckpts)):
        if not any(u.get("model_family") == family for u in unresolved):
            unresolved.append(
                {
                    "path_or_object": f"family_checkpoint[{family}]",
                    "expected": str(models_path / "{techindi|no_techindi}" / "<stem>" / "out" / "*" / "metadata.json"),
                    "notebook_cell": "Configuration / Model loading",
                    "model_id": None,
                    "model_family": family,
                    "reason": "Family present in metrics catalog but checkpoint missing",
                }
            )

    report = pd.DataFrame(rows)
    return records, report, unresolved
