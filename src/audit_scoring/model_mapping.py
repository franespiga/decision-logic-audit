"""Model ID and family mapping for diagnostic PKLs."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from .config import BENCHMARK_FAMILIES, AuditConfig
from .io import (
    load_pickle,
    path_has_partial,
    read_table,
    technical_indicator_flag_from_path,
    discover_metric_file,
)

LOGGER = logging.getLogger("audit_scoring.model_mapping")

EXPERIMENT_RE = re.compile(r"^(?P<exp>RL_AG_DT_refdate_\d{4}-\d{2}-\d{2}_\d+)$")
MODEL_ID_RE = re.compile(
    r"^(?P<family>.+)_ar_(?P<ar>-?\d+(?:\.\d+)?)_re_(?P<re>\d+)_it_(?P<it>True|False)"
    r"_tro_(?P<tro>True|False)_rd_(?P<rd>\d{4}-\d{2}-\d{2})$"
)
OUTPUT_SUFFIXES = (
    "_model_diagnostics",
    "model_diagnostics_",
    "_diagnostics",
    "_attributions",
)


@dataclass
class MappingResult:
    source_path: str
    experiment_name: str
    canonical_model_id: Optional[str]
    model_family: Optional[str]
    model_type: Optional[str]
    technical_indicators: Optional[bool]
    mapping_method: str
    mapping_confidence: str
    mapping_status: str
    ambiguity_reason: str = ""


def experiment_name_from_filename(path: Path) -> str:
    stem = path.stem
    if stem.startswith("model_diagnostics_"):
        return stem[len("model_diagnostics_") :]
    return stem


def normalize_name(text: str) -> str:
    value = str(text).strip()
    for suffix in OUTPUT_SUFFIXES:
        if value.startswith(suffix):
            value = value[len(suffix) :]
        if value.endswith(suffix):
            value = value[: -len(suffix)]
    return value.lower().replace("-", "_")


def ti_from_family(model_family: str) -> Optional[bool]:
    if model_family.endswith("_TI_True"):
        return True
    if model_family.endswith("_TI_False"):
        return False
    return None


def model_type_from_family(model_family: str) -> str:
    if model_family.startswith("DT_"):
        return "decision_transformer_variant"
    if "efficient" in model_family or model_family.startswith("max_"):
        return "portfolio_objective_conditioned_dt"
    if model_family.endswith("_full_TI_True") or model_family.endswith("_full_TI_False"):
        return "full_feature_dt"
    return "decision_transformer"


def load_expected_families(metrics_dir: Path) -> List[str]:
    path = discover_metric_file(metrics_dir, "family_cost_sensitivity")
    df = read_table(path)
    if "model_family" not in df.columns:
        raise ValueError(f"{path} missing model_family column")
    families = sorted(
        f
        for f in df["model_family"].astype(str).unique()
        if f not in BENCHMARK_FAMILIES
    )
    return families


def load_model_cost_catalog(metrics_dir: Path) -> pd.DataFrame:
    path = discover_metric_file(metrics_dir, "model_cost_sensitivity")
    df = read_table(path)
    cols = [c for c in ["model_id", "model_family"] if c in df.columns]
    if len(cols) < 2:
        return pd.DataFrame(columns=["model_id", "model_family"])
    catalog = (
        df.loc[:, cols]
        .dropna()
        .astype(str)
        .drop_duplicates()
    )
    catalog = catalog[~catalog["model_family"].isin(BENCHMARK_FAMILIES)].copy()
    return catalog.reset_index(drop=True)


def load_family_mapping_report(path: Path) -> pd.DataFrame:
    if path is None or not Path(path).exists():
        return pd.DataFrame()
    df = read_table(path)
    if "model_path" in df.columns:
        df = df.copy()
        df["experiment_name"] = df["model_path"].map(_experiment_from_path)
    return df


def _experiment_from_path(path_value: Any) -> Optional[str]:
    parts = str(path_value).replace("/", "\\").split("\\")
    for part in parts:
        if part.startswith("RL_AG_DT_"):
            return part
    return None


def build_experiment_to_family(mapping_report: pd.DataFrame) -> Dict[str, str]:
    if mapping_report.empty or "experiment_name" not in mapping_report.columns:
        return {}
    pairs = (
        mapping_report.dropna(subset=["experiment_name", "model_family"])
        .loc[:, ["experiment_name", "model_family"]]
        .astype(str)
        .drop_duplicates()
    )
    out: Dict[str, str] = {}
    conflicts: Dict[str, set] = {}
    for _, row in pairs.iterrows():
        exp = row["experiment_name"]
        fam = row["model_family"]
        if exp in out and out[exp] != fam:
            conflicts.setdefault(exp, {out[exp]}).add(fam)
        else:
            out[exp] = fam
    for exp, fams in conflicts.items():
        LOGGER.warning("Ambiguous experiment→family mapping for %s: %s", exp, sorted(fams))
        out.pop(exp, None)
    return out


def expected_models_per_family(mapping_report: pd.DataFrame, families: Sequence[str]) -> Dict[str, int]:
    counts = {f: 0 for f in families}
    if mapping_report.empty or "model_family" not in mapping_report.columns:
        # Family-checkpoint diagnostics: one diagnostic unit expected per family.
        return {f: 1 for f in families}
    grouped = mapping_report.groupby("model_family")["model_id"].nunique()
    for fam in families:
        counts[fam] = int(grouped.get(fam, 0)) or 1
    return counts


def _embedded_metadata(payload: Mapping[str, Any]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    for key in (
        "model_id",
        "canonical_model_id",
        "model_family",
        "experiment_name",
        "metadata",
        "config",
    ):
        if key in payload:
            meta[key] = payload[key]
    nested = payload.get("metadata")
    if isinstance(nested, Mapping):
        for key in ("model_id", "model_family", "experiment_name", "technical_indicators"):
            if key in nested and key not in meta:
                meta[key] = nested[key]
    return meta


def map_one_pkl(
    path: Path,
    root: Path,
    *,
    expected_families: Sequence[str],
    model_catalog: pd.DataFrame,
    experiment_to_family: Mapping[str, str],
    assessment_index: Mapping[str, Mapping[str, Any]] | None = None,
    payload_meta: Mapping[str, Any] | None = None,
) -> MappingResult:
    experiment = experiment_name_from_filename(path)
    folder_ti = technical_indicator_flag_from_path(path)
    meta = dict(payload_meta or {})

    # 1. Embedded metadata
    emb_id = meta.get("model_id") or meta.get("canonical_model_id")
    emb_family = meta.get("model_family")
    if emb_id and emb_family and str(emb_family) in set(expected_families):
        return MappingResult(
            source_path=str(path),
            experiment_name=experiment,
            canonical_model_id=str(emb_id),
            model_family=str(emb_family),
            model_type=model_type_from_family(str(emb_family)),
            technical_indicators=ti_from_family(str(emb_family))
            if ti_from_family(str(emb_family)) is not None
            else folder_ti,
            mapping_method="embedded_metadata",
            mapping_confidence="high",
            mapping_status="resolved",
        )

    # 2. Exact experiment / checkpoint-name match against model_cost_sensitivity
    if not model_catalog.empty:
        hits = model_catalog[model_catalog["model_id"].astype(str) == experiment]
        if len(hits) == 1:
            fam = str(hits.iloc[0]["model_family"])
            mid = str(hits.iloc[0]["model_id"])
            return MappingResult(
                source_path=str(path),
                experiment_name=experiment,
                canonical_model_id=mid,
                model_family=fam,
                model_type=model_type_from_family(fam),
                technical_indicators=ti_from_family(fam),
                mapping_method="model_cost_sensitivity_exact_experiment",
                mapping_confidence="high",
                mapping_status="resolved",
            )
        if len(hits) > 1:
            return MappingResult(
                source_path=str(path),
                experiment_name=experiment,
                canonical_model_id=None,
                model_family=None,
                model_type=None,
                technical_indicators=folder_ti,
                mapping_method="model_cost_sensitivity_exact_experiment",
                mapping_confidence="low",
                mapping_status="ambiguous",
                ambiguity_reason=f"Multiple model_cost_sensitivity rows for experiment={experiment}",
            )

    # 3. Exact filename-stem match against model_id
    if not model_catalog.empty:
        hits = model_catalog[model_catalog["model_id"].astype(str) == path.stem]
        if len(hits) == 1:
            fam = str(hits.iloc[0]["model_family"])
            mid = str(hits.iloc[0]["model_id"])
            return MappingResult(
                source_path=str(path),
                experiment_name=experiment,
                canonical_model_id=mid,
                model_family=fam,
                model_type=model_type_from_family(fam),
                technical_indicators=ti_from_family(fam),
                mapping_method="model_id_stem_exact",
                mapping_confidence="high",
                mapping_status="resolved",
            )

    # 4. Unique normalized-name match
    if not model_catalog.empty:
        norm_exp = normalize_name(experiment)
        norms = model_catalog.assign(model_id_norm=model_catalog["model_id"].map(normalize_name))
        hits = norms[norms["model_id_norm"] == norm_exp]
        if len(hits) == 1:
            fam = str(hits.iloc[0]["model_family"])
            mid = str(hits.iloc[0]["model_id"])
            return MappingResult(
                source_path=str(path),
                experiment_name=experiment,
                canonical_model_id=mid,
                model_family=fam,
                model_type=model_type_from_family(fam),
                technical_indicators=ti_from_family(fam),
                mapping_method="normalized_unique_match",
                mapping_confidence="medium",
                mapping_status="resolved",
            )
        if len(hits) > 1:
            return MappingResult(
                source_path=str(path),
                experiment_name=experiment,
                canonical_model_id=None,
                model_family=None,
                model_type=None,
                technical_indicators=folder_ti,
                mapping_method="normalized_unique_match",
                mapping_confidence="low",
                mapping_status="ambiguous",
                ambiguity_reason="Multiple normalized model_id matches",
            )

    # 5. family_mapping_report / assessment experiment map
    if experiment in experiment_to_family:
        fam = experiment_to_family[experiment]
        if fam in set(expected_families):
            # Family-checkpoint diagnostic: canonical ID names the checkpoint unit.
            mid = f"{fam}__family_checkpoint__{experiment}"
            ti = ti_from_family(fam)
            if folder_ti is not None and ti is not None and folder_ti != ti:
                return MappingResult(
                    source_path=str(path),
                    experiment_name=experiment,
                    canonical_model_id=mid,
                    model_family=fam,
                    model_type=model_type_from_family(fam),
                    technical_indicators=ti,
                    mapping_method="family_mapping_report_experiment",
                    mapping_confidence="medium",
                    mapping_status="ambiguous",
                    ambiguity_reason=(
                        f"Folder TI flag={folder_ti} inconsistent with family TI={ti}"
                    ),
                )
            return MappingResult(
                source_path=str(path),
                experiment_name=experiment,
                canonical_model_id=mid,
                model_family=fam,
                model_type=model_type_from_family(fam),
                technical_indicators=ti if ti is not None else folder_ti,
                mapping_method="family_mapping_report_experiment",
                mapping_confidence="high",
                mapping_status="resolved",
            )

    if assessment_index and experiment in assessment_index:
        info = assessment_index[experiment]
        fam = str(info.get("model_family", ""))
        mid = str(info.get("model_id", f"{fam}__family_checkpoint__{experiment}"))
        if fam in set(expected_families):
            return MappingResult(
                source_path=str(path),
                experiment_name=experiment,
                canonical_model_id=mid,
                model_family=fam,
                model_type=model_type_from_family(fam),
                technical_indicators=ti_from_family(fam),
                mapping_method="model_assessment_pickle",
                mapping_confidence="medium",
                mapping_status="resolved",
            )

    return MappingResult(
        source_path=str(path),
        experiment_name=experiment,
        canonical_model_id=None,
        model_family=None,
        model_type=None,
        technical_indicators=folder_ti,
        mapping_method="unresolved",
        mapping_confidence="none",
        mapping_status="unresolved",
        ambiguity_reason="No authoritative model_id/family mapping found",
    )


def deduplicate_paths(
    paths: Sequence[Path],
    root: Path,
    prefer_complete_over_partial: bool = True,
) -> Tuple[List[Path], List[MappingResult]]:
    """Prefer non-partial PKLs when the same experiment appears twice."""
    by_experiment: Dict[str, List[Path]] = {}
    for path in paths:
        by_experiment.setdefault(experiment_name_from_filename(path), []).append(path)

    kept: List[Path] = []
    duplicate_rows: List[MappingResult] = []
    for experiment, group in sorted(by_experiment.items()):
        if len(group) == 1:
            kept.append(group[0])
            continue
        complete = [p for p in group if not path_has_partial(p, root)]
        partial = [p for p in group if path_has_partial(p, root)]
        if prefer_complete_over_partial and complete:
            chosen = sorted(complete)[-1]
            kept.append(chosen)
            for other in group:
                if other == chosen:
                    continue
                duplicate_rows.append(
                    MappingResult(
                        source_path=str(other),
                        experiment_name=experiment,
                        canonical_model_id=None,
                        model_family=None,
                        model_type=None,
                        technical_indicators=technical_indicator_flag_from_path(other),
                        mapping_method="duplicate_experiment_partial_or_multi",
                        mapping_confidence="none",
                        mapping_status="duplicate",
                        ambiguity_reason=f"Superseded by {chosen}",
                    )
                )
        else:
            # Ambiguous duplicates with no complete preference winner.
            for other in group:
                duplicate_rows.append(
                    MappingResult(
                        source_path=str(other),
                        experiment_name=experiment,
                        canonical_model_id=None,
                        model_family=None,
                        model_type=None,
                        technical_indicators=technical_indicator_flag_from_path(other),
                        mapping_method="duplicate_experiment",
                        mapping_confidence="none",
                        mapping_status="duplicate",
                        ambiguity_reason="Multiple PKLs for the same experiment",
                    )
                )
    return kept, duplicate_rows


def load_assessment_index(path: Path) -> Dict[str, Dict[str, Any]]:
    """Optional fallback index from model_assessment.pkl (expensive)."""
    if path is None or not Path(path).exists():
        return {}
    LOGGER.warning("Loading model_assessment.pkl (large); this may take several minutes.")
    payload = load_pickle(path)
    index: Dict[str, Dict[str, Any]] = {}
    if not isinstance(payload, Mapping):
        return index
    weight_meta = payload.get("weight_metadata", {})
    if isinstance(weight_meta, Mapping):
        for model_id, meta in weight_meta.items():
            if not isinstance(meta, Mapping):
                continue
            family = meta.get("model_family") or meta.get("family")
            exp = meta.get("experiment_name") or meta.get("checkpoint_name")
            if exp:
                index[str(exp)] = {
                    "model_id": str(model_id),
                    "model_family": str(family) if family else None,
                }
    return index


def mappings_to_frame(rows: Sequence[MappingResult]) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in rows])
