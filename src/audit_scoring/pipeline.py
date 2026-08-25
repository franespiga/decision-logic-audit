"""End-to-end Batch A pipeline orchestration."""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .aggregation import aggregate_family_raw
from .config import (
    AuditConfig,
    CORE_CONTRIBUTION_METRICS,
    CORE_NECESSITY_METRICS,
    CORE_STRUCTURE_METRICS,
    EXPECTED_N_FAMILIES,
    PROFILES,
)
from .contribution import compute_contribution_diagnostics
from .decision_rules import (
    assign_performance_bands,
    build_validation_master,
    compute_diagnostic_thresholds,
    load_validation_performance,
)
from .feature_groups import build_taxonomy
from .io import (
    discover_diagnostic_pkls,
    ensure_output_dir,
    load_pickle,
    path_has_partial,
    scalar_float,
    to_numpy,
    write_json,
    write_table,
    write_yaml,
)
from .model_mapping import (
    build_experiment_to_family,
    deduplicate_paths,
    expected_models_per_family,
    load_assessment_index,
    load_expected_families,
    load_family_mapping_report,
    load_model_cost_catalog,
    map_one_pkl,
    mappings_to_frame,
    experiment_name_from_filename,
)
from .necessity import compute_necessity_diagnostics
from .reporting import (
    write_batch_a_report,
    write_metric_registry,
    write_missing_inputs,
    write_qc_report,
    write_methodology_tex,
)
from .scoring import score_family_medians, score_model_metrics
from .structure import compute_structure_diagnostics
from .schemas import (
    ATTENTION_KEYS,
    CONTRIBUTION_KEYS,
    LATENT_KEYS,
    LAYER_KEYS,
    OCCLUSION_KEYS,
    PROBE_KEYS,
    ROLLING_KEYS,
    SEPARABILITY_KEYS,
)

LOGGER = logging.getLogger("audit_scoring.pipeline")


def _top_level_keys(payload: Any) -> List[str]:
    if isinstance(payload, dict):
        return sorted(map(str, payload.keys()))
    return [type(payload).__name__]


def _has_any(payload: Dict[str, Any], keys: Sequence[str]) -> bool:
    return any(k in payload and payload.get(k) is not None for k in keys)


def inventory_one(
    path: Path,
    root: Path,
    mapping_row: Dict[str, Any],
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    owned = False
    if payload is None:
        payload = load_pickle(path)
        owned = True
    keys = _top_level_keys(payload) if isinstance(payload, dict) else []
    earliest = latest = None
    notes = []
    if isinstance(payload, dict):
        dates = None
        ab = payload.get("ablation_attributions")
        if isinstance(ab, dict) and ab.get("dates") is not None:
            dates = to_numpy(ab["dates"])
        if dates is not None and getattr(dates, "size", 0):
            di = pd.to_datetime(dates.reshape(-1))
            earliest = str(pd.Timestamp(di.min()).date())
            latest = str(pd.Timestamp(di.max()).date())
        if path_has_partial(path, root):
            notes.append("partial_directory")
    row = {
        "source_path": str(path),
        "relative_path": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
        "filename": path.name,
        "experiment_name": mapping_row.get("experiment_name") or experiment_name_from_filename(path),
        "canonical_model_id": mapping_row.get("canonical_model_id"),
        "model_family": mapping_row.get("model_family"),
        "model_type": mapping_row.get("model_type"),
        "technical_indicators": mapping_row.get("technical_indicators"),
        "mapping_method": mapping_row.get("mapping_method"),
        "mapping_confidence": mapping_row.get("mapping_confidence"),
        "mapping_status": mapping_row.get("mapping_status"),
        "top_level_keys": "|".join(keys),
        "has_contribution_inputs": _has_any(payload, CONTRIBUTION_KEYS) if isinstance(payload, dict) else False,
        "has_occlusion_results": _has_any(payload, OCCLUSION_KEYS) if isinstance(payload, dict) else False,
        "has_rolling_stability": _has_any(payload, ROLLING_KEYS) if isinstance(payload, dict) else False,
        "has_layer_representations": _has_any(payload, LAYER_KEYS) if isinstance(payload, dict) else False,
        "has_latent_representations": _has_any(payload, LATENT_KEYS) if isinstance(payload, dict) else False,
        "has_linear_probe_results": _has_any(payload, PROBE_KEYS) if isinstance(payload, dict) else False,
        "has_separability_metrics": _has_any(payload, SEPARABILITY_KEYS) if isinstance(payload, dict) else False,
        "has_attention_results": _has_any(payload, ATTENTION_KEYS) if isinstance(payload, dict) else False,
        "earliest_date": earliest,
        "latest_date": latest,
        "notes": "; ".join(notes),
    }
    if owned:
        del payload
        gc.collect()
    return row


def extract_model_diagnostics(
    path: Path,
    mapping: Dict[str, Any],
    config: AuditConfig,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[str]]:
    """Extract raw metrics for one PKL; release large arrays afterwards."""
    warnings: List[str] = []
    descriptive_rows: List[Dict[str, Any]] = []
    payload = load_pickle(path)
    if not isinstance(payload, dict):
        return (
            {
                "model_id": mapping.get("canonical_model_id"),
                "model_family": mapping.get("model_family"),
                "overall_raw_validity_status": "failed",
                "warnings": "payload_not_dict",
                "source_path": str(path),
            },
            [],
            ["payload_not_dict"],
        )

    ab = payload.get("ablation_attributions") if isinstance(payload.get("ablation_attributions"), dict) else {}
    dates = ab.get("dates")
    labels = None
    latent = payload.get("latent_representations")
    if isinstance(latent, dict):
        labels = latent.get("labels")

    # Temporal gate
    temporal_status = "unavailable"
    n_after = 0
    if dates is not None:
        di = pd.to_datetime(to_numpy(dates).reshape(-1))
        n_after = int((di > config.validation_end).sum())
        n_before = int((di < config.validation_start).sum())
        temporal_status = "verified"
        if n_after > 0:
            warnings.append(f"filtered_{n_after}_post_validation_dates")
        if n_before > 0:
            warnings.append(f"filtered_{n_before}_pre_validation_dates")

    contrib = compute_contribution_diagnostics(
        payload.get("postprocessed_attributions"),
        payload.get("postprocessed_feature_names"),
        dates=dates,
        labels=labels,
        bootstrap_replicates=config.bootstrap_replicates,
        random_seed=config.random_seed,
        top_k=config.top_k_families,
        min_observations=config.min_attribution_observations,
        min_feature_families=config.min_feature_families,
        validation_start=config.validation_start,
        validation_end=config.validation_end,
    )
    warnings.extend(contrib.warnings)

    nec = compute_necessity_diagnostics(
        payload.get("occlusion_results"),
        payload.get("rolling_stability"),
        eps=config.occlusion_effect_eps,
        nondegeneracy_tol=config.occlusion_nondegeneracy_tol,
    )
    warnings.extend(nec.warnings)

    struct = compute_structure_diagnostics(
        latent_representations=payload.get("latent_representations"),
        layer_representations=payload.get("layer_representations"),
        linear_probe_results=payload.get("linear_probe_results"),
        separability_metrics=payload.get("separability_metrics"),
        regime_labels=labels,
    )
    warnings.extend(struct.warnings)

    # Feature taxonomy incremental names
    names = payload.get("postprocessed_feature_names")
    if names is not None:
        flat = to_numpy(names).reshape(-1).astype(str)
        for fam_name, value in (contrib.descriptive.get("family_abs_profile") or {}).items():
            descriptive_rows.append(
                {
                    "model_id": mapping.get("canonical_model_id"),
                    "metric_id": "contribution_abs_family_profile",
                    "metric_value": value,
                    "feature_family": fam_name,
                    "notes": "",
                }
            )
        taxonomy_preview = build_taxonomy(flat.tolist())
        unclassified = taxonomy_preview[taxonomy_preview["mapping_status"] == "unclassified"]
        if not unclassified.empty:
            warnings.append(f"unclassified_features={len(unclassified)}")

    for key, value in {
        **{f"contrib::{k}": v for k, v in contrib.descriptive.items() if np.isscalar(v)},
        **{f"necessity::{k}": v for k, v in nec.descriptive.items() if np.isscalar(v)},
        **{f"structure::{k}": v for k, v in struct.descriptive.items() if np.isscalar(v)},
    }.items():
        descriptive_rows.append(
            {
                "model_id": mapping.get("canonical_model_id"),
                "metric_id": key,
                "metric_value": value,
                "feature_family": "",
                "notes": "descriptive",
            }
        )

    overall = "passed"
    statuses = [contrib.validity_status, nec.validity_status, struct.validity_status]
    if "failed" in statuses:
        overall = "failed"
    elif "incomplete" in statuses:
        overall = "incomplete"

    # Mandatory gate: occlusion non-degeneracy when occlusion present
    mandatory_failed = False
    if payload.get("occlusion_results") is not None and not nec.necessity_non_degeneracy_pass:
        mandatory_failed = True
        overall = "failed"
        warnings.append("mandatory_gate:occlusion_non_degeneracy")

    row = {
        "model_id": mapping.get("canonical_model_id"),
        "model_family": mapping.get("model_family"),
        "model_type": mapping.get("model_type"),
        "technical_indicators": mapping.get("technical_indicators"),
        "source_path": str(path),
        "validation_start": config.validation_start.date().isoformat(),
        "validation_end": config.validation_end.date().isoformat(),
        "n_attribution_observations": contrib.n_attribution_observations,
        "n_occlusion_samples": nec.n_occlusion_samples,
        "n_latent_samples": struct.n_latent_samples,
        "contribution_rank_stability": contrib.contribution_rank_stability,
        "contribution_sign_stability": contrib.contribution_sign_stability,
        "necessity_temporal_profile_stability": nec.necessity_temporal_profile_stability,
        "necessity_temporal_instability": nec.necessity_temporal_instability,
        "necessity_normalized_occlusion_effect": nec.necessity_normalized_occlusion_effect,
        "necessity_non_degeneracy_pass": nec.necessity_non_degeneracy_pass,
        "structure_effective_rank_ratio": struct.structure_effective_rank_ratio,
        "structure_probe_margin": struct.structure_probe_margin,
        "structure_probe_cv_std": struct.structure_probe_cv_std,
        "contribution_validity_status": contrib.validity_status,
        "necessity_validity_status": nec.validity_status,
        "structure_validity_status": struct.validity_status,
        "overall_raw_validity_status": overall,
        "mandatory_validity_status": "failed" if mandatory_failed else "passed",
        "temporal_status": temporal_status,
        "warnings": "; ".join(dict.fromkeys(warnings)),
        "_feature_names": to_numpy(names).reshape(-1).astype(str).tolist() if names is not None else [],
    }

    del payload
    gc.collect()
    return row, descriptive_rows, warnings


def inspect_inventory(config: AuditConfig) -> Dict[str, Any]:
    """Inventory PKLs and mappings without scoring."""
    root = Path(config.diagnostic_pkl_root)
    out_dir = ensure_output_dir(config.output_dir, force=config.force)
    paths = discover_diagnostic_pkls(root)
    families = load_expected_families(config.metrics_dir)
    catalog = load_model_cost_catalog(config.metrics_dir)
    mapping_report = load_family_mapping_report(config.family_mapping_report)
    exp_map = build_experiment_to_family(mapping_report)
    kept, dup_rows = deduplicate_paths(paths, root, config.prefer_complete_over_partial)

    assessment_index = {}
    if config.load_model_assessment:
        assessment_index = load_assessment_index(config.model_assessment_pickle)

    mapping_rows = list(dup_rows)
    inventory_rows = []
    for path in kept:
        # lightweight mapping without loading payload metadata first
        m = map_one_pkl(
            path,
            root,
            expected_families=families,
            model_catalog=catalog,
            experiment_to_family=exp_map,
            assessment_index=assessment_index,
        )
        mapping_rows.append(m)
        inventory_rows.append(inventory_one(path, root, m.__dict__))

    inv = pd.DataFrame(inventory_rows)
    maps = mappings_to_frame(mapping_rows)
    write_table(inv, out_dir / "audit_pkl_inventory.csv", also_parquet=True)
    write_table(maps, out_dir / "audit_model_mapping.csv", also_parquet=True)
    return {"inventory": inv, "mapping": maps, "expected_families": families}


def run_batch_a(config: AuditConfig) -> Dict[str, Any]:
    """Execute the full Batch A audit scoring workflow."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = Path(config.diagnostic_pkl_root)
    out_dir = ensure_output_dir(config.output_dir, force=True if config.force else True)
    # Always writable generated outputs under audit_screening; never overwrite source PKLs/metrics.

    paths = discover_diagnostic_pkls(root)
    LOGGER.info("Discovered %d diagnostic PKLs under %s", len(paths), root)
    families = load_expected_families(config.metrics_dir)
    if len(families) != EXPECTED_N_FAMILIES:
        LOGGER.warning("Expected %d families, found %d", EXPECTED_N_FAMILIES, len(families))

    catalog = load_model_cost_catalog(config.metrics_dir)
    mapping_report = load_family_mapping_report(config.family_mapping_report)
    exp_map = build_experiment_to_family(mapping_report)
    kept, dup_rows = deduplicate_paths(paths, root, config.prefer_complete_over_partial)

    assessment_index = {}
    if config.load_model_assessment:
        assessment_index = load_assessment_index(config.model_assessment_pickle)

    mapping_rows = list(dup_rows)
    inventory_rows: List[Dict[str, Any]] = []
    # Inventory every discovered PKL (including duplicates) for Phase 0 completeness.
    for path in paths:
        prelim = map_one_pkl(
            path,
            root,
            expected_families=families,
            model_catalog=catalog,
            experiment_to_family=exp_map,
            assessment_index=assessment_index,
        )
        # Mark duplicates explicitly in inventory notes when superseded.
        inv = inventory_one(path, root, prelim.__dict__)
        if any(str(d.source_path) == str(path) for d in dup_rows):
            inv["mapping_status"] = "duplicate"
            inv["notes"] = (inv.get("notes") or "") + "; duplicate_excluded_from_scoring"
        inventory_rows.append(inv)

    raw_rows: List[Dict[str, Any]] = []
    descriptive_rows: List[Dict[str, Any]] = []
    all_feature_names: List[str] = []
    unresolved_paths: List[str] = []

    for path in kept:
        m = map_one_pkl(
            path,
            root,
            expected_families=families,
            model_catalog=catalog,
            experiment_to_family=exp_map,
            assessment_index=assessment_index,
        )
        mapping_rows.append(m)
        mdict = m.__dict__
        if m.mapping_status != "resolved":
            unresolved_paths.append(str(path))
            continue

        row, desc, _ = extract_model_diagnostics(path, mdict, config)
        feats = row.pop("_feature_names", [])
        all_feature_names.extend(feats)
        # Enrich matching inventory row with extracted flags
        for inv in inventory_rows:
            if inv["source_path"] == str(path):
                inv["canonical_model_id"] = m.canonical_model_id
                inv["model_family"] = m.model_family
                inv["mapping_status"] = m.mapping_status
                inv["mapping_method"] = m.mapping_method
                inv["has_contribution_inputs"] = row.get("n_attribution_observations", 0) > 0
                inv["has_occlusion_results"] = row.get("n_occlusion_samples", 0) > 0
                inv["has_rolling_stability"] = bool(
                    np.isfinite(row.get("necessity_temporal_profile_stability", np.nan))
                )
                inv["has_latent_representations"] = row.get("n_latent_samples", 0) > 0
                inv["has_linear_probe_results"] = bool(
                    np.isfinite(row.get("structure_probe_margin", np.nan))
                )
                inv["notes"] = row.get("warnings", inv.get("notes", ""))
                break
        raw_rows.append(row)
        descriptive_rows.extend(desc)
        LOGGER.info("Scored diagnostics for %s (%s)", m.canonical_model_id, m.model_family)

    inv_df = pd.DataFrame(inventory_rows)
    map_df = mappings_to_frame(mapping_rows)
    raw_df = pd.DataFrame(raw_rows)
    desc_df = pd.DataFrame(descriptive_rows)
    taxonomy = build_taxonomy(sorted(set(all_feature_names)))

    # Expected model counts: family-checkpoint diagnostics => expected 1 per available family unit
    catalog_counts = expected_models_per_family(mapping_report, families)
    if not raw_df.empty and raw_df["model_id"].astype(str).str.contains("family_checkpoint").all():
        expected_counts = {f: 1 for f in families}
        coverage_mode = "family_checkpoint_diagnostic_units"
    else:
        expected_counts = catalog_counts
        coverage_mode = "individual_model_catalog"

    if not raw_df.empty:
        model_scored = score_model_metrics(raw_df)
    else:
        model_scored = pd.DataFrame()

    family_raw = aggregate_family_raw(raw_df, families, expected_counts)
    weights = config.primary_weights()
    family_scored = score_family_medians(family_raw, weights=weights)

    # Merge mandatory validity from models to families
    if not raw_df.empty and "mandatory_validity_status" in raw_df.columns:
        gate = (
            raw_df.groupby("model_family")["mandatory_validity_status"]
            .apply(lambda s: "failed" if (s.astype(str) == "failed").any() else "passed")
            .rename("mandatory_validity_status")
        )
        family_scored = family_scored.merge(gate, left_on="model_family", right_index=True, how="left")
    else:
        family_scored["mandatory_validity_status"] = "passed"

    diag_thr = compute_diagnostic_thresholds(family_scored)
    perf = load_validation_performance(config.metrics_dir)
    # Restrict to expected families (exclude benchmarks)
    perf = perf[perf["model_family"].isin(families)].copy()
    perf_banded, perf_thr = assign_performance_bands(perf)
    master, primary = build_validation_master(
        family_scored,
        perf_banded,
        diag_thr,
        coverage_ratio_threshold=config.coverage_ratio_threshold,
        primary_cost_bps=config.primary_cost_bps,
    )

    # Persist outputs
    write_table(inv_df, out_dir / "audit_pkl_inventory.csv", also_parquet=True)
    write_table(map_df, out_dir / "audit_model_mapping.csv", also_parquet=True)
    write_table(taxonomy, out_dir / "audit_feature_taxonomy.csv", also_parquet=True)
    write_table(raw_df.drop(columns=[c for c in raw_df.columns if c.startswith("_")], errors="ignore"),
                out_dir / "audit_model_diagnostics_raw.csv", also_parquet=True)
    write_table(model_scored, out_dir / "audit_model_diagnostics_scored.csv", also_parquet=True)
    write_table(desc_df, out_dir / "audit_model_diagnostics_descriptive.csv", also_parquet=True)
    write_table(family_raw, out_dir / "audit_family_diagnostics_raw.csv", also_parquet=True)
    write_table(family_scored, out_dir / "audit_family_diagnostics_scored.csv", also_parquet=True)
    write_table(master, out_dir / "audit_validation_master_all_costs.csv", also_parquet=True)
    write_table(primary, out_dir / f"audit_validation_decisions_primary_{config.primary_cost_bps}bps.csv", also_parquet=True)

    thresholds_payload = {
        "validation_window": {
            "start": config.validation_start.isoformat(),
            "end": config.validation_end.isoformat(),
        },
        "holdout_window_untouched": {
            "start": "2024-05-01",
            "end": "2026-06-30",
        },
        "primary_cost_bps": config.primary_cost_bps,
        "audit_profile": config.audit_profile,
        "weights": weights,
        "coverage_ratio_threshold": config.coverage_ratio_threshold,
        "coverage_mode": coverage_mode,
        "diagnostic_score_thresholds": diag_thr,
        "performance_thresholds_by_cost": perf_thr,
        "bootstrap_replicates": config.bootstrap_replicates,
        "random_seed": config.random_seed,
        "forbidden_holdout_files": [
            "holdout_metrics_family",
            "holdout_metrics_individual",
        ],
    }
    write_json(thresholds_payload, out_dir / "audit_validation_thresholds.json")
    write_yaml(PROFILES, out_dir / "audit_profiles.yaml")
    write_metric_registry(out_dir / "audit_metric_registry.yaml")

    qc = write_qc_report(
        out_dir=out_dir,
        config=config,
        inventory=inv_df,
        mapping=map_df,
        raw=raw_df,
        model_scored=model_scored,
        family_raw=family_raw,
        family_scored=family_scored,
        master=master,
        primary=primary,
        expected_families=families,
        unresolved_paths=unresolved_paths,
        coverage_mode=coverage_mode,
    )
    write_batch_a_report(
        out_dir=out_dir,
        config=config,
        inventory=inv_df,
        mapping=map_df,
        raw=raw_df,
        family_scored=family_scored,
        primary=primary,
        qc=qc,
    )
    write_methodology_tex(out_dir / "unified_audit_protocol_revised.tex")
    if unresolved_paths or (len(families) - raw_df["model_family"].nunique() if not raw_df.empty else len(families)):
        write_missing_inputs(
            out_dir / "BATCH_A_MISSING_INPUTS.md",
            unresolved_paths=unresolved_paths,
            expected_families=families,
            available_families=sorted(raw_df["model_family"].unique()) if not raw_df.empty else [],
            duplicate_rows=map_df[map_df["mapping_status"] == "duplicate"] if not map_df.empty else pd.DataFrame(),
        )

    summary = {
        "pkls_discovered": len(paths),
        "models_resolved": int((map_df["mapping_status"] == "resolved").sum()) if not map_df.empty else 0,
        "models_scored": int((model_scored.get("model_score_status") == "scored").sum()) if not model_scored.empty else 0,
        "unresolved_or_ambiguous": int(map_df["mapping_status"].isin(["unresolved", "ambiguous"]).sum()) if not map_df.empty else 0,
        "families_represented": int(raw_df["model_family"].nunique()) if not raw_df.empty else 0,
        "primary_cost_bps": config.primary_cost_bps,
        "output_dir": str(out_dir),
        "qc": qc,
    }
    LOGGER.info("Batch A complete: %s", summary)
    return summary
