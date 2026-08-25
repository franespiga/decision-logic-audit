"""Batch C end-to-end diagnostic-validity pipeline."""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from .baseline_validity import baselines_payload, run_baseline_concordance
from .checkpoint_loading import (
    DecisionTransformerAdapter,
    discover_family_checkpoints,
    load_validation_pool_from_pkl,
    write_execution_inventory,
)
from .faithfulness_validity import run_deletion_insertion
from .holdout_assessment import INCOMPLETE_EVIDENCE_FAMILY, sha256_file
from .io import ensure_output_dir, write_json, write_table
from .randomization_validity import run_parameter_randomization
from .validation_sampling import (
    assert_validation_dates,
    build_sample_manifest,
    stratified_sample_indices,
)
from .validity_aggregation import build_evidence_overlay, status_sensitivity_table
from .validity_config import MIN_EVALUABLE_FAMILIES, ValidityConfig
from .validity_guards import SourceAccessLog, run_phase1_guards

LOGGER = logging.getLogger("audit_scoring.validity_pipeline")


def _tensor_batch(pool: Mapping[str, Any], indices: np.ndarray, device: torch.device) -> Dict[str, torch.Tensor]:
    def _sel(t):
        if t is None:
            return None
        if not torch.is_tensor(t):
            t = torch.as_tensor(t)
        t = t.detach().to("cpu")
        idx = torch.as_tensor(indices, dtype=torch.long, device="cpu")
        return t.index_select(0, idx).to(device)

    rtg = _sel(pool["returns_to_go"])
    if rtg is not None and rtg.ndim == 2:
        rtg = rtg.unsqueeze(-1)
    out = {
        "states": _sel(pool["states"]).float(),
        "actions": _sel(pool["actions"]).float(),
        "returns_to_go": rtg.float(),
        "timesteps": _sel(pool["timesteps"]).long(),
    }
    attn = pool.get("attention_mask")
    if attn is not None:
        out["attention_mask"] = _sel(attn)
    return out


def _config_hash(cfg: ValidityConfig) -> str:
    payload = json.dumps(cfg.to_dict(), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _family_cache_path(output_dir: Path, family: str) -> Path:
    safe = family.replace("/", "_")
    return output_dir / "intermediates" / f"{safe}__validity.json"


def _save_family_result(path: Path, payload: Mapping[str, Any]) -> None:
    ensure_output_dir(path.parent)
    serializable = {}
    for k, v in payload.items():
        if isinstance(v, (pd.DataFrame, pd.Series)):
            continue
        if isinstance(v, (np.ndarray, torch.Tensor)):
            continue
        serializable[k] = v
    path.write_text(json.dumps(serializable, indent=2, default=str), encoding="utf-8")


def run_inspect_only(cfg: ValidityConfig) -> Dict[str, Any]:
    ensure_output_dir(cfg.output_dir)
    log = SourceAccessLog()
    phase1 = run_phase1_guards(cfg, log)
    records = discover_family_checkpoints(cfg)
    inv = write_execution_inventory(records, cfg.output_dir / "audit_validity_execution_inventory.csv")
    missing_lines = ["# Batch C missing inputs", ""]
    for r in records:
        problems = []
        if not r.checkpoint_exists:
            problems.append(f"missing checkpoint: {r.checkpoint_path}")
        if not r.pkl_exists:
            problems.append(f"missing diagnostic PKL for experiment {r.experiment_name}")
        if problems:
            r.loading_status = "missing_inputs"
            missing_lines.append(f"## {r.model_family}")
            for p in problems:
                missing_lines.append(f"- {p}")
            missing_lines.append("")
    if len(missing_lines) > 2:
        (cfg.output_dir / "BATCH_C_MISSING_INPUTS.md").write_text("\n".join(missing_lines), encoding="utf-8")
    write_table(log.to_frame(), cfg.output_dir / "audit_validity_source_lineage.csv")
    return {"phase1": phase1, "inventory": inv, "n_records": len(records), "source_violations": log.violations}


def evaluate_family(
    cfg: ValidityConfig,
    record,
    adapter: DecisionTransformerAdapter,
    log: SourceAccessLog,
) -> Dict[str, Any]:
    family = record.model_family
    notes: List[str] = []
    if record.is_incomplete_family:
        return {
            "model_family": family,
            "baseline_status": "INCOMPLETE",
            "randomization_status": "INCOMPLETE",
            "deletion_status": "INCOMPLETE",
            "insertion_status": "INCOMPLETE",
            "notes": "Preserved INCOMPLETE for frozen incomplete-evidence family.",
            "skipped_heavy": True,
            "tables": {},
        }

    if not record.checkpoint_exists or not record.pkl_exists or record.diagnostic_pkl is None:
        return {
            "model_family": family,
            "baseline_status": "INCOMPLETE",
            "randomization_status": "INCOMPLETE",
            "deletion_status": "INCOMPLETE",
            "insertion_status": "INCOMPLETE",
            "notes": "Missing checkpoint or diagnostic PKL.",
            "tables": {},
        }

    log.record(
        record.diagnostic_pkl,
        source_role="diagnostic_pkl",
        validation_or_holdout="validation",
        loaded=True,
        permitted=True,
        notes="Validation tensors only.",
    )
    pool = load_validation_pool_from_pkl(
        record.diagnostic_pkl,
        validation_start=cfg.validation_start,
        validation_end=cfg.validation_end,
    )
    dates = pool["dates"]
    primary_idx, labels, primary_rule = stratified_sample_indices(
        dates,
        n_samples=cfg.primary_samples,
        per_regime=cfg.primary_per_regime,
        random_seed=cfg.random_seed,
    )
    # Randomization subset: prefer subset of primary for consistency when possible
    rand_idx, _, rand_rule = stratified_sample_indices(
        dates,
        n_samples=cfg.randomization_samples,
        per_regime=cfg.randomization_per_regime,
        random_seed=cfg.random_seed + 1,
    )
    # Prefer overlap with primary
    overlap = np.intersect1d(primary_idx, rand_idx)
    if len(overlap) >= min(cfg.randomization_samples, len(primary_idx) // 2):
        # take first randomization_samples from primary in deterministic order
        rng = np.random.default_rng(cfg.random_seed)
        if len(primary_idx) >= cfg.randomization_samples:
            # stratified within primary
            prim_dates = dates[primary_idx]
            local, _, rand_rule = stratified_sample_indices(
                prim_dates,
                n_samples=cfg.randomization_samples,
                per_regime=cfg.randomization_per_regime,
                random_seed=cfg.random_seed + 1,
            )
            rand_idx = primary_idx[local]
            rand_rule = "subset_of_primary+" + rand_rule
        else:
            rand_idx = primary_idx
    bg_idx, _, bg_rule = stratified_sample_indices(
        dates,
        n_samples=cfg.background_samples,
        per_regime=cfg.background_per_regime,
        random_seed=cfg.random_seed + 2,
    )

    assert_validation_dates(dates[primary_idx], cfg.validation_end)
    assert_validation_dates(dates[rand_idx], cfg.validation_end)
    assert_validation_dates(dates[bg_idx], cfg.validation_end)

    manifest = build_sample_manifest(
        family,
        dates,
        primary_idx,
        rand_idx,
        bg_idx,
        labels=labels,
        random_seed=cfg.random_seed,
        primary_rule=primary_rule,
        randomization_rule=rand_rule,
        background_rule=bg_rule,
    )

    device = adapter.device
    pool_states = pool["states"]
    if not torch.is_tensor(pool_states):
        pool_states = torch.as_tensor(pool_states)
    pool_states = pool_states.detach().to("cpu").float().to(device)
    primary_batch = _tensor_batch(pool, primary_idx, device)
    rand_batch = _tensor_batch(pool, rand_idx, device)
    bg_idx_t = torch.as_tensor(bg_idx, dtype=torch.long, device="cpu")
    bg_states = (
        pool["states"].detach().to("cpu").float().index_select(0, bg_idx_t).to(device)
        if torch.is_tensor(pool["states"])
        else torch.as_tensor(pool["states"]).float().index_select(0, bg_idx_t).to(device)
    )
    family_ids = pool["family_ids"]
    cash_idx = int(pool["cash_idx"])

    adapter.load_fresh_copy(record.checkpoint_path)
    log.record(
        record.checkpoint_path,
        source_role="trained_checkpoint",
        validation_or_holdout="validation",
        loaded=True,
        permitted=True,
        notes="Loaded read-only via from_pretrained.",
    )

    baseline_result = run_baseline_concordance(
        adapter,
        primary_batch,
        pool_states,
        bg_states,
        family_ids,
        model_family=family,
        n_steps=cfg.ig_steps,
        batch_size=cfg.attribution_batch_size,
        top_k=cfg.top_k_families,
    )
    baseline_ref = float(baseline_result["metrics"].get("median_kendall_tau_b", float("nan")))

    rand_result = run_parameter_randomization(
        adapter,
        record.checkpoint_path,
        rand_batch,
        pool_states,
        family_ids,
        baseline_result["abs_profiles"]["empirical_median"],
        baseline_result["signed_profiles"]["empirical_median"],
        model_family=family,
        seeds=cfg.randomization_seeds,
        n_steps=cfg.ig_steps,
        batch_size=cfg.attribution_batch_size,
        top_k=cfg.top_k_families,
        cash_idx=cash_idx,
        baseline_reference_concordance=baseline_ref,
    )

    adapter.load_fresh_copy(record.checkpoint_path)
    faith = run_deletion_insertion(
        adapter,
        primary_batch,
        pool_states,
        family_ids,
        baseline_result["abs_profiles"]["empirical_median"],
        model_family=family,
        max_depth=cfg.max_deletion_depth,
        random_control_repeats=cfg.random_control_repeats,
        bootstrap_replicates=cfg.bootstrap_replicates,
        random_seed=cfg.random_seed,
        cash_idx=cash_idx,
        inference_batch_size=cfg.inference_batch_size,
    )

    # baselines json fragment
    base_def = baselines_payload(
        zero=baseline_result["zero_baseline"],
        empirical_median=baseline_result["empirical_median_baseline"],
        background_states=bg_states,
        background_indices=bg_idx.tolist(),
    )

    adapter.release()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model_family": family,
        "baseline_status": baseline_result["status"],
        "baseline_median_tau": baseline_ref,
        "randomization_status": rand_result["status"],
        "full_randomization_similarity": rand_result["metrics"].get("full_randomization_similarity"),
        "randomization_gap": rand_result["metrics"].get("randomization_gap"),
        "deletion_status": faith["deletion_status"],
        "deletion_gap": faith["metrics"].get("deletion_gap"),
        "deletion_ci_lower": faith["metrics"].get("deletion_ci_lower"),
        "insertion_status": faith["insertion_status"],
        "insertion_gap": faith["metrics"].get("insertion_gap"),
        "insertion_ci_lower": faith["metrics"].get("insertion_ci_lower"),
        "notes": "; ".join(notes),
        "sample_manifest": manifest,
        "baseline_definitions": base_def,
        "tables": {
            "baseline_per_family": baseline_result["per_family"],
            "baseline_pairwise": baseline_result["pairwise"],
            "baseline_sensitivity": baseline_result["sensitivity"],
            "randomization_per_family": rand_result["per_family"],
            "randomization_by_level_seed": rand_result["by_level_seed"],
            "randomization_progression": rand_result["progression"],
            "deletion_per_family": faith["deletion_per_family"],
            "deletion_curves": faith["deletion_curves"],
            "deletion_random": faith["deletion_random"],
            "insertion_per_family": faith["insertion_per_family"],
            "insertion_curves": faith["insertion_curves"],
            "insertion_random": faith["insertion_random"],
            "action_effects": faith["action_effects"],
        },
        "bootstrap_replicates_used": cfg.bootstrap_replicates,
        "bootstrap_replicates_configured": cfg.bootstrap_replicates,
    }


def run_batch_c(cfg: ValidityConfig) -> Dict[str, Any]:
    ensure_output_dir(cfg.output_dir)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.time()
    log = SourceAccessLog()
    phase1 = run_phase1_guards(cfg, log)
    if not phase1["scores"]["ok"] and not cfg.force:
        raise RuntimeError("Batch A freeze verification failed; refuse to continue without --force")

    records = discover_family_checkpoints(cfg)
    if cfg.families:
        allow = set(cfg.families)
        records = [r for r in records if r.model_family in allow]
    inv = write_execution_inventory(records, cfg.output_dir / "audit_validity_execution_inventory.csv")

    if cfg.inspect_only:
        return run_inspect_only(cfg)

    adapter = DecisionTransformerAdapter(cfg)
    family_results: Dict[str, Dict[str, Any]] = {}
    exec_log_rows: List[Dict[str, Any]] = []
    all_tables: Dict[str, List[pd.DataFrame]] = {
        "baseline_per_family": [],
        "baseline_pairwise": [],
        "baseline_sensitivity": [],
        "randomization_per_family": [],
        "randomization_by_level_seed": [],
        "randomization_progression": [],
        "deletion_per_family": [],
        "deletion_curves": [],
        "deletion_random": [],
        "insertion_per_family": [],
        "insertion_curves": [],
        "insertion_random": [],
        "action_effects": [],
        "sample_manifest": [],
    }
    baselines_all: Dict[str, Any] = {}
    missing_md: List[str] = ["# Batch C missing inputs", ""]

    cfg_hash = _config_hash(cfg)
    evaluable = 0

    for record in records:
        family = record.model_family
        cache = _family_cache_path(cfg.output_dir, family)
        started = time.time()
        status = "pending"
        err = ""
        try:
            if cfg.resume and cache.exists() and not cfg.force:
                cached = json.loads(cache.read_text(encoding="utf-8"))
                if cached.get("config_hash") == cfg_hash:
                    LOGGER.info("Resume cache hit for %s", family)
                    family_results[family] = cached
                    status = "resumed"
                    # Reload table shards if present
                    shard_dir = cfg.output_dir / "intermediates" / family
                    for key in list(all_tables.keys()):
                        shard = shard_dir / f"{key}.csv"
                        if shard.exists():
                            all_tables[key].append(pd.read_csv(shard))
                    if cached.get("baseline_status") not in {None, "INCOMPLETE"} or family == INCOMPLETE_EVIDENCE_FAMILY:
                        evaluable += 0 if family == INCOMPLETE_EVIDENCE_FAMILY else 1
                    exec_log_rows.append(
                        {
                            "model_family": family,
                            "status": status,
                            "elapsed_sec": time.time() - started,
                            "error": "",
                        }
                    )
                    continue

            LOGGER.info("Evaluating family %s", family)
            result = evaluate_family(cfg, record, adapter, log)
            result["config_hash"] = cfg_hash
            family_results[family] = result
            status = "completed"
            if result.get("baseline_status") != "INCOMPLETE" or family == INCOMPLETE_EVIDENCE_FAMILY:
                if family != INCOMPLETE_EVIDENCE_FAMILY and result.get("baseline_status") != "INCOMPLETE":
                    evaluable += 1
            if result.get("sample_manifest") is not None:
                all_tables["sample_manifest"].append(result["sample_manifest"])
            for key, df in result.get("tables", {}).items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    all_tables[key].append(df)
                    shard_dir = cfg.output_dir / "intermediates" / family
                    ensure_output_dir(shard_dir)
                    write_table(df, shard_dir / f"{key}.csv")
            if result.get("baseline_definitions"):
                baselines_all[family] = result["baseline_definitions"]
            _save_family_result(cache, result)
            if result.get("baseline_status") == "INCOMPLETE" and family != INCOMPLETE_EVIDENCE_FAMILY:
                missing_md.append(f"## {family}")
                missing_md.append(f"- {result.get('notes')}")
                missing_md.append("")
        except Exception as exc:
            status = "failed"
            err = f"{exc}\n{traceback.format_exc()}"
            LOGGER.exception("Family %s failed", family)
            family_results[family] = {
                "model_family": family,
                "baseline_status": "INCOMPLETE",
                "randomization_status": "INCOMPLETE",
                "deletion_status": "INCOMPLETE",
                "insertion_status": "INCOMPLETE",
                "notes": str(exc),
                "config_hash": cfg_hash,
            }
            missing_md.append(f"## {family}")
            missing_md.append(f"- exception: {exc}")
            missing_md.append("")
            _save_family_result(cache, family_results[family])
        exec_log_rows.append(
            {
                "model_family": family,
                "status": status,
                "elapsed_sec": time.time() - started,
                "error": err[:2000],
            }
        )

    if evaluable < MIN_EVALUABLE_FAMILIES and len(records) >= EXPECTED_HINT(cfg):
        LOGGER.error(
            "Fewer than %s checkpoints evaluated successfully (%s). Aggregate conclusions suppressed.",
            MIN_EVALUABLE_FAMILIES,
            evaluable,
        )

    # Concatenate tables
    def _cat(key: str, path_name: str) -> pd.DataFrame:
        frames = all_tables.get(key) or []
        if not frames:
            df = pd.DataFrame()
        else:
            df = pd.concat(frames, ignore_index=True)
        write_table(df, cfg.output_dir / path_name)
        return df

    sample_manifest = _cat("sample_manifest", "audit_validity_sample_manifest.csv")
    baseline_per = _cat("baseline_per_family", "audit_baseline_concordance_per_family.csv")
    baseline_pair = _cat("baseline_pairwise", "audit_baseline_concordance_pairwise.csv")
    baseline_sens = _cat("baseline_sensitivity", "audit_baseline_concordance_sensitivity.csv")
    rand_per = _cat("randomization_per_family", "audit_randomization_per_family.csv")
    rand_seed = _cat("randomization_by_level_seed", "audit_randomization_by_level_seed.csv")
    rand_prog = _cat("randomization_progression", "audit_randomization_progression.csv")
    del_per = _cat("deletion_per_family", "audit_deletion_per_family.csv")
    del_curves = _cat("deletion_curves", "audit_deletion_curves.csv")
    del_rand = _cat("deletion_random", "audit_deletion_random_controls.csv")
    ins_per = _cat("insertion_per_family", "audit_insertion_per_family.csv")
    ins_curves = _cat("insertion_curves", "audit_insertion_curves.csv")
    ins_rand = _cat("insertion_random", "audit_insertion_random_controls.csv")
    action_fx = _cat("action_effects", "audit_validity_portfolio_action_effects.csv")

    write_json(baselines_all, cfg.output_dir / "audit_validity_baselines.json")
    write_table(pd.DataFrame(exec_log_rows), cfg.output_dir / "audit_validity_execution_log.csv")
    write_table(log.to_frame(), cfg.output_dir / "audit_validity_source_lineage.csv")
    if len(missing_md) > 2:
        (cfg.output_dir / "BATCH_C_MISSING_INPUTS.md").write_text("\n".join(missing_md), encoding="utf-8")

    # Load Batch A overlay inputs
    batch_a_primary = pd.read_csv(cfg.batch_a_dir / "audit_validation_decisions_primary_5bps.csv")
    raw_path = cfg.batch_a_dir / "audit_model_diagnostics_raw.csv"
    batch_a_raw = pd.read_csv(raw_path) if raw_path.exists() else None
    log.record(
        raw_path if raw_path.exists() else cfg.batch_a_dir / "audit_validation_decisions_primary_5bps.csv",
        source_role="batch_a_overlay_metadata",
        validation_or_holdout="validation",
        loaded=True,
        permitted=True,
        notes="Batch A scores retained unchanged.",
    )

    # Ensure all 18 families present in results
    for fam in batch_a_primary["model_family"].astype(str):
        if fam not in family_results:
            family_results[fam] = {
                "model_family": fam,
                "baseline_status": "INCOMPLETE",
                "randomization_status": "INCOMPLETE",
                "deletion_status": "INCOMPLETE",
                "insertion_status": "INCOMPLETE",
                "notes": "Not evaluated in this run.",
            }

    overlay = build_evidence_overlay(batch_a_primary, batch_a_raw, family_results)
    write_table(overlay, cfg.output_dir / "audit_validity_evidence_overlay.csv")
    combined = overlay[
        [
            "model_family",
            "baseline_concordance_status",
            "randomization_status",
            "deletion_status",
            "insertion_status",
            "combined_validity_status",
            "combined_validity_reason",
            "frozen_screening_decision",
            "evidence_complete",
        ]
    ].copy()
    write_table(combined, cfg.output_dir / "audit_validity_combined_status.csv")
    sens = status_sensitivity_table(overlay)
    write_table(sens, cfg.output_dir / "audit_validity_status_sensitivity.csv")

    # Import reporting lazily to avoid circular imports at module load for tests
    from .validity_reporting import finalize_batch_c_outputs

    summary = finalize_batch_c_outputs(
        cfg,
        phase1=phase1,
        inventory=inv,
        overlay=overlay,
        combined=combined,
        sensitivity=sens,
        tables={
            "baseline_per_family": baseline_per,
            "baseline_pairwise": baseline_pair,
            "baseline_sensitivity": baseline_sens,
            "randomization_per_family": rand_per,
            "randomization_by_level_seed": rand_seed,
            "randomization_progression": rand_prog,
            "deletion_per_family": del_per,
            "deletion_curves": del_curves,
            "insertion_per_family": ins_per,
            "insertion_curves": ins_curves,
            "action_effects": action_fx,
            "sample_manifest": sample_manifest,
        },
        source_log=log,
        family_results=family_results,
        baselines_all=baselines_all,
        elapsed_sec=time.time() - t0,
        evaluable=evaluable,
    )
    return summary


def EXPECTED_HINT(cfg: ValidityConfig) -> int:
    return 18 if not cfg.families else len(cfg.families)
