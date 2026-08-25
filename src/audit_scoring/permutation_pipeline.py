"""Batch C3 input-permutation orchestration pipeline."""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .baseline_validity import (
    build_empirical_median_baseline,
    ranking_from_abs_profile,
)
from .checkpoint_loading import (
    DecisionTransformerAdapter,
    discover_family_checkpoints,
    load_validation_pool_from_pkl,
)
from .holdout_assessment import (
    INCOMPLETE_EVIDENCE_FAMILY,
    sha256_file,
    verify_batch_a_freeze,
)
from .io import ensure_output_dir, write_json, write_table
from .permutation_config import ABOVE_MARKET_FAMILIES, MANDATORY_FAMILY, PermutationConfig
from .permutation_controls import run_all_state_control, run_identity_control
from .permutation_operator import build_regime_donor_map, permute_state_families, valid_permutation_mask
from .permutation_validity import (
    combined_permutation_status,
    global_permutation_attribution,
    guided_vs_random_permutation,
    importance_effect_concordance,
    importance_effect_sensitivity,
    single_family_permutation_effects,
)
from .perturbation_plausibility import assess_plausibility
from .targeted_validity import (
    TARGET_CASH,
    TARGET_L2_EQUAL,
    TARGET_VECTOR_L1,
    attribution_matrix_from_tensor,
    compute_ig_with_target,
    profiles_from_attr,
)
from .validation_sampling import assert_validation_dates, assign_regimes, stratified_sample_indices
from .validity_config import ValidityConfig, resolve_device
from .validity_guards import SourceAccessLog, load_permitted

LOGGER = logging.getLogger("audit_scoring.permutation_pipeline")


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
        "returns_to_go": rtg.float() if rtg is not None else None,
        "timesteps": _sel(pool["timesteps"]).long() if pool.get("timesteps") is not None else None,
        "attention_mask": _sel(pool["attention_mask"]).float() if pool.get("attention_mask") is not None else None,
    }
    return {k: v for k, v in out.items() if v is not None}


def _snapshot_hashes(directory: Path, *, skip_intermediates: bool = True) -> Dict[str, str]:
    out = {}
    if not Path(directory).exists():
        return out
    for p in sorted(Path(directory).rglob("*")):
        if not p.is_file():
            continue
        if skip_intermediates and "intermediates" in str(p):
            continue
        if p.suffix.lower() in {".csv", ".json", ".parquet", ".md", ".tex", ".png", ".pdf"}:
            out[str(p)] = sha256_file(p)
    return out


def _read_verification_verdict(path: Path, marker: str) -> str:
    if not path.exists():
        return "UNKNOWN"
    text = path.read_text(encoding="utf-8", errors="ignore")
    upper = text.upper()
    if "VERIFIED WITH METHODOLOGICAL QUALIFICATIONS" in upper:
        return "VERIFIED WITH METHODOLOGICAL QUALIFICATIONS"
    if "VERIFIED WITH WARNINGS" in upper:
        return "VERIFIED WITH WARNINGS"
    if "VERIFIED" in upper:
        return "VERIFIED"
    if "FAILED" in upper:
        return "FAILED"
    return "UNKNOWN"


def verify_frozen_priors(cfg: PermutationConfig, log: SourceAccessLog) -> Dict[str, Any]:
    freeze_a = verify_batch_a_freeze(cfg.batch_a_dir, repo_root=cfg.repository_root)
    log.record(
        cfg.batch_a_dir / "BATCH_A_FROZEN_MANIFEST.json",
        source_role="batch_a_frozen_manifest",
        validation_or_holdout="validation",
        loaded=True,
        permitted=True,
        notes="Hash verification only.",
    )
    c_manifest = cfg.batch_c_dir / "BATCH_C_VALIDITY_FROZEN_MANIFEST.json"
    c2_manifest = cfg.batch_c2_dir / "BATCH_C2_FROZEN_MANIFEST.json"
    c_ok = c_manifest.exists()
    c2_ok = c2_manifest.exists()
    if c_ok:
        log.record(c_manifest, source_role="batch_c_frozen_manifest", validation_or_holdout="validation", loaded=True, permitted=True)
    if c2_ok:
        log.record(c2_manifest, source_role="batch_c2_frozen_manifest", validation_or_holdout="validation", loaded=True, permitted=True)

    # Record Batch B only as provenance (do not load outcome tables)
    log.record(
        cfg.batch_b_dir,
        source_role="batch_b_dir_provenance",
        validation_or_holdout="provenance",
        loaded=False,
        permitted=True,
        notes="Referenced for provenance only; outcome tables not loaded.",
    )

    c_verdict = _read_verification_verdict(
        cfg.batch_c_verification_dir / "BATCH_C_VERIFICATION_REPORT.md", "C"
    )
    c2_verdict = _read_verification_verdict(
        cfg.batch_c2_verification_dir / "BATCH_C2_VERIFICATION_REPORT.md", "C2"
    )
    snap_a = _snapshot_hashes(cfg.batch_a_dir)
    snap_c = _snapshot_hashes(cfg.batch_c_dir)
    snap_c2 = _snapshot_hashes(cfg.batch_c2_dir)
    return {
        "batch_a_freeze_ok": bool(freeze_a.ok),
        "batch_a_changed": freeze_a.changed_files,
        "batch_c_manifest_exists": c_ok,
        "batch_c2_manifest_exists": c2_ok,
        "batch_c_verdict": c_verdict,
        "batch_c2_verdict": c2_verdict,
        "snapshot_a": snap_a,
        "snapshot_c": snap_c,
        "snapshot_c2": snap_c2,
        "expected_c_verdict": "VERIFIED WITH METHODOLOGICAL QUALIFICATIONS",
        "expected_c2_verdict": "VERIFIED WITH WARNINGS",
    }


def _checkpoint_hash(path: Path) -> str:
    path = Path(path)
    if path.is_file():
        return sha256_file(path)
    if path.is_dir():
        cands = sorted(path.glob("*.bin")) + sorted(path.glob("*.safetensors"))
        if cands:
            return sha256_file(cands[0])
    return ""


def build_execution_inventory(
    cfg: PermutationConfig,
    records: Sequence[Any],
    log: SourceAccessLog,
) -> pd.DataFrame:
    sample_path = cfg.batch_c_dir / "audit_validity_sample_manifest.csv"
    sample_df = None
    if sample_path.exists():
        sample_df = load_permitted(
            sample_path,
            role="batch_c_sample_manifest",
            log=log,
            loader=pd.read_csv,
        )
    rows = []
    for rec in records:
        fam = rec.model_family
        n_samp = 0
        src = "unavailable"
        if sample_df is not None:
            sub = sample_df[(sample_df.model_family == fam)]
            if "sample_role" in sub.columns:
                sub = sub[sub.sample_role == "primary"]
            n_samp = int(sub["observation_index"].nunique()) if not sub.empty else 0
            src = "batch_c_sample_manifest" if n_samp else "reconstruct"
        complete = fam != INCOMPLETE_EVIDENCE_FAMILY and bool(rec.pkl_exists) and bool(rec.checkpoint_exists)
        rows.append(
            {
                "model_family": fam,
                "checkpoint_path": str(rec.checkpoint_path),
                "checkpoint_hash": _checkpoint_hash(rec.checkpoint_path) if rec.checkpoint_exists else "",
                "checkpoint_exists": bool(rec.checkpoint_exists),
                "diagnostic_complete": bool(complete),
                "ti_flag": bool(str(fam).endswith("_TI_True")),
                "validation_sample_source": src if complete else "incomplete",
                "validation_sample_count": n_samp if complete else 0,
                "feature_count": "",
                "feature_family_count": "",
                "regime_source": "validation_dates",
                "cash_index": "",
                "model_loading_status": "pending" if complete else "INCOMPLETE",
                "notes": "" if complete else "Prespecified incomplete-evidence family or missing inputs.",
            }
        )
    return pd.DataFrame(rows)


def _ig_once(
    adapter: DecisionTransformerAdapter,
    batch: Mapping[str, torch.Tensor],
    baseline: torch.Tensor,
    *,
    target_name: str,
    n_steps: int,
    cash_idx: int,
    batch_size: int,
) -> Tuple[pd.Series, pd.Series, np.ndarray]:
    n = batch["states"].shape[0]
    mats = []
    for start in range(0, n, batch_size):
        sl = slice(start, min(n, start + batch_size))
        sub = {k: v[sl] for k, v in batch.items() if torch.is_tensor(v)}
        base = baseline[sl] if baseline.shape[0] == n else baseline
        attr, _ = compute_ig_with_target(
            adapter, sub, base, target_name=target_name, n_steps=n_steps, cash_idx=cash_idx
        )
        mats.append(attribution_matrix_from_tensor(attr))
    mat = np.concatenate(mats, axis=0)
    return mat


def evaluate_family_c3(
    cfg: PermutationConfig,
    record,
    adapter: DecisionTransformerAdapter,
    *,
    log: SourceAccessLog,
    batch_c_status: str,
    batch_c2_status: str,
    screening_decision: str,
) -> Dict[str, Any]:
    family = record.model_family
    if family == INCOMPLETE_EVIDENCE_FAMILY or not record.pkl_exists or not record.checkpoint_exists:
        return {
            "model_family": family,
            "incomplete": True,
            "permutation_status": "INCOMPLETE",
            "ti_flag": bool(str(family).endswith("_TI_True")),
            "batch_c_status": batch_c_status,
            "batch_c2_status": batch_c2_status,
            "screening_decision": screening_decision,
            "notes": "Incomplete-evidence family or missing checkpoint/pkl.",
        }

    log.record(record.diagnostic_pkl, source_role="diagnostic_pkl", validation_or_holdout="validation", loaded=True, permitted=True)
    pool = load_validation_pool_from_pkl(
        record.diagnostic_pkl,
        validation_start=cfg.validation_start,
        validation_end=cfg.validation_end,
    )
    dates = pool["dates"]
    sample_path = Path(cfg.batch_c_dir) / "audit_validity_sample_manifest.csv"
    primary_idx = None
    reused = False
    sample_notes = ""
    if sample_path.exists():
        man = pd.read_csv(sample_path)
        sub = man[man.model_family == family]
        if "sample_role" in sub.columns:
            sub = sub[sub.sample_role == "primary"]
        if not sub.empty:
            primary_idx = sub["observation_index"].astype(int).to_numpy()
            reused = True
            log.record(sample_path, source_role="batch_c_sample_manifest", validation_or_holdout="validation", loaded=True, permitted=True)
    if primary_idx is None:
        primary_idx, _, sample_notes = stratified_sample_indices(
            dates, n_samples=cfg.primary_samples, per_regime=32, random_seed=cfg.random_seed
        )
        sample_notes = "reconstructed_batch_c_rule:" + sample_notes
    assert_validation_dates(dates[primary_idx], cfg.validation_end)
    regimes = assign_regimes(dates[primary_idx])

    sample_manifest = pd.DataFrame(
        {
            "model_family": family,
            "observation_index": primary_idx,
            "date": pd.to_datetime(dates[primary_idx]).astype(str),
            "regime": regimes,
            "sample_source": "batch_c" if reused else "reconstructed",
            "random_seed": cfg.random_seed,
            "reused_from_batch_c": reused,
            "notes": sample_notes,
        }
    )

    device = adapter.device
    pool_states = (
        pool["states"].detach().to("cpu").float().to(device)
        if torch.is_tensor(pool["states"])
        else torch.as_tensor(pool["states"]).float().to(device)
    )
    batch = _tensor_batch(pool, primary_idx, device)
    family_ids = pool["family_ids"]
    cash_idx = int(pool["cash_idx"])
    feature_families = sorted(set(map(str, family_ids.tolist())))

    ckpt_hash = _checkpoint_hash(record.checkpoint_path)
    adapter.load_fresh_copy(record.checkpoint_path)
    log.record(record.checkpoint_path, source_role="checkpoint", validation_or_holdout="validation", loaded=True, permitted=True)

    median = build_empirical_median_baseline(pool_states, batch["states"])
    # Original IG (L2-equal, empirical-median)
    mat0 = _ig_once(
        adapter,
        batch,
        median,
        target_name=TARGET_L2_EQUAL,
        n_steps=cfg.ig_steps,
        cash_idx=cash_idx,
        batch_size=cfg.attribution_batch_size,
    )
    original_abs, original_signed = profiles_from_attr(mat0, family_ids)
    ranked = ranking_from_abs_profile(original_abs)

    # Donor maps per seed
    assignment_rows = []
    donor_by_seed: Dict[int, np.ndarray] = {}
    for seed in cfg.permutation_seeds:
        donors, assign_df = build_regime_donor_map(regimes, seed=seed)
        donor_by_seed[int(seed)] = donors
        for _, r in assign_df.iterrows():
            ri = int(r["recipient_local_index"])
            di = int(r["donor_local_index"])
            assignment_rows.append(
                {
                    "model_family": family,
                    "seed": int(seed),
                    "regime": r["regime"],
                    "recipient_index": int(primary_idx[ri]),
                    "donor_index": int(primary_idx[di]) if di >= 0 else -1,
                    "recipient_date": str(pd.Timestamp(dates[primary_idx[ri]]).date()),
                    "donor_date": str(pd.Timestamp(dates[primary_idx[di]]).date()) if di >= 0 else "",
                    "same_regime": bool(r["same_regime"]),
                    "is_derangement": bool(r["is_derangement"]),
                    "notes": r.get("notes", ""),
                }
            )
    assignments = pd.DataFrame(assignment_rows)

    # Controls
    id_ctrl = run_identity_control(adapter, batch, model_family=family, cash_idx=cash_idx)
    control_rows = [id_ctrl]
    for seed, donors in donor_by_seed.items():
        control_rows.append(
            run_all_state_control(
                adapter, batch, donors, model_family=family, seed=seed, cash_idx=cash_idx
            )
        )
    controls = pd.DataFrame(control_rows)

    # Single-family effects
    effect_parts = []
    attr_parts = []
    recompute_attr = not cfg.skip_attribution_recompute
    for seed, donors in donor_by_seed.items():
        # Recompute attribution only on first seed if many families (still all seeds for actions)
        do_attr = recompute_attr and (seed == cfg.permutation_seeds[0] or len(feature_families) <= 8)
        eff, attr = single_family_permutation_effects(
            adapter,
            batch,
            donors,
            family_ids,
            feature_families,
            original_abs,
            original_signed,
            median,
            model_family=family,
            seed=seed,
            cash_idx=cash_idx,
            n_steps=cfg.ig_steps,
            batch_size=cfg.attribution_batch_size,
            recompute_attribution=do_attr,
        )
        effect_parts.append(eff)
        if not attr.empty:
            attr_parts.append(attr)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    effects = pd.concat(effect_parts, ignore_index=True) if effect_parts else pd.DataFrame()
    attr_changes = pd.concat(attr_parts, ignore_index=True) if attr_parts else pd.DataFrame()

    conc = importance_effect_concordance(
        effects,
        model_family=family,
        bootstrap_replicates=min(cfg.bootstrap_replicates, 2000),
        random_seed=cfg.random_seed,
    )
    sens = importance_effect_sensitivity(conc)

    # Guided vs random
    guided = guided_vs_random_permutation(
        adapter,
        batch,
        donor_by_seed,
        family_ids,
        ranked,
        model_family=family,
        max_depth=cfg.max_depth,
        random_control_repeats=cfg.random_control_repeats,
        bootstrap_replicates=min(cfg.bootstrap_replicates, 5000),
        random_seed=cfg.random_seed,
        cash_idx=cash_idx,
    )

    # Global permutation attribution (seed 101)
    seed0 = int(cfg.permutation_seeds[0])
    glob = global_permutation_attribution(
        adapter,
        batch,
        donor_by_seed[seed0],
        family_ids,
        original_abs,
        original_signed,
        median,
        model_family=family,
        seed=seed0,
        cash_idx=cash_idx,
        n_steps=cfg.ig_steps,
        batch_size=cfg.attribution_batch_size,
    )

    # State plausibility at focus depth
    focus = min(3, len(ranked))
    donors0 = donor_by_seed[seed0]
    guided_fams = ranked[:focus]
    from .cardinality_controls import family_raw_cardinality, guided_raw_count, sample_cardinality_matched_families

    card = family_raw_cardinality(family_ids)
    target_raw = guided_raw_count(family_ids, guided_fams)
    rng = np.random.default_rng(cfg.random_seed + 99)
    rand_fams, _, _ = sample_cardinality_matched_families(
        ranked, card, guided_families=guided_fams, n_families=focus, target_raw_count=target_raw, rng=rng
    )
    states_g = permute_state_families(batch["states"], donors0, family_ids, guided_fams)
    states_r = permute_state_families(batch["states"], donors0, family_ids, rand_fams)
    plaus = assess_plausibility(
        batch["states"], states_g, states_r, pool_states, model_family=family, depth=focus
    )
    plaus_status = str(plaus.get("perturbed_state_plausibility_status", "MIXED"))
    if float(plaus.get("mean_guided_range_violation", 0) or 0) > 0.25:
        plaus_status = "MATERIAL_RANGE_VIOLATIONS"

    ie_status = conc["importance_effect_status"]
    gp_status = guided["guided_permutation_status"]
    identity_pass = bool(id_ctrl.get("pass", False))
    perm_status, perm_reason = combined_permutation_status(
        importance_effect_status=ie_status,
        guided_permutation_status=gp_status,
        identity_pass=identity_pass,
    )

    # Portfolio action summary
    focus_depth = min(3, len(ranked)) if ranked else 0
    action_row = {
        "model_family": family,
        "median_tv_single_family_top_ig": float(
            effects.loc[effects.feature_family == ranked[0], "median_tv"].median()
        )
        if ranked and not effects.empty
        else float("nan"),
        "median_tv_guided_focus": float(guided["summary"]["median_guided_tv_focus"].iloc[0])
        if not guided["summary"].empty
        else float("nan"),
        "median_cash_guided_focus": float(guided["summary"]["median_guided_cash_focus"].iloc[0])
        if not guided["summary"].empty
        else float("nan"),
        "guided_vs_random_gap": float(guided["metrics"].get("guided_permutation_gap", float("nan"))),
        "focus_depth": focus_depth,
    }

    # Mandatory target sensitivity
    mandatory_rows = []
    if family == cfg.mandatory_family:
        for tgt in (TARGET_L2_EQUAL, TARGET_CASH, TARGET_VECTOR_L1):
            mat_t = _ig_once(
                adapter,
                batch,
                median,
                target_name=tgt,
                n_steps=cfg.ig_steps,
                cash_idx=cash_idx,
                batch_size=cfg.attribution_batch_size,
            )
            abs_t, _ = profiles_from_attr(mat_t, family_ids)
            ranked_t = ranking_from_abs_profile(abs_t)
            # Single-family TV for concordance under this ranking importance
            # Reuse L2 action effects but importance from target-specific IG
            eff_t = effects.copy()
            if not eff_t.empty:
                map_abs = abs_t.to_dict()
                eff_t["original_ig_abs"] = eff_t["feature_family"].map(lambda f: float(map_abs.get(f, 0.0)))
            conc_t = importance_effect_concordance(
                eff_t, model_family=family, bootstrap_replicates=500, random_seed=cfg.random_seed
            )
            guided_t = guided_vs_random_permutation(
                adapter,
                batch,
                donor_by_seed,
                family_ids,
                ranked_t,
                model_family=family,
                max_depth=min(cfg.max_depth, 3),
                random_control_repeats=min(cfg.random_control_repeats, 50),
                bootstrap_replicates=1000,
                random_seed=cfg.random_seed + 7,
                cash_idx=cash_idx,
            )
            mandatory_rows.append(
                {
                    "model_family": family,
                    "target": tgt,
                    "importance_effect_status": conc_t["importance_effect_status"],
                    "importance_effect_spearman": conc_t["spearman"],
                    "guided_permutation_status": guided_t["guided_permutation_status"],
                    "guided_permutation_gap": guided_t["metrics"].get("guided_permutation_gap", float("nan")),
                    "guided_permutation_ci_lower": guided_t["metrics"].get("guided_permutation_ci_lower", float("nan")),
                    "notes": "Mandatory-family target sensitivity; uses same donors/observations.",
                }
            )

    overlay = {
        "model_family": family,
        "ti_flag": bool(str(family).endswith("_TI_True")),
        "batch_c_status": batch_c_status,
        "batch_c2_status": batch_c2_status,
        "importance_effect_status": ie_status,
        "importance_effect_spearman": conc.get("spearman", float("nan")),
        "importance_effect_ci_lower": conc.get("ci_lower", float("nan")),
        "guided_permutation_status": gp_status,
        "guided_permutation_gap": guided["metrics"].get("guided_permutation_gap", float("nan")),
        "guided_permutation_ci_lower": guided["metrics"].get("guided_permutation_ci_lower", float("nan")),
        "global_permutation_kendall": glob.get("kendall_tau_b", float("nan")),
        "identity_control_pass": identity_pass,
        "state_plausibility_status": plaus_status,
        "permutation_evidence_status": perm_status,
        "permutation_evidence_reason": perm_reason,
        "screening_decision": screening_decision,
        "evidence_complete": True,
        "notes": "",
        "checkpoint_hash": ckpt_hash,
        "feature_count": int(len(family_ids)),
        "feature_family_count": int(len(feature_families)),
        "cash_index": cash_idx,
        "validation_sample_count": int(len(primary_idx)),
    }

    return {
        "model_family": family,
        "incomplete": False,
        "sample_manifest": sample_manifest,
        "assignments": assignments,
        "controls": controls,
        "effects": effects,
        "attr_changes": attr_changes,
        "concordance": pd.DataFrame([conc]),
        "sensitivity": sens,
        "curves": guided["curves"],
        "random_controls": guided["controls"],
        "cardinality_matching": guided["matching"],
        "guided_summary": guided["summary"],
        "global_attr": pd.DataFrame([glob]),
        "plausibility": pd.DataFrame([{**plaus, "state_plausibility_status": plaus_status}]),
        "action_effects": pd.DataFrame([action_row]),
        "mandatory": pd.DataFrame(mandatory_rows),
        "overlay": overlay,
        "inventory_update": {
            "feature_count": int(len(family_ids)),
            "feature_family_count": int(len(feature_families)),
            "cash_index": cash_idx,
            "model_loading_status": "ok",
            "validation_sample_count": int(len(primary_idx)),
            "validation_sample_source": "batch_c" if reused else "reconstructed",
        },
    }


def run_batch_c3(cfg: PermutationConfig) -> Dict[str, Any]:
    ensure_output_dir(cfg.output_dir)
    (cfg.output_dir / "intermediates").mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "latex").mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "logs").mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = SourceAccessLog()
    priors = verify_frozen_priors(cfg, log)

    # Build a ValidityConfig-compatible object for discovery
    vcfg = ValidityConfig(
        repository_root=cfg.repository_root,
        diagnostic_pkl_root=cfg.diagnostic_pkl_root,
        batch_a_dir=cfg.batch_a_dir,
        family_mapping_report=cfg.family_mapping_report,
        model_code_root=cfg.model_code_root,
        output_dir=cfg.output_dir,
        validation_start=cfg.validation_start,
        validation_end=cfg.validation_end,
        device=cfg.device,
    )
    records = discover_family_checkpoints(vcfg)
    inventory = build_execution_inventory(cfg, records, log)
    write_table(inventory, cfg.output_dir / "audit_c3_execution_inventory.csv")

    loadable = int(inventory["diagnostic_complete"].sum()) if not inventory.empty else 0
    if loadable < 15:
        (cfg.output_dir / "BATCH_C3_MISSING_INPUTS.md").write_text(
            f"# Batch C3 missing inputs\n\nOnly {loadable} diagnostically complete checkpoints are available "
            f"(need >= 15). Stopping before aggregate conclusions.\n",
            encoding="utf-8",
        )
        write_table(log.to_frame(), cfg.output_dir / "audit_c3_source_lineage.csv")
        return {"stopped": True, "loadable": loadable, "priors": priors}

    if cfg.inspect_only:
        write_table(log.to_frame(), cfg.output_dir / "audit_c3_source_lineage.csv")
        return {"inspect_only": True, "inventory": inventory, "priors": priors}

    # Overlay metadata from Batch A/C/C2 (validation-only)
    decisions = load_permitted(
        cfg.batch_a_dir / "audit_validation_decisions_primary_5bps.csv",
        role="batch_a_decisions",
        log=log,
        loader=pd.read_csv,
    )
    c_overlay = load_permitted(
        cfg.batch_c_dir / "audit_validity_combined_status.csv",
        role="batch_c_status",
        log=log,
        loader=pd.read_csv,
    )
    c2_path = cfg.batch_c2_dir / "audit_c2_targeted_validity_overlay.csv"
    c2_overlay = pd.read_csv(c2_path) if c2_path.exists() else pd.DataFrame()
    if c2_path.exists():
        log.record(c2_path, source_role="batch_c2_overlay", validation_or_holdout="validation", loaded=True, permitted=True)

    def _status_map(df: pd.DataFrame, fam_col: str, status_col: str) -> Dict[str, str]:
        if df is None or df.empty or status_col not in df.columns:
            return {}
        return {str(r[fam_col]): str(r[status_col]) for _, r in df.iterrows()}

    c_status = _status_map(c_overlay, "model_family", "combined_validity_status")
    if not c_status and "validity_status" in (c_overlay.columns if not c_overlay.empty else []):
        c_status = _status_map(c_overlay, "model_family", "validity_status")
    c2_status = {}
    if not c2_overlay.empty:
        col = "targeted_evidence_status" if "targeted_evidence_status" in c2_overlay.columns else None
        if col:
            c2_status = _status_map(c2_overlay, "model_family", col)
    screen = _status_map(decisions, "model_family", "operational_recommendation")

    device = resolve_device(cfg.device)
    adapter = DecisionTransformerAdapter(vcfg)

    families_filter = set(cfg.families) if cfg.families else None
    tables: Dict[str, List[pd.DataFrame]] = {k: [] for k in [
        "sample_manifest", "assignments", "controls", "effects", "attr_changes",
        "concordance", "sensitivity", "curves", "random_controls", "cardinality_matching",
        "guided_summary", "global_attr", "plausibility", "action_effects", "mandatory", "overlay",
    ]}
    exec_log = []

    for rec in records:
        fam = rec.model_family
        # Always retain incomplete-evidence family in summaries even when --families filters others.
        if families_filter and fam not in families_filter and fam != cfg.incomplete_family:
            continue
        if families_filter and fam == cfg.incomplete_family and fam not in families_filter:
            # Still emit INCOMPLETE overlay row without loading tensors.
            tables["overlay"].append(
                pd.DataFrame(
                    [
                        {
                            "model_family": fam,
                            "ti_flag": bool(str(fam).endswith("_TI_True")),
                            "batch_c_status": c_status.get(fam, "INCOMPLETE"),
                            "batch_c2_status": c2_status.get(fam, ""),
                            "importance_effect_status": "INCOMPLETE",
                            "importance_effect_spearman": float("nan"),
                            "importance_effect_ci_lower": float("nan"),
                            "guided_permutation_status": "INCOMPLETE",
                            "guided_permutation_gap": float("nan"),
                            "guided_permutation_ci_lower": float("nan"),
                            "global_permutation_kendall": float("nan"),
                            "identity_control_pass": True,
                            "state_plausibility_status": "INCOMPLETE",
                            "permutation_evidence_status": "INCOMPLETE",
                            "permutation_evidence_reason": "Incomplete-evidence family.",
                            "screening_decision": screen.get(fam, "REJECT"),
                            "evidence_complete": False,
                            "notes": "Prespecified incomplete; retained in summary tables.",
                        }
                    ]
                )
            )
            exec_log.append({"model_family": fam, "status": "incomplete_retained", "seconds": 0})
            continue
        inter = cfg.output_dir / "intermediates" / f"{fam}__c3.json"
        if cfg.resume and inter.exists() and not cfg.force:
            LOGGER.info("Resume skip %s", fam)
            cached = json.loads(inter.read_text(encoding="utf-8"))
            # reload tables from companion csvs if present
            for key in tables:
                p = cfg.output_dir / "intermediates" / f"{fam}__{key}.csv"
                if p.exists():
                    tables[key].append(pd.read_csv(p))
            if "overlay" in cached:
                tables["overlay"].append(pd.DataFrame([cached["overlay"]]))
            exec_log.append({"model_family": fam, "status": "resumed", "seconds": 0})
            continue

        t0 = time.time()
        LOGGER.info("Evaluating Batch C3 family %s", fam)
        try:
            result = evaluate_family_c3(
                cfg,
                rec,
                adapter,
                log=log,
                batch_c_status=c_status.get(fam, ""),
                batch_c2_status=c2_status.get(fam, ""),
                screening_decision=screen.get(fam, ""),
            )
            if result.get("incomplete"):
                tables["overlay"].append(
                    pd.DataFrame(
                        [
                            {
                                "model_family": fam,
                                "ti_flag": bool(str(fam).endswith("_TI_True")),
                                "batch_c_status": c_status.get(fam, "INCOMPLETE"),
                                "batch_c2_status": c2_status.get(fam, ""),
                                "importance_effect_status": "INCOMPLETE",
                                "importance_effect_spearman": float("nan"),
                                "importance_effect_ci_lower": float("nan"),
                                "guided_permutation_status": "INCOMPLETE",
                                "guided_permutation_gap": float("nan"),
                                "guided_permutation_ci_lower": float("nan"),
                                "global_permutation_kendall": float("nan"),
                                "identity_control_pass": True,
                                "state_plausibility_status": "INCOMPLETE",
                                "permutation_evidence_status": "INCOMPLETE",
                                "permutation_evidence_reason": "Incomplete-evidence family.",
                                "screening_decision": screen.get(fam, "REJECT"),
                                "evidence_complete": False,
                                "notes": result.get("notes", ""),
                            }
                        ]
                    )
                )
            else:
                for key in tables:
                    if key in result and isinstance(result[key], pd.DataFrame) and not result[key].empty:
                        tables[key].append(result[key])
                        result[key].to_csv(
                            cfg.output_dir / "intermediates" / f"{fam}__{key}.csv", index=False
                        )
                    elif key == "overlay" and "overlay" in result:
                        tables["overlay"].append(pd.DataFrame([result["overlay"]]))
                # update inventory
                inv_u = result.get("inventory_update", {})
                for k, v in inv_u.items():
                    inventory.loc[inventory.model_family == fam, k] = v
                inter.write_text(
                    json.dumps({"overlay": result.get("overlay", {}), "seconds": time.time() - t0}, indent=2),
                    encoding="utf-8",
                )
            exec_log.append({"model_family": fam, "status": "ok", "seconds": time.time() - t0})
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Family %s failed: %s", fam, exc)
            exec_log.append({"model_family": fam, "status": f"error:{exc}", "seconds": time.time() - t0})
            tables["overlay"].append(
                pd.DataFrame(
                    [
                        {
                            "model_family": fam,
                            "ti_flag": bool(str(fam).endswith("_TI_True")),
                            "permutation_evidence_status": "INCOMPLETE",
                            "permutation_evidence_reason": f"Execution error: {exc}",
                            "evidence_complete": False,
                            "identity_control_pass": False,
                            "batch_c_status": c_status.get(fam, ""),
                            "batch_c2_status": c2_status.get(fam, ""),
                            "screening_decision": screen.get(fam, ""),
                            "notes": str(exc),
                        }
                    ]
                )
            )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _cat(key: str, name: str) -> pd.DataFrame:
        frames = tables.get(key, [])
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        write_table(df, cfg.output_dir / name)
        return df

    write_table(inventory, cfg.output_dir / "audit_c3_execution_inventory.csv")
    write_table(log.to_frame(), cfg.output_dir / "audit_c3_source_lineage.csv")
    write_table(pd.DataFrame(exec_log), cfg.output_dir / "audit_c3_execution_log.csv")

    sample_manifest = _cat("sample_manifest", "audit_c3_sample_manifest.csv")
    assignments = _cat("assignments", "audit_c3_permutation_assignments.csv")
    controls = _cat("controls", "audit_c3_control_results.csv")
    effects = _cat("effects", "audit_c3_single_family_effects.csv")
    attr_changes = _cat("attr_changes", "audit_c3_single_family_attribution_changes.csv")
    concordance = _cat("concordance", "audit_c3_importance_effect_concordance.csv")
    sensitivity = _cat("sensitivity", "audit_c3_importance_effect_sensitivity.csv")
    curves = _cat("curves", "audit_c3_guided_permutation_curves.csv")
    random_controls = _cat("random_controls", "audit_c3_random_permutation_controls.csv")
    matching = _cat("cardinality_matching", "audit_c3_cardinality_matching.csv")
    guided_summary = _cat("guided_summary", "audit_c3_guided_vs_random.csv")
    global_attr = _cat("global_attr", "audit_c3_global_permutation_attribution.csv")
    plausibility = _cat("plausibility", "audit_c3_state_plausibility.csv")
    action_effects = _cat("action_effects", "audit_c3_portfolio_action_effects.csv")
    mandatory = _cat("mandatory", "audit_c3_mandatory_target_sensitivity.csv")
    overlay = _cat("overlay", "audit_c3_permutation_overlay.csv")

    # Status sensitivity
    sens_rows = []
    for _, r in overlay.iterrows():
        ie = str(r.get("importance_effect_status", "INCOMPLETE"))
        gp = str(r.get("guided_permutation_status", "INCOMPLETE"))
        sens_rows.append(
            {
                "model_family": r["model_family"],
                "interpretation": "concordance_only",
                "status": ie,
            }
        )
        sens_rows.append(
            {
                "model_family": r["model_family"],
                "interpretation": "guided_gap_only",
                "status": gp,
            }
        )
        sens_rows.append(
            {
                "model_family": r["model_family"],
                "interpretation": "strict_conjunctive",
                "status": "SUPPORTED"
                if ie == "SUPPORTED" and gp == "SUPPORTED" and bool(r.get("identity_control_pass", False))
                else (
                    "INCOMPLETE"
                    if ie == "INCOMPLETE" or gp == "INCOMPLETE"
                    else (
                        "UNSUPPORTED"
                        if ie == "UNSUPPORTED" or gp == "UNSUPPORTED" or not bool(r.get("identity_control_pass", True))
                        else "MIXED"
                    )
                ),
            }
        )
    status_sens = pd.DataFrame(sens_rows)
    write_table(status_sens, cfg.output_dir / "audit_c3_status_sensitivity.csv")

    # Subgroup summaries
    if not overlay.empty:
        ti_rows = []
        for ti_val, label in ((True, "TI_True"), (False, "TI_False")):
            sub = overlay[overlay["ti_flag"] == ti_val]
            if sub.empty:
                continue
            ti_rows.append(
                {
                    "subgroup": label,
                    "n": int(len(sub)),
                    "median_importance_effect_spearman": float(pd.to_numeric(sub["importance_effect_spearman"], errors="coerce").median()),
                    "median_guided_gap": float(pd.to_numeric(sub["guided_permutation_gap"], errors="coerce").median()),
                    "n_supported": int((sub["permutation_evidence_status"] == "SUPPORTED").sum()),
                    "n_mixed": int((sub["permutation_evidence_status"] == "MIXED").sum()),
                    "n_unsupported": int((sub["permutation_evidence_status"] == "UNSUPPORTED").sum()),
                    "n_incomplete": int((sub["permutation_evidence_status"] == "INCOMPLETE").sum()),
                    "median_global_kendall": float(pd.to_numeric(sub["global_permutation_kendall"], errors="coerce").median()),
                    "notes": "Descriptive TI subgroup contrast; not a causal TI effect estimate.",
                }
            )
        ti_summary = pd.DataFrame(ti_rows)
    else:
        ti_summary = pd.DataFrame()
    write_table(ti_summary, cfg.output_dir / "audit_c3_ti_subgroup_summary.csv")

    above = overlay[overlay["model_family"].isin(list(cfg.above_market_families))].copy() if not overlay.empty else pd.DataFrame()
    write_table(above, cfg.output_dir / "audit_c3_above_market_summary.csv")

    # Post-run integrity: ensure prior snapshots unchanged
    snap_a2 = _snapshot_hashes(cfg.batch_a_dir)
    snap_c2b = _snapshot_hashes(cfg.batch_c_dir)
    snap_c22 = _snapshot_hashes(cfg.batch_c2_dir)
    prior_ok = (
        snap_a2 == priors["snapshot_a"]
        and snap_c2b == priors["snapshot_c"]
        and snap_c22 == priors["snapshot_c2"]
    )

    from .permutation_reporting import finalize_batch_c3_outputs

    summary = finalize_batch_c3_outputs(
        cfg,
        tables={
            "inventory": inventory,
            "sample_manifest": sample_manifest,
            "assignments": assignments,
            "controls": controls,
            "effects": effects,
            "attr_changes": attr_changes,
            "concordance": concordance,
            "sensitivity": sensitivity,
            "curves": curves,
            "random_controls": random_controls,
            "matching": matching,
            "guided_summary": guided_summary,
            "global_attr": global_attr,
            "plausibility": plausibility,
            "action_effects": action_effects,
            "mandatory": mandatory,
            "overlay": overlay,
            "status_sensitivity": status_sens,
            "ti_summary": ti_summary,
            "above_market": above,
            "source_lineage": log.to_frame(),
            "execution_log": pd.DataFrame(exec_log),
        },
        priors=priors,
        prior_outputs_unchanged=prior_ok,
        source_violations=list(log.violations),
    )
    return summary
