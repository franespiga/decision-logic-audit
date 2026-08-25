"""Batch C2 orchestration pipeline."""

from __future__ import annotations

import copy
import gc
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .baseline_validity import (
    build_empirical_median_baseline,
    build_zero_baseline,
    classify_baseline_status,
    pairwise_concordance,
)
from .benchmark_discovery import (
    discover_market_benchmark_candidates,
    select_above_market_families,
    select_authoritative_benchmark,
    write_benchmark_candidates,
)
from .checkpoint_loading import (
    DecisionTransformerAdapter,
    allocation_distance,
    cash_and_risky_deltas,
    discover_family_checkpoints,
    load_validation_pool_from_pkl,
)
from .holdout_assessment import (
    INCOMPLETE_EVIDENCE_FAMILY,
    sha256_file,
    verify_batch_a_freeze,
)
from .io import ensure_output_dir, write_json, write_table
from .perturbation_plausibility import run_plausibility_for_family
from .raw_feature_validity import resolution_comparison, summarize_asset_output_attributions
from .targeted_validity import (
    TARGET_CASH,
    TARGET_L2_EQUAL,
    TARGET_LARGEST_RISKY,
    TARGET_SUM_WEIGHTS,
    TARGET_VECTOR_L1,
    attribution_matrix_from_tensor,
    compare_rank_maps,
    compute_ig_with_target,
    config_hash,
    parameter_change_stats,
    profiles_from_attr,
    raw_feature_ranks,
    run_cardinality_matched_deletion_insertion,
    selection_list_hash,
)
from .validation_sampling import assert_validation_dates, stratified_sample_indices
from .validity_config import ValidityConfig, resolve_device
from .validity_guards import SourceAccessLog

LOGGER = logging.getLogger("audit_scoring.targeted_validity_pipeline")

MANDATORY_FAMILY_DEFAULT = "max_quadratic_utility_TI_True"


@dataclass
class TargetedConfig:
    repository_root: Path = Path(__file__).resolve().parents[2]
    diagnostic_pkl_root: Path = repository_root / "results" / "diagnostics"
    batch_a_dir: Path = repository_root / "results" / "mendeley" / "dt_metrics" / "audit_screening"
    batch_b_dir: Path = repository_root / "results" / "mendeley" / "dt_metrics" / "audit_holdout"
    batch_c_dir: Path = repository_root / "results" / "mendeley" / "dt_metrics" / "audit_validity"
    batch_c_verification_dir: Path = (
        repository_root / "results" / "mendeley" / "dt_metrics" / "audit_validity_verification"
    )
    metrics_dir: Path = repository_root / "results" / "mendeley" / "dt_metrics"
    output_dir: Path = (
        repository_root / "results" / "mendeley" / "dt_metrics" / "audit_validity_targeted"
    )
    family_mapping_report: Path = (
        repository_root / "results" / "mendeley" / "dt_metrics" / "audit_inputs" / "family_mapping_report.csv"
    )
    model_code_root: Path = repository_root
    mandatory_full_family: str = MANDATORY_FAMILY_DEFAULT
    primary_cost_bps: int = 5
    validation_start: pd.Timestamp = pd.Timestamp("2016-01-01")
    validation_end: pd.Timestamp = pd.Timestamp("2024-04-30")
    primary_samples: int = 128
    randomization_samples: int = 64
    background_samples: int = 16
    randomization_seeds: Tuple[int, ...] = (101, 202, 303, 404, 505)
    matched_random_repeats_full: int = 100
    matched_random_repeats_partial: int = 50
    bootstrap_replicates: int = 10000
    random_seed: int = 42
    ig_steps: int = 16
    attribution_batch_size: int = 8
    device: str = "auto"
    force: bool = False
    resume: bool = True
    inspect_only: bool = False
    discover_benchmark_only: bool = False
    skip_figures: bool = False
    families: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, Path):
                d[k] = str(v)
            elif isinstance(v, pd.Timestamp):
                d[k] = v.isoformat()
            elif isinstance(v, tuple):
                d[k] = list(v)
        return d


def _snapshot_hashes(directory: Path) -> Dict[str, str]:
    out = {}
    if not Path(directory).exists():
        return out
    for p in sorted(Path(directory).rglob("*")):
        if p.is_file() and p.suffix.lower() in {".csv", ".json", ".parquet", ".md", ".tex", ".png", ".pdf"}:
            # skip huge intermediates optionally
            if "intermediates" in str(p):
                continue
            out[str(p)] = sha256_file(p)
    return out


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
    if pool.get("attention_mask") is not None:
        out["attention_mask"] = _sel(pool["attention_mask"])
    return out


def verify_freezes(cfg: TargetedConfig, log: SourceAccessLog) -> Dict[str, Any]:
    freeze_a = verify_batch_a_freeze(cfg.batch_a_dir, repo_root=cfg.repository_root)
    log.record(
        cfg.batch_a_dir / "BATCH_A_FROZEN_MANIFEST.json",
        source_role="batch_a_manifest",
        validation_or_holdout="validation",
        loaded=True,
        permitted=True,
    )
    c_manifest = Path(cfg.batch_c_dir) / "BATCH_C_VALIDITY_FROZEN_MANIFEST.json"
    if not c_manifest.exists():
        raise FileNotFoundError(c_manifest)
    c_data = json.loads(c_manifest.read_text(encoding="utf-8"))
    log.record(c_manifest, source_role="batch_c_manifest", validation_or_holdout="validation", loaded=True, permitted=True)
    # Verify Batch C analytical hashes for key CSVs
    changed = []
    for entry in c_data.get("output_files") or []:
        # Manifest may use output_sha256 dict form
        pass
    sha_map = c_data.get("output_sha256") or {}
    for path_str, stored in sha_map.items():
        p = Path(path_str)
        name = p.name.lower()
        if not any(name.endswith(ext) for ext in (".csv", ".json")):
            continue
        if "freeze_summary" in name or "run_stdout" in name:
            continue
        if not p.exists():
            changed.append({"path": path_str, "issue": "missing"})
            continue
        cur = sha256_file(p)
        if cur.lower() != str(stored).lower():
            changed.append({"path": path_str, "stored": stored, "current": cur})
    batch_a_hashes = _snapshot_hashes(cfg.batch_a_dir)
    batch_b_hashes = _snapshot_hashes(cfg.batch_b_dir)
    batch_c_hashes = _snapshot_hashes(cfg.batch_c_dir)
    verdict_path = cfg.batch_c_verification_dir / "BATCH_C_VERIFICATION_REPORT.md"
    verdict = "UNKNOWN"
    if verdict_path.exists():
        text = verdict_path.read_text(encoding="utf-8", errors="ignore")
        if "VERIFIED WITH METHODOLOGICAL QUALIFICATIONS" in text:
            verdict = "VERIFIED WITH METHODOLOGICAL QUALIFICATIONS"
        elif "VERIFIED" in text.upper():
            verdict = "VERIFIED"
        log.record(
            verdict_path,
            source_role="batch_c_verification_verdict",
            validation_or_holdout="provenance",
            loaded=True,
            permitted=True,
            notes="Verdict only; not used analytically.",
        )
    return {
        "batch_a_ok": bool(freeze_a.ok),
        "batch_a_changed": freeze_a.changed_files,
        "batch_c_analytical_changed": changed,
        "batch_c_verification_verdict": verdict,
        "batch_a_hashes": batch_a_hashes,
        "batch_b_hashes": batch_b_hashes,
        "batch_c_hashes": batch_c_hashes,
    }


def _ig_batched(
    adapter: DecisionTransformerAdapter,
    batch: Mapping[str, torch.Tensor],
    baseline: torch.Tensor,
    *,
    target_name: str,
    n_steps: int,
    cash_idx: int,
    batch_size: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    n = batch["states"].shape[0]
    mats = []
    diag_acc: List[Dict[str, float]] = []
    for start in range(0, n, batch_size):
        sl = slice(start, min(n, start + batch_size))
        sub = {k: v[sl] for k, v in batch.items() if torch.is_tensor(v)}
        base = baseline[sl] if baseline.shape[0] == n else baseline
        attr, diag = compute_ig_with_target(
            adapter, sub, base, target_name=target_name, n_steps=n_steps, cash_idx=cash_idx
        )
        mats.append(attribution_matrix_from_tensor(attr))
        diag_acc.append(diag)
    matrix = np.concatenate(mats, axis=0)
    # average diagnostics
    keys = diag_acc[0].keys()
    diag = {k: float(np.mean([d[k] for d in diag_acc])) for k in keys}
    return matrix, diag


def evaluate_family_c2(
    cfg: TargetedConfig,
    record,
    adapter: DecisionTransformerAdapter,
    *,
    full: bool,
    log: SourceAccessLog,
) -> Dict[str, Any]:
    family = record.model_family
    if family == INCOMPLETE_EVIDENCE_FAMILY or not record.pkl_exists or not record.checkpoint_exists:
        return {
            "model_family": family,
            "incomplete": True,
            "notes": "Missing inputs or incomplete-evidence family.",
        }

    log.record(record.diagnostic_pkl, source_role="diagnostic_pkl", validation_or_holdout="validation", loaded=True, permitted=True)
    pool = load_validation_pool_from_pkl(
        record.diagnostic_pkl,
        validation_start=cfg.validation_start,
        validation_end=cfg.validation_end,
    )
    dates = pool["dates"]
    # Prefer Batch C sample indices if present
    sample_path = Path(cfg.batch_c_dir) / "audit_validity_sample_manifest.csv"
    primary_idx = None
    if sample_path.exists():
        man = pd.read_csv(sample_path)
        sub = man[(man.model_family == family) & (man.sample_role == "primary")]
        if not sub.empty:
            primary_idx = sub["observation_index"].astype(int).to_numpy()
            log.record(sample_path, source_role="batch_c_sample_manifest", validation_or_holdout="validation", loaded=True, permitted=True)
    if primary_idx is None:
        primary_idx, _, _ = stratified_sample_indices(
            dates, n_samples=cfg.primary_samples, per_regime=32, random_seed=cfg.random_seed
        )
    rand_local, _, _ = stratified_sample_indices(
        dates[primary_idx],
        n_samples=min(cfg.randomization_samples, len(primary_idx)),
        per_regime=16,
        random_seed=cfg.random_seed + 1,
    )
    rand_idx = primary_idx[rand_local]
    bg_idx, _, _ = stratified_sample_indices(
        dates, n_samples=cfg.background_samples, per_regime=4, random_seed=cfg.random_seed + 2
    )
    assert_validation_dates(dates[primary_idx], cfg.validation_end)

    device = adapter.device
    pool_states = pool["states"].detach().to("cpu").float().to(device) if torch.is_tensor(pool["states"]) else torch.as_tensor(pool["states"]).float().to(device)
    primary = _tensor_batch(pool, primary_idx, device)
    rand_batch = _tensor_batch(pool, rand_idx, device)
    bg_states = pool["states"].detach().to("cpu").float().index_select(0, torch.as_tensor(bg_idx, dtype=torch.long)).to(device)
    family_ids = pool["family_ids"]
    feature_names = pool["feature_names"]
    cash_idx = int(pool["cash_idx"])

    adapter.load_fresh_copy(record.checkpoint_path)
    log.record(record.checkpoint_path, source_role="checkpoint", validation_or_holdout="validation", loaded=True, permitted=True)

    zero = build_zero_baseline(primary["states"])
    median = build_empirical_median_baseline(pool_states, primary["states"])

    # --- Baseline concordance under L2-equal ---
    abs_profiles = {}
    signed_profiles = {}
    for bname, baseline in (("zero", zero), ("empirical_median", median)):
        mat, _ = _ig_batched(
            adapter, primary, baseline, target_name=TARGET_L2_EQUAL, n_steps=cfg.ig_steps, cash_idx=cash_idx, batch_size=cfg.attribution_batch_size
        )
        abs_p, signed_p = profiles_from_attr(mat, family_ids)
        abs_profiles[bname] = abs_p
        signed_profiles[bname] = signed_p
    # sampled background average
    bg_mats = []
    for i in range(bg_states.shape[0]):
        base = bg_states[i : i + 1].expand_as(primary["states"])
        mat, _ = _ig_batched(
            adapter, primary, base, target_name=TARGET_L2_EQUAL, n_steps=cfg.ig_steps, cash_idx=cash_idx, batch_size=cfg.attribution_batch_size
        )
        bg_mats.append(mat)
    bg_mat = np.mean(np.stack(bg_mats, axis=0), axis=0)
    abs_profiles["sampled_background"], signed_profiles["sampled_background"] = profiles_from_attr(bg_mat, family_ids)
    pairwise = pairwise_concordance(abs_profiles, signed_profiles, top_k=5)
    pairwise.insert(0, "model_family", family)
    baseline_status, baseline_metrics = classify_baseline_status(pairwise)
    # Separate rank vs sign
    median_tau = float(baseline_metrics.get("median_kendall_tau_b", float("nan")))
    mean_sign = float(baseline_metrics.get("mean_top_k_sign_agreement", float("nan")))
    rank_status = "SUPPORTED" if median_tau >= 0.5 else ("MIXED" if median_tau >= 0.2 else "UNSUPPORTED")
    sign_status = "SUPPORTED" if mean_sign >= 0.7 else ("MIXED" if mean_sign >= 0.4 else "UNSUPPORTED")

    median_abs = abs_profiles["empirical_median"]
    median_signed = signed_profiles["empirical_median"]
    median_mat, _ = _ig_batched(
        adapter, primary, median, target_name=TARGET_L2_EQUAL, n_steps=cfg.ig_steps, cash_idx=cash_idx, batch_size=cfg.attribution_batch_size
    )

    # --- Full-model randomization (L2) ---
    seed_rows = []
    raw_rand_rows = []
    orig_actions = adapter.forward_actions(rand_batch).detach().cpu().numpy()
    # store original raw ranks on rand subset
    rand_median = build_empirical_median_baseline(pool_states, rand_batch["states"])
    mat_r0, _ = _ig_batched(
        adapter, rand_batch, rand_median, target_name=TARGET_L2_EQUAL, n_steps=cfg.ig_steps, cash_idx=cash_idx, batch_size=cfg.attribution_batch_size
    )
    abs_r0, signed_r0 = profiles_from_attr(mat_r0, family_ids)
    raw0 = raw_feature_ranks(mat_r0, feature_names)

    full_sims = []
    raw_sims = []
    for seed in cfg.randomization_seeds:
        # load fresh trained copy then randomize
        adapter.load_fresh_copy(record.checkpoint_path)
        before = copy.deepcopy(adapter.model).cpu()
        adapter.reinitialize_modules("full_model", int(seed))
        after = adapter.model
        pstats = parameter_change_stats(before, after)
        del before
        act = adapter.forward_actions(rand_batch).detach().cpu().numpy()
        dist, _ = allocation_distance(orig_actions, act)
        cash_d, risky_d = cash_and_risky_deltas(orig_actions, act, cash_idx)
        mat_r, diag = _ig_batched(
            adapter, rand_batch, rand_median, target_name=TARGET_L2_EQUAL, n_steps=cfg.ig_steps, cash_idx=cash_idx, batch_size=cfg.attribution_batch_size
        )
        abs_r, signed_r = profiles_from_attr(mat_r, family_ids)
        comps = compare_rank_maps(abs_r0, abs_r, signed_r0, signed_r, top_k=5)
        full_sims.append(comps["kendall_tau_b"])
        raw_r = raw_feature_ranks(mat_r, feature_names)
        res = resolution_comparison(
            model_family=family,
            family_abs=abs_r0,
            family_abs_rand=abs_r,
            raw_abs_ranks=raw0,
            raw_abs_ranks_rand=raw_r,
        )
        raw_sims.append(res["raw_randomization_kendall"])
        seed_rows.append(
            {
                "model_family": family,
                "seed": int(seed),
                "target": TARGET_L2_EQUAL,
                "level": "full_model",
                "median_allocation_distance": float(np.nanmedian(dist)),
                "median_cash_abs_change": float(np.nanmedian(cash_d)),
                "median_risky_exposure_change": float(np.nanmedian(risky_d)),
                **comps,
                **pstats,
                **{f"diag_{k}": v for k, v in diag.items()},
            }
        )
        raw_rand_rows.append({"model_family": family, "seed": int(seed), **res})
        adapter.release()
        gc.collect()

    full_sim = float(np.nanmedian(full_sims)) if full_sims else float("nan")
    # reuse Batch C style thresholds loosely
    from .validity_config import RAND_SUPPORTED_FULL_SIM, RAND_MIXED_FULL_SIM

    if not np.isfinite(full_sim):
        rand_status = "INCOMPLETE"
    elif full_sim <= RAND_SUPPORTED_FULL_SIM:
        rand_status = "SUPPORTED"
    elif full_sim <= RAND_MIXED_FULL_SIM:
        rand_status = "MIXED"
    else:
        rand_status = "UNSUPPORTED"

    # --- Cardinality-matched deletion ---
    adapter.load_fresh_copy(record.checkpoint_path)
    repeats = cfg.matched_random_repeats_full if full else cfg.matched_random_repeats_partial
    faith = run_cardinality_matched_deletion_insertion(
        adapter,
        primary,
        pool_states,
        family_ids,
        median_abs,
        model_family=family,
        max_depth=5,
        random_control_repeats=repeats,
        bootstrap_replicates=min(cfg.bootstrap_replicates, 5000 if not full else cfg.bootstrap_replicates),
        random_seed=cfg.random_seed,
        cash_idx=cash_idx,
        do_insertion=full,
    )

    plaus = run_plausibility_for_family(
        primary["states"], pool_states, family_ids, median_abs, model_family=family, max_depth=3, random_seed=cfg.random_seed
    )
    plaus_status = (
        plaus["perturbed_state_plausibility_status"].mode().iloc[0]
        if not plaus.empty
        else "INCOMPLETE"
    )

    result: Dict[str, Any] = {
        "model_family": family,
        "full": full,
        "baseline_status": baseline_status,
        "c2_baseline_rank_status": rank_status,
        "c2_baseline_sign_status": sign_status,
        "baseline_median_tau": median_tau,
        "baseline_mean_sign": mean_sign,
        "c2_randomization_status": rand_status,
        "full_randomization_similarity": full_sim,
        "c2_cardinality_matched_deletion_status": faith["deletion_status"],
        "c2_insertion_status": faith["insertion_status"] if full else "NOT_RUN",
        "raw_feature_randomization_status": (
            pd.DataFrame(raw_rand_rows)["raw_feature_randomization_status"].mode().iloc[0]
            if raw_rand_rows
            else "INCOMPLETE"
        ),
        "perturbed_state_plausibility_status": plaus_status,
        "deletion_gap": faith["metrics"].get("deletion_gap"),
        "insertion_gap": faith["metrics"].get("insertion_gap"),
        "tables": {
            "baseline_pairwise": pairwise,
            "randomization": pd.DataFrame(seed_rows),
            "randomization_raw": pd.DataFrame(raw_rand_rows),
            "deletion": faith["deletion"],
            "insertion": faith["insertion"],
            "controls": faith["controls"],
            "deletion_summary": faith["summary"],
            "plausibility": plaus,
            "raw_features": raw_feature_ranks(median_mat, feature_names).assign(model_family=family, target=TARGET_L2_EQUAL),
            "asset_outputs": summarize_asset_output_attributions(
                median_mat, feature_names, model_family=family, target_name=TARGET_L2_EQUAL
            ),
        },
    }

    # Full-package extras for mandatory family
    if full:
        target_rows = []
        target_profiles = {}
        for tname in (TARGET_L2_EQUAL, TARGET_CASH, TARGET_VECTOR_L1, TARGET_LARGEST_RISKY, TARGET_SUM_WEIGHTS):
            mat, diag = _ig_batched(
                adapter, primary, median, target_name=tname, n_steps=cfg.ig_steps, cash_idx=cash_idx, batch_size=cfg.attribution_batch_size
            )
            abs_p, signed_p = profiles_from_attr(mat, family_ids)
            target_profiles[tname] = (abs_p, signed_p, mat, diag)
            target_rows.append({"model_family": family, "target": tname, **diag, "attr_mean_abs": float(np.nanmean(np.abs(mat)))})
        # pairwise target concordance vs L2
        sens_rows = []
        base_abs, base_signed, _, _ = target_profiles[TARGET_L2_EQUAL]
        for tname, (abs_p, signed_p, mat, diag) in target_profiles.items():
            comps = compare_rank_maps(base_abs, abs_p, base_signed, signed_p, top_k=5)
            sens_rows.append({"model_family": family, "target_a": TARGET_L2_EQUAL, "target_b": tname, **comps, **diag})
        # target-specific randomization for CASH and vector
        tgt_rand_rows = []
        for tname in (TARGET_CASH, TARGET_VECTOR_L1):
            adapter.load_fresh_copy(record.checkpoint_path)
            mat0, _ = _ig_batched(
                adapter, rand_batch, rand_median, target_name=tname, n_steps=cfg.ig_steps, cash_idx=cash_idx, batch_size=cfg.attribution_batch_size
            )
            abs0, signed0 = profiles_from_attr(mat0, family_ids)
            sims = []
            for seed in cfg.randomization_seeds:
                adapter.load_fresh_copy(record.checkpoint_path)
                adapter.reinitialize_modules("full_model", int(seed))
                mat1, _ = _ig_batched(
                    adapter, rand_batch, rand_median, target_name=tname, n_steps=cfg.ig_steps, cash_idx=cash_idx, batch_size=cfg.attribution_batch_size
                )
                abs1, signed1 = profiles_from_attr(mat1, family_ids)
                comps = compare_rank_maps(abs0, abs1, signed0, signed1, top_k=5)
                sims.append(comps["kendall_tau_b"])
                tgt_rand_rows.append({"model_family": family, "target": tname, "seed": int(seed), **comps})
                adapter.release()
            sens_rows.append(
                {
                    "model_family": family,
                    "target_a": tname,
                    "target_b": "full_model_randomization_median_tau",
                    "kendall_tau_b": float(np.nanmedian(sims)),
                }
            )
        # also output-head / final-block for L2
        level_rows = []
        for level in ("output_head", "final_block_and_head", "full_model"):
            for seed in cfg.randomization_seeds:
                adapter.load_fresh_copy(record.checkpoint_path)
                try:
                    adapter.reinitialize_modules(level, int(seed))
                except Exception as exc:
                    level_rows.append({"model_family": family, "level": level, "seed": int(seed), "status": "INCOMPLETE", "notes": str(exc)})
                    continue
                mat1, _ = _ig_batched(
                    adapter, rand_batch, rand_median, target_name=TARGET_L2_EQUAL, n_steps=cfg.ig_steps, cash_idx=cash_idx, batch_size=cfg.attribution_batch_size
                )
                abs1, signed1 = profiles_from_attr(mat1, family_ids)
                comps = compare_rank_maps(abs_r0, abs1, signed_r0, signed1, top_k=5)
                level_rows.append({"model_family": family, "level": level, "seed": int(seed), "status": "computed", **comps})
                adapter.release()

        result["tables"]["target_sensitivity"] = pd.DataFrame(sens_rows)
        result["tables"]["target_diagnostics"] = pd.DataFrame(target_rows)
        result["tables"]["randomization_target_comparison"] = pd.DataFrame(tgt_rand_rows)
        result["tables"]["randomization_levels"] = pd.DataFrame(level_rows)
        result["asset_output_validity_status"] = (
            "MIXED" if result["raw_feature_randomization_status"] == "MIXED" else result["raw_feature_randomization_status"]
        )
    else:
        # reduced target sensitivity for TI_True
        if family.endswith("_TI_True"):
            tgt_rand_rows = []
            for tname in (TARGET_CASH, TARGET_VECTOR_L1):
                adapter.load_fresh_copy(record.checkpoint_path)
                mat0, _ = _ig_batched(
                    adapter, rand_batch, rand_median, target_name=tname, n_steps=max(8, cfg.ig_steps // 2), cash_idx=cash_idx, batch_size=cfg.attribution_batch_size
                )
                abs0, signed0 = profiles_from_attr(mat0, family_ids)
                for seed in cfg.randomization_seeds[:3]:
                    adapter.load_fresh_copy(record.checkpoint_path)
                    adapter.reinitialize_modules("full_model", int(seed))
                    mat1, _ = _ig_batched(
                        adapter, rand_batch, rand_median, target_name=tname, n_steps=max(8, cfg.ig_steps // 2), cash_idx=cash_idx, batch_size=cfg.attribution_batch_size
                    )
                    abs1, signed1 = profiles_from_attr(mat1, family_ids)
                    comps = compare_rank_maps(abs0, abs1, signed0, signed1, top_k=5)
                    tgt_rand_rows.append({"model_family": family, "target": tname, "seed": int(seed), **comps})
                    adapter.release()
            result["tables"]["randomization_target_comparison"] = pd.DataFrame(tgt_rand_rows)
        result["asset_output_validity_status"] = "NOT_RUN"

    adapter.release()
    return result


def _targeted_status(row: Mapping[str, Any]) -> Tuple[str, str]:
    if row.get("incomplete"):
        return "INCOMPLETE", "Required execution evidence unavailable."
    core = [
        row.get("c2_baseline_rank_status"),
        row.get("c2_randomization_status"),
        row.get("c2_cardinality_matched_deletion_status"),
    ]
    if any(x == "INCOMPLETE" for x in core):
        return "INCOMPLETE", "Core check incomplete."
    n_sup = sum(x == "SUPPORTED" for x in core)
    if row.get("c2_randomization_status") == "UNSUPPORTED" and row.get("c2_cardinality_matched_deletion_status") == "UNSUPPORTED":
        return "UNSUPPORTED", "Randomization and cardinality-matched deletion both unsupported."
    if any(x == "UNSUPPORTED" for x in core):
        if n_sup >= 1:
            return "MIXED", "Mixed supportive and unsupported targeted checks."
        return "UNSUPPORTED", "Unsupported core targeted checks."
    if n_sup >= 2 and all(x != "UNSUPPORTED" for x in core):
        return "SUPPORTED", "At least two core targeted checks supported; none unsupported."
    return "MIXED", "Mixed or uncertain targeted evidence."


def run_batch_c2(cfg: TargetedConfig) -> Dict[str, Any]:
    ensure_output_dir(cfg.output_dir)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.time()
    log = SourceAccessLog()
    freeze = verify_freezes(cfg, log)
    if (not freeze["batch_a_ok"] or freeze["batch_c_analytical_changed"]) and not cfg.force:
        raise RuntimeError("Freeze verification failed; pass --force to continue")

    candidates = discover_market_benchmark_candidates(cfg.metrics_dir, cost_bps=cfg.primary_cost_bps)
    write_benchmark_candidates(candidates, cfg.output_dir / "audit_market_benchmark_candidates.csv")
    selected = select_authoritative_benchmark(candidates, cfg.output_dir)
    if selected is None:
        return {"stopped": "benchmark_confirmation_required", "freeze": freeze}
    if cfg.discover_benchmark_only:
        return {"benchmark": selected.to_dict(), "freeze": freeze}

    selection = select_above_market_families(
        cfg.metrics_dir,
        market_name=selected.candidate_name,
        market_sharpe=selected.sharpe,
        cost_bps=cfg.primary_cost_bps,
        mandatory_family=cfg.mandatory_full_family,
    )
    write_table(selection, cfg.output_dir / "audit_above_market_family_selection.csv")
    selected_families = selection.loc[selection.selected_for_batch_c2, "model_family"].astype(str).tolist()
    sel_hash = selection_list_hash(selected_families)
    write_json(
        {"families": selected_families, "selection_list_hash": sel_hash, "market": selected.to_dict()},
        cfg.output_dir / "audit_c2_selection_freeze.json",
    )
    LOGGER.info("Frozen selection %s hash=%s", selected_families, sel_hash)

    if cfg.inspect_only:
        write_table(log.to_frame(), cfg.output_dir / "audit_c2_source_lineage.csv")
        return {"selection": selected_families, "benchmark": selected.to_dict(), "inspect_only": True}

    records = discover_family_checkpoints(
        ValidityConfig(
            repository_root=cfg.repository_root,
            diagnostic_pkl_root=cfg.diagnostic_pkl_root,
            family_mapping_report=cfg.family_mapping_report,
            model_code_root=cfg.model_code_root,
            incomplete_family=INCOMPLETE_EVIDENCE_FAMILY,
        )
    )
    rec_map = {r.model_family: r for r in records}
    if cfg.families:
        run_families = [f for f in selected_families if f in set(cfg.families)]
    else:
        run_families = selected_families

    adapter = DecisionTransformerAdapter(
        ValidityConfig(
            repository_root=cfg.repository_root,
            model_code_root=cfg.model_code_root,
            device=cfg.device,
            ig_steps=cfg.ig_steps,
        )
    )

    # Load frozen Batch C overlay for comparison
    batch_c_overlay = pd.read_csv(cfg.batch_c_dir / "audit_validity_evidence_overlay.csv")
    batch_a_primary = pd.read_csv(cfg.batch_a_dir / "audit_validation_decisions_primary_5bps.csv")
    log.record(
        cfg.batch_c_dir / "audit_validity_evidence_overlay.csv",
        source_role="batch_c_overlay_readonly",
        validation_or_holdout="validation",
        loaded=True,
        permitted=True,
        notes="Read-only comparison; not overwritten.",
    )

    all_tables: Dict[str, List[pd.DataFrame]] = {}
    family_results: Dict[str, Dict[str, Any]] = {}
    exec_rows = []
    cfg_hash = config_hash(cfg.to_dict())

    for fam in run_families:
        cache = cfg.output_dir / "intermediates" / f"{fam}__c2.json"
        started = time.time()
        try:
            if cfg.resume and cache.exists() and not cfg.force:
                cached = json.loads(cache.read_text(encoding="utf-8"))
                if cached.get("config_hash") == cfg_hash:
                    family_results[fam] = cached
                    shard = cfg.output_dir / "intermediates" / fam
                    for p in shard.glob("*.csv"):
                        all_tables.setdefault(p.stem, []).append(pd.read_csv(p))
                    exec_rows.append({"model_family": fam, "status": "resumed", "elapsed_sec": 0.0, "error": ""})
                    continue
            record = rec_map.get(fam)
            if record is None:
                raise FileNotFoundError(f"No checkpoint record for {fam}")
            full = fam == cfg.mandatory_full_family
            LOGGER.info("Batch C2 evaluating %s (full=%s)", fam, full)
            result = evaluate_family_c2(cfg, record, adapter, full=full, log=log)
            result["config_hash"] = cfg_hash
            family_results[fam] = result
            shard = cfg.output_dir / "intermediates" / fam
            ensure_output_dir(shard)
            for key, df in result.get("tables", {}).items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    write_table(df, shard / f"{key}.csv")
                    all_tables.setdefault(key, []).append(df)
            # save slim json
            slim = {k: v for k, v in result.items() if k != "tables"}
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(slim, indent=2, default=str), encoding="utf-8")
            exec_rows.append({"model_family": fam, "status": "completed", "elapsed_sec": time.time() - started, "error": ""})
        except Exception as exc:
            LOGGER.exception("C2 failed for %s", fam)
            family_results[fam] = {
                "model_family": fam,
                "incomplete": True,
                "notes": str(exc),
                "config_hash": cfg_hash,
                "c2_baseline_rank_status": "INCOMPLETE",
                "c2_baseline_sign_status": "INCOMPLETE",
                "c2_randomization_status": "INCOMPLETE",
                "c2_cardinality_matched_deletion_status": "INCOMPLETE",
                "c2_insertion_status": "INCOMPLETE",
                "raw_feature_randomization_status": "INCOMPLETE",
                "perturbed_state_plausibility_status": "INCOMPLETE",
                "asset_output_validity_status": "INCOMPLETE",
            }
            exec_rows.append({"model_family": fam, "status": "failed", "elapsed_sec": time.time() - started, "error": str(exc)})

    def cat(key: str, out_name: str) -> pd.DataFrame:
        frames = all_tables.get(key) or []
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        write_table(df, cfg.output_dir / out_name)
        return df

    baseline_am = cat("baseline_pairwise", "audit_c2_above_market_baseline_concordance.csv")
    rand_am = cat("randomization", "audit_c2_above_market_randomization.csv")
    del_am = cat("deletion_summary", "audit_c2_above_market_deletion.csv")
    # also store detailed deletion curves
    cat("deletion", "audit_c2_cardinality_matched_deletion.csv")
    cat("insertion", "audit_c2_cardinality_matched_insertion.csv")
    cat("controls", "audit_c2_control_comparison.csv")
    cat("target_sensitivity", "audit_c2_target_sensitivity_full.csv")
    cat("target_diagnostics", "audit_c2_target_sensitivity_summary.csv")
    cat("randomization_levels", "audit_c2_randomization_full.csv")
    cat("randomization_raw", "audit_c2_randomization_raw_feature.csv")
    cat("randomization_target_comparison", "audit_c2_randomization_target_comparison.csv")
    cat("raw_features", "audit_c2_raw_feature_attributions.csv")
    cat("asset_outputs", "audit_c2_asset_output_attributions.csv")
    cat("plausibility", "audit_c2_perturbed_state_plausibility_full.csv")

    # resolution comparison from raw randomization rows
    res_frames = all_tables.get("randomization_raw") or []
    write_table(pd.concat(res_frames, ignore_index=True) if res_frames else pd.DataFrame(), cfg.output_dir / "audit_c2_resolution_comparison.csv")

    overlay_rows = []
    for fam in selected_families:
        fres = family_results.get(fam, {"incomplete": True})
        bc = batch_c_overlay[batch_c_overlay.model_family == fam]
        ba = batch_a_primary[batch_a_primary.model_family == fam]
        sel = selection[selection.model_family == fam].iloc[0]
        status, reason = _targeted_status(fres)
        overlay_rows.append(
            {
                "model_family": fam,
                "mandatory_full_validation": fam == cfg.mandatory_full_family,
                "above_market_validation_sharpe": float(sel.validation_median_sharpe),
                "market_benchmark_sharpe": float(selected.sharpe),
                "original_batch_c_baseline_status": bc.iloc[0]["baseline_concordance_status"] if not bc.empty else "",
                "c2_baseline_rank_status": fres.get("c2_baseline_rank_status", "INCOMPLETE"),
                "c2_baseline_sign_status": fres.get("c2_baseline_sign_status", "INCOMPLETE"),
                "original_batch_c_randomization_status": bc.iloc[0]["randomization_status"] if not bc.empty else "",
                "c2_randomization_status": fres.get("c2_randomization_status", "INCOMPLETE"),
                "original_batch_c_deletion_status": bc.iloc[0]["deletion_status"] if not bc.empty else "",
                "c2_cardinality_matched_deletion_status": fres.get("c2_cardinality_matched_deletion_status", "INCOMPLETE"),
                "c2_insertion_status": fres.get("c2_insertion_status", "NOT_RUN"),
                "raw_feature_randomization_status": fres.get("raw_feature_randomization_status", "INCOMPLETE"),
                "asset_output_validity_status": fres.get("asset_output_validity_status", "NOT_RUN"),
                "perturbed_state_plausibility_status": fres.get("perturbed_state_plausibility_status", "INCOMPLETE"),
                "targeted_evidence_status": status,
                "targeted_evidence_reason": reason,
                "frozen_screening_decision": ba.iloc[0]["operational_recommendation"] if not ba.empty else "",
                "frozen_batch_c_status": bc.iloc[0]["combined_validity_status"] if not bc.empty else "",
                "notes": fres.get("notes", ""),
            }
        )
    overlay = pd.DataFrame(overlay_rows)
    write_table(overlay, cfg.output_dir / "audit_c2_targeted_validity_overlay.csv")
    write_table(pd.DataFrame(exec_rows), cfg.output_dir / "audit_c2_execution_log.csv")
    write_table(log.to_frame(), cfg.output_dir / "audit_c2_source_lineage.csv")

    from .targeted_validity_reporting import finalize_batch_c2

    summary = finalize_batch_c2(
        cfg,
        freeze=freeze,
        benchmark=selected.to_dict(),
        selection=selection,
        selection_hash=sel_hash,
        overlay=overlay,
        family_results=family_results,
        source_log=log,
        elapsed_sec=time.time() - t0,
        batch_a_hashes_before=freeze["batch_a_hashes"],
        batch_b_hashes_before=freeze["batch_b_hashes"],
        batch_c_hashes_before=freeze["batch_c_hashes"],
    )
    return summary
