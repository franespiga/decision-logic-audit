"""Checkpoint discovery, policy adapters, and Integrated Gradients attribution.

Uses the same TrainableDT.from_pretrained pathway and validation tensors stored
in diagnostic PKLs (ablation_attributions) to avoid reconstructing a divergent
input pipeline.
"""

from __future__ import annotations

import gc
import hashlib
import logging
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .feature_groups import parse_indicator_token
from .holdout_assessment import INCOMPLETE_EVIDENCE_FAMILY, sha256_file
from .io import ensure_output_dir, load_pickle, write_table
from .model_mapping import experiment_name_from_filename, load_family_mapping_report
from .validity_config import ValidityConfig, resolve_device

LOGGER = logging.getLogger("audit_scoring.checkpoint_loading")


def add_model_import_paths(repository_root: Path, model_code_root: Path) -> None:
    candidates = [
        Path(repository_root),
        Path(repository_root) / "src",
        Path(model_code_root),
        Path(model_code_root) / "src",
        Path(model_code_root) / "src" / "portfolio_optimization",
    ]
    for path in candidates:
        text = str(path)
        if path.exists() and text not in sys.path:
            sys.path.insert(0, text)


def import_trainable_dt(repository_root: Path, model_code_root: Path):
    add_model_import_paths(repository_root, model_code_root)
    errors: List[str] = []
    for module_name in ("models.model", "src.models.model"):
        try:
            mod = __import__(module_name, fromlist=["TrainableDT"])
            return getattr(mod, "TrainableDT")
        except Exception as exc:  # pragma: no cover
            errors.append(f"{module_name}: {exc}")
    raise ImportError("Unable to import TrainableDT: " + "; ".join(errors))


def import_forward_action(repository_root: Path, model_code_root: Path):
    add_model_import_paths(repository_root, model_code_root)
    for module_name in ("ablation", "src.ablation"):
        try:
            mod = __import__(module_name, fromlist=["_forward_action"])
            return getattr(mod, "_forward_action")
        except Exception:
            continue
    raise ImportError("Unable to import _forward_action")


def cash_index_from_instruments(instruments: Optional[Sequence[str]], act_dim: int) -> int:
    if instruments:
        lowered = [str(x).upper() for x in instruments]
        if "CASH" in lowered:
            return int(lowered.index("CASH"))
    return max(0, act_dim - 1)


def total_variation_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """TV(a,b)=0.5*sum|a-b| for allocation vectors that approximately sum to one."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim == 1:
        a = a.reshape(1, -1)
        b = b.reshape(1, -1)
    return 0.5 * np.abs(a - b).sum(axis=-1)


def allocation_distance(a: np.ndarray, b: np.ndarray, tol: float = 1e-2) -> Tuple[np.ndarray, str]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim == 1:
        a = a.reshape(1, -1)
        b = b.reshape(1, -1)
    sums_ok = np.all(np.abs(a.sum(axis=-1) - 1.0) < tol) and np.all(np.abs(b.sum(axis=-1) - 1.0) < tol)
    if sums_ok:
        return total_variation_distance(a, b), "total_variation"
    denom = np.maximum(np.abs(a).sum(axis=-1), 1e-12)
    return (np.abs(a - b).sum(axis=-1) / denom), "normalized_l1"


def cash_and_risky_deltas(
    original: np.ndarray,
    perturbed: np.ndarray,
    cash_idx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    original = np.asarray(original, dtype=float)
    perturbed = np.asarray(perturbed, dtype=float)
    if original.ndim == 1:
        original = original.reshape(1, -1)
        perturbed = perturbed.reshape(1, -1)
    cash_delta = np.abs(perturbed[:, cash_idx] - original[:, cash_idx])
    risky_orig = 1.0 - original[:, cash_idx]
    risky_pert = 1.0 - perturbed[:, cash_idx]
    risky_delta = risky_pert - risky_orig
    return cash_delta, risky_delta


def flatten_feature_names(labels: Any, n_features: int) -> np.ndarray:
    if isinstance(labels, dict):
        names = []
        for i in range(n_features):
            if i in labels:
                names.append(str(labels[i]))
            elif str(i) in labels:
                names.append(str(labels[str(i)]))
            else:
                names.append(f"feature_{i}")
        return np.asarray(names, dtype=object)
    arr = np.asarray(labels, dtype=object).reshape(-1)
    if len(arr) == n_features:
        return arr
    out = np.asarray([f"feature_{i}" for i in range(n_features)], dtype=object)
    out[: min(len(arr), n_features)] = arr[:n_features]
    return out


def indicator_family_ids(feature_names: Sequence[str]) -> np.ndarray:
    return np.asarray([parse_indicator_token(str(n)) for n in feature_names], dtype=object)


def aggregate_family_profile(
    attribution_matrix: np.ndarray,
    family_ids: np.ndarray,
    *,
    signed: bool,
) -> pd.Series:
    matrix = np.asarray(attribution_matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    values = np.nanmean(matrix if signed else np.abs(matrix), axis=0)
    families = np.asarray(family_ids, dtype=object)
    out: Dict[str, float] = {}
    for fam in sorted(set(families.tolist())):
        mask = families == fam
        out[str(fam)] = float(np.nanmean(values[mask])) if np.any(mask) else 0.0
    return pd.Series(out, dtype=float).sort_index()


@dataclass
class FamilyCheckpointRecord:
    model_family: str
    checkpoint_path: Path
    checkpoint_exists: bool
    diagnostic_pkl: Optional[Path]
    pkl_exists: bool
    experiment_name: str
    model_class: str = "TrainableDT"
    model_type: str = "decision_transformer"
    config_source: str = ""
    preprocessing_source: str = "diagnostic_pkl.ablation_attributions"
    validation_data_source: str = "diagnostic_pkl.ablation_attributions"
    feature_names_source: str = "ablation_attributions.labels"
    regime_labels_source: str = "audit_scoring.config.REGIME_BOUNDARIES"
    attribution_target: str = "allocation_l2_deviation_from_equal_weight_ig"
    original_baseline: str = "zero"
    loading_status: str = "pending"
    notes: str = ""
    is_incomplete_family: bool = False


@dataclass
class PolicyBatch:
    states: torch.Tensor
    actions: torch.Tensor
    returns_to_go: torch.Tensor
    timesteps: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    dates: pd.DatetimeIndex
    feature_names: np.ndarray
    family_ids: np.ndarray
    instruments: List[str]
    cash_idx: int
    source_indices: np.ndarray  # indices into filtered validation pool


class DecisionTransformerAdapter:
    """Architecture-neutral policy-output adapter for Decision Transformer checkpoints."""

    def __init__(self, cfg: ValidityConfig):
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.TrainableDT = import_trainable_dt(cfg.repository_root, cfg.model_code_root)
        self._forward_action = import_forward_action(cfg.repository_root, cfg.model_code_root)
        self._model: Optional[nn.Module] = None
        self._checkpoint_path: Optional[Path] = None
        self._checkpoint_hash: Optional[str] = None

    def load_model(self, checkpoint_path: Path) -> nn.Module:
        checkpoint_path = Path(checkpoint_path)
        before_hash = sha256_file(checkpoint_path) if checkpoint_path.is_file() else None
        # HuggingFace-style directories hash a representative weight file when present.
        weight_candidates = sorted(checkpoint_path.glob("*.bin")) + sorted(checkpoint_path.glob("*.safetensors"))
        if checkpoint_path.is_dir() and weight_candidates:
            before_hash = sha256_file(weight_candidates[0])
        model = self.TrainableDT.from_pretrained(str(checkpoint_path))
        model = model.to(self.device).eval()
        after_hash = before_hash
        if weight_candidates:
            after_hash = sha256_file(weight_candidates[0])
        if before_hash and after_hash and before_hash != after_hash:
            raise RuntimeError(f"Checkpoint mutation detected while loading {checkpoint_path}")
        self._model = model
        self._checkpoint_path = checkpoint_path
        self._checkpoint_hash = before_hash
        return model

    @property
    def model(self) -> nn.Module:
        if self._model is None:
            raise RuntimeError("Model not loaded")
        return self._model

    def release(self) -> None:
        self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _prepare_rtg(self, rtg: torch.Tensor) -> torch.Tensor:
        if rtg.ndim == 2:
            return rtg.unsqueeze(-1)
        return rtg

    def forward_actions(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        states = batch["states"]
        actions = batch["actions"]
        rtg = self._prepare_rtg(batch["returns_to_go"])
        timesteps = batch["timesteps"]
        attention_mask = batch.get("attention_mask")
        with torch.no_grad():
            out = self._forward_action(
                self.model,
                states,
                actions,
                rtg,
                timesteps,
                attention_mask=attention_mask,
                target_action_idx=None,
            )
        return out.detach()

    def compute_integrated_gradients(
        self,
        batch: Mapping[str, torch.Tensor],
        baseline: torch.Tensor,
        *,
        n_steps: int,
    ) -> torch.Tensor:
        """IG of a non-degenerate functional of the allocation vector.

        Because Decision Transformer portfolio actions approximately lie on the
        simplex, summing action coordinates is nearly constant and yields
        vanishing attributions. We therefore attribute the squared Euclidean
        deviation from equal-weight allocation, which remains a pure function of
        the action output and is sensitive to learned allocation structure.
        """
        from captum.attr import IntegratedGradients

        states = batch["states"].detach().clone().requires_grad_(True)
        actions = batch["actions"]
        rtg = self._prepare_rtg(batch["returns_to_go"])
        timesteps = batch["timesteps"]
        attention_mask = batch.get("attention_mask")
        baseline = baseline.to(states.device).detach()
        if baseline.shape != states.shape:
            if baseline.shape[0] == 1 and states.shape[0] > 1:
                baseline = baseline.expand_as(states)
            else:
                raise ValueError(f"Baseline shape {tuple(baseline.shape)} != states {tuple(states.shape)}")

        def forward_wrapper(input_states: torch.Tensor) -> torch.Tensor:
            preds = self._forward_action(
                self.model,
                input_states,
                actions,
                rtg,
                timesteps,
                attention_mask=attention_mask,
                target_action_idx=None,
            )
            equal = 1.0 / float(preds.shape[-1])
            return ((preds - equal) ** 2).sum(dim=-1)

        ig = IntegratedGradients(forward_wrapper)
        attr = ig.attribute(states, baselines=baseline, n_steps=n_steps)
        return attr.detach()

    def reinitialize_modules(self, level: str, seed: int) -> List[str]:
        """Reinitialize selected modules in-place on a deep-copied model."""
        torch.manual_seed(int(seed))
        np.random.seed(int(seed) % (2**32 - 1))
        model = self.model
        reset_names: List[str] = []

        def _reset(module: nn.Module, name: str) -> None:
            if hasattr(module, "reset_parameters") and callable(module.reset_parameters):
                module.reset_parameters()
                reset_names.append(name)
            else:
                for child_name, child in module.named_children():
                    _reset(child, f"{name}.{child_name}" if name else child_name)

        if level == "output_head":
            candidates = []
            for name, module in model.named_modules():
                lname = name.lower()
                if any(tok in lname for tok in ("predict_action", "action_pred", "action_head", "embed_action")):
                    if hasattr(module, "reset_parameters"):
                        candidates.append((name, module))
            if not candidates:
                # Fallback: last Linear with out_features == act_dim if discoverable.
                linears = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
                if linears:
                    candidates = [linears[-1]]
            for name, module in candidates:
                module.reset_parameters()
                reset_names.append(name)
            if not reset_names:
                raise RuntimeError("output_head randomization: no resettable action head found")
            return reset_names

        if level == "final_block_and_head":
            blocks = []
            # Prefer GPT2-style blocks under transformer.h or encoder.h.
            for attr in ("transformer", "gpt2", "encoder"):
                container = getattr(model, attr, None)
                if container is not None and hasattr(container, "h"):
                    blocks = list(container.h)
                    block_prefix = f"{attr}.h"
                    break
            else:
                block_prefix = "block"
            if blocks:
                last = blocks[-1]
                _reset(last, f"{block_prefix}.-1")
            # Also reset action head
            head_names = self.reinitialize_modules("output_head", seed + 17)
            reset_names.extend(head_names)
            if not reset_names:
                raise RuntimeError("final_block_and_head unavailable")
            return reset_names

        if level == "full_model":
            for name, module in model.named_modules():
                if name and hasattr(module, "reset_parameters") and callable(module.reset_parameters):
                    # Avoid double-resetting children if parent resets; call on leaves with params.
                    if any(True for _ in module.children()):
                        continue
                    module.reset_parameters()
                    reset_names.append(name)
            if not reset_names:
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        torch.nn.init.normal_(param, mean=0.0, std=0.02)
                        reset_names.append(name)
            return reset_names

        raise ValueError(f"Unknown randomization level: {level}")

    def load_fresh_copy(self, checkpoint_path: Path) -> nn.Module:
        self.release()
        return self.load_model(checkpoint_path)


def discover_family_checkpoints(cfg: ValidityConfig) -> List[FamilyCheckpointRecord]:
    mapping = load_family_mapping_report(cfg.family_mapping_report)
    if mapping.empty:
        raise FileNotFoundError(f"Family mapping report missing: {cfg.family_mapping_report}")
    fam_paths = (
        mapping.dropna(subset=["model_family", "model_path"])
        .drop_duplicates(subset=["model_family"])
        .sort_values("model_family")
    )
    records: List[FamilyCheckpointRecord] = []
    for _, row in fam_paths.iterrows():
        family = str(row["model_family"])
        ckpt = Path(str(row["model_path"]))
        experiment = ckpt.parent.name if ckpt.parent.name.startswith("RL_AG_DT") else ckpt.name
        pkl = cfg.diagnostic_pkl_root / f"model_diagnostics_{experiment}.pkl"
        partial = cfg.diagnostic_pkl_root / "partial" / f"model_diagnostics_{experiment}.pkl"
        pkl_path = pkl if pkl.exists() else (partial if partial.exists() else pkl)
        notes = ""
        incomplete = family == cfg.incomplete_family
        if incomplete:
            notes = "Frozen incomplete-evidence family; Batch C status must remain INCOMPLETE."
        if not pkl_path.exists():
            notes = (notes + " Missing diagnostic PKL.").strip()
        records.append(
            FamilyCheckpointRecord(
                model_family=family,
                checkpoint_path=ckpt,
                checkpoint_exists=ckpt.exists(),
                diagnostic_pkl=pkl_path if pkl_path.exists() else None,
                pkl_exists=pkl_path.exists(),
                experiment_name=experiment,
                config_source=str(cfg.family_mapping_report),
                loading_status="discovered",
                notes=notes,
                is_incomplete_family=incomplete,
            )
        )
    return records


def load_validation_pool_from_pkl(
    pkl_path: Path,
    *,
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
) -> Dict[str, Any]:
    payload = load_pickle(pkl_path)
    if not isinstance(payload, dict) or "ablation_attributions" not in payload:
        raise KeyError(f"{pkl_path} missing ablation_attributions")
    aa = payload["ablation_attributions"]
    dates = pd.to_datetime(pd.Index(aa["dates"]))
    keep = (dates >= validation_start) & (dates <= validation_end)
    if not np.any(keep):
        raise ValueError(f"No validation-period observations in {pkl_path}")
    idx = np.flatnonzero(np.asarray(keep))
    states = aa["states"][idx]
    actions = aa["actions"][idx]
    rtg = aa["returns_to_go"][idx]
    timesteps = aa["timesteps"][idx]
    attn = aa.get("attention_mask")
    if attn is not None:
        attn = attn[idx]
    dates = dates[idx]
    feature_names = flatten_feature_names(aa.get("labels"), int(states.shape[-1]))
    family_ids = indicator_family_ids(feature_names)
    instruments: List[str] = []
    # Prefer metadata instruments when present in sibling config.
    meta = payload.get("metadata") or {}
    if isinstance(meta, dict) and meta.get("instruments"):
        instruments = list(meta["instruments"])
    cash_idx = cash_index_from_instruments(instruments, int(actions.shape[-1]))
    return {
        "states": states,
        "actions": actions,
        "returns_to_go": rtg,
        "timesteps": timesteps,
        "attention_mask": attn,
        "dates": dates,
        "feature_names": feature_names,
        "family_ids": family_ids,
        "instruments": instruments,
        "cash_idx": cash_idx,
        "pool_indices": idx,
        "payload_keys": sorted(payload.keys()),
    }


def write_execution_inventory(records: Sequence[FamilyCheckpointRecord], path: Path) -> pd.DataFrame:
    rows = []
    for r in records:
        rows.append(
            {
                "model_family": r.model_family,
                "checkpoint_path": str(r.checkpoint_path),
                "checkpoint_exists": bool(r.checkpoint_exists),
                "model_class": r.model_class,
                "model_type": r.model_type,
                "config_source": r.config_source,
                "preprocessing_source": r.preprocessing_source,
                "validation_data_source": r.validation_data_source,
                "feature_names_source": r.feature_names_source,
                "regime_labels_source": r.regime_labels_source,
                "attribution_target": r.attribution_target,
                "original_baseline": r.original_baseline,
                "loading_status": r.loading_status,
                "notes": r.notes,
                "diagnostic_pkl": str(r.diagnostic_pkl) if r.diagnostic_pkl else "",
                "pkl_exists": bool(r.pkl_exists),
            }
        )
    df = pd.DataFrame(rows).sort_values("model_family").reset_index(drop=True)
    ensure_output_dir(path.parent)
    write_table(df, path)
    return df
