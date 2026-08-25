"""
Multi-family (Decision Transformer + Stable-Baselines3) adapters for ablation studies.

Supports:
  - DT models via TrainableDT / original_forward
  - SB3 models with MLP or CNN feature extractors
  - Train / eval / holdout period splits
  - Unified ablation batches, weight rollouts, Captum-friendly forwards, and latents
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from captum.attr import FeatureAblation, IntegratedGradients
from tqdm import tqdm

from ablation.preprocessing import (
    create_feature_mapping_from_labels,
    create_return_matrices,
    prepare_ablation_inputs,
)

try:
    from data import load_datasets
except ImportError:  # pragma: no cover
    from src.data import load_datasets

ModelFamily = Literal["DT", "SB3"]
FeatureExtractorKind = Literal["MLP", "CNN"]
ActionSpaceKind = Literal["continuous", "discrete"]
PeriodName = Literal["train", "eval", "holdout"]


DEFAULT_PERIODS: Dict[str, Tuple[Optional[str], Optional[str]]] = {
    "train": (None, "2015-12-31"),
    "eval": ("2016-01-01", "2024-04-30"),
    "holdout": ("2024-05-01", None),
}


@dataclass
class ModelSpec:
    """Specification for a model to analyse."""

    label: str
    type: ModelFamily
    path: str
    feature_extractor: FeatureExtractorKind = "MLP"
    algorithm: Optional[str] = None  # e.g. PPO, SAC, A2C for SB3
    action_space: ActionSpaceKind = "continuous"
    metadata_path: Optional[str] = None
    data_path: Optional[str] = None
    lookback: Optional[int] = None
    obs_shape: Optional[Tuple[int, ...]] = None
    instruments: Optional[List[str]] = None
    env_state: Optional[List[Any]] = None
    technical_indicators: Optional[List[str]] = None
    rebalance_every: int = 1
    target_return: float = 0.15
    reference_date: Optional[str] = None  # start of evaluation walk
    cnn_layout: str = "features_x_time"  # or "time_x_features"
    drop_cash_from_obs: bool = False
    custom_obs_transform: Optional[Callable[[pd.DataFrame], torch.Tensor]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def normalized_type(self) -> ModelFamily:
        t = str(self.type).upper()
        if t in {"DT", "DECISION_TRANSFORMER", "DECISIONTRANSFORMER"}:
            return "DT"
        if t in {"SB3", "STABLE_BASELINES3", "STABLEBASELINES3", "RL"}:
            return "SB3"
        raise ValueError(f"Unsupported model type '{self.type}' for {self.label}")


def _normalize_env_state(env_state: Optional[List[Any]]) -> Optional[List[Any]]:
    if env_state is None:
        return None
    out = []
    for item in env_state:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            out.append((item[0], item[1]))
        else:
            out.append(item)
    return out


def build_env_state_from_indicators(
    instruments: List[str],
    technical_indicators: List[str],
) -> List[Tuple[str, str]]:
    """Cartesian product matching training observation layout (indicator-major)."""
    return [(inst, ind) for ind in technical_indicators for inst in instruments]

def parse_models_dict(models: Dict[str, Dict[str, Any]]) -> Dict[str, ModelSpec]:
    """
    Convert a user-provided mapping into ModelSpec objects.

    Expected format:
      {
        "my_dt": {"type": "DT", "path": ".../model_dir", "metadata_path": ".../metadata.json"},
        "my_ppo": {
            "type": "SB3",
            "path": ".../model.zip",
            "feature_extractor": "CNN",
            "algorithm": "PPO",
            "action_space": "continuous",
            ...
        },
      }
    """
    specs: Dict[str, ModelSpec] = {}
    for label, cfg in models.items():
        if not isinstance(cfg, dict):
            raise TypeError(f"Model entry for '{label}' must be a dict, got {type(cfg)}")
        payload = dict(cfg)
        payload["label"] = label
        if "type" not in payload:
            raise KeyError(f"Model '{label}' is missing required key 'type' (DT|SB3)")
        if "path" not in payload:
            raise KeyError(f"Model '{label}' is missing required key 'path'")
        # Drop unknown keys that are useful in the manifest but not ModelSpec fields
        known = {k: v for k, v in payload.items() if k in ModelSpec.__dataclass_fields__}
        if "env_state" in known:
            known["env_state"] = _normalize_env_state(known["env_state"])
        specs[label] = ModelSpec(**known)
        # Keep extras for debugging
        extras = {k: v for k, v in payload.items() if k not in ModelSpec.__dataclass_fields__ and k != "label"}
        if extras:
            specs[label].extra.update(extras)
    return specs


def load_models_manifest(path: str | Path) -> Dict[str, Dict[str, Any]]:
    """Load models dict from models_manifest.json (supports wrapped or flat formats)."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "models" in payload and isinstance(payload["models"], dict):
        return payload["models"]
    if isinstance(payload, dict):
        return payload
    raise TypeError(f"Unexpected manifest format in {path}")

def slice_period(
    df: pd.DataFrame,
    period: PeriodName | str,
    periods: Optional[Dict[str, Tuple[Optional[str], Optional[str]]]] = None,
) -> pd.DataFrame:
    """Slice a DataFrame by named train/eval/holdout period."""
    periods = periods or DEFAULT_PERIODS
    if period not in periods:
        raise KeyError(f"Unknown period '{period}'. Expected one of {list(periods)}")
    start, end = periods[period]
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    if start is not None:
        out = out.loc[pd.to_datetime(start) :]
    if end is not None:
        out = out.loc[: pd.to_datetime(end)]
    return out


def load_market_data(
    data_path: str,
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load instruments / technical indicators / close observations from HDF5."""
    cfg = cfg or {"start_date": None}
    return load_datasets(data_path, cfg)


def load_optional_metadata(path: Optional[str]) -> Dict[str, Any]:
    if path is None:
        return {}
    path = str(path)
    if os.path.isdir(path):
        candidate = os.path.join(path, "metadata.json")
        if not os.path.exists(candidate):
            return {}
        path = candidate
    elif not os.path.exists(path):
        # allow directory containing metadata.json (legacy callers)
        candidate = os.path.join(path, "metadata.json")
        if os.path.exists(candidate):
            path = candidate
        else:
            return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def enrich_spec_from_metadata(spec: ModelSpec) -> ModelSpec:
    """Fill missing ModelSpec fields from adjacent metadata.json when available."""
    meta = load_optional_metadata(spec.metadata_path)
    if not meta and spec.path:
        parent = Path(spec.path)
        # DT: path is often .../run_id/out ; SB3: .../model.zip
        for candidate in [
            parent / "metadata.json",
            parent.parent / "metadata.json",
            parent.parent.parent / "metadata.json",
        ]:
            if candidate.exists():
                meta = load_optional_metadata(str(candidate))
                if meta:
                    spec.metadata_path = str(candidate)
                    break

    if not meta:
        return spec

    exp = meta.get("experiment_args", meta)
    spec.lookback = spec.lookback or meta.get("lookback") or exp.get("lookback")
    spec.instruments = spec.instruments or meta.get("instruments") or exp.get("instruments")
    tis = (
        spec.technical_indicators
        or meta.get("technical_indicators")
        or exp.get("technical_indicators")
    )
    if tis:
        spec.technical_indicators = list(tis)
    spec.env_state = _normalize_env_state(
        spec.env_state or meta.get("env_state") or exp.get("env_state")
    )
    if spec.env_state is None and spec.instruments and spec.technical_indicators:
        # instruments list for actions may include CASH; state uses traded names only
        traded = [i for i in spec.instruments if i != "CASH"]
        # if cash already not present, use as-is
        base_inst = traded if traded else list(spec.instruments)
        spec.env_state = build_env_state_from_indicators(base_inst, spec.technical_indicators)
    if spec.instruments and meta.get("cash_instrument") and "CASH" not in spec.instruments:
        spec.instruments = list(spec.instruments) + ["CASH"]
    spec.data_path = spec.data_path or meta.get("data_path") or exp.get("data_path")
    spec.target_return = (
        spec.target_return
        if spec.target_return is not None
        else meta.get("target_return", exp.get("target_return", 0.15))
    )
    spec.reference_date = (
        spec.reference_date
        or exp.get("reference_date")
        or meta.get("reference_date")
        or DEFAULT_PERIODS["eval"][0]
    )
    if not spec.rebalance_every:
        spec.rebalance_every = int(exp.get("rebalance_every", 1) or 1)
    elif meta.get("rebalance_every") and spec.rebalance_every == 1:
        # keep explicit name-derived value if already set from manifest
        pass
    fe = exp.get("feature_extractor") or meta.get("feature_extractor")
    if fe:
        spec.feature_extractor = str(fe).upper()  # type: ignore[assignment]
    algo = exp.get("algorithm") or meta.get("algorithm") or exp.get("agent_algorithm")
    if algo and not spec.algorithm:
        spec.algorithm = str(algo)
    space = exp.get("action_space") or meta.get("action_space")
    if space:
        spec.action_space = str(space).lower()  # type: ignore[assignment]
    elif spec.algorithm and str(spec.algorithm).upper() in {"DQN", "PPO"}:
        spec.action_space = "discrete"
    if "drop_cash_from_obs" in meta:
        spec.drop_cash_from_obs = bool(meta["drop_cash_from_obs"])
    elif spec.feature_extractor == "CNN":
        spec.drop_cash_from_obs = True
    spec.extra["metadata"] = meta
    return spec

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _resolve_sb3_class(algorithm: Optional[str]):
    try:
        from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
        from stable_baselines3.common.base_class import BaseAlgorithm
    except ImportError as e:
        raise ImportError(
            "stable-baselines3 is required to load SB3 models. "
            "Install with: pip install stable-baselines3"
        ) from e

    mapping = {
        "PPO": PPO,
        "SAC": SAC,
        "A2C": A2C,
        "DDPG": DDPG,
        "TD3": TD3,
        "DQN": DQN,
    }
    if algorithm is None:
        return BaseAlgorithm
    key = str(algorithm).upper()
    if key not in mapping:
        raise ValueError(f"Unsupported SB3 algorithm '{algorithm}'. Choose from {list(mapping)}")
    return mapping[key]


def load_dt_model(path: str, device: torch.device):
    try:
        from models.model import TrainableDT
    except ImportError:
        from src.models.model import TrainableDT

    model = TrainableDT.from_pretrained(path)
    model.to(device)
    model.eval()
    return model


def register_rl_factory_shim() -> None:
    """Register CustomCNN under rl_factory.utils.feature_extractors for SB3 unpickling."""
    import importlib.util
    import types

    if "rl_factory.utils.feature_extractors" in sys.modules:
        return
    shim_file = Path(__file__).resolve().parents[1] / "rl_factory_shim" / "feature_extractors.py"
    rf = types.ModuleType("rl_factory")
    rf_utils = types.ModuleType("rl_factory.utils")
    sys.modules["rl_factory"] = rf
    sys.modules["rl_factory.utils"] = rf_utils
    spec = importlib.util.spec_from_file_location(
        "rl_factory.utils.feature_extractors", shim_file
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["rl_factory.utils.feature_extractors"] = mod
    spec.loader.exec_module(mod)
    rf_utils.feature_extractors = mod


def load_sb3_model(path: str, algorithm: Optional[str], device: torch.device):
    register_rl_factory_shim()
    cls = _resolve_sb3_class(algorithm)
    # SB3 load accepts zip path; device via policy kwargs after load
    model = cls.load(path, device=str(device))
    model.policy.eval()
    return model


def load_model(spec: ModelSpec, device: Optional[torch.device] = None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = enrich_spec_from_metadata(spec)
    family = spec.normalized_type()
    if family == "DT":
        return load_dt_model(spec.path, device), spec
    model = load_sb3_model(spec.path, spec.algorithm, device)
    # Trust the checkpoint observation space over metadata/folder guesses
    try:
        shape = tuple(int(x) for x in model.observation_space.shape)
        spec.obs_shape = shape
        if str(spec.feature_extractor).upper() == "CNN" and len(shape) >= 2:
            spec.lookback = int(shape[1])
        elif len(shape) >= 1:
            # MLP Box is typically (lookback, n_features)
            spec.lookback = int(shape[0])
    except Exception:
        pass
    return model, spec


def model_capabilities(spec: ModelSpec) -> Dict[str, bool]:
    family = spec.normalized_type()
    return {
        "family": family == "DT",  # kept for backwards clarity below
        "is_dt": family == "DT",
        "is_sb3": family == "SB3",
        "supports_rtg": family == "DT",
        "supports_timesteps": family == "DT",
        "supports_layer_attribution": family == "DT",
        "supports_head_specialization": family == "DT",
        "supports_feature_attribution": True,
        "supports_temporal_occlusion": True,
        "supports_latent_umap": True,
        "feature_extractor": spec.feature_extractor if family == "SB3" else "DT",
    }


# ---------------------------------------------------------------------------
# Observation builders (MLP / CNN)
# ---------------------------------------------------------------------------

def build_observation_frame(
    data: pd.DataFrame,
    indicators: pd.DataFrame,
    drop_cash: bool = False,
) -> pd.DataFrame:
    """
    Rebuild training observations: technical indicators merged with OHLC prices.
    Matches ECI sb3_trainer with use_prices_as_indicators=True.
    """
    merged = indicators.merge(data, right_index=True, left_index=True)
    if drop_cash:
        cash_cols = [c for c in merged.columns if isinstance(c, tuple) and c[0] == "CASH"]
        merged = merged.drop(columns=cash_cols)
    return merged.dropna()


def observation_frame_to_cnn_tensor(frame: pd.DataFrame) -> torch.Tensor:
    """
    Port of PortfolioOptimizationEnv.return_obs_frame_as_tensor:
    channels = instruments, height = lookback, width = indicator/feature names.
    """
    import collections

    indicator_names = [i[1] for i in frame.columns]
    occurrences = collections.Counter(indicator_names)
    multiple = [k for k, v in occurrences.items() if v > 1]
    single = [k for k, v in occurrences.items() if v == 1]
    indicator_columns = [i for i in frame.columns if i[1] in multiple]
    global_columns = [i for i in frame.columns if i[1] in single]
    ind_names = list(dict.fromkeys([i[1] for i in indicator_columns]))
    inst_names = list(dict.fromkeys([i[0] for i in indicator_columns]))

    indicators_tensor = torch.stack(
        [torch.as_tensor(frame.loc[:, [(i, j) for j in ind_names if (i, j) in frame.columns]].values, dtype=torch.float32)
         for i in inst_names],
        dim=0,
    )
    # shape: (n_inst, lookback, n_feats) — ensure feature order
    # rebuild explicitly
    mats = []
    for i in inst_names:
        cols = [(i, j) for j in ind_names if (i, j) in frame.columns]
        mats.append(torch.as_tensor(frame.loc[:, cols].values, dtype=torch.float32))
    indicators_tensor = torch.stack(mats, dim=0)

    if global_columns:
        global_tensor = torch.as_tensor(frame.loc[:, global_columns].values, dtype=torch.float32)
        global_tensor = global_tensor.unsqueeze(0).expand(len(inst_names), -1, -1)
        return torch.cat([indicators_tensor, global_tensor], dim=2)
    return indicators_tensor


def observation_frame_to_tensor(
    frame: pd.DataFrame,
    feature_extractor: FeatureExtractorKind | str = "MLP",
    cnn_layout: str = "features_x_time",
    custom_transform: Optional[Callable[[pd.DataFrame], torch.Tensor]] = None,
) -> torch.Tensor:
    """
    Convert a lookback observation frame to the tensor expected by the policy.

    MLP: (lookback, n_features) float tensor.
    CNN: (n_instruments, lookback, n_features) via rl_factory tensor layout.
    """
    if custom_transform is not None:
        tensor = custom_transform(frame)
        return tensor if isinstance(tensor, torch.Tensor) else torch.as_tensor(tensor, dtype=torch.float32)

    kind = str(feature_extractor).upper()
    if kind == "MLP":
        return torch.from_numpy(frame.values.astype(np.float32))

    if kind != "CNN":
        raise ValueError(f"Unknown feature_extractor '{feature_extractor}'")

    # Prefer rl_factory tensor layout (channels=instruments)
    try:
        return observation_frame_to_cnn_tensor(frame)
    except Exception:
        values = frame.values.astype(np.float32)
        if cnn_layout == "features_x_time":
            return torch.from_numpy(values.T)
        if cnn_layout == "time_x_features":
            return torch.from_numpy(values).unsqueeze(0)
        if cnn_layout == "raw":
            return torch.from_numpy(values)
        raise


def flatten_obs_for_attribution(obs: torch.Tensor) -> torch.Tensor:
    """Flatten observation to (batch, 1, state_dim) for lag-wise attribution APIs."""
    if obs.dim() == 2:
        # (lookback, feat) or (batch, feat)
        return obs.reshape(1, 1, -1)
    if obs.dim() == 3:
        # (C,H,W) or (B, feat_dim) already batched oddly
        if obs.shape[0] == 1 and obs.shape[1] > 1:
            return obs.reshape(1, 1, -1)
        return obs.reshape(obs.shape[0], 1, -1)
    if obs.dim() == 4:
        # (B, C, H, W)
        b = obs.shape[0]
        return obs.reshape(b, 1, -1)
    return obs.reshape(1, 1, -1)


# ---------------------------------------------------------------------------
# SB3 differentiable policy wrapper
# ---------------------------------------------------------------------------

class SB3PolicyWrapper(nn.Module):
    """
    Differentiable wrapper around an SB3 policy for Captum attribution.

    Returns a continuous weight-like vector of shape (batch, act_dim):
      - continuous: raw / normalized action mean
      - discrete: softmax over action logits / Q-values
    """

    def __init__(
        self,
        sb3_model,
        action_space: ActionSpaceKind = "continuous",
        n_instruments: Optional[int] = None,
        normalize_weights: bool = True,
    ):
        super().__init__()
        self.policy = sb3_model.policy
        self.action_space = action_space
        self.n_instruments = n_instruments
        self.normalize_weights = normalize_weights
        self._algo = type(sb3_model).__name__.upper()

    def _actor(self):
        """Off-policy continuous policies expose the Actor under policy.actor."""
        return getattr(self.policy, "actor", None)

    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        # SB3>=2.x: extract_features(obs, features_extractor) is required on many policies.
        # Prefer an explicit features_extractor module call as a robust fallback.
        policy = self.policy
        actor = self._actor()
        q_net = getattr(policy, "q_net", None)

        def _call(mod) -> Optional[torch.Tensor]:
            if mod is None:
                return None
            fe = getattr(mod, "features_extractor", None)
            if fe is None and mod is not policy:
                fe = getattr(policy, "features_extractor", None)
            if fe is not None:
                try:
                    feats = mod.extract_features(obs, fe)
                except TypeError:
                    feats = fe(obs)
                except Exception:
                    feats = fe(obs)
            else:
                try:
                    feats = mod.extract_features(obs)
                except TypeError:
                    return None
            if isinstance(feats, tuple):
                feats = feats[0]
            return feats

        for mod in (actor, q_net, policy):
            feats = _call(mod)
            if feats is not None:
                return feats
        raise RuntimeError(f"Could not extract features for algorithm={self._algo}")

    def _normalize_continuous(self, weights: torch.Tensor) -> torch.Tensor:
        if not self.normalize_weights:
            return weights
        if (weights < 0).any():
            weights = torch.relu(weights)
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return weights / denom

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        obs = obs.float()
        policy = self.policy

        # DQN: use Q-network logits → softmax soft assignment
        if self._algo == "DQN" or (self.action_space == "discrete" and hasattr(policy, "q_net")):
            q_net = getattr(policy, "q_net", None)
            if q_net is not None:
                q_values = q_net(obs)
            else:
                features = self._features(obs)
                q_values = policy.action_net(features) if hasattr(policy, "action_net") else features
            return torch.softmax(q_values, dim=-1)

        # DDPG / TD3 / SAC: differentiable path goes through policy.actor
        actor = self._actor()
        if actor is not None and self.action_space == "continuous":
            if self._algo == "SAC":
                actions = actor(obs, deterministic=True)
            else:
                actions = actor(obs)
            return self._normalize_continuous(actions)

        features = self._features(obs)
        if hasattr(policy, "mlp_extractor"):
            latent_pi, _latent_vf = policy.mlp_extractor(features)
        else:
            latent_pi = features

        if self.action_space == "discrete":
            logits = policy.action_net(latent_pi)
            return torch.softmax(logits, dim=-1)

        actions = policy.action_net(latent_pi)
        if hasattr(policy, "squash_output") and policy.squash_output:
            actions = torch.tanh(actions)
        return self._normalize_continuous(actions)

def action_to_weights(
    action: Any,
    instruments: List[str],
    action_space: ActionSpaceKind = "continuous",
) -> np.ndarray:
    """Map SB3 predict() output to a portfolio weight vector."""
    if isinstance(action, tuple):
        action = action[0]
    action = np.asarray(action).reshape(-1)
    if action_space == "discrete":
        weights = np.zeros(len(instruments), dtype=np.float64)
        idx = int(action[0]) if action.size else 0
        idx = min(max(idx, 0), len(instruments) - 1)
        weights[idx] = 1.0
        return weights
    weights = action.astype(np.float64)
    if weights.size != len(instruments):
        # pad / truncate
        out = np.zeros(len(instruments), dtype=np.float64)
        n = min(len(instruments), weights.size)
        out[:n] = weights[:n]
        weights = out
    if weights.sum() == 0:
        weights[:] = 1.0 / len(instruments)
    else:
        weights = weights / weights.sum()
    return weights


# ---------------------------------------------------------------------------
# Unified forward for attribution
# ---------------------------------------------------------------------------

def forward_actions(
    model,
    spec: ModelSpec,
    states: torch.Tensor,
    actions: Optional[torch.Tensor] = None,
    returns_to_go: Optional[torch.Tensor] = None,
    timesteps: Optional[torch.Tensor] = None,
    target_action_idx: Optional[int] = None,
    sb3_wrapper: Optional[SB3PolicyWrapper] = None,
) -> torch.Tensor:
    """Family-aware forward returning action/weight predictions (batch, act_dim) or scalar slice."""
    family = spec.normalized_type()
    if family == "DT":
        from . import _forward_action

        return _forward_action(
            model,
            states,
            actions,
            returns_to_go,
            timesteps,
            target_action_idx=target_action_idx,
        )

    wrapper = sb3_wrapper or SB3PolicyWrapper(
        model,
        action_space=spec.action_space,
        n_instruments=len(spec.instruments) if spec.instruments else None,
    )
    # states for attribution are flat (B, 1, D); reshape to policy obs if possible
    obs = _states_to_sb3_obs(states, spec)
    preds = wrapper(obs)
    if target_action_idx is not None:
        return preds[:, target_action_idx]
    return preds


def _states_to_sb3_obs(states: torch.Tensor, spec: ModelSpec) -> torch.Tensor:
    """
    Convert flat attribution states (B, 1, state_dim) back to SB3 observation layout.
    Flat vectors must be the C-order flatten of the policy observation tensor.
    """
    if states.dim() == 3 and states.shape[1] == 1:
        flat = states[:, 0, :]
    elif states.dim() == 2:
        flat = states
    else:
        flat = states.reshape(states.shape[0], -1)

    if str(spec.feature_extractor).upper() == "CNN":
        shape = spec.obs_shape
        if shape is None:
            lookback = int(spec.lookback or 1)
            # Fallback: assume (n_inst, lookback, n_feat) with n_inst == n_feat
            # when flat size = lookback * k^2 (common in this project: 6xTx6).
            dim = int(flat.shape[-1])
            if lookback > 0 and dim % lookback == 0:
                spatial = dim // lookback
                root = int(round(spatial ** 0.5))
                if root * root == spatial:
                    shape = (root, lookback, root)
        if shape is not None and flat.shape[-1] == int(np.prod(shape)):
            return flat.view(flat.shape[0], *shape)
        raise ValueError(
            f"Cannot reshape CNN flat states of dim {flat.shape[-1]} "
            f"to obs_shape={shape} (lookback={spec.lookback})"
        )

    # MLP: FlattenExtractor accepts (B, lookback * n_feat)
    return flat


# ---------------------------------------------------------------------------
# Ablation batch preparation
# ---------------------------------------------------------------------------

def build_eval_state_frame(
    data: pd.DataFrame,
    indicators: pd.DataFrame,
    env_state: Optional[List[Any]] = None,
    drop_cash: bool = False,
) -> pd.DataFrame:
    """Build observation frame; optionally subset to env_state columns."""
    frame = build_observation_frame(data, indicators, drop_cash=drop_cash)
    if env_state:
        cols = []
        for item in env_state:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                cols.append((item[0], item[1]))
            else:
                cols.append(item)
        missing = [c for c in cols if c not in frame.columns]
        if missing:
            raise KeyError(f"env_state columns missing from observation frame: {missing[:5]}...")
        frame = frame.loc[:, cols]
    return frame


def prepare_ablation_batch_dt(
    model,
    spec: ModelSpec,
    eval_data: pd.DataFrame,
    ohlc_data: pd.DataFrame,
    device: torch.device,
    ref_date: Optional[str] = None,
) -> Dict[str, Any]:
    ref = ref_date or spec.reference_date or DEFAULT_PERIODS["eval"][0]
    batch = prepare_ablation_inputs(
        model=model,
        backtest_data=eval_data,
        ref_date=ref,
        ohlc_data=ohlc_data,
        lookback=spec.lookback or 16,
        rebalance_every=spec.rebalance_every,
        annualized_return=spec.target_return,
        use_trajectory_return_objective=False,
        use_incremental_timesteps=True,
        ppy=252,
        device=device,
    )
    batch["family"] = "DT"
    batch["label"] = spec.label
    return batch


def prepare_ablation_batch_sb3(
    model,
    spec: ModelSpec,
    eval_data: pd.DataFrame,
    ohlc_data: pd.DataFrame,
    device: torch.device,
    ref_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Walk the evaluation window and collect states / predicted actions for SB3.
    Mirrors the user evaluation loop while producing a DT-compatible batch schema.
    """
    instruments = list(spec.instruments or [])
    if not instruments:
        raise ValueError(f"SB3 model '{spec.label}' requires 'instruments' in the model spec/metadata")
    lookback = int(spec.lookback or 16)
    ref = pd.to_datetime(ref_date or spec.reference_date or DEFAULT_PERIODS["eval"][0])

    feature_mapping = create_feature_mapping_from_labels(eval_data.columns, lookback)
    R_h, R_b, R_s = create_return_matrices(ohlc_data)
    for R in (R_h, R_b, R_s):
        R.index = pd.to_datetime(R.index)

    states_list: List[torch.Tensor] = []
    actions_list: List[torch.Tensor] = []
    dates: List[pd.Timestamp] = []
    decision_series: Dict[pd.Timestamp, np.ndarray] = {}

    wrapper = SB3PolicyWrapper(
        model,
        action_space=spec.action_space,
        n_instruments=len(instruments),
    ).to(device)
    wrapper.eval()

    walk = eval_data.loc[ref:].dropna()
    # need lookback history before ref; use full eval_data for windows
    for idx, _row in tqdm(walk.iterrows(), total=len(walk), desc=f"SB3 batch [{spec.label}]"):
        observation_frame = eval_data.loc[:idx, :].tail(lookback)
        if observation_frame.shape[0] < lookback:
            continue

        obs = observation_frame_to_tensor(
            observation_frame,
            feature_extractor=spec.feature_extractor,
            cnn_layout=spec.cnn_layout,
            custom_transform=spec.custom_obs_transform,
        ).to(device)

        with torch.no_grad():
            # Prefer differentiable wrapper; fall back to SB3 predict()
            try:
                obs_b = obs if obs.dim() >= 3 and obs.shape[0] == 1 else obs.unsqueeze(0)
                # MLP (T, F) -> batch (1, T, F); CNN (C, T) -> (1, C, T)
                pred = wrapper(obs_b)
                if pred.dim() == 1:
                    pred = pred.unsqueeze(0)
                weights = pred[0].detach().cpu().numpy()
            except Exception:
                action = model.predict(
                    obs.detach().cpu().numpy(),
                    deterministic=True,
                )
                weights = action_to_weights(action, instruments, spec.action_space)

        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if weights.size != len(instruments):
            weights = action_to_weights(weights, instruments, spec.action_space)
        if weights.sum() != 0:
            weights = weights / weights.sum()

        # Flat state for attribution: C-order flatten of the policy observation
        flat_state = obs.detach().cpu().float().reshape(1, 1, -1)
        states_list.append(flat_state)
        actions_list.append(torch.from_numpy(weights.astype(np.float32)).view(1, 1, -1))
        dates.append(pd.to_datetime(idx))
        decision_series[pd.to_datetime(idx)] = weights

    if not states_list:
        raise RuntimeError(f"No SB3 ablation samples collected for '{spec.label}'")

    n = len(states_list)
    dummy_rtg = torch.zeros(n, 1, dtype=torch.float32)
    dummy_ts = torch.arange(n, dtype=torch.long).view(n, 1)
    dummy_mask = torch.ones(n, 1, dtype=torch.float32)
    dummy_rewards = torch.zeros(n, dtype=torch.float32)

    df_weights = pd.DataFrame.from_dict(decision_series, orient="index", columns=instruments)
    df_weights = df_weights.div(df_weights.sum(axis=1), axis=0)
    df_weights.index = pd.to_datetime(df_weights.index)

    return {
        "states": torch.cat(states_list, dim=0),
        "actions": torch.cat(actions_list, dim=0),
        "returns_to_go": dummy_rtg,
        "timesteps": dummy_ts,
        "attention_mask": dummy_mask,
        "rewards": dummy_rewards,
        "instrument_returns": R_h.loc[ref:] if len(R_h.loc[ref:]) else R_h,
        "labels": feature_mapping,
        "dates": pd.DatetimeIndex(dates),
        "weights": df_weights,
        "family": "SB3",
        "label": spec.label,
        "sb3_wrapper": wrapper,
    }


def prepare_ablation_batch(
    model,
    spec: ModelSpec,
    eval_data: pd.DataFrame,
    ohlc_data: pd.DataFrame,
    device: torch.device,
    ref_date: Optional[str] = None,
) -> Dict[str, Any]:
    if spec.normalized_type() == "DT":
        return prepare_ablation_batch_dt(model, spec, eval_data, ohlc_data, device, ref_date)
    return prepare_ablation_batch_sb3(model, spec, eval_data, ohlc_data, device, ref_date)


# ---------------------------------------------------------------------------
# Weight / strategy evaluation
# ---------------------------------------------------------------------------

def compute_strategy_returns_from_weights(
    df_weights: pd.DataFrame,
    ohlc_data: pd.DataFrame,
    instruments: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Portfolio returns via hold/buy/sell decomposition when possible, else close-to-close.
    Returns (strategy_returns_df, comparison_df).
    """
    data = ohlc_data.copy()
    data.index = pd.to_datetime(data.index)
    df_weights = df_weights.copy()
    df_weights.index = pd.to_datetime(df_weights.index)

    # Select OHLC MultiIndex columns for traded names (CASH has flat returns = 0)
    traded = [i for i in instruments if i != "CASH"]
    if isinstance(data.columns, pd.MultiIndex):
        ohlc_cols = [c for c in data.columns if c[0] in traded]
        price_frame = data.loc[:, ohlc_cols].ffill().dropna()
    else:
        price_frame = data.loc[:, [c for c in traded if c in data.columns]].ffill().dropna()

    try:
        from rl_factory.common import decompose_weights_tensor  # type: ignore
    except Exception:
        try:
            from rl_factory.envs.common import decompose_weights_tensor  # type: ignore
        except Exception:
            decompose_weights_tensor = None

    R_h, R_b, R_s = create_return_matrices(price_frame)
    R_h.index = pd.to_datetime(R_h.index)
    R_b.index = pd.to_datetime(R_b.index)
    R_s.index = pd.to_datetime(R_s.index)

    # Align weight columns to return matrix instrument order (+ CASH=0)
    ret_instruments = list(R_h.columns)
    weight_cols = [c for c in ret_instruments if c in df_weights.columns]
    if "CASH" in df_weights.columns and "CASH" not in weight_cols:
        # keep CASH in weights but give zero asset return
        pass

    returns: Dict[pd.Timestamp, float] = {}
    current_weights = np.zeros(len(ret_instruments), dtype=np.float64)

    for idx, row in df_weights.iterrows():
        if idx not in R_h.index:
            continue
        new_weights = np.array([float(row.get(c, 0.0)) for c in ret_instruments], dtype=np.float64)
        if decompose_weights_tensor is not None:
            w_h, w_b, w_s = decompose_weights_tensor(
                torch.tensor(new_weights, dtype=torch.float32),
                torch.tensor(current_weights, dtype=torch.float32),
            )
            pf = (
                torch.dot(w_h, torch.tensor(R_h.loc[idx].values, dtype=torch.float32))
                + torch.dot(w_b, torch.tensor(R_b.loc[idx].values, dtype=torch.float32))
                + torch.dot(w_s, torch.tensor(R_s.loc[idx].values, dtype=torch.float32))
            )
            returns[idx] = float(pf.detach().numpy())
        else:
            r = float(np.dot(current_weights, R_h.loc[idx].values)) if current_weights.sum() else 0.0
            returns[idx] = r
        current_weights = new_weights

    df_strategy = pd.DataFrame.from_dict(returns, orient="index", columns=["strategy_returns"])
    df_strategy.index = pd.to_datetime(df_strategy.index)

    close_cols = [(i, "Close") for i in ret_instruments if (i, "Close") in data.columns]
    if close_cols:
        df_bh = data.loc[:, close_cols].pct_change(fill_method=None)
        df_bh.columns = [c[0] for c in df_bh.columns]
    else:
        df_bh = pd.DataFrame(index=df_strategy.index)
    df_bh["equal_weights"] = df_bh.mean(axis=1) if df_bh.shape[1] else 0.0
    df_bh.index = pd.to_datetime(df_bh.index)
    df_comparison = df_bh.merge(df_strategy, right_index=True, left_index=True)
    return df_strategy, df_comparison


def _weights_from_dt_batch(batch: Dict[str, Any], instruments: List[str]) -> pd.DataFrame:
    """Build a weight DataFrame from DT ablation-batch action predictions."""
    actions = batch["actions"].detach().cpu().numpy()
    # actions: (N, 1, act_dim) or (N, act_dim)
    if actions.ndim == 3:
        actions = actions[:, -1, :]
    elif actions.ndim == 1:
        actions = actions.reshape(1, -1)
    dates = pd.to_datetime(batch["dates"])
    n = min(len(dates), actions.shape[0])
    df = pd.DataFrame(actions[:n], index=dates[:n], columns=list(instruments)[: actions.shape[1]])
    # align columns if act_dim matches instruments
    if df.shape[1] != len(instruments):
        # pad / trim
        for col in instruments:
            if col not in df.columns:
                df[col] = 0.0
        df = df.loc[:, instruments]
    row_sums = df.sum(axis=1).replace(0, np.nan)
    df = df.div(row_sums, axis=0).fillna(1.0 / len(instruments))
    return df


def evaluate_model_weights(
    model,
    spec: ModelSpec,
    eval_data: pd.DataFrame,
    ohlc_data: pd.DataFrame,
    device: torch.device,
    ref_date: Optional[str] = None,
    batch: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Produce weights + strategy comparison for DT or SB3."""
    instruments = list(spec.instruments or [])
    if batch is not None and "weights" in batch:
        df_weights = batch["weights"]
    elif spec.normalized_type() == "SB3":
        batch = prepare_ablation_batch_sb3(model, spec, eval_data, ohlc_data, device, ref_date)
        df_weights = batch["weights"]
    else:
        if batch is None:
            batch = prepare_ablation_batch_dt(model, spec, eval_data, ohlc_data, device, ref_date)
        df_weights = _weights_from_dt_batch(batch, instruments)

    df_strategy, df_comparison = compute_strategy_returns_from_weights(
        df_weights, ohlc_data, instruments
    )
    return {
        "weights": df_weights,
        "returns": df_strategy,
        "comparison": df_comparison,
        "batch": batch,
    }


# ---------------------------------------------------------------------------
# Attribution helpers
# ---------------------------------------------------------------------------

def feature_importance_multifamily(
    model,
    spec: ModelSpec,
    batch: Dict[str, Any],
    baseline: Optional[torch.Tensor] = None,
    target_action_idx: Optional[int] = None,
) -> torch.Tensor:
    states = batch["states"].clone().detach().requires_grad_(True)
    sb3_wrapper = batch.get("sb3_wrapper")

    def forward_wrapper(input_states):
        return forward_actions(
            model,
            spec,
            input_states,
            actions=batch.get("actions"),
            returns_to_go=batch.get("returns_to_go"),
            timesteps=batch.get("timesteps"),
            target_action_idx=target_action_idx,
            sb3_wrapper=sb3_wrapper,
        )

    ablator = FeatureAblation(forward_wrapper)
    if baseline is None:
        baseline = torch.zeros_like(states)
    return ablator.attribute(states, baselines=baseline)


def decision_drivers_multifamily(
    model,
    spec: ModelSpec,
    batch: Dict[str, Any],
    target_action_idx: Optional[int] = None,
) -> Dict[str, Any]:
    states = batch["states"].clone().detach().requires_grad_(True)
    sb3_wrapper = batch.get("sb3_wrapper")

    def forward_wrapper(input_states):
        out = forward_actions(
            model,
            spec,
            input_states,
            actions=batch.get("actions"),
            returns_to_go=batch.get("returns_to_go"),
            timesteps=batch.get("timesteps"),
            target_action_idx=target_action_idx,
            sb3_wrapper=sb3_wrapper,
        )
        return out if target_action_idx is not None else out.sum(dim=-1)

    ig = IntegratedGradients(forward_wrapper)
    baseline = torch.zeros_like(states)
    attributions, delta = ig.attribute(states, baselines=baseline, return_convergence_delta=True)
    return {"attributions": attributions, "delta": delta}


def extract_latents(
    model,
    spec: ModelSpec,
    batch: Dict[str, Any],
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Extract latent representations:
      - DT: last transformer hidden state
      - SB3: policy feature extractor (+ mlp latent_pi when available)
    """
    device = device or batch["states"].device
    family = spec.normalized_type()
    states = batch["states"].to(device)

    if family == "DT":
        from . import latent_space_market_regime

        out = latent_space_market_regime(
            model,
            states,
            batch["actions"].to(device),
            batch["returns_to_go"].to(device),
            batch["timesteps"].to(device),
        )
        return out["hidden"]

    wrapper = batch.get("sb3_wrapper") or SB3PolicyWrapper(
        model, action_space=spec.action_space, n_instruments=len(spec.instruments or [])
    )
    wrapper = wrapper.to(device)
    wrapper.eval()
    latents = []
    with torch.no_grad():
        for i in range(states.shape[0]):
            obs = _states_to_sb3_obs(states[i : i + 1], spec)
            feats = wrapper._features(obs)
            if hasattr(wrapper.policy, "mlp_extractor"):
                latent_pi, _ = wrapper.policy.mlp_extractor(feats)
                latents.append(latent_pi.detach().cpu().numpy().reshape(-1))
            else:
                latents.append(feats.detach().cpu().numpy().reshape(-1))
    return np.stack(latents, axis=0)


def temporal_occlusion_multifamily(
    model,
    spec: ModelSpec,
    batch: Dict[str, Any],
    reshaped_states: torch.Tensor,
    regimes: Optional[np.ndarray] = None,
    target_action_idx: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Lag-wise occlusion using the family-aware forward.
    reshaped_states: (N, n_indicators, n_instruments, lags) or compatible.
    """
    states = batch["states"]
    n, _, state_dim = states.shape[0], states.shape[1], states.shape[-1]
    time_major = reshaped_states.dim() == 3  # (N, lookback, n_features)
    lags = reshaped_states.shape[1] if time_major else reshaped_states.shape[-1]
    sb3_wrapper = batch.get("sb3_wrapper")

    baseline = forward_actions(
        model,
        spec,
        states,
        batch.get("actions"),
        batch.get("returns_to_go"),
        batch.get("timesteps"),
        target_action_idx=target_action_idx,
        sb3_wrapper=sb3_wrapper,
    )
    if target_action_idx is not None:
        baseline = baseline.unsqueeze(-1)

    occlusion_effects = torch.zeros(n, baseline.shape[-1], lags)
    for t in range(lags):
        ablated = reshaped_states.clone()
        if time_major:
            ablated[:, t, :] = 0
            ablated_flat = ablated.reshape(n, 1, -1)
        else:
            ablated[:, :, :, t] = 0
            ablated_flat = ablated.reshape(n, 1, -1)
        # pad / trim to state_dim
        if ablated_flat.shape[-1] < state_dim:
            pad = torch.zeros(n, 1, state_dim - ablated_flat.shape[-1], device=ablated_flat.device)
            ablated_flat = torch.cat([pad, ablated_flat], dim=-1)
        elif ablated_flat.shape[-1] > state_dim:
            ablated_flat = ablated_flat[..., -state_dim:]
        ablated_out = forward_actions(
            model,
            spec,
            ablated_flat,
            batch.get("actions"),
            batch.get("returns_to_go"),
            batch.get("timesteps"),
            target_action_idx=target_action_idx,
            sb3_wrapper=sb3_wrapper,
        )
        if target_action_idx is not None:
            ablated_out = ablated_out.unsqueeze(-1)
        occlusion_effects[:, :, t] = (baseline - ablated_out).abs().detach().cpu()

    # For time-major MLP windows, index 0 is oldest; flip so lag 0 = most recent
    if time_major:
        occlusion_effects = occlusion_effects.flip(dims=[-1])

    result = {
        "occlusion_curves": occlusion_effects,
        "lags": lags,
        "family": spec.normalized_type(),
    }
    if regimes is not None:
        result["regimes"] = regimes
    return result


def save_evaluation_outputs(
    output_path: str,
    model_key: str,
    weights: pd.DataFrame,
    comparison: pd.DataFrame,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    os.makedirs(f"{output_path}/results", exist_ok=True)
    weights.to_csv(f"{output_path}/results/weights_{model_key}.csv")
    comparison.to_csv(f"{output_path}/results/strategy_comparison_{model_key}.csv")
    if metadata is not None:
        with open(f"{output_path}/metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, default=str, indent=2)
