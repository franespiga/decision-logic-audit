"""
Family-specific policy adapters exposing one common diagnostic interface.

Every adapter provides, for the same ``(N, window, n_features)`` observation
tensor:

* ``portfolio_weights``  the long-only, fully invested allocation actually implied
  by the policy's normal action-to-weight transformation;
* ``decision_target``    a differentiable scalar per observation that tracks the
  policy's actual decision (see ``DECISION_TARGETS``);
* ``latent``             a decision-proximal internal representation;
* ``layer_modules``      the comparable decision-processing layers.

Decision targets, per algorithm
-------------------------------
continuous simplex policies (Decision Transformer, DDPG)
    ``tau(x) = sum_k w_hat_k * w_k(x)`` where ``w_hat`` is the unperturbed
    allocation, detached. This is the allocation-aligned weight mass: it is
    differentiable, bounded in [0, 1], and weights each action head by the
    capital it actually receives instead of averaging indiscriminately over all
    heads.
discrete policies (DQN, PPO)
    ``tau(x) = q_sel(x)`` / ``tau(x) = logit_sel(x)`` for the action selected by
    the frozen deterministic policy at the unperturbed observation. The selected
    action is mapped to its resulting one-hot allocation for weight-based
    measures.

Complementary target
--------------------
Where a policy's squashing non-linearity is saturated, the primary target is
locally constant and gradient-based attribution is identically zero. For such
policies a pre-squash target is additionally exposed as
``decision_target(..., variant="complementary")``. Its use is recorded per model;
it is never silently substituted for the primary target.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .discovery import CANONICAL_INSTRUMENTS, ModelRecord

OFF_POLICY: Tuple[str, ...] = ("DQN", "DDPG", "TD3", "SAC")

DECISION_TARGETS: Dict[str, Dict[str, str]] = {
    "DecisionTransformer": {
        "action_space": "continuous (7-simplex via softmax head)",
        "primary_target": "allocation-aligned weight mass sum_k w_hat_k w_k(x)",
        "weight_mapping": "softmax action head is already a long-only fully invested simplex",
        "complementary_target": "not required (head is not saturating)",
    },
    "DDPG": {
        "action_space": "continuous Box(0, 1)^7",
        "primary_target": "allocation-aligned weight mass sum_k w_hat_k w_k(x)",
        "weight_mapping": "unscale_action((tanh output)) = (a+1)/2, then normalised to sum to one",
        "complementary_target": "allocation-aligned pre-tanh activation sum_k w_hat_k z_k(x)",
    },
    "DQN": {
        "action_space": "Discrete(7)",
        "primary_target": "Q value of the greedily selected action",
        "weight_mapping": "argmax action index -> one-hot allocation over the instrument universe",
        "complementary_target": "not required (Q head is linear)",
    },
    "PPO": {
        "action_space": "Discrete(7)",
        "primary_target": "policy logit of the greedily selected action",
        "weight_mapping": "argmax action index -> one-hot allocation over the instrument universe",
        "complementary_target": "not required (logit head is linear)",
    },
}


def tile_context(context: Dict[str, torch.Tensor], reps: int) -> Dict[str, torch.Tensor]:
    """
    Repeat auxiliary and reference tensors along the batch axis.

    Perturbation and path-integration methods evaluate several copies of every
    observation in one forward pass. The policy context (previous action,
    return-to-go, timestep, reference allocation, selected action) must be tiled
    the same way so that each row stays paired with its own observation.
    """
    if reps <= 1:
        return context
    out: Dict[str, torch.Tensor] = {}
    for key, value in context.items():
        if value.dim() > 1:
            out[key] = value.repeat(reps, *([1] * (value.dim() - 1)))
        else:
            out[key] = value.repeat(reps)
    return out


@dataclass
class Rollout:
    """Per-date observations, auxiliary policy context and realised allocations."""

    dates: pd.DatetimeIndex
    states: torch.Tensor  # (N, window, n_features), float32, CPU
    weights: pd.DataFrame  # (N, 7) canonical instrument order
    aux: Dict[str, torch.Tensor] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.states.shape[0])

    def select(self, positions: Sequence[int]) -> "Rollout":
        pos = np.asarray(positions, dtype=int)
        return Rollout(
            dates=pd.DatetimeIndex(self.dates[pos]),
            states=self.states[pos].clone(),
            weights=self.weights.iloc[pos].copy(),
            aux={k: v[pos].clone() for k, v in self.aux.items()},
        )


class PolicyAdapter(ABC):
    """Common diagnostic interface over a frozen policy."""

    def __init__(self, record: ModelRecord, device: torch.device):
        self.record = record
        self.device = device
        self.window = int(record.window_length)
        self.feature_labels: List[Tuple[str, str]] = [tuple(f) for f in record.state_features]
        self.n_features = len(self.feature_labels)
        self.action_instruments: List[str] = list(record.instruments)
        self._canonical_index = [self.action_instruments.index(i) for i in CANONICAL_INSTRUMENTS]

    # -- observations -------------------------------------------------------

    def observation_frame(self, merged: pd.DataFrame) -> pd.DataFrame:
        """Subset the merged market frame to this policy's observation columns."""
        missing = [c for c in self.feature_labels if c not in merged.columns]
        if missing:
            raise KeyError(f"{self.record.model_id}: missing observation columns {missing[:5]}")
        return merged.loc[:, self.feature_labels]

    def buildable_dates(self, frame: pd.DataFrame, dates: Sequence[pd.Timestamp]) -> List[pd.Timestamp]:
        """Dates for which a complete observation window ending on that date exists."""
        out = []
        for dt in dates:
            if dt not in frame.index:
                continue
            if frame.index.get_loc(dt) - self.window + 1 >= 0:
                out.append(dt)
        return out

    def build_states(self, frame: pd.DataFrame, dates: Sequence[pd.Timestamp]) -> torch.Tensor:
        """
        Stack observation windows ending on each date.

        The returned tensor is time-major with row 0 the oldest observation, which
        is the layout consumed by both the DT (``window.values`` flattened in
        C order) and the SB3 ``FlattenExtractor``.
        """
        blocks = []
        for dt in dates:
            end = frame.index.get_loc(dt)
            start = end - self.window + 1
            if start < 0:
                raise ValueError(f"{self.record.model_id}: incomplete window at {dt}")
            blocks.append(frame.iloc[start : end + 1].values)
        arr = np.stack(blocks).astype(np.float32)
        return torch.from_numpy(arr)

    def lag_labels(self) -> np.ndarray:
        """Relative lag index per observation row; 0 is the most recent row."""
        return np.arange(self.window - 1, -1, -1)

    # -- policy interface --------------------------------------------------

    @abstractmethod
    def _forward_weights(self, states: torch.Tensor, aux: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Differentiable allocation in this policy's own action order, (N, A)."""

    @abstractmethod
    def latent(self, states: torch.Tensor, aux: Dict[str, torch.Tensor]) -> np.ndarray:
        """Decision-proximal internal representation, (N, d)."""

    @abstractmethod
    def layer_modules(self) -> Dict[str, nn.Module]:
        """Ordered comparable decision-processing layers."""

    @abstractmethod
    def latent_layer_name(self) -> str:
        """Human-readable description of the layer used for ``latent``."""

    def is_discrete(self) -> bool:
        return self.record.action_space == "discrete"

    def portfolio_weights(
        self, states: torch.Tensor, aux: Optional[Dict[str, torch.Tensor]] = None
    ) -> np.ndarray:
        """Allocation in canonical instrument order, (N, 7), rows summing to one."""
        aux = aux or {}
        with torch.no_grad():
            w = self._forward_weights(states.to(self.device), self._aux_to_device(aux))
        w = w.detach().cpu().numpy().astype(np.float64)
        return w[:, self._canonical_index]

    def selected_actions(
        self, states: torch.Tensor, aux: Optional[Dict[str, torch.Tensor]] = None
    ) -> Optional[np.ndarray]:
        if not self.is_discrete():
            return None
        with torch.no_grad():
            scores = self._action_scores(states.to(self.device), self._aux_to_device(aux or {}))
        return scores.argmax(dim=-1).cpu().numpy()

    def _action_scores(self, states: torch.Tensor, aux: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def _pre_squash(self, states: torch.Tensor, aux: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError(f"{self.record.model_id} has no complementary pre-squash target")

    def has_complementary_target(self) -> bool:
        return False

    def _aux_to_device(self, aux: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.to(self.device) for k, v in aux.items()}

    def release(self) -> None:
        """Move the policy off the accelerator once its diagnostics are computed.

        The record and the layer names stay available for the reports; only the
        parameters are relocated, so a released adapter must not be used again
        for a forward pass on the original device.
        """
        for attr in ("model", "policy"):
            module = getattr(self, attr, None)
            if isinstance(module, nn.Module):
                module.to("cpu")
            elif module is not None and hasattr(module, "policy"):
                module.policy.to("cpu")
        self.device = torch.device("cpu")

    # -- decision target ---------------------------------------------------

    def reference(self, states: torch.Tensor, aux: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Detached reference quantities at the unperturbed observation.

        These fix the decision target so that a perturbed observation is scored
        against the policy's *original* decision rather than its perturbed one.
        """
        aux = self._aux_to_device(aux or {})
        states = states.to(self.device)
        with torch.no_grad():
            if self.is_discrete():
                scores = self._action_scores(states, aux)
                return {"selected": scores.argmax(dim=-1).detach()}
            w = self._forward_weights(states, aux)
            return {"weights": w.detach()}

    def decision_target(
        self,
        states: torch.Tensor,
        aux: Optional[Dict[str, torch.Tensor]],
        reference: Dict[str, torch.Tensor],
        variant: str = "primary",
    ) -> torch.Tensor:
        """Differentiable scalar decision target, (N,)."""
        aux = self._aux_to_device(aux or {})
        states = states.to(self.device)
        if variant == "complementary":
            z = self._pre_squash(states, aux)
            w_hat = reference["weights"]
            return (w_hat * z).sum(dim=-1)
        if self.is_discrete():
            scores = self._action_scores(states, aux)
            sel = reference["selected"].view(-1, 1)
            return scores.gather(1, sel).squeeze(1)
        w = self._forward_weights(states, aux)
        return (reference["weights"] * w).sum(dim=-1)

    # -- rollout -----------------------------------------------------------

    @abstractmethod
    def rollout(self, frame: pd.DataFrame, dates: Sequence[pd.Timestamp]) -> Rollout:
        """Walk the evaluation dates and collect states, context and allocations."""

    def _weights_frame(self, weights: np.ndarray, dates: Sequence[pd.Timestamp]) -> pd.DataFrame:
        df = pd.DataFrame(weights, index=pd.DatetimeIndex(dates), columns=list(CANONICAL_INSTRUMENTS))
        return df


# ---------------------------------------------------------------------------
# Decision Transformer
# ---------------------------------------------------------------------------


class DTAdapter(PolicyAdapter):
    """Decision Transformer adapter.

    The policy is evaluated in the single-step mode used throughout this project:
    one ``(return-to-go, state, previous action)`` token triple per decision, with
    the previous action taken from the model's own preceding prediction and an
    incrementing timestep.
    """

    def __init__(self, record: ModelRecord, device: torch.device):
        super().__init__(record, device)
        try:
            from models.model import TrainableDT
        except ImportError:  # pragma: no cover
            from src.models.model import TrainableDT
        self.model = TrainableDT.from_pretrained(record.checkpoint_path)
        self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.act_dim = int(self.model.config.act_dim)
        self.state_dim = int(self.model.config.state_dim)
        if self.state_dim != self.window * self.n_features:
            raise ValueError(
                f"{record.model_id}: state_dim {self.state_dim} != window {self.window} x "
                f"features {self.n_features}"
            )
        self.target_return = float(record.target_return if record.target_return is not None else 0.15)

    def _flat(self, states: torch.Tensor) -> torch.Tensor:
        return states.reshape(states.shape[0], 1, -1)

    def _forward(self, states: torch.Tensor, aux: Dict[str, torch.Tensor], hidden: bool = False):
        n = states.shape[0]
        flat = self._flat(states)
        actions = aux.get("actions")
        if actions is None:
            actions = torch.zeros(n, 1, self.act_dim, device=states.device)
            actions[:, :, -1] = 1.0
        rtg = aux.get("returns_to_go")
        if rtg is None:
            rtg = torch.full((n, 1, 1), self.target_return, device=states.device)
        timesteps = aux.get("timesteps")
        if timesteps is None:
            timesteps = torch.zeros(n, 1, dtype=torch.long, device=states.device)
        out = self.model.original_forward(
            states=flat,
            actions=actions,
            rewards=None,
            returns_to_go=rtg,
            timesteps=timesteps,
            attention_mask=torch.ones(n, 1, dtype=torch.long, device=states.device),
            output_hidden_states=hidden,
            return_dict=True,
        )
        # DecisionTransformerModel.forward appends every input tensor to an
        # unbounded ``knowledge`` buffer used for NaN imputation. The diagnostics
        # run tens of thousands of forward passes, so the buffer is emptied after
        # each call. The frozen observation frame carries no NaN, which makes the
        # imputation path unreachable; if that ever changed the model would raise
        # on an empty buffer rather than impute silently.
        for buffer in getattr(self.model, "knowledge", {}).values():
            buffer.clear()
        return out

    def _forward_weights(self, states: torch.Tensor, aux: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self._forward(states, aux)
        return out[1][:, -1, :]

    def latent(self, states: torch.Tensor, aux: Dict[str, torch.Tensor]) -> np.ndarray:
        with torch.no_grad():
            out = self._forward(states.to(self.device), self._aux_to_device(aux), hidden=True)
            # Token order per timestep is (return-to-go, state, action); the action
            # head reads the state token, so index 1 is the decision-proximal state.
            hidden = out["hidden_states"][-1][:, 1, :]
        return hidden.detach().cpu().numpy()

    def latent_layer_name(self) -> str:
        return "final transformer block, state-token hidden state (input of the action head)"

    def layer_modules(self) -> Dict[str, nn.Module]:
        return {f"block_{i}": blk for i, blk in enumerate(self.model.encoder.h)}

    def rollout(self, frame: pd.DataFrame, dates: Sequence[pd.Timestamp]) -> Rollout:
        states_list, weights, actions_list, rtg_list, ts_list, used = [], [], [], [], [], []
        prev = torch.zeros(1, 1, self.act_dim, device=self.device)
        prev[:, :, -1] = 1.0
        rtg = torch.full((1, 1, 1), self.target_return, device=self.device)
        step = 0
        for dt in dates:
            end = frame.index.get_loc(dt)
            start = end - self.window + 1
            if start < 0:
                continue
            block = torch.from_numpy(frame.iloc[start : end + 1].values.astype(np.float32))
            ts = torch.tensor([[step]], dtype=torch.long, device=self.device)
            aux = {"actions": prev, "returns_to_go": rtg, "timesteps": ts}
            with torch.no_grad():
                pred = self._forward_weights(block.unsqueeze(0).to(self.device), aux)
            states_list.append(block)
            actions_list.append(prev.detach().cpu().clone())
            rtg_list.append(rtg.detach().cpu().clone())
            ts_list.append(ts.detach().cpu().clone())
            weights.append(pred.detach().cpu().numpy().reshape(-1))
            used.append(dt)
            prev = pred.detach().view(1, 1, -1)
            step += 1
        w = np.stack(weights).astype(np.float64)
        w = w[:, self._canonical_index]
        return Rollout(
            dates=pd.DatetimeIndex(used),
            states=torch.stack(states_list),
            weights=self._weights_frame(w, used),
            aux={
                "actions": torch.cat(actions_list, dim=0),
                "returns_to_go": torch.cat(rtg_list, dim=0),
                "timesteps": torch.cat(ts_list, dim=0),
            },
        )


# ---------------------------------------------------------------------------
# Stable-Baselines3
# ---------------------------------------------------------------------------


class SB3Adapter(PolicyAdapter):
    """Base adapter for Stable-Baselines3 policies with a FlattenExtractor."""

    def __init__(self, record: ModelRecord, device: torch.device):
        super().__init__(record, device)
        from ablation.multi_family import register_rl_factory_shim

        register_rl_factory_shim()
        cls = self._sb3_class(record.algorithm)
        # SB3 rebuilds the training replay buffer on load, which for these
        # checkpoints is about 17 GB of host memory per off-policy agent
        # (1e6 transitions of a 28x77 observation). Diagnostics never train or
        # sample from it, so the buffer is loaded at minimum size; policy
        # parameters are unaffected.
        overrides = {"buffer_size": 1} if record.algorithm.upper() in OFF_POLICY else {}
        self.model = cls.load(record.checkpoint_path, device=str(device), **overrides)
        self.policy = self.model.policy
        self.policy.eval()
        for p in self.policy.parameters():
            p.requires_grad_(False)
        obs_shape = tuple(int(x) for x in self.model.observation_space.shape)
        if obs_shape != (self.window, self.n_features):
            raise ValueError(
                f"{record.model_id}: checkpoint observation space {obs_shape} does not match "
                f"resolved layout (window={self.window}, features={self.n_features})"
            )
        # The action space is read from the checkpoint rather than inferred from the
        # algorithm name, and is required to agree with the declared universe.
        space = self.model.action_space
        if type(space).__name__ == "Discrete":
            record.action_space = "discrete"
            record.n_actions = int(space.n)
        else:
            record.action_space = "continuous"
            record.n_actions = int(np.prod(space.shape))
        if record.n_actions != len(self.action_instruments):
            raise ValueError(
                f"{record.model_id}: checkpoint action space has {record.n_actions} actions but "
                f"metadata declares {len(self.action_instruments)} instruments"
            )

    @staticmethod
    def _sb3_class(algorithm: str):
        from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3

        mapping = {"PPO": PPO, "A2C": A2C, "DQN": DQN, "DDPG": DDPG, "TD3": TD3, "SAC": SAC}
        key = str(algorithm).upper()
        if key not in mapping:
            raise ValueError(f"unsupported SB3 algorithm '{algorithm}'")
        return mapping[key]

    def rollout(self, frame: pd.DataFrame, dates: Sequence[pd.Timestamp]) -> Rollout:
        used = self.buildable_dates(frame, dates)
        states = self.build_states(frame, used)
        weights = self.portfolio_weights(states)
        return Rollout(dates=pd.DatetimeIndex(used), states=states, weights=self._weights_frame(weights, used))


class DDPGAdapter(SB3Adapter):
    """DDPG / TD3 deterministic continuous actor."""

    def __init__(self, record: ModelRecord, device: torch.device):
        super().__init__(record, device)
        self.actor = self.policy.actor
        self.mu = self.actor.mu
        low = np.asarray(self.model.action_space.low, dtype=np.float64)
        high = np.asarray(self.model.action_space.high, dtype=np.float64)
        self._low = torch.tensor(low, dtype=torch.float32, device=device)
        self._high = torch.tensor(high, dtype=torch.float32, device=device)

    def _features(self, states: torch.Tensor) -> torch.Tensor:
        return self.actor.extract_features(states, self.actor.features_extractor)

    def _pre_squash(self, states: torch.Tensor, aux: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.mu[:-1](self._features(states))

    def has_complementary_target(self) -> bool:
        return True

    def _forward_weights(self, states: torch.Tensor, aux: Dict[str, torch.Tensor]) -> torch.Tensor:
        squashed = self.mu(self._features(states))
        # SB3 unscale_action for a Box action space, kept differentiable.
        raw = self._low + 0.5 * (squashed + 1.0) * (self._high - self._low)
        raw = raw.clamp_min(0.0)
        total = raw.sum(dim=-1, keepdim=True)
        uniform = torch.full_like(raw, 1.0 / raw.shape[-1])
        return torch.where(total > 1e-12, raw / total.clamp_min(1e-12), uniform)

    def latent(self, states: torch.Tensor, aux: Dict[str, torch.Tensor]) -> np.ndarray:
        with torch.no_grad():
            lat = self.mu[:-2](self._features(states.to(self.device)))
        return lat.detach().cpu().numpy()

    def latent_layer_name(self) -> str:
        return "actor MLP penultimate activation (input of the action head)"

    def layer_modules(self) -> Dict[str, nn.Module]:
        return {
            f"actor_linear_{i}": m for i, m in enumerate(self.mu) if isinstance(m, nn.Linear)
        }


class DQNAdapter(SB3Adapter):
    """DQN discrete Q-network."""

    def __init__(self, record: ModelRecord, device: torch.device):
        super().__init__(record, device)
        self.q_net = self.policy.q_net
        self.trunk = self.q_net.q_net

    def _features(self, states: torch.Tensor) -> torch.Tensor:
        return self.q_net.extract_features(states, self.q_net.features_extractor)

    def _action_scores(self, states: torch.Tensor, aux: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.trunk(self._features(states))

    def _forward_weights(self, states: torch.Tensor, aux: Dict[str, torch.Tensor]) -> torch.Tensor:
        scores = self._action_scores(states, aux)
        idx = scores.argmax(dim=-1)
        return torch.nn.functional.one_hot(idx, num_classes=scores.shape[-1]).float()

    def latent(self, states: torch.Tensor, aux: Dict[str, torch.Tensor]) -> np.ndarray:
        with torch.no_grad():
            lat = self.trunk[:-1](self._features(states.to(self.device)))
        return lat.detach().cpu().numpy()

    def latent_layer_name(self) -> str:
        return "Q-network penultimate activation (input of the Q head)"

    def layer_modules(self) -> Dict[str, nn.Module]:
        return {f"q_linear_{i}": m for i, m in enumerate(self.trunk) if isinstance(m, nn.Linear)}


class PPOAdapter(SB3Adapter):
    """PPO / A2C actor-critic policy with a discrete action space."""

    def __init__(self, record: ModelRecord, device: torch.device):
        super().__init__(record, device)
        self.extractor = self.policy.mlp_extractor
        self.action_net = self.policy.action_net

    def _features(self, states: torch.Tensor) -> torch.Tensor:
        return self.policy.extract_features(states, self.policy.pi_features_extractor)

    def _latent_pi(self, states: torch.Tensor) -> torch.Tensor:
        return self.extractor.forward_actor(self._features(states))

    def _action_scores(self, states: torch.Tensor, aux: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.action_net(self._latent_pi(states))

    def _forward_weights(self, states: torch.Tensor, aux: Dict[str, torch.Tensor]) -> torch.Tensor:
        scores = self._action_scores(states, aux)
        idx = scores.argmax(dim=-1)
        return torch.nn.functional.one_hot(idx, num_classes=scores.shape[-1]).float()

    def latent(self, states: torch.Tensor, aux: Dict[str, torch.Tensor]) -> np.ndarray:
        with torch.no_grad():
            lat = self._latent_pi(states.to(self.device))
        return lat.detach().cpu().numpy()

    def latent_layer_name(self) -> str:
        return "policy-network latent representation (input of the action head)"

    def layer_modules(self) -> Dict[str, nn.Module]:
        mods: Dict[str, nn.Module] = {
            f"policy_linear_{i}": m
            for i, m in enumerate(self.extractor.policy_net)
            if isinstance(m, nn.Linear)
        }
        mods["action_head"] = self.action_net
        return mods


ADAPTERS = {
    "DecisionTransformer": DTAdapter,
    "DDPG": DDPGAdapter,
    "TD3": DDPGAdapter,
    "DQN": DQNAdapter,
    "PPO": PPOAdapter,
    "A2C": PPOAdapter,
}


def build_adapter(record: ModelRecord, device: torch.device) -> PolicyAdapter:
    cls = ADAPTERS.get(record.algorithm)
    if cls is None:
        raise ValueError(f"no adapter for algorithm '{record.algorithm}'")
    return cls(record, device)


def decision_target_table(adapters: Dict[str, PolicyAdapter]) -> pd.DataFrame:
    """Documented decision target per model (STEP 3 artefact)."""
    rows = []
    for model_id, ad in adapters.items():
        spec = DECISION_TARGETS.get(ad.record.algorithm, {})
        rows.append(
            {
                "model_id": model_id,
                "family": ad.record.family,
                "algorithm": ad.record.algorithm,
                "action_space": spec.get("action_space", ad.record.action_space),
                "primary_decision_target": spec.get("primary_target", ""),
                "action_to_weight_mapping": spec.get("weight_mapping", ""),
                "complementary_target": spec.get("complementary_target", ""),
                "complementary_available": ad.has_complementary_target(),
                "weight_based_degradation": "total-variation distance between allocations",
                "flip_based_degradation": "selected-action flip rate" if ad.is_discrete() else "not applicable",
                "latent_layer": ad.latent_layer_name(),
                "layer_modules": ", ".join(ad.layer_modules().keys()),
                "observation_window_rows": ad.window,
                "n_observation_features": ad.n_features,
                "action_index_order": ", ".join(ad.action_instruments),
            }
        )
    return pd.DataFrame(rows)
