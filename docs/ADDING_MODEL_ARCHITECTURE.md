# Adding another model architecture

This note describes the extension interface that version **1.0.0** already
implements. It does not add an adapter, a checkpoint, or a new action space.

The cross-architecture diagnostics call one object per frozen policy:
`PolicyAdapter` in `src/ablation/cross_architecture/targets.py`.
`build_adapter` selects the concrete class from the `ADAPTERS` dictionary
using `ModelRecord.algorithm`.

Two scopes stay separate:

- **Another portfolio-policy architecture** means a new network that still
  emits a long-only, fully invested allocation on the seven-instrument
  universe in `CANONICAL_INSTRUMENTS` (`^GSPC`, `^GDAXI`, `^IXIC`, `^RUT`,
  `^N225`, `^FTSE`, `CASH`). That is what the current contract accepts.
- **Another sequential decision domain** (a different action meaning, state,
  or constraint) needs a domain-specific action and output layer. Version
  1.0.0 does not provide that layer. See the last section.

`src/ablation/multi_family.py` is a separate helper used while loading
Stable-Baselines3 checkpoints (`register_rl_factory_shim`, and an older
`SB3PolicyWrapper` for Captum). It is not the registry the diagnostics
dispatch on.

## Adding another portfolio-policy architecture

### Contract

`PolicyAdapter` is an abstract base class. Its constructor stores the
`ModelRecord` and device, then builds `_canonical_index` by looking up every
name in `CANONICAL_INSTRUMENTS` inside `record.instruments`. A record whose
instrument list does not contain that universe fails at construction.

The five methods declared with `@abstractmethod` are the minimum a subclass
must implement:

| Method | Role |
| --- | --- |
| `_forward_weights(states, aux)` | Differentiable allocation in the policy's own action order, shape `(N, A)`. |
| `latent(states, aux)` | Decision-proximal internal representation, shape `(N, d)`, returned as a NumPy array. |
| `layer_modules()` | Mapping of comparable decision-processing layers (`dict[str, nn.Module]`). |
| `latent_layer_name()` | Short description of the layer `latent` reads. |
| `rollout(frame, dates)` | Walk the evaluation dates and return a `Rollout` of states, auxiliary context, and allocations. |

`states` is a time-major tensor `(N, window, n_features)` whose last row is
the decision date. `aux` carries policy context such as the previous action,
return-to-go, or timestep. Continuous adapters that have no extra context may
ignore `aux`.

### What a subclass inherits

These methods are concrete. A new portfolio adapter reuses them.

| Method | Behaviour |
| --- | --- |
| `reference(states, aux)` | Detached quantities at the unperturbed observation. Continuous policies store `weights` from `_forward_weights`. Discrete policies (`record.action_space == "discrete"`) store `selected`, the argmax of `_action_scores`. |
| `decision_target(states, aux, reference, variant="primary")` | Scalar per observation. Primary continuous target: `(reference["weights"] * _forward_weights(...)).sum(-1)`, the allocation-aligned weight mass. Primary discrete target: the score of `reference["selected"]`. `variant="complementary"` uses `(reference["weights"] * _pre_squash(...)).sum(-1)`. |
| `portfolio_weights(states, aux)` | Calls `_forward_weights` under `torch.no_grad()` and reorders columns onto `CANONICAL_INSTRUMENTS`. |
| `selected_actions(states, aux)` | For discrete policies, the argmax of `_action_scores`. Returns `None` for continuous policies. |
| `is_discrete()` | True when `record.action_space == "discrete"`. |
| `has_complementary_target()` | Returns `False` until a subclass overrides it. |
| `observation_frame`, `buildable_dates`, `build_states`, `lag_labels` | Slice the merged market frame and stack windows that end on each date. |
| `release()` | Moves a `model` or `policy` module to CPU after diagnostics. |

The contribution routine calls `decision_target` and `reference` directly. It
switches to `variant="complementary"` only when `has_complementary_target()`
is true and the primary target is locally constant or has a zero gradient on
the diagnostic sample. That switch is recorded. It is not a silent replacement.

Two further hooks are ordinary methods that raise `NotImplementedError`.
Implement them when the inherited target needs them:

- Discrete policies implement `_action_scores(states, aux)`. `reference`,
  `decision_target`, and `selected_actions` call it. The shipped discrete
  adapters (`DQNAdapter`, `PPOAdapter`) also implement `_forward_weights` as
  a one-hot allocation of the greedy action.
- A saturating continuous policy that should expose a pre-squash target
  implements `_pre_squash(states, aux)` and overrides
  `has_complementary_target` to return `True`. `DDPGAdapter` (also used for
  TD3) is the current example. Policies without a saturating head leave the
  default in place.

### Registration

`ADAPTERS` maps the algorithm string on the model record to a class.
`build_adapter(record, device)` looks up `record.algorithm` and instantiates
that class. An unknown name raises `ValueError`.

There is no setuptools entry point. Registering an architecture means adding
one entry to that dictionary, then supplying a `ModelRecord` whose
`algorithm` field is that key and whose instruments, window, and checkpoint
satisfy the existing inclusion rules in
`src/ablation/cross_architecture/discovery.py`.

### Illustrative skeleton

The class below lists only the abstract methods. Architecture-specific
loading and forward logic are left unimplemented on purpose.

```python
from ablation.cross_architecture.targets import ADAPTERS, PolicyAdapter


class NewPortfolioAdapter(PolicyAdapter):
    """Additional portfolio-policy architecture on the existing universe."""

    def _forward_weights(self, states, aux):
        """Differentiable allocation in this policy's own action order, (N, A)."""
        raise NotImplementedError

    def latent(self, states, aux):
        """Decision-proximal internal representation, (N, d)."""
        raise NotImplementedError

    def layer_modules(self):
        """Ordered comparable decision-processing layers."""
        raise NotImplementedError

    def latent_layer_name(self):
        """Human-readable description of the layer used for latent."""
        raise NotImplementedError

    def rollout(self, frame, dates):
        """Walk the evaluation dates and collect states, context and allocations."""
        raise NotImplementedError


ADAPTERS["NEW_ALGORITHM"] = NewPortfolioAdapter
```

`build_adapter` will then construct `NewPortfolioAdapter` when
`record.algorithm == "NEW_ALGORITHM"`. A discrete architecture still has to
implement `_action_scores` before `decision_target` can run. A complementary
pre-squash target still has to implement `_pre_squash` and
`has_complementary_target`.

## Using the diagnostics outside portfolio allocation

The diagnostic *roles* — attribute a scalar decision, perturb inputs and
measure how the decision moves, and inspect latent neighbourhoods and layer
influence — are the same ideas one would use for another sequential policy.
Version 1.0.0 wires those roles to a portfolio action.

`PolicyAdapter.__init__`, `portfolio_weights`, `_weights_frame`, and the
inherited `decision_target` assume a seven-column allocation (or a discrete
action mapped onto one). `CANONICAL_INSTRUMENTS` is fixed in
`discovery.py`. Inclusion rejects any other instrument universe. Observation
windows are slices of a merged market frame whose columns are
`(instrument, indicator)` pairs. Regime labels, the train/eval/holdout
calendar, portfolio returns, the cash instrument, and the 20-basis-point
transaction-cost convention live in the sampling and outcome modules.

Using the software on another sequential domain therefore means supplying a
domain-specific replacement for that action and output layer, and for the
state, constraint, and outcome pieces that assume this market. Dropping a
checkpoint in and calling `build_adapter` does not do that.

Once such an adapter exposes a scalar decision target, a latent vector, and
named layers, the following roles are the ones the current package already
factors through `PolicyAdapter`:

| Role | Where it lives | What must be adapted with the domain |
| --- | --- | --- |
| Scalar target used by attribution and layer influence | `reference`, `decision_target` | Replace the allocation-mass and selected-action formulas with the domain's decision scalar. |
| Occlusion and integrated gradients | `contribution.py` | Keep the perturbation mechanics only after the target, the observation tensor, and the feature grouping match the new state. Baselines are drawn from the training market frame. |
| Deletion and temporal occlusion *procedure* | `necessity.py` | The mask-and-replace steps follow the observation tensor. The reported degradation (total variation, Jensen–Shannon divergence, cash-weight change, discrete action-flip rate) measures portfolio allocations and has to be redefined. |
| Latent neighbourhood stability | `structure.py` (`latent`, k-nearest-neighbour overlap) | Reusable on the array `latent` returns. Input-noise and feature-mask checks still assume this observation layout. |
| Layer-influence concentration | `structure.py` (`layer_modules`, conductance or layer ablation) | Reusable once `layer_modules` and the scalar target belong to the new policy. Cross-model comparison uses normalised shares, not raw magnitudes. |
| Oriented C / N / S average | `scoring.py` | The clipping and equal-weight average do not read instrument names. `deletion_faithfulness` is defined from allocation change, so that input has to be a domain-appropriate necessity statistic before the score means the same thing. |

These pieces stay specific to this portfolio study and are not reused by
changing only the network class:

- State construction from the frozen market HDF5 (`observation_frame`,
  `build_states`, `ablation.cross_architecture.data`).
- Action semantics: long-only simplex, one-hot index-to-instrument map,
  canonical column order.
- Inclusion, metadata resolution, and the seven-instrument universe
  (`discovery.py`).
- Market regimes and the evaluation calendar (`sampling.py`).
- Portfolio outcomes: close-to-close returns, cash return of zero, holding
  the weight from date `t` over the next trading interval, transaction cost
  `0.002` (`outcomes.py`).
- The Batch A–C `audit_scoring` pipeline. It scores stored Decision
  Transformer diagnostic tables. It does not call `PolicyAdapter`.

## Existing reference implementations

All of the registered adapters are in
`src/ablation/cross_architecture/targets.py`.

| Class | `ADAPTERS` keys | What it wraps |
| --- | --- | --- |
| `DTAdapter` | `DecisionTransformer` | Frozen `TrainableDT` evaluated one `(return-to-go, state, previous action)` step at a time. The action head is already a simplex, so the primary target is the allocation-aligned weight mass and no complementary target is declared. |
| `SB3Adapter` | *(base class, not registered)* | Loads a Stable-Baselines3 checkpoint. Off-policy algorithms load with `buffer_size=1` because diagnostics do not sample the replay buffer. Subclasses read the action space from the checkpoint. |
| `DDPGAdapter` | `DDPG`, `TD3` | Deterministic continuous actor. Weights are the unscaled `tanh` action, clipped at zero and normalised to sum to one. `has_complementary_target` is true; the complementary target is the allocation-aligned pre-tanh activation. |
| `DQNAdapter` | `DQN` | Discrete Q-network. `_action_scores` is the Q-head. `_forward_weights` is the one-hot greedy action. |
| `PPOAdapter` | `PPO`, `A2C` | Discrete actor-critic. `_action_scores` is the policy logits. `_forward_weights` is the one-hot greedy action. |

`SB3Adapter._sb3_class` lists `SAC` among the Stable-Baselines3 classes it
knows how to import. `ADAPTERS` has no `SAC` entry, so `build_adapter`
raises `ValueError` for that algorithm. A SAC policy is not a supported
diagnostic adapter in v1.0.0.

`tests/test_cross_architecture.py` exercises this contract on frozen
checkpoints when those files are present (simplex rows, decision-target
identity, latent shape, layer modules). Without the checkpoints those tests
are skipped.
