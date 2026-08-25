"""
Step 1 artefacts: the implementation audit and the missing-input report.

The audit records what was verified empirically (model loading, observation
reconstruction, action-to-weight mapping, coverage) together with the defects
found in the pre-existing multi-family implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from .discovery import INCLUSION_CRITERIA, ModelRecord
from .sampling import PERIODS, REGIMES

# Defects and incorrect assumptions found in the pre-existing implementation
# while verifying it. Each entry was confirmed by direct inspection or by running
# the code path against the frozen checkpoints.
EXISTING_IMPLEMENTATION_FINDINGS: List[Dict[str, str]] = [
    {
        "location": "src/ablation/multi_family.py :: SB3PolicyWrapper._normalize_continuous",
        "issue": "For a continuous actor the wrapper applies relu() to the tanh output and "
        "renormalises. The correct transformation is SB3's unscale_action, which maps the "
        "tanh output from [-1, 1] onto the action-space range, i.e. (a+1)/2 for Box(0, 1).",
        "impact": "relu() maps every negative pre-activation to exactly zero, which destroys "
        "both the weight values and the gradient for any actor whose output is negative. This "
        "is the direct cause of the all-zero DDPG attributions recorded in "
        "results/RL_comparison/run_stdout.txt.",
        "resolution": "targets.DDPGAdapter implements the differentiable unscale_action mapping.",
    },
    {
        "location": "src/ablation/multi_family.py :: enrich_spec_from_metadata",
        "issue": "The action space is inferred from the algorithm name with the rule "
        "'DQN or PPO implies discrete'. This is an assumption about the training "
        "configuration, not a fact about the checkpoint.",
        "impact": "The rule happens to be right for these three runs but would silently "
        "mislabel a continuous PPO policy, and it never checks the checkpoint.",
        "resolution": "The action space is now read from the deserialised checkpoint's "
        "action_space and cross-checked against the observation space.",
    },
    {
        "location": "src/ablation/multi_family.py :: extract_latents (SB3 branch)",
        "issue": "For policies without an mlp_extractor (DQN, DDPG) the function returns "
        "wrapper._features(obs), which for a FlattenExtractor is the flattened raw "
        "observation, not an internal representation.",
        "impact": "The recorded DDPG and DQN latents had dimension 2156, exactly the flattened "
        "observation size, so any latent-structure result for those models described the input "
        "data rather than the policy.",
        "resolution": "targets.DQNAdapter and targets.DDPGAdapter extract the penultimate "
        "activation of the Q network and of the actor MLP respectively.",
    },
    {
        "location": "src/ablation/__init__.py :: latent_space_market_regime",
        "issue": "The DT latent is taken as hidden_states[-1][:, -1, :]. With the token order "
        "(return-to-go, state, action) the action head reads the state token at index 1, so "
        "index -1 is the action token.",
        "impact": "The extracted representation is not the one that produces the allocation.",
        "resolution": "targets.DTAdapter uses the state-token hidden state at index 1.",
    },
    {
        "location": "notebooks/2_ablation_walkthrough_multifamily.ipynb, portfolio-returns cell",
        "issue": "Weights are reindexed onto pd.date_range(..., freq='D') and forward-filled "
        "before returns are computed.",
        "impact": "Weekends and holidays enter the return series as artificial observations "
        "(1918 trading rows become 2172), which biases every annualised statistic and the "
        "drawdown path.",
        "resolution": "outcomes.portfolio_returns operates only on the trading calendar of the "
        "frozen market data; no calendar reindexing or forward filling is performed.",
    },
    {
        "location": "notebooks/2_ablation_walkthrough_multifamily.ipynb, attribution cell",
        "issue": "Attribution is computed for every action head and then averaged with "
        "abs().mean(dim=0) for all algorithms, including DQN.",
        "impact": "For a discrete policy this mixes the attributions of six Q heads that the "
        "policy never selects into the explanation of the decision it does take.",
        "resolution": "Discrete policies are attributed on the selected action's Q value or "
        "logit; continuous simplex policies use an allocation-weighted target.",
    },
    {
        "location": "notebooks/2_ablation_walkthrough_multifamily.ipynb, regime cell",
        "issue": "The regime configuration starts the crash period on 2020-03-01 while the bull "
        "market ends on 2020-02-15, leaving 17--28 February 2020 unlabelled. Ten evaluation "
        "dates were labelled 'Unknown' and retained in the regime tables.",
        "impact": "An Unknown category appears in a formally four-regime analysis.",
        "resolution": "sampling.REGIMES places 15--28 February 2020 inside the crash period, and "
        "sampling.assign_regimes raises if any date is unlabelled.",
    },
    {
        "location": "src/ablation/preprocessing.py :: prepare_ablation_inputs",
        "issue": "The function opens with a loop over the whole backtest frame whose body only "
        "rebuilds 'obs' and is then discarded, and it passes attention_mask=zeros.",
        "impact": "The dead loop doubles the cost of building the DT batch. The all-zero mask "
        "marks the only supplied token as padding, so attention is spread uniformly over the "
        "(return-to-go, state, action) triple instead of respecting causality, and the "
        "allocation changes: over the evaluation period the mean L1 difference against the "
        "correct mask is 0.008 for DT_16_10 (max 0.063, no change of the largest holding) and "
        "0.368 for DT_max_quadratic_utility (max 1.541, largest holding changes on 21.7% of "
        "dates). The effect is therefore material for at least one checkpoint.",
        "resolution": "targets.DTAdapter builds the rollout in a single pass with an all-ones "
        "attention mask. tests/test_cross_architecture.py asserts that the adapter uses the "
        "all-ones mask and that the legacy all-zero mask changes the allocation, so the "
        "discrepancy cannot be reintroduced unnoticed. No DT weight file is stored under "
        "models/comparison/dt, so the size of the effect on previously published DT numbers "
        "cannot be checked here and all DT outcomes in this comparison are recomputed.",
    },
    {
        "location": "notebooks/2_ablation_walkthrough_multifamily.ipynb, model-loading cell",
        "issue": "The Stable-Baselines3 checkpoints are loaded with the default arguments, so "
        "each off-policy agent rebuilds its training replay buffer.",
        "impact": "Each DDPG or DQN checkpoint asks for about 17 GB of host memory (one million "
        "transitions of a 28x77 observation). Loading two of them exhausted the machine and the "
        "run aborted in an unrelated place with a numpy allocation error.",
        "resolution": "targets.SB3Adapter loads off-policy agents with buffer_size=1. The "
        "diagnostics never train and never sample from the buffer, the policy parameters are "
        "read from the checkpoint unchanged, and host memory for all three RL policies together "
        "stays near 0.7 GB.",
    },
    {
        "location": "src/models/model.py :: DecisionTransformerModel.forward",
        "issue": "Every call appends the states, actions and returns-to-go tensors to an "
        "unbounded self.knowledge dictionary, which is only read to impute NaN inputs.",
        "impact": "Memory grows with the number of forward passes: the attribution stage of a "
        "single DT retained about 3.9 GiB of input tensors on the device and exhausted an 8 GiB "
        "card on the second policy. It also means a diagnostic pass leaves state behind on the "
        "model, so the imputation reference would depend on how many perturbed observations had "
        "been scored earlier.",
        "resolution": "targets.DTAdapter empties the buffer after each forward pass. The frozen "
        "observation frame contains no NaN, so the imputation path is never taken; the shared "
        "model class is left unmodified so that existing training and backtest code is "
        "unaffected.",
    },
    {
        "location": "src/ablation/postprocessing.py :: aggregate_attributions",
        "issue": "The 'indicator_wise' aggregate-all branch references an undefined name "
        "attr_reshaped.",
        "impact": "That code path raises NameError; it is unused by the new pipeline.",
        "resolution": "Not used. The new pipeline aggregates through "
        "contribution.aggregate_cube.",
    },
    {
        "location": "scripts/build_rl_manifest.py :: env_state derivation",
        "issue": "The observation column order for the SB3 policies is derived by a script "
        "rather than recorded by the training code, and metadata.json lists an indicator "
        "(STOCHF) that the frozen data file does not contain.",
        "impact": "The feature ordering is an inference. If it were wrong, every per-feature "
        "attribution label would be wrong.",
        "resolution": "Candidate orderings were compared against the stored allocations of the "
        "one non-degenerate RL policy before any diagnostic was computed. The "
        "indicator-block-then-price-block order (the order produced by the training code's "
        "indicators.merge(prices) call) agreed with the stored greedy actions far more often "
        "than any alternative ordering tested, every one of which stayed at chance level. That "
        "order is adopted for all RL policies, the 77-column layout is cross-checked against "
        "each checkpoint's observation space, the indicator named in metadata but absent from "
        "the frozen file is dropped, and the residual uncertainty is recorded in "
        "missing_inputs.md.",
    },
]


def render_audit(
    records: Sequence[ModelRecord],
    load_report: pd.DataFrame,
    coverage: pd.DataFrame,
    data_files: pd.DataFrame,
    reconstruction: pd.DataFrame,
    decision_targets: pd.DataFrame,
    common_dates_n: int,
    settings: Dict[str, object],
) -> str:
    def table(frame: pd.DataFrame) -> str:
        from .reporting import markdown_table

        return markdown_table(frame) + "\n"

    lines: List[str] = []
    lines.append("# Implementation audit: cross-architecture diagnostic comparison\n")
    lines.append(
        "This document records what was verified before the final experiment was run. "
        "Model loading, observation construction and the action-to-portfolio-weight mapping "
        "were checked against the frozen checkpoints; the defects found in the pre-existing "
        "implementation are listed in the final section.\n"
    )

    lines.append("## 1. Models discovered\n")
    cols = [
        c
        for c in [
            "model_id", "family", "algorithm", "checkpoint_path", "metadata_path",
            "data_path", "feature_extractor", "lookback", "window_length",
            "n_state_features", "action_space", "n_actions", "state_dim",
            "reference_date", "rebalancing_frequency", "inclusion_status",
            "exclusion_reason",
        ]
        if c in load_report.columns
    ]
    lines.append(table(load_report.loc[:, cols]))

    lines.append("## 2. Frozen market data\n")
    lines.append(
        "All included policies are evaluated on a single frozen market data file so that the "
        "asset universe, the indicator definitions and the trading calendar are identical "
        "across architectures.\n"
    )
    lines.append(table(data_files))

    lines.append("## 3. Observation construction and required metadata\n")
    lines.append(
        "Both families consume a time-major observation window whose first row is the oldest "
        "observation; this is the layout produced by flattening the window in C order, which is "
        "what the Decision Transformer state vector and the Stable-Baselines3 `FlattenExtractor` "
        "both expect. A Decision Transformer consumes `lookback + 1` rows, an SB3 policy "
        "consumes `lookback` rows.\n"
    )
    lines.append(
        "* Decision Transformer observation columns come from `metadata.json:env_state` "
        "(66 columns: seven technical indicators for six indices, plus Open/High/Low/Close for "
        "six indices). This is written by the training code and is authoritative.\n"
        "* Stable-Baselines3 observation columns are the technical-indicator block followed by "
        "the price block of the frozen data file (77 columns: 42 indicator columns, plus "
        "Open/High/Low/Close/Volume for six indices and for CASH). The SB3 metadata does not "
        "record `env_state`, so this ordering is reconstructed and was verified empirically "
        "(see section 6).\n"
        "* The two families therefore do **not** share an identical input set: the RL policies "
        "additionally observe traded volume and the CASH price series. Cross-family attribution "
        "comparisons are restricted to shares, ranks and concentration indices computed within "
        "each model, and the indicator families common to both are reported separately.\n"
    )

    lines.append("### State-feature and lag ordering\n")
    lines.append(
        "The lag axis used throughout the diagnostics is defined with index 0 as the most recent "
        "observation row, independently of the storage order. Feature labels are "
        "`(instrument, indicator)` pairs taken from the resolved observation columns.\n"
    )

    lines.append("## 4. Action space and output-to-portfolio mapping\n")
    keep = [
        c
        for c in [
            "model_id", "algorithm", "action_space", "primary_decision_target",
            "action_to_weight_mapping", "complementary_target", "latent_layer",
            "layer_modules", "action_index_order",
        ]
        if c in decision_targets.columns
    ]
    lines.append(table(decision_targets.loc[:, keep]))
    lines.append(
        "The action index order is taken from each run's own `metadata.json:instruments` "
        "(with CASH appended when `cash_instrument` is set). Allocations are then re-indexed "
        "onto a single canonical instrument order before anything is compared.\n"
    )

    lines.append("## 5. Evaluation coverage\n")
    lines.append(
        f"Period definitions: {PERIODS}. The common diagnostic set contains "
        f"{common_dates_n} trading dates available to every included policy.\n"
    )
    lines.append(table(coverage))

    lines.append("## 6. Verified reconstruction of allocations\n")
    lines.append(
        "Allocations were reconstructed from the frozen policies and, where a legacy "
        "`results/weights_BEST.csv` exists next to the checkpoint, compared with it. "
        "`legacy_action_agreement` is the fraction of shared dates on which the largest weight "
        "falls on the same instrument; `legacy_mean_l1` is the mean absolute allocation "
        "difference; `legacy_mean_l1_order_invariant` is that distance after sorting each weight "
        "vector, which separates whether the policy output itself is reproduced from whether the "
        "instrument labelling agrees. `distinct_allocations` counts the distinct weight vectors the policy emits "
        "over the whole evaluation window, and `max_sum_deviation` verifies the fully invested "
        "constraint. The two gradient columns separate the decision target the policy actually "
        "acts on (`grad_nonzero_primary_target`) from the one the gradient-based estimator was "
        "run on (`grad_nonzero_target_used`), which differ only where the primary target is "
        "saturated.\n"
    )
    lines.append(table(reconstruction))
    lines.append(
        "Interpretation. Three separate questions have to be kept apart here.\n\n"
        "*Are the simplex constraints satisfied?* Yes, for every policy: weights are "
        "non-negative and every row sums to one to within numerical tolerance, so the "
        "action-to-weight mapping is internally valid.\n\n"
        "*Is the policy output reproduced?* The order-invariant distance answers this without "
        "depending on the instrument labelling. It is essentially zero for the continuous policy, "
        "which means the reconstructed observation windows drive the frozen actor to the same "
        "allocation values that the legacy backtest recorded. For the two discrete policies the "
        "order-invariant distance is zero by construction -- every allocation is one-hot -- so it "
        "carries no information for them, and the labelled agreement column is the relevant one.\n\n"
        "*Does the instrument labelling agree?* Not consistently. Note the two order columns in "
        "the table: the legacy CSV files are labelled in the instrument order of the frozen data "
        "file, whereas `metadata.json` declares a different instrument order for the action space. "
        "Under the declared order the value-based policy's greedy action matches the legacy file "
        "on every shared date, the continuous policy's allocation differs on one instrument, and "
        "the stochastic policy's action matches on far fewer dates than the declared order would "
        "predict -- with a diffuse rather than permutation-like disagreement pattern, which is "
        "what sampling from the categorical policy in the legacy backtest would produce, since "
        "this pipeline always takes the greedy action.\n\n"
        "Consequently the legacy files are treated as **not authoritative** for the action-index "
        "mapping. The ordering declared in each run's own `metadata.json:instruments` is used "
        "instead, identically for every policy, and it was fixed before any diagnostic was "
        "computed. Every diagnostic and every reported portfolio outcome is computed from the "
        "frozen policy rather than read from a legacy file, and the artefact that would settle "
        "the residual ambiguity is specified in `missing_inputs.md`.\n"
    )

    lines.append("## 7. Which diagnostics can be computed\n")
    lines.append(
        "| Diagnostic | Decision Transformer | DDPG | DQN | PPO |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| Feature attribution (perturbation) | yes | yes | yes | yes |\n"
        "| Feature attribution (integrated gradients) | yes | only on the complementary "
        "pre-squash target | yes | vanishes: saturated tanh MLP |\n"
        "| Attribution-guided deletion | yes | allocation is input-invariant | selected action "
        "is input-invariant | yes |\n"
        "| Temporal occlusion | yes | yes (on the decision target) | yes | yes |\n"
        "| Latent extraction | yes | yes | yes | yes |\n"
        "| Latent neighbourhood stability | yes | yes | yes | yes |\n"
        "| Layer influence | yes | yes | yes | yes (controlled ablation) |\n"
        "| Return-to-go conditioning | yes | not applicable | not applicable | not applicable |\n"
        "| Attention-head analysis | yes | not applicable | not applicable | not applicable |\n"
    )

    lines.append("## 8. Incompatibilities and degenerate policies found\n")
    lines.append(
        "These are properties of the frozen policies, not of the diagnostic framework, and they "
        "are reported as findings rather than worked around:\n\n"
        "1. **DDPG actor saturation.** The pre-tanh activations measured at audit time reach "
        "magnitudes of order 1e8, so every action component saturates at the bounds of the "
        "action space. Section 6 shows the consequence: the allocation takes only a handful of "
        "distinct values over the whole evaluation window, and the gradient of the allocation "
        "target with respect to the observation is exactly zero "
        "(`grad_nonzero_primary_target` = 0). The documented complementary pre-squash target "
        "restores a usable gradient (`grad_nonzero_target_used`), so the gradient-based "
        "estimators run on it, and necessity measured on the allocation itself is genuinely low "
        "rather than unmeasured.\n"
        "2. **DQN action collapse.** The greedy action is the same instrument on every "
        "evaluation date, so the allocation is constant (`distinct_allocations` = 1). The Q "
        "values themselves do vary strongly with the observation (`decision_target_std` in "
        "section 6), so the selected-action Q value remains a usable decision target and the "
        "contribution and structure dimensions stay measurable.\n"
        "3. **PPO gradient saturation.** The policy MLP applies tanh activations to unnormalised "
        "inputs -- traded volume is of order 1e9 -- so local gradients vanish even though the "
        "policy is far from constant (`grad_nonzero_target_used` = 0 while the policy emits "
        "seven distinct allocations). Perturbation-based attribution is informative where "
        "integrated gradients are not, and which estimator supplied the primary ranking is "
        "recorded per model.\n"
        "4. **Input-set mismatch.** The two families do not observe the same columns "
        "(66 versus 77), which is why only normalised, rank-based comparisons are made.\n"
        "5. **Cost model.** Net performance charges the project's documented 20 basis points on "
        "the one-way traded fraction. The 0.5% slippage that `src/README.md` also documents for "
        "the training environment is not applied here, because no implementation of it exists in "
        "this repository; net figures are therefore an upper bound on realised performance.\n"
    )

    lines.append("## 9. Defects found in the pre-existing implementation\n")
    lines.append(table(pd.DataFrame(EXISTING_IMPLEMENTATION_FINDINGS)))

    lines.append("## 10. Pre-specified inclusion criteria\n")
    for c in INCLUSION_CRITERIA:
        lines.append(f"* {c}")
    lines.append("")
    lines.append(f"Regime configuration used: `{REGIMES}`\n")
    lines.append("## 11. Run settings\n")
    lines.append(table(pd.DataFrame([settings]).T.reset_index().rename(columns={"index": "setting", 0: "value"})))
    return "\n".join(lines)


def render_missing_inputs(items: Sequence[Dict[str, str]]) -> str:
    lines = ["# Missing inputs\n"]
    if not items:
        lines.append(
            "No input is missing. Every included policy was loaded from its frozen checkpoint, "
            "every observation window was rebuilt from the frozen market data file, and every "
            "portfolio outcome was reconstructed from the policy itself, so no externally "
            "supplied weight file or performance summary is required.\n"
        )
        return "\n".join(lines)
    lines.append(
        "The analysis below completed without these inputs; they would remove documented "
        "residual uncertainty rather than unblock the diagnostics. Each entry states exactly "
        "what is missing and why it matters.\n"
    )
    for i, item in enumerate(items, start=1):
        lines.append(f"## {i}. {item['title']}\n")
        for key in (
            "affected_models",
            "exact_missing_file",
            "required_schema",
            "expected_columns",
            "expected_date_range",
            "expected_frequency",
            "expected_weight_convention",
            "why_necessary",
            "current_workaround",
        ):
            if item.get(key):
                label = key.replace("_", " ").capitalize()
                lines.append(f"* **{label}:** {item[key]}")
        lines.append("")
    return "\n".join(lines)
