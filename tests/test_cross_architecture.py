"""
Validation suite for the cross-architecture diagnostic comparison.

The suite is written against the stdlib ``unittest`` runner so that it needs no
extra dependency:

    python -m unittest discover -s tests -v

It is also collected unchanged by pytest if that is available.

Model-dependent tests load the frozen checkpoints once for the whole module. If
the model directories are absent the corresponding tests are skipped rather than
failed, so the suite remains usable in a checkout without the large artefacts.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ablation.cross_architecture import (  # noqa: E402
    contribution,
    data as market,
    discovery,
    necessity,
    outcomes,
    reporting,
    sampling,
    scoring,
    structure,
    targets,
)

DT_DIR = ROOT / "models" / "comparison" / "dt"
RL_DIR = ROOT / "models" / "comparison" / "rl"
DATA_PATH = DT_DIR / "DT_16_10" / "data.h5"
RESULTS_DIR = ROOT / "results" / "cross_architecture"
N_TEST_DATES = 12

_STATE: dict = {}


def _fixture() -> dict:
    """Load the market data, records and adapters once for the whole module."""
    if _STATE:
        return _STATE
    md = market.load_market_data(DATA_PATH)
    records = discovery.discover_models(DT_DIR, RL_DIR)
    for record in records:
        discovery.resolve_record(
            record,
            indicator_columns=md.indicator_columns,
            price_columns=md.price_columns,
            default_data_path=str(md.path),
        )
    device = torch.device("cpu")
    adapters, rollouts = {}, {}
    period = sampling.period_dates(md.merged.index, "eval")
    for record in records:
        adapter = targets.build_adapter(record, device)
        adapters[record.model_id] = adapter
        frame = adapter.observation_frame(md.merged)
        dates = adapter.buildable_dates(frame, period)
        # A short, regime-spread slice keeps the suite fast while still exercising
        # every code path on real observations.
        step = max(1, len(dates) // N_TEST_DATES)
        picked = list(dates)[::step][:N_TEST_DATES]
        rollouts[record.model_id] = adapter.rollout(frame, picked)
        _STATE.setdefault("frames", {})[record.model_id] = frame
    _STATE.update(
        {"md": md, "records": records, "adapters": adapters, "rollouts": rollouts, "period": period}
    )
    return _STATE


def _models_available() -> bool:
    return DATA_PATH.exists() and any(DT_DIR.rglob("model/config.json")) and any(
        RL_DIR.rglob("best_model.zip")
    )


needs_models = unittest.skipUnless(
    _models_available(), "frozen checkpoints or market data not present in this checkout"
)


# ---------------------------------------------------------------------------
# Discovery and manifest
# ---------------------------------------------------------------------------


@needs_models
class TestDiscovery(unittest.TestCase):
    def test_finds_both_families(self):
        records = discovery.discover_models(DT_DIR, RL_DIR)
        families = {r.family for r in records}
        self.assertIn("DT", families)
        self.assertIn("RL", families)

    def test_model_ids_unique(self):
        records = discovery.discover_models(DT_DIR, RL_DIR)
        ids = [r.model_id for r in records]
        self.assertEqual(len(ids), len(set(ids)))

    def test_every_record_has_checkpoint_and_metadata(self):
        for record in discovery.discover_models(DT_DIR, RL_DIR):
            self.assertTrue(Path(record.checkpoint_path).exists(), record.model_id)
            self.assertIsNotNone(record.metadata_path, record.model_id)

    def test_resolution_fills_required_fields(self):
        state = _fixture()
        for record in state["records"]:
            self.assertTrue(record.state_features, record.model_id)
            self.assertIsNotNone(record.window_length, record.model_id)
            self.assertEqual(
                sorted(record.instruments), sorted(discovery.CANONICAL_INSTRUMENTS), record.model_id
            )

    def test_inclusion_rules_reject_missing_columns(self):
        record = discovery.ModelRecord(
            model_id="synthetic",
            family="RL",
            algorithm="PPO",
            checkpoint_path=str(DATA_PATH),
            metadata_path=str(DATA_PATH),
            state_features=[("^NOPE", "RSI")],
            window_length=4,
            instruments=list(discovery.CANONICAL_INSTRUMENTS),
        )
        discovery.apply_inclusion_rules(
            record,
            loadable=True,
            load_error=None,
            available_columns=[("^GSPC", "RSI")],
            coverage_fraction=1.0,
        )
        self.assertEqual(record.inclusion_status, "excluded")
        self.assertIn("absent from the frozen data file", record.exclusion_reason)

    def test_inclusion_rules_reject_low_coverage(self):
        record = discovery.ModelRecord(
            model_id="synthetic",
            family="RL",
            algorithm="PPO",
            checkpoint_path=str(DATA_PATH),
            metadata_path=str(DATA_PATH),
            state_features=[("^GSPC", "RSI")],
            window_length=4,
            instruments=list(discovery.CANONICAL_INSTRUMENTS),
        )
        discovery.apply_inclusion_rules(
            record,
            loadable=True,
            load_error=None,
            available_columns=[("^GSPC", "RSI")],
            coverage_fraction=0.10,
        )
        self.assertEqual(record.inclusion_status, "excluded")
        self.assertIn("complete observation window", record.exclusion_reason)

    def test_manifest_round_trip(self):
        state = _fixture()
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.yaml"
            discovery.write_manifest(path, state["records"], {"seed": 42})
            back, meta = discovery.read_manifest(path)
            self.assertEqual(meta["seed"], 42)
            self.assertEqual([r.model_id for r in back], [r.model_id for r in state["records"]])
            self.assertEqual(back[0].state_features, state["records"][0].state_features)


# ---------------------------------------------------------------------------
# Feature and lag ordering
# ---------------------------------------------------------------------------


@needs_models
class TestOrdering(unittest.TestCase):
    def test_observation_columns_exist_and_match_checkpoint(self):
        state = _fixture()
        for model_id, adapter in state["adapters"].items():
            self.assertEqual(
                adapter.n_features * adapter.window,
                adapter.window * len(adapter.feature_labels),
                model_id,
            )
            frame = state["frames"][model_id]
            self.assertEqual(list(frame.columns), adapter.feature_labels, model_id)

    def test_rl_layout_is_indicator_block_then_price_block(self):
        state = _fixture()
        md = state["md"]
        expected = md.indicator_columns + md.price_columns
        for adapter in state["adapters"].values():
            if adapter.record.family == "RL":
                self.assertEqual(adapter.feature_labels, expected)

    def test_dt_window_is_lookback_plus_one(self):
        state = _fixture()
        for adapter in state["adapters"].values():
            if adapter.record.family == "DT":
                self.assertEqual(adapter.window, int(adapter.record.lookback) + 1)
            else:
                self.assertEqual(adapter.window, int(adapter.record.lookback))

    def test_state_window_is_time_major_and_ends_on_the_decision_date(self):
        state = _fixture()
        for model_id, adapter in state["adapters"].items():
            frame = state["frames"][model_id]
            date = state["rollouts"][model_id].dates[-1]
            block = adapter.build_states(frame, [date])[0].numpy()
            self.assertEqual(block.shape, (adapter.window, adapter.n_features))
            np.testing.assert_allclose(
                block[-1], frame.loc[date].values.astype(np.float32), rtol=1e-6
            )
            position = frame.index.get_loc(date)
            np.testing.assert_allclose(
                block[0],
                frame.iloc[position - adapter.window + 1].values.astype(np.float32),
                rtol=1e-6,
            )

    def test_lag_index_zero_is_the_most_recent_row(self):
        state = _fixture()
        for adapter in state["adapters"].values():
            lags = adapter.lag_labels()
            self.assertEqual(lags[-1], 0)
            self.assertEqual(lags[0], adapter.window - 1)


# ---------------------------------------------------------------------------
# Regimes and sampling
# ---------------------------------------------------------------------------


class TestRegimes(unittest.TestCase):
    def test_regimes_partition_the_evaluation_window(self):
        idx = pd.bdate_range(*sampling.PERIODS["eval"])
        labels = sampling.assign_regimes(idx)
        self.assertEqual(len(labels), len(idx))
        self.assertNotIn("Unknown", set(labels))
        self.assertEqual(set(labels), set(sampling.REGIME_ORDER))

    def test_february_2020_gap_is_inside_the_crash_period(self):
        for day in ("2020-02-15", "2020-02-20", "2020-02-28"):
            self.assertEqual(sampling.assign_regimes([pd.Timestamp(day)])[0], "crash_period")

    def test_unlabelled_date_raises(self):
        with self.assertRaises(ValueError):
            sampling.assign_regimes([pd.Timestamp("2015-06-01")])

    def test_overlapping_configuration_raises(self):
        bad = {"a": ("2020-01-01", "2020-06-30"), "b": ("2020-03-01", "2020-12-31")}
        with self.assertRaises(ValueError):
            sampling.assign_regimes([pd.Timestamp("2020-04-01")], bad)

    def test_common_dates_is_the_intersection(self):
        a = pd.bdate_range("2020-01-01", "2020-02-01")
        b = pd.bdate_range("2020-01-15", "2020-03-01")
        common = sampling.common_dates({"a": a, "b": b})
        self.assertEqual(set(common), set(a).intersection(set(b)))
        self.assertTrue(common.is_monotonic_increasing)

    def test_stratified_sample_is_balanced_and_deterministic(self):
        idx = pd.bdate_range(*sampling.PERIODS["eval"])
        first, composition = sampling.stratified_sample(idx, 256, seed=42)
        second, _ = sampling.stratified_sample(idx, 256, seed=42)
        self.assertEqual(len(first), 256)
        self.assertTrue(first.equals(second))
        counts = pd.Series(sampling.assign_regimes(first)).value_counts()
        self.assertEqual(len(counts), 4)
        self.assertLessEqual(counts.max() - counts.min(), 1)
        self.assertEqual(int(composition["sampled"].sum()), 256)
        self.assertAlmostEqual(float(composition["share_of_sample"].sum()), 1.0)

    def test_validate_sample_accepts_a_clean_sample(self):
        idx = pd.bdate_range("2020-06-01", "2020-06-30")
        checks = sampling.validate_sample(idx)
        self.assertEqual(checks["unknown"], 0)
        self.assertEqual(checks["n"], len(idx))

    def test_validate_sample_rejects_duplicates(self):
        idx = pd.bdate_range("2020-06-01", "2020-06-30")
        with self.assertRaises(AssertionError):
            sampling.validate_sample(idx.append(idx[:1]))

    def test_validate_sample_rejects_holdout_leakage(self):
        idx = pd.bdate_range("2020-06-01", "2020-06-30")
        with self.assertRaises(AssertionError):
            sampling.validate_sample(idx, holdout_start="2020-06-15")

    def test_holdout_dates_carry_no_four_regime_label(self):
        with self.assertRaises(ValueError):
            sampling.assign_regimes([pd.Timestamp(sampling.PERIODS["holdout"][0])])

    def test_sample_size_table_covers_every_regime(self):
        idx = pd.bdate_range(*sampling.PERIODS["eval"])
        sample, _ = sampling.stratified_sample(idx, 64, seed=1)
        table = sampling.sample_size_table({"m": idx}, sample)
        self.assertEqual(set(table["regime"]), set(sampling.REGIME_ORDER) | {"all"})
        total = table.loc[table["regime"] == "all", "n_diagnostic_sample"].iloc[0]
        self.assertEqual(total, 64)

    def test_train_reference_statistics_use_only_training_data(self):
        frame = pd.DataFrame(
            {"x": [1.0, 1.0, 100.0]},
            index=pd.to_datetime(["2010-01-01", "2015-01-01", "2020-01-01"]),
        )
        mean, std, train = sampling.train_reference_statistics(frame, train_end="2015-12-31")
        self.assertEqual(len(train), 2)
        self.assertAlmostEqual(float(mean[0]), 1.0)
        self.assertGreater(float(std[0]), 0.0)


# ---------------------------------------------------------------------------
# Policy adapters, targets and latents
# ---------------------------------------------------------------------------


@needs_models
class TestAdapters(unittest.TestCase):
    def test_weights_lie_on_the_simplex(self):
        state = _fixture()
        for model_id, rollout in state["rollouts"].items():
            w = rollout.weights
            self.assertEqual(list(w.columns), list(discovery.CANONICAL_INSTRUMENTS), model_id)
            self.assertGreaterEqual(float(w.values.min()), -1e-9, model_id)
            np.testing.assert_allclose(w.values.sum(axis=1), 1.0, atol=1e-5)

    def test_weights_are_indexed_by_trading_dates_only(self):
        state = _fixture()
        md = state["md"]
        for model_id, rollout in state["rollouts"].items():
            self.assertTrue(set(rollout.weights.index) <= set(md.merged.index), model_id)
            self.assertEqual(rollout.weights.index.dayofweek.max() <= 4, True, model_id)

    def test_discrete_policies_select_a_valid_action(self):
        state = _fixture()
        found_discrete = False
        for model_id, adapter in state["adapters"].items():
            rollout = state["rollouts"][model_id]
            selected = adapter.selected_actions(rollout.states, rollout.aux)
            if adapter.is_discrete():
                found_discrete = True
                self.assertEqual(selected.shape, (len(rollout),))
                self.assertTrue(((selected >= 0) & (selected < adapter.record.n_actions)).all())
                # the allocation of a discrete policy is the one-hot of its action
                one_hot = rollout.weights.values.argmax(axis=1)
                canonical = [
                    list(discovery.CANONICAL_INSTRUMENTS).index(adapter.action_instruments[a])
                    for a in selected
                ]
                np.testing.assert_array_equal(one_hot, canonical)
            else:
                self.assertIsNone(selected)
        self.assertTrue(found_discrete, "no discrete policy was exercised")

    def test_decision_target_is_finite_and_per_observation(self):
        state = _fixture()
        for model_id, adapter in state["adapters"].items():
            rollout = state["rollouts"][model_id]
            reference = adapter.reference(rollout.states, rollout.aux)
            tau = adapter.decision_target(rollout.states, rollout.aux, reference)
            self.assertEqual(tuple(tau.shape), (len(rollout),), model_id)
            self.assertTrue(torch.isfinite(tau).all(), model_id)

    def test_continuous_target_equals_the_allocation_weighted_mass(self):
        state = _fixture()
        for model_id, adapter in state["adapters"].items():
            if adapter.is_discrete():
                continue
            rollout = state["rollouts"][model_id]
            reference = adapter.reference(rollout.states, rollout.aux)
            tau = adapter.decision_target(rollout.states, rollout.aux, reference).detach().numpy()
            w = adapter.portfolio_weights(rollout.states, rollout.aux)
            np.testing.assert_allclose(tau, (w**2).sum(axis=1), rtol=1e-4, atol=1e-5)

    def test_complementary_target_is_only_used_where_declared(self):
        state = _fixture()
        for adapter in state["adapters"].values():
            if not adapter.has_complementary_target():
                with self.assertRaises(NotImplementedError):
                    adapter._pre_squash(
                        state["rollouts"][adapter.record.model_id].states, {}
                    )

    def test_latent_is_an_internal_representation_not_the_input(self):
        state = _fixture()
        for model_id, adapter in state["adapters"].items():
            rollout = state["rollouts"][model_id]
            latent = adapter.latent(rollout.states, rollout.aux)
            self.assertEqual(latent.shape[0], len(rollout), model_id)
            self.assertLess(
                latent.shape[1], adapter.window * adapter.n_features, f"{model_id} latent"
            )
            self.assertTrue(np.isfinite(latent).all(), model_id)

    def test_off_policy_checkpoints_do_not_materialise_their_replay_buffer(self):
        """A full buffer is ~17 GB per agent and is never used by the diagnostics."""
        state = _fixture()
        checked = 0
        for model_id, adapter in state["adapters"].items():
            if adapter.record.algorithm.upper() not in targets.OFF_POLICY:
                continue
            buffer = getattr(adapter.model, "replay_buffer", None)
            self.assertIsNotNone(buffer, model_id)
            self.assertEqual(int(buffer.buffer_size), 1, model_id)
            checked += 1
        self.assertGreater(checked, 0, "no off-policy checkpoint was available to check")

    def test_layer_modules_are_comparable_and_non_empty(self):
        state = _fixture()
        for model_id, adapter in state["adapters"].items():
            modules = adapter.layer_modules()
            self.assertGreaterEqual(len(modules), 2, model_id)
            self.assertTrue(all(isinstance(k, str) for k in modules))

    def _dt_forward_with_mask(self, adapter, rollout, mask_value: float, n: int = 8):
        states = rollout.states[:n]
        n = states.shape[0]
        aux = {k: v[:n] for k, v in rollout.aux.items()}
        with torch.no_grad():
            out = adapter.model.original_forward(
                states=states.reshape(n, 1, -1),
                actions=aux["actions"],
                rewards=None,
                returns_to_go=aux["returns_to_go"],
                timesteps=aux["timesteps"],
                attention_mask=torch.full((n, 1), mask_value, dtype=torch.long),
                return_dict=True,
            )
        return out[1][:, -1, :].numpy()

    def test_dt_forward_uses_a_valid_all_ones_attention_mask(self):
        """The adapter must mark the single supplied token as observed."""
        state = _fixture()
        for model_id, adapter in state["adapters"].items():
            if adapter.record.family != "DT":
                continue
            rollout = state["rollouts"][model_id]
            n = min(8, rollout.states.shape[0])
            aux = {k: v[:n] for k, v in rollout.aux.items()}
            with torch.no_grad():
                got = adapter._forward_weights(rollout.states[:n], aux).numpy()
            np.testing.assert_allclose(
                got, self._dt_forward_with_mask(adapter, rollout, 1.0, n), atol=1e-6
            )

    def test_dt_forward_does_not_accumulate_inputs_on_the_model(self):
        """The shared DT class buffers every input; the adapter must not let it grow."""
        state = _fixture()
        for model_id, adapter in state["adapters"].items():
            if adapter.record.family != "DT":
                continue
            rollout = state["rollouts"][model_id]
            aux = {k: v[:2] for k, v in rollout.aux.items()}
            for _ in range(3):
                adapter.portfolio_weights(rollout.states[:2], aux)
            sizes = {k: len(v) for k, v in adapter.model.knowledge.items()}
            self.assertEqual(sizes, {"states": 0, "actions": 0, "rtgs": 0}, model_id)

    def test_legacy_all_zero_attention_mask_changes_the_dt_allocation(self):
        """Guards the audit finding: the legacy zero mask is not harmless.

        ``prepare_ablation_inputs`` passed ``attention_mask=zeros``, which marks
        the only token as padding; attention then spreads over the whole
        (return, state, action) triple instead of respecting causality. The
        resulting allocations differ, so the audit must not claim equivalence.
        """
        state = _fixture()
        checked = 0
        for model_id, adapter in state["adapters"].items():
            if adapter.record.family != "DT":
                continue
            rollout = state["rollouts"][model_id]
            n = min(8, rollout.states.shape[0])
            ones = self._dt_forward_with_mask(adapter, rollout, 1.0, n)
            zeros = self._dt_forward_with_mask(adapter, rollout, 0.0, n)
            self.assertGreater(np.abs(ones - zeros).sum(axis=1).mean(), 1e-4, model_id)
            checked += 1
        self.assertGreater(checked, 0, "no DT policy was available to check")


# ---------------------------------------------------------------------------
# Attribution, deletion and occlusion mechanics
# ---------------------------------------------------------------------------


@needs_models
class TestAttribution(unittest.TestCase):
    def _small(self, adapter, rollout, n: int = 4):
        return rollout.select(range(min(n, len(rollout))))

    def test_baselines_have_the_observation_shape(self):
        state = _fixture()
        for model_id, adapter in state["adapters"].items():
            frame = state["frames"][model_id].loc[: sampling.PERIODS["train"][1]]
            baselines = contribution.build_baselines(adapter, 4, frame, seed=0)
            self.assertEqual(set(baselines), set(contribution.BASELINE_NAMES))
            for name, tensor in baselines.items():
                self.assertEqual(
                    tuple(tensor.shape), (4, adapter.window, adapter.n_features), f"{model_id}/{name}"
                )
            self.assertEqual(float(baselines["zero"].abs().sum()), 0.0)

    def test_attribution_cubes_have_the_documented_shape(self):
        state = _fixture()
        for model_id, adapter in state["adapters"].items():
            sample = self._small(adapter, state["rollouts"][model_id])
            frame = state["frames"][model_id].loc[: sampling.PERIODS["train"][1]]
            baseline = contribution.build_baselines(adapter, len(sample), frame, seed=0)["train_mean"]
            occ = contribution.occlusion_attribution_cube(adapter, sample, baseline, coord_chunk=64)
            self.assertEqual(occ.shape, (len(sample), adapter.n_features, adapter.window))
            self.assertTrue(np.isfinite(occ).all())
            self.assertGreaterEqual(float(occ.min()), 0.0)
            ig, delta = contribution.integrated_gradients_cube(
                adapter, sample, baseline, n_steps=4, batch_size=len(sample)
            )
            self.assertEqual(ig.shape, occ.shape)
            self.assertTrue(np.isfinite(ig).all())
            self.assertTrue(np.isfinite(delta))

    def test_shares_are_normalised(self):
        state = _fixture()
        model_id, adapter = next(iter(state["adapters"].items()))
        sample = self._small(adapter, state["rollouts"][model_id])
        cube = np.abs(np.random.default_rng(0).normal(size=(len(sample), adapter.n_features, adapter.window)))
        shares = contribution.aggregate_cube(cube, adapter.feature_labels, adapter.window)
        for axis, series in shares.items():
            self.assertAlmostEqual(float(series.sum()), 1.0, places=6, msg=axis)
            self.assertGreaterEqual(float(series.min()), 0.0, axis)
        self.assertEqual(len(shares["lag"]), adapter.window)

    def test_deletion_mask_touches_only_the_selected_features(self):
        state = _fixture()
        model_id, adapter = next(iter(state["adapters"].items()))
        sample = self._small(adapter, state["rollouts"][model_id])
        replacement = torch.zeros_like(sample.states)
        chosen = [1, 3, 5]
        out = necessity._apply_group_deletion(sample.states, replacement, chosen)
        self.assertEqual(out.shape, sample.states.shape)
        for f in range(adapter.n_features):
            if f in chosen:
                self.assertTrue(torch.allclose(out[:, :, f], replacement[:, :, f]))
            else:
                self.assertTrue(torch.equal(out[:, :, f], sample.states[:, :, f]))

    def test_temporal_occlusion_replaces_whole_timesteps(self):
        state = _fixture()
        model_id, adapter = next(iter(state["adapters"].items()))
        sample = self._small(adapter, state["rollouts"][model_id])
        replacement = torch.zeros_like(sample.states)
        profiles = necessity.temporal_occlusion_profiles(adapter, sample, replacement, "primary")
        for key, profile in profiles.items():
            self.assertEqual(profile.shape, (len(sample), adapter.window), key)
            self.assertTrue(np.isfinite(profile).all(), key)
            self.assertGreaterEqual(float(profile.min()), 0.0, key)

    def test_feature_ranking_is_a_permutation(self):
        rng = np.random.default_rng(0)
        cube = np.abs(rng.normal(size=(5, 8, 3)))
        labels = [(f"i{k}", "RSI") for k in range(8)]
        order = necessity.feature_group_ranking(cube, labels)
        self.assertEqual(sorted(order.tolist()), list(range(8)))
        scores = np.abs(cube).sum(axis=2).mean(axis=0)
        self.assertEqual(order[0], int(np.argmax(scores)))

    def test_bootstrap_rank_stability_is_deterministic(self):
        rng = np.random.default_rng(0)
        cube = np.abs(rng.normal(size=(16, 6, 4)))
        labels = [(f"i{k}", "RSI") for k in range(6)]
        regimes = np.array(["bull_market"] * 8 + ["crash_period"] * 8, dtype=object)
        a = contribution.bootstrap_rank_stability(cube, labels, regimes, n_bootstrap=20, seed=7)
        b = contribution.bootstrap_rank_stability(cube, labels, regimes, n_bootstrap=20, seed=7)
        self.assertEqual(a["median_spearman"], b["median_spearman"])
        self.assertLessEqual(a["median_spearman"], 1.0)
        self.assertEqual(len(a["groups"]), 6)

    def test_degenerate_attribution_is_reported_not_scored(self):
        cube = np.zeros((8, 6, 4))
        labels = [(f"i{k}", "RSI") for k in range(6)]
        regimes = np.array(["bull_market"] * 8, dtype=object)
        result = contribution.bootstrap_rank_stability(cube, labels, regimes, n_bootstrap=5, seed=1)
        self.assertTrue(result["degenerate"])
        self.assertTrue(np.isnan(result["median_spearman"]))


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


class TestStructure(unittest.TestCase):
    def test_neighbourhood_overlap_bounds(self):
        a = np.array([[1, 2], [0, 2], [0, 1]])
        self.assertAlmostEqual(structure.neighbourhood_overlap(a, a), 1.0)
        b = np.array([[3, 4], [3, 4], [3, 4]])
        self.assertAlmostEqual(structure.neighbourhood_overlap(a, b), 0.0)

    def test_knn_excludes_self_and_returns_k_neighbours(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(20, 5))
        idx = structure.knn_indices(x, k=3)
        self.assertEqual(idx.shape, (20, 3))
        for row in range(20):
            self.assertNotIn(row, idx[row].tolist())

    def test_standardise_drops_constant_dimensions(self):
        x = np.column_stack([np.arange(10.0), np.ones(10)])
        out = structure.standardise(x)
        self.assertEqual(out.shape[1], 1)
        self.assertAlmostEqual(float(out.mean()), 0.0, places=6)

    def test_zero_layer_influence_is_undefined_not_perfect(self):
        """A policy no layer affects must not be scored as perfectly dispersed."""
        from unittest import mock

        with mock.patch.object(
            structure, "_layer_ablation", return_value={"a": 0.0, "b": 0.0, "c": 0.0}
        ):
            out = structure.layer_influence(None, None, None, "primary", gradients_usable=False)
        self.assertFalse(out["measurable"])
        self.assertTrue(np.isnan(out["metrics"]["non_concentrated_layer_influence"]))
        self.assertTrue(np.isnan(out["metrics"]["layer_influence_max_share"]))
        self.assertEqual(out["metrics"]["n_layers"], 3)

    def test_measurable_layer_influence_is_scored(self):
        from unittest import mock

        with mock.patch.object(
            structure, "_layer_ablation", return_value={"a": 1.0, "b": 1.0, "c": 1.0}
        ):
            even = structure.layer_influence(None, None, None, "primary", gradients_usable=False)
        with mock.patch.object(
            structure, "_layer_ablation", return_value={"a": 1.0, "b": 0.0, "c": 0.0}
        ):
            skewed = structure.layer_influence(None, None, None, "primary", gradients_usable=False)
        self.assertAlmostEqual(even["metrics"]["non_concentrated_layer_influence"], 1.0)
        self.assertAlmostEqual(skewed["metrics"]["non_concentrated_layer_influence"], 0.0)
        self.assertAlmostEqual(even["metrics"]["layer_influence_max_share"], 1 / 3)

    def test_concentration_indices_are_bounded(self):
        uniform = np.full(4, 0.25)
        concentrated = np.array([1.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(contribution.normalised_entropy(uniform), 1.0)
        self.assertAlmostEqual(contribution.normalised_entropy(concentrated), 0.0)
        self.assertAlmostEqual(contribution.hhi(uniform), 0.25)
        self.assertAlmostEqual(contribution.hhi(concentrated), 1.0)
        self.assertAlmostEqual(contribution.effective_number(uniform), 4.0)


# ---------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------


class TestOutcomes(unittest.TestCase):
    def _weights(self, index):
        w = pd.DataFrame(0.0, index=index, columns=list(discovery.CANONICAL_INSTRUMENTS))
        w["^GSPC"] = 1.0
        return w

    def test_simplex_check_flags_violations(self):
        index = pd.bdate_range("2020-06-01", periods=5)
        good = self._weights(index)
        self.assertFalse(outcomes.check_simplex(good)["violates_long_only"])
        self.assertFalse(outcomes.check_simplex(good)["violates_full_investment"])
        bad = good.copy()
        bad.iloc[0, 0] = -0.5
        self.assertTrue(outcomes.check_simplex(bad)["violates_long_only"])
        unnormalised = good * 2.0
        self.assertTrue(outcomes.check_simplex(unnormalised)["violates_full_investment"])

    def test_cash_earns_zero(self):
        index = pd.bdate_range("2020-06-01", periods=6)
        prices = pd.DataFrame(
            {("CASH", "Close"): np.linspace(100, 110, 6), ("^GSPC", "Close"): np.linspace(1, 2, 6)},
            index=index,
        )
        prices.columns = pd.MultiIndex.from_tuples(prices.columns)
        rets = outcomes.hold_returns(prices, list(discovery.CANONICAL_INSTRUMENTS))
        self.assertTrue((rets["CASH"] == 0.0).all())
        self.assertEqual(list(rets.columns), list(discovery.CANONICAL_INSTRUMENTS))

    def test_returns_stay_on_the_trading_calendar(self):
        index = pd.bdate_range("2020-06-01", periods=10)
        prices = pd.DataFrame(
            {("^GSPC", "Close"): np.linspace(100, 110, 10)}, index=index
        )
        prices.columns = pd.MultiIndex.from_tuples(prices.columns)
        rets = outcomes.hold_returns(prices, list(discovery.CANONICAL_INSTRUMENTS))
        pnl = outcomes.portfolio_returns(self._weights(index), rets)
        self.assertTrue(set(pnl.index) <= set(index))
        self.assertEqual(pnl.index.dayofweek.max() <= 4, True)
        self.assertEqual(len(pnl), len(index) - 1)

    def test_weights_are_held_over_the_following_interval(self):
        index = pd.bdate_range("2020-06-01", periods=4)
        rets = pd.DataFrame(
            0.0, index=index, columns=list(discovery.CANONICAL_INSTRUMENTS)
        )
        rets.loc[index[2], "^GSPC"] = 0.10
        pnl = outcomes.portfolio_returns(self._weights(index), rets, cost=0.0)
        self.assertAlmostEqual(float(pnl.loc[index[2], "gross_return"]), 0.10)
        self.assertAlmostEqual(float(pnl.loc[index[1], "gross_return"]), 0.0)

    def test_transaction_costs_reduce_net_return(self):
        index = pd.bdate_range("2020-06-01", periods=4)
        rets = pd.DataFrame(0.0, index=index, columns=list(discovery.CANONICAL_INSTRUMENTS))
        w = self._weights(index)
        w.loc[index[2], "^GSPC"] = 0.0
        w.loc[index[2], "CASH"] = 1.0
        pnl = outcomes.portfolio_returns(w, rets, cost=outcomes.TRANSACTION_COST)
        self.assertAlmostEqual(float(pnl.loc[index[2], "traded_fraction"]), 1.0)
        self.assertAlmostEqual(
            float(pnl.loc[index[2], "net_return"]), -outcomes.TRANSACTION_COST, places=9
        )
        self.assertLessEqual(pnl["net_return"].sum(), pnl["gross_return"].sum())

    def test_performance_metrics_are_finite_for_a_normal_series(self):
        rng = np.random.default_rng(0)
        series = pd.Series(rng.normal(0.0005, 0.01, 500))
        metrics = outcomes.performance_metrics(series)
        for key in ("annualised_return", "annualised_volatility", "sharpe_ratio", "max_drawdown"):
            self.assertTrue(np.isfinite(metrics[key]), key)
        self.assertLessEqual(metrics["max_drawdown"], 0.0)

    def test_cash_by_regime_covers_all_regimes(self):
        index = pd.bdate_range(*sampling.PERIODS["eval"])
        mean, frame = outcomes.cash_allocation(self._weights(index))
        self.assertAlmostEqual(mean, 0.0)
        self.assertEqual(list(frame["regime"]), list(sampling.REGIME_ORDER))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


class TestScoring(unittest.TestCase):
    def test_orientation_clips_to_the_unit_interval(self):
        self.assertEqual(scoring.orient(-0.4, "clip_rank_correlation"), 0.0)
        self.assertEqual(scoring.orient(1.6, "identity_unit"), 1.0)
        self.assertTrue(np.isnan(scoring.orient(float("nan"), "identity_unit")))

    def test_dimension_scores_average_their_two_metrics(self):
        metrics = {name: 0.5 for name in scoring.METRIC_SPEC}
        out = scoring.score_model(metrics)
        for dim in scoring.DIMENSIONS:
            self.assertAlmostEqual(out[f"{dim}_score"], 0.5)
            self.assertEqual(out[f"{dim}_metrics_available"], 2)
        self.assertAlmostEqual(out["aggregate_diagnostic_score"], 0.5)

    def test_larger_is_better_for_every_metric(self):
        low = scoring.score_model({name: 0.2 for name in scoring.METRIC_SPEC})
        high = scoring.score_model({name: 0.8 for name in scoring.METRIC_SPEC})
        self.assertLess(low["aggregate_diagnostic_score"], high["aggregate_diagnostic_score"])

    def test_missing_dimension_makes_the_aggregate_undefined(self):
        metrics = {name: 0.5 for name in scoring.METRIC_SPEC}
        for name, spec in scoring.METRIC_SPEC.items():
            if spec["dimension"] == "structure":
                metrics[name] = float("nan")
        out = scoring.score_model(metrics)
        self.assertTrue(np.isnan(out["structure_score"]))
        self.assertTrue(np.isnan(out["aggregate_diagnostic_score"]))
        self.assertFalse(np.isnan(out["aggregate_partial_score"]))

    def test_partial_dimension_uses_the_available_metric(self):
        metrics = {name: 0.5 for name in scoring.METRIC_SPEC}
        metrics["non_concentrated_layer_influence"] = float("nan")
        out = scoring.score_model(metrics)
        self.assertAlmostEqual(out["structure_score"], 0.5)
        self.assertEqual(out["structure_metrics_available"], 1)

    def test_bootstrap_ci_brackets_the_mean(self):
        values = [0.1, 0.4, 0.5, 0.9]
        stat = scoring.bootstrap_ci(values, n_bootstrap=500, seed=3)
        self.assertLessEqual(stat["lower"], stat["mean"])
        self.assertLessEqual(stat["mean"], stat["upper"])
        self.assertEqual(stat["n"], 4)

    def test_rank_correlation_is_withheld_at_small_n(self):
        table = pd.DataFrame(
            {
                "model_id": ["a", "b"],
                "family": ["DT", "RL"],
                "aggregate_diagnostic_score": [0.4, 0.6],
                "gross_sharpe_ratio": [0.1, 0.2],
            }
        )
        out = scoring.rank_correlations(table, ["aggregate_diagnostic_score"], "gross_sharpe_ratio")
        self.assertTrue(np.isnan(out.iloc[0]["spearman"]))
        self.assertIn("not computed", out.iloc[0]["note"])

    def test_metric_dictionary_documents_every_metric(self):
        frame = scoring.metric_dictionary()
        self.assertEqual(set(frame["metric"]), set(scoring.METRIC_SPEC))
        self.assertEqual(set(frame["direction"]), {"larger is better"})


# ---------------------------------------------------------------------------
# Reporting and generated artefacts
# ---------------------------------------------------------------------------


class TestReporting(unittest.TestCase):
    def _table(self):
        rows = []
        for i, (mid, fam, algo) in enumerate(
            [("DT_a", "DT", "DecisionTransformer"), ("RL_b", "RL", "PPO")]
        ):
            metrics = {name: 0.4 + 0.1 * i for name in scoring.METRIC_SPEC}
            row = {
                "model_id": mid,
                "family": fam,
                "algorithm": algo,
                "gross_sharpe_ratio": 0.5 + i,
                **metrics,
                **scoring.score_model(metrics),
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def test_main_table_has_no_placeholder(self):
        tex = reporting.main_latex_table(self._table())
        self.assertIn(r"\begin{tabular}", tex)
        for rule in (r"\toprule", r"\midrule", r"\bottomrule"):
            self.assertIn(rule, tex)
        self.assertIn(r"\resizebox{\textwidth}", tex)
        for token in ("TODO", "XXX", "PLACEHOLDER", "nan"):
            self.assertNotIn(token, tex)
        self.assertIn("gross", tex)

    def test_markdown_table_handles_missing_values(self):
        frame = pd.DataFrame({"a": [1.0, float("nan")], "b": ["x", "y"]})
        text = reporting.markdown_table(frame)
        self.assertIn("n/a", text)
        self.assertEqual(text.count("\n"), 3)

    def test_figures_are_written_in_both_formats(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            paths = reporting.figure_diagnostics_heatmap(self._table(), Path(tmp) / "heatmap")
            self.assertEqual({p.suffix for p in paths}, {".png", ".pdf"})
            for p in paths:
                self.assertTrue(p.exists() and p.stat().st_size > 0)


class TestGeneratedArtefacts(unittest.TestCase):
    """Verify the artefacts of the reported run, when it has been executed."""

    EXPECTED = [
        "audit/implementation_audit.md",
        "audit/decision_targets.csv",
        "audit/reconstruction_checks.csv",
        "data/common_diagnostic_dates.csv",
        "tables/sample_sizes_by_model_regime.csv",
        "tables/cross_architecture_model_summary.csv",
        "manuscript/cross_architecture_summary.tex",
        "manuscript/non_transformer_benchmark_methods.tex",
        "manuscript/cross_architecture_results.tex",
        "manuscript/cross_architecture_discussion.tex",
        "figures/F_cross_architecture_diagnostics_heatmap.png",
        "figures/F_cross_architecture_diagnostics_heatmap.pdf",
        "figures/F_cross_architecture_functional_comparison.png",
        "figures/F_cross_architecture_functional_comparison.pdf",
        "supplementary/cross_architecture_supplement.tex",
        "supplementary/model_registry.csv",
        "missing_inputs.md",
        "run_report.md",
    ]

    @unittest.skipUnless(
        (RESULTS_DIR / "run_report.md").exists(), "the experiment has not been run yet"
    )
    def test_every_expected_artefact_exists(self):
        for rel in self.EXPECTED:
            path = RESULTS_DIR / rel
            self.assertTrue(path.exists(), rel)
            self.assertGreater(path.stat().st_size, 0, rel)

    @unittest.skipUnless(
        (RESULTS_DIR / "data" / "common_diagnostic_dates.csv").exists(),
        "the experiment has not been run yet",
    )
    def test_reported_sample_has_no_unknown_regime_and_no_leakage(self):
        frame = pd.read_csv(RESULTS_DIR / "data" / "common_diagnostic_dates.csv", parse_dates=["date"])
        self.assertFalse(frame["date"].duplicated().any())
        self.assertNotIn("Unknown", set(frame["regime"]))
        self.assertLess(frame["date"].max(), pd.Timestamp(sampling.PERIODS["holdout"][0]))
        self.assertEqual(set(frame["regime"]), set(sampling.REGIME_ORDER))

    @unittest.skipUnless(
        (RESULTS_DIR / "tables" / "cross_architecture_model_summary.csv").exists(),
        "the experiment has not been run yet",
    )
    def test_reported_scores_are_in_range_and_cover_both_families(self):
        frame = pd.read_csv(RESULTS_DIR / "tables" / "cross_architecture_model_summary.csv")
        self.assertTrue({"DT", "RL"} <= set(frame["family"]))
        for col in [f"oriented_{m}" for m in scoring.METRIC_SPEC] + [
            "contribution_score",
            "necessity_score",
            "structure_score",
            "aggregate_diagnostic_score",
        ]:
            values = pd.to_numeric(frame[col], errors="coerce").dropna()
            self.assertTrue(((values >= 0.0) & (values <= 1.0)).all(), col)
        weights = pd.to_numeric(frame["simplex_max_abs_sum_deviation"], errors="coerce")
        self.assertTrue((weights < 1e-4).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
