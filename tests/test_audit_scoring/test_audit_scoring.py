"""Unit tests for Batch A audit scoring (synthetic data only)."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from audit_scoring.aggregation import aggregate_family_raw
from audit_scoring.config import PROFILES, validate_weights
from audit_scoring.contribution import (
    align_attribution_matrix,
    compute_contribution_diagnostics,
)
from audit_scoring.decision_rules import (
    DECISION_MATRIX,
    assign_diagnostic_band,
    downgrade_band,
    operational_decision,
)
from audit_scoring.feature_groups import build_taxonomy, map_feature_name
from audit_scoring.io import assert_not_holdout_path, load_pickle, to_numpy
from audit_scoring.io import technical_indicator_flag_from_path
from audit_scoring.model_mapping import (
    deduplicate_paths,
    experiment_name_from_filename,
    map_one_pkl,
    normalize_name,
)
from audit_scoring.necessity import (
    compute_necessity_diagnostics,
    compute_normalized_occlusion_effect,
)
from audit_scoring.scoring import average_rank, percentile_scores, score_family_medians, score_model_metrics
from audit_scoring.structure import effective_rank_ratio, majority_class_rate, compute_structure_diagnostics


def test_holdout_loading_prohibited(tmp_path: Path):
    banned = tmp_path / "holdout_metrics_family.csv"
    banned.write_text("x\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        assert_not_holdout_path(banned)


def test_cloudpickle_loading(tmp_path: Path):
    path = tmp_path / "toy.pkl"
    payload = {"a": np.arange(3), "b": {"c": 1}}
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    loaded = load_pickle(path)
    assert loaded["b"]["c"] == 1
    assert np.allclose(to_numpy(loaded["a"]), np.arange(3))


def test_technical_indicator_folder_extraction(tmp_path: Path):
    p1 = tmp_path / "techindi" / "DT_16_1" / "model_diagnostics_x.pkl"
    p2 = tmp_path / "no_techindi" / "DT_16_1" / "model_diagnostics_x.pkl"
    p1.parent.mkdir(parents=True)
    p2.parent.mkdir(parents=True)
    p1.write_bytes(b"")
    p2.write_bytes(b"")
    assert technical_indicator_flag_from_path(p1) is True
    assert technical_indicator_flag_from_path(p2) is False


def test_ambiguous_mapping_detection(tmp_path: Path):
    catalog = pd.DataFrame(
        {
            "model_id": ["same_norm_a", "same_norm_b"],
            "model_family": ["DT_16_1_TI_True", "DT_16_1_TI_False"],
        }
    )
    # Force normalized collision by monkeypatching names to identical normalize
    path = tmp_path / "model_diagnostics_same_norm.pkl"
    path.write_bytes(b"")
    # Without experiment map and without exact match -> unresolved rather than guess
    result = map_one_pkl(
        path,
        tmp_path,
        expected_families=["DT_16_1_TI_True", "DT_16_1_TI_False"],
        model_catalog=catalog,
        experiment_to_family={},
    )
    assert result.mapping_status in {"unresolved", "ambiguous"}


def test_model_name_reconciliation():
    assert experiment_name_from_filename(Path("model_diagnostics_RL_AG_DT_refdate_2016-01-01_20250101_0000.pkl")).startswith(
        "RL_AG_DT_"
    )
    assert normalize_name("Model_Diagnostics_ABC") == "abc" or "abc" in normalize_name("model_diagnostics_abc")


def test_date_filtering():
    dates = pd.DatetimeIndex(
        list(pd.date_range("2016-01-01", periods=10, freq="D"))
        + list(pd.date_range("2024-05-01", periods=5, freq="D"))
    )
    n = len(dates)
    attr = np.random.randn(n, 4)
    names = np.array(["^GSPC_Close_0", "^GSPC_RSI_0", "^GSPC_ATR_0", "rtg_0"])
    matrix, _, di, _, meta = align_attribution_matrix(
        attr,
        names,
        dates=dates,
        validation_start=pd.Timestamp("2016-01-01"),
        validation_end=pd.Timestamp("2024-04-30"),
    )
    assert meta["n_after_validation"] == 5
    assert matrix.shape[0] == 10
    assert di.max() <= pd.Timestamp("2024-04-30")


def test_attribution_feature_alignment():
    attr = np.random.randn(20, 2, 3)
    names = np.array([["a_0", "a_1", "a_2"], ["b_0", "b_1", "b_2"]])
    matrix, flat, *_ = align_attribution_matrix(attr, names)
    assert matrix.shape == (20, 6)
    assert len(flat) == 6


def test_bootstrap_determinism():
    rng = np.random.default_rng(0)
    attr = rng.normal(size=(80, 6))
    names = np.array(["^GSPC_Close_0", "^GSPC_RSI_0", "^GSPC_ATR_0", "^GSPC_Volume_0", "rtg_0", "reward_0"])
    dates = pd.date_range("2016-01-01", periods=80, freq="D")
    labels = np.array(["bull_market"] * 40 + ["crash_period"] * 40)
    a = compute_contribution_diagnostics(
        attr, names, dates=dates, labels=labels, bootstrap_replicates=20, random_seed=123
    )
    b = compute_contribution_diagnostics(
        attr, names, dates=dates, labels=labels, bootstrap_replicates=20, random_seed=123
    )
    assert a.contribution_rank_stability == pytest.approx(b.contribution_rank_stability)
    assert a.contribution_sign_stability == pytest.approx(b.contribution_sign_stability)


def test_kendall_and_sign_stability_finite():
    rng = np.random.default_rng(1)
    attr = rng.normal(size=(60, 5))
    # Make family structure stable
    attr[:, 0] += 2
    names = np.array(["x_Close_0", "x_RSI_0", "x_ATR_0", "x_Volume_0", "rtg_0"])
    dates = pd.date_range("2016-01-01", periods=60, freq="D")
    res = compute_contribution_diagnostics(
        attr, names, dates=dates, bootstrap_replicates=15, random_seed=7, min_observations=30
    )
    assert np.isfinite(res.contribution_rank_stability)
    assert np.isfinite(res.contribution_sign_stability)


def test_temporal_instability_and_occlusion_gate():
    baseline = np.ones((30, 3))
    curves = baseline[:, :, None] + np.linspace(0.1, 0.5, 5).reshape(1, 1, 5)
    effect = compute_normalized_occlusion_effect(curves, baseline)
    assert effect > 0
    nec = compute_necessity_diagnostics(
        {
            "occlusion_curves": curves,
            "baseline_output": baseline,
        },
        {
            "stability_metrics": {
                "mean_window_correlation": 0.9,
                "curve_coefficient_of_variation": 0.2,
                "half_life_coefficient_of_variation": 0.3,
                "auc_coefficient_of_variation": 0.25,
            }
        },
    )
    assert nec.necessity_non_degeneracy_pass
    assert nec.necessity_temporal_instability == pytest.approx(0.25)


def test_effective_rank_and_probe_margin():
    rng = np.random.default_rng(2)
    hidden = rng.normal(size=(100, 8))
    ratio, n = effective_rank_ratio(hidden)
    assert n == 100
    assert 0 < ratio <= 1
    labels = np.array(["a"] * 70 + ["b"] * 30)
    assert majority_class_rate(labels) == pytest.approx(0.7)
    struct = compute_structure_diagnostics(
        latent_representations={"hidden": hidden, "labels": labels},
        linear_probe_results={
            "layer_0": {"cv_accuracy_mean": 0.8, "cv_accuracy_std": 0.05},
            "layer_1": {"cv_accuracy_mean": 0.85, "cv_accuracy_std": 0.04},
        },
        regime_labels=labels,
    )
    assert struct.structure_probe_margin == pytest.approx(0.15)


def test_percentile_directions_and_ties():
    higher = percentile_scores([1.0, 2.0, 3.0], higher_is_better=True)
    lower = percentile_scores([1.0, 2.0, 3.0], higher_is_better=False)
    assert higher[-1] > higher[0]
    assert lower[0] > lower[-1]
    ranks = average_rank([1.0, 1.0, 2.0], higher_is_better=True)
    assert ranks[0] == ranks[1]


def test_missing_core_metrics_not_imputed():
    raw = pd.DataFrame(
        {
            "model_id": ["m1", "m2"],
            "model_family": ["F1", "F1"],
            "model_type": ["dt", "dt"],
            "technical_indicators": [True, True],
            "contribution_rank_stability": [0.8, np.nan],
            "contribution_sign_stability": [0.7, 0.6],
            "necessity_temporal_profile_stability": [0.9, 0.8],
            "necessity_temporal_instability": [0.2, 0.3],
            "structure_effective_rank_ratio": [0.4, 0.5],
            "structure_probe_margin": [0.1, 0.2],
            "overall_raw_validity_status": ["passed", "incomplete"],
            "warnings": ["", "missing"],
        }
    )
    scored = score_model_metrics(raw)
    assert scored.loc[1, "model_score_status"] == "not_scored_missing_core_dimension"
    assert np.isnan(scored.loc[1, "diagnostic_score_model"])


def test_family_aggregation_and_scoring():
    raw = pd.DataFrame(
        {
            "model_id": [f"m{i}" for i in range(6)],
            "model_family": ["A"] * 3 + ["B"] * 3,
            "contribution_rank_stability": [0.9, 0.8, 0.85, 0.2, 0.25, 0.22],
            "contribution_sign_stability": [0.9, 0.88, 0.91, 0.3, 0.28, 0.33],
            "necessity_temporal_profile_stability": [0.95, 0.94, 0.96, 0.4, 0.41, 0.39],
            "necessity_temporal_instability": [0.1, 0.12, 0.11, 0.5, 0.55, 0.52],
            "structure_effective_rank_ratio": [0.6, 0.62, 0.61, 0.2, 0.22, 0.21],
            "structure_probe_margin": [0.2, 0.22, 0.21, 0.01, 0.0, 0.02],
        }
    )
    fam_raw = aggregate_family_raw(raw, ["A", "B"], {"A": 3, "B": 3})
    assert fam_raw.loc[fam_raw.model_family == "A", "median_contribution_rank_stability"].iloc[0] == pytest.approx(0.85)
    fam_scored = score_family_medians(fam_raw)
    assert fam_scored.loc[fam_scored.model_family == "A", "diagnostic_score"].iloc[0] > fam_scored.loc[
        fam_scored.model_family == "B", "diagnostic_score"
    ].iloc[0]


def test_weight_constraints():
    validate_weights(PROFILES["balanced"])
    with pytest.raises(ValueError):
        validate_weights({"contribution": 0.9, "necessity": 0.05, "structure": 0.05})


def test_diagnostic_band_precedence():
    thr = {"D_p25": 25.0, "D_p50": 50.0, "D_p75": 75.0}
    band, reason = assign_diagnostic_band(
        {
            "diagnostic_score": 90,
            "contribution_score": 80,
            "necessity_score": 80,
            "structure_score": 80,
            "model_coverage_ratio": 1.0,
            "high_within_family_dispersion": False,
            "family_score_status": "scored",
            "mandatory_validity_status": "failed",
        },
        thr,
    )
    assert band == "POOR"
    assert "mandatory" in reason


def test_performance_band_downgrade():
    assert downgrade_band("STRONG", 1) == "ADEQUATE"
    assert downgrade_band("WEAK", 5) == "POOR"


def test_every_decision_matrix_cell():
    for perf in ["STRONG", "ADEQUATE", "WEAK", "POOR"]:
        for diag in ["STRONG", "ADEQUATE", "WEAK", "POOR"]:
            decision, _ = operational_decision(perf, diag)
            assert decision == DECISION_MATRIX[(perf, diag)]
    decision, reason = operational_decision("STRONG", "STRONG", mandatory_gate_failed=True)
    assert decision == "REJECT"
    assert "override" in reason


def test_feature_taxonomy_unclassified():
    tax = build_taxonomy(["^GSPC_Close_3", "weird_xyz_feature", "rtg_0", "CASH_0"])
    assert "price_level" in set(tax["feature_family"])
    assert "unclassified" in set(tax["feature_family"])
    assert map_feature_name("rtg_5").feature_family == "return_to_go"


def test_duplicate_path_preference(tmp_path: Path):
    root = tmp_path
    complete = root / "model_diagnostics_expA.pkl"
    partial = root / "partial" / "model_diagnostics_expA.pkl"
    partial.parent.mkdir(parents=True)
    complete.write_bytes(b"1")
    partial.write_bytes(b"1")
    kept, dups = deduplicate_paths([complete, partial], root, prefer_complete_over_partial=True)
    assert kept == [complete]
    assert dups and dups[0].mapping_status == "duplicate"


def test_deterministic_sorting():
    df = pd.DataFrame(
        {
            "operational_recommendation": ["REJECT", "ADVANCE", "MONITOR"],
            "diagnostic_score": [10, 20, 90],
            "median_sharpe": [1.0, 0.5, 0.2],
        }
    )
    order = {"ADVANCE": 0, "MONITOR": 1, "INVESTIGATE": 2, "REJECT": 3}
    df["decision_rank"] = df["operational_recommendation"].map(order)
    out = df.sort_values(
        ["decision_rank", "diagnostic_score", "median_sharpe"],
        ascending=[True, False, False],
    )
    assert list(out["operational_recommendation"]) == ["ADVANCE", "MONITOR", "REJECT"]
