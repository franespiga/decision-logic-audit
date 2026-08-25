"""Unit tests for Batch C diagnostic-validity package (synthetic data)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from src.audit_scoring.baseline_validity import (
    build_empirical_median_baseline,
    build_zero_baseline,
    classify_baseline_status,
    pairwise_concordance,
    signed_agreement,
    top_k_jaccard,
)
from src.audit_scoring.checkpoint_loading import (
    allocation_distance,
    cash_and_risky_deltas,
    total_variation_distance,
)
from src.audit_scoring.faithfulness_validity import (
    classify_faithfulness,
    paired_bootstrap_ci,
    replace_families_with_baseline,
    restore_families_from_original,
)
from src.audit_scoring.randomization_validity import classify_randomization_status
from src.audit_scoring.validation_sampling import assert_validation_dates, stratified_sample_indices
from src.audit_scoring.validity_aggregation import combined_validity_status
from src.audit_scoring.validity_guards import SourceAccessLog, assert_permitted_analytical_load
from src.audit_scoring.holdout_assessment import INCOMPLETE_EVIDENCE_FAMILY


def test_validation_date_filtering():
    dates = pd.date_range("2016-01-01", periods=10, freq="B")
    assert_validation_dates(dates, pd.Timestamp("2024-04-30"))
    with pytest.raises(AssertionError):
        assert_validation_dates(pd.to_datetime(["2024-05-01"]), pd.Timestamp("2024-04-30"))


def test_stratified_deterministic_sampling():
    dates = pd.to_datetime(
        list(pd.date_range("2016-01-01", periods=40, freq="30D"))
        + list(pd.date_range("2020-03-01", periods=20, freq="3D"))
        + list(pd.date_range("2020-07-01", periods=40, freq="10D"))
        + list(pd.date_range("2022-01-01", periods=40, freq="15D"))
    )
    a, _, _ = stratified_sample_indices(dates, n_samples=64, per_regime=16, random_seed=42)
    b, _, _ = stratified_sample_indices(dates, n_samples=64, per_regime=16, random_seed=42)
    assert np.array_equal(a, b)
    assert len(a) <= 64


def test_zero_and_median_baselines():
    states = torch.randn(10, 2, 5)
    z = build_zero_baseline(states)
    assert torch.all(z == 0)
    med = build_empirical_median_baseline(states, states[:3])
    assert med.shape == states[:3].shape


def test_kendall_and_jaccard_and_sign():
    abs_a = pd.Series({"a": 3.0, "b": 2.0, "c": 1.0, "d": 0.5})
    abs_b = pd.Series({"a": 2.5, "b": 2.2, "c": 0.8, "d": 0.4})
    signed_a = pd.Series({"a": 1.0, "b": -1.0, "c": 1.0, "d": -1.0})
    signed_b = pd.Series({"a": 0.5, "b": -0.2, "c": 0.1, "d": 0.3})
    pair = pairwise_concordance(
        {"zero": abs_a, "median": abs_b, "bg": abs_a},
        {"zero": signed_a, "median": signed_b, "bg": signed_a},
        top_k=2,
    )
    assert not pair.empty
    assert top_k_jaccard(["a", "b", "c"], ["a", "c", "d"], 2) == pytest.approx(1.0 / 3.0)
    assert signed_agreement(signed_a, signed_b, ["a", "b"]) == 1.0


def test_baseline_status_rules():
    pair = pd.DataFrame(
        {
            "kendall_tau_b": [0.7, 0.6, 0.55],
            "top_k_jaccard": [0.6, 0.6, 0.6],
            "top_k_sign_agreement": [0.8, 0.8, 0.8],
        }
    )
    status, _ = classify_baseline_status(pair)
    assert status == "SUPPORTED"


def test_tv_and_cash():
    a = np.array([[0.2, 0.3, 0.5]])
    b = np.array([[0.1, 0.4, 0.5]])
    d = total_variation_distance(a, b)
    assert np.isclose(d[0], 0.1)
    dist, name = allocation_distance(a, b)
    assert name == "total_variation"
    cash, risky = cash_and_risky_deltas(a, b, cash_idx=2)
    assert cash[0] == 0.0


def test_randomization_gap_and_status():
    status, metrics = classify_randomization_status(
        baseline_reference_concordance=0.6,
        full_randomization_similarity=0.1,
        median_action_distance=0.2,
    )
    assert status == "SUPPORTED"
    assert metrics["randomization_gap"] == pytest.approx(0.5)


def test_grouped_deletion_insertion_masks():
    states = torch.arange(12, dtype=torch.float32).view(1, 2, 6)
    baseline = torch.zeros_like(states)
    family_ids = np.array(["A", "A", "B", "B", "C", "C"])
    out = replace_families_with_baseline(states, baseline, family_ids, ["A"])
    assert torch.all(out[..., :2] == 0)
    assert torch.all(out[..., 2:] == states[..., 2:])
    restored = restore_families_from_original(baseline, states, family_ids, ["B"])
    assert torch.all(restored[..., 2:4] == states[..., 2:4])
    assert torch.all(restored[..., :2] == 0)


def test_matched_random_counts_and_bootstrap():
    gaps = np.linspace(0.1, 0.3, 20)
    mean, lo, hi, eff = paired_bootstrap_ci(gaps, replicates=200, seed=42)
    mean2, lo2, hi2, _ = paired_bootstrap_ci(gaps, replicates=200, seed=42)
    assert mean == mean2 and lo == lo2 and hi == hi2
    assert classify_faithfulness(0.2, 0.05, 0.5) == "SUPPORTED"
    assert classify_faithfulness(0.2, -0.01, 0.5) == "MIXED"
    assert classify_faithfulness(-0.1, -0.2, 0.1) == "UNSUPPORTED"


def test_combined_status_precedence():
    st, _ = combined_validity_status(
        baseline_status="SUPPORTED",
        randomization_status="SUPPORTED",
        deletion_status="SUPPORTED",
        insertion_status="MIXED",
        batch_a_mandatory_failed=False,
    )
    assert st == "SUPPORTED"
    st, _ = combined_validity_status(
        baseline_status="SUPPORTED",
        randomization_status="UNSUPPORTED",
        deletion_status="SUPPORTED",
        insertion_status="SUPPORTED",
        batch_a_mandatory_failed=False,
    )
    assert st == "UNSUPPORTED"
    st, _ = combined_validity_status(
        baseline_status="SUPPORTED",
        randomization_status="SUPPORTED",
        deletion_status="SUPPORTED",
        insertion_status="SUPPORTED",
        batch_a_mandatory_failed=False,
        force_incomplete=True,
    )
    assert st == "INCOMPLETE"


def test_holdout_source_prohibition(tmp_path: Path):
    log = SourceAccessLog()
    bad = tmp_path / "holdout_metrics_family.csv"
    bad.write_text("x\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        assert_permitted_analytical_load(bad, role="test", log=log)


def test_module_randomization_and_restore():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.predict_action = nn.Linear(4, 3)
            self.other = nn.Linear(4, 4)

        def forward(self, x):
            return self.predict_action(x)

    m = Tiny()
    w0 = m.predict_action.weight.detach().clone()
    torch.manual_seed(0)
    m.predict_action.reset_parameters()
    assert not torch.allclose(w0, m.predict_action.weight)
    # restore
    with torch.no_grad():
        m.predict_action.weight.copy_(w0)
    assert torch.allclose(w0, m.predict_action.weight)


def test_incomplete_family_constant():
    assert INCOMPLETE_EVIDENCE_FAMILY == "no_techindi_full_TI_False"


def test_deterministic_output_ordering():
    idx1, _, _ = stratified_sample_indices(
        pd.date_range("2016-01-01", periods=200, freq="B"),
        n_samples=32,
        per_regime=8,
        random_seed=42,
    )
    idx2, _, _ = stratified_sample_indices(
        pd.date_range("2016-01-01", periods=200, freq="B"),
        n_samples=32,
        per_regime=8,
        random_seed=42,
    )
    assert list(idx1) == list(idx2)
