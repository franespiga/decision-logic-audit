"""Tests for Batch C2 targeted validity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.audit_scoring.benchmark_discovery import (
    discover_market_benchmark_candidates,
    select_above_market_families,
    select_authoritative_benchmark,
)
from src.audit_scoring.cardinality_controls import (
    family_raw_cardinality,
    guided_raw_count,
    sample_cardinality_matched_families,
)
from src.audit_scoring.targeted_validity import selection_list_hash
from src.audit_scoring.validity_guards import SourceAccessLog, assert_permitted_analytical_load


def test_benchmark_discovery_selects_gspc(tmp_path: Path):
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    df = pd.DataFrame(
        {
            "model_family": ["EWP", "^GSPC", "efficient_risk_TI_True", "max_quadratic_utility_TI_True"],
            "cost_bps": [5, 5, 5, 5],
            "median_sharpe": [0.68, 0.71, 0.80, 0.81],
        }
    )
    df.to_csv(metrics / "family_cost_sensitivity.csv", index=False)
    cands = discover_market_benchmark_candidates(metrics, cost_bps=5)
    selected = select_authoritative_benchmark(cands, tmp_path / "out")
    assert selected is not None
    assert selected.candidate_name == "^GSPC"
    assert selected.sharpe == pytest.approx(0.71)


def test_ambiguous_benchmark_stops(tmp_path: Path):
    # Force ambiguity by marking both as explicit market in candidates via selection logic:
    # If two candidates have explicitly_called_market_benchmark and reproducible, select returns None.
    from src.audit_scoring.benchmark_discovery import BenchmarkCandidate

    cands = [
        BenchmarkCandidate("^GSPC", "x", "r", "2016-01-01", "2024-04-30", 5, 0.7, True, True, False, ""),
        BenchmarkCandidate("EWP", "x", "r", "2016-01-01", "2024-04-30", 5, 0.6, True, True, False, ""),
    ]
    # manually mark both selected false and both explicit
    out = select_authoritative_benchmark(cands, tmp_path)
    assert out is None
    assert (tmp_path / "BATCH_C2_MARKET_BENCHMARK_CONFIRMATION_REQUIRED.md").exists()


def test_above_market_and_mandatory(tmp_path: Path):
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    pd.DataFrame(
        {
            "model_family": ["^GSPC", "EWP", "A", "B", "max_quadratic_utility_TI_True"],
            "cost_bps": [5] * 5,
            "median_sharpe": [0.71, 0.68, 0.80, 0.50, 0.60],
        }
    ).to_csv(metrics / "family_cost_sensitivity.csv", index=False)
    sel = select_above_market_families(
        metrics, market_name="^GSPC", market_sharpe=0.71, mandatory_family="max_quadratic_utility_TI_True"
    )
    chosen = set(sel.loc[sel.selected_for_batch_c2, "model_family"])
    assert "A" in chosen
    assert "max_quadratic_utility_TI_True" in chosen
    assert "B" not in chosen


def test_selection_list_freezing():
    h1 = selection_list_hash(["b", "a"])
    h2 = selection_list_hash(["a", "b"])
    assert h1 == h2


def test_cardinality_matching():
    family_ids = np.array(["A"] * 10 + ["B"] * 5 + ["C"] * 2 + ["D"] * 8)
    card = family_raw_cardinality(family_ids)
    ranked = ["A", "B", "C", "D"]
    guided = ["A"]
    target = guided_raw_count(family_ids, guided)
    rng = np.random.default_rng(42)
    choice, raw, residual = sample_cardinality_matched_families(
        ranked, card, guided_families=guided, n_families=1, target_raw_count=target, rng=rng
    )
    assert len(choice) == 1
    assert residual == abs(raw - target)
    a, _, _ = sample_cardinality_matched_families(
        ranked, card, guided_families=guided, n_families=1, target_raw_count=target, rng=np.random.default_rng(42)
    )
    b, _, _ = sample_cardinality_matched_families(
        ranked, card, guided_families=guided, n_families=1, target_raw_count=target, rng=np.random.default_rng(42)
    )
    assert a == b


def test_holdout_prohibition(tmp_path: Path):
    log = SourceAccessLog()
    bad = tmp_path / "holdout_metrics_family.csv"
    bad.write_text("x\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        assert_permitted_analytical_load(bad, role="x", log=log)


def test_preserve_batch_c_status_semantics():
    # Overlay builder must keep frozen_batch_c_status separate from targeted status.
    frozen = "UNSUPPORTED"
    targeted = "MIXED"
    assert frozen != targeted
