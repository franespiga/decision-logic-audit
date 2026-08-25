"""Tests for Batch C3 input-permutation validity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from audit_scoring.permutation_config import ABOVE_MARKET_FAMILIES, PermutationConfig
from audit_scoring.permutation_operator import (
    build_regime_donor_map,
    generate_derangement,
    permute_all_state_families,
    permute_state_families,
    valid_permutation_mask,
)
from audit_scoring.permutation_validity import (
    combined_permutation_status,
    importance_effect_concordance,
)
from audit_scoring.validity_guards import SourceAccessLog, assert_permitted_analytical_load


def test_derangement_no_fixed_points():
    rng = np.random.default_rng(0)
    for n in range(2, 12):
        perm = generate_derangement(n, rng)
        assert len(perm) == n
        assert not np.any(perm == np.arange(n))


def test_small_regime_handling():
    regimes = ["a", "a", "b"]  # b has size 1
    donors, df = build_regime_donor_map(regimes, seed=101)
    assert donors[2] == -1
    assert (df.loc[df.regime == "b", "donor_local_index"] == -1).all()
    assert valid_permutation_mask(donors).sum() == 2


def test_whole_family_block_and_temporal_order():
    # states [N,T,F]
    states = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    family_ids = np.array(["A", "A", "B", "B"])
    donors = np.array([1, 0])
    out = permute_state_families(states, donors, family_ids, ["A"])
    # Family A features (0,1) of obs0 come from obs1; temporal order preserved
    assert torch.equal(out[0, :, :2], states[1, :, :2])
    assert torch.equal(out[0, :, 2:], states[0, :, 2:])  # B unchanged
    # Within-block temporal ordering from donor
    assert torch.equal(out[0, 0, :2], states[1, 0, :2])
    assert torch.equal(out[0, 1, :2], states[1, 1, :2])


def test_non_state_inputs_conceptually_unpermuted():
    # Operator only touches states; actions/rtg would remain caller-owned.
    states = torch.randn(4, 2, 6)
    donors = np.array([1, 0, 3, 2])
    family_ids = np.array(["X"] * 3 + ["Y"] * 3)
    out = permute_state_families(states, donors, family_ids, ["X"])
    assert out.shape == states.shape
    # Y block unchanged
    assert torch.equal(out[:, :, 3:], states[:, :, 3:])


def test_all_state_permutation():
    states = torch.randn(3, 2, 4)
    donors = np.array([1, 2, 0])
    out = permute_all_state_families(states, donors)
    assert torch.equal(out[0], states[1])
    assert torch.equal(out[1], states[2])
    assert torch.equal(out[2], states[0])


def test_deterministic_seeds():
    regimes = ["r1"] * 5 + ["r2"] * 5
    d1, _ = build_regime_donor_map(regimes, seed=101)
    d2, _ = build_regime_donor_map(regimes, seed=101)
    d3, _ = build_regime_donor_map(regimes, seed=202)
    assert np.array_equal(d1, d2)
    assert not np.array_equal(d1, d3)


def test_importance_effect_concordance_statuses():
    # Positive association -> SUPPORTED or MIXED
    fams = [f"F{i}" for i in range(8)]
    rows = []
    for i, f in enumerate(fams):
        rows.append(
            {
                "model_family": "m",
                "feature_family": f,
                "median_tv": float(i),
                "original_ig_abs": float(i),
                "seed": 101,
            }
        )
    df = pd.DataFrame(rows)
    out = importance_effect_concordance(df, model_family="m", bootstrap_replicates=200, random_seed=42)
    assert out["spearman"] > 0.9
    assert out["importance_effect_status"] in {"SUPPORTED", "MIXED"}


def test_combined_status_rules():
    assert combined_permutation_status(
        importance_effect_status="SUPPORTED", guided_permutation_status="SUPPORTED", identity_pass=True
    )[0] == "SUPPORTED"
    assert combined_permutation_status(
        importance_effect_status="UNSUPPORTED", guided_permutation_status="SUPPORTED", identity_pass=True
    )[0] == "UNSUPPORTED"
    assert combined_permutation_status(
        importance_effect_status="SUPPORTED", guided_permutation_status="SUPPORTED", identity_pass=False
    )[0] == "UNSUPPORTED"
    assert combined_permutation_status(
        importance_effect_status="MIXED", guided_permutation_status="SUPPORTED", identity_pass=True
    )[0] == "MIXED"


def test_above_market_frozen_subset():
    assert "max_quadratic_utility_TI_True" in ABOVE_MARKET_FAMILIES
    assert len(ABOVE_MARKET_FAMILIES) == 5


def test_holdout_source_prohibition(tmp_path: Path):
    log = SourceAccessLog()
    bad = tmp_path / "holdout_metrics_family.csv"
    bad.write_text("x\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        assert_permitted_analytical_load(bad, role="test", log=log)


def test_incomplete_family_config():
    cfg = PermutationConfig()
    assert cfg.incomplete_family == "no_techindi_full_TI_False"
    assert cfg.mandatory_family == "max_quadratic_utility_TI_True"


def test_topk_and_cardinality_reuse():
    from audit_scoring.baseline_validity import ranking_from_abs_profile, top_k_jaccard
    from audit_scoring.cardinality_controls import (
        family_raw_cardinality,
        guided_raw_count,
        sample_cardinality_matched_families,
    )

    s = pd.Series({"A": 3.0, "B": 2.0, "C": 1.0})
    r = ranking_from_abs_profile(s)
    assert top_k_jaccard(r, r, 2) == 1.0
    family_ids = np.array(["A"] * 4 + ["B"] * 4 + ["C"] * 2)
    card = family_raw_cardinality(family_ids)
    guided = ["A"]
    target = guided_raw_count(family_ids, guided)
    rng = np.random.default_rng(42)
    choice, raw, residual = sample_cardinality_matched_families(
        ["A", "B", "C"], card, guided_families=guided, n_families=1, target_raw_count=target, rng=rng
    )
    assert "A" not in choice
    assert residual == abs(raw - target)
