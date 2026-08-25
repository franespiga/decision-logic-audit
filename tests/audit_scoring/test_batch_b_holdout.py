"""Unit tests for Batch B holdout assessment (synthetic data)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.audit_scoring.case_analysis import classify_cases
from src.audit_scoring.concordance import pairwise_concordance
from src.audit_scoring.enrichment import category_summary, top_bottom_coverage
from src.audit_scoring.holdout_assessment import (
    INCOMPLETE_EVIDENCE_FAMILY,
    assert_allowed_holdout_path,
    parse_family_design_factors,
    sha256_file,
    verify_batch_a_freeze,
)
from src.audit_scoring.neighbour_analysis import (
    build_neighbour_candidates,
    hamming_distance,
    select_neighbour_pairs,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_combined_file_prohibited(tmp_path: Path):
    p = tmp_path / "validation_holdout_metrics_family.csv"
    p.write_text("x\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        assert_allowed_holdout_path(p)


def test_holdout_file_allowed(tmp_path: Path):
    p = tmp_path / "holdout_metrics_family.csv"
    p.write_text("x\n", encoding="utf-8")
    assert assert_allowed_holdout_path(p) == p


def test_sha256_and_changed_detection(tmp_path: Path):
    f = tmp_path / "a.csv"
    f.write_text("abc", encoding="utf-8")
    h1 = sha256_file(f)
    f.write_text("abcd", encoding="utf-8")
    h2 = sha256_file(f)
    assert h1 != h2


def test_freeze_verification_detects_hash_change(tmp_path: Path):
    batch_a = tmp_path / "audit_screening"
    batch_a.mkdir()
    out = batch_a / "audit_validation_decisions_primary_5bps.csv"
    families = [f"F{i}_TI_True" for i in range(17)] + [INCOMPLETE_EVIDENCE_FAMILY]
    decisions = ["MONITOR"] * 4 + ["INVESTIGATE"] * 7 + ["REJECT"] * 6 + ["REJECT"]
    # Ensure counts 0/4/7/7
    decisions = (
        ["MONITOR"] * 4 + ["INVESTIGATE"] * 7 + ["REJECT"] * 7
    )
    assert len(families) == 18 and len(decisions) == 18
    df = pd.DataFrame(
        {
            "model_family": families,
            "operational_recommendation": decisions,
            "diagnostic_band": ["ADEQUATE"] * 17 + ["POOR"],
            "diagnostic_score": [50.0] * 17 + [np.nan],
            "decision_reason": ["matrix|provisional_pending_additional_validity_checks"] * 18,
        }
    )
    df.to_csv(out, index=False)
    # dummy hashed files
    inp = tmp_path / "family_cost_sensitivity.csv"
    inp.write_text("ok", encoding="utf-8")
    digest = sha256_file(inp)
    manifest = {
        "git_commit": "abc",
        "primary_cost_level_bps": 5,
        "audit_profile": "balanced",
        "n_families": 18,
        "input_files": [{"path": str(inp), "sha256": digest}],
        "output_files": [{"path": str(out), "sha256": sha256_file(out)}],
        "provisional_decisions": 18,
    }
    (batch_a / "BATCH_A_FROZEN_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    ok = verify_batch_a_freeze(batch_a, repo_root=tmp_path)
    assert ok.ok
    # change input
    inp.write_text("changed", encoding="utf-8")
    bad = verify_batch_a_freeze(batch_a, repo_root=tmp_path)
    assert not bad.ok
    assert bad.changed_files


def test_variant_family_uniqueness_and_counts():
    rows = []
    for fam_i in range(18):
        fam = f"FAM{fam_i}_TI_True"
        for v in range(96):
            rows.append({"model_id": f"{fam}_v{v}", "model_family": fam, "cost_bps": 5, "sharpe": 0.1})
    df = pd.DataFrame(rows)
    assert df.model_id.nunique() == 1728
    assert (df.groupby("model_family").model_id.nunique() == 96).all()
    assert (df.groupby("model_id").model_family.nunique() == 1).all()


def test_complete_case_correlation_excludes_incomplete():
    df = pd.DataFrame(
        {
            "model_family": [f"F{i}" for i in range(5)] + [INCOMPLETE_EVIDENCE_FAMILY],
            "diagnostic_score": [10, 20, 30, 40, 50, np.nan],
            "holdout_median_sharpe": [0.1, 0.2, 0.3, 0.4, 0.5, 0.9],
        }
    )
    row = pairwise_concordance(
        df,
        "diagnostic_score",
        "holdout_median_sharpe",
        method="spearman",
        n_boot=50,
        seed=42,
        exclude_families=[INCOMPLETE_EVIDENCE_FAMILY],
    )
    assert row["n_families"] == 5
    assert INCOMPLETE_EVIDENCE_FAMILY in str(row["excluded_families"])


def test_empty_category_not_fabricated():
    df = pd.DataFrame(
        {
            "model_family": ["A", "B", "C"],
            "frozen_screening_decision": ["MONITOR", "INVESTIGATE", "REJECT"],
            "holdout_median_sharpe": [1.0, 0.5, 0.2],
            "holdout_median_sortino": [1.0, 0.5, 0.2],
            "holdout_median_calmar": [1.0, 0.5, 0.2],
            "holdout_median_annualized_return": [0.1, 0.05, 0.01],
            "holdout_median_max_drawdown": [-0.1, -0.2, -0.3],
            "holdout_median_annualized_turnover": [1.0, 2.0, 3.0],
        }
    )
    cat = category_summary(df, n_boot=20, seed=1)
    adv = cat[cat.frozen_screening_decision == "ADVANCE"].iloc[0]
    assert adv["n_families"] == 0
    assert pd.isna(adv["median_holdout_sharpe"])
    assert "not_estimable" in str(adv["notes"])


def test_ci_suppressed_for_small_groups():
    df = pd.DataFrame(
        {
            "model_family": ["A", "B"],
            "frozen_screening_decision": ["MONITOR", "MONITOR"],
            "holdout_median_sharpe": [1.0, 0.8],
            "holdout_median_sortino": [1.0, 0.8],
            "holdout_median_calmar": [1.0, 0.8],
            "holdout_median_annualized_return": [0.1, 0.08],
            "holdout_median_max_drawdown": [-0.1, -0.12],
            "holdout_median_annualized_turnover": [1.0, 1.1],
        }
    )
    cat = category_summary(df, n_boot=20, seed=1)
    mon = cat[cat.frozen_screening_decision == "MONITOR"].iloc[0]
    assert pd.isna(mon["ci_low_median_holdout_sharpe"])


def test_topk_coverage_and_cases():
    rng = np.random.default_rng(0)
    fams = [f"F{i}_TI_True" for i in range(17)] + [INCOMPLETE_EVIDENCE_FAMILY]
    df = pd.DataFrame(
        {
            "model_family": fams,
            "frozen_screening_decision": ["MONITOR"] * 4 + ["INVESTIGATE"] * 7 + ["REJECT"] * 7,
            "diagnostic_band": ["ADEQUATE"] * 17 + ["POOR"],
            "validation_performance_band": ["STRONG"] * 9 + ["WEAK"] * 9,
            "rejection_basis": [""] * 17 + ["incomplete_diagnostic_evidence"],
            "holdout_median_sharpe": np.linspace(1.0, 0.0, 18),
            "holdout_median_sortino": np.linspace(1.0, 0.0, 18),
            "holdout_median_calmar": np.linspace(1.0, 0.0, 18),
            "holdout_median_annualized_return": np.linspace(0.2, -0.1, 18),
            "holdout_median_max_drawdown": -np.linspace(0.05, 0.4, 18),
            "holdout_median_annualized_turnover": np.linspace(1, 10, 18),
            "diagnostic_score": list(np.linspace(90, 10, 17)) + [np.nan],
            "contribution_score": list(np.linspace(90, 10, 17)) + [np.nan],
            "necessity_score": list(np.linspace(90, 10, 17)) + [np.nan],
            "structure_score": list(np.linspace(90, 10, 17)) + [np.nan],
            "validation_median_sharpe": np.linspace(1.0, 0.2, 18),
            "validation_median_calmar": np.linspace(1.0, 0.2, 18),
            "diagnostic_score_available": [True] * 17 + [False],
            "evidence_complete_for_primary_score": [True] * 17 + [False],
        }
    )
    cov = top_bottom_coverage(df)
    assert not cov.empty
    cases = classify_cases(df)
    assert INCOMPLETE_EVIDENCE_FAMILY in set(
        cases.loc[cases.case_label == "evidence_incomplete_exclusion", "model_family"]
    )


def test_neighbour_selection_excludes_incomplete_and_is_deterministic():
    fams = [
        "DT_16_1_TI_True",
        "DT_16_1_TI_False",
        "DT_16_10_TI_True",
        "DT_28_1_TI_True",
        "efficient_risk_TI_True",
        "efficient_risk_TI_False",
        "max_sharpe_TI_True",
        "max_sharpe_TI_False",
        INCOMPLETE_EVIDENCE_FAMILY,
    ]
    n = len(fams)
    df = pd.DataFrame(
        {
            "model_family": fams,
            "diagnostic_score": list(np.linspace(80, 20, n - 1)) + [np.nan],
            "contribution_score": list(np.linspace(80, 20, n - 1)) + [np.nan],
            "necessity_score": list(np.linspace(70, 30, n - 1)) + [np.nan],
            "structure_score": list(np.linspace(60, 40, n - 1)) + [np.nan],
            "validation_median_sharpe": list(np.linspace(0.8, 0.5, n - 1)) + [0.7],
            "validation_median_calmar": list(np.linspace(0.7, 0.4, n - 1)) + [0.6],
            "holdout_median_sharpe": list(np.linspace(0.9, 0.2, n - 1)) + [0.5],
            "holdout_median_calmar": list(np.linspace(0.8, 0.1, n - 1)) + [0.4],
            "holdout_median_max_drawdown": [-0.1] * n,
            "holdout_median_annualized_turnover": [1.0] * n,
            "diagnostic_score_available": [True] * (n - 1) + [False],
            "evidence_complete_for_primary_score": [True] * (n - 1) + [False],
            "frozen_screening_decision": ["MONITOR"] * n,
        }
    )
    cand = build_neighbour_candidates(df)
    assert INCOMPLETE_EVIDENCE_FAMILY not in set(cand.family_a) | set(cand.family_b)
    a = select_neighbour_pairs(cand, n_pairs=3)
    b = select_neighbour_pairs(cand, n_pairs=3)
    assert list(a.family_a) == list(b.family_a)
    assert list(a.family_b) == list(b.family_b)
    # Hamming helper
    d, shared, differing = hamming_distance("DT_16_1_TI_True", "DT_16_1_TI_False")
    assert d == 1


def test_design_factor_parse():
    f = parse_family_design_factors("DT_28_10_TI_False")
    assert f["lookback"] == "28"
    assert f["rebalancing_frequency"] == "10"
    assert f["technical_indicators"] == "False"


def test_no_batch_a_recalibration_in_cli_defaults():
    from src.audit_scoring.holdout_cli import build_parser

    p = build_parser()
    args = p.parse_args([])
    assert args.primary_cost_bps == 5
