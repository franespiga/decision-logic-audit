"""Batch C3 input-permutation configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from .config import REPO_ROOT, VALIDATION_END, VALIDATION_START
from .holdout_assessment import INCOMPLETE_EVIDENCE_FAMILY

DEFAULT_PERMUTATION_SEEDS: Tuple[int, ...] = (101, 202, 303, 404, 505)

ABOVE_MARKET_FAMILIES: Tuple[str, ...] = (
    "max_quadratic_utility_TI_True",
    "efficient_risk_TI_False",
    "efficient_risk_TI_True",
    "techindi_full_TI_True",
    "efficient_return_TI_True",
)

MANDATORY_FAMILY = "max_quadratic_utility_TI_True"

# Importance-effect concordance thresholds
IE_SUPPORTED_SPEARMAN = 0.30
IE_SUPPORTED_TOPK = 0.50
IE_SENSITIVITY = (0.20, 0.30, 0.40)

# Guided permutation faithfulness
PERM_EFFECT_SIZE = 0.20
IDENTITY_ACTION_TOL = 1e-6


@dataclass
class PermutationConfig:
    repository_root: Path = REPO_ROOT
    diagnostic_pkl_root: Path = Path(r"C:\git\tesis\results\ablation_ESWA_all_models")
    batch_a_dir: Path = REPO_ROOT / "results" / "mendeley" / "dt_metrics" / "audit_screening"
    batch_b_dir: Path = REPO_ROOT / "results" / "mendeley" / "dt_metrics" / "audit_holdout"
    batch_c_dir: Path = REPO_ROOT / "results" / "mendeley" / "dt_metrics" / "audit_validity"
    batch_c_verification_dir: Path = (
        REPO_ROOT / "results" / "mendeley" / "dt_metrics" / "audit_validity_verification"
    )
    batch_c2_dir: Path = REPO_ROOT / "results" / "mendeley" / "dt_metrics" / "audit_validity_targeted"
    batch_c2_verification_dir: Path = (
        REPO_ROOT / "results" / "mendeley" / "dt_metrics" / "audit_validity_targeted_verification"
    )
    family_mapping_report: Path = (
        REPO_ROOT / "results" / "mendeley" / "dt_metrics" / "audit_inputs" / "family_mapping_report.csv"
    )
    model_code_root: Path = Path(r"C:\git\DT-DIO-ETSII-UPM")
    output_dir: Path = REPO_ROOT / "results" / "mendeley" / "dt_metrics" / "audit_validity_permutation"
    validation_start: pd.Timestamp = VALIDATION_START
    validation_end: pd.Timestamp = VALIDATION_END
    primary_samples: int = 128
    permutation_seeds: Tuple[int, ...] = DEFAULT_PERMUTATION_SEEDS
    random_control_repeats: int = 100
    max_depth: int = 5
    bootstrap_replicates: int = 10000
    random_seed: int = 42
    ig_steps: int = 16
    attribution_batch_size: int = 8
    device: str = "auto"
    incomplete_family: str = INCOMPLETE_EVIDENCE_FAMILY
    mandatory_family: str = MANDATORY_FAMILY
    above_market_families: Tuple[str, ...] = ABOVE_MARKET_FAMILIES
    force: bool = False
    resume: bool = True
    inspect_only: bool = False
    skip_figures: bool = False
    skip_attribution_recompute: bool = False
    families: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, Path):
                d[k] = str(v)
            elif isinstance(v, pd.Timestamp):
                d[k] = v.isoformat()
            elif isinstance(v, tuple):
                d[k] = list(v)
        return d
