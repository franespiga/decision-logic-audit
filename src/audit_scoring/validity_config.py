"""Batch C diagnostic-validity configuration and operational thresholds.

Thresholds are study-specific operational rules for this manuscript revision.
They are not universal standards and do not certify deployment readiness.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd

from .config import REPO_ROOT, VALIDATION_END, VALIDATION_START
from .holdout_assessment import INCOMPLETE_EVIDENCE_FAMILY

EXPECTED_N_FAMILIES = 18
MIN_EVALUABLE_FAMILIES = 15

DEFAULT_DIAGNOSTIC_PKL_ROOT = Path(r"C:\git\tesis\results\ablation_ESWA_all_models")
DEFAULT_BATCH_A_DIR = REPO_ROOT / "results" / "mendeley" / "dt_metrics" / "audit_screening"
DEFAULT_BATCH_B_DIR = REPO_ROOT / "results" / "mendeley" / "dt_metrics" / "audit_holdout"
DEFAULT_BATCH_B_VERIFICATION_DIR = (
    REPO_ROOT / "results" / "mendeley" / "dt_metrics" / "audit_holdout_verification"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "mendeley" / "dt_metrics" / "audit_validity"
DEFAULT_FAMILY_MAPPING_REPORT = (
    REPO_ROOT / "results" / "mendeley" / "dt_metrics" / "audit_inputs" / "family_mapping_report.csv"
)
DEFAULT_MODEL_CODE_ROOT = Path(r"C:\git\DT-DIO-ETSII-UPM")
DEFAULT_MODEL_ASSESSMENT = Path(
    r"C:\git\DT-DIO-ETSII-UPM\src\portfolio_optimization\results\ESWA_wHoldout\model_assessment.pkl"
)

PRIMARY_REGIMES: Tuple[str, ...] = (
    "bull_market",
    "crash_period",
    "recovery_growth",
    "sideways_uncertain",
)

# Analytical loading of these basenames is forbidden in Batch C.
FORBIDDEN_ANALYTICAL_BASENAMES: Tuple[str, ...] = (
    "holdout_metrics_family",
    "holdout_metrics_individual",
    "validation_holdout_metrics_family",
    "validation_holdout_metrics_individual",
)

FORBIDDEN_BATCH_B_OUTCOME_STEMS: Tuple[str, ...] = (
    "audit_holdout_family_outcomes",
    "audit_holdout_score_concordance",
    "audit_holdout_case_classification",
    "audit_holdout_neighbour",
    "audit_holdout_master",
)

DEFAULT_RANDOMIZATION_SEEDS: Tuple[int, ...] = (101, 202, 303, 404, 505)

# Baseline concordance (primary operational rules)
BASELINE_SUPPORTED_MEDIAN_TAU = 0.50
BASELINE_SUPPORTED_MIN_JACCARD = 0.50
BASELINE_SUPPORTED_SIGN_AGREE = 0.70
BASELINE_MIXED_MEDIAN_TAU = 0.20
BASELINE_SENSITIVITY_TAUS: Tuple[float, ...] = (0.40, 0.50, 0.60)

# Randomization
RAND_SUPPORTED_FULL_SIM = 0.20
RAND_SUPPORTED_GAP = 0.30
RAND_MIXED_FULL_SIM = 0.40
RAND_MIXED_GAP = 0.15
ACTION_DISTANCE_TOL = 1e-6

# Faithfulness
FAITHFULNESS_EFFECT_SIZE = 0.20


@dataclass
class ValidityConfig:
    repository_root: Path = REPO_ROOT
    diagnostic_pkl_root: Path = DEFAULT_DIAGNOSTIC_PKL_ROOT
    batch_a_dir: Path = DEFAULT_BATCH_A_DIR
    batch_b_dir: Path = DEFAULT_BATCH_B_DIR
    batch_b_verification_dir: Path = DEFAULT_BATCH_B_VERIFICATION_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    family_mapping_report: Path = DEFAULT_FAMILY_MAPPING_REPORT
    model_assessment_pickle: Path = DEFAULT_MODEL_ASSESSMENT
    model_code_root: Path = DEFAULT_MODEL_CODE_ROOT
    validation_start: pd.Timestamp = VALIDATION_START
    validation_end: pd.Timestamp = VALIDATION_END
    primary_samples: int = 128
    randomization_samples: int = 64
    background_samples: int = 16
    primary_per_regime: int = 32
    randomization_per_regime: int = 16
    background_per_regime: int = 4
    random_seed: int = 42
    randomization_seeds: Tuple[int, ...] = DEFAULT_RANDOMIZATION_SEEDS
    random_control_repeats: int = 50
    bootstrap_replicates: int = 10000
    ig_steps: int = 16
    attribution_batch_size: int = 8
    inference_batch_size: int = 32
    top_k_families: int = 5
    max_deletion_depth: int = 5
    device: str = "auto"
    force: bool = False
    inspect_only: bool = False
    resume: bool = True
    skip_figures: bool = False
    families: Tuple[str, ...] = ()
    incomplete_family: str = INCOMPLETE_EVIDENCE_FAMILY

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, Path):
                payload[key] = str(value)
            elif isinstance(value, pd.Timestamp):
                payload[key] = value.isoformat()
            elif isinstance(value, tuple):
                payload[key] = list(value)
        return payload


DEFAULT_VALIDITY_CONFIG = ValidityConfig()


def resolve_device(preference: str = "auto"):
    import torch

    pref = (preference or "auto").lower()
    if pref == "cpu":
        return torch.device("cpu")
    if pref.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(pref if ":" in pref else "cuda")
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
