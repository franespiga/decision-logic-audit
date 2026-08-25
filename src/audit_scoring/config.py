"""Configuration constants for Batch A audit scoring."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else default


DEFAULT_DIAGNOSTIC_PKL_ROOT = _env_path(
    "DECISION_LOGIC_AUDIT_DIAGNOSTIC_PKL_ROOT",
    REPO_ROOT / "results" / "diagnostics",
)
DEFAULT_METRICS_DIR = _env_path(
    "DECISION_LOGIC_AUDIT_METRICS_DIR",
    REPO_ROOT / "results" / "mendeley" / "dt_metrics",
)
DEFAULT_OUTPUT_DIR = DEFAULT_METRICS_DIR / "audit_screening"
DEFAULT_MODEL_ASSESSMENT = _env_path(
    "DECISION_LOGIC_AUDIT_MODEL_ASSESSMENT",
    REPO_ROOT / "results" / "model_assessment.pkl",
)
DEFAULT_FAMILY_MAPPING_REPORT = (
    DEFAULT_METRICS_DIR / "audit_inputs" / "family_mapping_report.csv"
)

VALIDATION_START = pd.Timestamp("2016-01-01")
VALIDATION_END = pd.Timestamp("2024-04-30")
HOLDOUT_START = pd.Timestamp("2024-05-01")
HOLDOUT_END = pd.Timestamp("2026-06-30")

FORBIDDEN_HOLDOUT_BASENAMES = (
    "holdout_metrics_family",
    "holdout_metrics_individual",
)

BENCHMARK_FAMILIES = frozenset({"EWP", "^GSPC"})

EXPECTED_N_FAMILIES = 18
PRIMARY_COST_BPS = 5
COST_LEVELS_BPS: Tuple[int, ...] = (0, 2, 5, 10)

DEFAULT_BOOTSTRAP_REPLICATES = 100
DEFAULT_RANDOM_SEED = 42
DEFAULT_TOP_K_FAMILIES = 5
MIN_ATTRIBUTION_OBSERVATIONS = 30
MIN_FEATURE_FAMILIES = 2
OCCLUSION_EFFECT_EPS = 1e-8
OCCLUSION_NONDEGENERACY_TOL = 1e-6
COVERAGE_RATIO_THRESHOLD = 0.80
DISPERSION_IQR_THRESHOLD = 40.0  # percentile-score IQR proxy on raw-metric scale flags

WEIGHT_MIN = 0.15

PROFILES: Dict[str, Dict[str, float]] = {
    "balanced": {
        "contribution": 1.0 / 3.0,
        "necessity": 1.0 / 3.0,
        "structure": 1.0 / 3.0,
    },
    "contribution_oriented": {
        "contribution": 0.50,
        "necessity": 0.25,
        "structure": 0.25,
    },
    "necessity_oriented": {
        "contribution": 0.25,
        "necessity": 0.50,
        "structure": 0.25,
    },
    "structure_oriented": {
        "contribution": 0.25,
        "necessity": 0.25,
        "structure": 0.50,
    },
}

HIGHER_IS_BETTER_METRICS = (
    "contribution_rank_stability",
    "contribution_sign_stability",
    "necessity_temporal_profile_stability",
    "structure_effective_rank_ratio",
    "structure_probe_margin",
)

LOWER_IS_BETTER_METRICS = (
    "necessity_temporal_instability",
)

CORE_CONTRIBUTION_METRICS = (
    "contribution_rank_stability",
    "contribution_sign_stability",
)

CORE_NECESSITY_METRICS = (
    "necessity_temporal_profile_stability",
    "necessity_temporal_instability",
)

CORE_STRUCTURE_METRICS = (
    "structure_effective_rank_ratio",
    "structure_probe_margin",
)

DECISION_MATRIX: Dict[Tuple[str, str], str] = {
    ("STRONG", "STRONG"): "ADVANCE",
    ("STRONG", "ADEQUATE"): "MONITOR",
    ("STRONG", "WEAK"): "INVESTIGATE",
    ("STRONG", "POOR"): "INVESTIGATE",
    ("ADEQUATE", "STRONG"): "MONITOR",
    ("ADEQUATE", "ADEQUATE"): "MONITOR",
    ("ADEQUATE", "WEAK"): "INVESTIGATE",
    ("ADEQUATE", "POOR"): "INVESTIGATE",
    ("WEAK", "STRONG"): "INVESTIGATE",
    ("WEAK", "ADEQUATE"): "INVESTIGATE",
    ("WEAK", "WEAK"): "INVESTIGATE",
    ("WEAK", "POOR"): "REJECT",
    ("POOR", "STRONG"): "INVESTIGATE",
    ("POOR", "ADEQUATE"): "INVESTIGATE",
    ("POOR", "WEAK"): "REJECT",
    ("POOR", "POOR"): "REJECT",
}

DECISION_ORDER = ("ADVANCE", "MONITOR", "INVESTIGATE", "REJECT")
BAND_ORDER = ("STRONG", "ADEQUATE", "WEAK", "POOR")

REGIME_BOUNDARIES: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]] = {
    "bull_market": (pd.Timestamp("2016-01-01"), pd.Timestamp("2020-02-15")),
    "crash_period": (pd.Timestamp("2020-02-16"), pd.Timestamp("2020-05-31")),
    "recovery_growth": (pd.Timestamp("2020-06-01"), pd.Timestamp("2021-12-31")),
    "sideways_uncertain": (pd.Timestamp("2022-01-01"), pd.Timestamp("2024-04-30")),
}


@dataclass
class AuditConfig:
    """Runtime configuration for the Batch A audit pipeline."""

    repository_root: Path = REPO_ROOT
    diagnostic_pkl_root: Path = DEFAULT_DIAGNOSTIC_PKL_ROOT
    metrics_dir: Path = DEFAULT_METRICS_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    model_assessment_pickle: Path = DEFAULT_MODEL_ASSESSMENT
    family_mapping_report: Path = DEFAULT_FAMILY_MAPPING_REPORT
    primary_cost_bps: int = PRIMARY_COST_BPS
    audit_profile: str = "balanced"
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES
    random_seed: int = DEFAULT_RANDOM_SEED
    top_k_families: int = DEFAULT_TOP_K_FAMILIES
    coverage_ratio_threshold: float = COVERAGE_RATIO_THRESHOLD
    min_attribution_observations: int = MIN_ATTRIBUTION_OBSERVATIONS
    min_feature_families: int = MIN_FEATURE_FAMILIES
    occlusion_effect_eps: float = OCCLUSION_EFFECT_EPS
    occlusion_nondegeneracy_tol: float = OCCLUSION_NONDEGENERACY_TOL
    validation_start: pd.Timestamp = VALIDATION_START
    validation_end: pd.Timestamp = VALIDATION_END
    prefer_complete_over_partial: bool = True
    load_model_assessment: bool = False
    force: bool = False
    inspect_only: bool = False
    model_id: str | None = None
    model_family: str | None = None

    def profiles(self) -> Dict[str, Dict[str, float]]:
        return {k: dict(v) for k, v in PROFILES.items()}

    def primary_weights(self) -> Dict[str, float]:
        if self.audit_profile not in PROFILES:
            raise ValueError(
                f"Unknown audit profile {self.audit_profile!r}. "
                f"Choose from {sorted(PROFILES)}"
            )
        weights = dict(PROFILES[self.audit_profile])
        validate_weights(weights)
        return weights

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, Path):
                payload[key] = str(value)
            elif isinstance(value, pd.Timestamp):
                payload[key] = value.isoformat()
        return payload


DEFAULT_CONFIG = AuditConfig()


def validate_weights(weights: Dict[str, float]) -> None:
    required = {"contribution", "necessity", "structure"}
    if set(weights) != required:
        raise ValueError(f"Weights must contain exactly {sorted(required)}")
    total = float(sum(weights.values()))
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"Weights must sum to 1; got {total}")
    for name, value in weights.items():
        if float(value) < WEIGHT_MIN - 1e-12:
            raise ValueError(f"Weight {name}={value} is below minimum {WEIGHT_MIN}")


def regime_label_for_date(ts: pd.Timestamp) -> str:
    for name, (start, end) in REGIME_BOUNDARIES.items():
        if start <= ts <= end:
            return name
    return "unclassified"
