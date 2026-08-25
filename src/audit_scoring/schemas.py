"""Lightweight schema helpers and status enums for Batch A."""

from __future__ import annotations

from typing import Final

MAPPING_STATUSES: Final = ("resolved", "unresolved", "ambiguous", "duplicate")
VALIDITY_STATUSES: Final = ("passed", "failed", "incomplete", "not_evaluated")
SCORE_STATUSES: Final = (
    "scored",
    "not_scored_missing_core_dimension",
    "excluded_unresolved_mapping",
    "excluded_temporally_invalid",
)
DIAGNOSTIC_BANDS: Final = ("STRONG", "ADEQUATE", "WEAK", "POOR")
PERFORMANCE_BANDS: Final = ("STRONG", "ADEQUATE", "WEAK", "POOR")
DECISIONS: Final = ("ADVANCE", "MONITOR", "INVESTIGATE", "REJECT")

CONTRIBUTION_KEYS = (
    "postprocessed_attributions",
    "postprocessed_feature_names",
    "feature_attributions",
    "ablation_attributions",
)
OCCLUSION_KEYS = ("occlusion_results",)
ROLLING_KEYS = ("rolling_stability",)
LAYER_KEYS = ("layer_representations",)
LATENT_KEYS = ("latent_representations",)
PROBE_KEYS = ("linear_probe_results",)
SEPARABILITY_KEYS = ("separability_metrics",)
ATTENTION_KEYS = ("attention_activations", "head_specialization")
