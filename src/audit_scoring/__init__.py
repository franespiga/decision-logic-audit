"""Validation-calibrated contribution–necessity–structure audit scoring (Batch A)
plus holdout assessment (Batch B) and diagnostic-validity overlay (Batch C).

Diagnostic scores and validity overlays do not predict future performance and do
not certify deployment readiness.
"""

from __future__ import annotations

from .config import AuditConfig, DEFAULT_CONFIG
from .pipeline import run_batch_a, inspect_inventory
from .validity_config import ValidityConfig, DEFAULT_VALIDITY_CONFIG
from .validity_pipeline import run_batch_c, run_inspect_only

__all__ = [
    "AuditConfig",
    "DEFAULT_CONFIG",
    "ValidityConfig",
    "DEFAULT_VALIDITY_CONFIG",
    "run_batch_a",
    "inspect_inventory",
    "run_batch_c",
    "run_inspect_only",
]

__version__ = "0.2.0"
