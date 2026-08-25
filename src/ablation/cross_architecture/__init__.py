"""
Cross-architecture diagnostic comparison between Decision Transformers and
conventional deep RL policies.

The package applies one common set of diagnostics -- contribution, necessity and
structure -- to frozen policies of different architectures. All cross-model
quantities are dimensionless (normalised shares, ranks, stabilities,
concentration indices); raw attribution and activation magnitudes are never
compared across architectures.

Modules
-------
discovery   model discovery, manifest resolution and inclusion rules
targets     family-specific policy adapters and decision targets
sampling    regimes, common evaluation dates and stratified sampling
contribution feature attribution, rank stability, baseline concordance
necessity   attribution-guided deletion and temporal occlusion
structure   latent neighbourhood stability and layer-influence concentration
outcomes    portfolio weight reconstruction and performance metrics
scoring     oriented dimension scores C, N, S and the aggregate D
reporting   tables, LaTeX drop-ins, figures and supplementary output
"""

from __future__ import annotations

__all__ = [
    "discovery",
    "targets",
    "sampling",
    "contribution",
    "necessity",
    "structure",
    "outcomes",
    "scoring",
    "reporting",
]

SCHEMA_VERSION = "1.0.0"
