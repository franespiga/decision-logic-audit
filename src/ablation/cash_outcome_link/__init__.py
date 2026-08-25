"""
Diagnostic-to-allocation-to-outcome link for Reviewer 1 Comment 3 (Tier B).

Uses paper validation-period weight CSVs for the higher- and lower-Sharpe
Decision Transformer policies, filtered to the trading calendar.
"""

from .analysis import run_tier_b_analysis
from .regimes import PAPER_REGIMES, REGIME_ORDER, assign_paper_regimes

__all__ = [
    "PAPER_REGIMES",
    "REGIME_ORDER",
    "assign_paper_regimes",
    "run_tier_b_analysis",
]
