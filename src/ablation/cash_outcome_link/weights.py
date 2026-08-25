"""Load paper weight CSVs and restrict them to the trading calendar."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

INSTRUMENTS = ["^RUT", "^GDAXI", "^N225", "^IXIC", "^FTSE", "^GSPC", "CASH"]


def load_weight_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index)
    missing = [c for c in INSTRUMENTS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return df.loc[:, INSTRUMENTS].sort_index().astype(float)


def filter_to_trading_calendar(
    weights: pd.DataFrame,
    trading_index: pd.DatetimeIndex,
    start: str = "2016-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Keep only dates that appear in the market trading calendar.

    This removes weekend/holiday rows introduced by calendar forward-fill.
    """
    trading = pd.DatetimeIndex(trading_index).sort_values()
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) if end else min(weights.index.max(), trading.max())
    window = trading[(trading >= start_ts) & (trading <= end_ts)]
    common = weights.index.intersection(window)
    if len(common) == 0:
        raise ValueError("no overlap between weights and trading calendar")
    out = weights.loc[common].sort_index()
    # Drop exact duplicate index rows if any
    out = out[~out.index.duplicated(keep="last")]
    return out


def load_filtered_pair(
    best_path: str | Path,
    worst_path: str | Path,
    trading_index: pd.DatetimeIndex,
    start: str = "2016-01-01",
    end: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    best = filter_to_trading_calendar(load_weight_csv(best_path), trading_index, start, end)
    worst = filter_to_trading_calendar(load_weight_csv(worst_path), trading_index, start, end)
    common = best.index.intersection(worst.index)
    return {
        "DT_max_quadratic_utility": best.loc[common],
        "DT_16_10": worst.loc[common],
        "common_dates": common.sort_values(),
    }


def simplex_report(weights: pd.DataFrame, tol: float = 1e-4) -> Dict[str, float]:
    row_sums = weights.sum(axis=1)
    return {
        "n_rows": int(len(weights)),
        "min_weight": float(weights.values.min()),
        "max_weight": float(weights.values.max()),
        "max_abs_sum_deviation": float(np.abs(row_sums - 1.0).max()),
        "violates_long_only": bool(weights.values.min() < -1e-8),
        "violates_full_investment": bool(np.abs(row_sums - 1.0).max() > tol),
        "n_weekend_like_removed_hint": None,
    }
