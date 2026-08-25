"""
Portfolio outcomes reconstructed from the frozen policies.

Allocations are aligned to the trading calendar of the frozen market data: the
weight index is the set of trading dates on which the policy was evaluated. No
calendar-day reindexing and no forward-filling of weekends or holidays is
performed, so no artificial observations enter the return series.

A weight decided on trading date t is held over the following trading interval,
so the portfolio return attributed to t+1 is ``w_t . r_{t+1}``. This convention
contains no look-ahead.

Net performance uses the project's existing assumption, ``transaction_costs =
0.002`` (20 basis points, documented in ``src/README.md``), charged on the
one-way traded fraction of the portfolio.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .discovery import CANONICAL_INSTRUMENTS
from .sampling import REGIME_ORDER, assign_regimes

PERIODS_PER_YEAR = 252
TRANSACTION_COST = 0.002


def hold_returns(ohlc: pd.DataFrame, instruments: Sequence[str]) -> pd.DataFrame:
    """Close-to-close returns per instrument; CASH earns zero."""
    cols = [(i, "Close") for i in instruments if (i, "Close") in ohlc.columns]
    prices = ohlc.loc[:, cols].copy()
    prices.columns = [c[0] for c in cols]
    rets = prices.ffill().pct_change(fill_method=None)
    for inst in instruments:
        if inst not in rets.columns:
            rets[inst] = 0.0
    rets["CASH"] = 0.0
    return rets.loc[:, list(instruments)]


def portfolio_returns(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    cost: float = TRANSACTION_COST,
) -> pd.DataFrame:
    """
    Gross and net portfolio returns on the trading calendar.

    ``weights`` is indexed by decision date; the realised return is booked on the
    next available trading date of ``returns``.
    """
    w = weights.reindex(columns=list(returns.columns)).fillna(0.0)
    common = returns.index.intersection(w.index)
    w = w.loc[common].sort_index()
    r = returns.loc[common].sort_index()

    # w_t applied to r_{t+1}
    aligned = w.shift(1)
    gross = (aligned * r).sum(axis=1)
    traded = 0.5 * (w - w.shift(1)).abs().sum(axis=1)
    net = gross - cost * traded.shift(0).reindex(gross.index).fillna(0.0)
    out = pd.DataFrame(
        {"gross_return": gross, "net_return": net, "traded_fraction": traded}
    ).iloc[1:]
    return out.dropna(subset=["gross_return"])


def _max_drawdown(returns: pd.Series) -> float:
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    peak = wealth.cummax()
    return float(((wealth - peak) / peak).min())


def performance_metrics(returns: pd.Series, label: str = "") -> Dict[str, float]:
    r = returns.dropna().astype(float)
    n = len(r)
    if n < 2:
        return {k: float("nan") for k in
                ("cumulative_return", "annualised_return", "annualised_volatility",
                 "sharpe_ratio", "sortino_ratio", "max_drawdown", "n_observations")}
    cumulative = float((1.0 + r).prod() - 1.0)
    ann_ret = float((1.0 + cumulative) ** (PERIODS_PER_YEAR / n) - 1.0)
    ann_vol = float(r.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR))
    downside = r[r < 0]
    down_vol = float(downside.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR)) if len(downside) > 1 else float("nan")
    return {
        "cumulative_return": cumulative,
        "annualised_return": ann_ret,
        "annualised_volatility": ann_vol,
        "sharpe_ratio": float(ann_ret / ann_vol) if ann_vol > 0 else float("nan"),
        "sortino_ratio": float(ann_ret / down_vol) if down_vol and down_vol > 0 else float("nan"),
        "max_drawdown": _max_drawdown(r),
        "n_observations": int(n),
    }


def turnover_metrics(weights: pd.DataFrame) -> Dict[str, float]:
    traded = 0.5 * (weights - weights.shift(1)).abs().sum(axis=1)
    traded = traded.iloc[1:]
    return {
        "mean_daily_turnover": float(traded.mean()),
        "annualised_turnover": float(traded.mean() * PERIODS_PER_YEAR),
        "n_rebalances": int((traded > 1e-9).sum()),
    }


def cash_allocation(weights: pd.DataFrame) -> Tuple[float, pd.DataFrame]:
    cash = weights["CASH"] if "CASH" in weights.columns else pd.Series(0.0, index=weights.index)
    labels = assign_regimes(weights.index)
    frame = (
        pd.DataFrame({"regime": labels, "cash_weight": cash.values}, index=weights.index)
        .groupby("regime")["cash_weight"]
        .agg(["mean", "std", "count"])
        .reindex(list(REGIME_ORDER))
        .reset_index()
    )
    return float(cash.mean()), frame


def evaluate_outcomes(
    model_id: str,
    weights: pd.DataFrame,
    ohlc: pd.DataFrame,
    provenance: str = "reconstructed",
    cost: float = TRANSACTION_COST,
) -> Dict[str, object]:
    """Full outcome bundle for one allocation series."""
    rets = hold_returns(ohlc, list(CANONICAL_INSTRUMENTS))
    pnl = portfolio_returns(weights, rets, cost=cost)
    gross = performance_metrics(pnl["gross_return"])
    net = performance_metrics(pnl["net_return"])
    turn = turnover_metrics(weights)
    mean_cash, cash_by_regime = cash_allocation(weights)

    metrics = {f"gross_{k}": v for k, v in gross.items()}
    metrics.update({f"net_{k}": v for k, v in net.items()})
    metrics.update(turn)
    metrics["mean_cash_allocation"] = mean_cash
    metrics["unique_allocations"] = int(len(np.unique(np.round(weights.values, 8), axis=0)))
    metrics["allocation_variability"] = float(weights.std(axis=0).mean())
    metrics["weights_source"] = provenance
    metrics["performance_source"] = provenance
    metrics["transaction_cost"] = cost
    return {
        "model_id": model_id,
        "returns": pnl,
        "metrics": metrics,
        "cash_by_regime": cash_by_regime,
    }


def check_simplex(weights: pd.DataFrame, tol: float = 1e-6) -> Dict[str, float]:
    """Verify the long-only, fully invested simplex constraints."""
    row_sums = weights.sum(axis=1)
    return {
        "min_weight": float(weights.values.min()),
        "max_abs_sum_deviation": float(np.abs(row_sums - 1.0).max()),
        "violates_long_only": bool(weights.values.min() < -tol),
        "violates_full_investment": bool(np.abs(row_sums - 1.0).max() > 1e-4),
    }


def load_legacy_weights(path: Optional[str], dates: pd.DatetimeIndex) -> Optional[pd.DataFrame]:
    """Load a previously generated weight file, restricted to the evaluation dates."""
    if not path or not Path(path).exists():
        return None
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index)
    missing = [c for c in CANONICAL_INSTRUMENTS if c not in df.columns]
    if missing:
        return None
    df = df.loc[:, list(CANONICAL_INSTRUMENTS)]
    common = df.index.intersection(dates)
    if len(common) < 0.5 * len(dates):
        return None
    return df.loc[common]


def benchmark_outcomes(ohlc: pd.DataFrame, dates: pd.DatetimeIndex) -> Dict[str, object]:
    """Equal-weight buy-and-hold over the six equity indices, for context only."""
    risky = [i for i in CANONICAL_INSTRUMENTS if i != "CASH"]
    w = pd.DataFrame(1.0 / len(risky), index=dates, columns=risky)
    w["CASH"] = 0.0
    return evaluate_outcomes("EqualWeight_benchmark", w, ohlc, provenance="benchmark")
