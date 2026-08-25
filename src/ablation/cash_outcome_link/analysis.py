"""Core Tier-B computations: regime CASH, contrasts, outcomes, family backdrop."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ablation.cross_architecture.data import load_market_data
from ablation.cross_architecture.discovery import CANONICAL_INSTRUMENTS
from ablation.cross_architecture.outcomes import (
    TRANSACTION_COST,
    check_simplex,
    hold_returns,
    performance_metrics,
    portfolio_returns,
    turnover_metrics,
)

from .bootstrap import (
    block_bootstrap_ci,
    default_block_length,
    paired_block_bootstrap_diff_ci,
)
from .regimes import (
    PAPER_REGIMES,
    REGIME_ORDER,
    assign_paper_regimes,
    count_regime_episodes,
)
from .weights import filter_to_trading_calendar, load_filtered_pair, load_weight_csv

ROOT = Path(__file__).resolve().parents[3]

MODEL_LABELS = {
    "DT_max_quadratic_utility": "Higher-Sharpe DT (max quadratic utility)",
    "DT_16_10": "Lower-Sharpe DT (DT_16_10)",
}


def _cash_series(weights: pd.DataFrame) -> pd.Series:
    return weights["CASH"].astype(float)


def regime_cash_table(
    model_id: str,
    weights: pd.DataFrame,
    outcome_metrics: Dict[str, float],
    *,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    cash = _cash_series(weights)
    labels = assign_paper_regimes(weights.index)
    rows = []
    for regime in REGIME_ORDER:
        mask = labels == regime
        vals = cash.values[mask]
        n = int(mask.sum())
        n_ep = count_regime_episodes(labels, regime)
        if n == 0:
            mean = med = iqr = lo = hi = float("nan")
            L = 1
        else:
            mean, lo, hi, L, _ = block_bootstrap_ci(
                vals,
                n_bootstrap=n_bootstrap,
                seed=seed + 10 * (1 + list(REGIME_ORDER).index(regime)),
            )
            med = float(np.median(vals))
            q75, q25 = np.percentile(vals, [75, 25])
            iqr = float(q75 - q25)
        rows.append(
            {
                "model_id": model_id,
                "regime": regime,
                "n_decisions": n,
                "n_episodes": n_ep,
                "mean_cash": mean,
                "ci95_low": lo,
                "ci95_high": hi,
                "median_cash": med,
                "iqr_cash": iqr,
                "mean_equity_exposure": 1.0 - mean if np.isfinite(mean) else float("nan"),
                "block_length": L,
                "n_bootstrap": n_bootstrap,
                "annualised_turnover": outcome_metrics.get("annualised_turnover", float("nan")),
                "annualised_volatility": outcome_metrics.get(
                    "gross_annualised_volatility", float("nan")
                ),
                "max_drawdown": outcome_metrics.get("gross_max_drawdown", float("nan")),
                "sharpe_ratio": outcome_metrics.get("gross_sharpe_ratio", float("nan")),
            }
        )
    return pd.DataFrame(rows)


def cash_shift_stats(
    model_id: str,
    weights: pd.DataFrame,
    *,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    cash = _cash_series(weights)
    labels = assign_paper_regimes(weights.index)
    pairs = [
        ("crash_minus_bull", "crash_period", "bull_market"),
        ("crash_minus_sideways", "crash_period", "sideways_uncertain"),
        ("recovery_minus_crash", "recovery_growth", "crash_period"),
    ]
    rows = []
    for name, a, b in pairs:
        va = cash.values[labels == a]
        vb = cash.values[labels == b]
        # Difference of means with separate block bootstraps, then percentile of
        # bootstrapped mean_a - mean_b via independent draws (conservative).
        # Prefer: bootstrap each regime series, then difference of bootstrap means.
        La = default_block_length(len(va)) if len(va) else 1
        Lb = default_block_length(len(vb)) if len(vb) else 1
        pair_seeds = {
            "crash_minus_bull": 101,
            "crash_minus_sideways": 102,
            "recovery_minus_crash": 103,
        }
        rng = np.random.default_rng(seed + pair_seeds[name])
        point = float(np.mean(va) - np.mean(vb)) if len(va) and len(vb) else float("nan")
        if len(va) and len(vb):
            from .bootstrap import moving_block_indices

            draws = np.empty(n_bootstrap)
            for i in range(n_bootstrap):
                ia = moving_block_indices(len(va), La, rng)
                ib = moving_block_indices(len(vb), Lb, rng)
                draws[i] = float(np.mean(va[ia]) - np.mean(vb[ib]))
            lo, hi = np.quantile(draws, [0.025, 0.975])
        else:
            lo = hi = float("nan")
        rows.append(
            {
                "model_id": model_id,
                "contrast": name,
                "difference": point,
                "ci95_low": float(lo),
                "ci95_high": float(hi),
                "n_left": int(len(va)),
                "n_right": int(len(vb)),
                "block_length_left": La,
                "block_length_right": Lb,
                "n_bootstrap": n_bootstrap,
            }
        )
    return pd.DataFrame(rows)


def evaluate_model(
    model_id: str,
    weights: pd.DataFrame,
    ohlc: pd.DataFrame,
    cost: float = TRANSACTION_COST,
) -> Dict[str, object]:
    # Reindex weights onto canonical order used by outcomes helpers
    w = weights.reindex(columns=list(CANONICAL_INSTRUMENTS)).fillna(0.0)
    rets = hold_returns(ohlc, list(CANONICAL_INSTRUMENTS))
    pnl = portfolio_returns(w, rets, cost=cost)
    gross = performance_metrics(pnl["gross_return"])
    net = performance_metrics(pnl["net_return"])
    turn = turnover_metrics(w)
    metrics = {f"gross_{k}": v for k, v in gross.items()}
    metrics.update({f"net_{k}": v for k, v in net.items()})
    metrics.update(turn)
    metrics["mean_cash_allocation"] = float(w["CASH"].mean())
    metrics["transaction_cost"] = cost
    metrics["weights_source"] = "paper_validation_csv_trading_calendar"
    return {
        "model_id": model_id,
        "returns": pnl,
        "metrics": metrics,
        "simplex": check_simplex(w),
        "weights": w,
    }


def paired_outcome_link(
    best: Dict[str, object],
    worst: Dict[str, object],
    *,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    wb = best["weights"]
    ww = worst["weights"]
    common = wb.index.intersection(ww.index)
    cash_b = wb.loc[common, "CASH"].values
    cash_w = ww.loc[common, "CASH"].values
    equity_b = 1.0 - cash_b
    equity_w = 1.0 - cash_w

    # Daily turnover on common dates
    turn_b = 0.5 * (wb.loc[common] - wb.loc[common].shift(1)).abs().sum(axis=1).iloc[1:]
    turn_w = 0.5 * (ww.loc[common] - ww.loc[common].shift(1)).abs().sum(axis=1).iloc[1:]
    turn_idx = turn_b.index.intersection(turn_w.index)

    rb = best["returns"]["gross_return"]
    rw = worst["returns"]["gross_return"]
    r_common = rb.index.intersection(rw.index)

    rows = []

    def add_pair(metric, a, b, aligned=True):
        if aligned:
            point, lo, hi, L, B = paired_block_bootstrap_diff_ci(
                a, b, n_bootstrap=n_bootstrap, seed=seed
            )
            rows.append(
                {
                    "metric": metric,
                    "DT_max_quadratic_utility": float(np.mean(a)),
                    "DT_16_10": float(np.mean(b)),
                    "difference": point,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "n": int(len(a)),
                    "block_length": L,
                    "n_bootstrap": B,
                    "inference": "paired_moving_block_bootstrap",
                }
            )
        else:
            rows.append(
                {
                    "metric": metric,
                    "DT_max_quadratic_utility": float(a),
                    "DT_16_10": float(b),
                    "difference": float(a) - float(b),
                    "ci95_low": float("nan"),
                    "ci95_high": float("nan"),
                    "n": int(best["metrics"].get("gross_n_observations", np.nan)),
                    "block_length": np.nan,
                    "n_bootstrap": np.nan,
                    "inference": "full_sample_point_estimate",
                }
            )

    add_pair("mean_cash_weight", cash_b, cash_w)
    add_pair("mean_equity_exposure", equity_b, equity_w)
    add_pair(
        "mean_daily_turnover",
        turn_b.loc[turn_idx].values,
        turn_w.loc[turn_idx].values,
    )
    add_pair(
        "mean_daily_gross_return",
        rb.loc[r_common].values,
        rw.loc[r_common].values,
    )

    # Path metrics: point estimates only (full-sample)
    mb, mw = best["metrics"], worst["metrics"]
    for key, label in [
        ("gross_sharpe_ratio", "sharpe_ratio"),
        ("gross_annualised_volatility", "annualised_volatility"),
        ("gross_max_drawdown", "max_drawdown"),
        ("annualised_turnover", "annualised_turnover"),
        ("mean_cash_allocation", "mean_cash_allocation"),
        ("net_sharpe_ratio", "net_sharpe_ratio"),
    ]:
        add_pair(label, mb[key], mw[key], aligned=False)

    # Crash-minus-bull for each model as rows for convenience
    for model_id, w in [
        ("DT_max_quadratic_utility", wb),
        ("DT_16_10", ww),
    ]:
        labels = assign_paper_regimes(w.index)
        cash = w["CASH"].values
        crash = cash[labels == "crash_period"]
        bull = cash[labels == "bull_market"]
        # store later via cash_shift_stats; skip here

    return pd.DataFrame(rows)


def load_diagnostics() -> pd.DataFrame:
    path = ROOT / "results/cross_architecture/tables/cross_architecture_model_summary.csv"
    df = pd.read_csv(path)
    keep = [
        "model_id",
        "temporal_recency_index",
        "temporal_half_life",
        "temporal_normalised_auc",
        "contribution_score",
        "necessity_score",
        "structure_score",
        "aggregate_diagnostic_score",
        "gross_sharpe_ratio",
        "mean_cash_allocation",
        "annualised_turnover",
        "gross_max_drawdown",
        "gross_annualised_volatility",
    ]
    return df.loc[df["model_id"].isin(["DT_max_quadratic_utility", "DT_16_10"]), keep]


def family_backdrop() -> pd.DataFrame:
    """Family performance/turnover at 0 bps; CASH not available for all variants."""
    path = ROOT / "results/metrics/family_cost_sensitivity.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df.loc[df["cost_bps"] == 0].copy()
    # Cross-architecture cash means for the two selected families if present
    cash_path = ROOT / "results/cross_architecture/tables/cash_allocation_by_regime.csv"
    cash = pd.read_csv(cash_path) if cash_path.exists() else pd.DataFrame()
    rows = []
    for _, r in df.iterrows():
        fam = r["model_family"]
        row = {
            "unit": "dt_family_performance",
            "unit_id": fam,
            "median_crash_cash": np.nan,
            "median_bull_cash": np.nan,
            "median_crash_minus_bull_cash": np.nan,
            "iqr_crash_minus_bull": np.nan,
            "median_turnover": r["median_annualized_turnover"],
            "median_max_drawdown": r["median_max_drawdown"],
            "median_sharpe": r["median_sharpe"],
            "diagnostic_summary": np.nan,
            "notes": "CASH-by-regime not available for 96-variant families; performance/turnover only",
        }
        rows.append(row)

    # Add selected-policy CASH from this analysis placeholders filled by caller
    return pd.DataFrame(rows)


def selected_outcome_summary(results: Dict[str, Dict]) -> pd.DataFrame:
    rows = []
    for mid, bundle in results.items():
        m = bundle["metrics"]
        rows.append(
            {
                "model_id": mid,
                "role": "higher_sharpe" if mid == "DT_max_quadratic_utility" else "lower_sharpe",
                "mean_cash_allocation": m["mean_cash_allocation"],
                "annualised_turnover": m["annualised_turnover"],
                "gross_annualised_return": m["gross_annualised_return"],
                "gross_annualised_volatility": m["gross_annualised_volatility"],
                "gross_sharpe_ratio": m["gross_sharpe_ratio"],
                "gross_sortino_ratio": m["gross_sortino_ratio"],
                "gross_max_drawdown": m["gross_max_drawdown"],
                "net_sharpe_ratio": m["net_sharpe_ratio"],
                "n_observations": m["gross_n_observations"],
                "weights_source": m["weights_source"],
                "transaction_cost": m["transaction_cost"],
            }
        )
    return pd.DataFrame(rows)


def run_tier_b_analysis(
    best_weights_path: str | Path,
    worst_weights_path: str | Path,
    data_h5: str | Path,
    out_dir: str | Path,
    *,
    n_bootstrap: int = 2000,
    seed: int = 42,
    cost: float = TRANSACTION_COST,
) -> Dict[str, object]:
    out_dir = Path(out_dir)
    for sub in ("data", "tables", "figures", "supplementary", "latex", "logs", "tests"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    md = load_market_data(data_h5)
    trading = md.merged.index  # trading calendar after indicator merge

    # End = min(weight end, sideways end, last trading date)
    raw_best = load_weight_csv(best_weights_path)
    raw_worst = load_weight_csv(worst_weights_path)
    end = min(
        raw_best.index.max(),
        raw_worst.index.max(),
        pd.Timestamp(PAPER_REGIMES["sideways_uncertain"][1]),
        trading.max(),
    )

    pair = load_filtered_pair(
        best_weights_path,
        worst_weights_path,
        trading,
        start="2016-01-01",
        end=str(end.date()),
    )
    n_raw = len(raw_best)
    n_filt = len(pair["DT_max_quadratic_utility"])

    bundles = {}
    cash_by_regime_frames = []
    shift_frames = []
    for mid in ("DT_max_quadratic_utility", "DT_16_10"):
        w = pair[mid]
        # Persist filtered weights
        w.to_csv(out_dir / "data" / f"weights_trading__{mid}.csv")
        bundle = evaluate_model(mid, w, md.prices, cost=cost)
        bundles[mid] = bundle
        bundle["returns"].to_csv(out_dir / "data" / f"portfolio_returns__{mid}.csv")
        cash_by_regime_frames.append(
            regime_cash_table(
                mid, bundle["weights"], bundle["metrics"], n_bootstrap=n_bootstrap, seed=seed
            )
        )
        shift_frames.append(
            cash_shift_stats(mid, bundle["weights"], n_bootstrap=n_bootstrap, seed=seed)
        )

    cash_regime = pd.concat(cash_by_regime_frames, ignore_index=True)
    shifts = pd.concat(shift_frames, ignore_index=True)
    outcome_link = paired_outcome_link(
        bundles["DT_max_quadratic_utility"],
        bundles["DT_16_10"],
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    selected_summary = selected_outcome_summary(bundles)
    diagnostics = load_diagnostics()

    # Family backdrop + selected-policy CASH rows
    fam = family_backdrop()
    selected_family_rows = []
    for mid, bundle in bundles.items():
        labels = assign_paper_regimes(bundle["weights"].index)
        cash = bundle["weights"]["CASH"]
        crash = float(cash[labels == "crash_period"].mean())
        bull = float(cash[labels == "bull_market"].mean())
        diag = diagnostics.loc[diagnostics["model_id"] == mid]
        selected_family_rows.append(
            {
                "unit": "selected_policy",
                "unit_id": mid,
                "median_crash_cash": crash,
                "median_bull_cash": bull,
                "median_crash_minus_bull_cash": crash - bull,
                "iqr_crash_minus_bull": np.nan,
                "median_turnover": bundle["metrics"]["annualised_turnover"],
                "median_max_drawdown": bundle["metrics"]["gross_max_drawdown"],
                "median_sharpe": bundle["metrics"]["gross_sharpe_ratio"],
                "diagnostic_summary": float(diag["aggregate_diagnostic_score"].iloc[0])
                if len(diag)
                else np.nan,
                "notes": "Paper validation weights filtered to trading calendar",
            }
        )
    family_summary = pd.concat(
        [pd.DataFrame(selected_family_rows), fam], ignore_index=True
    )

    # Statistical tests table = shifts + paired cash metrics
    stats = pd.concat(
        [
            shifts.assign(table="regime_cash_shift"),
            outcome_link.assign(table="paired_model_contrast"),
        ],
        ignore_index=True,
        sort=False,
    )

    # Also append cross-architecture 5-model cash means for supplement context
    xa_cash = ROOT / "results/cross_architecture/tables/cash_allocation_by_regime.csv"
    xa_frame = pd.read_csv(xa_cash) if xa_cash.exists() else pd.DataFrame()

    # Write primary tables
    cash_regime.to_csv(out_dir / "tables" / "selected_model_cash_by_regime.csv", index=False)
    outcome_link.to_csv(out_dir / "tables" / "selected_model_outcome_link.csv", index=False)
    family_summary.to_csv(out_dir / "tables" / "family_cash_outcome_summary.csv", index=False)
    stats.to_csv(out_dir / "tables" / "statistical_tests.csv", index=False)
    selected_summary.to_csv(out_dir / "tables" / "selected_model_performance.csv", index=False)
    shifts.to_csv(out_dir / "tables" / "cash_regime_shifts.csv", index=False)
    diagnostics.to_csv(out_dir / "tables" / "selected_model_diagnostics.csv", index=False)
    if len(xa_frame):
        xa_frame.to_csv(out_dir / "tables" / "cross_architecture_cash_by_regime.csv", index=False)

    meta = {
        "n_raw_calendar_rows_best": int(n_raw),
        "n_trading_rows": int(n_filt),
        "n_weekends_holidays_removed_approx": int(n_raw - n_filt),
        "start": str(pair["common_dates"].min().date()),
        "end": str(pair["common_dates"].max().date()),
        "regimes": PAPER_REGIMES,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "transaction_cost": cost,
        "best_weights_path": str(best_weights_path),
        "worst_weights_path": str(worst_weights_path),
        "data_h5": str(data_h5),
        "simplex_best": bundles["DT_max_quadratic_utility"]["simplex"],
        "simplex_worst": bundles["DT_16_10"]["simplex"],
    }
    pd.Series({k: str(v) for k, v in meta.items()}).to_csv(
        out_dir / "logs" / "run_settings.csv"
    )

    return {
        "bundles": bundles,
        "cash_regime": cash_regime,
        "shifts": shifts,
        "outcome_link": outcome_link,
        "family_summary": family_summary,
        "selected_summary": selected_summary,
        "diagnostics": diagnostics,
        "xa_cash": xa_frame,
        "meta": meta,
        "out_dir": out_dir,
    }
