"""Main and supplementary figures for Comment 3."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .regimes import REGIME_ORDER

REGIME_LABELS = {
    "bull_market": "Bull",
    "crash_period": "Crash",
    "recovery_growth": "Recovery",
    "sideways_uncertain": "Sideways",
}

MODEL_STYLE = {
    "DT_max_quadratic_utility": {"label": "Higher-Sharpe DT", "color": "#1f4e79"},
    "DT_16_10": {"label": "Lower-Sharpe DT", "color": "#c45c26"},
}


def plot_cash_by_regime_main(
    cash_regime: pd.DataFrame,
    out_png: str | Path,
    out_pdf: str | Path,
    *,
    width: float = 6.5,
    height: float = 3.8,
    dpi: int = 300,
) -> None:
    fig, ax = plt.subplots(figsize=(width, height))
    regimes = list(REGIME_ORDER)
    x = np.arange(len(regimes))
    width_bar = 0.36

    for i, model_id in enumerate(["DT_max_quadratic_utility", "DT_16_10"]):
        sub = cash_regime.loc[cash_regime["model_id"] == model_id].set_index("regime")
        means = [sub.loc[r, "mean_cash"] if r in sub.index else np.nan for r in regimes]
        lo = [sub.loc[r, "ci95_low"] if r in sub.index else np.nan for r in regimes]
        hi = [sub.loc[r, "ci95_high"] if r in sub.index else np.nan for r in regimes]
        yerr = np.vstack(
            [
                np.array(means) - np.array(lo),
                np.array(hi) - np.array(means),
            ]
        )
        style = MODEL_STYLE[model_id]
        offset = -width_bar / 2 if i == 0 else width_bar / 2
        ax.bar(
            x + offset,
            means,
            width=width_bar,
            color=style["color"],
            label=style["label"],
            yerr=yerr,
            capsize=3,
            error_kw={"elinewidth": 1.0, "capthick": 1.0},
        )

    ax.set_xticks(x)
    ax.set_xticklabels([REGIME_LABELS[r] for r in regimes])
    ax.set_ylabel("Mean CASH weight")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Market regime")
    ax.legend(frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Regime-conditioned CASH allocation (95% moving-block bootstrap CI)")
    fig.tight_layout()
    out_png = Path(out_png)
    out_pdf = Path(out_pdf)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_covid_event_panel(
    bundles: Dict[str, Dict],
    out_png: str | Path,
    out_pdf: str | Path,
    *,
    start: str = "2020-01-15",
    end: str = "2020-06-30",
    width: float = 6.5,
    height: float = 5.2,
    dpi: int = 300,
) -> None:
    """Supplementary event panel: CASH and drawdown around the COVID crash."""
    fig, axes = plt.subplots(2, 1, figsize=(width, height), sharex=True)
    for model_id, style in MODEL_STYLE.items():
        w = bundles[model_id]["weights"]
        r = bundles[model_id]["returns"]["gross_return"]
        mask_w = (w.index >= start) & (w.index <= end)
        mask_r = (r.index >= start) & (r.index <= end)
        cash = w.loc[mask_w, "CASH"]
        wealth = (1.0 + r.loc[mask_r]).cumprod()
        dd = wealth / wealth.cummax() - 1.0
        axes[0].plot(cash.index, cash.values, color=style["color"], label=style["label"], lw=1.4)
        axes[1].plot(dd.index, dd.values, color=style["color"], label=style["label"], lw=1.4)

    for ax in axes:
        ax.axvspan(pd.Timestamp("2020-02-16"), pd.Timestamp("2020-05-31"), color="#dddddd", alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("CASH weight")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(frameon=False, loc="best")
    axes[0].set_title("CASH allocation and portfolio drawdown around the COVID crash")
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
