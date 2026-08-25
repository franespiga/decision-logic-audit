#!/usr/bin/env python
"""Run Tier-B cash–outcome link for Reviewer 1 Comment 3."""

from __future__ import annotations

import json
import argparse
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ablation.cash_outcome_link.analysis import run_tier_b_analysis
from ablation.cash_outcome_link.plotting import plot_cash_by_regime_main, plot_covid_event_panel
from ablation.cash_outcome_link.reporting import (
    write_discussion_tex,
    write_figure_tex,
    write_main_table_tex,
    write_reviewer_response,
    write_subsection_tex,
    write_supplement_tex,
)
from ablation.cash_outcome_link.regimes import assign_paper_regimes


def run_validation(result: dict, out_dir: Path) -> list[str]:
    errors = []
    bundles = result["bundles"]
    cash = result["cash_regime"]
    meta = result["meta"]

    for mid, b in bundles.items():
        w = b["weights"]
        sx = b["simplex"]
        if sx["violates_long_only"]:
            errors.append(f"{mid}: long-only violated")
        if sx["violates_full_investment"]:
            errors.append(f"{mid}: full investment violated (max |sum-1|={sx['max_abs_sum_deviation']})")
        if "CASH" not in w.columns:
            errors.append(f"{mid}: CASH missing")
        eq = (1.0 - w["CASH"]) - w.drop(columns=["CASH"]).sum(axis=1)
        if float(eq.abs().max()) > 1e-6:
            errors.append(f"{mid}: equity exposure != 1-CASH")
        labels = assign_paper_regimes(w.index)
        if (labels == "Unknown").any():
            errors.append(f"{mid}: Unknown regimes present")

    # Common dates
    wb = bundles["DT_max_quadratic_utility"]["weights"].index
    ww = bundles["DT_16_10"]["weights"].index
    if not wb.equals(ww):
        errors.append("common-date weight indices differ")

    # Weekend check: no Saturday/Sunday
    for mid, b in bundles.items():
        wd = b["weights"].index.dayofweek
        if (wd >= 5).any():
            errors.append(f"{mid}: weekend dates remain after filtering")

    # Table decimals for latex will be handled in writers; check CSV finite means
    if cash["mean_cash"].isna().any():
        errors.append("NaN mean_cash in regime table")

    # Causal language check on generated tex
    for p in (out_dir / "latex").glob("*.tex"):
        txt = p.read_text(encoding="utf-8").lower()
        # Flag assertive causal claims; allow explicit disclaimers ("not a causal guarantee").
        for bad in (
            "causes drawdown protection",
            "prevents drawdown",
            "drives sharpe",
            "proves that",
            "guarantees drawdown",
        ):
            if bad in txt:
                errors.append(f"causal/forbidden language in {p.name}: {bad}")

    # Main table matches CSV
    main_csv = out_dir / "tables" / "selected_model_cash_by_regime.csv"
    if not main_csv.exists():
        errors.append("missing selected_model_cash_by_regime.csv")

    report = out_dir / "tests" / "validation_results.txt"
    report.write_text(
        "PASS\n" if not errors else "FAIL\n" + "\n".join(errors) + "\n",
        encoding="utf-8",
    )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Tier-B cash-outcome link analysis.")
    parser.add_argument(
        "--cost-bps",
        type=float,
        default=20.0,
        help="One-way transaction cost in basis points for net outcomes (default: 20).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/reviewer1_comment3",
        help="Output directory relative to repository root.",
    )
    args = parser.parse_args()

    t0 = time.time()
    out_dir = ROOT / args.out_dir
    best = ROOT / "notebooks" / "df_weights_best.csv"
    worst = ROOT / "notebooks" / "df_weights_worst.csv"
    data_h5 = ROOT / "models" / "comparison" / "dt" / "DT_16_10" / "data.h5"
    cost = float(args.cost_bps) / 10000.0

    print("Running Tier B cash–outcome analysis...")
    print(f"  best weights:  {best}")
    print(f"  worst weights: {worst}")
    print(f"  market data:   {data_h5}")
    print(f"  out dir:       {out_dir}")
    print(f"  cost (bps):    {args.cost_bps:g}")

    result = run_tier_b_analysis(
        best,
        worst,
        data_h5,
        out_dir,
        n_bootstrap=2000,
        seed=42,
        cost=cost,
    )

    plot_cash_by_regime_main(
        result["cash_regime"],
        out_dir / "figures" / "F_cash_outcome_main.png",
        out_dir / "figures" / "F_cash_outcome_main.pdf",
    )
    plot_covid_event_panel(
        result["bundles"],
        out_dir / "figures" / "F_cash_covid_event_panel.png",
        out_dir / "figures" / "F_cash_covid_event_panel.pdf",
    )

    write_main_table_tex(
        result["cash_regime"],
        result["shifts"],
        result["selected_summary"],
        out_dir / "latex" / "main_cash_outcome_table.tex",
        cost_bps=args.cost_bps,
    )
    write_figure_tex(out_dir / "latex" / "main_cash_outcome_figure.tex")
    write_subsection_tex(
        result["cash_regime"],
        result["shifts"],
        result["selected_summary"],
        result["diagnostics"],
        out_dir / "latex" / "main_cash_outcome_subsection.tex",
        cost_bps=args.cost_bps,
    )
    write_discussion_tex(out_dir / "latex" / "discussion_cash_outcome_paragraph.tex")
    write_reviewer_response(
        result["cash_regime"],
        result["shifts"],
        result["selected_summary"],
        out_dir / "latex" / "reviewer1_comment3_response.tex",
    )
    write_supplement_tex(
        result["cash_regime"],
        result["shifts"],
        result["outcome_link"],
        result["selected_summary"],
        result["meta"],
        out_dir / "supplementary" / "supplementary_cash_diagnostics.tex",
        cost_bps=args.cost_bps,
    )

    errors = run_validation(result, out_dir)

    # Final findings
    cash = result["cash_regime"]
    shifts = result["shifts"]
    sel = result["selected_summary"].set_index("model_id")
    lines = [
        "# Final findings — Reviewer 1 Comment 3 (Tier B)",
        "",
        "## Models",
        "- Higher-Sharpe: `DT_max_quadratic_utility` (paper `df_weights_best.csv`)",
        "- Lower-Sharpe: `DT_16_10` (paper `df_weights_worst.csv`)",
        "",
        f"## Sample",
        f"- Trading dates: {result['meta']['start']} to {result['meta']['end']}",
        f"- Rows after calendar filter: {result['meta']['n_trading_rows']}",
        f"- Approx. weekends/holidays removed: {result['meta']['n_weekends_holidays_removed_approx']}",
        "",
        "## Principal CASH results",
    ]
    for mid in ("DT_max_quadratic_utility", "DT_16_10"):
        sub = cash.loc[cash["model_id"] == mid].set_index("regime")
        sh = shifts.loc[
            (shifts["model_id"] == mid) & (shifts["contrast"] == "crash_minus_bull")
        ].iloc[0]
        lines.append(
            f"- **{mid}**: bull CASH={sub.loc['bull_market','mean_cash']:.3f} "
            f"[{sub.loc['bull_market','ci95_low']:.3f}, {sub.loc['bull_market','ci95_high']:.3f}]; "
            f"crash CASH={sub.loc['crash_period','mean_cash']:.3f} "
            f"[{sub.loc['crash_period','ci95_low']:.3f}, {sub.loc['crash_period','ci95_high']:.3f}]; "
            f"crash−bull={sh['difference']:.3f} [{sh['ci95_low']:.3f}, {sh['ci95_high']:.3f}]"
        )
    lines += [
        "",
        "## Outcomes (gross)",
        f"- Higher-Sharpe: Sharpe={sel.loc['DT_max_quadratic_utility','gross_sharpe_ratio']:.3f}, "
        f"vol={sel.loc['DT_max_quadratic_utility','gross_annualised_volatility']:.3f}, "
        f"MDD={sel.loc['DT_max_quadratic_utility','gross_max_drawdown']:.3f}, "
        f"turnover={sel.loc['DT_max_quadratic_utility','annualised_turnover']:.3f}",
        f"- Lower-Sharpe: Sharpe={sel.loc['DT_16_10','gross_sharpe_ratio']:.3f}, "
        f"vol={sel.loc['DT_16_10','gross_annualised_volatility']:.3f}, "
        f"MDD={sel.loc['DT_16_10','gross_max_drawdown']:.3f}, "
        f"turnover={sel.loc['DT_16_10','annualised_turnover']:.3f}",
        "",
        "## Validation",
        "PASS" if not errors else "FAIL: " + "; ".join(errors),
        "",
        "## Association statement",
        "Diagnostic temporal patterns coincide with differing crash CASH responses; "
        "allocation profiles are reflected in volatility, drawdown, turnover and Sharpe. "
        "No causal claim is made.",
        "",
    ]
    (out_dir / "final_findings.md").write_text("\n".join(lines), encoding="utf-8")

    elapsed = time.time() - t0
    run_report = f"""# Run report: Reviewer 1 Comment 3 (Tier B)

Completed in {elapsed:.1f} s.

## Command

```
python scripts/run_cash_outcome_link.py --cost-bps {args.cost_bps:g} --out-dir {args.out_dir}
```

## Inputs

- Best weights: `{best}`
- Worst weights: `{worst}`
- Market data: `{data_h5}`
- Regimes: paper schedule (bull through 2020-02-15; crash from 2020-02-16; sideways through 2024-05-30)
- Bootstrap: B=2000, seed=42, block length round(T^(1/3))
- Transaction cost: {cost:.4f} one-way ({args.cost_bps:g} bp)

## Sample

- Trading dates: {result['meta']['start']} to {result['meta']['end']}
- n trading rows: {result['meta']['n_trading_rows']}
- Approx. calendar rows removed: {result['meta']['n_weekends_holidays_removed_approx']}

## Artefacts

- `tables/selected_model_cash_by_regime.csv`
- `tables/selected_model_outcome_link.csv`
- `tables/family_cash_outcome_summary.csv`
- `tables/statistical_tests.csv`
- `figures/F_cash_outcome_main.png` / `.pdf`
- `figures/F_cash_covid_event_panel.png` / `.pdf` (supplement)
- `latex/main_cash_outcome_subsection.tex`
- `latex/main_cash_outcome_table.tex`
- `latex/main_cash_outcome_figure.tex`
- `latex/discussion_cash_outcome_paragraph.tex`
- `latex/reviewer1_comment3_response.tex`
- `supplementary/supplementary_cash_diagnostics.tex`
- `final_findings.md`

## Validation

{"PASS" if not errors else "FAIL"}
{chr(10).join(errors) if errors else "All checks passed."}

## Simplex

- Best: {json.dumps(result['meta']['simplex_best'])}
- Worst: {json.dumps(result['meta']['simplex_worst'])}
"""
    (out_dir / "run_report.md").write_text(run_report, encoding="utf-8")
    print(run_report)
    return 0 if not errors else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
