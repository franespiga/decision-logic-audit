"""Market-benchmark discovery for Batch C2 targeted validity.

Selects the authoritative validation-period market comparator from manuscript
and experiment evidence. Does not use holdout outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .config import BENCHMARK_FAMILIES, VALIDATION_END, VALIDATION_START
from .io import ensure_output_dir, write_table


@dataclass
class BenchmarkCandidate:
    candidate_name: str
    source_path: str
    source_role: str
    validation_start: str
    validation_end: str
    cost_bps: float
    sharpe: float
    explicitly_called_market_benchmark: bool
    reproducible: bool
    selected: bool
    selection_reason: str
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def load_family_sharpe_at_cost(metrics_dir: Path, cost_bps: int = 5) -> pd.DataFrame:
    path = Path(metrics_dir) / "family_cost_sensitivity.csv"
    if not path.exists():
        alt = Path(metrics_dir) / "family_cost_sensitivity.parquet"
        path = alt if alt.exists() else path
    if not path.exists():
        raise FileNotFoundError(f"Missing family_cost_sensitivity under {metrics_dir}")
    df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
    if "cost_bps" not in df.columns or "median_sharpe" not in df.columns:
        raise ValueError(f"{path} missing cost_bps/median_sharpe")
    out = df[df["cost_bps"].astype(float) == float(cost_bps)].copy()
    return out


def discover_market_benchmark_candidates(
    metrics_dir: Path,
    *,
    cost_bps: int = 5,
    validation_start: pd.Timestamp = VALIDATION_START,
    validation_end: pd.Timestamp = VALIDATION_END,
    manuscript_evidence: Optional[Sequence[Tuple[str, str]]] = None,
) -> List[BenchmarkCandidate]:
    """Return ranked candidates; selection flags are set when unambiguous."""
    perf = load_family_sharpe_at_cost(metrics_dir, cost_bps=cost_bps)
    metrics_path = str(Path(metrics_dir) / "family_cost_sensitivity.csv")
    candidates: List[BenchmarkCandidate] = []

    # Manuscript / experiment identifiers for market portfolio.
    explicit_market = {
        "^GSPC": (
            True,
            "Manuscript states S&P 500 (^GSPC) is used as a proxy for the market portfolio; "
            "ESWA revision package sets BENCHMARK_IDENTIFIER='^GSPC'.",
        ),
        "EWP": (
            False,
            "Equal-weighted portfolio is an explicit comparator but is distinguished from "
            "the market portfolio in the manuscript.",
        ),
    }

    for fam in sorted(BENCHMARK_FAMILIES):
        rows = perf[perf["model_family"].astype(str) == fam]
        if rows.empty:
            candidates.append(
                BenchmarkCandidate(
                    candidate_name=fam,
                    source_path=metrics_path,
                    source_role="family_cost_sensitivity_missing_row",
                    validation_start=str(validation_start.date()),
                    validation_end=str(validation_end.date()),
                    cost_bps=float(cost_bps),
                    sharpe=float("nan"),
                    explicitly_called_market_benchmark=bool(explicit_market.get(fam, (False, ""))[0]),
                    reproducible=False,
                    selected=False,
                    selection_reason="missing_from_family_cost_sensitivity",
                    notes="Benchmark family listed in BENCHMARK_FAMILIES but absent from metrics table.",
                )
            )
            continue
        sharpe = float(rows.iloc[0]["median_sharpe"])
        is_market, reason = explicit_market.get(fam, (False, "Listed in BENCHMARK_FAMILIES only."))
        candidates.append(
            BenchmarkCandidate(
                candidate_name=fam,
                source_path=metrics_path,
                source_role="family_cost_sensitivity_validation_median_sharpe",
                validation_start=str(validation_start.date()),
                validation_end=str(validation_end.date()),
                cost_bps=float(cost_bps),
                sharpe=sharpe,
                explicitly_called_market_benchmark=is_market,
                reproducible=True,
                selected=False,
                selection_reason="",
                notes=reason,
            )
        )

    if manuscript_evidence:
        for name, note in manuscript_evidence:
            if any(c.candidate_name == name for c in candidates):
                continue
            candidates.append(
                BenchmarkCandidate(
                    candidate_name=name,
                    source_path="manuscript_evidence",
                    source_role="textual",
                    validation_start=str(validation_start.date()),
                    validation_end=str(validation_end.date()),
                    cost_bps=float(cost_bps),
                    sharpe=float("nan"),
                    explicitly_called_market_benchmark=True,
                    reproducible=False,
                    selected=False,
                    selection_reason="textual_only_not_reproducible",
                    notes=note,
                )
            )

    marketish = [c for c in candidates if c.explicitly_called_market_benchmark and c.reproducible]
    if len(marketish) == 1:
        marketish[0].selected = True
        marketish[0].selection_reason = (
            "Unique authoritative market portfolio proxy with reproducible validation Sharpe."
        )
    elif len(marketish) > 1:
        for c in marketish:
            c.selection_reason = "ambiguous_multiple_explicit_market_benchmarks"
    else:
        # Fall back: if only one reproducible BENCHMARK_FAMILIES row marked market-like by name
        gspc = [c for c in candidates if c.candidate_name == "^GSPC" and c.reproducible]
        if len(gspc) == 1:
            gspc[0].selected = True
            gspc[0].explicitly_called_market_benchmark = True
            gspc[0].selection_reason = (
                "Selected ^GSPC as market portfolio proxy per manuscript language."
            )

    return candidates


def write_benchmark_candidates(candidates: Sequence[BenchmarkCandidate], path: Path) -> pd.DataFrame:
    df = pd.DataFrame([c.to_dict() for c in candidates]).sort_values(
        ["explicitly_called_market_benchmark", "candidate_name"], ascending=[False, True]
    )
    ensure_output_dir(path.parent)
    write_table(df, path)
    return df


def select_authoritative_benchmark(
    candidates: Sequence[BenchmarkCandidate],
    output_dir: Path,
) -> Optional[BenchmarkCandidate]:
    selected = [c for c in candidates if c.selected]
    if len(selected) == 1:
        return selected[0]
    # Ambiguity stop
    lines = [
        "# Batch C2 market-benchmark confirmation required",
        "",
        "Multiple or zero authoritative market benchmarks remain after discovery.",
        "Do not select a benchmark according to which produces a preferred family subset.",
        "",
    ]
    for c in candidates:
        lines.append(
            f"- `{c.candidate_name}`: Sharpe={c.sharpe}, explicit_market={c.explicitly_called_market_benchmark}, "
            f"reproducible={c.reproducible}, selected={c.selected}. {c.notes}"
        )
    ensure_output_dir(output_dir)
    (Path(output_dir) / "BATCH_C2_MARKET_BENCHMARK_CONFIRMATION_REQUIRED.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return None


def select_above_market_families(
    metrics_dir: Path,
    *,
    market_name: str,
    market_sharpe: float,
    cost_bps: int = 5,
    mandatory_family: str = "max_quadratic_utility_TI_True",
    exclude_benchmarks: Sequence[str] = ("EWP", "^GSPC"),
) -> pd.DataFrame:
    perf = load_family_sharpe_at_cost(metrics_dir, cost_bps=cost_bps)
    rows = []
    for _, r in perf.iterrows():
        fam = str(r["model_family"])
        if fam in set(exclude_benchmarks):
            continue
        sharpe = float(r["median_sharpe"])
        above = bool(sharpe > float(market_sharpe))
        mandatory = fam == mandatory_family
        if not (above or mandatory):
            # still record for transparency? Spec wants selection table of selected;
            # include all DT families with flags for auditability.
            pass
        scope = "full" if mandatory else ("partial" if above else "excluded")
        reason = []
        if mandatory:
            reason.append("mandatory_full_validation")
        if above:
            reason.append("validation_median_sharpe_strictly_above_market")
        if not reason:
            reason.append("below_or_equal_market_and_not_mandatory")
        rows.append(
            {
                "model_family": fam,
                "validation_median_sharpe": sharpe,
                "market_benchmark_name": market_name,
                "market_benchmark_sharpe": float(market_sharpe),
                "sharpe_difference": sharpe - float(market_sharpe),
                "above_market": above,
                "mandatory_full_validation": mandatory,
                "validation_scope": scope if (above or mandatory) else "excluded",
                "selection_reason": "+".join(reason),
                "selected_for_batch_c2": bool(above or mandatory),
            }
        )
    df = pd.DataFrame(rows).sort_values(
        ["selected_for_batch_c2", "mandatory_full_validation", "validation_median_sharpe"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return df
