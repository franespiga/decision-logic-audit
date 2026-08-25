"""Combine Batch A evidence with Batch C validity overlays."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from .holdout_assessment import INCOMPLETE_EVIDENCE_FAMILY
from .validity_config import INCOMPLETE_EVIDENCE_FAMILY as _INC  # noqa: F401


def combined_validity_status(
    *,
    baseline_status: str,
    randomization_status: str,
    deletion_status: str,
    insertion_status: str,
    batch_a_mandatory_failed: bool,
    force_incomplete: bool = False,
) -> tuple[str, str]:
    if force_incomplete:
        return "INCOMPLETE", "Frozen incomplete diagnostic evidence family."
    statuses = {
        "baseline": baseline_status,
        "randomization": randomization_status,
        "deletion": deletion_status,
        "insertion": insertion_status,
    }
    if any(s == "INCOMPLETE" for s in (baseline_status, randomization_status, deletion_status)):
        return "INCOMPLETE", "One or more core validity checks unavailable."
    if batch_a_mandatory_failed:
        return "UNSUPPORTED", "Existing Batch A mandatory validity gate failed."
    if any(s == "UNSUPPORTED" for s in (baseline_status, randomization_status, deletion_status)):
        failed = [k for k, v in statuses.items() if v == "UNSUPPORTED" and k != "insertion"]
        return "UNSUPPORTED", f"Core check(s) unsupported: {', '.join(failed)}."
    if (
        baseline_status == "SUPPORTED"
        and randomization_status == "SUPPORTED"
        and deletion_status == "SUPPORTED"
        and insertion_status in {"SUPPORTED", "MIXED"}
    ):
        return "SUPPORTED", "Core checks supported; insertion confirmatory supported/mixed."
    if insertion_status == "UNSUPPORTED" and all(
        s == "SUPPORTED" for s in (baseline_status, randomization_status, deletion_status)
    ):
        return "MIXED", "Core checks supported but confirmatory insertion unsupported."
    if any(s == "MIXED" for s in (baseline_status, randomization_status, deletion_status)):
        return "MIXED", "At least one core check mixed; none unsupported."
    return "MIXED", "Mixed validity evidence under primary operational rules."


def build_evidence_overlay(
    batch_a_primary: pd.DataFrame,
    batch_a_raw: Optional[pd.DataFrame],
    family_results: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    raw_by_family = {}
    if batch_a_raw is not None and not batch_a_raw.empty and "model_family" in batch_a_raw.columns:
        for fam, grp in batch_a_raw.groupby("model_family"):
            raw_by_family[str(fam)] = grp.iloc[0]

    for _, prow in batch_a_primary.sort_values("model_family").iterrows():
        fam = str(prow["model_family"])
        fres = family_results.get(fam, {})
        raw = raw_by_family.get(fam, {})
        baseline = fres.get("baseline_status", "INCOMPLETE")
        randomization = fres.get("randomization_status", "INCOMPLETE")
        deletion = fres.get("deletion_status", "INCOMPLETE")
        insertion = fres.get("insertion_status", "INCOMPLETE")
        mandatory = str(prow.get("mandatory_validity_status", "")).lower()
        batch_a_failed = mandatory in {"fail", "failed", "false"}
        force_inc = fam == INCOMPLETE_EVIDENCE_FAMILY
        combined, reason = combined_validity_status(
            baseline_status=baseline,
            randomization_status=randomization,
            deletion_status=deletion,
            insertion_status=insertion,
            batch_a_mandatory_failed=batch_a_failed,
            force_incomplete=force_inc,
        )
        # Batch A validity status from contribution if present
        batch_a_validity = str(
            raw.get("contribution_validity_status", prow.get("family_score_status", "unknown"))
            if hasattr(raw, "get")
            else prow.get("family_score_status", "unknown")
        )
        rows.append(
            {
                "model_family": fam,
                "batch_a_diagnostic_score": prow.get("diagnostic_score"),
                "batch_a_diagnostic_band": prow.get("diagnostic_band"),
                "frozen_screening_decision": prow.get("operational_recommendation"),
                "batch_a_validity_status": batch_a_validity,
                "contribution_rank_stability": _get(raw, "contribution_rank_stability"),
                "contribution_sign_stability": _get(raw, "contribution_sign_stability"),
                "baseline_concordance_status": baseline,
                "baseline_concordance_median_tau": fres.get("baseline_median_tau"),
                "randomization_status": randomization,
                "full_randomization_similarity": fres.get("full_randomization_similarity"),
                "randomization_gap": fres.get("randomization_gap"),
                "deletion_status": deletion,
                "deletion_gap": fres.get("deletion_gap"),
                "deletion_ci_lower": fres.get("deletion_ci_lower"),
                "insertion_status": insertion,
                "insertion_gap": fres.get("insertion_gap"),
                "insertion_ci_lower": fres.get("insertion_ci_lower"),
                "combined_validity_status": combined,
                "combined_validity_reason": reason,
                "evidence_complete": combined != "INCOMPLETE",
                "notes": fres.get("notes", ""),
            }
        )
    return pd.DataFrame(rows)


def _get(obj, key):
    if obj is None:
        return float("nan")
    if hasattr(obj, "get"):
        return obj.get(key, float("nan"))
    try:
        return obj[key]
    except Exception:
        return float("nan")


def status_sensitivity_table(overlay: pd.DataFrame) -> pd.DataFrame:
    """Less-strict / primary / more-strict combined-status variants."""
    rows = []
    for _, r in overlay.iterrows():
        fam = r["model_family"]
        b, rnd, d, ins = (
            r["baseline_concordance_status"],
            r["randomization_status"],
            r["deletion_status"],
            r["insertion_status"],
        )
        primary = r["combined_validity_status"]
        # Less strict: treat MIXED core as acceptable for SUPPORTED if none UNSUPPORTED
        if fam == INCOMPLETE_EVIDENCE_FAMILY:
            less = more = primary = "INCOMPLETE"
        else:
            core = [b, rnd, d]
            if any(x == "INCOMPLETE" for x in core):
                less = more = "INCOMPLETE"
            elif any(x == "UNSUPPORTED" for x in core):
                less = "UNSUPPORTED"
                more = "UNSUPPORTED"
            else:
                less = "SUPPORTED" if all(x in {"SUPPORTED", "MIXED"} for x in core) else "MIXED"
                # More strict: require all four SUPPORTED
                more = (
                    "SUPPORTED"
                    if all(x == "SUPPORTED" for x in (b, rnd, d, ins))
                    else ("MIXED" if "UNSUPPORTED" not in (b, rnd, d, ins) else "UNSUPPORTED")
                )
        rows.append(
            {
                "model_family": fam,
                "primary_combined_status": primary,
                "less_strict_combined_status": less,
                "more_strict_combined_status": more,
                "frozen_screening_decision": r["frozen_screening_decision"],
            }
        )
    return pd.DataFrame(rows)
