"""Source-access guards and Batch A/B freeze verification for Batch C."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

from .holdout_assessment import (
    EXPECTED_DECISION_COUNTS,
    FreezeVerificationResult,
    sha256_file,
    verify_batch_a_freeze,
)
from .io import ensure_output_dir, write_table
from .validity_config import (
    FORBIDDEN_ANALYTICAL_BASENAMES,
    FORBIDDEN_BATCH_B_OUTCOME_STEMS,
    ValidityConfig,
)

LOGGER = logging.getLogger("audit_scoring.validity_guards")


@dataclass
class SourceAccessLog:
    rows: List[Dict[str, Any]] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)

    def record(
        self,
        source_path: Path | str,
        *,
        source_role: str,
        validation_or_holdout: str,
        loaded: bool,
        permitted: bool,
        notes: str = "",
    ) -> None:
        path = Path(source_path)
        self.rows.append(
            {
                "source_path": str(path),
                "source_role": source_role,
                "validation_or_holdout": validation_or_holdout,
                "loaded": bool(loaded),
                "permitted": bool(permitted),
                "notes": notes,
            }
        )
        if loaded and not permitted:
            msg = f"FORBIDDEN analytical load: {path} ({source_role})"
            self.violations.append(msg)
            raise RuntimeError(msg)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def classify_path_scope(path: Path | str) -> str:
    text = str(path).replace("\\", "/").lower()
    name = Path(path).name.lower()
    stem = Path(path).stem.lower()
    if any(stem.startswith(b) or b in stem for b in FORBIDDEN_ANALYTICAL_BASENAMES):
        return "holdout"
    if "audit_holdout" in text and "verification" not in text:
        if any(s in stem for s in FORBIDDEN_BATCH_B_OUTCOME_STEMS) or "holdout" in stem:
            return "holdout"
    if "holdout" in name or "holdout" in text:
        if "verification" in text or "frozen" in text or "manifest" in text:
            return "provenance"
        return "holdout"
    return "validation"


def assert_permitted_analytical_load(path: Path | str, *, role: str, log: SourceAccessLog) -> Path:
    path = Path(path)
    scope = classify_path_scope(path)
    permitted = scope != "holdout"
    # Provenance-only Batch B artefacts may be referenced but not used analytically.
    if scope == "provenance":
        log.record(
            path,
            source_role=role,
            validation_or_holdout="provenance",
            loaded=False,
            permitted=True,
            notes="Referenced as frozen provenance only; not used analytically.",
        )
        return path
    if not permitted:
        log.record(
            path,
            source_role=role,
            validation_or_holdout="holdout",
            loaded=False,
            permitted=False,
            notes="Blocked before load.",
        )
        raise RuntimeError(f"Batch C forbids analytical use of holdout/Batch-B outcomes: {path}")
    return path


def load_permitted(
    path: Path | str,
    *,
    role: str,
    log: SourceAccessLog,
    loader,
    validation_or_holdout: str = "validation",
    notes: str = "",
):
    path = assert_permitted_analytical_load(path, role=role, log=log)
    value = loader(path)
    log.record(
        path,
        source_role=role,
        validation_or_holdout=validation_or_holdout,
        loaded=True,
        permitted=True,
        notes=notes,
    )
    return value


def verify_batch_a_scores_unchanged(batch_a_dir: Path, freeze: FreezeVerificationResult) -> Dict[str, Any]:
    primary = pd.read_csv(Path(batch_a_dir) / "audit_validation_decisions_primary_5bps.csv")
    counts = primary["operational_recommendation"].value_counts().to_dict()
    ok = freeze.ok and not freeze.changed_files
    return {
        "freeze_ok": bool(freeze.ok),
        "hash_changes": freeze.changed_files,
        "decision_counts": counts,
        "scores_present": "diagnostic_score" in primary.columns,
        "n_families": int(primary["model_family"].nunique()),
        "ok": ok,
    }


def read_batch_b_verification_verdict(verification_dir: Path) -> Dict[str, Any]:
    verification_dir = Path(verification_dir)
    report = verification_dir / "BATCH_B_VERIFICATION_REPORT.md"
    verdict = "UNKNOWN"
    if report.exists():
        text = report.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            if "VERDICT" in line.upper() or "VERIFIED" in line.upper():
                if "VERIFIED WITH WARNINGS" in line.upper():
                    verdict = "VERIFIED WITH WARNINGS"
                    break
                if "VERIFIED" in line.upper():
                    verdict = "VERIFIED"
                    break
                if "FAILED" in line.upper():
                    verdict = "FAILED"
                    break
        if verdict == "UNKNOWN" and "VERIFIED WITH WARNINGS" in text.upper():
            verdict = "VERIFIED WITH WARNINGS"
    return {
        "verification_dir": str(verification_dir),
        "report_path": str(report) if report.exists() else "",
        "verdict": verdict,
    }


def snapshot_batch_b_hashes(batch_b_dir: Path) -> Dict[str, str]:
    batch_b_dir = Path(batch_b_dir)
    out: Dict[str, str] = {}
    if not batch_b_dir.exists():
        return out
    for path in sorted(batch_b_dir.rglob("*")):
        if path.is_file():
            out[str(path)] = sha256_file(path)
    return out


def run_phase1_guards(cfg: ValidityConfig, log: SourceAccessLog) -> Dict[str, Any]:
    freeze = verify_batch_a_freeze(cfg.batch_a_dir, repo_root=cfg.repository_root)
    scores = verify_batch_a_scores_unchanged(cfg.batch_a_dir, freeze)
    log.record(
        cfg.batch_a_dir / "BATCH_A_FROZEN_MANIFEST.json",
        source_role="batch_a_frozen_manifest",
        validation_or_holdout="validation",
        loaded=True,
        permitted=True,
        notes="Hash verification only.",
    )
    log.record(
        cfg.batch_a_dir / "audit_validation_decisions_primary_5bps.csv",
        source_role="batch_a_primary_decisions",
        validation_or_holdout="validation",
        loaded=True,
        permitted=True,
        notes="Loaded for overlay metadata; scores not modified.",
    )
    batch_b_verdict = read_batch_b_verification_verdict(cfg.batch_b_verification_dir)
    log.record(
        cfg.batch_b_verification_dir / "BATCH_B_VERIFICATION_REPORT.md",
        source_role="batch_b_verification_verdict",
        validation_or_holdout="provenance",
        loaded=True,
        permitted=True,
        notes="Verdict recorded as provenance; Batch B outcomes not used analytically.",
    )
    batch_b_hashes = snapshot_batch_b_hashes(cfg.batch_b_dir)
    return {
        "freeze": freeze.to_dict(),
        "scores": scores,
        "batch_b_verification": batch_b_verdict,
        "batch_b_file_count": len(batch_b_hashes),
        "batch_b_hashes": batch_b_hashes,
    }
