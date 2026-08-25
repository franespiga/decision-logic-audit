"""Batch B: temporally separated holdout assessment of frozen Batch A decisions.

Diagnostics remain frozen. This module never recalibrates Batch A thresholds,
weights, bands, or recommendations.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

LOGGER = logging.getLogger("audit_scoring.holdout_assessment")

INCOMPLETE_EVIDENCE_FAMILY = "no_techindi_full_TI_False"
EXPECTED_N_FAMILIES = 18
EXPECTED_VARIANTS_PER_FAMILY = 96
EXPECTED_DECISION_COUNTS = {"ADVANCE": 0, "MONITOR": 4, "INVESTIGATE": 7, "REJECT": 7}
FORBIDDEN_COMBINED_PREFIX = "validation_holdout_metrics_"


@dataclass
class FreezeVerificationResult:
    ok: bool
    git_commit_stored: str
    git_commit_current: str
    working_tree_dirty: bool
    primary_cost_bps: int
    audit_profile: str
    decision_counts: Dict[str, int]
    provisional_count: int
    incomplete_family_ok: bool
    changed_files: List[Dict[str, str]] = field(default_factory=list)
    messages: List[str] = field(default_factory=list)
    manifest: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "git_commit_stored": self.git_commit_stored,
            "git_commit_current": self.git_commit_current,
            "working_tree_dirty": self.working_tree_dirty,
            "primary_cost_bps": self.primary_cost_bps,
            "audit_profile": self.audit_profile,
            "decision_counts": self.decision_counts,
            "provisional_count": self.provisional_count,
            "incomplete_family_ok": self.incomplete_family_ok,
            "changed_files": self.changed_files,
            "messages": self.messages,
        }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def _git_dirty(repo_root: Path) -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return bool(out.strip())
    except Exception:
        return False


def load_manifest(batch_a_dir: Path) -> Dict[str, Any]:
    path = Path(batch_a_dir) / "BATCH_A_FROZEN_MANIFEST.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing Batch A freeze manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_batch_a_freeze(
    batch_a_dir: Path,
    *,
    repo_root: Optional[Path] = None,
    expected_decision_counts: Mapping[str, int] = EXPECTED_DECISION_COUNTS,
) -> FreezeVerificationResult:
    """Verify Batch A artefact hashes and frozen decision properties before holdout load."""
    batch_a_dir = Path(batch_a_dir)
    repo_root = Path(repo_root) if repo_root else batch_a_dir.parents[3] if len(batch_a_dir.parts) > 3 else Path.cwd()
    # Prefer repository root containing src/
    for candidate in [Path(r"C:\git\tesis\DT_ABLATION_RISK"), batch_a_dir.parent.parent.parent.parent]:
        if (candidate / "src" / "audit_scoring").exists():
            repo_root = candidate
            break

    manifest = load_manifest(batch_a_dir)
    messages: List[str] = []
    changed: List[Dict[str, str]] = []

    for section in ("input_files", "output_files"):
        entries = manifest.get(section, [])
        for entry in entries:
            path = Path(entry["path"])
            stored = str(entry["sha256"]).lower()
            if not path.exists():
                changed.append({"path": str(path), "stored_sha256": stored, "current_sha256": "MISSING"})
                continue
            current = sha256_file(path).lower()
            if current != stored:
                changed.append(
                    {
                        "path": str(path),
                        "stored_sha256": stored,
                        "current_sha256": current,
                    }
                )

    primary = pd.read_csv(batch_a_dir / "audit_validation_decisions_primary_5bps.csv")
    counts = primary["operational_recommendation"].value_counts().to_dict()
    for key in expected_decision_counts:
        counts.setdefault(key, 0)
    counts_ok = all(int(counts.get(k, 0)) == int(v) for k, v in expected_decision_counts.items())
    if not counts_ok:
        messages.append(f"Decision counts mismatch: {counts} vs expected {dict(expected_decision_counts)}")

    provisional = True
    if "decision_reason" in primary.columns:
        provisional = bool(primary["decision_reason"].astype(str).str.contains("provisional").all())
    elif "provisional" in primary.columns:
        provisional = bool(primary["provisional"].astype(bool).all())
    provisional_count = int(len(primary)) if provisional else int(
        primary["decision_reason"].astype(str).str.contains("provisional").sum()
        if "decision_reason" in primary.columns
        else 0
    )
    if provisional_count != 18:
        messages.append(f"Expected 18 provisional decisions; found {provisional_count}")

    incomplete_row = primary[primary["model_family"] == INCOMPLETE_EVIDENCE_FAMILY]
    incomplete_ok = False
    if incomplete_row.empty:
        messages.append(f"Missing incomplete-evidence family {INCOMPLETE_EVIDENCE_FAMILY}")
    else:
        rec = str(incomplete_row.iloc[0]["operational_recommendation"])
        band = str(incomplete_row.iloc[0].get("diagnostic_band", ""))
        score = incomplete_row.iloc[0].get("diagnostic_score")
        incomplete_ok = (
            rec == "REJECT"
            and band == "POOR"
            and (pd.isna(score) or str(score).lower() in {"", "nan", "none"})
        )
        if not incomplete_ok:
            messages.append(
                f"{INCOMPLETE_EVIDENCE_FAMILY} freeze properties violated: "
                f"decision={rec}, band={band}, score={score}"
            )

    primary_cost = int(manifest.get("primary_cost_level_bps", -1))
    profile = str(manifest.get("audit_profile", ""))
    if primary_cost != 5:
        messages.append(f"primary_cost_level_bps={primary_cost}, expected 5")
    if profile != "balanced":
        messages.append(f"audit_profile={profile!r}, expected 'balanced'")

    stored_commit = str(manifest.get("git_commit", ""))
    current_commit = _git_commit(repo_root)
    dirty = _git_dirty(repo_root)
    if stored_commit and current_commit != "unknown" and stored_commit != current_commit:
        messages.append(
            "Current Git commit differs from Batch A freeze commit; "
            "allowed when only Batch B implementation changed."
        )

    ok = len(changed) == 0 and counts_ok and incomplete_ok and primary_cost == 5 and profile == "balanced"
    if changed:
        messages.append(f"{len(changed)} Batch A artefact hash mismatch(es)")

    return FreezeVerificationResult(
        ok=ok,
        git_commit_stored=stored_commit,
        git_commit_current=current_commit,
        working_tree_dirty=dirty,
        primary_cost_bps=primary_cost,
        audit_profile=profile,
        decision_counts={k: int(counts.get(k, 0)) for k in expected_decision_counts},
        provisional_count=provisional_count,
        incomplete_family_ok=incomplete_ok,
        changed_files=changed,
        messages=messages,
        manifest=manifest,
    )


def write_blocked_report(path: Path, result: FreezeVerificationResult) -> None:
    lines = [
        "# Batch B blocked: Batch A artefacts changed",
        "",
        "Holdout metrics were **not** loaded.",
        "",
        f"- Stored Git commit: `{result.git_commit_stored}`",
        f"- Current Git commit: `{result.git_commit_current}`",
        f"- Working tree dirty: `{result.working_tree_dirty}`",
        "",
        "## Changed files",
        "",
    ]
    for row in result.changed_files:
        lines.append(f"- `{row['path']}`")
        lines.append(f"  - stored: `{row['stored_sha256']}`")
        lines.append(f"  - current: `{row['current_sha256']}`")
    lines.extend(["", "## Messages", ""])
    lines.extend(f"- {m}" for m in result.messages)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def assert_allowed_holdout_path(path: Path | str) -> Path:
    """Allow holdout_metrics_* but forbid combined validation_holdout_metrics_*."""
    p = Path(path)
    if p.name.startswith(FORBIDDEN_COMBINED_PREFIX):
        raise RuntimeError(
            f"Combined validation-holdout metrics file is forbidden in Batch B: {p}"
        )
    return p


def discover_metrics_table(metrics_dir: Path, basename: str) -> Path:
    metrics_dir = Path(metrics_dir)
    assert_allowed_holdout_path(basename)
    for ext in (".csv", ".parquet", ".pkl"):
        path = metrics_dir / f"{basename}{ext}"
        if path.exists():
            return assert_allowed_holdout_path(path)
    raise FileNotFoundError(f"No table for {basename!r} under {metrics_dir}")


def read_metrics_table(path: Path) -> pd.DataFrame:
    path = assert_allowed_holdout_path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported metrics format: {path}")


def parse_family_design_factors(model_family: str) -> Dict[str, str]:
    """Extract ex-ante design factors from family labels."""
    fam = str(model_family)
    ti = "True" if fam.endswith("_TI_True") else "False" if fam.endswith("_TI_False") else "unknown"
    stem = fam.replace("_TI_True", "").replace("_TI_False", "")
    lookback = "unknown"
    rebalance = "unknown"
    objective = stem
    if stem.startswith("DT_"):
        parts = stem.split("_")
        # DT_{LB}_{RE}
        if len(parts) >= 3 and parts[1].isdigit() and parts[2].isdigit():
            lookback = parts[1]
            rebalance = parts[2]
            objective = "dt_baseline"
        else:
            objective = "dt_other"
    elif stem.endswith("_full"):
        objective = "full_feature"
        lookback = "full"
        rebalance = "full"
    return {
        "technical_indicators": ti,
        "lookback": lookback,
        "rebalancing_frequency": rebalance,
        "objective_or_conditioning": objective,
        "family_stem": stem,
    }


def design_factor_summary(model_family: str) -> str:
    factors = parse_family_design_factors(model_family)
    return (
        f"TI={factors['technical_indicators']};"
        f"LB={factors['lookback']};"
        f"RE={factors['rebalancing_frequency']};"
        f"OBJ={factors['objective_or_conditioning']}"
    )


def load_frozen_batch_a(batch_a_dir: Path) -> Dict[str, Any]:
    batch_a_dir = Path(batch_a_dir)
    primary = pd.read_csv(batch_a_dir / "audit_validation_decisions_primary_5bps.csv")
    family_scored = pd.read_csv(batch_a_dir / "audit_family_diagnostics_scored.csv")
    master = pd.read_csv(batch_a_dir / "audit_validation_master_all_costs.csv")
    thresholds = json.loads((batch_a_dir / "audit_validation_thresholds.json").read_text(encoding="utf-8"))
    mapping = pd.read_csv(batch_a_dir / "audit_model_mapping.csv")
    raw = pd.read_csv(batch_a_dir / "audit_model_diagnostics_raw.csv")
    return {
        "primary": primary,
        "family_scored": family_scored,
        "master": master,
        "thresholds": thresholds,
        "mapping": mapping,
        "raw": raw,
    }


def drawdown_severity(drawdown: pd.Series) -> pd.Series:
    """Convert signed drawdown to severity (larger = worse)."""
    dd = pd.to_numeric(drawdown, errors="coerce")
    return (-dd).where(dd < 0, dd.abs())
