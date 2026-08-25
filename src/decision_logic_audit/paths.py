"""Cache, repository, and environment-path helpers.

None of these helpers participate in attribution, scoring, or model equations.
"""

from __future__ import annotations

import os
from pathlib import Path

CACHE_ENV = "DECISION_LOGIC_AUDIT_CACHE"
CACHE_SCHEMA_VERSION = "v1"
DEFAULT_CACHE_NAME = "decision-logic-audit"


def env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else default


def cache_root() -> Path:
    override = os.environ.get(CACHE_ENV)
    if override:
        return Path(override).expanduser().resolve()
    return (Path.home() / ".cache" / DEFAULT_CACHE_NAME).resolve()


def cache_version_root() -> Path:
    return cache_root() / CACHE_SCHEMA_VERSION


def repository_root() -> Path | None:
    """Return the source checkout root when the tree is available."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists() and (parent / "src").is_dir():
            return parent
    return None
