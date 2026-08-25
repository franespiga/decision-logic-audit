#!/usr/bin/env python
"""Thin wrapper around the packaged cross-architecture comparison runner."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "src"
if src.is_dir() and str(src) not in sys.path:
    sys.path.insert(0, str(src))

from ablation.cross_architecture.run import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
