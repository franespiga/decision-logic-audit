"""I/O helpers for Batch A audit scoring."""

from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import yaml

from .config import FORBIDDEN_HOLDOUT_BASENAMES

LOGGER = logging.getLogger("audit_scoring.io")

_NUMPY2_SHIM_INSTALLED = False


def install_numpy2_pickle_shim() -> None:
    """Allow loading NumPy-2 pickles under a NumPy-1.x runtime."""
    global _NUMPY2_SHIM_INSTALLED
    if _NUMPY2_SHIM_INSTALLED:
        return
    import numpy.core as nc

    sys.modules.setdefault("numpy._core", nc)
    aliases = {
        "numeric": "numeric",
        "multiarray": "multiarray",
        "umath": "umath",
        "_dtype_ctypes": "_dtype_ctypes",
        "_internal": "_internal",
        "fromnumeric": "fromnumeric",
        "shape_base": "shape_base",
        "overrides": "overrides",
        "_multiarray_umath": "_multiarray_umath",
    }
    for suffix, attr in aliases.items():
        mod = getattr(nc, attr, None)
        if mod is not None:
            sys.modules.setdefault(f"numpy._core.{suffix}", mod)
    _NUMPY2_SHIM_INSTALLED = True


def assert_not_holdout_path(path: Path | str) -> Path:
    """Raise if a forbidden holdout metrics basename is requested."""
    p = Path(path)
    stem = p.name
    for banned in FORBIDDEN_HOLDOUT_BASENAMES:
        if stem.startswith(banned):
            raise RuntimeError(
                f"Holdout metrics file is forbidden during Batch A: {p}. "
                "Batch A may use validation/calibration information only."
            )
    return p


def load_pickle(path: Path | str) -> Any:
    """Load a trusted local pickle with cloudpickle when available."""
    path = Path(path)
    install_numpy2_pickle_shim()
    try:
        import cloudpickle
    except ImportError:  # pragma: no cover
        cloudpickle = None
    with path.open("rb") as handle:
        if cloudpickle is not None:
            return cloudpickle.load(handle)
        return pickle.load(handle)


def to_numpy(value: Any) -> np.ndarray:
    """Detach tensors, move to CPU, and return a NumPy array."""
    if value is None:
        return np.asarray([])
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        try:
            return np.asarray(value.numpy())
        except Exception:
            return np.asarray(value)
    return np.asarray(value)


def scalar_float(value: Any) -> float:
    if value is None:
        return float("nan")
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:
            pass
    arr = np.asarray(value)
    if arr.size == 0:
        return float("nan")
    if arr.size == 1:
        return float(arr.reshape(-1)[0])
    return float(np.nanmean(arr))


def discover_metric_file(metrics_dir: Path, basename: str) -> Path:
    """Discover CSV/Parquet for an allowed validation metrics basename."""
    assert_not_holdout_path(basename)
    metrics_dir = Path(metrics_dir)
    candidates = [
        metrics_dir / f"{basename}.csv",
        metrics_dir / f"{basename}.parquet",
        metrics_dir / f"{basename}.pkl",
    ]
    for path in candidates:
        if path.exists():
            assert_not_holdout_path(path)
            return path
    raise FileNotFoundError(f"No metrics file found for basename={basename!r} under {metrics_dir}")


def read_table(path: Path | str) -> pd.DataFrame:
    path = assert_not_holdout_path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".pkl", ".pickle"}:
        obj = load_pickle(path)
        if isinstance(obj, pd.DataFrame):
            return obj
        raise TypeError(f"Pickle at {path} is not a DataFrame")
    raise ValueError(f"Unsupported table format: {path}")


def write_table(df: pd.DataFrame, path: Path, also_parquet: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    if also_parquet:
        parquet_path = path.with_suffix(".parquet")
        try:
            df.to_parquet(parquet_path, index=False)
        except Exception:
            # Fall back to stringified object columns for mixed-type long tables.
            coerced = df.copy()
            for col in coerced.columns:
                if coerced[col].dtype == object:
                    coerced[col] = coerced[col].map(
                        lambda x: "" if x is None or (isinstance(x, float) and pd.isna(x)) else str(x)
                    )
            coerced.to_parquet(parquet_path, index=False)


def write_json(payload: Mapping[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(obj: Any) -> Any:
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if pd.isna(obj):
            return None
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_default)


def write_yaml(payload: Mapping[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(payload), handle, sort_keys=False)


def ensure_output_dir(path: Path, force: bool = True) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if not force:
        existing = list(path.glob("audit_*.csv")) + list(path.glob("BATCH_A_REPORT.md"))
        if existing:
            LOGGER.info(
                "Output directory already contains Batch A artifacts; "
                "pass --force to overwrite (%s files).",
                len(existing),
            )
    return path


def discover_diagnostic_pkls(root: Path) -> List[Path]:
    """Recursively discover model_diagnostics_*.pkl files."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Diagnostic PKL root does not exist: {root}")
    paths = sorted(root.rglob("model_diagnostics_*.pkl"))
    return paths


def path_has_partial(path: Path, root: Path) -> bool:
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = Path(*path.parts)
    return "partial" in {p.lower() for p in rel.parts}


def technical_indicator_flag_from_path(path: Path) -> Optional[bool]:
    parts = {p.lower() for p in Path(path).parts}
    if "techindi" in parts and "no_techindi" not in parts:
        return True
    if "no_techindi" in parts:
        return False
    text = str(path).lower().replace("\\", "/")
    if "/no_techindi/" in text:
        return False
    if "/techindi/" in text:
        return True
    return None
