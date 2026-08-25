"""Materialize a discovery-friendly runtime layout from the verified cache."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from .fixtures import bundled_metadata_path, load_manifest, require_cached
from .paths import cache_version_root


def _link_or_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def runtime_root() -> Path:
    return cache_version_root() / "runtime"


def materialize_runtime_layout(cached: dict[str, Path] | None = None) -> dict[str, Path]:
    """Create the directory shape expected by ``ablation.cross_architecture.discovery``.

    Binary bytes are hard-linked or copied unchanged. Sanitized metadata JSON
    shipped with the software is used in place of the original files that
    contained researcher home-directory paths.
    """
    files = cached or require_cached()
    root = runtime_root()
    dt_root = root / "dt" / "max_quadratic_utility_TI_True"
    rl_root = root / "rl" / "PPO_16_1_TI_True"
    model_dir = dt_root / "out" / "checkpoint" / "model"

    _link_or_copy(files["max_quadratic_utility_TI_True.config"], model_dir / "config.json")
    _link_or_copy(files["max_quadratic_utility_TI_True.weights"], model_dir / "model.safetensors")
    _link_or_copy(files["aligned_market_data"], dt_root / "data.h5")
    shutil.copy2(
        bundled_metadata_path("max_quadratic_utility_TI_True"),
        dt_root / "out" / "checkpoint" / "metadata.json",
    )

    _link_or_copy(files["PPO_16_1_TI_True.weights"], rl_root / "best_model.zip")
    shutil.copy2(bundled_metadata_path("PPO_16_1_TI_True"), rl_root / "metadata.json")

    return {
        "dt_model_dir": root / "dt",
        "rl_model_dir": root / "rl",
        "data_path": dt_root / "data.h5",
        "dt_checkpoint": model_dir,
        "ppo_checkpoint": rl_root / "best_model.zip",
        "runtime_root": root,
    }


def describe_layout() -> dict[str, str]:
    payload = load_manifest()
    return {
        "cache_root": str(cache_version_root()),
        "runtime_root": str(runtime_root()),
        "figshare_article_id": str(payload.get("figshare_article_id")),
        "figshare_doi": str(payload.get("figshare_doi")),
    }
