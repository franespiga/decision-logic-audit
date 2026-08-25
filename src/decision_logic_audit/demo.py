"""Convenience demo that calls the existing scientific comparison runner."""

from __future__ import annotations

import os
from pathlib import Path

from .fixtures import FixtureError, fetch_all, require_cached
from .layout import materialize_runtime_layout
from .paths import repository_root


def default_output_dir() -> Path:
    repo = repository_root()
    base = repo if repo is not None else Path.cwd()
    return base / "outputs" / "demo"


def run_demo(
    *,
    output_dir: Path | None = None,
    device: str = "cpu",
    skip_fetch: bool = False,
    from_dir: Path | None = None,
    force_fetch: bool = False,
    extra_args: list[str] | None = None,
) -> int:
    if not skip_fetch:
        try:
            fetch_all(force=force_fetch, from_dir=from_dir)
        except FixtureError as exc:
            raise SystemExit(str(exc)) from exc
    try:
        cached = require_cached()
        layout = materialize_runtime_layout(cached)
    except FixtureError as exc:
        raise SystemExit(
            f"{exc}\nRun `decision-logic-audit fetch-fixtures` first."
        ) from exc

    out = Path(output_dir) if output_dir else default_output_dir()
    out.mkdir(parents=True, exist_ok=True)
    argv = [
        "--dt-model-dir",
        str(layout["dt_model_dir"]),
        "--rl-model-dir",
        str(layout["rl_model_dir"]),
        "--data-path",
        str(layout["data_path"]),
        "--config",
        str(out / "comparison_manifest.yaml"),
        "--output-dir",
        str(out),
        "--period",
        "eval",
        "--seed",
        "42",
        "--device",
        device,
        "--pilot",
        "--skip-tests",
    ]
    if extra_args:
        argv.extend(extra_args)

    # Isolate the public demo from the original research repository.
    os.environ.pop("PYTHONPATH", None)
    from ablation.cross_architecture.run import main as comparison_main

    print("decision-logic-audit demo: smoke/pilot run (not a reproduction of paper values)")
    print(f"  DT dir:  {layout['dt_model_dir']}")
    print(f"  RL dir:  {layout['rl_model_dir']}")
    print(f"  data:    {layout['data_path']}")
    print(f"  output:  {out}")
    return int(comparison_main(argv))
