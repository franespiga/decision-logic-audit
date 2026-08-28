"""Public console entry point: ``decision-logic-audit``."""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

from . import __version__
from .fixtures import FixtureError, fetch_all, load_manifest, print_results
from .paths import cache_root, cache_version_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="decision-logic-audit",
        description=(
            "Fetch public fixtures and run a bounded cross-architecture "
            "diagnostic demonstration. This is research software, not an "
            "investment tool."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch-fixtures", help="download or stage the public fixture bundle")
    fetch.add_argument("--force", action="store_true", help="re-download even if a cached file verifies")
    fetch.add_argument(
        "--from-dir",
        type=Path,
        default=None,
        help="copy Figshare-named files from a local directory instead of downloading",
    )

    demo = sub.add_parser(
        "demo",
        help="run the bounded pilot demonstration (not a paper reproduction)",
    )
    demo.add_argument("--output-dir", type=Path, default=None)
    demo.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    demo.add_argument("--skip-fetch", action="store_true", help="require an already-verified cache")
    demo.add_argument("--from-dir", type=Path, default=None, help="stage fixtures from a local directory first")
    demo.add_argument("--force-fetch", action="store_true")

    sub.add_parser("diagnose", help="print environment and cache diagnostics")
    return parser


def _cmd_fetch(args: argparse.Namespace) -> int:
    try:
        rows = fetch_all(force=args.force, from_dir=args.from_dir)
    except FixtureError as exc:
        print(exc, file=sys.stderr)
        return 1
    print_results(rows)
    print(f"cache: {cache_version_root()}")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    from .demo import run_demo

    return run_demo(
        output_dir=args.output_dir,
        device=args.device,
        skip_fetch=args.skip_fetch,
        from_dir=args.from_dir,
        force_fetch=args.force_fetch,
    )


def _cmd_diagnose(_args: argparse.Namespace) -> int:
    print(f"decision-logic-audit {__version__}")
    print(f"python: {sys.version.split()[0]} ({platform.system()} {platform.machine()})")
    print(f"executable: {sys.executable}")
    print(f"cache_root: {cache_root()}")
    print(f"cache_version_root: {cache_version_root()}")
    try:
        manifest = load_manifest()
        print(f"manifest: loaded (schema {manifest.get('schema_version')})")
        print(f"figshare_article_id: {manifest.get('figshare_article_id')}")
        print(f"figshare_doi: {manifest.get('figshare_doi')}")
    except Exception as exc:  # noqa: BLE001
        print(f"manifest: {exc}")
    try:
        import torch

        print(f"torch: {torch.__version__} cuda={torch.cuda.is_available()}")
    except Exception as exc:  # noqa: BLE001
        print(f"torch: not importable ({exc})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "fetch-fixtures":
        return _cmd_fetch(args)
    if args.command == "demo":
        return _cmd_demo(args)
    if args.command == "diagnose":
        return _cmd_diagnose(args)
    parser.error("unknown command")
    return 2
