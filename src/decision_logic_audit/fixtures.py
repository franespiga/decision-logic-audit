"""Figshare fixture manifest, download, cache, and SHA-256 verification."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from .paths import cache_version_root, repository_root

FIGSHARE_API = "https://api.figshare.com/v2/articles/{article_id}"
CHUNK_SIZE = 1024 * 1024


class FixtureError(RuntimeError):
    """Raised when a fixture cannot be located, downloaded, or verified."""


@dataclass(frozen=True)
class FixtureSpec:
    logical_id: str
    figshare_filename: str
    cache_relative_path: str
    sha256: str
    loader: str
    role: str
    model_class: str | None = None

    @property
    def cache_path(self) -> Path:
        return cache_version_root() / Path(self.cache_relative_path)


def sha256_file(path: Path, chunk_size: int = CHUNK_SIZE) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest().upper()


def _candidate_manifest_paths() -> list[Path]:
    paths: list[Path] = []
    here = Path(__file__).resolve().parent
    paths.append(here / "data" / "manifest.yaml")
    repo = repository_root()
    if repo is not None:
        paths.append(repo / "fixtures" / "manifest.yaml")
    env = os.environ.get("DECISION_LOGIC_AUDIT_MANIFEST")
    if env:
        paths.insert(0, Path(env).expanduser())
    return paths


def manifest_path() -> Path:
    for candidate in _candidate_manifest_paths():
        if candidate.is_file():
            return candidate
    raise FixtureError(
        "Fixture manifest not found. Expected package data "
        "`decision_logic_audit/data/manifest.yaml` or `fixtures/manifest.yaml` "
        "in a source checkout."
    )


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    target = path or manifest_path()
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FixtureError(f"Manifest {target} is not a mapping")
    return payload


def fixture_specs(manifest: Mapping[str, Any] | None = None) -> list[FixtureSpec]:
    payload = dict(manifest or load_manifest())
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise FixtureError("Manifest is missing a non-empty 'files' list")
    specs: list[FixtureSpec] = []
    for row in files:
        specs.append(
            FixtureSpec(
                logical_id=str(row["logical_id"]),
                figshare_filename=str(row["figshare_filename"]),
                cache_relative_path=str(row["cache_relative_path"]),
                sha256=str(row["sha256"]).upper(),
                loader=str(row.get("loader", "")),
                role=str(row.get("role", "")),
                model_class=row.get("model_class"),
            )
        )
    return specs


def bundled_metadata_path(logical_id: str) -> Path:
    filename = f"{logical_id}.json"
    here = Path(__file__).resolve().parent / "data" / "metadata" / filename
    if here.is_file():
        return here
    repo = repository_root()
    if repo is not None:
        candidate = repo / "fixtures" / "metadata" / filename
        if candidate.is_file():
            return candidate
    raise FixtureError(
        f"Sanitized metadata for {logical_id!r} is not bundled with the software."
    )


def verify_file(path: Path, expected_sha256: str) -> str:
    if not path.is_file():
        raise FixtureError(f"Missing fixture file: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256.upper():
        raise FixtureError(
            f"SHA-256 mismatch for {path.name}: expected {expected_sha256.upper()}, "
            f"got {actual}. The file is treated as corrupt and must not be used."
        )
    return actual


def _atomic_replace(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.replace(src, dest)


def _copy_verified(src: Path, dest: Path, expected_sha256: str, force: bool) -> str:
    if dest.is_file() and not force:
        return verify_file(dest, expected_sha256)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=dest.name + ".", suffix=".part", dir=dest.parent)
    tmp = Path(tmp_name)
    os.close(fd)
    try:
        shutil.copyfile(src, tmp)
        actual = verify_file(tmp, expected_sha256)
        _atomic_replace(tmp, dest)
        return actual
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def _stream_download(url: str, dest_part: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "decision-logic-audit/1.0.0"})
    with urllib.request.urlopen(request, timeout=120) as response, dest_part.open("wb") as handle:
        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break
            handle.write(chunk)


def query_figshare_files(article_id: int) -> list[dict[str, Any]]:
    url = FIGSHARE_API.format(article_id=int(article_id))
    request = urllib.request.Request(url, headers={"User-Agent": "decision-logic-audit/1.0.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise FixtureError(
                f"Figshare article {article_id} is not publicly available (HTTP 404). "
                "Publish the fixture record, or stage files with "
                "`decision-logic-audit fetch-fixtures --from-dir PATH`."
            ) from exc
        raise FixtureError(f"Figshare API request failed: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise FixtureError(
            "Could not reach the Figshare API. Check network access, or stage "
            "files offline with `--from-dir`."
        ) from exc
    files = payload.get("files")
    if not isinstance(files, list):
        raise FixtureError(f"Figshare article {article_id} did not return a file list")
    return files


def _download_url_for(filename: str, files: Sequence[Mapping[str, Any]]) -> str:
    for row in files:
        if str(row.get("name")) == filename:
            url = row.get("download_url") or row.get("url")
            if not url:
                raise FixtureError(f"Figshare file {filename!r} has no download URL")
            return str(url)
    available = ", ".join(sorted(str(r.get("name")) for r in files)) or "(none)"
    raise FixtureError(
        f"Figshare record does not contain {filename!r}. Available filenames: {available}. "
        "Update fixtures/manifest.yaml to match the published record."
    )


def fetch_one(
    spec: FixtureSpec,
    *,
    article_id: int,
    figshare_files: Sequence[Mapping[str, Any]] | None = None,
    from_dir: Path | None = None,
    force: bool = False,
) -> dict[str, str]:
    dest = spec.cache_path
    if dest.is_file() and not force:
        actual = verify_file(dest, spec.sha256)
        return {
            "logical_id": spec.logical_id,
            "status": "cached",
            "path": str(dest),
            "sha256": actual,
        }

    if from_dir is not None:
        source = Path(from_dir) / spec.figshare_filename
        if not source.is_file():
            raise FixtureError(
                f"Local staging directory {from_dir} does not contain {spec.figshare_filename}. "
                "Place files using the exact Figshare filenames from the manifest."
            )
        actual = _copy_verified(source, dest, spec.sha256, force=True)
        return {
            "logical_id": spec.logical_id,
            "status": "copied",
            "path": str(dest),
            "sha256": actual,
        }

    files = list(figshare_files) if figshare_files is not None else query_figshare_files(article_id)
    url = _download_url_for(spec.figshare_filename, files)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=dest.name + ".", suffix=".part", dir=dest.parent)
    tmp = Path(tmp_name)
    os.close(fd)
    try:
        _stream_download(url, tmp)
        actual = verify_file(tmp, spec.sha256)
        _atomic_replace(tmp, dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        if dest.exists():
            dest.unlink()
        raise
    return {
        "logical_id": spec.logical_id,
        "status": "downloaded",
        "path": str(dest),
        "sha256": actual,
    }


def fetch_all(
    *,
    force: bool = False,
    from_dir: Path | None = None,
    manifest: Mapping[str, Any] | None = None,
) -> list[dict[str, str]]:
    payload = dict(manifest or load_manifest())
    article_id = int(payload["figshare_article_id"])
    specs = fixture_specs(payload)
    if from_dir is None and not force and all(spec.cache_path.is_file() for spec in specs):
        return [
            fetch_one(spec, article_id=article_id, from_dir=None, force=False)
            for spec in specs
        ]
    remote_files = None if from_dir is not None else query_figshare_files(article_id)
    results = []
    for spec in specs:
        results.append(
            fetch_one(
                spec,
                article_id=article_id,
                figshare_files=remote_files,
                from_dir=from_dir,
                force=force,
            )
        )
    return results


def require_cached(manifest: Mapping[str, Any] | None = None) -> dict[str, Path]:
    payload = dict(manifest or load_manifest())
    out: dict[str, Path] = {}
    for spec in fixture_specs(payload):
        verify_file(spec.cache_path, spec.sha256)
        out[spec.logical_id] = spec.cache_path
    return out


def print_results(rows: Iterable[Mapping[str, str]], stream=None) -> None:
    stream = stream or sys.stdout
    for row in rows:
        stream.write(
            f"{row['status']:10} {row['logical_id']:32} {row['sha256']}  {row['path']}\n"
        )
