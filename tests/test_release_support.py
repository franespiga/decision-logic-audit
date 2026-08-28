"""Release-support tests: manifest, checksums, cache behaviour, imports."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from decision_logic_audit.fixtures import (
    FixtureError,
    _download_url_for,
    embargo_or_unpublished_message,
    fetch_all,
    fixture_specs,
    load_manifest,
    query_figshare_files,
    sha256_file,
    verify_file,
)
from decision_logic_audit.paths import CACHE_ENV


def test_manifest_parses_and_has_required_fields():
    payload = load_manifest()
    assert payload["schema_version"] == "1.0.0"
    assert payload["fixture_license"] == "CC-BY-4.0"
    assert payload["figshare_article_id"] == 33331437
    specs = fixture_specs(payload)
    ids = {s.logical_id for s in specs}
    assert "max_quadratic_utility_TI_True.weights" in ids
    assert "PPO_16_1_TI_True.weights" in ids
    assert "aligned_market_data" in ids
    for spec in specs:
        assert len(spec.sha256) == 64
        assert spec.figshare_filename
        assert spec.cache_relative_path
    folders = {s.logical_id: s.figshare_folder for s in specs}
    assert folders["max_quadratic_utility_TI_True.weights"] == "fixtures/decision_transformer"
    assert folders["PPO_16_1_TI_True.weights"] == "fixtures/ppo"
    assert folders["aligned_market_data"] == "fixtures/decision_transformer"


def test_sha256_matches_known_bytes(tmp_path: Path):
    path = tmp_path / "blob.bin"
    payload = b"decision-logic-audit-fixture-test"
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest().upper()
    assert sha256_file(path) == digest
    assert verify_file(path, digest) == digest


def test_bad_checksum_is_refused(tmp_path: Path):
    path = tmp_path / "blob.bin"
    path.write_bytes(b"not-the-expected-bytes")
    with pytest.raises(FixtureError, match="SHA-256 mismatch"):
        verify_file(path, "0" * 64)


def test_from_dir_staging_and_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cache = tmp_path / "cache"
    monkeypatch.setenv(CACHE_ENV, str(cache))
    payload = load_manifest()
    specs = fixture_specs(payload)
    staging = tmp_path / "staging"
    staging.mkdir()
    local = dict(payload)
    local_files = []
    for spec in specs:
        body = f"payload-for-{spec.logical_id}".encode("utf-8")
        digest = hashlib.sha256(body).hexdigest().upper()
        (staging / spec.figshare_filename).write_bytes(body)
        local_files.append(
            {
                "logical_id": spec.logical_id,
                "figshare_filename": spec.figshare_filename,
                "figshare_folder": spec.figshare_folder,
                "cache_relative_path": spec.cache_relative_path,
                "sha256": digest,
                "loader": spec.loader,
                "role": spec.role,
                "model_class": spec.model_class,
            }
        )
    local["files"] = local_files
    results = fetch_all(from_dir=staging, manifest=local)
    assert {r["status"] for r in results} == {"copied"}
    again = fetch_all(from_dir=staging, manifest=local)
    assert {r["status"] for r in again} == {"cached"}
    first = fixture_specs(local)[0]
    corrupt = cache / "v1" / Path(first.cache_relative_path)
    corrupt.write_bytes(b"corrupt")
    with pytest.raises(FixtureError, match="SHA-256 mismatch"):
        fetch_all(from_dir=staging, manifest=local)


def test_duplicate_figshare_basename_uses_folder():
    files = [
        {
            "name": "data.h5",
            "folder": "fixtures/decision_transformer",
            "download_url": "https://example.test/nested",
        },
        {
            "name": "data.h5",
            "folder": "",
            "download_url": "https://example.test/root",
        },
    ]
    assert _download_url_for("data.h5", files, folder="") == "https://example.test/root"
    assert (
        _download_url_for("data.h5", files, folder="fixtures/decision_transformer")
        == "https://example.test/nested"
    )


def test_from_dir_nested_figshare_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cache = tmp_path / "cache"
    monkeypatch.setenv(CACHE_ENV, str(cache))
    payload = load_manifest()
    specs = fixture_specs(payload)
    staging = tmp_path / "staging"
    local = dict(payload)
    local_files = []
    for spec in specs:
        body = f"nested-{spec.logical_id}".encode("utf-8")
        digest = hashlib.sha256(body).hexdigest().upper()
        dest = staging / spec.figshare_folder / spec.figshare_filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(body)
        local_files.append(
            {
                "logical_id": spec.logical_id,
                "figshare_filename": spec.figshare_filename,
                "figshare_folder": spec.figshare_folder,
                "cache_relative_path": spec.cache_relative_path,
                "sha256": digest,
                "loader": spec.loader,
                "role": spec.role,
                "model_class": spec.model_class,
            }
        )
    local["files"] = local_files
    results = fetch_all(from_dir=staging, manifest=local)
    assert {r["status"] for r in results} == {"copied"}


def test_figshare_embargo_message_points_to_doi_and_from_dir():
    text = embargo_or_unpublished_message(33331437, http_code=404)
    assert "10.6084/m9.figshare.33331437" in text
    assert "under embargo" in text
    assert "--from-dir" in text


def test_figshare_api_404_is_reported_as_embargo(monkeypatch: pytest.MonkeyPatch):
    import io
    import urllib.error
    import urllib.request

    def _raise(_request, timeout=60):  # noqa: ARG001
        raise urllib.error.HTTPError(
            url="https://api.figshare.com/v2/articles/33331437",
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=io.BytesIO(b""),
        )

    monkeypatch.setattr(urllib.request, "urlopen", _raise)
    with pytest.raises(FixtureError, match="under embargo"):
        query_figshare_files(33331437)


def test_figshare_embargoed_payload_without_download_urls(monkeypatch: pytest.MonkeyPatch):
    import json
    import urllib.request

    class _Response:
        def read(self) -> bytes:
            return json.dumps(
                {
                    "is_embargoed": True,
                    "files": [{"name": "aligned_market_data.h5"}],
                }
            ).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Response())
    with pytest.raises(FixtureError, match="under embargo"):
        query_figshare_files(33331437)


def test_cli_help_parsers():
    from decision_logic_audit.cli import build_parser

    parser = build_parser()
    help_text = parser.format_help()
    assert "fetch-fixtures" in help_text
    assert "demo" in help_text
    sub = parser._subparsers._group_actions[0].choices
    assert "from-dir" in sub["fetch-fixtures"].format_help()
    assert "--output-dir" in sub["demo"].format_help()



def test_scientific_packages_importable():
    import ablation
    import audit_scoring
    import models
    import regime_representation
    import rl_factory_shim

    assert ablation.__name__ == "ablation"
    assert audit_scoring.__name__ == "audit_scoring"
    assert models.__name__ == "models"
    assert regime_representation.__name__ == "regime_representation"
    assert rl_factory_shim.__name__ == "rl_factory_shim"
