"""Loader tests against the verified public fixture cache.

Skipped when the Figshare/cache files are absent so unit CI stays fixture-free.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from decision_logic_audit.fixtures import FixtureError, load_manifest, require_cached, sha256_file
from decision_logic_audit.layout import materialize_runtime_layout


def _cached_or_skip():
    try:
        return require_cached()
    except FixtureError as exc:
        pytest.skip(str(exc))


def test_cached_fixture_hashes_match_manifest():
    payload = load_manifest()
    cached = _cached_or_skip()
    expected = {row["logical_id"]: row["sha256"].upper() for row in payload["files"]}
    for logical_id, path in cached.items():
        assert sha256_file(path) == expected[logical_id]


def test_dt_ppo_and_hdf5_load_from_cache_layout():
    cached = _cached_or_skip()
    layout = materialize_runtime_layout(cached)

    from ablation.cross_architecture import data as market
    from models.model import TrainableDT
    from stable_baselines3 import PPO

    md = market.load_market_data(layout["data_path"])
    assert md.merged.shape[0] > 100
    assert ("^GSPC", "Close") in md.columns or ("^GSPC", "Close") in list(md.prices.columns)

    dt = TrainableDT.from_pretrained(str(layout["dt_checkpoint"]))
    assert int(dt.config.state_dim) == 1122
    assert int(dt.config.act_dim) == 7

    ppo = PPO.load(str(layout["ppo_checkpoint"]), device="cpu")
    assert ppo is not None
    n_actions = getattr(ppo.action_space, "n", None)
    assert n_actions == 7
