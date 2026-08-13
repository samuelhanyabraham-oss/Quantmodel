"""Logger round-trip: what goes in must come back out, append-only."""

import json

import pytest

from regime import experiment_log as xl


def test_log_run_round_trips(tmp_path):
    log = tmp_path / "experiments.jsonl"
    cfg = {"model": "dummy", "features": ["a", "b"], "embargo_days": 10}

    written = xl.log_run(
        cfg,
        {"auc": 0.5, "brier": 0.25},
        seed=42,
        data_snapshot_hash="deadbeef",
        notes="phase-0 smoke run",
        log_path=log,
        git_sha="abc123",
    )

    runs = xl.read_runs(log)
    assert len(runs) == 1
    assert runs[0] == written
    assert runs[0].config_hash == xl.config_hash(cfg)
    assert runs[0].seed == 42
    assert runs[0].metrics == {"auc": 0.5, "brier": 0.25}


def test_run_count_is_append_only(tmp_path):
    log = tmp_path / "experiments.jsonl"
    for i in range(3):
        xl.log_run(
            {"trial": i},
            {"auc": 0.5},
            seed=i,
            data_snapshot_hash="deadbeef",
            log_path=log,
            git_sha="abc123",
        )
    assert xl.run_count(log) == 3


def test_config_hash_is_order_invariant():
    assert xl.config_hash({"a": 1, "b": 2}) == xl.config_hash({"b": 2, "a": 1})
    assert xl.config_hash({"a": 1}) != xl.config_hash({"a": 2})


def test_missing_snapshot_hash_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="data_snapshot_hash"):
        xl.log_run(
            {"model": "dummy"},
            {"auc": 0.5},
            seed=0,
            data_snapshot_hash="",
            log_path=tmp_path / "experiments.jsonl",
        )


def test_corrupt_entry_fails_loudly(tmp_path):
    log = tmp_path / "experiments.jsonl"
    log.write_text(json.dumps({"timestamp": "t", "metrics": {}}) + "\n")
    with pytest.raises(ValueError, match="missing fields"):
        xl.read_runs(log)
