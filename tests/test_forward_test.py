"""Forward-test record hygiene: de-duplication, label resolution, and the
one-entry-per-day guard. Synthetic frames only — no snapshot access."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import forward_test_score as fts  # noqa: E402
import predict_today  # noqa: E402


def _write_log(path, entries):
    path.write_text("".join(json.dumps(e) + "\n" for e in entries))


def test_read_forward_log_dedupes_on_asof_config_universe(tmp_path, monkeypatch):
    log = tmp_path / "fwd.jsonl"
    e = {"asof": "2026-08-12", "config": "c", "signal": 0, "band_lo": 0, "band_hi": 0.1}
    _write_log(log, [e, e, {**e, "universe": "book"}, {**e, "universe": "book"}, {**e, "asof": "2026-08-13"}])
    monkeypatch.setattr(fts, "FORWARD_LOG", log)
    entries, dupes = fts.read_forward_log()
    assert len(entries) == 3 and dupes == 2


def test_already_logged_guard(tmp_path, monkeypatch):
    log = tmp_path / "fwd.jsonl"
    _write_log(log, [{"asof": "2026-08-12", "config": "c", "universe": "book"}])
    monkeypatch.setattr(predict_today, "FORWARD_LOG", log)
    assert predict_today.already_logged("2026-08-12", "c", "book")
    assert not predict_today.already_logged("2026-08-12", "c")  # SPY universe (None) distinct
    assert not predict_today.already_logged("2026-08-13", "c", "book")


def test_score_resolves_only_closed_windows():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2024-01-01", periods=400)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))), index=idx)
    frame = fts._resolve_frame(close)
    asof_resolved = str(idx[-30].date())
    asof_pending = str(idx[-5].date())  # fewer than HORIZON bars after it
    entries = [
        {"asof": asof_resolved, "config": "c", "signal": 1, "band_lo": 0.25, "band_hi": 0.5},
        {"asof": asof_pending, "config": "c", "signal": 0, "band_lo": 0.0, "band_hi": 0.1},
    ]
    rows, skipped = fts.score(entries, {None: frame})
    assert skipped == []
    assert [r["resolved"] for r in rows] == [True, False]
    assert rows[0]["label"] in (0, 1)
    assert rows[0]["outcome"].startswith(("TP", "FP"))
    agg = fts.aggregate(rows, {None: frame})
    (key,) = agg.keys()
    assert key == "SPY | c"
    assert agg[key]["n_resolved"] == 1 and agg[key]["n_pending"] == 1
    assert agg[key]["skill_test"].startswith("not attempted")


def test_score_label_matches_label_module():
    """The scorer's realized label must be the charter label, not a re-derivation."""
    from regime import labels
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2024-01-01", periods=400)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))), index=idx)
    frame = fts._resolve_frame(close)
    lab = labels.build_labels(close)
    pd.testing.assert_series_equal(frame["label"], lab["label"])
