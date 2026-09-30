"""Book composition / refresh mechanics on synthetic data (no brokerage data)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from regime import book


def _bars(dates, closes):
    return [{"begins_at": d.strftime("%Y-%m-%dT00:00:00Z"), "close_price": f"{c:.4f}"} for d, c in zip(dates, closes)]


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(book, "DATA_DIR", tmp_path)
    (tmp_path / "raw" / "book").mkdir(parents=True)
    (tmp_path / "snapshots").mkdir()
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2024-01-01", periods=300)
    closes = {s: 50 * np.exp(np.cumsum(rng.normal(0, 0.02, 300))) for s in ["AAA", "BBB", "CCC"]}
    batch = {"data": {"results": [{"symbol": s, "bars": _bars(dates[:280], c[:280])} for s, c in closes.items()]}}
    (tmp_path / "raw" / "book" / "batch1.json").write_text(json.dumps(batch))
    # refresh dir with the last 20 bars + 5 overlapping
    rd = tmp_path / "raw" / "refresh_x"; rd.mkdir()
    rows = ["symbol,date,close"]
    for s, c in closes.items():
        for d, v in zip(dates[275:], c[275:]):
            rows.append(f"{s},{d.date()},{v:.4f}")
    (rd / "book_bars_1.csv").write_text("\n".join(rows) + "\n")
    pos1 = {"asof": "2024-12-01", "shares": {"AAA": 10, "BBB": 5}, "last_close": {"date": "x", "AAA": 50.0, "BBB": 100.0}}
    pos2 = {"asof": "2025-02-01", "shares": {"BBB": 5, "CCC": 20}, "last_close": {"date": "x", "BBB": 100.0, "CCC": 10.0}}
    (tmp_path / "book_positions.json").write_text(json.dumps(pos1))
    (tmp_path / "book_positions_c2.json").write_text(json.dumps(pos2))
    return tmp_path, dates, closes


def test_build_two_compositions_and_activate(sandbox):
    tmp, dates, closes = sandbox
    m1 = book.build_book_snapshot(1)
    m2 = book.build_book_snapshot(2)
    assert m1["target_weights"] == {"BBB": 0.5, "AAA": 0.5}
    assert set(m2["target_weights"]) == {"BBB", "CCC"}
    assert book.active_composition() == 1
    book.set_active_composition(2, "test")
    assert book.active_composition() == 2
    assert book.book_snapshot_hash() == m2["book_sha256"]
    assert book.book_snapshot_hash(1) == m1["book_sha256"]
    assert len(book.frozen_book_hashes()) == 2
    with pytest.raises(RuntimeError):
        book.build_book_snapshot(2)  # immutable


def test_refresh_extends_and_refuses_restated_history(sandbox):
    tmp, dates, closes = sandbox
    book.build_book_snapshot(1)
    entry = book.refresh_book_snapshot("refresh_x", composition=1, source="synthetic")
    assert entry["rows_added"] == 20 and entry["version"] == 2
    panel = book.load_book_panel(1)
    assert len(panel) == 299 and book.book_snapshot_hash(1) == entry["sha256"]
    # a second refresh whose overlap disagrees must be refused
    rd = tmp / "raw" / "refresh_y"; rd.mkdir()
    rows = ["symbol,date,close"]
    for s, c in closes.items():
        for d, v in zip(dates[290:], c[290:] * (1.5 if s == "AAA" else 1.0)):
            rows.append(f"{s},{d.date()},{v:.4f}")
    (rd / "book_bars_1.csv").write_text("\n".join(rows) + "\n")
    with pytest.raises(RuntimeError, match="restated history"):
        book.refresh_book_snapshot("refresh_y", composition=1, source="synthetic")


def test_missing_name_refuses(sandbox):
    tmp, dates, closes = sandbox
    (tmp / "book_positions_c3.json").write_text(json.dumps(
        {"asof": "x", "shares": {"AAA": 1, "ZZZ": 1}, "last_close": {"AAA": 1.0, "ZZZ": 1.0}}))
    with pytest.raises(RuntimeError, match="no raw closes"):
        book.build_book_snapshot(3)
