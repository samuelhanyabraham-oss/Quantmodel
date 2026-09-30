"""Portfolio ('book') index: the owner's actual holdings as one series, so
the regime machinery can be pointed at the thing being hedged instead of SPY.

Construction, and the assumptions it buys:
- Weights are CURRENT market-value weights (data/book_positions.json),
  renormalized each day over the names with real (non-interpolated) data.
  This is a deliberate, documented distortion: it asks "how would today's
  book have behaved," not "what did I actually hold" — early history is
  therefore a backcast of the current composition, and names that IPO'd
  late (CRWV 2025-03, NBIS 2024-10, DRAM, SHAZ, TE...) only enter when
  their data begins. Before those dates the index is effectively the miner
  cluster (IREN/CLSK/APLD/BTDR/CORZ).
- Interpolated bars from the source are dropped, not trusted.
- The book series is snapshotted and hash-locked (data/BOOK_MANIFEST.json)
  like every other input; experiments log that hash.
- Refreshes (forward test) rebuild the index from raw batches + refresh
  close files with the SAME weights and guards, verify the rebuilt series
  reproduces the frozen one on the overlap, and write a NEW versioned file
  (book_vN.csv). Frozen files are never modified.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_BOOK = ROOT / "data" / "raw" / "book"
POSITIONS_PATH = ROOT / "data" / "book_positions.json"
BOOK_SNAP = ROOT / "data" / "snapshots" / "book_v1.csv"
BOOK_MANIFEST = ROOT / "data" / "BOOK_MANIFEST.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_closes(refresh_dirs: list[Path] | None = None) -> pd.DataFrame:
    """Per-symbol close series from the frozen raw batches, optionally
    extended by refresh pulls (long CSVs: symbol,date,close). Where a
    refresh overlaps the batches the batch value wins — the overlap is
    checked for agreement by refresh_book_snapshot(), not papered over."""
    raw: dict[str, list[pd.Series]] = {}
    for f in sorted(RAW_BOOK.glob("batch*.json")):
        for r in json.loads(f.read_text())["data"]["results"]:
            bars = [b for b in r["bars"] if not b.get("interpolated")]
            if not bars:
                continue
            raw.setdefault(r["symbol"], []).append(pd.Series(
                [float(b["close_price"]) for b in bars],
                index=pd.to_datetime([b["begins_at"] for b in bars]).tz_localize(None).normalize(),
                name=r["symbol"],
            ))
    for d in refresh_dirs or []:
        for f in sorted(d.glob("book_bars_*.csv")):
            df = pd.read_csv(f)
            for sym, sub in df.groupby("symbol"):
                raw.setdefault(sym, []).append(pd.Series(
                    sub["close"].to_numpy(dtype=float),
                    index=pd.to_datetime(sub["date"]).dt.normalize().to_numpy(),
                    name=sym,
                ))
    closes = {}
    for sym, parts in raw.items():
        s = pd.concat(parts).sort_index()
        s.name = sym
        s = s[s > 0]  # a zero close is a data error, not a price
        s = s[~s.index.duplicated(keep="first")]  # batch value wins on overlap
        # Ticker-reuse guard: keep only the longest clean suffix — data
        # from the first bar after the LAST |log return| > 0.7 day.
        # Predecessor listings under a reused symbol (e.g. SHAZ pre-2025:
        # hundreds of +-160% oscillations) are data poison; a genuine
        # crash rarely exceeds -50% in a day, and losing one real jump
        # (CORZ's post-bankruptcy relist print) costs less than keeping
        # a fake history. Cut dates are recorded in the manifest.
        lr = np.log(s).diff().abs()
        bad = lr[lr > 0.7]
        if len(bad):
            s = s[s.index > bad.index[-1]]
        closes[sym] = s
    return pd.DataFrame(closes)


def _target_weights() -> dict[str, float]:
    pos = json.loads(POSITIONS_PATH.read_text())
    shares = pos["shares"]
    last_close = pos["last_close_2026_08_12"]
    target_w = {s: shares[s] * last_close[s] for s in shares}
    total = sum(target_w.values())
    return {s: v / total for s, v in target_w.items()}


def _build_index(closes: pd.DataFrame, target_w: dict[str, float]) -> pd.DataFrame:
    rets = np.log(closes).diff()

    # Daily book return: current weights renormalized over names with data.
    w = pd.DataFrame(
        {s: np.where(rets[s].notna(), target_w.get(s, 0.0), 0.0) for s in rets.columns},
        index=rets.index,
    )
    w = w.div(w.sum(axis=1), axis=0)
    book_ret = (w * rets.fillna(0.0)).sum(axis=1)
    book_ret = book_ret[w.sum(axis=1) > 0]
    book_index = 100.0 * np.exp(book_ret.cumsum())
    if not np.isfinite(book_index).all():
        raise RuntimeError("book index contains non-finite values — refuse to freeze")

    out = pd.DataFrame({"BOOK_close": book_index, "n_names": (w > 0).sum(axis=1)})
    out = out.dropna()  # the return series' seed day has no index value
    out.index.name = "date"
    return out


def build_book_snapshot() -> dict:
    """One-time: raw bars + positions -> hashed book-index snapshot."""
    if BOOK_MANIFEST.exists():
        raise RuntimeError("book snapshot already frozen (data/BOOK_MANIFEST.json)")
    pos = json.loads(POSITIONS_PATH.read_text())
    target_w = _target_weights()
    out = _build_index(_load_closes(), target_w)
    BOOK_SNAP.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(BOOK_SNAP, float_format="%.6f")

    manifest = {
        "created_utc": pd.Timestamp.now("UTC").isoformat(),
        "book_sha256": _sha256(BOOK_SNAP),
        "rows": len(out),
        "start": str(out.index.min().date()),
        "end": str(out.index.max().date()),
        "weights_asof": pos["asof"],
        "target_weights": {k: round(v, 4) for k, v in sorted(target_w.items(), key=lambda kv: -kv[1])},
        "source": "Robinhood MCP get_equity_historicals, day bars, split-adjusted, RTH",
        "excluded": ["GPUSB (no market data)", "BMNR Jan-2027 option"],
    }
    BOOK_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_book_manifest() -> dict:
    return json.loads(BOOK_MANIFEST.read_text())


def _active(m: dict) -> tuple[Path, str]:
    """(path, sha256) of the active book version."""
    versions = m.get("versions", [])
    if versions:
        active = m.get("active_version")
        for v in versions:
            if v["version"] == active:
                return BOOK_SNAP.parent / v["path"], v["sha256"]
        raise RuntimeError(f"book manifest active_version {active!r} not in versions")
    return BOOK_SNAP, m["book_sha256"]


def book_snapshot_hash() -> str:
    return _active(load_book_manifest())[1]


def frozen_book_hashes() -> set[str]:
    m = load_book_manifest()
    return {m["book_sha256"], *(v["sha256"] for v in m.get("versions", []))}


def load_book_panel(verify: bool = True) -> pd.DataFrame:
    path, want = _active(load_book_manifest())
    if verify and _sha256(path) != want:
        raise RuntimeError("book snapshot hash mismatch — snapshot was modified")
    return pd.read_csv(path, index_col="date", parse_dates=["date"])


def refresh_book_snapshot(refresh_tag: str, *, source: str, note: str = "") -> dict:
    """Rebuild the book index with the frozen weights over raw batches +
    the refresh close files, verify it reproduces the active frozen series
    on the overlap (rel tol 1e-9 — same code, same inputs, must match), and
    freeze the extension as a NEW version. Composition is NOT updated here:
    a re-weight is a separate, logged decision."""
    m = load_book_manifest()
    current = load_book_panel()
    refresh_dir = ROOT / "data" / "raw" / refresh_tag
    prior = [ROOT / "data" / "raw" / Path(v["refresh_dir"]).name for v in m.get("versions", [])]
    closes = _load_closes([*prior, refresh_dir])
    rebuilt = _build_index(closes, _target_weights())

    overlap = rebuilt.index.intersection(current.index)
    if len(overlap) != len(current.index):
        raise RuntimeError("rebuilt book index does not cover the frozen series")
    a = rebuilt.loc[overlap, "BOOK_close"].to_numpy()
    b = current.loc[overlap, "BOOK_close"].to_numpy()
    if not np.allclose(a, b, rtol=1e-6, atol=0.0):
        worst = overlap[np.argmax(np.abs(a / b - 1.0))]
        raise RuntimeError(f"rebuilt book index diverges from frozen on {worst.date()} — refuse")
    add = rebuilt[rebuilt.index > current.index.max()]
    if add.empty:
        raise RuntimeError("refresh adds no rows after the active book end")
    if (add["n_names"] < current["n_names"].iloc[-1]).any():
        raise RuntimeError("a book name went missing in the refresh — refuse to freeze a silently shrunken book")
    out = pd.concat([current, add])

    version = len(m.get("versions", [])) + 2  # v1 = the original freeze
    path = BOOK_SNAP.parent / f"book_v{version}.csv"
    if path.exists():
        raise RuntimeError(f"{path.name} already exists — versions are immutable")
    out.to_csv(path, float_format="%.6f")
    entry = {
        "version": version,
        "path": path.name,
        "sha256": _sha256(path),
        "created_utc": pd.Timestamp.now("UTC").isoformat(),
        "rows": len(out),
        "start": str(out.index.min().date()),
        "end": str(out.index.max().date()),
        "rows_added": len(add),
        "refresh_dir": f"data/raw/{refresh_tag}",
        "source": source,
        "weights_asof": m["weights_asof"],
        "note": note,
    }
    m.setdefault("versions", []).append(entry)
    m["active_version"] = version
    BOOK_MANIFEST.write_text(json.dumps(m, indent=2) + "\n")
    return entry
