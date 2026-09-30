"""Portfolio ('book') index: the owner's actual holdings as one series, so
the regime machinery can be pointed at the thing being hedged instead of SPY.

Construction, and the assumptions it buys:
- Weights are CURRENT market-value weights (a positions file), renormalized
  each day over the names with real (non-interpolated) data. This is a
  deliberate, documented distortion: it asks "how would today's book have
  behaved," not "what did I actually hold" — early history is a backcast of
  the current composition, and names that IPO'd late only enter when their
  data begins.
- Interpolated bars from the source are dropped, not trusted.
- Every book series is snapshotted and hash-locked like every other input;
  experiments log that hash.

Compositions (2026-09-30): the holdings change, so the book is versioned in
two dimensions. A COMPOSITION is a positions file (which names, what
weights); composition 1 is data/book_positions.json + data/BOOK_MANIFEST.json
(legacy paths). Composition N >= 2 lives at data/book_positions_cN.json +
data/BOOK_CN_MANIFEST.json and its snapshots at book_cN_v*.csv. Inside a
composition, REFRESHES append bars as new versions (book_v2.csv, ...).
data/BOOK_ACTIVE.json names the active composition (absent -> 1). Forward-
test entries record which composition they were predicted on and are
scored against that one; every composition's hashes stay legal forever.

Refreshes rebuild the index from raw batches + refresh close files with the
SAME weights and guards, verify every overlapping close per symbol and the
rebuilt series on the frozen range, and write a NEW versioned file. Frozen
files are never modified.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"  # tests monkeypatch this

TICKER_REUSE_TOL = 0.7  # |log return| above which a predecessor listing is assumed


def _raw_book() -> Path:
    return DATA_DIR / "raw" / "book"


def _snap_dir() -> Path:
    return DATA_DIR / "snapshots"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- compositions

def active_composition() -> int:
    p = DATA_DIR / "BOOK_ACTIVE.json"
    return int(json.loads(p.read_text())["composition"]) if p.exists() else 1


def composition_paths(c: int | None = None) -> dict:
    c = active_composition() if c is None else int(c)
    if c == 1:
        return {"composition": 1, "manifest": DATA_DIR / "BOOK_MANIFEST.json",
                "positions": DATA_DIR / "book_positions.json", "snap_prefix": "book"}
    return {"composition": c, "manifest": DATA_DIR / f"BOOK_C{c}_MANIFEST.json",
            "positions": DATA_DIR / f"book_positions_c{c}.json", "snap_prefix": f"book_c{c}"}


def set_active_composition(c: int, reason: str) -> dict:
    paths = composition_paths(c)
    if not paths["manifest"].exists():
        raise RuntimeError(f"composition {c} is not built ({paths['manifest'].name} missing)")
    rec = {"composition": int(c), "set_utc": pd.Timestamp.now("UTC").isoformat(), "reason": reason}
    (DATA_DIR / "BOOK_ACTIVE.json").write_text(json.dumps(rec, indent=2) + "\n")
    return rec


def _target_weights(positions_path: Path) -> dict[str, float]:
    pos = json.loads(positions_path.read_text())
    shares = pos["shares"]
    key = next(k for k in pos if k.startswith("last_close"))
    last_close = {s: v for s, v in pos[key].items() if s in shares}
    missing = [s for s in shares if s not in last_close]
    if missing:
        raise ValueError(f"positions file lacks a last close for {missing}")
    mv = {s: shares[s] * last_close[s] for s in shares}
    total = sum(mv.values())
    return {s: v / total for s, v in mv.items()}


# ---------------------------------------------------------------- raw closes

def _load_closes(refresh_dirs: list[Path] | None = None) -> pd.DataFrame:
    """Per-symbol close series from the frozen raw batches, optionally
    extended by refresh pulls (long CSVs: symbol,date,close). Where a
    refresh overlaps the batches the batch value wins; agreement on the
    overlap is checked by refresh_book_snapshot(), never papered over."""
    raw: dict[str, list[pd.Series]] = {}
    for f in sorted(_raw_book().glob("batch*.json")):
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
        for f in sorted(Path(d).glob("book_bars_*.csv")):
            df = pd.read_csv(f)
            for sym, sub in df.groupby("symbol"):
                raw.setdefault(sym, []).append(pd.Series(
                    sub["close"].to_numpy(dtype=float),
                    index=pd.to_datetime(sub["date"]).dt.normalize().to_numpy(),
                    name=sym,
                ))
    closes = {}
    for sym, parts in raw.items():
        s = pd.concat(parts)
        s = s[~s.index.duplicated(keep="first")].sort_index()  # batch value wins on overlap
        s.name = sym
        s = s[s > 0]  # a zero close is a data error, not a price
        # Ticker-reuse guard: keep only the longest clean suffix — data
        # from the first bar after the LAST |log return| > TICKER_REUSE_TOL
        # day. Predecessor listings under a reused symbol (e.g. SHAZ
        # pre-2025: hundreds of +-160% oscillations) are data poison; a
        # genuine crash rarely exceeds -50% in a day.
        lr = np.log(s).diff().abs()
        bad = lr[lr > TICKER_REUSE_TOL]
        if len(bad):
            s = s[s.index > bad.index[-1]]
        closes[sym] = s
    return pd.DataFrame(closes)


def _build_index(closes: pd.DataFrame, target_w: dict[str, float]) -> pd.DataFrame:
    names = [s for s in target_w if s in closes.columns]
    absent = [s for s in target_w if s not in closes.columns]
    if absent:
        raise RuntimeError(f"no raw closes for book names {absent} — pull them first")
    rets = np.log(closes[names]).diff()
    w = pd.DataFrame({s: np.where(rets[s].notna(), target_w[s], 0.0) for s in names}, index=rets.index)
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


# ---------------------------------------------------------------- build / load

def build_book_snapshot(composition: int = 1, *, refresh_dirs: list[Path] | None = None,
                        source: str = "", note: str = "") -> dict:
    """One-time per composition: raw bars + positions -> hashed book-index
    snapshot (book[_cN]_v1.csv) + manifest. Refuses to overwrite."""
    paths = composition_paths(composition)
    if paths["manifest"].exists():
        raise RuntimeError(f"composition {composition} already frozen ({paths['manifest'].name})")
    if not paths["positions"].exists():
        raise FileNotFoundError(f"positions file for composition {composition}: {paths['positions']}")
    pos = json.loads(paths["positions"].read_text())
    target_w = _target_weights(paths["positions"])
    out = _build_index(_load_closes(refresh_dirs), target_w)
    snap = _snap_dir() / f"{paths['snap_prefix']}_v1.csv"
    snap.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(snap, float_format="%.6f")
    manifest = {
        "composition": int(composition),
        "created_utc": pd.Timestamp.now("UTC").isoformat(),
        "book_sha256": _sha256(snap),
        "path": snap.name,
        "rows": len(out),
        "start": str(out.index.min().date()),
        "end": str(out.index.max().date()),
        "weights_asof": pos["asof"],
        "positions_file": paths["positions"].name,
        "target_weights": {k: round(v, 4) for k, v in sorted(target_w.items(), key=lambda kv: -kv[1])},
        "source": source or pos.get("source", ""),
        "refresh_dirs_used": [Path(d).name for d in (refresh_dirs or [])],  # under data/raw/
        "excluded": pos.get("excluded", []),
        "note": note or pos.get("note", ""),
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_book_manifest(composition: int | None = None) -> dict:
    return json.loads(composition_paths(composition)["manifest"].read_text())


def _active(m: dict, prefix: str) -> tuple[Path, str]:
    """(path, sha256) of the active version inside one composition."""
    versions = m.get("versions", [])
    if versions:
        active = m.get("active_version")
        for v in versions:
            if v["version"] == active:
                return _snap_dir() / v["path"], v["sha256"]
        raise RuntimeError(f"book manifest active_version {active!r} not in versions")
    return _snap_dir() / m.get("path", f"{prefix}_v1.csv"), m["book_sha256"]


def book_snapshot_hash(composition: int | None = None) -> str:
    p = composition_paths(composition)
    return _active(load_book_manifest(p["composition"]), p["snap_prefix"])[1]


def frozen_book_hashes() -> set[str]:
    out = set()
    for mp in sorted(DATA_DIR.glob("BOOK*_MANIFEST.json")):
        m = json.loads(mp.read_text())
        out.add(m["book_sha256"])
        out |= {v["sha256"] for v in m.get("versions", [])}
    return out


def load_book_panel(composition: int | None = None, verify: bool = True) -> pd.DataFrame:
    p = composition_paths(composition)
    path, want = _active(load_book_manifest(p["composition"]), p["snap_prefix"])
    if verify and _sha256(path) != want:
        raise RuntimeError(f"book snapshot hash mismatch — {path.name} was modified")
    return pd.read_csv(path, index_col="date", parse_dates=["date"])


# ---------------------------------------------------------------- refresh

def refresh_book_snapshot(refresh_tag: str, *, composition: int | None = None,
                          source: str, note: str = "") -> dict:
    """Extend one composition's book index with newly pulled closes as a NEW
    version. Composition is NOT changed here: a re-weight is
    build_book_snapshot() on a new positions file."""
    p = composition_paths(composition)
    m = load_book_manifest(p["composition"])
    current = load_book_panel(p["composition"])
    refresh_dir = DATA_DIR / "raw" / refresh_tag
    prior = [DATA_DIR / "raw" / Path(v["refresh_dir"]).name for v in m.get("versions", [])]
    prior += [DATA_DIR / "raw" / Path(d).name for d in m.get("refresh_dirs_used", [])]

    # Per-symbol continuity: every refresh close on a date the frozen raw
    # batches (or an earlier refresh) already hold must agree to 1e-6 — a
    # restated or re-adjusted history is refused, never blended.
    frozen_closes = _load_closes(prior)
    files = sorted(refresh_dir.glob("book_bars_*.csv"))
    if not files:
        raise FileNotFoundError(f"no book_bars_*.csv in {refresh_dir}")
    fresh = pd.concat([pd.read_csv(f) for f in files])
    fresh["date"] = pd.to_datetime(fresh["date"]).dt.normalize()
    n_compared = {}
    for sym, sub in fresh.groupby("symbol"):
        if sym not in frozen_closes.columns:
            continue
        s_new = sub.set_index("date")["close"].astype(float)
        common = s_new.index.intersection(frozen_closes.index[frozen_closes[sym].notna()])
        n_compared[sym] = int(len(common))
        if len(common) == 0:
            raise RuntimeError(f"{sym}: refresh has no overlap with the frozen raw closes — cannot verify continuity")
        diff = (s_new.loc[common] - frozen_closes.loc[common, sym]).abs()
        if (diff > 1e-6).any():
            raise RuntimeError(f"{sym}: refresh close disagrees with frozen raw close on "
                               f"{list(common[diff > 1e-6].date)} — restated history refused")
    closes = _load_closes([*prior, refresh_dir])
    rebuilt = _build_index(closes, _target_weights(p["positions"]))

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

    version = len(m.get("versions", [])) + 2  # v1 = the composition's original freeze
    path = _snap_dir() / f"{p['snap_prefix']}_v{version}.csv"
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
        "overlap_closes_compared_per_symbol": n_compared,
        "note": note,
    }
    m.setdefault("versions", []).append(entry)
    m["active_version"] = version
    p["manifest"].write_text(json.dumps(m, indent=2) + "\n")
    return entry
