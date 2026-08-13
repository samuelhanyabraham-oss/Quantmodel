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


def _load_closes() -> pd.DataFrame:
    closes = {}
    for f in sorted(RAW_BOOK.glob("batch*.json")):
        for r in json.loads(f.read_text())["data"]["results"]:
            bars = [b for b in r["bars"] if not b.get("interpolated")]
            if not bars:
                continue
            s = pd.Series(
                [float(b["close_price"]) for b in bars],
                index=pd.to_datetime([b["begins_at"] for b in bars]).tz_localize(None).normalize(),
                name=r["symbol"],
            )
            s = s[s > 0]  # a zero close is a data error, not a price
            s = s[~s.index.duplicated(keep="last")]
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
            closes[r["symbol"]] = s
    return pd.DataFrame(closes)


def build_book_snapshot() -> dict:
    """One-time: raw bars + positions -> hashed book-index snapshot."""
    if BOOK_MANIFEST.exists():
        raise RuntimeError("book snapshot already frozen (data/BOOK_MANIFEST.json)")
    pos = json.loads(POSITIONS_PATH.read_text())
    shares = pos["shares"]
    last_close = pos["last_close_2026_08_12"]
    target_w = {s: shares[s] * last_close[s] for s in shares}
    total = sum(target_w.values())
    target_w = {s: v / total for s, v in target_w.items()}

    closes = _load_closes()
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
    BOOK_SNAP.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(BOOK_SNAP, float_format="%.6f")

    manifest = {
        "created_utc": pd.Timestamp.utcnow().isoformat(),
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


def book_snapshot_hash() -> str:
    return load_book_manifest()["book_sha256"]


def load_book_panel(verify: bool = True) -> pd.DataFrame:
    m = load_book_manifest()
    if verify and _sha256(BOOK_SNAP) != m["book_sha256"]:
        raise RuntimeError("book snapshot hash mismatch — snapshot was modified")
    return pd.read_csv(BOOK_SNAP, index_col="date", parse_dates=["date"])
