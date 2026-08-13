"""Data snapshotting and loading.

Raw pulls (data/raw/*.json, IBKR MCP daily bars) are converted once into a
canonical CSV snapshot in data/snapshots/, content-hashed, and split at the
holdout boundary. Experiments load ONLY from the snapshot — never from the raw
pulls and never from a live API — so a restated series can't silently change a
past result. The dev/holdout split is physical (two files): experiment code
imports load_dev_panel() and simply has no path to holdout rows.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
SNAP_DIR = ROOT / "data" / "snapshots"
MANIFEST_PATH = ROOT / "data" / "HOLDOUT_MANIFEST.json"

SERIES = ["SPY", "QQQ", "IWM", "HYG", "LQD", "VIX", "VIX3M"]
HOLDOUT_MONTHS = 18


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_raw_series(name: str) -> pd.DataFrame:
    d = json.loads((RAW_DIR / f"{name}.json").read_text())
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(d["time"]).tz_localize(None).normalize(),
            f"{name}_close": d["close"],
            f"{name}_high": d["high"],
            f"{name}_low": d["low"],
        }
    )
    return df.set_index("date")


def build_panel_from_raw() -> pd.DataFrame:
    """Merge raw series into one daily panel on SPY trading dates."""
    panel = _load_raw_series("SPY")
    for name in SERIES[1:]:
        panel = panel.join(_load_raw_series(name), how="left")
    # Non-equity series may miss the odd date; a 3-day ffill limit covers
    # exchange quirks without papering over real gaps.
    panel = panel.ffill(limit=3)
    panel = panel.dropna()
    return panel


def build_snapshot() -> dict:
    """One-time: raw -> hashed dev/holdout snapshot + holdout manifest.

    Refuses to overwrite an existing snapshot: the snapshot is frozen once.
    """
    if MANIFEST_PATH.exists():
        raise RuntimeError(
            "Snapshot already exists — the snapshot is frozen. Delete requires "
            "explicit human action and a charter changelog entry."
        )
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    panel = build_panel_from_raw()

    boundary = panel.index.max() - pd.DateOffset(months=HOLDOUT_MONTHS)
    dev = panel[panel.index <= boundary]
    holdout = panel[panel.index > boundary]

    dev_path = SNAP_DIR / "panel_dev.csv"
    holdout_path = SNAP_DIR / "panel_holdout.csv"
    dev.to_csv(dev_path, float_format="%.6f")
    holdout.to_csv(holdout_path, float_format="%.6f")

    manifest = {
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "holdout_boundary_date": str(boundary.date()),
        "holdout_months": HOLDOUT_MONTHS,
        "dev_rows": len(dev),
        "holdout_rows": len(holdout),
        "dev_start": str(dev.index.min().date()),
        "dev_end": str(dev.index.max().date()),
        "holdout_start": str(holdout.index.min().date()),
        "holdout_end": str(holdout.index.max().date()),
        "dev_sha256": sha256_file(dev_path),
        "holdout_sha256": sha256_file(holdout_path),
        "series": SERIES,
        "source": "IBKR MCP get_price_history, ONE_DAY bars, FIVE_YEARS, RTH only",
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def snapshot_hash() -> str:
    """The data hash experiments must log (dev snapshot only)."""
    return load_manifest()["dev_sha256"]


def load_dev_panel(verify: bool = True) -> pd.DataFrame:
    """Load the development panel. This is the ONLY loader experiment code
    may use before holdout is unlocked."""
    m = load_manifest()
    path = SNAP_DIR / "panel_dev.csv"
    if verify and sha256_file(path) != m["dev_sha256"]:
        raise RuntimeError("dev snapshot hash mismatch — snapshot was modified")
    df = pd.read_csv(path, index_col="date", parse_dates=["date"])
    assert str(df.index.max().date()) <= m["holdout_boundary_date"], (
        "dev panel crosses the holdout boundary"
    )
    return df


def load_holdout_panel(*, i_have_explicit_permission: bool = False) -> pd.DataFrame:
    """Holdout loader. The flag is a tripwire, not security: it must appear
    (grep-ably) at any call site, and no committed experiment code may set it
    until the holdout is unlocked in writing."""
    if not i_have_explicit_permission:
        raise PermissionError(
            "Holdout is locked (CLAUDE.md). Requires explicit permission in the "
            "unlocking message."
        )
    m = load_manifest()
    path = SNAP_DIR / "panel_holdout.csv"
    if sha256_file(path) != m["holdout_sha256"]:
        raise RuntimeError("holdout snapshot hash mismatch — snapshot was modified")
    return pd.read_csv(path, index_col="date", parse_dates=["date"])
