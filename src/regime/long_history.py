"""Long-history market panel (2000 -> present), for the replication study.

Why this exists: every REPORT.md conclusion so far rests on ~5 years of data
with one bear market. Robinhood's bar API serves daily bars back to 2000
for the ETFs and to 2002 for VIX, plus the S&P 500 index itself (SPX),
which is used here purely as a CHECK series for SPY prints.

Construction (all recorded in data/LONG_MANIFEST.json):
- Long pulls (data/raw/long_history/*.csv, long format) become one wide
  panel with the SAME columns as the frozen panel. Rows are SPY trading
  dates. Series that start later (HYG 2007, LQD 2002, VIX 2002) are NaN
  before their first print. VIX3M exists ONLY inside the frozen segment.
- The frozen active panel (2021-08-16 onward) is appended verbatim; on the
  overlap days every close must agree to the cent or the build refuses.
- SPY print repair: SPY/SPX is stable to ~0.04%/day. A close whose ratio
  deviates from its 21-day centered median by more than SPY_DEV_TOL is a
  vendor print error (32 days, all in 2000/2005/2008-09) and is
  replaced by SPX x local-median ratio; high/low are widened to contain
  the repaired close. Every repaired day is listed in the manifest.
- Feature-ETF spike repair: a bar whose |log return| > 0.25 is immediately
  reversed by an opposite move > 0.25 (IWM 2005-02-17 and 2005-02-24, both
  half-price prints) is set to NaN and forward-filled (limit 3), the same
  gap policy as the original build. Listed in the manifest.
No other value is touched. The result is hash-locked like every input.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import data

ROOT = Path(__file__).resolve().parents[2]
LONG_RAW = ROOT / "data" / "raw" / "long_history"
LONG_SNAP = ROOT / "data" / "snapshots" / "panel_long_v1.csv"
LONG_MANIFEST = ROOT / "data" / "LONG_MANIFEST.json"
SPY_DEV_TOL = 0.01       # 1% of index-implied price (~25x a typical day; 0.5% would reach into 2008 tracking noise)
SPIKE_TOL = 0.25         # single-bar reversed |log return| for feature ETFs
RATIO_WINDOW = 21


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _wide(csv: Path) -> pd.DataFrame:
    d = pd.read_csv(csv, parse_dates=["date"])
    d["date"] = d["date"].dt.normalize()
    out = {}
    for sym, g in d.groupby("symbol"):
        g = g.drop_duplicates("date").set_index("date").sort_index()
        for col in ("close", "high", "low"):
            out[f"{sym}_{col}"] = g[col]
    return pd.DataFrame(out)


def build_long_panel() -> dict:
    if LONG_MANIFEST.exists():
        raise RuntimeError("long panel already frozen (data/LONG_MANIFEST.json)")
    etf = _wide(LONG_RAW / "etf_bars.csv")
    vix = _wide(LONG_RAW / "vix_bars.csv")
    spx = _wide(LONG_RAW / "spx_bars.csv")["SPX_close"]
    long = etf.join(vix, how="left")
    long = long[long["SPY_close"].notna()]

    repairs = {"spy_index_check": [], "feature_etf_spikes": []}

    # --- SPY print repair against the index ---
    j = pd.concat([long["SPY_close"], spx], axis=1).dropna()
    ratio = j["SPY_close"] / j["SPX_close"]
    med = ratio.rolling(RATIO_WINDOW, center=True, min_periods=5).median()
    dev = ratio / med - 1.0
    flagged = dev[dev.abs() > SPY_DEV_TOL].index
    for t in flagged:
        old = float(long.at[t, "SPY_close"])
        new = float(j.at[t, "SPX_close"] * med[t])
        long.at[t, "SPY_close"] = new
        long.at[t, "SPY_high"] = max(float(long.at[t, "SPY_high"]), new)
        long.at[t, "SPY_low"] = min(float(long.at[t, "SPY_low"]), new)
        repairs["spy_index_check"].append(
            {"date": str(t.date()), "vendor_close": round(old, 4), "repaired_close": round(new, 4),
             "deviation": round(float(dev[t]), 5)})

    # --- feature ETF single-bar spikes ---
    for sym in ("QQQ", "IWM", "HYG", "LQD"):
        c = long[f"{sym}_close"]
        lr = np.log(c).diff()
        nxt = lr.shift(-1)
        spike = (lr.abs() > SPIKE_TOL) & (nxt.abs() > SPIKE_TOL) & (np.sign(lr) != np.sign(nxt))
        for t in c.index[spike.fillna(False)]:
            repairs["feature_etf_spikes"].append({"symbol": sym, "date": str(t.date()), "vendor_close": float(c[t])})
            for col in ("close", "high", "low"):
                long.at[t, f"{sym}_{col}"] = np.nan
    long = long.ffill(limit=3)

    # --- append the frozen segment; verify the overlap ---
    frozen = data.load_dev_panel()
    long["VIX3M_close"] = np.nan; long["VIX3M_high"] = np.nan; long["VIX3M_low"] = np.nan
    long = long[frozen.columns]
    overlap = long.index.intersection(frozen.index)
    if len(overlap) < 3:
        raise RuntimeError("long pull does not overlap the frozen panel enough to verify")
    if any(pd.Timestamp(r["date"]) in overlap for r in repairs["spy_index_check"]):
        raise RuntimeError("a repair landed inside the frozen overlap — refuse")
    for col in frozen.columns:
        if col.startswith("VIX3M"):
            continue
        a, b = long.loc[overlap, col], frozen.loc[overlap, col]
        tol = 1e-6 if col.endswith("_close") else data.HL_REL_TOL * b
        if ((a - b).abs() > tol).any():
            raise RuntimeError(f"long pull disagrees with frozen values for {col} on the overlap")
    panel = pd.concat([long[long.index < frozen.index.min()], frozen])
    panel.index.name = "date"

    LONG_SNAP.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(LONG_SNAP, float_format="%.6f")
    coverage = {s: {"first": str(panel[f"{s}_close"].first_valid_index().date()),
                    "last": str(panel[f"{s}_close"].last_valid_index().date()),
                    "rows": int(panel[f"{s}_close"].notna().sum())} for s in data.SERIES}
    manifest = {
        "created_utc": pd.Timestamp.now("UTC").isoformat(),
        "path": LONG_SNAP.name,
        "sha256": _sha256(LONG_SNAP),
        "rows": len(panel),
        "start": str(panel.index.min().date()),
        "end": str(panel.index.max().date()),
        "frozen_segment_from": str(frozen.index.min().date()),
        "frozen_segment_hash": data.snapshot_hash(),
        "overlap_rows_verified": len(overlap),
        "source_long": "Robinhood MCP get_equity_historicals (SPY/QQQ/IWM/HYG/LQD, day bars, RTH, split-adjusted; 2000-01-01 -> 2021-08-19 in decade chunks) and get_index_historicals (VIX, SPX; 2-year chunks). Pulled 2026-09-30.",
        "check_series": "SPX (S&P 500 index) used only to validate SPY prints; not a panel column",
        "coverage": coverage,
        "spy_dev_tol": SPY_DEV_TOL,
        "spike_tol": SPIKE_TOL,
        "repairs": repairs,
        "n_spy_repaired": len(repairs["spy_index_check"]),
        "n_spike_repaired": len(repairs["feature_etf_spikes"]),
    }
    LONG_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_long_manifest() -> dict:
    return json.loads(LONG_MANIFEST.read_text())


def long_snapshot_hash() -> str:
    return load_long_manifest()["sha256"]


def frozen_long_hashes() -> set[str]:
    return {load_long_manifest()["sha256"]} if LONG_MANIFEST.exists() else set()


def load_long_panel(verify: bool = True) -> pd.DataFrame:
    m = load_long_manifest()
    if verify and _sha256(LONG_SNAP) != m["sha256"]:
        raise RuntimeError("long panel hash mismatch — snapshot was modified")
    return pd.read_csv(LONG_SNAP, index_col="date", parse_dates=["date"])
