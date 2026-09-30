"""Data snapshotting and loading.

Raw pulls (data/raw/*.json, IBKR MCP daily bars) are converted once into a
canonical CSV snapshot in data/snapshots/, content-hashed, and split at the
holdout boundary. Experiments load ONLY from the snapshot — never from the raw
pulls and never from a live API — so a restated series can't silently change a
past result. The dev/holdout split is physical (two files): experiment code
imports load_dev_panel() and simply has no path to holdout rows.

Refresh protocol (forward test, REPORT.md "Next steps"): new bars are never
appended to a frozen file. refresh_snapshot() writes a NEW versioned file
(panel_vN.csv), records its hash in the manifest's `versions` list, and
moves the `active_version` pointer. Every hash ever frozen stays legal for
the experiment log; the frozen files themselves are immutable.
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
HL_REL_TOL = 0.001  # vendor tolerance on overlap highs/lows (closes: exact)


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
        "created_utc": pd.Timestamp.now("UTC").isoformat(),
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


def dissolve_holdout() -> dict:
    """Owner-ordered dissolution (charter changelog 2026-08-13): merge dev +
    holdout into a frozen full snapshot (v2). One-way; requires the recorded
    unlock. After this, load_dev_panel() serves the full panel and the
    holdout loader refuses (nothing is 'held out' anymore)."""
    unlock = ROOT / "data" / "HOLDOUT_UNLOCK.json"
    if not (unlock.exists() and json.loads(unlock.read_text()).get("unlocked")):
        raise PermissionError("dissolution requires the recorded owner unlock")
    m = load_manifest()
    if m.get("dissolved"):
        raise RuntimeError("holdout already dissolved")
    dev = pd.read_csv(SNAP_DIR / "panel_dev.csv", index_col="date", parse_dates=["date"])
    holdout = pd.read_csv(
        SNAP_DIR / "panel_holdout.csv", index_col="date", parse_dates=["date"]
    )
    full = pd.concat([dev, holdout])
    full_path = SNAP_DIR / "panel_full.csv"
    full.to_csv(full_path, float_format="%.6f")
    m.update(
        {
            "dissolved": True,
            "dissolved_utc": pd.Timestamp.now("UTC").isoformat(),
            "full_rows": len(full),
            "full_sha256": sha256_file(full_path),
        }
    )
    MANIFEST_PATH.write_text(json.dumps(m, indent=2) + "\n")
    return m


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def _active_version(m: dict) -> dict | None:
    """The newest refreshed version, or None if no refresh has happened."""
    versions = m.get("versions", [])
    if not versions:
        return None
    active = m.get("active_version")
    for v in versions:
        if v["version"] == active:
            return v
    raise RuntimeError(f"manifest active_version {active!r} not in versions")


def snapshot_hash() -> str:
    """The data hash experiments must log: the active refreshed version if
    any, else the full snapshot after dissolution, else the dev snapshot."""
    m = load_manifest()
    v = _active_version(m)
    if v is not None:
        return v["sha256"]
    return m["full_sha256"] if m.get("dissolved") else m["dev_sha256"]


def frozen_hashes() -> set[str]:
    """Every market-panel hash that has ever been frozen. An experiment
    entry must carry one of these (or a frozen book hash) — nothing else."""
    m = load_manifest()
    out = {m["dev_sha256"]}
    if m.get("dissolved"):
        out.add(m["full_sha256"])
    for v in m.get("versions", []):
        out.add(v["sha256"])
    return out


def _read_refresh_bars(refresh_dir: Path) -> pd.DataFrame:
    """Refresh pulls are long-format CSVs (symbol,date,close,high,low) in
    data/raw/<refresh>/ — one per source call. Wide panel out, NaN where a
    series was not delivered. No forward-fill across the refresh boundary:
    a missing series stays missing, so downstream code fails loudly rather
    than scoring a stale value."""
    frames = []
    for f in sorted(refresh_dir.glob("*_bars.csv")):
        d = pd.read_csv(f)
        if not {"symbol", "date", "close", "high", "low"} <= set(d.columns):
            continue  # book close-only files live in the same folder
        frames.append(d)
    if not frames:
        raise FileNotFoundError(f"no *_bars.csv with close/high/low in {refresh_dir}")
    long = pd.concat(frames, ignore_index=True)
    long["date"] = pd.to_datetime(long["date"]).dt.normalize()
    wide = pd.DataFrame(index=pd.DatetimeIndex(sorted(long["date"].unique()), name="date"))
    for name in SERIES:
        sub = long[long["symbol"] == name].set_index("date").sort_index()
        sub = sub[~sub.index.duplicated(keep="last")]
        for col in ("close", "high", "low"):
            wide[f"{name}_{col}"] = sub[col] if len(sub) else float("nan")
    return wide


def refresh_snapshot(refresh_tag: str, *, source: str, note: str = "") -> dict:
    """Append newly pulled bars as a NEW frozen version.

    Safety: rows the new pull shares with the active panel must agree with
    the frozen values (abs tol 1e-6) — a restated history is refused, never
    silently adopted. Rows are kept only where SPY (the label series)
    printed; other series may be NaN and are recorded as such in the
    manifest so the gap is visible."""
    m = load_manifest()
    if not m.get("dissolved"):
        raise RuntimeError("refresh is a post-dissolution (forward-test) operation")
    current = load_dev_panel()
    new = _read_refresh_bars(RAW_DIR / refresh_tag)

    overlap = new.index.intersection(current.index)
    if len(overlap) == 0:
        raise RuntimeError("refresh pull does not overlap the frozen panel — cannot verify continuity")
    # Closes must agree exactly: they drive every label, baseline and frozen
    # feature. Intraday highs/lows may differ by cents between vendors
    # (auction / odd-lot extreme prints); tolerated up to HL_REL_TOL of the
    # close and recorded in the manifest, never silently.
    hl_worst = 0.0
    for col in current.columns:
        a = new.loc[overlap, col]
        b = current.loc[overlap, col]
        if col.endswith("_close"):
            ok = a.isna() | ((a - b).abs() <= 1e-6)
        else:
            rel = ((a - b).abs() / b).fillna(0.0)
            hl_worst = max(hl_worst, float(rel.max()))
            ok = a.isna() | (rel <= HL_REL_TOL)
        if not ok.all():
            bad = overlap[~ok.to_numpy()]
            raise RuntimeError(f"refresh disagrees with frozen values for {col} on {list(bad.date)}")

    add = new[new.index > current.index.max()]
    add = add[add["SPY_close"].notna()]
    if add.empty:
        raise RuntimeError("refresh contains no rows after the active panel end")
    panel = pd.concat([current, add[current.columns]])

    version = len(m.get("versions", [])) + 3  # v1 = dev, v2 = full (dissolved)
    path = SNAP_DIR / f"panel_v{version}.csv"
    if path.exists():
        raise RuntimeError(f"{path.name} already exists — versions are immutable")
    panel.to_csv(path, float_format="%.6f")
    missing = {
        s: str(add.index[add[f"{s}_close"].isna()].min().date())
        for s in SERIES if add[f"{s}_close"].isna().any()
    }
    entry = {
        "version": version,
        "path": path.name,
        "sha256": sha256_file(path),
        "created_utc": pd.Timestamp.now("UTC").isoformat(),
        "rows": len(panel),
        "start": str(panel.index.min().date()),
        "end": str(panel.index.max().date()),
        "rows_added": len(add),
        "refresh_dir": str(Path("data/raw") / refresh_tag),
        "source": source,
        "missing_series_from": missing,
        "overlap_rows_checked": len(overlap),
        "overlap_high_low_max_rel_diff": round(hl_worst, 6),
        "note": note,
    }
    m.setdefault("versions", []).append(entry)
    m["active_version"] = version
    MANIFEST_PATH.write_text(json.dumps(m, indent=2) + "\n")
    return entry


def load_dev_panel(verify: bool = True) -> pd.DataFrame:
    """Load the development panel: the dev snapshot, or the full snapshot
    after the owner-ordered dissolution (charter changelog 2026-08-13), or
    the active refreshed version once the forward test has begun."""
    m = load_manifest()
    v = _active_version(m)
    if v is not None:
        path = SNAP_DIR / v["path"]
        want = v["sha256"]
    elif m.get("dissolved"):
        path = SNAP_DIR / "panel_full.csv"
        want = m["full_sha256"]
    else:
        path = SNAP_DIR / "panel_dev.csv"
        want = m["dev_sha256"]
    if verify and sha256_file(path) != want:
        raise RuntimeError("snapshot hash mismatch — snapshot was modified")
    df = pd.read_csv(path, index_col="date", parse_dates=["date"])
    if not m.get("dissolved"):
        assert str(df.index.max().date()) <= m["holdout_boundary_date"], (
            "dev panel crosses the holdout boundary"
        )
    return df


def load_holdout_panel(*, i_have_explicit_permission: bool = False) -> pd.DataFrame:
    """Holdout loader. The flag is a tripwire, not security: it must appear
    (grep-ably) at any call site. After dissolution there is no holdout to
    load — refuse loudly rather than pretend."""
    m = load_manifest()
    if m.get("dissolved"):
        raise RuntimeError(
            "holdout was dissolved into the full snapshot (charter changelog "
            "2026-08-13); nothing is held out — use load_dev_panel()"
        )
    if not i_have_explicit_permission:
        raise PermissionError(
            "Holdout is locked (CLAUDE.md). Requires explicit permission in the "
            "unlocking message."
        )
    path = SNAP_DIR / "panel_holdout.csv"
    if sha256_file(path) != m["holdout_sha256"]:
        raise RuntimeError("holdout snapshot hash mismatch — snapshot was modified")
    return pd.read_csv(path, index_col="date", parse_dates=["date"])
