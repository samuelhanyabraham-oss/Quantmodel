"""Score the forward-test record against resolved labels.

Reads forward_test.jsonl (append-only; duplicates de-duplicated on read,
never deleted), resolves every entry whose 10-day forward window has
closed in the ACTIVE snapshot, and reports — per entry and in aggregate —
the realized label, forward RV, forward max drawdown, and what the
persistence baseline said on the same day. Writes results/forward_test_record.json
and logs the scoring as an experiment run (it is one).

It does NOT claim skill. The freeze-v2 rule stands: a claim needs the
accumulated record to beat persistence under block bootstrap + Bonferroni
at whatever M then stands; that machinery is only invoked once at least
MIN_ROWS_FOR_TEST resolved rows exist per universe, and even then the
output here is a p-value, not a verdict.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regime import baselines, book, data, evaluate, experiment_log, labels  # noqa: E402
from regime.costs import net_protection_score  # noqa: E402

SEED = 20260813
FORWARD_LOG = ROOT / "forward_test.jsonl"
OUT = ROOT / "results" / "forward_test_record.json"
MIN_ROWS_FOR_TEST = 60  # ~6 effective obs at a 10-day label; below this no test is even attempted


def read_forward_log() -> tuple[list[dict], int]:
    """Entries de-duplicated on (asof, config, universe); returns (entries, n_dupes)."""
    seen, out, dupes = set(), [], 0
    if not FORWARD_LOG.exists():
        return out, 0
    for line in FORWARD_LOG.read_text().splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        key = (e["asof"], e["config"], e.get("universe"))
        if key in seen:
            dupes += 1
            continue
        seen.add(key)
        out.append(e)
    return out, dupes


def _universe_frames() -> dict[str, pd.DataFrame]:
    frames = {}
    spy = data.load_dev_panel()["SPY_close"]
    frames[None] = _resolve_frame(spy)
    if (ROOT / "data" / "BOOK_MANIFEST.json").exists():
        frames["book"] = _resolve_frame(book.load_book_panel()["BOOK_close"])
    return frames


def _resolve_frame(close: pd.Series) -> pd.DataFrame:
    lab = labels.build_labels(close)
    lab["persistence"] = baselines.persistence(close)
    lab["fwd_ret_10"] = close.shift(-labels.HORIZON) / close - 1.0
    lab["latest_bar"] = close.index.max()
    return lab


def score(entries: list[dict], frames: dict[str, pd.DataFrame]) -> list[dict]:
    rows = []
    for e in entries:
        f = frames.get(e.get("universe"))
        if f is None:
            continue
        t = pd.Timestamp(e["asof"])
        if t not in f.index:
            continue
        r = f.loc[t]
        resolved = not pd.isna(r["label"])
        row = {
            "asof": e["asof"], "universe": e.get("universe") or "SPY",
            "config": e["config"], "signal": int(e["signal"]),
            "band": [e["band_lo"], e["band_hi"]],
            "resolved": bool(resolved),
        }
        if resolved:
            y = int(r["label"])
            row.update({
                "label": y,
                "rv_fwd_ann": round(float(r["rv_fwd"]), 4),
                "rv_thresh_ann": round(float(r["rv_thresh"]), 4),
                "mdd_fwd": round(float(r["mdd_fwd"]), 4),
                "fwd_ret_10": round(float(r["fwd_ret_10"]), 4),
                "persistence_signal": int(r["persistence"]),
                "outcome": {(1, 1): "TP", (1, 0): "FP", (0, 1): "FN (unhedged into regime)", (0, 0): "TN"}[
                    (int(e["signal"]), y)],
            })
        rows.append(row)
    return rows


def aggregate(rows: list[dict], frames: dict[str, pd.DataFrame]) -> dict:
    agg = {}
    for uni in sorted({r["universe"] for r in rows}):
        rs = [r for r in rows if r["universe"] == uni and r["resolved"]]
        pend = [r for r in rows if r["universe"] == uni and not r["resolved"]]
        a = {"n_logged": len(rs) + len(pend), "n_resolved": len(rs), "n_pending": len(pend)}
        if rs:
            s = np.array([r["signal"] for r in rs], float)
            y = np.array([r["label"] for r in rs], float)
            ps = np.array([r["persistence_signal"] for r in rs], float)
            ret = np.array([r["fwd_ret_10"] for r in rs], float)
            a.update({
                "confusion": {k: int(sum(1 for r in rs if r["outcome"].startswith(k))) for k in ("TP", "FP", "FN", "TN")},
                "false_negatives_unhedged_into_regime": int(((s == 0) & (y == 1)).sum()),
                "false_positives_bleed_in_calm": int(((s == 1) & (y == 0)).sum()),
                "persistence_agreement": float(np.mean(s == ps)),
                "nps_net_ann_signal": round(net_protection_score(s, ret), 5),
                "nps_net_ann_persistence": round(net_protection_score(ps, ret), 5),
                "effective_n_approx": round(len(rs) / labels.HORIZON, 1),
                "skill_test": "not attempted: n_resolved < MIN_ROWS_FOR_TEST" if len(rs) < MIN_ROWS_FOR_TEST else "see p_raw",
            })
            if len(rs) >= MIN_ROWS_FOR_TEST and len(set(y)) == 2:
                diff, p_raw = evaluate.auc_diff_pvalue(y, s, ps, seed=SEED)
                M = experiment_log.run_count()
                a.update({"auc_minus_persistence": round(diff, 4), "p_raw": round(p_raw, 4),
                          "M": M, "p_bonferroni": round(min(1.0, p_raw * M), 4),
                          "decision_rule": "claim only if p_bonferroni < 0.10 (docs/multiple_testing.md)"})
        agg[uni] = a
    return agg


def main() -> None:
    entries, dupes = read_forward_log()
    frames = _universe_frames()
    rows = score(entries, frames)
    agg = aggregate(rows, frames)
    record = {
        "scored_utc": pd.Timestamp.now("UTC").isoformat(),
        "panel_hash": data.snapshot_hash(),
        "book_hash": book.book_snapshot_hash() if "book" in frames else None,
        "latest_bar": {("SPY" if k is None else k): str(v["latest_bar"].iloc[0].date()) for k, v in frames.items()},
        "duplicate_entries_ignored": dupes,
        "aggregate": agg,
        "rows": rows,
        "claim": "none — forward record too short for any test (see aggregate.skill_test)",
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(record, indent=2) + "\n")
    experiment_log.log_run(
        {"strategy": "forward_test_scoring", "universes": sorted(agg)},
        {u: {k: v for k, v in a.items() if k in ("n_resolved", "n_pending", "nps_net_ann_signal",
                                                   "nps_net_ann_persistence", "p_raw", "p_bonferroni")}
         for u, a in agg.items()},
        seed=SEED, data_snapshot_hash=data.snapshot_hash(),
        notes=f"forward-test scoring; {dupes} duplicate log entries ignored; latest bar {record['latest_bar']}",
    )
    for r in rows:
        tag = r.get("outcome", "pending")
        extra = (f" label={r['label']} rv_fwd={r['rv_fwd_ann']:.2f} vs thr={r['rv_thresh_ann']:.2f} "
                 f"mdd={r['mdd_fwd']:+.3f} persist={r['persistence_signal']}") if r["resolved"] else ""
        print(f"{r['asof']} {r['universe']:5s} signal={r['signal']} band={r['band']} -> {tag}{extra}")
    print(json.dumps(agg, indent=2))
    print(f"duplicates ignored: {dupes}; written {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
