"""Diagnostic for the owner's open decision on the BOOK target definition.

Question: on a ~65%-vol book, how often does the charter's vol-percentile
label (forward 10-day RV above its trailing 1-year 75th percentile) read
"calm" (0) while the book nonetheless suffers a large 10-day drawdown?
And how often does the operational persistence rule fire ahead of those
drawdowns? Frozen data only (active book snapshot + SPY for comparison);
no model, no threshold change — this informs a decision, it does not make
one. Logged as an experiment run because it is an evaluation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regime import baselines, book, data, experiment_log, labels  # noqa: E402

SEED = 20260813


def table(close: pd.Series, cuts: list[float]) -> dict:
    lab = labels.build_labels(close)
    lab["persist"] = baselines.persistence(close)
    d = lab.dropna(subset=["label", "mdd_fwd", "persist"])
    y = d["label"].to_numpy()
    mdd = d["mdd_fwd"].to_numpy()
    s = d["persist"].to_numpy()
    out = {"n_rows": int(len(d)), "effective_n_approx": round(len(d) / labels.HORIZON, 1),
           "base_rate": round(float(y.mean()), 3), "persistence_on_frac": round(float(s.mean()), 3)}
    for c in cuts:
        big = mdd <= c
        out[f"mdd<={c:+.0%}"] = {
            "days": int(big.sum()),
            "share_of_all_days": round(float(big.mean()), 3),
            "label_recall": round(float(y[big].mean()), 3) if big.any() else None,   # P(label=1 | big DD)
            "persistence_recall": round(float(s[big].mean()), 3) if big.any() else None,
            "P(big DD | label=0)": round(float(big[y == 0].mean()), 3),
            "P(big DD | label=1)": round(float(big[y == 1].mean()), 3),
        }
    return out


def main() -> None:
    bk = book.load_book_panel()["BOOK_close"]
    spy = data.load_dev_panel()["SPY_close"]
    res = {"book_composition1": table(bk, [-0.10, -0.15, -0.20, -0.30]),
           "spy": table(spy, [-0.03, -0.05, -0.10])}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "book_label_diagnostic.json").write_text(json.dumps(res, indent=2) + "\n")
    experiment_log.log_run(
        {"strategy": "book_label_diagnostic", "type": "diagnostic", "universe": "book+spy"},
        {u: {k: v for k, v in r.items() if not isinstance(v, dict)} for u, r in res.items()},
        seed=SEED, data_snapshot_hash=book.book_snapshot_hash(),
        notes="label vs forward-drawdown cross-tab; informs the owner's target-definition decision, changes nothing")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
