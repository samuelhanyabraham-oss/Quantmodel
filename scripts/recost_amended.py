"""Re-cost every comparison the 2026-09-30 cost-model amendment touches
(bleed 2 -> 4 bp/day), on the long panel and the composition-1 book.

The strategies are re-derived deterministically (same seeds, same frozen
data, same pre-registered specs) and scored under BOTH the original and the
amended cost, so the record shows the change is purely the yardstick's.
Each re-derivation is a run and is logged. Nothing frozen is modified;
the original results files stay as written.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regime import book, costs, evaluate, experiment_log, long_history, longrun, models, stats  # noqa: E402
from regime.validation import purged_walk_forward  # noqa: E402

SEED = 20260813
L3 = ["rv10", "vix", "rv_ratio_10_63"]
L4 = ["rv10", "rv21", "rv63", "rv10_chg5", "rv_ratio_10_63", "volofvol21", "vix", "vix_chg5", "vrp",
      "qqq_rv10", "iwm_rv10", "dd_from_peak63", "ret21", "ret5", "skew63", "range5"]


def nps_at(signal, r, bleed):
    s = np.asarray(signal, float); r = np.asarray(r, float)
    daily = s * np.maximum(0.0, -r) * costs.CAPTURE / 10.0 - bleed * s - costs.C_SWITCH * np.abs(np.diff(s, prepend=s[0]))
    return float(np.nanmean(daily) * costs.ANN_DAYS)


def both(signal, r):
    orig, new = nps_at(signal, r, costs.C_BLEED_ORIGINAL), nps_at(signal, r, costs.C_BLEED)
    tv = stats.timing_value_test(signal, r, lambda s_, r_: nps_at(s_, r_, costs.C_BLEED), seed=SEED)
    return {"nps_2bp": round(orig, 5), "nps_4bp": round(new, 5), "hedge_on": round(float(np.mean(signal)), 4),
            "timing_excess_4bp": tv["excess_over_null"], "timing_p_4bp": tv["p_one_sided"]}


def adaptive_signal(probs):
    roll_q = probs.rolling(126, min_periods=60).quantile(0.70)
    return (probs >= roll_q).astype(float)


def main() -> None:
    out = {"amendment": "C_BLEED 0.0002 -> 0.0004 (2026-09-30)", "long_panel": {}, "book_c1": {}}
    snap = long_history.long_snapshot_hash()

    # --- long panel: baselines on the L2 rows; models on their own pre-registered rows ---
    df = longrun.long_dataset()
    folds = purged_walk_forward(len(df), n_folds=10); te = np.concatenate([f.test_idx for f in folds]); d = df.iloc[te]
    r = d["fwd_ret_10"].to_numpy()
    out["long_panel"]["always_hedged"] = both(np.ones(len(d)), r)
    for n in ["persistence", "trailing_pctl"]:
        out["long_panel"][n] = both(d[n].to_numpy(float), r)
    for tag, feats, kinds in [("L3", L3, ["logistic"]), ("L4", L4, ["logistic", "gbm"])]:
        dfm = longrun.long_dataset(feats)
        fm = purged_walk_forward(len(dfm), n_folds=10); tem = np.concatenate([f.test_idx for f in fm]); dm = dfm.iloc[tem]
        rm = dm["fwd_ret_10"].to_numpy()
        for kind in kinds:
            probs = models.walk_forward_probs(dfm[feats], dfm["label"], kind, seed=SEED, n_folds=10)
            sig = adaptive_signal(probs).iloc[tem].to_numpy()
            res = both(sig, rm)
            res["auc"] = round(float(evaluate.auc(dm["label"].to_numpy(), probs.iloc[tem].to_numpy())), 4)
            out["long_panel"][f"{tag}_{kind}"] = res
            experiment_log.log_run({"strategy": f"{tag}_{kind}_recost", "type": "recost", "panel": "long_v1", "features": feats},
                                   {k: v for k, v in res.items()}, seed=SEED, data_snapshot_hash=snap,
                                   notes="re-derivation of a pre-registered run for the 4 bp cost amendment; AUC unchanged by construction")
        out["long_panel"][f"{tag}_persistence_same_rows"] = both(dm["persistence"].to_numpy(float), rm)

    # --- book composition 1 persistence (walk-forward rows as in book_pipeline) ---
    bk = book.load_book_panel(1)["BOOK_close"]
    from regime import baselines, labels
    b = pd.DataFrame({"close": bk}).join(labels.build_labels(bk))
    b["fwd_ret_10"] = bk.shift(-labels.HORIZON) / bk - 1.0
    b["persistence"] = baselines.persistence(bk)
    b = b.dropna()
    fb = purged_walk_forward(len(b)); teb = np.concatenate([f.test_idx for f in fb]); db = b.iloc[teb]
    out["book_c1"]["persistence"] = both(db["persistence"].to_numpy(float), db["fwd_ret_10"].to_numpy())
    out["book_c1"]["always_hedged"] = both(np.ones(len(db)), db["fwd_ret_10"].to_numpy())
    experiment_log.log_run({"strategy": "recost_summary", "type": "recost"}, {"n_strategies": len(out["long_panel"]) + len(out["book_c1"])},
                           seed=SEED, data_snapshot_hash=snap, notes="cost-model amendment 2->4 bp: recost of all affected comparisons")
    (ROOT / "results" / "recost_4bp.json").write_text(json.dumps(out, indent=2) + "\n")
    for grp, rows in out.items():
        if isinstance(rows, dict):
            print(f"== {grp}")
            for n, v in rows.items():
                print(f"  {n:28s} 2bp {v['nps_2bp']:+.4f}  4bp {v['nps_4bp']:+.4f}  on {v['hedge_on']:.2f}  timing@4bp {v['timing_excess_4bp']:+.4f} p={v['timing_p_4bp']}" + (f"  AUC {v['auc']}" if 'auc' in v else ""))


if __name__ == "__main__":
    main()
