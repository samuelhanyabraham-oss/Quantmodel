"""Walk-forward evaluation of baselines and models on the dev snapshot.

Every invocation logs every strategy it evaluates to experiments.jsonl.
Results land in results/walk_forward.json for the report. Holdout is never
touched here (load_dev_panel only).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regime import baselines, data, evaluate, experiment_log, features, labels, models
from regime.validation import purged_walk_forward

SEED = 20260813


def build_dataset() -> pd.DataFrame:
    panel = data.load_dev_panel()
    lab = labels.build_labels(panel["SPY_close"])
    feats = features.build_features(panel)
    df = panel.join(lab).join(feats)
    df["fwd_ret_10"] = panel["SPY_close"].shift(-labels.HORIZON) / panel["SPY_close"] - 1.0
    df = df.dropna()
    return df


def main() -> None:
    df = build_dataset()
    snap = data.snapshot_hash()
    folds = purged_walk_forward(len(df))
    test_rows = np.concatenate([f.test_idx for f in folds])
    test_df = df.iloc[test_rows]

    results = {}

    # --- baselines (binary; prob == signal) ---
    base_signals = {
        "persistence": baselines.persistence(df["SPY_close"]),
        "term_structure": baselines.term_structure(df["VIX_close"], df["VIX3M_close"]),
        "trailing_pctl": baselines.trailing_pctl_rule(df["SPY_close"]),
    }
    for name, sig in base_signals.items():
        sig_te = sig.iloc[test_rows]
        res = evaluate.evaluate_strategy(sig_te, sig_te, test_df, name=name, seed=SEED)
        results[name] = res
        experiment_log.log_run(
            {"strategy": name, "type": "baseline", "folds": len(folds)},
            res, seed=SEED, data_snapshot_hash=snap,
            notes="walk-forward test folds, dev snapshot",
        )

    # --- models (probabilities; threshold applied later in Phase 5, but a
    # provisional 0.5 signal is evaluated here for the same metric set) ---
    X = df[features.FEATURE_NAMES]
    y = df["label"]
    for kind in ["logistic", "gbm"]:
        probs = models.walk_forward_probs(X, y, kind, seed=SEED)
        p_te = probs.iloc[test_rows]
        s_te = (p_te >= 0.5).astype(float)
        res = evaluate.evaluate_strategy(p_te, s_te, test_df, name=kind, seed=SEED)
        # skill-above-persistence: AUC diff + one-sided bootstrap p
        diff, p_raw = evaluate.auc_diff_pvalue(
            test_df["label"].to_numpy(),
            p_te.to_numpy(),
            base_signals["persistence"].iloc[test_rows].to_numpy(dtype=float),
            seed=SEED,
        )
        res["auc_minus_persistence"] = round(diff, 4)
        res["p_raw_vs_persistence"] = round(p_raw, 4)
        results[kind] = res
        experiment_log.log_run(
            {"strategy": kind, "type": "model", "features": features.FEATURE_NAMES,
             "folds": len(folds), "calibration": "platt-chrono-tail"},
            res, seed=SEED, data_snapshot_hash=snap,
            notes="walk-forward test folds, dev snapshot",
        )
        results.setdefault("_probs", {})[kind] = {
            str(d.date()): round(float(v), 4) for d, v in p_te.items()
        }

    m = experiment_log.run_count()
    for kind in ["logistic", "gbm"]:
        p_raw = results[kind]["p_raw_vs_persistence"]
        results[kind]["run_count_M"] = m
        results[kind]["p_bonferroni"] = round(min(1.0, p_raw * m), 4)

    out = ROOT / "results" / "walk_forward.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")
    for name, r in results.items():
        if name.startswith("_"):
            continue
        print(f"{name:14s} AUC={r['auc']:.3f} {r['auc_ci90']} Brier={r['brier']:.3f} "
              f"NPS={r['nps_net_ann']:+.4f} on={r['hedge_on_frac']:.2f} "
              f"lead_med={r['lead_median_lead_days']}")
    print(f"total logged runs M={m}")


if __name__ == "__main__":
    main()
