"""Phase 6: the ONE pre-registered holdout evaluation (docs/freeze.md).

Authorization: data/HOLDOUT_UNLOCK.json (owner, 2026-08-13). This script is
written to run once on the real holdout. --dry-run validates the entire code
path on a pseudo-split inside the dev sample and never touches the holdout.

Frozen choices exercised here (nothing else):
- model gbm_small4 on [rv10, vix, vix_slope, rv_ratio_10_63], seed 20260813
- fit on first 75% of usable dev rows, Platt-calibrate on the chronological
  tail 25% (mirrors walk_forward_probs fold semantics exactly)
- last EMBARGO dev rows dropped from training (train labels already cannot
  see past the boundary because they were built on the dev panel alone)
- operating threshold 0.20; baselines evaluated identically
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.preprocessing import StandardScaler

from regime import baselines, data, evaluate, experiment_log, features, labels, models
from regime.validation import EMBARGO

SEED = 20260813
SMALL = ["rv10", "vix", "vix_slope", "rv_ratio_10_63"]
THRESHOLD = 0.20


def frozen_probs(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> np.ndarray:
    cut = int(len(X_train) * (1 - models.CAL_FRAC))
    Xv, yv = X_train.to_numpy(dtype=float), y_train.to_numpy(dtype=float)
    scaler = StandardScaler().fit(Xv[:cut])
    model = models.make_model("gbm", SEED)
    model.fit(scaler.transform(Xv[:cut]), yv[:cut])
    raw_cal = model.predict_proba(scaler.transform(Xv[cut:]))[:, 1]
    calibrate = models._fit_sigmoid_calibrator(raw_cal, yv[cut:])
    raw_te = model.predict_proba(scaler.transform(X_test.to_numpy(dtype=float)))[:, 1]
    return calibrate(raw_te)


def main(dry_run: bool) -> None:
    dev_panel = data.load_dev_panel()
    if dry_run:
        # pseudo-split inside dev: last 250 rows play the holdout. Never
        # reads the holdout file.
        full_panel = dev_panel
        boundary = dev_panel.index[-251]
        tag = "DRY-RUN (pseudo-holdout inside dev)"
    else:
        holdout_panel = data.load_holdout_panel(i_have_explicit_permission=True)
        full_panel = pd.concat([dev_panel, holdout_panel])
        boundary = pd.Timestamp(data.load_manifest()["holdout_boundary_date"])
        tag = "HOLDOUT"

    # Training data: labels built from the dev panel ALONE (labels of late
    # dev rows are NaN there because their forward window has no data — that
    # is the purge). Then drop EMBARGO more rows.
    dev_lab = labels.build_labels(dev_panel["SPY_close"])
    dev_feat = features.build_features(dev_panel)
    train_df = dev_panel.join(dev_lab).join(dev_feat).dropna()
    if dry_run:
        train_df = train_df[train_df.index <= boundary]
    train_df = train_df.iloc[:-EMBARGO]

    # Evaluation data: labels/features on the full panel (trailing windows
    # legitimately use pre-boundary history), restricted to post-boundary rows.
    full_lab = labels.build_labels(full_panel["SPY_close"])
    full_feat = features.build_features(full_panel)
    full_df = full_panel.join(full_lab).join(full_feat)
    full_df["fwd_ret_10"] = (
        full_panel["SPY_close"].shift(-labels.HORIZON) / full_panel["SPY_close"] - 1.0
    )
    eval_df = full_df[full_df.index > boundary].dropna()

    p = pd.Series(
        frozen_probs(train_df[SMALL], train_df["label"], eval_df[SMALL]),
        index=eval_df.index,
    )
    s = (p >= THRESHOLD).astype(float)

    results = {}
    res = evaluate.evaluate_strategy(p, s, eval_df, name="gbm_small4_frozen", seed=SEED)
    base_sigs = {
        "persistence": baselines.persistence(full_df["SPY_close"]),
        "term_structure": baselines.term_structure(full_df["VIX_close"], full_df["VIX3M_close"]),
        "trailing_pctl": baselines.trailing_pctl_rule(full_df["SPY_close"]),
    }
    persist_eval = base_sigs["persistence"].reindex(eval_df.index).astype(float)
    diff, p_raw = evaluate.auc_diff_pvalue(
        eval_df["label"].to_numpy(), p.to_numpy(), persist_eval.to_numpy(), seed=SEED
    )
    res["auc_minus_persistence"] = round(diff, 4)
    res["p_raw_vs_persistence"] = round(p_raw, 4)
    results["gbm_small4_frozen"] = res

    for name, sig in base_sigs.items():
        sig_e = sig.reindex(eval_df.index).astype(float)
        results[name] = evaluate.evaluate_strategy(sig_e, sig_e, eval_df, name=name, seed=SEED)

    snap_hash = data.snapshot_hash() if dry_run else data.load_manifest()["holdout_sha256"]
    for name, r in results.items():
        experiment_log.log_run(
            {"strategy": name, "type": "holdout_eval" if not dry_run else "holdout_dry_run",
             "threshold": THRESHOLD, "features": SMALL if "gbm" in name else None},
            r, seed=SEED, data_snapshot_hash=snap_hash,
            notes=f"{tag}: pre-registered evaluation per docs/freeze.md",
        )

    M = experiment_log.run_count()
    res["run_count_M"] = M
    res["p_bonferroni"] = round(min(1.0, res["p_raw_vs_persistence"] * M), 4)
    res["skill_claimed"] = bool(res["p_bonferroni"] < 0.10)

    out = ROOT / "results" / ("holdout_dry_run.json" if dry_run else "holdout.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")

    print(f"== {tag} ==  eval rows: {len(eval_df)}  ({eval_df.index.min().date()} -> {eval_df.index.max().date()})")
    for name, r in results.items():
        print(f"{name:20s} AUC={r['auc']:.3f} {r['auc_ci90']} Brier={r['brier']:.3f} "
              f"NPS={r['nps_net_ann']:+.4f} on={r['hedge_on_frac']:.2f} "
              f"FNrate={r['false_negative_rate']:.2f}")
    print(f"model minus persistence AUC: {res['auc_minus_persistence']:+.4f}  "
          f"p_raw={res['p_raw_vs_persistence']}  M={M}  p_adj={res['p_bonferroni']}  "
          f"skill_claimed={res['skill_claimed']}")


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv)
