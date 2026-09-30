"""Erratum audit (2026-09-30): re-score the persistence and trailing-percentile
baselines on the ORIGINAL frozen data (panel_full.csv = snapshot v2, and
book_v1.csv), on exactly the folds the original scripts used, two ways:

  buggy  — baseline computed on the dropna'd frame (what run_experiments.py
           and book_pipeline.py evaluate did): threshold undefined for the
           first 261 rows -> signal silently 0 there
  fixed  — baseline computed on the full close, aligned to the same rows

Reports both, the number of test rows affected, and the AUC/NPS gap. Logged
as a run; changes nothing frozen. The results files from the original runs
are left as they are (they are the record); REPORT.md carries the erratum.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regime import baselines, book, data, evaluate, experiment_log, features, labels  # noqa: E402
from regime.validation import purged_walk_forward  # noqa: E402

SEED = 20260813


def old_style(close):
    """What the old code did: (rv > thresh).astype(float) with 0 in warm-up."""
    rv = labels.realized_vol_trailing(close, labels.HORIZON)
    th = rv.rolling(labels.TRAIL_WINDOW, min_periods=labels.TRAIL_WINDOW).quantile(labels.PCTL)
    return (rv > th).astype(float)


def audit_spy(which: str = "full") -> dict:
    m = data.load_manifest()
    path = data.SNAP_DIR / f"panel_{which}.csv"
    assert data.sha256_file(path) == m[f"{which}_sha256"]
    panel = pd.read_csv(path, index_col="date", parse_dates=["date"])
    lab = labels.build_labels(panel["SPY_close"]); feats = features.build_features(panel)
    df = panel.join(lab).join(feats)
    df["fwd_ret_10"] = panel["SPY_close"].shift(-labels.HORIZON) / panel["SPY_close"] - 1.0
    fixed = {"persistence": baselines.persistence(panel["SPY_close"]),
             "trailing_pctl": baselines.trailing_pctl_rule(panel["SPY_close"])}
    df = df.dropna()
    folds = purged_walk_forward(len(df))
    te = np.concatenate([f.test_idx for f in folds]); test_df = df.iloc[te]
    out = {"snapshot": f"panel_{which}.csv", "rows": int(len(df)), "test_rows": int(len(te))}
    for name in fixed:
        buggy_sig = (old_style(df["SPY_close"]) if name == "persistence" else
                     (lambda c: (labels.realized_vol_trailing(c, 10) > labels.realized_vol_trailing(c, 10).rolling(252, min_periods=252).quantile(0.60)).astype(float))(df["SPY_close"])).iloc[te]
        fixed_sig = fixed[name].reindex(df.index).iloc[te]
        rb = evaluate.evaluate_strategy(buggy_sig, buggy_sig, test_df, name=f"{name}_buggy", seed=SEED)
        rf = evaluate.evaluate_strategy(fixed_sig, fixed_sig, test_df, name=f"{name}_fixed", seed=SEED)
        forced_off = int(((buggy_sig == 0) & (fixed_sig == 1)).sum())
        out[name] = {"buggy": {k: rb[k] for k in ("auc", "auc_ci90", "nps_net_ann", "hedge_on_frac", "false_negative_rate")},
                     "fixed": {k: rf[k] for k in ("auc", "auc_ci90", "nps_net_ann", "hedge_on_frac", "false_negative_rate")},
                     "test_rows_forced_off_by_bug": forced_off,
                     "auc_understated_by": round(rf["auc"] - rb["auc"], 4)}
    return out


def audit_book() -> dict:
    m = book.load_book_manifest()
    path = book.BOOK_SNAP
    assert book._sha256(path) == m["book_sha256"]
    bk = pd.read_csv(path, index_col="date", parse_dates=["date"])
    mkt = pd.read_csv(data.SNAP_DIR / "panel_full.csv", index_col="date", parse_dates=["date"])
    df = bk.join(mkt, how="inner")
    close = df["BOOK_close"]
    df = df.join(labels.build_labels(close))
    df["fwd_ret_10"] = close.shift(-labels.HORIZON) / close - 1.0
    # the original book dataset also carried 11 features incl. beta63 (63-day warm-up) and vix_slope
    from regime.labels import realized_vol_trailing
    f = pd.DataFrame(index=df.index)
    f["brv10"] = realized_vol_trailing(close, 10); f["brv21"] = realized_vol_trailing(close, 21)
    f["brv_ratio_10_63"] = f["brv10"] / realized_vol_trailing(close, 63)
    f["brv10_chg5"] = f["brv10"] - f["brv10"].shift(5); f["bdd_from_peak63"] = close / close.rolling(63).max() - 1.0
    f["vix"] = df["VIX_close"]; f["vix_slope"] = df["VIX_close"] / df["VIX3M_close"] - 1.0
    f["vix_chg5"] = df["VIX_close"].pct_change(5); f["credit_ratio_chg21"] = (df["HYG_close"] / df["LQD_close"]).pct_change(21)
    f["spy_rv10"] = realized_vol_trailing(df["SPY_close"], 10)
    br, sr = np.log(close).diff(), np.log(df["SPY_close"]).diff(); f["beta63"] = br.rolling(63).cov(sr) / sr.rolling(63).var()
    df = df.join(f)
    fixed_full = baselines.persistence(close)
    df = df.dropna()
    folds = purged_walk_forward(len(df))
    te = np.concatenate([f_.test_idx for f_ in folds]); test_df = df.iloc[te]
    buggy_sig = old_style(df["BOOK_close"]).iloc[te]
    fixed_sig = fixed_full.reindex(df.index).iloc[te]
    rb = evaluate.evaluate_strategy(buggy_sig, buggy_sig, test_df, name="book_persistence_buggy", seed=SEED)
    rf = evaluate.evaluate_strategy(fixed_sig, fixed_sig, test_df, name="book_persistence_fixed", seed=SEED)
    return {"snapshot": "book_v1.csv", "rows": int(len(df)), "test_rows": int(len(te)),
            "book_persistence": {"buggy": {k: rb[k] for k in ("auc", "auc_ci90", "nps_net_ann", "hedge_on_frac", "false_negative_rate")},
                                 "fixed": {k: rf[k] for k in ("auc", "auc_ci90", "nps_net_ann", "hedge_on_frac", "false_negative_rate")},
                                 "test_rows_forced_off_by_bug": int(((buggy_sig == 0) & (fixed_sig == 1)).sum()),
                                 "auc_understated_by": round(rf["auc"] - rb["auc"], 4)}}


def main() -> None:
    out = {"spy_dev_preholdout": audit_spy("dev"), "spy_full_v2": audit_spy("full"), "book": audit_book()}
    (ROOT / "results" / "baseline_bug_audit.json").write_text(json.dumps(out, indent=2) + "\n")
    experiment_log.log_run({"strategy": "baseline_bug_audit", "type": "erratum", "panels": ["panel_full v2", "book_v1"]},
                           {"spy_dev_persistence_auc_buggy": out["spy_dev_preholdout"]["persistence"]["buggy"]["auc"],
                            "spy_dev_persistence_auc_fixed": out["spy_dev_preholdout"]["persistence"]["fixed"]["auc"],
                            "spy_full_persistence_auc_buggy": out["spy_full_v2"]["persistence"]["buggy"]["auc"],
                            "spy_full_persistence_auc_fixed": out["spy_full_v2"]["persistence"]["fixed"]["auc"],
                            "book_persistence_auc_buggy": out["book"]["book_persistence"]["buggy"]["auc"],
                            "book_persistence_auc_fixed": out["book"]["book_persistence"]["fixed"]["auc"]},
                           seed=SEED, data_snapshot_hash=data.load_manifest()["full_sha256"],
                           notes="erratum: baselines were computed on truncated closes in run_experiments/book_pipeline; re-scored both ways")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
