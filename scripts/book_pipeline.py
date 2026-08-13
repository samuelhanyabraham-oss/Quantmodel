"""Book (portfolio) regime pipeline: build | evaluate | predict.

Points the exact same machinery at the owner's actual holdings:
- labels: forward 10-day realized vol of the BOOK vs its own trailing
  1-year 75th percentile, plus the book's forward max drawdown
- features: book RV dynamics + the market panel (VIX term structure,
  credit proxy, SPY state) — the book is high-beta to exactly those
- validation: same purged walk-forward, same costs, same logging

`evaluate` reports honest walk-forward numbers vs the persistence baseline
computed ON THE BOOK. `predict` emits today's probability, rank, hedge band,
book beta vs SPY, and a beta-scaled hedge-notional hint, and appends to
forward_test.jsonl before the label resolves. Bands, never orders.
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

from regime import baselines, book, data, evaluate, experiment_log, labels, models
from regime.bands import hedge_band_from_rank
from regime.labels import realized_vol_trailing
from regime.validation import purged_walk_forward

SEED = 20260813
RANK_WINDOW = 126
BOOK_FEATURES = [
    "brv10", "brv21", "brv_ratio_10_63", "brv10_chg5", "bdd_from_peak63",
    "vix", "vix_slope", "vix_chg5", "credit_ratio_chg21", "spy_rv10", "beta63",
]


def build_dataset() -> pd.DataFrame:
    bk = book.load_book_panel()
    mkt = data.load_dev_panel()
    df = bk.join(mkt, how="inner")
    close = df["BOOK_close"]
    spy = df["SPY_close"]

    f = pd.DataFrame(index=df.index)
    f["brv10"] = realized_vol_trailing(close, 10)
    f["brv21"] = realized_vol_trailing(close, 21)
    f["brv_ratio_10_63"] = f["brv10"] / realized_vol_trailing(close, 63)
    f["brv10_chg5"] = f["brv10"] - f["brv10"].shift(5)
    f["bdd_from_peak63"] = close / close.rolling(63).max() - 1.0
    f["vix"] = df["VIX_close"]
    f["vix_slope"] = df["VIX_close"] / df["VIX3M_close"] - 1.0
    f["vix_chg5"] = df["VIX_close"].pct_change(5)
    f["credit_ratio_chg21"] = (df["HYG_close"] / df["LQD_close"]).pct_change(21)
    f["spy_rv10"] = realized_vol_trailing(spy, 10)
    br, sr = np.log(close).diff(), np.log(spy).diff()
    f["beta63"] = br.rolling(63).cov(sr) / sr.rolling(63).var()

    lab = labels.build_labels(close)
    out = df.join(lab).join(f)
    out["fwd_ret_10"] = close.shift(-labels.HORIZON) / close - 1.0
    return out


def cmd_evaluate() -> None:
    df = build_dataset().dropna()
    folds = purged_walk_forward(len(df))
    test_rows = np.concatenate([f.test_idx for f in folds])
    test_df = df.iloc[test_rows]
    snap = book.book_snapshot_hash()

    persist = baselines.persistence(df["BOOK_close"]).iloc[test_rows]
    res_p = evaluate.evaluate_strategy(persist, persist, test_df, name="book_persistence", seed=SEED)
    experiment_log.log_run(
        {"strategy": "book_persistence", "type": "baseline", "universe": "book"},
        res_p, seed=SEED, data_snapshot_hash=snap, notes="book walk-forward")

    probs = models.walk_forward_probs(df[BOOK_FEATURES], df["label"], "logistic", seed=SEED)
    p_te = probs.iloc[test_rows]
    roll_q = probs.rolling(RANK_WINDOW, min_periods=60).quantile(0.70)
    s_te = (probs >= roll_q).astype(float).iloc[test_rows]
    res_m = evaluate.evaluate_strategy(p_te, s_te, test_df, name="book_logistic_adaptive", seed=SEED)
    diff, p_raw = evaluate.auc_diff_pvalue(
        test_df["label"].to_numpy(), p_te.to_numpy(), persist.to_numpy(dtype=float), seed=SEED)
    res_m["auc_minus_persistence"] = round(diff, 4)
    res_m["p_raw_vs_persistence"] = round(p_raw, 4)
    experiment_log.log_run(
        {"strategy": "book_logistic_adaptive", "type": "model", "universe": "book",
         "features": BOOK_FEATURES}, res_m, seed=SEED, data_snapshot_hash=snap,
        notes="book walk-forward, adaptive q70 operating point")
    M = experiment_log.run_count()
    res_m["run_count_M"] = M
    res_m["p_bonferroni"] = round(min(1.0, p_raw * M), 4)

    out = {"book_persistence": res_p, "book_logistic_adaptive": res_m}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "book_walk_forward.json").write_text(json.dumps(out, indent=2) + "\n")
    for k, r in out.items():
        print(f"{k:24s} AUC={r['auc']:.3f} {r['auc_ci90']} Brier={r['brier']:.3f} "
              f"NPS={r['nps_net_ann']:+.4f} on={r['hedge_on_frac']:.2f} FNrate={r['false_negative_rate']:.2f}")
    print(f"model vs book-persistence: AUC diff {res_m['auc_minus_persistence']:+.4f} "
          f"p_raw={res_m['p_raw_vs_persistence']} M={M} p_adj={res_m['p_bonferroni']}")


def cmd_predict() -> None:
    df = build_dataset()
    train = df.dropna()
    cut = int(len(train) * (1 - models.CAL_FRAC))
    Xtr = train[BOOK_FEATURES].to_numpy(dtype=float)
    ytr = train["label"].to_numpy(dtype=float)
    scaler = StandardScaler().fit(Xtr[:cut])
    model = models.make_model("logistic", SEED)
    model.fit(scaler.transform(Xtr[:cut]), ytr[:cut])
    calibrate = models._fit_sigmoid_calibrator(
        model.predict_proba(scaler.transform(Xtr[cut:]))[:, 1], ytr[cut:])

    scorable = df[BOOK_FEATURES].dropna()
    recent = scorable.iloc[-(RANK_WINDOW + 1):]
    probs = calibrate(model.predict_proba(scaler.transform(recent.to_numpy(dtype=float)))[:, 1])
    p_today = float(probs[-1])
    rank = float(np.mean(probs[:-1] <= p_today))
    beta = float(df["beta63"].dropna().iloc[-1])
    brv = float(df["brv10"].dropna().iloc[-1])
    thresh = float(df["rv_thresh"].dropna().iloc[-1])

    # OPERATIONAL SIGNAL = the persistence rule on the book, not the model.
    # Book walk-forward (results/book_walk_forward.json): the fitted model
    # scored AUC 0.38 (anti-skill) while book-persistence scored 0.64 with
    # NPS +18%/yr — on a book whose 10-day drawdowns dwarf hedge bleed, the
    # simple rule is the only defensible signal. Model prob/rank stay in the
    # record as diagnostics and for the forward test.
    in_regime = brv > thresh
    band = hedge_band_from_rank(0.75 if in_regime else 0.0)

    entry = {
        "asof": str(recent.index[-1].date()),
        "universe": "book",
        "signal_rule": "book_persistence (rv10 > trailing 1y q75)",
        "in_regime": bool(in_regime),
        "signal": int(in_regime),
        "band_lo": band[0], "band_hi": band[1],
        "model_prob_diagnostic": round(p_today, 4),
        "model_rank_diagnostic": round(rank, 4),
        "book_rv10_ann": round(brv, 4),
        "book_regime_thresh": round(thresh, 4),
        "book_beta63_vs_spy": round(beta, 2),
        "spy_hedge_notional_per_100k_book": round(1e5 * (band[0] + band[1]) / 2 * beta),
        "config": "book-v1 persistence-operational, logistic diagnostic",
        "label_resolves_after": str((recent.index[-1] + pd.tseries.offsets.BDay(labels.HORIZON)).date()),
    }
    with open(ROOT / "forward_test.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")
    experiment_log.log_run(
        {"strategy": "book_forward_prediction", "config": entry["config"]},
        {k: entry[k] for k in ["model_prob_diagnostic", "model_rank_diagnostic",
                               "signal", "in_regime", "book_beta63_vs_spy"]},
        seed=SEED, data_snapshot_hash=book.book_snapshot_hash(),
        notes=f"book forward-test prediction asof {entry['asof']}")
    print(json.dumps(entry, indent=2))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "predict"
    if cmd == "build":
        print(json.dumps(book.build_book_snapshot(), indent=2))
    elif cmd == "evaluate":
        cmd_evaluate()
    elif cmd == "predict":
        cmd_predict()
    else:
        raise SystemExit("usage: book_pipeline.py [build|evaluate|predict]")
