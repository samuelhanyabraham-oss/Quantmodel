"""L3: the ONE pre-registered model check on the long panel.

Spec: docs/long_history_plan.md (committed before this ran, 77f207e).
Freeze-v2 procedure minus vix_slope: logistic C=0.1 balanced, per-fold
scaler + Platt tail-25% calibration (models.walk_forward_probs, unchanged),
features [rv10, vix, rv_ratio_10_63], purged walk-forward 10 folds,
adaptive rank operating point (q70 of trailing 126, min 60), frozen bands.
Compared with persistence and trailing percentile on the same pooled test
rows; one-sided block-bootstrap p vs persistence; Bonferroni over M at run
time. Decision rule: skill only if p_adj < 0.10 AND net NPS beats both
computable baselines. One run; the script refuses to run twice.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regime import baselines, evaluate, experiment_log, features, labels, long_history, models  # noqa: E402
from regime.costs import net_protection_score  # noqa: E402
from regime.validation import purged_walk_forward  # noqa: E402

SEED = 20260813
N_FOLDS = 10
FEATS = ["rv10", "vix", "rv_ratio_10_63"]
RANK_WINDOW, RANK_Q = 126, 0.70
OUT = ROOT / "results" / "long_model_check.json"
STRESS = json.loads((ROOT / "results" / "long_baselines.json").read_text())["stress_windows"]


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT.relative_to(ROOT)} exists — the pre-registered check was already run once")
    panel = long_history.load_long_panel()
    spy = panel["SPY_close"]
    df = panel.join(labels.build_labels(spy)).join(features.build_features(panel)[FEATS])
    df["fwd_ret_10"] = spy.shift(-labels.HORIZON) / spy - 1.0
    df["persistence"] = baselines.persistence(spy)
    df["trailing_pctl"] = baselines.trailing_pctl_rule(spy)
    df = df.dropna(subset=["label", "fwd_ret_10", "persistence", "trailing_pctl", *FEATS])
    snap = long_history.long_snapshot_hash()

    folds = purged_walk_forward(len(df), n_folds=N_FOLDS)
    test_rows = np.concatenate([f.test_idx for f in folds])
    test_df = df.iloc[test_rows]
    probs = models.walk_forward_probs(df[FEATS], df["label"], "logistic", seed=SEED, n_folds=N_FOLDS)
    roll_q = probs.rolling(RANK_WINDOW, min_periods=60).quantile(RANK_Q)
    signal = (probs >= roll_q).astype(float)
    p_te, s_te = probs.iloc[test_rows], signal.iloc[test_rows]

    res = evaluate.evaluate_strategy(p_te, s_te, test_df, name="logistic_small3_adaptive_long", seed=SEED)
    y = test_df["label"].to_numpy(); r = test_df["fwd_ret_10"].to_numpy()
    base = {}
    for n in ["persistence", "trailing_pctl"]:
        b = test_df[n]
        base[n] = evaluate.evaluate_strategy(b, b, test_df, name=n, seed=SEED)
    diff, p_raw = evaluate.auc_diff_pvalue(y, p_te.to_numpy(), test_df["persistence"].to_numpy(dtype=float), seed=SEED)
    always = net_protection_score(np.ones(len(y)), r)
    res.update({"auc_minus_persistence": round(diff, 4), "p_raw_vs_persistence": round(p_raw, 4),
                "timing_value": round(res["nps_net_ann"] - res["hedge_on_frac"] * always, 5)})
    for n in base:
        base[n]["timing_value"] = round(base[n]["nps_net_ann"] - base[n]["hedge_on_frac"] * always, 5)

    per_window = {}
    for k, (a, b) in {**{k: (v, w) for k, (v, w) in [
        ("dotcom_2001_02", ("2001-01-01", "2003-03-31")), ("gfc_2008_09", ("2008-09-01", "2009-03-31")),
        ("flash_crash_2010", ("2010-04-15", "2010-06-30")), ("eurozone_2011", ("2011-07-15", "2011-10-31")),
        ("china_2015", ("2015-08-01", "2015-09-30")), ("volmageddon_2018", ("2018-01-15", "2018-02-28")),
        ("q4_2018", ("2018-10-01", "2018-12-31")), ("covid_2020", ("2020-02-15", "2020-04-30")),
        ("bear_2022", ("2022-01-01", "2022-10-31")), ("tariff_2025", ("2025-03-01", "2025-05-31"))]}}.items():
        d = test_df.loc[a:b]
        if len(d) < 5:
            continue
        sw = signal.loc[d.index].to_numpy(); rw = d["fwd_ret_10"].to_numpy()
        lt = evaluate.lead_times(signal.loc[d.index], d["label"])
        per_window[k] = {"rows": int(len(d)), "model_nps": round(net_protection_score(sw, rw), 5),
                         "persistence_nps": round(net_protection_score(d["persistence"].to_numpy(dtype=float), rw), 5),
                         "model_on": round(float(sw.mean()), 3), "model_missed": lt["n_missed"], "onsets": lt["n_onsets"]}
    per_fold = []
    for k, f in enumerate(folds):
        d = df.iloc[f.test_idx]; sw = signal.iloc[f.test_idx].to_numpy(); rw = d["fwd_ret_10"].to_numpy()
        yy = d["label"].to_numpy(); pp = probs.iloc[f.test_idx].to_numpy()
        per_fold.append({"fold": k, "start": str(d.index[0].date()), "end": str(d.index[-1].date()),
                         "auc": round(float(evaluate.auc(yy, pp)), 3) if len(set(yy)) == 2 else None,
                         "persistence_auc": round(float(evaluate.auc(yy, d["persistence"].to_numpy())), 3) if len(set(yy)) == 2 else None,
                         "model_nps": round(net_protection_score(sw, rw), 5),
                         "persistence_nps": round(net_protection_score(d["persistence"].to_numpy(dtype=float), rw), 5)})

    experiment_log.log_run(
        {"strategy": "logistic_small3_adaptive", "type": "model", "panel": "long_v1", "features": FEATS,
         "folds": N_FOLDS, "operating_point": f"rank q{RANK_Q} w{RANK_WINDOW}", "preregistered": "docs/long_history_plan.md @ 77f207e"},
        res, seed=SEED, data_snapshot_hash=snap, notes="L3 pre-registered long-history model check, single run")
    M = experiment_log.run_count()
    p_adj = min(1.0, p_raw * M)
    beats_net = res["nps_net_ann"] > max(b["nps_net_ann"] for b in base.values())
    verdict = ("SKILL CLAIMED" if (p_adj < 0.10 and beats_net) else "NO SKILL CLAIMED")
    out = {"panel_hash": snap, "spec": "docs/long_history_plan.md @ 77f207e", "rows": int(len(df)),
           "span": [str(df.index[0].date()), str(df.index[-1].date())], "n_folds": len(folds),
           "model": res, "baselines": base, "always_hedged_nps": round(always, 5),
           "M": M, "p_raw": round(p_raw, 4), "p_bonferroni": round(p_adj, 4), "beats_both_baselines_net": bool(beats_net),
           "verdict": verdict, "per_fold": per_fold, "stress_windows": per_window}
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"rows {len(df)} span {out['span']} folds {len(folds)} effN {res['effective_n']}")
    print(f"model      AUC={res['auc']:.3f} {res['auc_ci90']} Brier={res['brier']:.3f} NPS={res['nps_net_ann']:+.4f} on={res['hedge_on_frac']:.2f} "
          f"FNrate={res['false_negative_rate']:.2f} lead={res['lead_median_lead_days']} missed={res['lead_n_missed']}/{res['lead_n_onsets']} timing={res['timing_value']:+.4f}")
    for n, b in base.items():
        print(f"{n:10s} AUC={b['auc']:.3f} {b['auc_ci90']} NPS={b['nps_net_ann']:+.4f} on={b['hedge_on_frac']:.2f} FNrate={b['false_negative_rate']:.2f} timing={b['timing_value']:+.4f}")
    print(f"always hedged NPS {always:+.4f}")
    print(f"AUC diff vs persistence {diff:+.4f}  p_raw={p_raw:.4f}  M={M}  p_adj={p_adj:.4f}  beats both net: {beats_net}  -> {verdict}")
    print("per fold (model AUC / persistence AUC / model NPS / persistence NPS):")
    for f in per_fold:
        print(f"  {f['start']}..{f['end']}  {f['auc']} / {f['persistence_auc']}  {f['model_nps']:+.4f} / {f['persistence_nps']:+.4f}")
    print("stress windows (model NPS / persistence NPS / model on / model missed):")
    for k, w in per_window.items():
        print(f"  {k:18s} {w['model_nps']:+.4f} / {w['persistence_nps']:+.4f}  on={w['model_on']:.2f}  missed={w['model_missed']}/{w['onsets']}")


if __name__ == "__main__":
    main()
