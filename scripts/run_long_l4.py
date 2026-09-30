"""L4: two pre-registered runs on the long panel (docs/long_history_plan.md,
section L4, committed before this ran). Refuses to run twice."""

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
FEATS = ["rv10", "rv21", "rv63", "rv10_chg5", "rv_ratio_10_63", "volofvol21", "vix", "vix_chg5", "vrp",
         "qqq_rv10", "iwm_rv10", "dd_from_peak63", "ret21", "ret5", "skew63", "range5"]
RANK_WINDOW, RANK_Q = 126, 0.70
OUT = ROOT / "results" / "long_l4.json"
STRESS = {"dotcom_2001_02": ("2001-01-01", "2003-03-31"), "gfc_2008_09": ("2008-09-01", "2009-03-31"),
          "flash_crash_2010": ("2010-04-15", "2010-06-30"), "eurozone_2011": ("2011-07-15", "2011-10-31"),
          "china_2015": ("2015-08-01", "2015-09-30"), "volmageddon_2018": ("2018-01-15", "2018-02-28"),
          "q4_2018": ("2018-10-01", "2018-12-31"), "covid_2020": ("2020-02-15", "2020-04-30"),
          "bear_2022": ("2022-01-01", "2022-10-31"), "tariff_2025": ("2025-03-01", "2025-05-31")}


def main() -> None:
    if OUT.exists():
        raise SystemExit("L4 already run once")
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
    y = test_df["label"].to_numpy(); r = test_df["fwd_ret_10"].to_numpy()
    always = net_protection_score(np.ones(len(y)), r)

    out = {"panel_hash": snap, "spec": "docs/long_history_plan.md L4", "rows": int(len(df)),
           "span": [str(df.index[0].date()), str(df.index[-1].date())], "n_folds": len(folds),
           "features": FEATS, "always_hedged_nps": round(always, 5), "baselines": {}, "models": {}}
    for n in ["persistence", "trailing_pctl"]:
        b = test_df[n]
        out["baselines"][n] = evaluate.evaluate_strategy(b, b, test_df, name=n, seed=SEED)

    for kind in ["logistic", "gbm"]:
        probs = models.walk_forward_probs(df[FEATS], df["label"], kind, seed=SEED, n_folds=N_FOLDS)
        roll_q = probs.rolling(RANK_WINDOW, min_periods=60).quantile(RANK_Q)
        signal = (probs >= roll_q).astype(float)
        p_te, s_te = probs.iloc[test_rows], signal.iloc[test_rows]
        res = evaluate.evaluate_strategy(p_te, s_te, test_df, name=f"{kind}_16f_adaptive_long", seed=SEED)
        diff, p_raw = evaluate.auc_diff_pvalue(y, p_te.to_numpy(), test_df["persistence"].to_numpy(dtype=float), seed=SEED)
        res["auc_minus_persistence"] = round(diff, 4); res["p_raw_vs_persistence"] = round(p_raw, 4)
        experiment_log.log_run({"strategy": f"{kind}_16f_adaptive", "type": "model", "panel": "long_v1",
                                "features": FEATS, "folds": N_FOLDS, "operating_point": f"rank q{RANK_Q} w{RANK_WINDOW}",
                                "preregistered": "docs/long_history_plan.md L4"},
                               res, seed=SEED, data_snapshot_hash=snap, notes=f"L4 pre-registered run: {kind}")
        per_fold, per_window = [], {}
        for k, f in enumerate(folds):
            d = df.iloc[f.test_idx]; yy = d["label"].to_numpy(); pp = probs.iloc[f.test_idx].to_numpy()
            sw = signal.iloc[f.test_idx].to_numpy(); rw = d["fwd_ret_10"].to_numpy()
            per_fold.append({"fold": k, "start": str(d.index[0].date()), "end": str(d.index[-1].date()),
                             "auc": round(float(evaluate.auc(yy, pp)), 3), "persistence_auc": round(float(evaluate.auc(yy, d["persistence"].to_numpy())), 3),
                             "model_nps": round(net_protection_score(sw, rw), 5), "persistence_nps": round(net_protection_score(d["persistence"].to_numpy(dtype=float), rw), 5)})
        for k, (a, b) in STRESS.items():
            d = test_df.loc[a:b]
            if len(d) < 5:
                continue
            sw = signal.loc[d.index].to_numpy(); rw = d["fwd_ret_10"].to_numpy(); lt = evaluate.lead_times(signal.loc[d.index], d["label"])
            per_window[k] = {"model_nps": round(net_protection_score(sw, rw), 5),
                             "persistence_nps": round(net_protection_score(d["persistence"].to_numpy(dtype=float), rw), 5),
                             "model_on": round(float(sw.mean()), 3), "model_missed": lt["n_missed"], "onsets": lt["n_onsets"]}
        out["models"][kind] = {"metrics": res, "per_fold": per_fold, "stress_windows": per_window}

    M = experiment_log.run_count()
    for kind, m in out["models"].items():
        res = m["metrics"]; p_adj = min(1.0, res["p_raw_vs_persistence"] * M)
        beats = res["nps_net_ann"] > max(b["nps_net_ann"] for b in out["baselines"].values())
        m["M"] = M; m["p_bonferroni"] = round(p_adj, 4); m["beats_both_baselines_net"] = bool(beats)
        m["verdict"] = "SKILL CLAIMED" if (p_adj < 0.10 and beats) else "NO SKILL CLAIMED"
    OUT.write_text(json.dumps(out, indent=2) + "\n")

    print(f"rows {len(df)} span {out['span']} effN {out['baselines']['persistence']['effective_n']} always-hedged {always:+.4f}")
    for n, b in out["baselines"].items():
        print(f"{n:12s} AUC={b['auc']:.3f} {b['auc_ci90']} NPS={b['nps_net_ann']:+.4f} on={b['hedge_on_frac']:.2f} timing_excess={b['timing_excess_nps']:+.4f} timing_p={b['timing_p']} lead={b['lead_median_lead_days']} missed={b['lead_n_missed']}/{b['lead_n_onsets']}")
    for kind, m in out["models"].items():
        res = m["metrics"]
        print(f"{kind:12s} AUC={res['auc']:.3f} {res['auc_ci90']} Brier={res['brier']:.3f} NPS={res['nps_net_ann']:+.4f} on={res['hedge_on_frac']:.2f} FNrate={res['false_negative_rate']:.2f} "
              f"timing_excess={res['timing_excess_nps']:+.4f} timing_p={res['timing_p']} lead={res['lead_median_lead_days']} missed={res['lead_n_missed']}/{res['lead_n_onsets']}")
        print(f"             dAUC vs persistence {res['auc_minus_persistence']:+.4f} p_raw={res['p_raw_vs_persistence']} M={m['M']} p_adj={m['p_bonferroni']} beats net={m['beats_both_baselines_net']} -> {m['verdict']}")
        print("             per fold AUC model/persist: " + "  ".join(f"{f['auc']:.2f}/{f['persistence_auc']:.2f}" for f in m["per_fold"]))
        print("             stress NPS model/persist:  " + "  ".join(f"{k[:8]} {w['model_nps']:+.3f}/{w['persistence_nps']:+.3f}" for k, w in m["stress_windows"].items()))


if __name__ == "__main__":
    main()
