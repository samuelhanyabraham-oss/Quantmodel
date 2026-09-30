"""L2: charter baselines on the long-history panel (docs/long_history_plan.md).

Persistence and trailing-percentile need only SPY, so they run 2001->2026.
Term structure needs VIX3M and is reported on the 2021+ segment only.
Pooled walk-forward test folds (10 folds), per fold, and inside named
stress windows (baselines have no training, so direct windows are fair).
Every evaluation is logged against the long panel's hash.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regime import baselines, evaluate, experiment_log, labels, long_history  # noqa: E402
from regime.costs import net_protection_score  # noqa: E402
from regime.validation import purged_walk_forward  # noqa: E402

SEED = 20260813
N_FOLDS = 10
STRESS = {
    "dotcom_2001_02": ("2001-01-01", "2003-03-31"),
    "gfc_2008_09": ("2008-09-01", "2009-03-31"),
    "flash_crash_2010": ("2010-04-15", "2010-06-30"),
    "eurozone_2011": ("2011-07-15", "2011-10-31"),
    "china_2015": ("2015-08-01", "2015-09-30"),
    "volmageddon_2018": ("2018-01-15", "2018-02-28"),
    "q4_2018": ("2018-10-01", "2018-12-31"),
    "covid_2020": ("2020-02-15", "2020-04-30"),
    "bear_2022": ("2022-01-01", "2022-10-31"),
    "tariff_2025": ("2025-03-01", "2025-05-31"),
}


def build() -> pd.DataFrame:
    panel = long_history.load_long_panel()
    spy = panel["SPY_close"]
    df = panel.join(labels.build_labels(spy))
    df["fwd_ret_10"] = spy.shift(-labels.HORIZON) / spy - 1.0
    df["persistence"] = baselines.persistence(spy)
    df["trailing_pctl"] = baselines.trailing_pctl_rule(spy)
    ts = baselines.term_structure(panel["VIX_close"], panel["VIX3M_close"])
    df["term_structure"] = ts.where(panel["VIX3M_close"].notna())
    return df


def window_stats(d: pd.DataFrame, names: list[str]) -> dict:
    y = d["label"].to_numpy(); r = d["fwd_ret_10"].to_numpy()
    out = {"rows": int(len(d)), "base_rate": round(float(y.mean()), 3),
           "worst_fwd10": round(float(np.nanmin(r)), 4)}
    for n in names:
        s = d[n].to_numpy(dtype=float)
        lt = evaluate.lead_times(d[n], d["label"])
        out[n] = {"nps_net_ann": round(net_protection_score(s, r), 5),
                  "hedge_on": round(float(s.mean()), 3),
                  "fn_rate": round(float(((s == 0) & (y == 1)).sum() / max(y.sum(), 1)), 3),
                  "onsets": lt["n_onsets"], "missed": lt["n_missed"],
                  "median_lead": lt["median_lead_days"]}
    out["always_hedged_nps"] = round(net_protection_score(np.ones_like(s), r), 5)
    return out


def main() -> None:
    df = build()
    snap = long_history.long_snapshot_hash()
    core = df.dropna(subset=["label", "fwd_ret_10", "persistence", "trailing_pctl"])
    folds = purged_walk_forward(len(core), n_folds=N_FOLDS)
    test_rows = np.concatenate([f.test_idx for f in folds])
    test_df = core.iloc[test_rows]
    results = {"panel_hash": snap, "rows_total": int(len(core)),
               "span": [str(core.index[0].date()), str(core.index[-1].date())],
               "n_folds": len(folds), "pooled": {}, "per_fold": [], "stress_windows": {}}

    for name in ["persistence", "trailing_pctl"]:
        sig = core[name].iloc[test_rows]
        res = evaluate.evaluate_strategy(sig, sig, test_df, name=name, seed=SEED)
        results["pooled"][name] = res
        experiment_log.log_run({"strategy": name, "type": "baseline", "panel": "long_v1", "folds": len(folds)},
                               res, seed=SEED, data_snapshot_hash=snap,
                               notes="long-history walk-forward test folds (L2, docs/long_history_plan.md)")
    # term structure: 2021+ segment only, pooled directly (no training)
    seg = df.dropna(subset=["label", "fwd_ret_10", "term_structure"])
    if len(seg) > 100:
        sig = seg["term_structure"]
        res = evaluate.evaluate_strategy(sig, sig, seg, name="term_structure_2021plus", seed=SEED)
        results["pooled"]["term_structure_2021plus"] = res
        experiment_log.log_run({"strategy": "term_structure", "type": "baseline", "panel": "long_v1", "segment": "2021+"},
                               res, seed=SEED, data_snapshot_hash=snap, notes="term structure only where VIX3M exists")
    results["pooled"]["always_hedged_nps"] = round(net_protection_score(
        np.ones(len(test_df)), test_df["fwd_ret_10"].to_numpy()), 5)
    results["pooled"]["never_hedged_nps"] = 0.0

    for k, f in enumerate(folds):
        d = core.iloc[f.test_idx]
        results["per_fold"].append({"fold": k, "start": str(d.index[0].date()), "end": str(d.index[-1].date()),
                                    **window_stats(d, ["persistence", "trailing_pctl"])})
    for k, (a, b) in STRESS.items():
        d = core.loc[a:b]
        if len(d):
            results["stress_windows"][k] = window_stats(d, ["persistence", "trailing_pctl"])
    calm = core[~core.index.isin(pd.concat([core.loc[a:b] for a, b in STRESS.values()]).index)]
    results["stress_windows"]["all_other_days"] = window_stats(calm, ["persistence", "trailing_pctl"])

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "long_baselines.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"rows {results['rows_total']} span {results['span']} folds {len(folds)}")
    for n, r in results["pooled"].items():
        if isinstance(r, dict):
            print(f"{n:24s} AUC={r['auc']:.3f} {r['auc_ci90']} Brier={r['brier']:.3f} NPS={r['nps_net_ann']:+.4f} "
                  f"on={r['hedge_on_frac']:.2f} FNrate={r['false_negative_rate']:.2f} effN={r['effective_n']} "
                  f"lead={r['lead_median_lead_days']} missed={r['lead_n_missed']}/{r['lead_n_onsets']}")
        else:
            print(f"{n:24s} {r:+.4f}")
    print("\nper fold (persistence NPS / trailing NPS / always-hedged NPS / base rate):")
    for f in results["per_fold"]:
        print(f"  {f['start']}..{f['end']}  {f['persistence']['nps_net_ann']:+.4f} / {f['trailing_pctl']['nps_net_ann']:+.4f} / "
              f"{f['always_hedged_nps']:+.4f}  base={f['base_rate']:.2f}")
    print("\nstress windows (persistence NPS / trailing NPS / always-hedged / base rate / persistence missed onsets):")
    for k, w in results["stress_windows"].items():
        print(f"  {k:18s} {w['persistence']['nps_net_ann']:+.4f} / {w['trailing_pctl']['nps_net_ann']:+.4f} / "
              f"{w['always_hedged_nps']:+.4f}  base={w['base_rate']:.2f}  missed={w['persistence']['missed']}/{w['persistence']['onsets']}")


if __name__ == "__main__":
    main()
