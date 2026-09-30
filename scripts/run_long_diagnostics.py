"""Two harness self-checks on the long panel (2026-09-30). Diagnostics of the
METHOD, not new models; each is logged as a run because it is an evaluation.

1. Dependence: how long does label dependence last, and how much do the
   90% CIs widen when the bootstrap block length is raised from the frozen
   20 days to 60 / 120 / 250? If they widen materially, every CI reported
   so far is overconfident (charter: "CIs must use the effective count").
2. Cost-model sensitivity: NPS of always-hedged, persistence and trailing
   percentile across bleed 2..8 bp/day and capture 0.5 / 0.35, and the
   break-even bleed at which each strategy's net goes to zero. The charter's
   Section-2 falsifier named 4 bp/day explicitly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regime import costs, experiment_log, long_history, longrun, stats  # noqa: E402
from regime.validation import purged_walk_forward  # noqa: E402

SEED = 20260813


def nps_with(signal, r, bleed, capture, switch=costs.C_SWITCH):
    s = np.asarray(signal, float); r = np.asarray(r, float)
    protection = s * np.maximum(0.0, -r) * capture / 10.0
    daily = protection - bleed * s - switch * np.abs(np.diff(s, prepend=s[0]))
    return float(np.nanmean(daily) * costs.ANN_DAYS)


def breakeven_bleed(signal, r, capture):
    on = float(np.mean(signal))
    if on == 0:
        return None
    lo, hi = 0.0, 0.01
    for _ in range(40):
        mid = (lo + hi) / 2
        if nps_with(signal, r, mid, capture) > 0:
            lo = mid
        else:
            hi = mid
    return round(mid * 1e4, 2)  # bp/day


def main() -> None:
    df = longrun.long_dataset()
    folds = purged_walk_forward(len(df), n_folds=10)
    test_rows = np.concatenate([f.test_idx for f in folds])
    te = df.iloc[test_rows]
    y = te["label"].to_numpy(); r = te["fwd_ret_10"].to_numpy()
    snap = long_history.long_snapshot_hash()

    # --- 1. dependence ---
    lab = df["label"]
    acf = {int(k): round(float(lab.autocorr(k)), 3) for k in [1, 5, 10, 20, 40, 60, 120, 250]}
    # integrated autocorrelation time -> Newey-West-style effective N
    rho = np.array([lab.autocorr(k) for k in range(1, 251)])
    rho = np.where(np.isfinite(rho), rho, 0.0)
    first_neg = int(np.argmax(rho <= 0)) if (rho <= 0).any() else len(rho)
    tau = 1 + 2 * float(rho[:first_neg].sum())
    eff = {"n_rows": int(len(lab)), "effective_n_frozen_rule(n/10)": round(len(lab) / 10, 1),
           "effective_n_autocorr(n/(1+2*sum rho))": round(len(lab) / tau, 1),
           "integrated_autocorr_time_days": round(tau, 1), "first_nonpositive_acf_lag": first_neg}
    ci_by_block = {}
    p = te["persistence"].to_numpy(float)
    for bl in [20, 60, 120, 250]:
        pt, lo, hi = stats.block_bootstrap_ci(roc_auc_score, y, p, block_len=bl, seed=SEED)
        ci_by_block[bl] = {"auc": round(pt, 4), "ci90": [round(lo, 4), round(hi, 4)], "width": round(hi - lo, 4)}
    dep = {"label_acf": acf, **eff, "persistence_auc_ci90_by_block_len": ci_by_block,
           "ci_width_ratio_250_vs_20": round(ci_by_block[250]["width"] / ci_by_block[20]["width"], 2)}

    # --- 2. cost sensitivity ---
    sigs = {"always_hedged": np.ones(len(y)), "persistence": p, "trailing_pctl": te["trailing_pctl"].to_numpy(float)}
    grid = {}
    for cap in [0.5, 0.35]:
        for bp in [2, 3, 4, 5, 6, 8]:
            grid[f"capture{cap}_bleed{bp}bp"] = {k: round(nps_with(s, r, bp * 1e-4, cap), 4) for k, s in sigs.items()}
    be = {cap: {k: breakeven_bleed(s, r, cap) for k, s in sigs.items()} for cap in [0.5, 0.35]}
    # at what bleed does a rule beat always-hedged? (net difference)
    cross = {}
    for cap in [0.5, 0.35]:
        for k in ["persistence", "trailing_pctl"]:
            lo, hi = 0.0, 0.02; found = None
            for _ in range(40):
                mid = (lo + hi) / 2
                if nps_with(sigs["always_hedged"], r, mid, cap) > nps_with(sigs[k], r, mid, cap):
                    lo = mid
                else:
                    hi = mid; found = mid
            cross[f"{k}_beats_always_from_bleed_bp(capture{cap})"] = round(found * 1e4, 2) if found else None
    cost = {"frozen": {"bleed_bp": costs.C_BLEED * 1e4, "switch_bp": costs.C_SWITCH * 1e4, "capture": costs.CAPTURE},
            "mean_fwd10_return": round(float(np.mean(r)), 5), "mean_negative_part": round(float(np.mean(np.maximum(0, -r))), 5),
            "grid": grid, "breakeven_bleed_bp": be, "crossover": cross}

    out = {"panel_hash": snap, "rows_test": int(len(te)), "dependence": dep, "cost_sensitivity": cost}
    (ROOT / "results" / "long_diagnostics.json").write_text(json.dumps(out, indent=2) + "\n")
    experiment_log.log_run({"strategy": "harness_diagnostics", "type": "diagnostic", "panel": "long_v1",
                            "checks": ["dependence/block-length", "cost-model sensitivity"]},
                           {"ci_width_ratio_250_vs_20": dep["ci_width_ratio_250_vs_20"],
                            "effective_n_autocorr": eff["effective_n_autocorr(n/(1+2*sum rho))"],
                            "breakeven_bleed_bp_capture0.5": be[0.5]},
                           seed=SEED, data_snapshot_hash=snap, notes="method self-checks on the long panel; no model")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
