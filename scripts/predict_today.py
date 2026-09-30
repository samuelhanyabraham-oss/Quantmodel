"""Operational daily prediction + forward-test log (freeze v3).

Emits, for the latest bar in the active frozen snapshot:
- OPERATIONAL: the persistence rule (charter baseline #1) — trailing
  10-day realized vol vs the 75th percentile of its trailing 252 values —
  as a state (in/out of regime), the continuous trailing-252 rank of rv10,
  and the freeze-v3 hedge band from that rank.
- DIAGNOSTIC: the freeze-v2 logistic model's probability and adaptive
  rank, when its features (incl. vix_slope -> VIX3M) exist on the bar;
  null otherwise. Never drives the band.
Appends the prediction to forward_test.jsonl BEFORE its label resolves.
Refuses a stale bar and refuses to log the same (asof, config) twice.
Output contract: bands only, never orders. See docs/freeze.md.
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

from regime import data, experiment_log, features, labels, models
from regime.bands import hedge_band_from_rv_rank
from regime.labels import PCTL, TRAIL_WINDOW, realized_vol_trailing

SEED = 20260813
SMALL = ["rv10", "vix", "vix_slope", "rv_ratio_10_63"]
RANK_WINDOW = 126
FORWARD_LOG = ROOT / "forward_test.jsonl"
CONFIG = "freeze-v3 persistence-operational, logistic_small4 diagnostic"


def already_logged(asof: str, config: str, universe: str | None = None) -> bool:
    """Idempotency: one forward-test entry per (asof, config, universe).
    The 2026-08-12 entries were written twice by two runs of the same
    script; the record is append-only, so they stay, and the scorer
    de-duplicates. From here on the append itself refuses a repeat."""
    if not FORWARD_LOG.exists():
        return False
    for line in FORWARD_LOG.read_text().splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        if e.get("asof") == asof and e.get("config") == config and e.get("universe") == universe:
            return True
    return False


def model_diagnostic(panel: pd.DataFrame, feats: pd.DataFrame, latest: pd.Timestamp) -> dict:
    """Freeze-v2 model probability + adaptive rank on the latest bar, or
    nulls with the reason. Trains on all labelled rows of the active
    snapshot, as freeze v2 specified."""
    if feats.loc[latest, SMALL].isna().any():
        missing = [c for c in SMALL if pd.isna(feats.loc[latest, c])]
        return {"model_prob_diagnostic": None, "model_rank_diagnostic": None,
                "model_diagnostic_note": f"features {missing} not computable on {latest.date()}"}
    lab = labels.build_labels(panel["SPY_close"])
    train = panel.join(lab).join(feats).dropna()
    cut = int(len(train) * (1 - models.CAL_FRAC))
    Xtr = train[SMALL].to_numpy(dtype=float)
    ytr = train["label"].to_numpy(dtype=float)
    scaler = StandardScaler().fit(Xtr[:cut])
    model = models.make_model("logistic", SEED)
    model.fit(scaler.transform(Xtr[:cut]), ytr[:cut])
    calibrate = models._fit_sigmoid_calibrator(
        model.predict_proba(scaler.transform(Xtr[cut:]))[:, 1], ytr[cut:])
    scorable = feats[SMALL].dropna()
    recent = scorable.loc[:latest].iloc[-(RANK_WINDOW + 1):]
    probs = calibrate(model.predict_proba(scaler.transform(recent.to_numpy(dtype=float)))[:, 1])
    p_today = float(probs[-1])
    rank = float(np.mean(probs[:-1] <= p_today))
    return {"model_prob_diagnostic": round(p_today, 4), "model_rank_diagnostic": round(rank, 4),
            "model_diagnostic_note": None}


def main() -> None:
    panel = data.load_dev_panel()
    latest = panel.index.max()
    spy = panel["SPY_close"]
    rv = realized_vol_trailing(spy, labels.HORIZON)
    thresh = rv.rolling(TRAIL_WINDOW, min_periods=TRAIL_WINDOW).quantile(PCTL)
    if pd.isna(rv.loc[latest]) or pd.isna(thresh.loc[latest]):
        raise SystemExit(f"REFUSED: persistence rule not computable on latest bar {latest.date()}")
    window = rv.loc[:latest].iloc[-TRAIL_WINDOW:]
    # STRICT rank (fraction of the trailing 252 rv10 values below today's):
    # with the label threshold being pandas' interpolated 75th percentile,
    # rank >= 0.75 <=> rv > thresh exactly; a non-strict rank disagreed at
    # rank == 0.75 on 4 of 1,025 historical bars (review 2026-09-30).
    rank = float(np.mean(window.to_numpy() < rv.loc[latest]))
    in_regime = bool(rv.loc[latest] > thresh.loc[latest])
    band = hedge_band_from_rv_rank(rank)
    feats = features.build_features(panel)
    diag = model_diagnostic(panel, feats, latest)
    asof = str(latest.date())

    entry = {
        "asof": asof,
        "signal_rule": "persistence (rv10 > trailing 252-day q75)",
        "in_regime": in_regime,
        "signal": int(in_regime),
        "rv10_ann": round(float(rv.loc[latest]), 4),
        "regime_thresh_ann": round(float(thresh.loc[latest]), 4),
        "rv10_rank252": round(rank, 4),
        "band_lo": band[0],
        "band_hi": band[1],
        **diag,
        "config": CONFIG,
        "label_resolves_after": str((latest + pd.tseries.offsets.BDay(labels.HORIZON)).date()),
    }
    if already_logged(asof, CONFIG):
        print(json.dumps(entry, indent=2))
        raise SystemExit(f"already logged for {asof} — not appended twice")
    with open(FORWARD_LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")
    experiment_log.log_run(
        {"strategy": "forward_test_prediction", "config": CONFIG},
        {k: entry[k] for k in ("signal", "rv10_rank252", "model_prob_diagnostic", "model_rank_diagnostic")},
        seed=SEED, data_snapshot_hash=data.snapshot_hash(),
        notes=f"forward-test prediction asof {asof} (freeze v3)",
    )
    print(json.dumps(entry, indent=2))


if __name__ == "__main__":
    main()
