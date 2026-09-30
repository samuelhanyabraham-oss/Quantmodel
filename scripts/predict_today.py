"""Operational daily prediction + forward-test log (freeze v2).

Emits, for the latest bar in the frozen snapshot: the model probability, its
trailing-window percentile rank, and the hedge-ratio suggestion band. Appends
the prediction to forward_test.jsonl BEFORE its label can resolve (the label
needs 10 more trading days) — that file is the accumulating out-of-sample
record, the only path back to a skill claim after the holdout dissolution.

Frozen config (docs/freeze.md v2): logistic on [rv10, vix, vix_slope,
rv_ratio_10_63], Platt tail calibration, adaptive rank operating point
(70th pct of trailing 126 probabilities). Output contract: bands only,
never orders.
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
from regime.bands import hedge_band_from_rank

SEED = 20260813
SMALL = ["rv10", "vix", "vix_slope", "rv_ratio_10_63"]
RANK_WINDOW = 126
FORWARD_LOG = ROOT / "forward_test.jsonl"


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


def main() -> None:
    panel = data.load_dev_panel()
    lab = labels.build_labels(panel["SPY_close"])
    feats = features.build_features(panel)

    train = panel.join(lab).join(feats).dropna()
    cut = int(len(train) * (1 - models.CAL_FRAC))
    Xtr = train[SMALL].to_numpy(dtype=float)
    ytr = train["label"].to_numpy(dtype=float)
    scaler = StandardScaler().fit(Xtr[:cut])
    model = models.make_model("logistic", SEED)
    model.fit(scaler.transform(Xtr[:cut]), ytr[:cut])
    calibrate = models._fit_sigmoid_calibrator(
        model.predict_proba(scaler.transform(Xtr[cut:]))[:, 1], ytr[cut:]
    )

    # Score the trailing window + the latest bar (features need no labels).
    scorable = feats[SMALL].dropna()
    # Staleness guard: the prediction must be for the panel's latest bar.
    # If a frozen feature can't be computed there (a series the refresh
    # could not deliver — VIX3M since 2026-08-13), refuse rather than
    # silently re-emit the last computable bar as if it were today's.
    latest = panel.index.max()
    if scorable.index[-1] != latest:
        missing = [c for c in SMALL if pd.isna(feats.loc[latest, c])]
        raise SystemExit(
            f"REFUSED: frozen features {missing} are not computable on the latest bar "
            f"{latest.date()} (last computable: {scorable.index[-1].date()}). No "
            "prediction logged. The frozen config needs those series refreshed."
        )
    recent = scorable.iloc[-(RANK_WINDOW + 1) :]
    probs = calibrate(model.predict_proba(scaler.transform(recent[SMALL].to_numpy(dtype=float)))[:, 1])
    p_today = float(probs[-1])
    rank = float(np.mean(probs[:-1] <= p_today))
    band = hedge_band_from_rank(rank)
    asof = str(recent.index[-1].date())

    entry = {
        "asof": asof,
        "prob": round(p_today, 4),
        "rank": round(rank, 4),
        "signal": int(rank >= 0.70),
        "band_lo": band[0],
        "band_hi": band[1],
        "config": "freeze-v2 logistic_small4 adaptive q70 w126",
        "label_resolves_after": str(
            (recent.index[-1] + pd.tseries.offsets.BDay(labels.HORIZON)).date()
        ),
    }
    if already_logged(asof, entry["config"]):
        print(json.dumps(entry, indent=2))
        raise SystemExit(f"already logged for {asof} — not appended twice")
    with open(FORWARD_LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")
    experiment_log.log_run(
        {"strategy": "forward_test_prediction", "config": entry["config"]},
        {"prob": entry["prob"], "rank": entry["rank"], "signal": entry["signal"]},
        seed=SEED, data_snapshot_hash=data.snapshot_hash(),
        notes=f"forward-test prediction asof {asof}",
    )
    print(json.dumps(entry, indent=2))


if __name__ == "__main__":
    main()
