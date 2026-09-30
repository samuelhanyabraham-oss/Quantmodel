"""Shared helpers for long-panel studies (docs/long_history_plan.md), so
every pre-registered run reports the same windows and the same per-fold /
per-window accounting. Nothing here trains or selects anything."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import baselines, evaluate, features, labels, long_history
from .costs import net_protection_score

STRESS_WINDOWS = {
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


def long_dataset(feature_names: list[str] | None = None) -> pd.DataFrame:
    """Long panel + labels + baselines (+ features), rows where everything
    requested is defined. Same construction every study uses."""
    panel = long_history.load_long_panel()
    spy = panel["SPY_close"]
    df = panel.join(labels.build_labels(spy))
    df["fwd_ret_10"] = spy.shift(-labels.HORIZON) / spy - 1.0
    df["persistence"] = baselines.persistence(spy)
    df["trailing_pctl"] = baselines.trailing_pctl_rule(spy)
    need = ["label", "fwd_ret_10", "persistence", "trailing_pctl"]
    if feature_names:
        df = df.join(features.build_features(panel)[feature_names])
        need += feature_names
    return df.dropna(subset=need)


def window_table(df: pd.DataFrame, signal_cols: list[str]) -> dict:
    out = {}
    for k, (a, b) in STRESS_WINDOWS.items():
        d = df.loc[a:b]
        if len(d) < 5:
            continue
        r = d["fwd_ret_10"].to_numpy()
        row = {"rows": int(len(d)), "base_rate": round(float(d["label"].mean()), 3),
               "always_hedged_nps": round(net_protection_score(np.ones(len(d)), r), 5)}
        for c in signal_cols:
            s = d[c].to_numpy(dtype=float)
            lt = evaluate.lead_times(d[c], d["label"])
            row[c] = {"nps": round(net_protection_score(s, r), 5), "on": round(float(s.mean()), 3),
                      "missed": lt["n_missed"], "onsets": lt["n_onsets"]}
        out[k] = row
    return out
