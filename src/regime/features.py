"""Point-in-time features. Every feature at bar t uses only closes/highs/lows
printed at or before t (all series print at the close of t; see
docs/data_dictionary.md). Deliberately boring, small set. No selection
against the label happens here or anywhere outside training folds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .labels import realized_vol_trailing


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    """~18 features from the frozen panel. Docstring per feature = its
    information timestamp (all: close of t)."""
    f = pd.DataFrame(index=panel.index)
    spy = panel["SPY_close"]
    logret = np.log(spy).diff()

    # --- realized vol levels & dynamics (SPY, known at t) ---
    f["rv10"] = realized_vol_trailing(spy, 10)
    f["rv21"] = realized_vol_trailing(spy, 21)
    f["rv63"] = realized_vol_trailing(spy, 63)
    f["rv10_chg5"] = f["rv10"] - f["rv10"].shift(5)
    f["rv_ratio_10_63"] = f["rv10"] / f["rv63"]
    f["volofvol21"] = f["rv10"].pct_change().rolling(21).std()

    # --- implied vol term structure (CBOE closes at t) ---
    f["vix"] = panel["VIX_close"]
    f["vix_slope"] = panel["VIX_close"] / panel["VIX3M_close"] - 1.0
    f["vix_chg5"] = panel["VIX_close"].pct_change(5)
    f["vrp"] = panel["VIX_close"] / 100.0 - f["rv21"]  # variance risk premium proxy

    # --- credit stress proxy (traded ETF closes at t) ---
    hyg_lqd = panel["HYG_close"] / panel["LQD_close"]
    f["credit_ratio_chg21"] = hyg_lqd.pct_change(21)
    f["credit_ratio_chg5"] = hyg_lqd.pct_change(5)

    # --- cross-asset confirmation (closes at t) ---
    f["qqq_rv10"] = realized_vol_trailing(panel["QQQ_close"], 10)
    f["iwm_rv10"] = realized_vol_trailing(panel["IWM_close"], 10)

    # --- price state (closes at t) ---
    f["dd_from_peak63"] = spy / spy.rolling(63).max() - 1.0
    f["ret21"] = spy.pct_change(21)
    f["ret5"] = spy.pct_change(5)
    f["skew63"] = logret.rolling(63).skew()

    # --- intraday range stress (high/low of t) ---
    tr = (panel["SPY_high"] - panel["SPY_low"]) / spy
    f["range5"] = tr.rolling(5).mean()

    return f


FEATURE_NAMES = [
    "rv10", "rv21", "rv63", "rv10_chg5", "rv_ratio_10_63", "volofvol21",
    "vix", "vix_slope", "vix_chg5", "vrp",
    "credit_ratio_chg21", "credit_ratio_chg5",
    "qqq_rv10", "iwm_rv10",
    "dd_from_peak63", "ret21", "ret5", "skew63", "range5",
]
