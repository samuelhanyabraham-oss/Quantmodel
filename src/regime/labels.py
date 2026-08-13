"""Label construction (CLAUDE.md "Target definition" — do not change).

Primary label at bar t: 1 if annualized realized volatility of SPY log
returns over t+1..t+10 exceeds the 75th percentile of the trailing 10-day
realized-vol series computed from data available at or before t (trailing
window: 252 obs). The threshold uses only past data; only the forward RV
looks ahead. Secondary labels: forward 10-day RV (continuous) and forward
max drawdown over the same window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZON = 10  # forward window, trading days
TRAIL_WINDOW = 252  # trailing window for the percentile threshold
PCTL = 0.75
ANN = np.sqrt(252.0)


def realized_vol_trailing(close: pd.Series, window: int = HORIZON) -> pd.Series:
    """Annualized RV of log returns over the PAST `window` days, known at t."""
    r = np.log(close).diff()
    return r.rolling(window).std(ddof=0) * ANN


def realized_vol_forward(close: pd.Series, horizon: int = HORIZON) -> pd.Series:
    """Annualized RV of log returns over t+1..t+horizon. Label component —
    must never be used as a feature."""
    r = np.log(close).diff()
    # r_{t+1}..r_{t+horizon}: reverse-rolling std via shift(-horizon)
    fwd = r.rolling(horizon).std(ddof=0).shift(-horizon) * ANN
    return fwd


def forward_max_drawdown(close: pd.Series, horizon: int = HORIZON) -> pd.Series:
    """Max peak-to-trough drawdown over t+1..t+horizon (negative number),
    with the close at t as the initial peak: the drawdown a holder standing
    at t would experience, which is what a hedge sized at t protects."""
    out = pd.Series(np.nan, index=close.index)
    vals = close.to_numpy(dtype=float)
    n = len(vals)
    for i in range(n - horizon):
        w = vals[i : i + 1 + horizon]  # include t as the starting peak
        peaks = np.maximum.accumulate(w)
        out.iloc[i] = float(np.min(w[1:] / peaks[1:] - 1.0))
    return out


def build_labels(close: pd.Series) -> pd.DataFrame:
    """All labels, plus the threshold series (kept for baselines/diagnostics).

    Rows where any component is undefined (warm-up head, forward-window tail)
    are NaN and must be dropped by the caller before modeling.
    """
    rv_trail = realized_vol_trailing(close)
    # Threshold at t: 75th pct of trailing-RV values available at or before t.
    # min_periods=TRAIL_WINDOW so the early sample doesn't use a flattering
    # tiny window.
    thresh = rv_trail.rolling(TRAIL_WINDOW, min_periods=TRAIL_WINDOW).quantile(PCTL)
    rv_fwd = realized_vol_forward(close)
    mdd_fwd = forward_max_drawdown(close)
    label = (rv_fwd > thresh).astype(float)
    label[rv_fwd.isna() | thresh.isna()] = np.nan
    return pd.DataFrame(
        {
            "label": label,
            "rv_fwd": rv_fwd,
            "mdd_fwd": mdd_fwd,
            "rv_trail": rv_trail,
            "rv_thresh": thresh,
        }
    )
