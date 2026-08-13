"""The three charter baselines. All use only information available at t.

Each returns a binary signal series aligned to the panel index. These are the
bar the model must clear NET (docs/cost_model.md); if it doesn't, the project
reports failure.
"""

from __future__ import annotations

import pandas as pd

from .labels import HORIZON, PCTL, TRAIL_WINDOW, realized_vol_trailing


def persistence(close: pd.Series) -> pd.Series:
    """Baseline 1 — the killer. Today's regime state (trailing RV above its
    own trailing 75th pct threshold, all known at t) predicted to continue."""
    rv = realized_vol_trailing(close, HORIZON)
    thresh = rv.rolling(TRAIL_WINDOW, min_periods=TRAIL_WINDOW).quantile(PCTL)
    return (rv > thresh).astype(float)


def term_structure(vix: pd.Series, vix3m: pd.Series) -> pd.Series:
    """Baseline 2 — implied-vol term-structure inversion: VIX >= VIX3M."""
    return (vix >= vix3m).astype(float)


def trailing_pctl_rule(close: pd.Series, q: float = 0.60) -> pd.Series:
    """Baseline 3 — single rolling rule: trailing RV above its trailing 60th
    percentile. A deliberately more trigger-happy cousin of persistence (the
    charter wants a distinct single-threshold rule, not a duplicate of #1)."""
    rv = realized_vol_trailing(close, HORIZON)
    thresh = rv.rolling(TRAIL_WINDOW, min_periods=TRAIL_WINDOW).quantile(q)
    return (rv > thresh).astype(float)
