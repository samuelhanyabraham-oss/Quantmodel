"""Net Protection Score — the frozen cost accounting from docs/cost_model.md.

Parameters are module constants ON PURPOSE: changing them is a charter
amendment, not a keyword argument. Applied identically to model and
baselines.
"""

from __future__ import annotations

import numpy as np

C_BLEED = 0.0002  # 2 bp/day while hedged
C_SWITCH = 0.0010  # 10 bp per on/off switch
CAPTURE = 0.5  # fraction of drawdown the overlay offsets
ANN_DAYS = 252


def net_protection_score(signal: np.ndarray, fwd_return_10d: np.ndarray) -> float:
    """Annualized NPS for a binary signal vs. forward 10-day returns."""
    s = np.asarray(signal, dtype=float)
    r = np.asarray(fwd_return_10d, dtype=float)
    if s.shape != r.shape:
        raise ValueError("signal/return length mismatch")
    protection = s * np.maximum(0.0, -r) * CAPTURE / 10.0  # per-day allocation
    bleed = C_BLEED * s
    switches = np.abs(np.diff(s, prepend=s[0]))
    switch_cost = C_SWITCH * switches
    daily = protection - bleed - switch_cost
    return float(np.nanmean(daily) * ANN_DAYS)
