"""Output contract: calibrated probability -> hedge-ratio suggestion band.

Frozen in docs/freeze.md (2026-08-13). Monotone, piecewise-constant, three
bands. This module never emits, places, or formats an order.
"""

from __future__ import annotations

OPERATING_THRESHOLD = 0.20

# (lower_p_inclusive, band_lo, band_hi) — must stay monotone in both edges.
BANDS = [
    (0.00, 0.00, 0.10),
    (0.20, 0.25, 0.50),
    (0.30, 0.50, 0.75),
]


def hedge_band(p: float) -> tuple[float, float]:
    """Suggested hedge-ratio range for a calibrated probability."""
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"probability out of range: {p}")
    lo, hi = BANDS[0][1], BANDS[0][2]
    for cut, b_lo, b_hi in BANDS:
        if p >= cut:
            lo, hi = b_lo, b_hi
    return lo, hi
