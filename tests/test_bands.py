import numpy as np
import pytest

from regime.bands import BANDS, RANK_BANDS, hedge_band, hedge_band_from_rank


def test_rank_bands_are_monotone():
    grid = np.linspace(0, 1, 201)
    prev = (-1.0, -1.0)
    for r in grid:
        b = hedge_band_from_rank(float(r))
        assert b[0] >= prev[0] and b[1] >= prev[1]
        prev = b


def test_rank_band_edges_match_freeze_v2():
    assert RANK_BANDS == [(0.00, 0.00, 0.10), (0.70, 0.25, 0.50), (0.85, 0.50, 0.75)]
    assert hedge_band_from_rank(0.69) == (0.0, 0.10)
    assert hedge_band_from_rank(0.70) == (0.25, 0.50)
    assert hedge_band_from_rank(0.90) == (0.50, 0.75)
    with pytest.raises(ValueError):
        hedge_band_from_rank(-0.1)


def test_bands_are_monotone():
    # Higher probability must never suggest a smaller hedge (charter).
    grid = np.linspace(0, 1, 201)
    prev_lo, prev_hi = -1.0, -1.0
    for p in grid:
        lo, hi = hedge_band(float(p))
        assert lo >= prev_lo and hi >= prev_hi
        prev_lo, prev_hi = lo, hi


def test_band_edges_match_freeze_doc():
    assert BANDS == [(0.00, 0.00, 0.10), (0.20, 0.25, 0.50), (0.30, 0.50, 0.75)]
    assert hedge_band(0.0) == (0.0, 0.10)
    assert hedge_band(0.19) == (0.0, 0.10)
    assert hedge_band(0.20) == (0.25, 0.50)
    assert hedge_band(0.75) == (0.50, 0.75)


def test_out_of_range_rejected():
    with pytest.raises(ValueError):
        hedge_band(1.5)
