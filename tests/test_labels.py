"""Label construction on synthetic series with hand-checkable answers."""

import numpy as np
import pandas as pd
import pytest

from regime import labels as L


def _idx(n):
    return pd.bdate_range("2015-01-01", periods=n)


def test_forward_rv_uses_only_future_returns():
    # Price flat until day k, then volatile. Forward RV must rise BEFORE the
    # vol arrives (it looks ahead), trailing RV only AFTER.
    n, k = 300, 200
    rng = np.random.default_rng(0)
    r = np.zeros(n)
    r[k:] = rng.normal(0, 0.03, n - k)
    close = pd.Series(100 * np.exp(np.cumsum(r)), index=_idx(n))
    rv_fwd = L.realized_vol_forward(close)
    rv_trail = L.realized_vol_trailing(close)
    assert rv_fwd.iloc[k - 5] > 0.1  # sees the coming vol
    assert rv_trail.iloc[k - 5] < 0.01  # still blind at t
    assert rv_trail.iloc[k + 15] > 0.1


def test_forward_rv_last_rows_are_nan():
    close = pd.Series(np.linspace(100, 110, 60), index=_idx(60))
    rv_fwd = L.realized_vol_forward(close)
    assert rv_fwd.iloc[-L.HORIZON :].isna().all()
    assert rv_fwd.iloc[: -L.HORIZON].notna().sum() > 0


def test_forward_mdd_known_by_hand():
    # 20 flat days, then a straight 10% fall over 5 days, then flat.
    vals = [100.0] * 20 + [98, 96, 94, 92, 90] + [90.0] * 20
    close = pd.Series(vals, index=_idx(len(vals)))
    mdd = L.forward_max_drawdown(close)
    # At t=19 (last flat day) the next 10 days contain the full fall to 90.
    assert mdd.iloc[19] == pytest.approx(-0.10)
    # Deep in the final flat stretch there is no drawdown ahead.
    assert mdd.iloc[30] == pytest.approx(0.0)


def test_label_threshold_uses_only_past_data():
    # The threshold at t must not move when future prices change.
    n = 600
    rng = np.random.default_rng(1)
    r = rng.normal(0, 0.01, n)
    close = pd.Series(100 * np.exp(np.cumsum(r)), index=_idx(n))
    t_check = 400
    thresh_a = L.build_labels(close)["rv_thresh"].iloc[t_check]
    r2 = r.copy()
    r2[t_check + 1 :] = rng.normal(0, 0.05, n - t_check - 1)  # rewrite future
    close2 = pd.Series(100 * np.exp(np.cumsum(r2)), index=_idx(n))
    thresh_b = L.build_labels(close2)["rv_thresh"].iloc[t_check]
    assert thresh_a == pytest.approx(thresh_b)


def test_label_base_rate_near_25pct():
    # By construction the base rate should be in the neighborhood of 25%.
    n = 1500
    rng = np.random.default_rng(2)
    close = pd.Series(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=_idx(n)
    )
    lab = L.build_labels(close)["label"].dropna()
    assert 0.10 < lab.mean() < 0.45
