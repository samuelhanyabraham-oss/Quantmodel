"""Leakage suite, long-history panel. Same charter tests as the dev panel,
run on the 2000->present panel with the VIX3M-free feature set the
replication study uses (regime/long_history.py). Skipped, never silently
passed, if the long panel has not been frozen."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from regime import features, labels, long_history, models
from regime.validation import purged_walk_forward

FEATS = ["rv10", "vix", "rv_ratio_10_63"]

pytestmark = pytest.mark.skipif(
    not long_history.LONG_MANIFEST.exists(), reason="long panel not frozen"
)


@pytest.fixture(scope="module")
def long_df():
    panel = long_history.load_long_panel()  # verifies the hash
    lab = labels.build_labels(panel["SPY_close"])
    feats = features.build_features(panel)[FEATS]
    return panel.join(lab).join(feats).dropna(subset=["label", *FEATS])


def _pooled_auc(X, y, seed=0):
    probs = models.walk_forward_probs(X, y, "logistic", seed=seed)
    mask = probs.notna()
    return roc_auc_score(y[mask], probs[mask])


def test_long_panel_repairs_do_not_touch_frozen_segment():
    m = long_history.load_long_manifest()
    cut = pd.Timestamp(m["frozen_segment_from"])
    for r in m["repairs"]["spy_index_check"]:
        assert pd.Timestamp(r["date"]) < cut
    for r in m["repairs"]["feature_etf_spikes"]:
        assert pd.Timestamp(r["date"]) < cut


def test_long_panel_labels_only_use_forward_data(long_df):
    """Truncating the panel after t must not change any label at or before
    t - HORIZON, and must erase labels inside the last HORIZON rows."""
    close = long_df["SPY_close"]
    full = labels.build_labels(close)["label"]
    cut = len(close) - 200
    part = labels.build_labels(close.iloc[:cut])["label"]
    a = full.iloc[: cut - labels.HORIZON]
    b = part.iloc[: cut - labels.HORIZON]
    pd.testing.assert_series_equal(a, b)
    assert part.iloc[-labels.HORIZON:].isna().all()


def test_long_panel_shuffled_labels_give_chance_auc(long_df):
    rng = np.random.default_rng(1)
    y = pd.Series(rng.permutation(long_df["label"].to_numpy()), index=long_df.index)
    auc = _pooled_auc(long_df[FEATS], y, seed=1)
    assert 0.44 <= auc <= 0.56, f"shuffled-label AUC {auc:.3f} — leakage"


def test_long_panel_forward_shift_collapses_performance(long_df):
    y = long_df["label"]
    auc_true = _pooled_auc(long_df[FEATS], y)
    shifted = long_df[FEATS].shift(-labels.HORIZON - 5).dropna()
    auc_shift = _pooled_auc(shifted, y.loc[shifted.index])
    # features from 15 bars later carry the label's own window -> should NOT
    # look like the true features; and true features must beat chance.
    assert auc_true > 0.55
    assert abs(auc_shift - auc_true) > 0.02 or auc_shift < auc_true
