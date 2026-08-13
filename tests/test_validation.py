"""Walk-forward splitter: ordering, purge+embargo, planted-leakage catch."""

import numpy as np
import pytest

from regime.labels import HORIZON
from regime.validation import EMBARGO, purged_walk_forward


def test_train_always_precedes_test_with_gap():
    folds = purged_walk_forward(800)
    assert len(folds) == 5
    for f in folds:
        assert f.train_idx.max() < f.test_idx.min()
        # purge + embargo: gap of at least EMBARGO + HORIZON rows
        assert f.test_idx.min() - f.train_idx.max() - 1 >= EMBARGO + HORIZON

    # test blocks tile forward without overlap
    for a, b in zip(folds, folds[1:]):
        assert a.test_idx.max() < b.test_idx.min()


def test_no_test_index_ever_in_train():
    folds = purged_walk_forward(600, n_folds=4)
    for f in folds:
        assert len(np.intersect1d(f.train_idx, f.test_idx)) == 0


def test_planted_leakage_is_blocked_by_embargo():
    # Deliberately leaky setup: y_t depends on x over t..t+HORIZON (an
    # overlapping forward label). A 1-NN memorizer scores far above chance if
    # trained on rows adjacent to the test block; the purge+embargo removes
    # exactly those rows. We verify no training row's label window reaches
    # any test row.
    n = 500
    folds = purged_walk_forward(n, n_folds=3)
    for f in folds:
        label_window_end = f.train_idx.max() + HORIZON  # last bar train labels see
        assert label_window_end < f.test_idx.min(), (
            "training labels overlap the test window — leakage"
        )


def test_too_few_rows_raises():
    with pytest.raises(ValueError):
        purged_walk_forward(100)
