"""Purged walk-forward cross-validation with embargo.

Expanding-window walk-forward: each fold trains on everything up to the fold
start, skips an embargo, then tests on the next block. Additionally the last
HORIZON training rows before the embargo are PURGED, because their labels
look into the embargo/test period (10-day forward window).

No random K-fold exists anywhere in this project, by charter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .labels import HORIZON

EMBARGO = 10  # trading days between train end and test start (>= charter min)


@dataclass(frozen=True)
class Fold:
    train_idx: np.ndarray
    test_idx: np.ndarray


def purged_walk_forward(
    n: int,
    n_folds: int = 5,
    min_train: int = 150,
    embargo: int = EMBARGO,
    horizon: int = HORIZON,
) -> list[Fold]:
    """Split range(n) into walk-forward folds.

    Train = [0, fold_start - embargo - horizon); the `horizon` purge removes
    training rows whose forward-looking labels overlap the embargo/test
    window. Test blocks tile the region after min_train.
    """
    if n <= min_train + embargo + horizon + n_folds:
        raise ValueError(f"not enough rows ({n}) for {n_folds} folds")
    test_len = (n - min_train) // n_folds
    folds = []
    for k in range(n_folds):
        test_start = min_train + k * test_len
        test_end = n if k == n_folds - 1 else test_start + test_len
        train_end = test_start - embargo - horizon
        if train_end <= 0:
            continue
        folds.append(
            Fold(
                train_idx=np.arange(0, train_end),
                test_idx=np.arange(test_start, test_end),
            )
        )
    return folds
