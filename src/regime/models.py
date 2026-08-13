"""Models + the walk-forward runner.

Order of attack per the gameplan: regularized logistic regression first,
shallow gradient boosting second. Scaler and calibrator are fit inside each
training fold only. Calibration uses a CHRONOLOGICAL tail split of the train
fold (no random splits anywhere, per charter).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .validation import purged_walk_forward

CAL_FRAC = 0.25  # chronological tail of train used to fit the calibrator


def _fit_sigmoid_calibrator(scores: np.ndarray, y: np.ndarray):
    """Platt scaling: 1-feature logistic on held-out chronological tail.
    Isotonic would overfit at this sample size."""
    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(scores.reshape(-1, 1), y)
    return lambda s: lr.predict_proba(s.reshape(-1, 1))[:, 1]


def make_model(kind: str, seed: int):
    if kind == "logistic":
        return LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced")
    if kind == "gbm":
        return HistGradientBoostingClassifier(
            max_depth=2,
            max_iter=100,
            learning_rate=0.05,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=seed,
        )
    raise ValueError(kind)


def walk_forward_probs(
    X: pd.DataFrame, y: pd.Series, kind: str, *, seed: int = 0, n_folds: int = 5
) -> pd.Series:
    """Out-of-sample probabilities on every test row across walk-forward
    folds. NaN outside test rows."""
    folds = purged_walk_forward(len(X), n_folds=n_folds)
    out = pd.Series(np.nan, index=X.index)
    Xv, yv = X.to_numpy(dtype=float), y.to_numpy(dtype=float)
    for f in folds:
        tr, te = f.train_idx, f.test_idx
        cut = int(len(tr) * (1 - CAL_FRAC))
        fit_idx, cal_idx = tr[:cut], tr[cut:]
        scaler = StandardScaler().fit(Xv[fit_idx])
        model = make_model(kind, seed)
        model.fit(scaler.transform(Xv[fit_idx]), yv[fit_idx])
        raw_cal = model.predict_proba(scaler.transform(Xv[cal_idx]))[:, 1]
        if len(np.unique(yv[cal_idx])) < 2:
            calibrate = lambda s: s  # degenerate tail: skip calibration
        else:
            calibrate = _fit_sigmoid_calibrator(raw_cal, yv[cal_idx])
        raw_te = model.predict_proba(scaler.transform(Xv[te]))[:, 1]
        out.iloc[te] = calibrate(raw_te)
    return out
