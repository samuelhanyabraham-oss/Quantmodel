"""Charter-mandated leakage suite (CLAUDE.md, "Leakage tests") — LIVE.

Implemented in Phase 3. A red test here blocks all merges; there is no
override. Tests that need the real pipeline run it on the dev snapshot.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from regime import data, features, labels, models
from regime.validation import purged_walk_forward

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def dev():
    panel = data.load_dev_panel()
    lab = labels.build_labels(panel["SPY_close"])
    feats = features.build_features(panel)
    df = panel.join(lab).join(feats).dropna()
    return df


def _pooled_auc(X, y, seed=0):
    from sklearn.metrics import roc_auc_score

    probs = models.walk_forward_probs(X, y, "gbm", seed=seed)
    mask = probs.notna()
    return roc_auc_score(y[mask], probs[mask])


def test_shuffled_labels_give_chance_auc(dev):
    """Training on shuffled labels must produce AUC ~ 0.50."""
    X = dev[features.FEATURE_NAMES]
    rng = np.random.default_rng(7)
    aucs = []
    for seed in range(3):
        y_shuf = pd.Series(
            rng.permutation(dev["label"].to_numpy()), index=dev.index
        )
        aucs.append(_pooled_auc(X, y_shuf, seed=seed))
    mean_auc = float(np.mean(aucs))
    assert abs(mean_auc - 0.5) < 0.10, (
        f"shuffled-label AUC {mean_auc:.3f} far from 0.5 — leakage"
    )


def test_forward_shifted_features_collapse_performance(dev):
    """Staleness must hurt. Tested directly on the signal-bearing features
    (single-feature AUC), not through a fitted model: a model with no skill
    (AUC ~ 0.5) turns the model-level version of this test into a coin flip,
    which is exactly what happened on first implementation — the GBM's
    'improvement' under lag was noise around chance, while the features
    below show clean monotone decay. For each feature: lag-1 must not
    improve its AUC beyond tolerance, and lag-5 must sit strictly below
    lag-1's (staleness monotonically destroys timing information)."""
    from sklearn.metrics import roc_auc_score

    y = dev["label"]
    for name in ["rv10", "vix", "vix_slope", "rv_ratio_10_63"]:
        s = dev[name]
        a0 = roc_auc_score(y, s)
        a1 = roc_auc_score(y.iloc[1:], s.shift(1).iloc[1:])
        a5 = roc_auc_score(y.iloc[5:], s.shift(5).iloc[5:])
        assert a1 <= a0 + 0.02, f"{name}: lag-1 improved AUC ({a1:.3f} > {a0:.3f})"
        assert a5 < a1, f"{name}: lag-5 ({a5:.3f}) not below lag-1 ({a1:.3f})"
        assert a5 < a0 - 0.02, f"{name}: staleness barely hurts ({a0:.3f}->{a5:.3f})"


def test_no_feature_uses_data_after_its_own_bar(dev):
    """Prefix invariance: recomputing features on a truncated panel must give
    identical values at the truncation point. Any lookahead breaks this."""
    panel = data.load_dev_panel()
    full = features.build_features(panel)
    for t in [300, 500, len(panel) - 1]:
        trunc = features.build_features(panel.iloc[: t + 1])
        a = full.iloc[t].to_numpy(dtype=float)
        b = trunc.iloc[-1].to_numpy(dtype=float)
        np.testing.assert_allclose(a, b, rtol=1e-10, err_msg=f"lookahead at row {t}")


def test_no_label_information_in_rolling_normalization(dev):
    """Fold isolation: probabilities for fold k's test rows must be identical
    when data AFTER fold k's test block is replaced with garbage. If anything
    fit on later rows (scaler, calibrator, model) touched them, this breaks."""
    X = dev[features.FEATURE_NAMES]
    y = dev["label"]
    folds = purged_walk_forward(len(X))
    k = 1  # an early fold with plenty of data after it
    end = folds[k].test_idx.max()

    probs_full = models.walk_forward_probs(X, y, "logistic", n_folds=5)

    X_g = X.copy()
    y_g = y.copy()
    X_g.iloc[end + 1 :] = 999.0
    y_g.iloc[end + 1 :] = 1.0
    probs_garbled = models.walk_forward_probs(X_g, y_g, "logistic", n_folds=5)

    a = probs_full.iloc[folds[k].test_idx].to_numpy(dtype=float)
    b = probs_garbled.iloc[folds[k].test_idx].to_numpy(dtype=float)
    np.testing.assert_allclose(a, b, rtol=1e-10, err_msg="future data leaked into fold")


def test_revised_series_used_at_first_print():
    """All snapshot series must be documented in docs/data_dictionary.md as
    revision-free (exchange/index prints). Adding a revisable series without
    first-print handling must turn this red."""
    manifest = data.load_manifest()
    dictionary = (ROOT / "docs" / "data_dictionary.md").read_text()
    revision_free = {"SPY", "QQQ", "IWM", "HYG", "LQD", "VIX", "VIX3M"}
    for s in manifest["series"]:
        assert s in revision_free, (
            f"series {s} not in the audited revision-free set — document its "
            "revision policy and first-print handling before use"
        )
        assert f"| {s} |" in dictionary, f"series {s} missing from data dictionary"


def test_no_experiment_crosses_holdout_boundary():
    """Every logged experiment must carry a frozen snapshot hash. Before the
    owner-ordered dissolution that means the dev hash (plus the holdout hash
    for the one authorized evaluation); after it, the frozen full-snapshot
    hash. Unfrozen or unknown data hashes are never legal."""
    manifest = data.load_manifest()
    assert manifest["dev_end"] <= manifest["holdout_boundary_date"]

    data.load_dev_panel()  # verifies the active snapshot hash internally

    # Holdout-hash entries are legal ONLY while the recorded owner
    # authorization exists (charter changelog 2026-08-13).
    unlock_path = ROOT / "data" / "HOLDOUT_UNLOCK.json"
    allowed = {manifest["dev_sha256"]}
    if unlock_path.exists() and json.loads(unlock_path.read_text()).get("unlocked"):
        allowed.add(manifest["holdout_sha256"])
    if manifest.get("dissolved"):
        allowed.add(manifest["full_sha256"])
    book_manifest = ROOT / "data" / "BOOK_MANIFEST.json"
    if book_manifest.exists():
        allowed.add(json.loads(book_manifest.read_text())["book_sha256"])

    log_path = ROOT / "experiments.jsonl"
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            entry = json.loads(line)
            assert entry["data_snapshot_hash"] in allowed, (
                f"experiment {entry['config_hash'][:8]} ran on unauthorized data"
            )
