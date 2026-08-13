"""Charter-mandated leakage suite (CLAUDE.md, "Leakage tests").

Phase 0 status: STUBS. Each test names its contract now so the suite's scope
is fixed before any feature or model code exists; implementations land in
Phase 3 (except the holdout-boundary check, which lands in Phase 1 with the
holdout manifest). The conftest counts these skips and prints the number of
unimplemented leakage tests on every run — the count must reach 0 by the end
of Phase 3, after which a red leakage test blocks all merges with no
override.
"""

import pytest

pytestmark = pytest.mark.leakage_stub


def test_shuffled_labels_give_chance_auc():
    """Training on shuffled labels must produce AUC ~ 0.50 on walk-forward
    folds. Materially above chance => label information is leaking into
    features or normalization."""
    raise NotImplementedError


def test_forward_shifted_features_collapse_performance():
    """Shifting all features one bar forward (so each row sees tomorrow's
    features) must NOT improve — and shifting them one bar backward must
    materially degrade — performance. If a one-bar lag doesn't hurt, the
    features aren't carrying the timing signal we think they are."""
    raise NotImplementedError


def test_no_feature_uses_data_after_its_own_bar():
    """Every feature at bar t may use only data published by the close of t,
    including publication lag from the data dictionary."""
    raise NotImplementedError


def test_no_label_information_in_rolling_normalization():
    """Scalers, imputers, and percentile thresholds must be fit on training
    folds only; no rolling window used for normalization may extend past the
    feature timestamp."""
    raise NotImplementedError


def test_revised_series_used_at_first_print():
    """Any series flagged as revisable in docs/data_dictionary.md must enter
    the dataset at its first-print value, or be excluded."""
    raise NotImplementedError


def test_no_experiment_crosses_holdout_boundary():
    """No experiment's data range may cross the boundary recorded in
    data/HOLDOUT_MANIFEST.json. (Implementable in Phase 1, with the
    manifest.)"""
    raise NotImplementedError
