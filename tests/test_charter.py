"""Charter-enforcement tests (added 2026-09-30, "self-improvement" pass).

CLAUDE.md states rules the code base had been following by convention.
These make the cheap ones mechanical so CI, not memory, holds the line.
"""

import json
import re
from pathlib import Path

import numpy as np
import pytest

from regime import data, experiment_log
from regime.costs import net_protection_score
from regime.stats import timing_value_test

ROOT = Path(__file__).resolve().parents[1]


def test_report_sections_1_and_2_come_first():
    text = (ROOT / "REPORT.md").read_text()
    heads = [m.group(1).strip() for m in re.finditer(r"^## (.+)$", text, flags=re.M)]
    assert heads[0].startswith("1. Why this might be wrong"), heads[:3]
    assert heads[1].startswith("2. What would falsify this"), heads[:3]


def test_raw_accuracy_is_never_a_headline():
    """'accuracy' may appear only in sentences that disclaim it."""
    text = (ROOT / "REPORT.md").read_text()
    for m in re.finditer(r"accuracy", text, flags=re.I):
        line = text[text.rfind("\n", 0, m.start()) + 1 : text.find("\n", m.end())]
        assert re.search(r"never|nowhere|not a headline|meaningless|not against|not appear", line, flags=re.I), line


def test_every_result_file_names_a_frozen_hash():
    from regime import book
    from regime import long_history
    allowed = data.frozen_hashes() | book.frozen_book_hashes() | long_history.frozen_long_hashes()
    for f in (ROOT / "results").glob("*.json"):
        d = json.loads(f.read_text())
        hashes = [v for k, v in d.items() if isinstance(v, str) and k.endswith("hash") and len(v) == 64]
        if hashes:  # older result files predate the field; new ones must carry it
            assert all(h in allowed for h in hashes), f.name


def test_experiment_log_is_append_only_and_complete():
    runs = experiment_log.read_runs()
    assert len(runs) == experiment_log.run_count()
    ts = [r.timestamp for r in runs]
    assert ts == sorted(ts), "experiment log timestamps are not monotone — was it edited?"
    for r in runs:
        assert r.data_snapshot_hash and r.git_sha and r.config_hash


def test_frozen_files_are_not_modified():
    """Every hash in every manifest must still verify against its file."""
    data.load_dev_panel()
    from regime import book, long_history
    book.load_book_panel()
    if long_history.LONG_MANIFEST.exists():
        long_history.load_long_panel()
    m = data.load_manifest()
    for v in m.get("versions", []):
        assert data.sha256_file(data.SNAP_DIR / v["path"]) == v["sha256"], v["path"]
    assert data.sha256_file(data.SNAP_DIR / "panel_dev.csv") == m["dev_sha256"]
    assert data.sha256_file(data.SNAP_DIR / "panel_full.csv") == m["full_sha256"]
    assert data.sha256_file(data.SNAP_DIR / "panel_holdout.csv") == m["holdout_sha256"]
    bm = book.load_book_manifest()
    assert book._sha256(book.BOOK_SNAP) == bm["book_sha256"]
    for v in bm.get("versions", []):
        assert book._sha256(book.BOOK_SNAP.parent / v["path"]) == v["sha256"], v["path"]


def test_timing_test_is_calibrated_on_noise_and_detects_real_timing():
    rng = np.random.default_rng(3)
    n = 3000
    r = rng.normal(0, 0.03, n)
    # a noise signal with realistic run lengths: on-fraction ~0.3
    noise = (np.convolve(rng.normal(size=n), np.ones(15) / 15, mode="same") > 0.1).astype(float)
    p_noise = timing_value_test(noise, r, net_protection_score, n_perm=400, seed=1)["p_one_sided"]
    assert p_noise > 0.05, p_noise
    # an oracle that is on exactly when the forward return is negative
    oracle = (r < 0).astype(float)
    res = timing_value_test(oracle, r, net_protection_score, n_perm=400, seed=1)
    assert res["p_one_sided"] < 0.01 and res["excess_over_null"] > 0


def test_baselines_are_nan_not_zero_during_warmup():
    """Erratum 2026-09-30: a baseline computed on a truncated close must not
    silently read 0 where its threshold is undefined."""
    import pandas as pd
    from regime import baselines
    rng = np.random.default_rng(7)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400))))
    s = baselines.persistence(close)
    assert s.iloc[:261].isna().all() and s.iloc[261:].notna().all()
    assert set(s.dropna().unique()) <= {0.0, 1.0}


def test_effective_n_acf_matches_iid_and_shrinks_under_dependence():
    from regime.stats import effective_n_acf
    rng = np.random.default_rng(5)
    iid = rng.integers(0, 2, 4000)
    assert 0.85 * 4000 < effective_n_acf(iid) <= 4000 * 1.05
    # a label that switches state every ~40 days: far fewer independent obs
    blocks = np.repeat(rng.integers(0, 2, 100), 40)
    assert effective_n_acf(blocks) < 4000 / 15
