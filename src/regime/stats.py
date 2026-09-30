"""Effective-sample-size-aware statistics.

With a 10-day overlapping label on daily bars, adjacent rows are strongly
dependent: raw row counts overstate the independent information by roughly
the overlap factor. Two tools, used everywhere and documented once here:

- effective_n: raw count deflated by the label-overlap factor (n / HORIZON).
  Reported alongside every raw count.
- Stationary (circular) block bootstrap for confidence intervals on any
  metric, block length = 2 * HORIZON (covers label overlap plus short-range
  vol clustering). All CIs in this project come from this function.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .labels import HORIZON

BLOCK_LEN = 2 * HORIZON
N_BOOT = 2000


def effective_n(n_rows: int, horizon: int = HORIZON) -> float:
    return n_rows / horizon


def block_bootstrap_ci(
    metric_fn,
    y: np.ndarray,
    p: np.ndarray,
    *,
    n_boot: int = N_BOOT,
    block_len: int = BLOCK_LEN,
    alpha: float = 0.10,
    seed: int = 0,
) -> tuple[float, float, float]:
    """(point, lo, hi) for metric_fn(y, p) under a circular block bootstrap.

    90% CI by default — pretending to 95% precision with effective N in the
    tens would be false comfort. Bootstrap draws that are degenerate for the
    metric (e.g. single-class resamples for AUC) are skipped.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    point = float(metric_fn(y, p))
    n_blocks = int(np.ceil(n / block_len))
    vals = []
    for _ in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate([(s + np.arange(block_len)) % n for s in starts])[:n]
        try:
            v = metric_fn(y[idx], p[idx])
        except ValueError:
            continue
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return point, np.nan, np.nan
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def timing_value_test(
    signal: np.ndarray,
    fwd_ret: np.ndarray,
    score_fn,
    *,
    n_perm: int = 2000,
    min_shift: int = 2 * HORIZON,
    seed: int = 0,
) -> dict:
    """Does a binary signal's TIMING add value beyond its hedge-on fraction?

    Added 2026-09-30 after the long-history study showed the frozen NPS is
    dominated by a constant (always-hedged beats every rule). NPS itself is
    unchanged — this is a supplementary test. Null: the same signal
    circularly shifted by a random offset >= min_shift, which preserves its
    on-fraction AND its run-length structure exactly, so the only thing
    destroyed is alignment with the returns. Reports the observed score,
    the null mean, the excess, and a one-sided p (P[null >= observed]).
    A low p means the signal's timing, not its size, earns its score.
    """
    s = np.asarray(signal, dtype=float)
    r = np.asarray(fwd_ret, dtype=float)
    n = len(s)
    if n <= 2 * min_shift + 1:
        return {"observed": round(float(score_fn(s, r)), 5), "null_mean": None, "null_sd": None,
                "excess_over_null": None, "p_one_sided": None, "n_perm": 0,
                "note": f"too few rows ({n}) for a shift null with min_shift={min_shift}"}
    rng = np.random.default_rng(seed)
    observed = float(score_fn(s, r))
    null = np.empty(n_perm)
    for i in range(n_perm):
        k = int(rng.integers(min_shift, n - min_shift))
        null[i] = score_fn(np.roll(s, k), r)
    return {
        "observed": round(observed, 5),
        "null_mean": round(float(null.mean()), 5),
        "null_sd": round(float(null.std()), 5),
        "excess_over_null": round(observed - float(null.mean()), 5),
        # finite-sample p: (1 + #null >= obs) / (1 + n_perm), never exactly 0
        "p_one_sided": round(float((1 + np.sum(null >= observed)) / (1 + n_perm)), 4),
        "n_perm": n_perm,
    }


def effective_n_acf(y: np.ndarray, max_lag: int = 250) -> float:
    """Autocorrelation-based effective sample size: n / (1 + 2 * sum of
    positive-run autocorrelations), i.e. n over the integrated autocorrelation
    time. Added 2026-09-30 after the long-panel check showed label
    dependence persists ~80 days, so n / HORIZON (the frozen rule, kept and
    reported first) overstates independent observations ~3x. Reported
    alongside, never instead; changing the headline rule is a charter
    amendment."""
    s = pd.Series(np.asarray(y, dtype=float))
    n = len(s)
    tau = 1.0
    for k in range(1, min(max_lag, n - 2) + 1):
        r = s.autocorr(k)
        if not np.isfinite(r) or r <= 0:
            break
        tau += 2.0 * r
    return n / tau
