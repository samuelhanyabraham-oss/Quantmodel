"""Shared evaluation: every strategy (model or baseline) is scored by this
module and nothing else, so comparisons can't diverge in accounting.

Metrics per charter: AUC, Brier, precision/recall at the operating threshold,
NPS (net, docs/cost_model.md), lead time. Raw accuracy is computed nowhere.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, precision_score, recall_score, roc_auc_score

from .costs import net_protection_score
from .labels import HORIZON
from .stats import block_bootstrap_ci, effective_n


def auc(y: np.ndarray, p: np.ndarray) -> float:
    return roc_auc_score(y, p)


def lead_times(signal: pd.Series, label: pd.Series, lookback: int = HORIZON) -> dict:
    """Days of warning before each regime onset (0->1 label transition).

    For each onset at t0: lead = t0 - (first day in [t0-lookback, t0] from
    which the signal stayed on through t0). 0 = fired the day of onset,
    missed = never on in the window.
    """
    lab = label.to_numpy()
    sig = signal.to_numpy()
    onsets = np.where((lab[1:] == 1) & (lab[:-1] == 0))[0] + 1
    leads, missed = [], 0
    for t0 in onsets:
        lo = max(0, t0 - lookback)
        window = sig[lo : t0 + 1]
        on = np.where(window > 0)[0]
        if len(on) == 0:
            missed += 1
            continue
        # earliest index from which the signal is on continuously through t0
        run_start = len(window) - 1
        while run_start > 0 and window[run_start - 1] > 0:
            run_start -= 1
        if window[-1] > 0:
            leads.append(len(window) - 1 - run_start)
        else:
            missed += 1
    return {
        "n_onsets": int(len(onsets)),
        "n_missed": int(missed),
        "median_lead_days": float(np.median(leads)) if leads else np.nan,
        "mean_lead_days": float(np.mean(leads)) if leads else np.nan,
    }


def evaluate_strategy(
    prob: pd.Series,
    signal: pd.Series,
    df: pd.DataFrame,
    *,
    name: str,
    seed: int = 0,
) -> dict:
    """Full charter metric set on aligned prob/signal vs df with columns
    label, fwd_ret_10. df must already be restricted to evaluation rows
    (test folds only)."""
    y = df["label"].to_numpy()
    p = prob.to_numpy(dtype=float)
    s = signal.to_numpy(dtype=float)
    r = df["fwd_ret_10"].to_numpy()

    auc_pt, auc_lo, auc_hi = block_bootstrap_ci(auc, y, p, seed=seed)
    brier_pt, brier_lo, brier_hi = block_bootstrap_ci(brier_score_loss, y, p, seed=seed)
    lt = lead_times(signal, df["label"])
    return {
        "name": name,
        "n_rows": int(len(y)),
        "effective_n": round(effective_n(len(y)), 1),
        "base_rate": round(float(np.mean(y)), 4),
        "auc": round(auc_pt, 4),
        "auc_ci90": [round(auc_lo, 4), round(auc_hi, 4)],
        "brier": round(brier_pt, 4),
        "brier_ci90": [round(brier_lo, 4), round(brier_hi, 4)],
        "precision": round(float(precision_score(y, s, zero_division=0)), 4),
        "recall": round(float(recall_score(y, s, zero_division=0)), 4),
        "false_negative_rate": round(float(np.mean((s == 0) & (y == 1)) / max(np.mean(y), 1e-9)), 4),
        "false_positive_days_frac": round(float(np.mean((s == 1) & (y == 0))), 4),
        "nps_net_ann": round(net_protection_score(s, r), 5),
        "hedge_on_frac": round(float(np.mean(s)), 4),
        **{f"lead_{k}": v for k, v in lt.items()},
    }


def auc_diff_pvalue(
    y: np.ndarray, p_model: np.ndarray, p_base: np.ndarray, *, seed: int = 0
) -> tuple[float, float]:
    """(auc_diff, one-sided bootstrap p) for H0: model AUC <= baseline AUC,
    via the same circular block bootstrap as every CI."""
    from .stats import BLOCK_LEN, N_BOOT

    rng = np.random.default_rng(seed)
    n = len(y)
    n_blocks = int(np.ceil(n / BLOCK_LEN))
    point = roc_auc_score(y, p_model) - roc_auc_score(y, p_base)
    diffs = []
    for _ in range(N_BOOT):
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate([(s + np.arange(BLOCK_LEN)) % n for s in starts])[:n]
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y[idx], p_model[idx]) - roc_auc_score(y[idx], p_base[idx]))
    diffs = np.asarray(diffs)
    # one-sided p: probability the bootstrap difference is <= 0
    p_raw = float(np.mean(diffs <= 0.0)) if len(diffs) else np.nan
    return float(point), p_raw
