# Pre-holdout freeze — 2026-08-13

Everything below is fixed BEFORE any holdout row is read. The holdout remains
locked until explicitly unlocked in writing (CLAUDE.md). Nothing in this file
may change after holdout evaluation begins.

## Frozen primary model

`gbm_small4`: HistGradientBoostingClassifier(max_depth=2, max_iter=100,
learning_rate=0.05, min_samples_leaf=20, l2_regularization=1.0), features
`[rv10, vix, vix_slope, rv_ratio_10_63]`, Platt calibration on the
chronological tail 25% of each training fold, seed 20260813. Selected on
walk-forward results among the 4 logged model variants; that selection is
part of the multiple-testing burden (M in experiments.jsonl).

## Frozen operating threshold

**p ≥ 0.20** ⇒ hedge signal on.

Justification against the cost asymmetry (not balanced accuracy): 0.20 is
the highest-recall operating point (recall 0.37) whose walk-forward NPS is
not negative; below it, false-positive bleed dominates; above it, recall
collapses toward zero and the expensive error (unhedged into drawdown)
explodes. Stated plainly: its +22 bp/yr NPS is a single positive cell in a
noisy grid with negative neighbors, and is NOT claimed as positive net skill.

## Frozen hedge-ratio band mapping (monotone, piecewise-constant)

| calibrated p | suggested hedge ratio band |
|---|---|
| p < 0.20 | 0–10% |
| 0.20 ≤ p < 0.30 | 25–50% |
| p ≥ 0.30 | 50–75% |

Output contract unchanged: bands only, never orders.

## Pre-registered holdout evaluation plan

One run, once, via the same `evaluate.evaluate_strategy` code path:
1. Build labels/features on the holdout panel (warm-up rows drawn only from
   data at or before each bar, as everywhere).
2. Score the frozen model trained on the full dev set (same hyperparameters,
   scaler and calibrator fit on dev only; calibrator on dev's chronological
   tail 25%).
3. Report the charter metric set + NPS at the frozen threshold, plus the same
   three baselines, with block-bootstrap CIs and the Bonferroni-M adjustment
   at whatever M then stands in experiments.jsonl.
4. The decision rule fixed in docs/multiple_testing.md applies: skill is
   claimed only if p_adj < 0.10. No re-runs, no threshold changes, no new
   variants after the first holdout read.
