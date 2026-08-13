# Project: Volatility / Drawdown Regime Model

## What this is

A research harness that estimates the probability of an elevated-volatility or
drawdown regime over the next 5–10 trading days, for the purpose of sizing
portfolio hedges.

It is **not** a directional forecaster. It does not predict price. It must never
be described, documented, or extended as one.

## Prime directive

Skepticism outranks performance. A negative result reported honestly is a
successful outcome. A high score is presumed to be a bug until proven otherwise.

## Workflow rules

- Plan mode first. Propose, wait for my approval, then write code.
- One phase at a time. Stop at every phase boundary and wait.
- Never touch the holdout set without explicit permission in that message.
- Log every experiment run to `experiments.jsonl` (config hash, timestamp,
  metrics). I need the total run count for multiple-testing adjustment.
- If you find yourself about to relax a rule in this file, stop and ask instead.
- Changes to this charter are themselves logged: every amendment gets an entry
  in the Charter changelog at the bottom of this file, with date and reason.

## Target definition — do not change without asking

Primary label: `1` if realized volatility over the **next** 10 trading days
exceeds its trailing 1-year 75th percentile, else `0`.

Secondary (continuous) labels, produced alongside:
- forward 10-day realized volatility
- forward maximum drawdown over the same window

All labels are computed from `t+1` forward only. No label may be derived from
information available at or before the feature timestamp.

Note the base rate is ~25% by construction. Accuracy is therefore doubly
meaningless (a constant `0` scores ~75%); this is another reason raw accuracy
never appears as a headline metric.

## Baselines — the model must beat ALL of these, net

1. **Persistence.** Today's regime equals tomorrow's regime. This is the killer
   baseline. Volatility is strongly autocorrelated, so most "85% accurate"
   regime models are nothing more than this restated. If the model does not beat
   persistence by a meaningful margin, the project has failed and the report says
   so.
2. **Vol term-structure rule.** Short-dated vs. longer-dated implied vol
   inversion as a simple binary signal.
3. **Trailing realized-vol percentile threshold.** A single rolling rule.

Report skill *above* these baselines. Raw accuracy is not a headline metric and
should not appear as one.

**"Net" means:** baselines and model are compared after the same cost
accounting — hedge bleed while the signal is on, and a per-switch friction for
turning the hedge on/off. The cost model is defined once (in
`docs/cost_model.md`), applied identically to model and baselines, and its
parameters are frozen before any model sees the test folds. A model that beats
baselines gross but loses net has failed.

## Validation

- Purged walk-forward cross-validation. No random K-fold, ever.
- Embargo of at least 10 trading days between train and test, because the labels
  overlap and will otherwise leak across the boundary.
- **Overlapping labels:** report effective sample size, not raw row count. With a
  10-day forward window on daily bars, the raw count overstates independent
  observations by roughly an order of magnitude. Confidence intervals must use
  the effective count (block bootstrap or Newey–West-style adjustment; method
  documented once and reused everywhere).
- Holdout: the most recent 18 months, untouched until I say otherwise. The
  holdout boundary date and a hash of the holdout rows are written to
  `data/HOLDOUT_MANIFEST.json` at carve-out time; CI fails if any experiment's
  data range crosses the boundary.
- Fit all scalers, imputers, and percentile thresholds on the training fold only.
- Metrics: AUC, Brier score, precision and recall at the chosen operating
  threshold, calibration curve, and **lead time** (how many days before a
  drawdown begins does the signal fire).
- **Multiple-testing adjustment:** the final report's headline skill estimate is
  adjusted for the total number of runs in `experiments.jsonl` (all runs count,
  including failures and dead ends). The adjustment method is chosen and
  documented in Phase 2, before any model results exist.
- **Threshold freeze:** the operating threshold and the hedge-band mapping are
  chosen on walk-forward results and frozen in writing before the holdout is
  ever evaluated.

## Asymmetric costs

The two error types are not equivalent and must be reported separately:
- **False negative:** unhedged into a drawdown. Expensive.
- **False positive:** hedge bleed in a calm tape. Survivable, and bounded.

Any threshold selection must be justified against this asymmetry, not against
balanced accuracy.

## Leakage tests — these must exist, run in CI, and fail loudly

- Shuffled-label run must produce AUC ≈ 0.50. If it doesn't, there is leakage.
- Shifting all features one bar forward must collapse performance. If it doesn't,
  the features aren't carrying the signal you think they are.
- Assert no feature uses data timestamped after its own bar.
- Assert no label information enters any rolling normalization.
- Assert no series with post-hoc revisions is used at its revised value.
- Assert no experiment's data range crosses the holdout boundary
  (`data/HOLDOUT_MANIFEST.json`).
- The leakage suite runs on every push. A red leakage test blocks all merges;
  there is no override label.

## Reproducibility

- Every run is reproducible from its `experiments.jsonl` entry: config hash,
  code commit, data snapshot hash, RNG seed.
- Raw data pulls are snapshotted (content-hashed) before use; experiments run
  against snapshots, never against a live API, so a restated series can't
  silently change a past result.
- One environment file (pinned versions). No "works on my machine."

## Data

Indices, ETFs, and volatility/credit series only. There is no single-name
universe, so survivorship bias does not apply to this project — do not add
survivorship machinery that isn't needed.

Every series must have a documented as-of and revision policy, recorded in
`docs/data_dictionary.md` (one row per series: source, publication lag,
revision behavior, first-print availability). Series that are restated after
initial publication must be used at their first-print value or excluded.

Sources: _[to be confirmed in Phase 1]_

## Output contract

The model's job is to emit a **hedge-ratio suggestion band**, not a trade. The
final deliverable maps a probability to a hedge range. It never emits, places, or
formats an order.

The mapping is monotone (higher probability never suggests a smaller hedge),
piecewise-constant with few bands, and frozen alongside the operating
threshold before holdout evaluation.

## REPORT.md structure

Section 1 is always "Why this might be wrong."
Section 2 is always "What would falsify this."
Performance comes after both.

Section 3 reports: skill above each baseline (net), effective sample size,
run count and the multiple-testing-adjusted estimate, calibration, lead time,
and the two error costs separately. Raw accuracy appears nowhere as a headline.

## Charter changelog

- 2026-08-13 — Holdout unlocked by owner ("go ahead and run it", in direct
  reply to the unlock question). Recorded in `data/HOLDOUT_UNLOCK.json`. The
  holdout-boundary leakage test now additionally accepts experiment entries
  bearing the holdout snapshot hash **only while that unlock file exists**;
  dev-hash-only enforcement remains for all other entries. Scope: the single
  pre-registered evaluation in docs/freeze.md.

- 2026-08-13 — Initial charter committed. Additions over the original draft:
  base-rate note under the target definition; explicit definition of "net" and
  a frozen cost model; holdout manifest + CI boundary check; multiple-testing
  adjustment and threshold-freeze requirements under Validation; two extra
  leakage assertions (holdout boundary, no-override CI); Reproducibility
  section (snapshots, seeds, pinned env); data dictionary requirement; monotone
  frozen hedge-band mapping; REPORT.md Section 3 contents; this changelog.
