# Gameplan

Phased plan for the volatility/drawdown regime project. One phase at a time;
each phase ends at a hard stop with a deliverable and exit criteria.

**Status (2026-08-13):** Phases 0–5 complete under a blanket "finish the
project" approval; per-phase artifacts are committed and every run is in
`experiments.jsonl`. Walk-forward outcome: negative result (see REPORT.md).
Phase 6 complete: the owner unlocked the holdout on 2026-08-13
(data/HOLDOUT_UNLOCK.json) and the single pre-registered evaluation ran —
final verdict in REPORT.md: no skill claimed; operating point degenerated
to always-hedged out-of-sample. Data-source deviation from this plan: the network
policy blocked Stooq/FRED/CBOE, so the frozen snapshot uses IBKR MCP daily
bars (5-year history cap — the binding limitation), with HYG/LQD replacing
FRED credit spreads; see docs/data_dictionary.md.

The ordering principle: **everything that could flatter a result is built and
frozen before any model exists.** Harness → data → labels/baselines/validation →
features → models → report. The holdout is last and gated separately.

---

## Phase 0 — Charter and scaffolding

**Goal:** a repo where cheating is structurally hard before any research starts.

Deliverables:
- `CLAUDE.md` (charter) and this gameplan. *(done — this commit)*
- Repo skeleton: `src/`, `tests/`, `data/` (gitignored raw), `docs/`,
  `experiments.jsonl` logger utility.
- Pinned environment (`requirements.txt` or `pyproject.toml`, exact versions).
- CI workflow that runs the test suite on every push. The leakage tests are
  created **now as failing stubs** (`xfail`-style is not allowed — they are
  skipped-with-a-counter until Phase 3 implements them, and CI prints the count
  of unimplemented leakage tests so it can't be forgotten).
- Experiment logger: every run appends `{timestamp, git_sha, config_hash,
  data_snapshot_hash, seed, metrics, notes}` to `experiments.jsonl`. Nothing in
  `src/` can train without going through the logger.

Exit criteria: CI green on a trivial test; logger round-trips a dummy run.
**Stop. Await approval.**

## Phase 1 — Data: sources, as-of policy, snapshot, holdout carve-out

**Goal:** a frozen, documented dataset with the holdout physically separated
before anyone has seen a label.

Proposed universe (to confirm at this phase's stop point):
- Equity indices/ETFs: S&P 500 (SPY or ^GSPC), Nasdaq-100 (QQQ), small caps
  (IWM). Prices only, for realized vol and drawdown labels + return features.
- Implied vol term structure: VIX, VIX3M (and VIX9D if history suffices) —
  needed for baseline #2 and features.
- Credit: HY OAS and IG OAS (FRED, ICE BofA series).
- Rates/curve: 10y–3m and 10y–2y spreads (FRED).
- Candidate sources: Stooq or Yahoo for prices; FRED for credit/rates; CBOE
  for vol indices. Free, daily, long history.

Revision policy work (the part that actually matters):
- `docs/data_dictionary.md`: per series — source, publication lag, revision
  behavior, decision (use / use-with-lag / exclude). FRED credit and rate
  series are near-final at first print but each gets an explicit entry; any
  series we can't document gets excluded, not waved through.
- Timestamp convention documented once: a feature at bar `t` may use only data
  published by the close of `t` (publication lag applied, not ignored).

Mechanics:
- Snapshot all raw pulls to `data/snapshots/` with content hashes.
- Carve out the holdout: most recent 18 months from snapshot date. Write
  `data/HOLDOUT_MANIFEST.json` (boundary date, row hashes). Wire the CI
  boundary-crossing assertion (this one is implementable now and goes live
  immediately).

Exit criteria: dictionary covers every series; snapshots hashed; holdout
manifest written; boundary test live in CI.
**Stop. Confirm sources and universe. Await approval.**

## Phase 2 — Labels, baselines, validation harness

**Goal:** the full evaluation machinery, proven on baselines alone, before any
model exists. If the harness is wrong we find out here, where it's cheap.

Deliverables:
- Label construction (primary binary + forward RV + forward max drawdown),
  from `t+1` forward only, with unit tests on synthetic series where the right
  answer is known by hand.
- Purged walk-forward CV with ≥10-day embargo; embargo unit-tested on
  synthetic data (deliberately planted leakage must be caught).
- Effective sample size machinery: choose and document the method (planned:
  stationary block bootstrap for CIs, block length ≳ label horizon), applied
  to every metric we report.
- Multiple-testing adjustment method chosen and documented **now**, before any
  model result exists (planned: report both a Bonferroni-style bound using the
  `experiments.jsonl` run count and a White's-reality-check-style bootstrap;
  final choice justified in writing).
- Cost model: `docs/cost_model.md` — hedge bleed per day-on, switch friction.
  Parameters frozen at the end of this phase.
- All three baselines implemented and evaluated through the harness on the
  walk-forward folds (never the holdout). Their numbers become the bar.

Exit criteria: baseline results table with effective-N confidence intervals;
synthetic-leakage tests pass; cost model frozen.
**Stop. This is where we agree on "the number to beat." Await approval.**

## Phase 3 — Features + live leakage suite

**Goal:** a small, documented, point-in-time-correct feature set — and the full
leakage suite armed in CI.

- Feature families (deliberately boring, ~15–25 features total): realized-vol
  levels/changes at several windows, vol-of-vol, term-structure slopes
  (VIX/VIX3M), credit spread levels/changes, curve slopes, drawdown state,
  return skew/kurt over trailing windows. Every feature gets a one-line
  docstring stating its information timestamp.
- No feature selection against the label outside the harness. Any pruning
  happens inside training folds only.
- Implement the remaining leakage tests from the charter: shuffled-label AUC ≈
  0.50, one-bar-forward shift collapse, no-future-timestamp assertion, no
  label info in rolling normalizations, first-print assertion. Skipped-stub
  counter reaches zero; suite blocks merges from here on.

Exit criteria: leakage suite fully live and green; feature docs complete.
**Stop. Await approval.**

## Phase 4 — Models

**Goal:** the smallest model that could work, evaluated identically to the
baselines.

- Order of attack: (1) regularized logistic regression, (2) gradient-boosted
  trees, small depth. Nothing deeper unless both fail *and* there's a stated
  reason to believe capacity is the problem.
- Calibration (isotonic or Platt) fit inside training folds only.
- Every run logged. Hyperparameter searches are runs; they all count toward
  the multiple-testing denominator.
- Compare against baselines net, with effective-N CIs. If nothing beats
  persistence net, that is the result, and we proceed to Phase 6 to report it.

Exit criteria: walk-forward results table, model vs. all baselines, net.
**Stop. Await approval.**

## Phase 5 — Threshold, hedge-band mapping, freeze

**Goal:** the decision layer, frozen before holdout.

- Operating threshold chosen against the asymmetric cost model (FN expensive,
  FP bounded), justified in writing — not balanced accuracy.
- Probability → hedge-ratio band mapping: monotone, piecewise-constant, few
  bands. Frozen in a committed file with its own hash.
- Lead-time analysis on walk-forward folds: distribution of days-before-drawdown
  the signal fires.

Exit criteria: `docs/freeze.md` containing threshold, mapping, and the
pre-registered holdout evaluation plan (exact metrics, exact code path).
**Stop. Await approval.**

## Phase 6 — Holdout evaluation and REPORT.md

**Gated: does not start without a message that explicitly grants holdout
access.**

- One evaluation, using the pre-registered plan from Phase 5, run once.
- REPORT.md: §1 "Why this might be wrong", §2 "What would falsify this",
  §3 performance (skill above baselines net, effective N, run count +
  adjusted estimate, calibration, lead time, FN/FP costs separately).
- If the model failed to beat persistence in Phase 4, this phase still runs —
  as a negative-result report with the same structure.

---

## Open questions to resolve at the Phase 1 stop

1. Which underlying defines "the" regime — SPY alone, or a small panel with
   one label per index? (Plan assumes SPY as primary; others as features.)
2. Data sources: are the free daily sources above acceptable, or do you have
   preferred paid feeds with proper point-in-time history?
3. Cost-model parameters: what hedge instrument should the bleed model assume
   (index puts, VIX calls, futures overlay)? This sets the FP cost scale.
4. Multiple-testing method preference, if any, beyond the Phase 2 proposal.

## Risk register (why this whole project might produce nothing)

- Persistence is genuinely hard to beat at a 10-day horizon; the honest prior
  is that added skill will be small and may not survive costs.
- Effective N is brutal: ~25 years of daily data with a 10-day overlapping
  label is on the order of only a few hundred independent observations; CIs
  will be wide, and many "improvements" will be indistinguishable from noise.
- Few independent crisis episodes (2008, 2011, 2015, 2018, 2020, 2022, …):
  the label's positive class clusters into a handful of regimes, so
  walk-forward folds are not exchangeable and fold variance will be high.
- Free data sources may lack usable first-print history for some series; the
  data dictionary work may shrink the feature set materially.
