# REPORT — Volatility/Drawdown Regime Model (final: holdout evaluated)

Status: Phases 0–6 complete. The holdout was unlocked by the owner on
2026-08-13 (data/HOLDOUT_UNLOCK.json) and evaluated exactly once, per the
pre-registered plan in docs/freeze.md. Sections 1–3 below are the
walk-forward analysis (written before the holdout was read and left
unedited); the holdout section follows them.

## 1. Why this might be wrong

- **The sample is tiny.** The data source caps history at 5 years; after the
  holdout carve-out and 252-day warm-up, evaluation rests on 457 test-fold
  rows ≈ **46 effective observations** (10-day overlapping labels). Every
  number below has confidence intervals wide enough to drive a truck through.
- **Few stress episodes.** The dev window contains essentially one bear
  market (2022) and one vol spike (2024-08). Anything "learned" may be a
  description of 2022, not of volatility.
- **Regime shift between folds.** Early training folds are dominated by the
  2022 bear; later test folds are calm. The full-feature logistic model
  scored AUC 0.37 — *anti*-skill — which is what fitting one regime and
  testing on another looks like. The restricted models' better numbers may
  be the same disease with milder symptoms.
- **Selection pressure.** The reported model (`gbm_small4`) was picked
  because it looked best among 4 variants; the threshold was picked from a
  scan. M = 8 logged runs feed the Bonferroni adjustment, but no adjustment
  fully repairs a garden of forking paths on 46 effective observations.
- **Cost-model realism.** Bleed (2 bp/day) and switch (10 bp) are round
  assumptions frozen before modeling; real overlay costs vary with the vol
  level precisely when the hedge is on. NPS could be systematically
  optimistic in high-vol periods.
- **Proxy drift.** HYG/LQD stands in for credit OAS; ETF prices embed
  duration and flows. VIX3M ETF-era conventions changed over time at CBOE.

## 2. What would falsify this

- **The frozen holdout evaluation** (docs/freeze.md): if the frozen model's
  AUC advantage over persistence disappears or reverses on 2025-02→2026-08
  data, the model is dead. The plan, threshold, and bands are pre-registered;
  one run, no do-overs.
- Shuffled-label AUC leaving the 0.5 band, or staleness ceasing to hurt the
  signal features (leakage suite runs in CI on every push) — either would
  mean the harness itself is broken and ALL results here are void.
- Longer history (a source without the 5-year cap): if the walk-forward
  advantage of `gbm_small4` over persistence does not replicate across
  2008/2011/2015/2018/2020 stress regimes, the current numbers were
  sample luck.
- Net accounting under a doubled bleed assumption (4 bp/day): the operating
  point's NPS goes decisively negative ⇒ the signal has no deployable value
  even if statistically real.

## 3. Performance (net, effective-N-adjusted, multiple-testing-adjusted)

Headline: **the project's success criterion is NOT met.** No model beats all
three baselines net, and no strategy — model or baseline — delivered
positive net protection over the evaluation window. The honest summary is a
negative result, as the charter contemplates.

Walk-forward test folds (457 rows, effective n ≈ 46, base rate 21.9%):

| strategy | AUC [90% CI] | Brier | NPS net (ann.) | hedge-on % | median lead (d) |
|---|---|---|---|---|---|
| persistence (baseline 1) | 0.596 [0.510, 0.673] | 0.238 | −0.0131 | 18% | 2.0 |
| term structure (baseline 2) | 0.532 [0.498, 0.570] | 0.177 | −0.0065 | 2% | n/a (fires rarely) |
| trailing pctl (baseline 3) | 0.654 [0.559, 0.745] | 0.256 | −0.0117 | 26% | 5.5 |
| logistic, 19 features | 0.370 [0.228, 0.527] | 0.286 | −0.0021 | 23% | 4.5 |
| gbm, 19 features | 0.505 [0.351, 0.651] | 0.217 | −0.0078 | 10% | n/a |
| logistic, 4 features | 0.540 [0.381, 0.690] | — | −0.0053 | — | — |
| **gbm, 4 features (frozen)** | **0.673 [0.530, 0.790]** | 0.217 | **+0.0022 @ thr 0.20** | 20% | — |

- **Skill above baselines:** `gbm_small4` beats persistence by +0.077 AUC
  points, p_raw = 0.204 (one-sided block bootstrap). **Bonferroni-adjusted
  over M = 8 logged runs: p_adj = 1.0.** Under the pre-registered rule
  (claim skill only if p_adj < 0.10): **no skill is claimed.** It does not
  clearly beat baseline 3 at all (0.673 vs 0.654, CIs almost coincident).
- **Net:** the frozen operating point's +22 bp/yr NPS is one positive cell
  in a threshold grid whose neighbors are negative; treated as noise. Every
  baseline is net-negative; "never hedge" (NPS = 0) beat everything else
  net over this mostly-calm window.
- **Effective sample size:** 457 raw test rows ⇒ ~46 effective. Reported
  everywhere; CIs use the circular block bootstrap (block = 20 days).
- **Run count / adjustment:** M = 8 (all runs, including failures and the
  threshold scan), Bonferroni per docs/multiple_testing.md.
- **Calibration (frozen model, 3 bins):** monotone but imperfect —
  predicted 0.03 / observed 0.16, predicted 0.22 / observed 0.20,
  predicted 0.32 / observed 0.48. Lows are overconfident.
- **Lead time:** baselines fire 2–5.5 trading days before regime onsets
  they catch; the frozen model's operating point catches 37% of onsets
  (misses 63%) — the expensive error type dominates.
- **Asymmetric costs, separately:** at the frozen threshold the model is
  unhedged into 63% of regime onsets (false negatives — the expensive
  error) and spends 13% of calm days paying bleed (false positives — the
  bounded error). Any deployment decision must weigh the former.

Raw accuracy appears nowhere above, by charter.

## What was actually learned

- Volatility persistence remains the bar: a one-line rule (trailing RV
  percentile) matched or beat every fitted model at a fraction of the
  complexity.
- The vol term-structure *level* (VIX/VIX3M − 1 as a continuous score, AUC
  0.716 on all dev rows) looked stronger than any model — an observation
  made mid-analysis, therefore a **hypothesis for future data, not a
  result**. It is noted here precisely so it can be tested honestly (on the
  holdout or longer history) instead of being quietly promoted.
- With ~46 effective observations, this dataset cannot distinguish a
  genuinely skillful regime model from luck. The binding constraint is
  data depth, not model capacity.

## Holdout evaluation (run once, 2026-08-13, per docs/freeze.md)

Window: 2025-02-13 → 2026-07-29 (365 rows, **effective n ≈ 37**, base rate
28.2%). Frozen model, frozen threshold, frozen bands; four experiment
entries logged; M stood at 16 at adjustment time.

| strategy | AUC [90% CI] | Brier | NPS net (ann.) | hedge-on % |
|---|---|---|---|---|
| **gbm_small4 (frozen)** | **0.744 [0.612, 0.853]** | 0.193 | +0.0465 | **100%** |
| persistence | 0.675 [0.542, 0.789] | 0.255 | +0.0095 | 26% |
| term structure | 0.613 [0.527, 0.683] | 0.238 | −0.0111 | 11% |
| trailing pctl | 0.630 [0.493, 0.746] | 0.345 | +0.0063 | 39% |

**Verdict under the pre-registered rule: NO SKILL CLAIMED.**
AUC advantage over persistence +0.068, p_raw = 0.0845 one-sided; Bonferroni
over M = 16 logged runs ⇒ p_adj = 1.0, far above the 0.10 rule fixed in
docs/multiple_testing.md before any model existed.

**The operating point failed outright.** Every holdout probability exceeded
the frozen 0.20 threshold: the deployed signal degenerates to "always
hedged" (recall 1.0 because it can't miss; precision = the base rate; bleed
paid on 72% of days). Its apparently-best NPS (+4.65%/yr) is exactly what a
constant always-hedge policy would have earned over a window containing the
spring-2025 drawdown — a constant is not a model, so this number is
attributed to the period, not to skill. The probability *ranking* (AUC
0.744) is genuinely encouraging but, at 37 effective observations and after
the adjustment the charter requires, indistinguishable from luck. The
calibration shift that broke the threshold (all probabilities elevated
out-of-sample) is itself evidence the Platt layer fitted on 2022-era dev
data did not transfer.

**Final project verdict: the success criterion is not met.** The model does
not demonstrably beat all baselines net of costs with the required
statistical support. The harness, the leakage discipline, and this negative
result are the deliverables.

## Next steps (require human decisions)

1. Secure a data source with 20+ years of daily history and re-run Phases
   1–5 unchanged (the harness is source-agnostic); the 5-year cap is the
   main reason this report can't say anything stronger. The holdout AUC
   ranking (0.744) and the vix_slope observation are the two hypotheses
   worth carrying into that re-run — as hypotheses, pre-registered.
2. If any deployment is contemplated despite "no skill claimed," the honest
   comparison is against a static always-hedged overlay, which the frozen
   operating point replicated at higher complexity.
