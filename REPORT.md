# REPORT — Volatility/Drawdown Regime Model (final: holdout evaluated; long-history replication failed)

Status: Phases 0–6 complete. The holdout was unlocked by the owner on
2026-08-13 (data/HOLDOUT_UNLOCK.json) and evaluated exactly once, per the
pre-registered plan in docs/freeze.md. Sections 1–3 below are the
walk-forward analysis (written before the holdout was read and left
unedited); the holdout section follows them.

**Reader's summary as of 2026-09-30** (details in the dated sections at
the end and in `docs/long_history_plan.md`): the project has now been run
on 25 years of data (effective N ≈ 200–600 depending on the rule used).
Three pre-registered model checks equal the persistence baseline on AUC
(0.716 vs 0.716) and score below it net. No skill is claimed, and the
2021–26 in-sample result is classified as sample luck. Under the frozen
cost model a constant hedge beats every signal, breaking even only at
5 bp/day of bleed; the one-line rules' *timing* is nonetheless real
(p ≈ 0.004 against a shift null). Nothing leads a regime onset. The
forward test continues for the book; the SPY leg is paused on a missing
VIX3M feed. M = 39 logged runs.

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

## Post-dissolution addendum (2026-08-13)

The owner dissolved the holdout after its single evaluation ("get rid of
the holdout then i want this to work"). Consequences, recorded plainly:

- The holdout verdict above remains the last clean out-of-sample result.
  Nothing after the dissolution can claim out-of-sample validity; the full
  five years are now development data (snapshot v2, hash-locked).
- **Re-running the walk-forward on the full sample — with the 2025–26
  stress episodes now inside the test folds — did not change the answer.**
  Best model variant (logistic, 4 features, adaptive rank threshold): AUC
  0.638 [0.518, 0.750], NPS +0.64%/yr, vs persistence 0.642 and trailing
  percentile 0.650. p_raw vs persistence 0.52; M = 23. The holdout's
  encouraging AUC (0.744) did not replicate as a general property — it was
  one period, not skill.
- The genuine defect the holdout exposed is fixed: the operating point is
  now the probability's trailing-quantile RANK (70th pct, 126-day window),
  immune to level shifts in calibration; bands are rank-based
  (freeze v2, `docs/freeze.md`).
- **"Working" now means:** `scripts/predict_today.py` emits a daily
  probability, rank, and hedge band from the frozen config, and appends
  every prediction to `forward_test.jsonl` before its label resolves. That
  accumulating record is the only remaining path to an honest skill claim.
  Until it delivers one, the correct reading of this system is: a
  disciplined hedging heuristic roughly on par with a one-line volatility
  rule — not a validated model.

## Book extension (2026-08-13): the owner's actual portfolio

The machinery was pointed at the owner's real holdings (12 names, Robinhood
margin + IRA, current-weight backcast; NBIS 25% / SHAZ 19% / IREN 12% top
weights — an AI-infrastructure/bitcoin-miner cluster). Data notes: two
symbols carried predecessor-listing garbage (SHAZ: 292 fake ±160% days),
removed by a longest-clean-suffix guard; book snapshot hash-locked in
`data/BOOK_MANIFEST.json`.

The book is NOT the SPY tape:
- annualized vol ≈ 65% (vs SPY ~13%); worst single day −35%; max drawdown
  in sample −85%; **currently ~24% below its sample peak**
- 63-day beta to SPY ≈ 4.4 — an SPY-based hedge must be sized on beta-
  equivalent notional, not book value

Book walk-forward (M=27 after these runs): the fitted model is anti-skill
(AUC 0.377 [0.28, 0.47]); **book-persistence** (trailing 10d RV above its
own 1y 75th pct) scores AUC 0.639 [0.56, 0.71] *(corrected 2026-09-30 to
0.612 [0.53, 0.69], see Erratum)* and — unlike every SPY
strategy — is decisively net-positive: **NPS +18.2%/yr**, because this
book's 10-day drawdowns dwarf hedge bleed. The operational signal in
`scripts/book_pipeline.py predict` is therefore the persistence rule; the
model is logged as a diagnostic only.

Current reading (2026-08-12): book trailing RV 135% > threshold 87% —
**the book is in an elevated-vol regime now** even though SPY is calm;
suggested hedge band 25–50% of book value, ≈ $167k SPY-equivalent short
notional per $100k of book at the band midpoint (beta 4.4). Same honesty
clause as everywhere: the persistence rule is in-sample-validated only;
its forward record accumulates in forward_test.jsonl.

## Forward test — first refresh (2026-09-30)

Operational continuation only; no modeling change, no new variant, no
threshold touched. Data refreshed through 2026-09-29 (33 trading days) as
new frozen versions (panel v3, book v2), Robinhood-sourced because IBKR was
unauthorized in this session; closes matched the frozen IBKR values to the
cent on the overlap. Two runs logged (scoring, book prediction): M = 31.

**Resolved predictions (the whole forward record so far):**

| asof | universe | signal / band | realized label | fwd RV vs thr | fwd 10d max DD | persistence said |
|---|---|---|---|---|---|---|
| 2026-08-12 | SPY (freeze v2 model) | 0 / 0–10% | 0 | 8% vs 14% | −2.0% | 0 |
| 2026-08-12 | book (persistence rule, operational) | 1 / 25–50% | 0 | 61% vs 87% | **−17.0%** | 1 |
| 2026-08-12 | book (logistic, diagnostic only) | 0 / 0–10% | 0 | 61% vs 87% | −17.0% | 1 |

Three rows ≈ 0.3 effective observations. No test attempted (the scorer
refuses below 60 resolved rows per universe); no claim of any kind.

**What the one book row exposes, honestly.** By the charter label the
operational hedge signal was a *false positive*: the book's forward 10-day
RV (61%) stayed under its 1-year 75th-percentile threshold (87%). Over the
same 10 days the book fell **17%** from the 08-12 close, and under the
frozen cost model being hedged was net-positive (NPS +0.45 ann. vs +1.16
for the always-on persistence rule). On a 65%-vol book the vol-percentile
label sets so high a bar that a −17% fortnight counts as "calm". That is a
property of the target definition applied to this universe, not a finding
about the signal — and the target definition is not changeable without the
owner's say-so. It is flagged here as an open question (below), not acted on.

**Current reading (2026-09-29):** book trailing 10-day RV 48% vs threshold
88% — **out of regime**; suggested band 0–10%; 63-day beta to SPY 4.65 (≈
$23k SPY-equivalent short notional per $100k of book at the band midpoint).
The book index is −7% since 2026-08-12. Model diagnostic unavailable
(needs VIX3M). The 2026-09-29 prediction is logged; its label resolves after
2026-10-13.

**SPY forward test: paused, not failed.** The frozen model's `vix_slope`
feature needs VIX3M, which no reachable source delivered after 2026-08-12
(Robinhood does not carry it; CBOE is blocked by the network policy; IBKR
needs authorization). The prediction script now refuses loudly on a bar
where a frozen feature is missing, rather than re-emitting the last
computable bar. The record of the two duplicate 2026-08-12 entries stays
(append-only); the scorer de-duplicates and the append is now idempotent.

**Weights are unchanged** from the 2026-08-13 positions snapshot. If the
holdings have changed, the book index is a backcast of a stale composition;
re-weighting is a logged decision, not something a refresh does silently.

### Addendum (2026-09-30, later): inputs to the open decisions

**Book label vs. large drawdowns (diagnostic, frozen composition-1 data,
in-sample, M = 32).** `results/book_label_diagnostic.json`. The single
−17% false positive above is not the norm: the charter label reads "1"
ahead of 72% of the book's ≤ −15% ten-day drawdowns and 83% of its ≤ −20%
ones (base rate 33%). The **operational persistence rule** is the weaker
link: it is on before only 49% of ≤ −15% drawdowns and 42% of ≤ −20%
ones, while being on 32% of all days. On SPY the same rule catches 51% of
≤ −5% drawdowns. So the honest reading of decision 2 is: the label is
defensible as-is; what is coin-flip on this book is the *rule*, and
nothing in this project has yet beaten that rule with statistical support.

**Positions (decision 3): the 2026-08-13 composition is stale.** Holdings
read from the connected brokerage on 2026-09-30 differ materially: four
names exited, one added, several resized. The active book index is a
backcast of a composition that no longer exists. A second composition was
NOT built — writing the new positions and the new name's history into the
repository was refused by this session's permission classifier as
brokerage-sourced data, so it needs the owner's explicit go-ahead in
session. Until then, book predictions continue on composition 1 and are
labelled as such; treat their hedge sizing as describing the old book.

**VIX3M (decision 1):** seven candidate hosts probed on 2026-09-30, all
denied by the environment's network policy (cdn.cboe.com, www.cboe.com,
query1/query2.finance.yahoo.com, stooq.com, fred.stlouisfed.org,
api.nasdaq.com). Allow-listing `cdn.cboe.com` is the smallest change that
would lift the SPY forward-test pause.

## Long-history replication (2026-09-30): the 5-year result did not replicate

The binding limitation named in Section 1 — five years, one bear market,
≈ 46 effective observations — was lifted: Robinhood's bar API serves daily
history to 2000 (IBKR's cap was the limit). Data, repairs and the
pre-registered plan are in `docs/long_history_plan.md`; the plan was
committed (77f207e) before the model check ran. Both new runs are logged.

**Baselines on 2001–2026 (effective N ≈ 630):** persistence AUC 0.714
[0.684, 0.744]; trailing percentile 0.710. Both net-positive over 25 years
(+2.3%, +2.8%/yr) though slightly net-negative outside the ten stress
windows. This is the bar; it was 0.60 on the short sample only because
that sample was mostly calm.

**The one pre-registered model check (freeze-v2 procedure minus
`vix_slope`, which no long history supplies): NO SKILL CLAIMED.** AUC
0.716 vs persistence 0.716 (Δ = +0.0006, p_raw 0.49, p_adj 1.0 at M = 36);
net NPS +1.4%/yr vs +2.1% (persistence) and +2.4% (trailing percentile).
The model is persistence restated. Its encouraging 2021–26 fold (AUC 0.84)
is one of ten and is now classified as sample luck.

**Two things the long history exposed that the short one hid:**

1. *The frozen cost model is dominated by a constant.* Hedging every day
   scores +6.8%/yr under `docs/cost_model.md` and beats every rule and
   model in every fold and every stress window: NPS credits half of every
   negative 10-day return while charging 2 bp/day, a net subsidy to being
   hedged on any long SPY sample. NPS therefore rewards hedge-on fraction
   at least as much as timing. Changing the cost model is a charter
   amendment and has not been made; a supplementary *timing value*
   (NPS minus hedge-on × always-hedged NPS) is proposed in the plan doc.
   Under it persistence adds +0.5%/yr over a static hedge of the same
   size; the model −0.7%/yr; trailing percentile −0.1%/yr.
2. *Lead time is structurally absent.* Every rule and the model fire after
   realized vol has risen: persistence misses 82% of onsets under the
   charter's lead-time definition, the model 70%. Nothing here leads.

**Second pre-registered check (L4, M = 38): the full VIX3M/HYG-free
feature set (16 features), logistic and small GBM, one run each — NO
SKILL CLAIMED, twice.** AUC 0.721 / 0.716 vs persistence 0.716 (p_raw 0.40
/ 0.48, p_adj 1.0); net NPS +0.8% / +0.6%/yr vs +2.1% (persistence). A new
supplementary timing test (`stats.timing_value_test`, circular-shift
null preserving on-fraction and run lengths) shows the one-line rules'
timing is real (persistence +1.6%/yr over its null, p ≈ 0.004) and the
models' is weaker (+0.8%, p 0.03–0.06): fitting loses timing information
the rule already has. Details: `docs/long_history_plan.md`.

**Harness self-checks (M = 39, `docs/long_history_plan.md`):** label
dependence lasts ~80 trading days (integrated autocorrelation time 33 d),
so the n/10 effective-N rule overstates independence ~3× (≈194, not 645,
on the long panel); the block-bootstrap CIs are only ~12% too narrow. The
harness now reports the autocorrelation-based count alongside. Cost
sensitivity: always-hedged breaks even at 5.0 bp/day and the one-line
rules overtake it only above ~4.8 bp/day; at the charter's 4 bp falsifier
the rules keep ~+1%/yr. The bleed parameter, not the models, decides
whether timing has cash value under this yardstick.

Section 1 and Section 2 stand unchanged. Section 2's falsifier
"if the walk-forward advantage does not replicate across 2008 / 2011 /
2015 / 2018 / 2020 stress regimes, the current numbers were sample luck"
has now been triggered.

## Freeze v3 (2026-09-30): the system as it now operates

Owner instruction: "keep going until it works." `docs/freeze.md` v3. The
operational SPY signal is the charter's persistence rule — the only SPY
signal with tested timing value on 25 years (p ≈ 0.004) and the one every
fitted model restated. Its continuous score, the trailing-252 percentile
rank of rv10, maps to three monotone bands (rank < 0.75 → 0–10%;
0.75–0.90 → 25–50%; ≥ 0.90 → 50–75%); the 0.90 cut is the only new
number and was fixed before any forward result under v3. The freeze-v2
model is computed as a diagnostic whenever VIX3M exists (currently never)
so the forward record can still test it. The book leg uses the same rule
and bands on the book's own rv10. `scripts/daily.py` runs refresh →
predict → predict → score in one command; every step refuses a stale bar
or a duplicate entry.

Readings logged for 2026-09-29 (labels resolve after 2026-10-13):

| leg | rv10 | threshold | rank | regime | band |
|---|---|---|---|---|---|
| SPY | 11.4% | 14.4% | 0.53 | off | 0–10% |
| book (composition 1, stale weights) | 48.0% | 87.9% | 0.11 | off | 0–10% |

What "works" does not mean: no skill is claimed for anything. The rule is
a disciplined, tested heuristic whose net value under the frozen cost
model is +2.1%/yr over 25 years and whose timing beats a same-size static
hedge by +1.6%/yr; a constant hedge still scores higher in absolute NPS
until bleed exceeds ~4.8 bp/day. The book composition is versioned
(`book_pipeline.py build --composition N --positions ...`) and waits on a
positions file from the owner.

## Charter amendments and book composition 2 (2026-09-30, owner-authorized)

The owner approved the four items put to them ("sounds good do that then").
Three were done; the fourth (VIX3M) still needs the network allow-list.

**Amendment 1 — cost model bleed 2 → 4 bp/day** (`docs/cost_model.md`,
charter changelog). Every affected comparison re-derived and re-costed
(`results/recost_4bp.json`; the L3/L4 AUCs reproduced exactly, confirming
determinism). Under the amended yardstick:

| strategy (long panel test rows) | NPS @2 bp | **NPS @4 bp** | timing excess @4 bp / p |
|---|---|---|---|
| always hedged | +7.6% | **+2.5%** | 0 |
| persistence | +2.3% | **+1.1%** | +1.7% / 0.004 |
| trailing pctl | +2.8% | **+0.9%** | +1.7% / 0.006 |
| L3 logistic (3f) | +1.4% | **−0.2%** | +1.0% / 0.009 |
| L4 logistic (16f) | +0.8% | **−0.9%** | +0.8% / 0.03 |
| L4 gbm (16f) | +0.6% | **−1.2%** | +0.7% / 0.06 |

The amendment does not rescue anything: the constant still leads, the
rules stay modestly positive, and every fitted model goes net-negative.
Book composition 1 persistence: +19.4% → +17.6%/yr.

**Amendment 2 — effective N is now autocorrelation-based** (`effective_n`;
the old n/10 rule stays as `effective_n_rule`). Read every earlier
"effective N" in this report as the old rule.

**Book composition 2 built and activated** (`data/book_positions_c2.json`,
`data/BOOK_C2_MANIFEST.json`, hash-locked; composition 1 retained). Current
holdings: SNDK 33%, NBIS 28%, SHAZ 17%, CRWV 9%, STM 6%, APLD 6%, KEEL,
CORZ, TE < 1% each. The new book runs ~59% annualized vol, worst day −41%,
max drawdown −74%, and sits 27% below its sample peak. Walk-forward on it
(M = 58, amended cost): **persistence AUC 0.634 [0.563, 0.700], net NPS
+13.2%/yr, hedge-on 39%, timing +6.8%/yr over its null (p ≈ 0.0005)**;
the fitted logistic model is anti-skill (AUC 0.44) with 168 test rows
unscorable because the first training fold contains no regime-on label.
Effective N ≈ 26 (rule: 84) — read the CI accordingly. Reading logged for
2026-09-29: rv10 48% vs threshold 91%, rank 0.10, **out of regime, band
0–10%**, beta to SPY 4.58 (≈ $23k SPY-equivalent short per $100k of book at
the band midpoint if the band were on).

## Erratum (2026-09-30): baseline warm-up bug, audited

An adversarial review of the code found that `run_experiments.py` and
`book_pipeline.py evaluate` computed the persistence and trailing-percentile
baselines on the *already-truncated* close series, so the 252-day
threshold was undefined for the first 261 evaluation rows and the signal
silently read 0 there. `baselines.py` now returns NaN where its threshold
is undefined, both scripts compute baselines on the full close, and a test
guards it. The holdout script and every long-history run were unaffected
(they already used the full close). Re-scoring the original frozen data
both ways (`results/baseline_bug_audit.json`, logged, M = 41):

| table | strategy | as reported | corrected | rows forced off |
|---|---|---|---|---|
| walk-forward, pre-holdout dev | persistence / trailing pctl | 0.596 / 0.654 | **unchanged** | 0 |
| walk-forward, full sample v2 | persistence / trailing pctl | 0.642 / 0.650 | **unchanged** | 0 |
| book walk-forward | book persistence | AUC 0.639, NPS +18.2% | **AUC 0.612 [0.529, 0.686], NPS +19.0%** | 31 |

On SPY the rows the bug forced off fell in calm 2023 where the rule was off
anyway, so no SPY number changes. On the book the bug *overstated* the
persistence AUC by 0.027 (the forced-off rows happened to be label-0
days); the operational conclusion (persistence net-positive on the book,
fitted model anti-skill) survives, the number does not. The original
results files are left as written; this section is the correction.

Other review findings fixed in the same pass: the book refresh could not
detect a retated/re-adjusted history (now compares every overlapping
close per symbol); the market-panel refresh could pass with zero
comparable SPY rows (now requires per-series overlap and a contiguous
calendar); the forward-test scorer pooled different configs of one
universe (now grouped per config); the timing test's p-value is now
finite-sample corrected and guarded for short samples.

## Next steps (require human decisions)

1. **VIX3M feed** (blocks the SPY forward test): authorize IBKR in a
   session, or allow-list `cdn.cboe.com`, or decide on a proxy — the last
   one changes the frozen feature set and needs an explicit instruction.
2. **Book label definition**: keep the charter's vol-percentile label for
   the book (consistent, but blind to a −17% fortnight), or add the
   drawdown label already produced alongside it as the book's operating
   target. Owner call; nothing changes until made.
3. **Positions**: confirm whether the 2026-08-13 weights still describe
   the book.


4. ~~Secure a data source with 20+ years of daily history~~ — done
   2026-09-30 (`docs/long_history_plan.md`); the replication failed, see
   the section above. Any further modeling on the long panel needs its
   own pre-registration and counts toward M.
5. Let the forward test run: refresh the snapshot periodically (new frozen
   version each time), run `predict_today.py` daily, and evaluate the
   forward record against persistence once it holds a few hundred rows
   (~2 years). No skill claim before then.
6. If any deployment is contemplated meanwhile, the honest comparison is
   against a static always-hedged overlay and the one-line trailing
   percentile rule, both of which matched or beat every model in-sample.
