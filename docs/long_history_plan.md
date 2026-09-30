# Long-history replication — plan and pre-registration (2026-09-30)

Owner instruction: "your goal is to research." Charter next step #1:
"secure a data source with 20+ years of daily history and re-run the
harness unchanged." This document is written and committed BEFORE any
result on the long panel exists, so nothing below can be tuned to one.

## L1 — Data (done in this commit)

`data/snapshots/panel_long_v1.csv`, hash in `data/LONG_MANIFEST.json`.
2000-01-03 → 2026-09-29 (6,722 SPY trading days); VIX from 2002-05-10;
HYG from 2007-04-11; VIX3M only 2021-08-16 → 2026-08-12. Source and
repair policy in `regime/long_history.py` and the manifest: 32 SPY closes
repaired against the S&P 500 index (vendor print errors > 1% off the
index-implied price), 2 IWM half-price bars removed. The frozen 2021+
segment is appended verbatim and verified on the overlap.

## L2 — Baselines on the long panel (this commit, results below the line)

Purged walk-forward (10-day embargo + purge, 10 folds ≈ 2.5 years each),
baselines 1 and 3 (persistence; trailing percentile), pooled over test
folds and per fold, plus NPS inside named stress windows. Baseline 2
(term structure) needs VIX3M and is reported only on the 2021+ segment.
Effective N, block-bootstrap 90% CIs, frozen cost model. Logged.

## L3 — One pre-registered model check (NOT yet run at commit time)

**Question:** does the frozen freeze-v2 *procedure* — minus the one
feature the long history cannot supply — beat persistence net across
2002–2026, or was the 2021–26 result sample luck?

**Exact spec (no choices left open):**
- Panel: `panel_long_v1` (hash above). Rows: SPY and VIX both present,
  label defined → starts ~2003-05 (252-day warm-up after VIX's first print).
- Features: `[rv10, vix, rv_ratio_10_63]` — freeze v2's set with
  `vix_slope` dropped because VIX3M does not exist before 2021. Nothing added.
- Model: logistic, C = 0.1, balanced class weights, StandardScaler fit on
  each training fold, Platt calibration on the chronological tail 25% of
  each training fold, seed 20260813 (`models.make_model("logistic")`,
  unchanged).
- Validation: `purged_walk_forward(n, n_folds=10)`, embargo 10, purge 10.
- Operating point: adaptive rank — signal on when the probability exceeds
  the 70th percentile of its trailing 126 values (min 60). Bands per
  `hedge_band_from_rank`. Unchanged from freeze v2.
- Comparison: AUC vs persistence on pooled test rows, one-sided block
  bootstrap p; NPS net vs persistence and vs trailing percentile; per-fold
  and per-stress-window NPS. Bonferroni over M = the `experiments.jsonl`
  count at run time. Decision rule unchanged: skill claimed only if
  p_adj < 0.10 AND net NPS exceeds both computable baselines.
- **One run.** No second feature set, no second model, no threshold scan.
  If it fails, the report says the 2021–26 result did not replicate.

Anything else run on the long panel after this is a new, separately
pre-registered experiment and counts toward M like everything else.

---

## L2 results (run 2026-09-30, `results/long_baselines.json`, M = 35 after these runs)

6,451 labelled rows 2001-01 → 2026-09, 10 purged folds, **effective N ≈ 630**
(vs ≈ 46 in the original study). Base rate 25.7%.

| strategy | AUC [90% CI] | Brier | NPS net (ann.) | hedge-on | FN rate | onsets missed |
|---|---|---|---|---|---|---|
| persistence | 0.714 [0.684, 0.744] | 0.212 | +2.29% | 24% | 0.44 | 111 / 136 |
| trailing pctl | 0.710 [0.682, 0.739] | 0.284 | +2.79% | 38% | 0.30 | 84 / 136 |
| term structure (2021+ only) | 0.562 [0.532, 0.593] | 0.251 | −0.87% | 5% | 0.86 | 32 / 32 |
| **always hedged** | — | — | **+7.56%** | 100% | 0 | 0 |
| never hedged | — | — | 0 | 0% | 1 | all |

What this says, bluntly:

1. **The baselines are real and stable.** Persistence's AUC of 0.71 on 630
   effective observations is the first number in this project with a tight
   CI. Over 25 years both one-line rules are net-positive (they were
   net-negative on the mostly calm 2021–24 window). Every "model" result
   in REPORT.md must now be read against 0.71, not 0.60.
2. **The frozen cost model is dominated by a constant.** Under
   `docs/cost_model.md`, hedging every day scores +7.6%/yr and beats both
   rules in every fold and every stress window. NPS credits half of every
   negative forward 10-day return while charging 2 bp/day; over any long
   sample of SPY that is a net subsidy to being hedged. NPS therefore
   rewards hedge-on fraction as much as timing. The cost model is frozen
   (changing it is a charter amendment); this is recorded as a Section-1
   "why this might be wrong" item, and a **supplementary** diagnostic is
   proposed, not adopted: *timing value* = NPS(signal) − hedge_on ×
   NPS(always). Persistence: +0.48%/yr; trailing pctl: −0.08%/yr. I.e. the
   rules add almost nothing beyond a static hedge of the same average size.
3. **Lead time is structurally absent.** Persistence fires after realized
   vol has already risen; it misses 82% of regime onsets under the
   charter's lead-time definition and catches the rest with a median 8-day
   run-up. Any model must be judged on whether it *leads*.
4. **Stress windows:** both rules were on and net-positive in every named
   episode (GFC +32%/+45%, COVID +49%, 2022 +11%/+16%), but always-hedged
   beat them in each. Outside the ten windows (74% of days, base rate 18%)
   both rules are slightly net-negative.

L3 is run next, once, exactly as pre-registered above.

## L3 result (run once, 2026-09-30, `results/long_model_check.json`, M = 36)

6,123 rows 2002-05 → 2026-09, 10 folds, **effective N ≈ 597**.

| strategy | AUC [90% CI] | Brier | NPS net | hedge-on | FN rate | timing value |
|---|---|---|---|---|---|---|
| logistic [rv10, vix, rv_ratio_10_63], adaptive rank | 0.716 [0.678, 0.751] | 0.171 | +1.39% | 31% | 0.49 | −0.68% |
| persistence | 0.716 [0.684, 0.746] | 0.212 | +2.08% | 24% | 0.44 | +0.46% |
| trailing pctl | 0.714 [0.684, 0.742] | 0.284 | +2.44% | 38% | 0.30 | −0.10% |
| always hedged | — | — | +6.76% | 100% | — | 0 |

AUC advantage over persistence: **+0.0006**, p_raw = 0.49, p_adj = 1.0.
Net NPS below both baselines. **Verdict under the pre-registered rule: NO
SKILL CLAIMED.** The model is the persistence rule restated — the exact
failure mode the charter's baseline #1 was written to expose. Per fold the
model's AUC swings from 0.49 (2003–05) to 0.84 (2022–24) while persistence
stays 0.55–0.81; the 2021–26 window that looked encouraging in the
original study is one of the model's best folds and does not generalize.
In the ten stress windows the model was hedged less often than persistence
in the two worst (GFC, 2022) and beat it only in the two 2018 episodes.

Consequence for the project: with 12× the effective sample of the
original study, the answer is unchanged and now statistically tight. The
only thing the long history adds beyond "no skill" is the (better)
calibration the Brier score shows, which has no cash value under the
frozen cost model. The 2021–26 in-sample result (`REPORT.md`) is now
classified as sample luck.

## L4 — pre-registered (written 2026-09-30 after L3, before any L4 number exists)

**Question.** L3 used three features and equalled persistence. Do the
charter's remaining boring features carry information beyond trailing
realized vol at effective N ≈ 600 — and does anything *lead*?

**Exact spec.**
- Panel `panel_long_v1`. Feature set = every `features.FEATURE_NAMES`
  entry computable without VIX3M or HYG (so the sample starts 2003, not
  2008): `rv10, rv21, rv63, rv10_chg5, rv_ratio_10_63, volofvol21, vix,
  vix_chg5, vrp, qqq_rv10, iwm_rv10, dd_from_peak63, ret21, ret5, skew63,
  range5` (16 features; `vix_slope`, `credit_ratio_chg21`,
  `credit_ratio_chg5` excluded). Nothing added, no selection.
- Two runs, both via `models.walk_forward_probs` unchanged: `logistic`
  (C=0.1, balanced) and `gbm` (depth 2, 100 iters, lr 0.05, leaf 20, l2 1),
  seed 20260813, 10 purged folds, Platt tail-25% calibration per fold.
- Operating point: adaptive rank q70 / w126 (min 60), frozen bands.
- Reported: charter metric set + timing test (new, supplementary) + per
  fold + per stress window + lead time vs persistence.
- Decision rule (unchanged): skill only if p_adj < 0.10 vs persistence
  (Bonferroni over M at run time, M = 39 after these two are logged) AND
  net NPS > both computable baselines. Timing p and lead time are
  descriptive. Whatever happens, no third feature set, no tuning.

## L4 result (run once, 2026-09-30, `results/long_l4.json`, M = 38)

6,118 rows 2002-05 → 2026-09, effective N ≈ 597. Timing test = supplementary
circular-shift null under the frozen cost model (`stats.timing_value_test`).

| strategy | AUC [90% CI] | net NPS | on | timing excess / p | lead (median) | onsets missed |
|---|---|---|---|---|---|---|
| persistence | 0.716 [0.683, 0.745] | +2.08% | 24% | +1.63% / **0.004** | 8 d | 102 / 125 |
| trailing pctl | 0.714 [0.684, 0.741] | +2.45% | 38% | +1.64% / **0.005** | 7 d | 78 / 125 |
| logistic, 16 features | 0.721 [0.680, 0.760] | +0.80% | 33% | +0.82% / 0.034 | 4 d | 86 / 125 |
| gbm, 16 features | 0.716 [0.678, 0.752] | +0.58% | 35% | +0.74% / 0.062 | 6 d | 86 / 125 |

Logistic ΔAUC vs persistence +0.0055 (p_raw 0.40); gbm −0.0002 (p_raw 0.48);
p_adj = 1.0 for both; both net below both baselines. **NO SKILL CLAIMED,
twice.** Thirteen extra boring features add nothing a one-line rule
doesn't already carry; the fitted models are *worse* net because they
hedge more often for the same ranking.

Two things the new timing test settles:

1. **The one-line rules have real timing value.** Circularly shifting
   persistence's signal (same on-fraction, same run lengths) costs
   1.6%/yr with p ≈ 0.004. So "always-hedged beats it" (true, +6.8%/yr)
   does NOT mean the rule is noise: under this cost model a static hedge
   is subsidized, and the rule's timing adds a real increment on top of
   its size. The two facts coexist; the report now states both.
2. **The models' timing value is weaker than the rules'** (+0.8%/yr, p
   0.03–0.06), i.e. fitting destroyed timing information the rule had.
   Consistent with the models being persistence plus noise.

Lead time: unchanged — nothing here leads regime onsets; the models' median
lead (4–6 d) is shorter than the rules' (7–8 d) because they fire later
and less decisively.

Closed questions after L2–L4 (all pre-registered, M = 38): on 25 years the
charter's feature set, with linear or small-tree models, does not beat
persistence; the project's negative result stands with tight intervals.
