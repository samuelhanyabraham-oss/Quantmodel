# Cost model — FROZEN 2026-08-13; AMENDED 2026-09-30 (bleed 2 → 4 bp/day)

## Amendment 2026-09-30 (owner-authorized; charter changelog entry)

`C_BLEED` is raised from 2 bp/day to **4 bp/day**. Everything else
(switch friction 10 bp, capture 0.5, the NPS formula) is unchanged.

Why: the long-history study (`docs/long_history_plan.md`, "Harness
self-checks") showed that at 2 bp/day a constant hedge scores +7.6%/yr and
beats every rule and model in every fold — the yardstick was subsidising
being hedged. 4 bp/day is (a) the value the charter's own Section-2
falsifier named in advance, and (b) the middle of the realistic carry range
for rolled ~5%-OTM 1–3-month index puts (~3–4 bp/day). It was NOT chosen to
flip a ranking: at 4 bp a constant hedge still out-scores the rules
(+2.5% vs +1.1%/yr); the rules overtake it only above ~4.8 bp. Every
comparison affected is re-costed in `results/recost_4bp.json` and the
REPORT; the original 2 bp results files are left as the record.

---



One cost accounting, applied identically to the model and to every baseline.
Defined before any model existed; parameters below do not move. If they ever
must change, that is a charter amendment with a changelog entry, and every
affected comparison is re-run.

## Signal cost accounting

For a binary hedge signal `s_t` (1 = hedge on), evaluated over N test days:

- **Hedge bleed:** 2 bp of NAV per day while the signal is on
  (`C_BLEED = 0.0002`). Approximates the daily carry of rolling ~5%-OTM
  1-3 month SPY index puts sized to cut drawdown beta roughly in half,
  averaged across vol environments. Deliberately round; precision here would
  be false.
- **Switch friction:** 10 bp per on/off transition (`C_SWITCH = 0.0010`).
  Spread + slippage on establishing/unwinding the overlay.

## Net Protection Score (NPS)

The single net metric on which "beats the baseline" is judged:

    NPS = mean_t [ s_t * max(0, -r_fwd10_t) * CAPTURE ]
        - C_BLEED * mean_t[s_t]
        - C_SWITCH * mean_t[|s_t - s_{t-1}|]

where `r_fwd10_t` is the SPY return over t+1..t+10 and `CAPTURE = 0.5` is the
fraction of the drawdown the overlay is assumed to offset. Units: NAV
fraction per day; reported annualized (x252). It rewards being hedged into
realized drawdowns and charges bleed and churn. It is an evaluation yardstick
for signal quality net of costs, not a P&L simulation.

Probabilistic outputs are converted to `s_t` with each strategy's own frozen
operating threshold; baselines are already binary.

## What this cost model is not

Not a backtest, not advice, not an order-generation rule. The output contract
(CLAUDE.md) still holds: probabilities to hedge-ratio bands, never trades.
