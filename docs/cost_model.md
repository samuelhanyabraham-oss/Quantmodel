# Cost model — FROZEN 2026-08-13, before any model saw any test fold

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
