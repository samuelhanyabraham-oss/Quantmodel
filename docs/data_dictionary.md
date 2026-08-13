# Data dictionary

Source for all series: IBKR MCP `get_price_history`, daily OHLC bars, regular
trading hours, `FIVE_YEARS` period (the API's maximum). Pulled once on
2026-08-13 and frozen (see `data/HOLDOUT_MANIFEST.json` for hashes). All
values are exchange prints of traded instruments or CBOE index closes — none
are subject to post-hoc statistical revision, so the first-print rule is
satisfied by construction for every series below.

| Series | Instrument | Publication lag | Revision behavior | Decision |
|---|---|---|---|---|
| SPY | SPDR S&P 500 ETF (ARCA), close/high/low | none (exchange print at close) | none | use — defines the regime |
| QQQ | Invesco QQQ (NASDAQ) | none | none | use (features only) |
| IWM | iShares Russell 2000 (ARCA) | none | none | use (features only) |
| HYG | iShares HY corporate bond ETF (ARCA) | none | none | use — credit-stress proxy (HYG/LQD) |
| LQD | iShares IG corporate bond ETF (ARCA) | none | none | use — credit-stress proxy denominator |
| VIX | CBOE Volatility Index close | none (disseminated intraday, settled at close) | none | use — term-structure numerator |
| VIX3M | CBOE 3-month Volatility Index close | none | none | use — term-structure denominator |

## Timestamp convention

A feature at bar `t` may use only values printed at or before the close of
`t`. All series above print at the close of `t`, so no additional lag is
applied. Labels are computed from `t+1` forward only.

## Excluded, and why

- **FRED credit spreads (HY/IG OAS), Treasury curve series** — the execution
  environment's network policy blocks FRED, and OAS series are also revised
  (evaluated prices), which would have required first-print (ALFRED) work.
  The HYG/LQD price ratio stands in as a credit-stress proxy: it is a traded
  price, never revised, available same-day. Documented limitation: it embeds
  duration and flow effects that pure OAS does not.
- **VIX9D** — insufficient overlap discipline vs. added value at a 10-day
  horizon with this short a sample; term structure is covered by VIX/VIX3M.

## Known limitations (stated once, loudly)

- **History depth:** the source caps history at 5 years (2021-08 → 2026-08).
  After the 18-month holdout carve-out and the 252-day warm-up for trailing
  percentiles, the development sample is ~3.5 calendar years and contains only
  a handful of distinct stress episodes (2022 bear market, 2024-08 vol spike,
  2025 episodes fall in the holdout). With a 10-day overlapping label the
  effective sample is on the order of tens of independent observations. All
  confidence intervals will be wide; conclusions are correspondingly weak.
  This is the project's single largest limitation.
- ETF closes (SPY vs. the S&P index itself) can differ slightly from index
  levels near the close; consistent across model and baselines, so
  comparisons are unaffected.
