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

## Refreshes (forward test)

The frozen 2026-08-13 pull is never modified. Each refresh is a new
versioned snapshot (`panel_vN.csv`, `book_vN.csv`) with its own hash in the
manifest; overlap rows must equal the frozen values (closes exactly; intraday
highs/lows within 0.1% of the close, a vendor extreme-print tolerance, worst
case recorded in the manifest). A series the refresh cannot deliver stays
NaN and is listed under `missing_series_from` — nothing is forward-filled
across a refresh boundary.

| Refresh | Source | Delivered | Not delivered |
|---|---|---|---|
| 2026-09-30 (panel v3, book v2; 33 bars, 2026-08-13 → 2026-09-29) | Robinhood MCP `get_equity_historicals` (ETFs, book names; day bars, RTH, split-adjusted) + `get_index_historicals` (VIX) | SPY, QQQ, IWM, HYG, LQD, VIX; all 12 book names | **VIX3M** — not carried by Robinhood; CBOE blocked by the network policy; IBKR MCP unauthorized in the refreshing session. Every feature using `vix_slope` is undefined from 2026-08-13 until VIX3M is refreshed. |

Robinhood ETF and VIX closes matched the IBKR-sourced frozen values to the
cent on the 2026-08-10..12 overlap, so the two vendors are treated as the
same exchange/CBOE prints; both remain revision-free.

## Long-history panel (replication study, 2026-09-30)

`data/snapshots/panel_long_v1.csv`, hash in `data/LONG_MANIFEST.json`;
construction and repair policy in `src/regime/long_history.py`. Same seven
columns-per-series layout as the frozen panel; the frozen 2021-08-16+
segment is appended verbatim (overlap verified to the cent).

| Series | Long-history source | First print | Notes |
|---|---|---|---|
| SPY, QQQ | Robinhood `get_equity_historicals`, day, RTH, split-adjusted | 2000-01-03 | SPY: 32 closes (2000, 2005, 2008–09) that sat > 1% off the S&P 500 index-implied price replaced by index × local ratio; each listed in the manifest |
| IWM | same | 2000-05-26 | two half-price bars (2005-02-17, 2005-02-24) removed and forward-filled |
| LQD | same | 2002-07-26 | |
| HYG | same | 2007-04-11 | credit proxy unavailable before this; features using it start 2007 |
| VIX | Robinhood `get_index_historicals`, day | 2002-05-10 | Robinhood has no VIX history before 2002-05 |
| VIX3M | frozen segment only | 2021-08-16 → 2026-08-12 | **absent** elsewhere; `vix_slope` and baseline 2 are undefined outside that window |
| SPX (check only) | Robinhood `get_index_historicals` | 2000-01-03 | S&P 500 index; used solely to validate SPY prints, never a feature |

All values remain exchange/CBOE prints, revision-free; the SPY repairs are
documented substitutions of the index the ETF tracks, not revisions.

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
