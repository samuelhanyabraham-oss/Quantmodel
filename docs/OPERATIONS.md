# Operations — running the forward test

Everything below is mechanical. Nothing here tunes, selects, or changes a
frozen configuration; if a step would, stop and read CLAUDE.md.

## One command

    python scripts/daily.py [--refresh <tag> --source "<how pulled>"]

runs the whole cycle below and prints the latest bands. Each step refuses
loudly (stale bar, duplicate entry, restated history) rather than
producing a wrong number; a refusal is reported and the run continues.

## Book compositions (re-weighting)

When holdings change, write a positions file `data/book_positions_cN.json`
(same schema as `data/book_positions.json`: `asof`, `shares`, and a
`last_close` map with the official prior-session closes), put any new
name's daily closes under `data/raw/book/batch*.json` or a refresh
`book_bars_*.csv`, then:

    python scripts/book_pipeline.py build --composition N \
        --refresh-dir data/raw/<tag> --activate "why"

The old composition's snapshots and hashes stay; forward-test entries
record the composition they were predicted on and are scored against it.

## Daily / weekly cycle

1. **Pull new bars** (any reachable source; Robinhood MCP was used on
   2026-09-30, IBKR MCP originally). Save them as long CSVs under
   `data/raw/<refresh_tag>/`:
   - market panel: `*_bars.csv` with columns `symbol,date,close,high,low`
     for SPY, QQQ, IWM, HYG, LQD, VIX, VIX3M
   - book names: `book_bars_*.csv` with `symbol,date,close`
   `scripts/parse_rh_bars.py out.csv <saved pull json>...` converts a saved
   Robinhood pull. Include at least 3 days that overlap the frozen panel.
2. **Freeze the refresh:**
   `python scripts/refresh_data.py <refresh_tag> --source "<how pulled>"`
   Refuses if any overlapping close differs from the frozen value, if a
   book name disappears, or if a version file already exists. A series the
   pull could not deliver stays NaN and is listed under
   `missing_series_from` in `data/HOLDOUT_MANIFEST.json`.
3. **Predict:**
   - `python scripts/predict_today.py` (SPY, freeze v3: persistence rule
     operational, freeze-v2 model as diagnostic when VIX3M exists).
     Refuses to log the same (asof, config) twice.
   - `python scripts/book_pipeline.py predict` (book, persistence rule
     operational, model diagnostic if computable). Same guards.
4. **Score resolved predictions:** `python scripts/forward_test_score.py`
   → `results/forward_test_record.json`. Attempts no skill test below 60
   resolved rows per universe. Never claims skill itself.
5. **Check the ledger:** `python scripts/ledger.py` — M is the
   multiple-testing denominator; every run above appends to it.
6. `pytest -q` must be green before committing (leakage suite, charter
   tests, hash verification of every frozen file).

## Things that need a human, and why

| item | why the scripts refuse to do it |
|---|---|
| VIX3M feed | frozen feature `vix_slope`; substituting a proxy changes the frozen feature set |
| book re-weight (composition 2) | brokerage-sourced data write; composition is a logged decision, not a refresh side-effect |
| cost-model parameters | charter amendment (`docs/cost_model.md`) |
| effective-N headline rule | charter amendment (Validation section); `effective_n_acf` is reported beside `effective_n` until then |
| any new model / feature set | must be pre-registered in writing and committed before it runs (`docs/long_history_plan.md` is the template) |

## Where the frozen state lives

- `data/HOLDOUT_MANIFEST.json` — market panel versions (`active_version`), hashes
- `data/BOOK_MANIFEST.json` — book versions, weights as-of date
- `data/LONG_MANIFEST.json` — long-history panel, repair list
- `docs/freeze.md` — model, operating point, bands (freeze v2)
- `docs/cost_model.md`, `docs/multiple_testing.md` — frozen yardsticks
- `forward_test.jsonl` — append-only prediction record (duplicates are
  ignored on read, never deleted)
- `experiments.jsonl` — append-only run ledger
