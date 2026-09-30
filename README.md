# Volatility / Drawdown Regime Model

A research harness that estimates whether SPY (and the owner's book) is
entering an elevated-volatility regime over the next 10 trading days, and
maps that to a **hedge-ratio suggestion band**. It never predicts price and
never emits an order.

**Current verdict (2026-09-30):** on 25 years of data no fitted model beats
the one-line persistence rule (AUC 0.716 vs 0.716); the rule's timing is
real (p ≈ 0.004). The operational signal is therefore that rule
(`docs/freeze.md`, v3). No skill is claimed for anything.

| read this for | file |
|---|---|
| the rules everything obeys | `CLAUDE.md` (charter) |
| the honest results, "why this might be wrong" first | `REPORT.md` |
| the long-history replication and its pre-registrations | `docs/long_history_plan.md` |
| how to run it day to day | `docs/OPERATIONS.md` |
| what is frozen and where | `docs/freeze.md`, `docs/cost_model.md`, `docs/multiple_testing.md`, `data/*MANIFEST.json` |

## Quickstart

```bash
pip install -r requirements.txt   # pinned; Python 3.11
pytest -q                          # leakage suite + charter tests + hash verification
python scripts/daily.py            # SPY + book bands for the latest frozen bar; scores closed windows
python scripts/ledger.py           # M, the multiple-testing denominator
```

Every training or evaluation run appends to `experiments.jsonl`; every
prediction appends to `forward_test.jsonl` before its label resolves; every
data file is hash-locked and verified in CI. Modifying a frozen file, using
an unfrozen data hash, or reporting raw accuracy as a headline fails the
test suite.
