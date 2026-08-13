# Multiple-testing adjustment — chosen 2026-08-13, before any model results

Every run appended to `experiments.jsonl` counts — failures, dead ends,
hyperparameter trials, baselines. Let M be that total at report time.

## Method (frozen now)

The headline claim is "the model's AUC exceeds the persistence baseline's".
The adjustment: the reported one-sided p-value for that comparison is
**Bonferroni-corrected by M** (`p_adj = min(1, M * p_raw)`), with `p_raw`
from the block-bootstrap distribution of the AUC difference
(`src/regime/stats.py`, same block length and seed policy as every other CI).

Bonferroni over the full run ledger is deliberately the most conservative
reasonable choice: with an effective sample in the tens, a flattering
adjustment (or one that requires modeling the dependence between runs, which
we cannot verify) would be exactly the kind of self-serving flexibility the
charter bans. If the result cannot survive Bonferroni, it is not reported as
skill.

The report states: M, p_raw, p_adj, and the decision rule "skill is claimed
only if p_adj < 0.10", fixed here before any model result existed.
