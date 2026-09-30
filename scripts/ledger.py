"""Print the experiment ledger: M (the multiple-testing denominator) and one
line per run. Read-only. `python scripts/ledger.py [--json]`."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from regime import experiment_log  # noqa: E402

runs = experiment_log.read_runs()
if "--json" in sys.argv:
    print(json.dumps([r.__dict__ for r in runs], indent=1, default=str)); raise SystemExit
print(f"M = {len(runs)}")
for i, r in enumerate(runs, 1):
    c = r.config
    what = c.get("strategy", "?") + (f" [{c['panel']}]" if "panel" in c else "") + (f" ({c['universe']})" if "universe" in c else "")
    key = {k: r.metrics[k] for k in ("auc", "nps_net_ann", "p_raw_vs_persistence") if k in r.metrics}
    print(f"{i:3d} {r.timestamp[:19]} {r.git_sha[:7]} {r.data_snapshot_hash[:8]} {what:48s} {key} {r.notes[:60]}")
