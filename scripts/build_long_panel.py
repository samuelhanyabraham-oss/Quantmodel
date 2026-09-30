"""Freeze the long-history panel (one-time). See regime/long_history.py."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from regime import long_history  # noqa: E402

m = long_history.build_long_panel()
print(json.dumps({k: v for k, v in m.items() if k != "repairs"}, indent=2))
print("SPY repairs:", [(r["date"], r["vendor_close"], r["repaired_close"]) for r in m["repairs"]["spy_index_check"]])
print("spike repairs:", m["repairs"]["feature_etf_spikes"])
