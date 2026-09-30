"""One-command operational run (freeze v3). See docs/OPERATIONS.md.

    python scripts/daily.py [--refresh <tag> --source "<how pulled>"]

Steps, each of which refuses loudly rather than producing a stale number:
  1. (optional) freeze a data refresh as new snapshot versions
  2. SPY prediction  — persistence rule operational, model diagnostic
  3. book prediction — persistence rule operational, model diagnostic
  4. score every forward-test prediction whose window has closed
  5. print today's bands and the ledger count
A step that refuses (e.g. already logged today) is reported, not fatal.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(label: str, *cmd: str) -> tuple[int, str]:
    r = subprocess.run([PY, *cmd], cwd=ROOT, capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    if r.returncode == 0:
        status = "ok"
    elif "already logged" in out:
        status = "skipped (already logged for this bar)"
    elif "REFUSED" in out:
        status = "refused: " + out.splitlines()[-1][:120]
    else:
        status = "FAILED: " + out.splitlines()[-1][:160]
    print(f"--- {label}: {status}")
    return r.returncode, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", metavar="TAG")
    ap.add_argument("--source", default="")
    a = ap.parse_args()
    if a.refresh:
        rc, out = run("refresh", "scripts/refresh_data.py", a.refresh, "--source", a.source or "unspecified")
        print(out[-1500:])
        if rc != 0:
            raise SystemExit("refresh refused — nothing else run")
    rc, out = run("SPY prediction", "scripts/predict_today.py"); print(out.splitlines()[-1] if rc else "")
    rc2, out2 = run("book prediction", "scripts/book_pipeline.py", "predict"); print(out2.splitlines()[-1] if rc2 else "")
    rc3, out3 = run("scoring", "scripts/forward_test_score.py"); print(out3.splitlines()[-1] if out3 else "")

    # today's bands: the last entry per (universe, config) in the forward log
    last = {}
    for line in (ROOT / "forward_test.jsonl").read_text().splitlines():
        if line.strip():
            e = json.loads(line); last[(e.get("universe") or "SPY", e["config"])] = e
    print("\n=== latest logged bands (freeze v3 entries only)")
    for (uni, cfg), e in sorted(last.items()):
        if "freeze-v3" not in cfg:
            continue
        extra = f"  beta {e['book_beta63_vs_spy']}  SPY-equiv/100k {e['spy_hedge_notional_per_100k_book']}" if uni == "book" else ""
        print(f"{e['asof']}  {uni:5s}  regime={'ON ' if e['signal'] else 'off'}  band {e['band_lo']:.0%}-{e['band_hi']:.0%}  "
              f"rank {e.get('rv10_rank252', e.get('book_rv10_rank252'))}{extra}")
    from regime import experiment_log  # noqa: E402
    sys.path.insert(0, str(ROOT / "src"))
    print(f"ledger M = {experiment_log.run_count()}")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    main()
