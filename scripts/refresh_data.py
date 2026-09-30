"""Freeze a data refresh as new snapshot versions (forward-test protocol).

    python scripts/refresh_data.py <refresh_tag> --source "<how it was pulled>"

Reads data/raw/<refresh_tag>/*_bars.csv (market panel: symbol,date,close,
high,low) and book_bars_*.csv (book names: symbol,date,close), verifies the
overlap against the frozen values, and writes panel_vN.csv / book_vN.csv
with hashes recorded in the manifests. Frozen files are never modified;
every version's hash stays legal for the experiment log. A series the pull
could not deliver stays NaN and is listed in the manifest — nothing is
forward-filled across the refresh boundary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regime import book, data  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("refresh_tag", help="folder name under data/raw/")
    ap.add_argument("--source", required=True)
    ap.add_argument("--note", default="")
    ap.add_argument("--skip-book", action="store_true")
    args = ap.parse_args()

    out = {"panel": data.refresh_snapshot(args.refresh_tag, source=args.source, note=args.note)}
    if not args.skip_book:
        out["book"] = book.refresh_book_snapshot(args.refresh_tag, source=args.source, note=args.note)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
