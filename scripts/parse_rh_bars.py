"""Convert saved Robinhood bar pulls (equity or index JSON) into the long
CSV format the refresh/snapshot code reads: symbol,date,close,high,low.
Interpolated bars are dropped. Usage: _parse_rh_bars.py <out.csv> <json>..."""
import json, sys
out, srcs = sys.argv[1], sys.argv[2:]
rows = []
for src in srcs:
    d = json.load(open(src))
    for r in d["data"]["results"]:
        sym = r["symbol"]
        for b in r["bars"]:
            if b.get("interpolated"):
                continue
            c = b.get("close_price", b.get("close_value"))
            h = b.get("high_price", b.get("high_value"))
            l = b.get("low_price", b.get("low_value"))
            rows.append((sym, b["begins_at"][:10], float(c), float(h), float(l)))
rows.sort()
with open(out, "w") as f:
    f.write("symbol,date,close,high,low\n")
    for r in rows:
        f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]}\n")
print(out, len(rows), "rows", rows[0][1] if rows else None, rows[-1][1] if rows else None)
