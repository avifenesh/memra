#!/usr/bin/env python3
"""Step rates and the memory book's margin from a deepseek-v4.1 server log ([dsv41-step] lines).
usage: rates.py SERVER_LOG OUT_JSON"""
import json
import re
import statistics
import sys

PAT = re.compile(
    r"\[dsv41-step\] calls=(\d+) prefill=(\d+) step_s=([\d.]+) transient_peak=(\d+) "
    r"predicted_live_need=(\d+) cache_slots=\[([\d, ]+)\] hits=(\d+) misses=(\d+) h2d_bytes=(\d+)"
)
rows = []
for line in open(sys.argv[1]):
    m = PAT.search(line)
    if m:
        c, p, s, tp, pn, slots, h, mi, hb = m.groups()
        rows.append(dict(calls=int(c), prefill=int(p), step_s=float(s), transient=int(tp), predicted=int(pn),
                         slots=[int(x) for x in slots.split(",")], hits=int(h), misses=int(mi), h2d=int(hb)))
out = {"steps": len(rows), "decode": {}, "prefill": []}
for c in sorted({r["calls"] for r in rows if r["prefill"] == 0}):
    v = sorted(r["step_s"] for r in rows if r["prefill"] == 0 and r["calls"] == c)
    med = statistics.median(v)
    out["decode"][c] = {"n": len(v), "median_s": round(med, 4), "p90_s": round(v[int(0.9 * (len(v) - 1))], 4),
                        "tokens_per_s": round(c / med, 2)}
prev = None
for r in rows:
    if r["prefill"] > 0:
        out["prefill"].append({"tokens": r["prefill"], "calls": r["calls"], "step_s": r["step_s"],
                               "transient_bytes": r["transient"], "predicted_live_need": r["predicted"],
                               "misses_in_step": r["misses"] - (prev["misses"] if prev else 0)})
    prev = r
out["transient_over_predicted_max"] = max(r["transient"] / r["predicted"] for r in rows) if rows else None
if rows:
    out["cache"] = {"slots": rows[-1]["slots"], "hits": rows[-1]["hits"], "misses": rows[-1]["misses"],
                    "hit_rate": round(rows[-1]["hits"] / max(1, rows[-1]["hits"] + rows[-1]["misses"]), 4),
                    "h2d_bytes": rows[-1]["h2d"]}
    dec = [(a, b) for a, b in zip(rows, rows[1:]) if b["prefill"] == 0 and b["calls"] == 1]
    if dec:
        per = sorted(b["misses"] - a["misses"] for a, b in dec)
        out["decode_misses_per_token"] = {"median": statistics.median(per), "p90": per[int(0.9 * (len(per) - 1))]}
json.dump(out, open(sys.argv[2], "w"), indent=1)
print(json.dumps(out["decode"]), out.get("decode_misses_per_token"), out["transient_over_predicted_max"])
for p in out["prefill"]:
    if p["tokens"] > 1000:
        print("long prefill", p)
