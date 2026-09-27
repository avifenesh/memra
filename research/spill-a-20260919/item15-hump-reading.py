#!/usr/bin/env python3
"""WP-A day 43 (DAY43.md section 1, OWED item 15): day 38 section 13e's hump reader, the program unchanged, over the
long-entry demote arm. Input: ROOT/bNN-<arm>/demote-long/receipt.json (stall_cell.py's demote-long arm, --n 8: 16 demote
runs per boot). Per boot: the tenant's pre-fire inter-token latency median
of every demote run (its first 22 ITLs), in run order; BASE = the median of demote runs 1 to 3; HUMP = the largest of
demote runs 4 to 16 minus BASE. Per arm: the median of its boots' HUMP; an arm HUMPS if that median exceeds 0.15 ms
(section 12a: x1 about +0.6 ms, x2 about +0.03). Printed per boot with the series, then per arm. A boot with fewer than
16 demote runs is INCOMPLETE and not read. Usage: item15-hump-reading.py ROOT"""
import glob, json, os, statistics as st, sys

root = sys.argv[1]
by = {}
for f in sorted(glob.glob(os.path.join(root, "b*-x*", "demote-long", "receipt.json"))):
    boot = f.split(os.sep)[-3]
    arm = boot.split("-")[1]
    j = json.load(open(f))
    rs = [r for r in j["runs"] if r.get("arm") == "demote-long" and r.get("itl_ms")]
    itl = [st.median(r["itl_ms"][:22]) for r in rs]
    if len(itl) < 16:
        print(f"HUMP boot={boot} runs={len(itl)} -> INCOMPLETE")
        continue
    base = st.median(itl[:3])
    hump = max(itl[3:16]) - base
    by.setdefault(arm, []).append(hump)
    print(f"HUMP boot={boot} base={base:.3f} hump={hump:+.3f} itl={[round(x, 2) for x in itl]}")
for arm in sorted(by):
    m = st.median(by[arm])
    print(f"HUMP arm={arm} boots={len(by[arm])} median-hump={m:+.3f} humps={m > 0.15}")
