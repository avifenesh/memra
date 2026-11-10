#!/usr/bin/env python3
"""tables.py NAME=KC.log ...: markdown tables for the lane README from kernel-check logs' timing rows.
1. per call, the 24/4 shape, one session's 4 rows by depth: seqs and rows ms per device, ratio;
2. the routing grid: rows/seqs per GQA shape and launch shape, one column per device, the floor marked."""
import re, sys
pat = re.compile(r"FA-VERIFY-ROWS time gqa=(\d+) (?:len0=(\d+) T=4|pooled (\d+) sessions x 4 rows depth~(\d+)) sp=(\d+): seqs ([\d.]+) ms \(([\d.]+)-([\d.]+)\), rows ([\d.]+) ms \(([\d.]+)-([\d.]+)\)")
data = {}  # dev -> (gqa, sessions, depth) -> (seqs, rows, apart, sp)
devs = []
for a in sys.argv[1:]:
    name, f = a.split("=", 1)
    devs.append(name)
    d = data.setdefault(name, {})
    for l in open(f, errors="replace"):
        m = pat.search(l)
        if not m:
            continue
        gqa, len0, sess, depth, sp, s, slo, shi, r, rlo, rhi = m.groups()
        apart = "<" if float(rhi) < float(slo) else (">" if float(rlo) > float(shi) else "~")
        d[(int(gqa), int(sess or 1), int(len0 or depth))] = (float(s), float(r), apart, int(sp))
floor = {4: 1024, 6: 2048, 8: 16384}
print("### per call, 24/4, one session's 4 rows\n")
print("| depth (keys) | " + " | ".join(f"{d} seqs | {d} rows | {d} rows/seqs" for d in devs) + " |")
print("|---:|" + "---:|---:|---:|" * len(devs))
for depth in sorted({k[2] for d in data.values() for k in d if k[0] == 6 and k[1] == 1}):
    cells = []
    for dv in devs:
        v = data[dv].get((6, 1, depth))
        cells.append(f"{v[0]:.3f} | {v[1]:.3f} | {v[1] / v[0]:.2f}" if v else " | | ")
    print(f"| {depth:,} | " + " | ".join(cells) + " |")
print("\n### routing grid, rows/seqs (a mark after the ratio: `<` rows faster with the arms' ranges apart, `>` seqs faster, `~` ranges overlap)\n")
print("| GQA | launch | depth (keys) | split keys | " + " | ".join(devs) + " | routed to |")
print("|---:|---|---:|---:|" + "---:|" * len(devs) + "---|")
for gqa in (4, 6, 8):
    keys = sorted({k for d in data.values() for k in d if k[0] == gqa}, key=lambda k: (k[1] != 1, k[1], k[2]))
    for k in keys:
        cells, sp = [], ""
        for dv in devs:
            v = data[dv].get(k)
            cells.append(f"{v[1] / v[0]:.2f} {v[2]}" if v else "")
            sp = v[3] if v else sp
        launch = "1 session x 4 rows" if k[1] == 1 else f"{k[1]} sessions x 4 rows"
        route = "rows walk" if k[2] + 1 >= floor[gqa] else "seqs"
        print(f"| {gqa} | {launch} | {k[2]:,} | {sp} | " + " | ".join(cells) + f" | {route} |")
