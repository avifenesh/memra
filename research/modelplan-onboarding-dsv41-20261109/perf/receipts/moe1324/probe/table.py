#!/usr/bin/env python3
"""Per rows: base (w13+swiglu, w2), main (w1w3 + swiglu + w2), and chosen variants' totals (w1w3 + swiglu + w2), us."""
import re, sys
from collections import defaultdict
from statistics import median
show = [a[7:] for a in sys.argv[1:] if a.startswith("--show=")][0].split(",")
files = [f for f in sys.argv[1:] if not f.startswith("--")]
main = defaultdict(list); sw = defaultdict(list); base = defaultdict(list); var = defaultdict(list)
for f in files:
    for line in open(f):
        if m := re.match(r"cell main rows\s+(\d+) (\w+)\s+n.* main\s+([\d.]+) us", line): main[(int(m[1]), m[2])].append(float(m[3]))
        elif m := re.match(r"cell main rows\s+(\d+) swiglu\s+([\d.]+)", line): sw[int(m[1])].append(float(m[2]))
        elif m := re.match(r"cell base rows\s+(\d+) w13\+swiglu\s+([\d.]+) us w2\s+([\d.]+)", line): base[int(m[1])].append(float(m[2]) + float(m[3]))
        elif m := re.match(r"var rows\s+(\d+) (\w+)\s+(\S+)\s+([\d.]+) us", line): var[(int(m[1]), m[2], m[3])].append(float(m[4]))
print("| rows | base | main | main/base | " + " | ".join(f"{v} | /base" for v in show) + " |")
print("|---|---|---|---|" + "---|---|" * len(show))
for r in sorted({k[0] for k in main}):
    s = median(sw[r]); b = median(base[r]) if base[r] else float("nan")
    m = median(main[(r, "w1w3")]) + s + median(main[(r, "w2")])
    cells = []
    for v in show:
        a, c = var.get((r, "w1w3", v)), var.get((r, "w2", v))
        t = median(a) + s + median(c) if a and c else float("nan")
        cells.append(f"{t:.1f} | {t / b:.3f}")
    print(f"| {r} | {b:.1f} | {m:.1f} | {m / b:.3f} | " + " | ".join(cells) + " |")
