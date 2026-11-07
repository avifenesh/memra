#!/usr/bin/env python3
"""Stall samples per SASS opcode from an ncu --page source --csv --print-source sass export.
usage: sass_stalls.py FILE [top]"""
import csv
import sys
from collections import Counter

path = sys.argv[1]
top = int(sys.argv[2]) if len(sys.argv) > 2 else 12
rows = list(csv.reader(open(path, newline="")))
hdr = next(i for i, r in enumerate(rows) if r and r[0] == "Address")
names = rows[hdr]
ix = {n: i for i, n in enumerate(names)}
samp, notiss, execd = Counter(), Counter(), Counter()
lines = []
for r in rows[hdr + 1:]:
    if len(r) < len(names):
        continue
    src = r[ix["Source"]].strip()
    op = src.split()[0] if src else "?"
    if op.startswith("@"):
        op = src.split()[1]
    base = op.split(".")[0]
    s = float(r[ix["Warp Stall Sampling (All Samples)"]] or 0)
    n = float(r[ix["Warp Stall Sampling (Not-issued Samples)"]] or 0)
    e = float((r[ix["Instructions Executed"]] or "0").replace(",", ""))
    samp[base] += s
    notiss[base] += n
    execd[base] += e
    lines.append((s, r[ix["Address"]], src[:70]))
tot = sum(samp.values())
print(f"samples {tot:.0f}; instructions executed {sum(execd.values()):.4g}")
print(f"{'opcode':<10} {'samples%':>9} {'not-issued%':>12} {'executed':>12} {'exec share':>10}")
te = sum(execd.values())
for op, s in samp.most_common(top):
    print(f"{op:<10} {100 * s / tot:9.1f} {100 * notiss[op] / tot:12.1f} {execd[op]:12.4g} {100 * execd[op] / te:9.1f}%")
print("top instructions by samples:")
for s, a, src in sorted(lines, reverse=True)[:top]:
    print(f"  {100 * s / tot:5.1f}%  {a}  {src}")
