#!/usr/bin/env python3
"""Per-arm speedup over the shipped v2 tile timed in the same process (the REF line before it), grouped
by (arm, token count T, shape); the "shipped (again)" lines give the same-process A/A spread.
usage: pairs.py LOG [LOG ...]"""
import re
import sys
from collections import defaultdict

LINE = re.compile(r"^(?P<shape>\S+ \S+)\s+(?P<arm>.+?)\s+(?P<ms>-?[0-9.]+) ms\s+(?P<tmac>-?[0-9.]+) TMAC/s\s+(?P<mac>-?[0-9.]+) MAC/clk/SM@2200\s+bitdiff=(?P<bad>\d+) (?P<tag>REF|OK|FAIL)")
ratios = defaultdict(list)
bad = defaultdict(int)
TLINE = re.compile(r"^T=(\d+) reps=")
for path in sys.argv[1:]:
    ref = {}
    T = 0
    for line in open(path):
        tm = TLINE.match(line.strip())
        if tm:
            T = int(tm.group(1))
            ref = {}
            continue
        m = LINE.match(line.strip())
        if not m:
            continue
        shape, arm, mac, tag = m["shape"], m["arm"].strip(), float(m["mac"]), m["tag"]
        if tag == "REF":
            ref[shape] = mac
            continue
        if shape not in ref or mac <= 0:
            continue
        key = ("A/A shipped (again)" if "shipped (again)" in arm else arm, T, shape)
        ratios[key].append(mac / ref[shape])
        bad[key] += int(m["bad"])
for (arm, T, shape), r in sorted(ratios.items()):
    mean = sum(r) / len(r)
    print(f"{arm:<36} T={T:<5} {shape:<20} n={len(r):<2} mean {100 * (mean - 1):+6.1f}%  range {100 * (min(r) - 1):+6.1f}% .. {100 * (max(r) - 1):+6.1f}%  bitdiff {bad[(arm, T, shape)]}")
