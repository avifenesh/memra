#!/usr/bin/env python3
"""Per-arm speedup over the shipped v2 tile timed in the same process (the REF line before it).
usage: pairs.py LOG [LOG ...] -> per arm and shape: n, mean and range of candidate / REF rate."""
import re
import sys
from collections import defaultdict

LINE = re.compile(r"^(?P<shape>\S+ \S+)\s+(?P<arm>.+?)\s+(?P<ms>-?[0-9.]+) ms\s+(?P<tmac>-?[0-9.]+) TMAC/s\s+(?P<mac>-?[0-9.]+) MAC/clk/SM@2200\s+bitdiff=(?P<bad>\d+) (?P<tag>REF|OK|FAIL)")
ratios = defaultdict(list)
bad = defaultdict(int)
for path in sys.argv[1:]:
    ref = {}
    for line in open(path):
        m = LINE.match(line.strip())
        if not m:
            continue
        shape, arm, mac, tag = m["shape"], m["arm"].strip(), float(m["mac"]), m["tag"]
        if tag == "REF":
            ref[shape] = mac
            continue
        if "shipped (again)" in arm or shape not in ref or mac <= 0:
            continue
        ratios[(arm, shape)].append(mac / ref[shape])
        bad[(arm, shape)] += int(m["bad"])
for (arm, shape), r in sorted(ratios.items()):
    mean = sum(r) / len(r)
    print(f"{arm:<36} {shape:<20} n={len(r):<2} mean {100 * (mean - 1):+6.1f}%  range {100 * (min(r) - 1):+6.1f}% .. {100 * (max(r) - 1):+6.1f}%  bitdiff {bad[(arm, shape)]}")
