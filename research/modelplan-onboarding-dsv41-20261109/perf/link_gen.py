#!/usr/bin/env python3
"""Each card's PCIe link generation during the runs of a phase: for every run in RUNS.txt (lines "NAME start
HH:MM:SS" / "NAME rc N end HH:MM:SS") the share of telemetry samples at each generation, per card.
usage: link_gen.py TELEMETRY.csv RUNS.txt [CARDS]"""
import collections
import re
import sys

tel, runs = sys.argv[1], sys.argv[2]
cards = sys.argv[3] if len(sys.argv) > 3 else "01"
samples = []
for line in open(tel, errors="replace"):
    p = [x.strip() for x in line.split(",")]
    if len(p) < 11:
        continue
    m = re.search(r"(\d\d):(\d\d):(\d\d)\.(\d+)", p[0])
    if not m:
        continue
    t = int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]) + int(m[4]) / 1000
    samples.append((t, p[1], p[9], p[7]))
span = {}
for line in open(runs):
    m = re.match(r"(\S+) start (\d\d):(\d\d):(\d\d)", line)
    if m:
        span.setdefault(m[1], [0, 0])[0] = int(m[2]) * 3600 + int(m[3]) * 60 + int(m[4])
    m = re.match(r"(\S+) rc \d+ end (\d\d):(\d\d):(\d\d)", line)
    if m:
        span.setdefault(m[1], [0, 0])[1] = int(m[2]) * 3600 + int(m[3]) * 60 + int(m[4]) + 1
print("| run | card | samples | link generation: share of samples | generation over the run, one char a second (lowest seen) |")
print("|---|---|---|---|---|")
for name, (a, b) in span.items():
    if not a or not b:
        continue
    for c in cards:
        v = [(t, g) for t, i, g, _ in samples if i == c and a <= t <= b]
        if not v:
            continue
        cnt = collections.Counter(g for _, g in v)
        share = ", ".join(f"gen {g}: {n / len(v):.0%}" for g, n in sorted(cnt.items()))
        sec = collections.defaultdict(list)
        for t, g in v:
            sec[int(t - a)].append(g)
        strip = "".join(min(sec[s]) if s in sec else "." for s in range(int(b - a) + 1))
        print(f"| {name} | {c} | {len(v)} | {share} | {strip} |")
