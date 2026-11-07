#!/usr/bin/env python3
"""Summarise gb10_bench.sh's ab.txt: per cell, the fp32 and bf16 run medians of ms/tick (each run is
itself a median of 3 reps of 16 steps), the change, tok/s, and whether every run of a class kept one
token fingerprint.  Usage: parse_bench.py ab.txt"""
import re
import statistics
import sys

rows = {}
for line in open(sys.argv[1]):
    m = re.match(r'(s(\d+)-d(\d+))-r(\d+) bf16=(\d) .*?([\d.]+) tok/s \(([\d.]+) ms/tick.*fp=(\w+)', line)
    if not m:
        print('UNPARSED', line.strip()[:120])
        continue
    cell = (int(m.group(2)), int(m.group(3)))
    cls = int(m.group(5))
    rows.setdefault(cell, {0: [], 1: []})[cls].append((float(m.group(7)), float(m.group(6)), m.group(8)))
print('| sessions x depth | fp32 ms/tick | bf16 ms/tick | change | fp32 tok/s | bf16 tok/s | runs | fingerprints |')
print('|---|---:|---:|---:|---:|---:|---|---|')
for (s, d), by in sorted(rows.items()):
    f32 = statistics.median(x[0] for x in by[0])
    b16 = statistics.median(x[0] for x in by[1])
    t32 = statistics.median(x[1] for x in by[0])
    t16 = statistics.median(x[1] for x in by[1])
    fps = {c: sorted({x[2] for x in by[c]}) for c in (0, 1)}
    sep = 'every bf16 run faster' if max(x[0] for x in by[1]) < min(x[0] for x in by[0]) else 'overlap'
    print(f'| {s} x {d} | {f32:.2f} | {b16:.2f} | {100 * (b16 - f32) / f32:+.1f}% | {t32:.1f} | {t16:.1f} | '
          f'{len(by[0])}+{len(by[1])}, {sep} | fp32 {"/".join(fps[0])}, bf16 {"/".join(fps[1])} |')
