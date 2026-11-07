#!/usr/bin/env python3
"""Decode-tick table from bench.sh logs: median ms/tick per arm per cell, change, token fingerprints.

usage: bench_table.py BENCH_DIR   (dbb-s<S>-d<D>-r<i>.log files and ab.txt)
The arm of each run comes from ab.txt (`fp16=0|1`); ms/tick from the CHUNKSWEEP line.
"""
import re
import statistics
import sys
from pathlib import Path

d = Path(sys.argv[1])
arms = {}
for line in (d / 'ab.txt').read_text().splitlines():
    m = re.match(r'(s\d+-d\d+-r\d+) fp16=(\d)', line)
    if m:
        fp = re.search(r'fp=(\w+)', line)
        arms[m.group(1)] = (m.group(2), fp.group(1) if fp else '')
cells = {}
for tag, (arm, fp) in arms.items():
    log = (d / f'dbb-{tag}.log').read_text()
    m = re.search(r'CHUNKSWEEP.*?([0-9.]+)\s*ms', log)
    nums = re.findall(r'([0-9.]+)\s*ms/(?:tick|step)', log)
    v = float(nums[-1]) if nums else (float(m.group(1)) if m else None)
    cell = tag.rsplit('-r', 1)[0]
    cells.setdefault(cell, {}).setdefault(arm, []).append((v, fp, tag))
print('| sessions x depth | fp32 ms/tick (runs) | fp16 ms/tick (runs) | change | fp32 fingerprints | fp16 fingerprints |')
print('|---|---|---|---:|---|---|')
for cell, by in sorted(cells.items(), key=lambda kv: (int(kv[0].split('-')[0][1:]), int(kv[0].split('-d')[1]))):
    a = [x[0] for x in by.get('0', []) if x[0] is not None]
    b = [x[0] for x in by.get('1', []) if x[0] is not None]
    ma = statistics.median(a) if a else float('nan')
    mb = statistics.median(b) if b else float('nan')
    s, dd = cell.split('-')
    print(f'| {s[1:]} x {dd[1:]} | {ma:.2f} ({", ".join(f"{x:.2f}" for x in a)}) | '
          f'{mb:.2f} ({", ".join(f"{x:.2f}" for x in b)}) | {100 * (mb / ma - 1):+.1f}% | '
          f'{",".join(sorted({x[1] for x in by.get("0", [])}))} | '
          f'{",".join(sorted({x[1] for x in by.get("1", [])}))} |')
