#!/usr/bin/env python3
"""Summarise the TTFT A B B A runs: cold TTFT per size, the 256K new turn, the MemAvailable floor.
Usage: parse_ttft.py PROBESDIR"""
import glob
import json
import os
import statistics
import sys

d = sys.argv[1]
rows = []
for log in sorted(glob.glob(os.path.join(d, 'ttft-r*-bf16?.log'))):
    tag = os.path.basename(log)[:-4]
    cls = tag[-1]
    recs = {r['size']: r for r in (json.loads(l) for l in open(log) if l.startswith('{'))}
    mem = [json.loads(l)['mem_available_gib'] for l in open(os.path.join(d, tag + '-memory.jsonl')) if l.strip()]
    rows.append((tag, cls, recs, min(mem) if mem else float('nan')))
print('| run | class | 32K cold | 128K cold | 256K cold (258,048) | 256K turn | MemAvailable floor |')
print('|---|---|---:|---:|---:|---:|---:|')
for tag, cls, r, m in rows:
    g = lambda s, k='cold_max_s': r.get(s, {}).get(k)
    turn = r.get(258048, {}).get('turn1024', [None])[0]
    print(f'| {tag.split("-")[1]} | {"bf16" if cls == "1" else "fp32"} | {g(32768)} s | {g(131072)} s | {g(258048)} s | {turn} s | {m:.2f} GiB |')
for cls in '01':
    sel = [r for r in rows if r[1] == cls]
    if sel:
        med = lambda s: statistics.median(x[2][s]['cold_max_s'] for x in sel if s in x[2])
        print(f'class {cls}: medians 32K {med(32768):.2f} 128K {med(131072):.2f} 258K {med(258048):.2f}; floor {min(x[3] for x in sel):.2f} GiB')
