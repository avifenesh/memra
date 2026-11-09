#!/usr/bin/env python3
"""Print every table RESULTS.md quotes, from the runs (receipts first, prose second).

  python3 results_tables.py RUNS_DIR > tables.md
"""
import collections
import json
import re
import subprocess
import sys
from pathlib import Path

runs = Path(sys.argv[1])
H = Path(__file__).resolve().parent


def sh(*args):
    return subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True).stdout


def batch_counts(run):
    log = (run / 'serve.log').read_text(errors='replace')
    sizes = [int(m) for m in re.findall(r'\[prime-batch\] B=(\d+)', log)]
    rows = [int(m) for m in re.findall(r'\[prime-batch\] B=\d+ tokens=(\d+)', log)]
    return len(sizes), (max(sizes) if sizes else 0), (max(rows) if rows else 0)


for phase in ('c8', 'c4', 'c1'):
    d = runs / phase
    if not d.exists():
        continue
    print(f'### {phase}\n')
    print(sh(H / 'serving_table.py', d, '--md'))
    print('| run | prime-batch calls | widest B | widest rows |')
    print('|---|---|---|---|')
    for r in sorted(p for p in d.iterdir() if p.is_dir() and (p / 'serve.log').exists()):
        n, b, t = batch_counts(r)
        print(f'| {r.name} | {n} | {b} | {t} |')
    print()

for phase in ('g8',):
    d = runs / phase
    if not d.exists():
        continue
    a = sorted(d.glob('*-A'))
    b = sorted(d.glob('*-B'))
    print(f'### {phase} (greedy exactness)\n')
    for ra in a:
        for rb in b:
            for ph in ('prime', 'measure'):
                print(f'{ra.name} vs {rb.name} {ph}:', sh(H / 'exact_cmp.py', ra, rb, '--phase', ph).strip())
    print()

d = runs / 'mixed'
if d.exists():
    print('### mixed\n')
    print(sh(H / 'score.py', d, '--md'))
    a = sorted(d.glob('*-A'))
    b = sorted(d.glob('*-B'))
    for ra in a[:1]:
        for rb in b:
            print(f'{ra.name} vs {rb.name}:', sh(H / 'exact_cmp.py', ra, rb).strip())
    print('| run | prime-batch calls | widest B | widest rows |')
    print('|---|---|---|---|')
    for r in sorted(p for p in d.iterdir() if p.is_dir() and (p / 'serve.log').exists()):
        n, bb, t = batch_counts(r)
        print(f'| {r.name} | {n} | {bb} | {t} |')
    print()

g = runs / 'grow'
if g.exists():
    print('### grow-rewind replay\n')
    print(subprocess.run([sys.executable, str(Path.home() / '.cache/memra-c8r2/ext/growrewind/summarize.py'), str(g)],
                         capture_output=True, text=True).stdout)
