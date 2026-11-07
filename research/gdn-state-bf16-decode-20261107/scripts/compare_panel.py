#!/usr/bin/env python3
"""Compare long-context panel runs task by task.
Usage: compare_panel.py LABEL=records.jsonl [LABEL=records.jsonl ...]
Prints each run's pass counts per task, then for every pair the discordant tasks (pass in one,
fail in the other) and the two-sided exact sign-test p over them."""
import json
import math
import sys
from itertools import combinations


def load(path):
    runs = {}
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        ok = r.get('pass', r.get('correct', r.get('ok')))
        runs[(r['task'], r['id'])] = bool(ok)
    return runs


def sign_p(a, b):
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


runs = {}
for arg in sys.argv[1:]:
    label, path = arg.split('=', 1)
    runs[label] = load(path)
for label, r in runs.items():
    by = {}
    for (task, _), ok in r.items():
        p, n = by.get(task, (0, 0))
        by[task] = (p + ok, n + 1)
    print(label, ' '.join(f'{t} {p}/{n}' for t, (p, n) in sorted(by.items())))
for a, b in combinations(runs, 2):
    ra, rb = runs[a], runs[b]
    keys = sorted(set(ra) & set(rb))
    only_a = [k for k in keys if ra[k] and not rb[k]]
    only_b = [k for k in keys if rb[k] and not ra[k]]
    for task in sorted({k[0] for k in keys}):
        oa = [k[1] for k in only_a if k[0] == task]
        ob = [k[1] for k in only_b if k[0] == task]
        print(f'{a} vs {b} {task}: {a}-only {oa} {b}-only {ob} p={sign_p(len(oa), len(ob)):.3f}')
