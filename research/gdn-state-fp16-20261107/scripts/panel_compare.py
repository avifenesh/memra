#!/usr/bin/env python3
"""Per-task comparison of quality panel runs (qual_panel.py records.jsonl).

usage: panel_compare.py NAME=records.jsonl [NAME=records.jsonl ...]
Prints each run's scores and failed HumanEval ids, then the discordant tasks against the first
run, with a two-sided exact sign test on the discordant pairs.
"""
import json
import math
import sys


def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return {r['id']: r for r in rows}


def sign_p(a, b):
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


runs = [(a.split('=', 1)[0], load(a.split('=', 1)[1])) for a in sys.argv[1:]]
for name, r in runs:
    he = [x for x in r.values() if x['task'] == 'long_humaneval']
    gs = [x for x in r.values() if x['task'] == 'long_gsm8k']
    fails = sorted((x['id'] for x in he if not x['pass']), key=lambda s: int(s.split('/')[1]))
    print(f'{name}: long HumanEval {sum(x["pass"] for x in he)}/{len(he)} '
          f'(truncated {sum(x["finish"] == "length" for x in he)}), long GSM8K '
          f'{sum(x["pass"] for x in gs)}/{len(gs)}; HumanEval fails: {", ".join(f.split("/")[1] for f in fails)}')
base_name, base = runs[0]
for name, r in runs[1:]:
    gained = sorted(k for k in base if k in r and not base[k]['pass'] and r[k]['pass'])
    lost = sorted(k for k in base if k in r and base[k]['pass'] and not r[k]['pass'])
    print(f'{name} vs {base_name}: lost {lost or "none"}, gained {gained or "none"}, '
          f'two-sided exact p = {sign_p(len(lost), len(gained)):.3f}')
