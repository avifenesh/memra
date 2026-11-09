#!/usr/bin/env python3
"""The grow-rewind replay as one table: per arm and cell, requests answered, HTTP 400/408,
generation tok/s, TTFT and E2E summed over the 29 requests, the second turns' paths and their
TTFT median (summarize.py's per-turn rows; the second turn is session_turn 2).

  python3 grow_table.py GROW_DIR
"""
import json
import subprocess
import sys
from pathlib import Path

g = Path(sys.argv[1])
S = Path('<growrewind-harness>/summarize.py')
lines = subprocess.run([sys.executable, str(S), str(g)], capture_output=True, text=True).stdout.splitlines()
cells, turns = {}, {}
for line in lines:
    if not line.startswith('{'):
        continue
    r = json.loads(line)
    if 'users' in r:
        cells[(r['arm'], r['users'])] = r
    elif 'cell' in r:
        users = int(r['cell'].rsplit('-c', 1)[1])
        turns[(r['arm'], users)] = r['turns']
print('| arm | c | answered | 400 / 408 | gen tok/s | TTFT sum s | E2E sum s | 2nd-turn paths | 2nd-turn TTFT median s |')
print('|---|---:|---|---|---|---|---|---|---|')
for (arm, users), c in sorted(cells.items(), key=lambda kv: (kv[0][1], kv[0][0])):
    t2 = turns.get((arm, users), {}).get('2') or turns.get((arm, users), {}).get(2) or {}
    paths = ', '.join(f'{k} {v}' for k, v in sorted(t2.get('paths', {}).items()))
    print(f"| {arm} | {users} | {c['ok']} of {c['turns']} | {c['http_400']} / {c['http_408']} | {c['aggregate_gen_tps']} | "
          f"{c['ttft_sum_s']} | {c['e2e_sum_s']} | {paths} | {t2.get('ttft_median_s')} |")
