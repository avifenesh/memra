#!/usr/bin/env python3
"""ABBA table for grow_ab.sh arms: per workload, cell and arm kind (base / fix), the runs' aggregate
tok/s, TTFT sum and median, E2E sum, HTTP 400/408, and the second turns' pool paths and TTFT; plus
the counters and the fixed residual per arm. Reads summarize.py's logic. usage: ab_table.py OUT"""
import collections
import json
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import summarize  # noqa: E402

root = pathlib.Path(sys.argv[1])
cells = collections.defaultdict(list)
turns = collections.defaultdict(list)
tails = []
for d in sorted(p for p in root.iterdir() if (p / "serve.log").exists() or (p / "serve.log.gz").exists()):
    _, kind, workload = d.name.split("-", 2)
    out, cs, tail = summarize.arm_summary(d)
    tails.append(tail)
    for c in cs:
        cells[(workload, c["users"], kind)].append(c)
    for o in out:
        users = int(o["cell"].rsplit("-c", 1)[1])
        turns[(workload, users, kind)].append(o["turns"].get("2") or o["turns"].get(2))


def fmt(xs, nd=2):
    xs = [x for x in xs if x is not None]
    return " / ".join(f"{x:.{nd}f}" for x in xs) if xs else "-"


print("| workload | c | arm | runs | tok/s | TTFT sum s | E2E sum s | 400 / 408 | 2nd-turn paths | 2nd-turn TTFT median s |")
print("|---|---:|---|---:|---|---|---|---|---|---|")
for key in sorted(cells):
    workload, users, kind = key
    cs = cells[key]
    ts = turns[key]
    paths = collections.Counter()
    for t in ts:
        for p, n in (t or {}).get("paths", {}).items():
            paths[p] += n
    print(f"| {workload} | {users} | {kind} | {len(cs)} | {fmt([c['aggregate_gen_tps'] for c in cs])} | "
          f"{fmt([c['ttft_sum_s'] for c in cs], 1)} | {fmt([c['e2e_sum_s'] for c in cs], 1)} | "
          f"{sum(c['http_400'] for c in cs)} / {sum(c['http_408'] for c in cs)} | "
          f"{', '.join(f'{p} {n}' for p, n in sorted(paths.items()))} | "
          f"{fmt([(t or {}).get('ttft_median_s') for t in ts], 3)} |")
print()
for t in tails:
    c = t["counters"]
    print(t["arm"], "rewound parks", c["dspark_pool_grid_park_rewinds"], "skips", c["dspark_pool_grid_park_skips"],
          "resumes", c["dspark_pool_grid_rewind_resumes"], "grows", c["dspark_pool_grid_rewind_grows"],
          "declines", c["dspark_pool_grid_declines"], "fixed residual MB", t["fixed_residual_mb_last"],
          "oom lines", t["oom_lines"])
