#!/usr/bin/env python3
"""Recompute land27-four B/A per call with chosen runs dropped.

Prefill cells: ttft_s per call (the steady.md figure), round 0 dropped (11 steady calls a run, matching steady.md's 44 per arm).
Decode c1: step_s entries, pooled. Pooled median over the kept runs per arm, then B/A.
Usage: recompute.py RUNS_DIR [drop ...]   e.g. recompute.py runs run1-A run2-B
"""
import json, os, statistics, sys

d = sys.argv[1]
drop = set(sys.argv[2:])
cells = ["prefill-4", "prefill-8", "prefill-9", "prefill-10", "prefill-12", "prefill-16", "decode-c1"]
pool = {(a, c): [] for a in "AB" for c in cells}
per_run = {}
for run in sorted(os.listdir(d)):
    p = os.path.join(d, run, "cells.jsonl")
    if not os.path.isfile(p) or run in drop:
        continue
    arm = run.split("-")[1]
    for line in open(p):
        r = json.loads(line)
        c = r["cell"]
        if c not in cells:
            continue
        if r["kind"] == "prefill":
            if r["round"] == 0:
                continue
            v = [r["ttft_s"]]
        else:
            s = r.get("step_s")
            v = s if isinstance(s, list) else [r["step_mean_s"]]
            if r["round"] == 0:
                continue
        pool[(arm, c)].extend(v)
        per_run.setdefault((run, c), []).extend(v)

print("kept:", sorted({k[0] for k in per_run}))
print("| call | A ms (n) | B ms (n) | B/A |")
print("|---|---|---|---|")
for c in cells:
    a, b = pool[("A", c)], pool[("B", c)]
    ma, mb = statistics.median(a) * 1e3, statistics.median(b) * 1e3
    print(f"| {c} | {ma:.3f} ({len(a)}) | {mb:.3f} ({len(b)}) | {mb/ma:.4f} |")
print()
print("per-run medians, ms:")
runs = sorted({k[0] for k in per_run})
print("| call | " + " | ".join(runs) + " |")
for c in cells:
    print(f"| {c} | " + " | ".join(f"{statistics.median(per_run[(r, c)])*1e3:.3f}" for r in runs) + " |")
