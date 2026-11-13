#!/usr/bin/env python3
"""Per-run steady medians of dsv41-tpbench runs: the rule of steady.py (calls from 0.5 s after a run's
first call, a run's first call of each kind left out), one median a run, then the arm's median of those
and [min, max] over its runs.

usage: per_run_steady.py DIR   (DIR holds runN-ARM/cells.jsonl)
"""
import json
import os
import statistics
import sys

STEADY = 0.5


def calls_of(path):
    out = []
    for line in open(path):
        if not line.strip():
            continue
        c = json.loads(line)
        if c.get("kind") == "prefill":
            out.append((c["start_unix"], f"{c['prompt_len']} rows", c["ttft_s"] * 1e3))
        elif c.get("kind") == "decode":
            t = c["start_unix"] + c.get("ttft_s", 0.0)
            for s in c.get("step_s", []):
                out.append((t, f"decode c{c.get('c', 1)}", s * 1e3))
                t += s
    return sorted(out)


d = sys.argv[1]
runs = sorted((x for x in os.listdir(d) if x.startswith("run") and os.path.isdir(os.path.join(d, x))),
              key=lambda x: int(x[3:].split("-")[0]))
per = {}
for run in runs:
    arm = run.split("-", 1)[1]
    calls = calls_of(os.path.join(d, run, "cells.jsonl"))
    t0 = calls[0][0]
    seen, by = set(), {}
    for t, kind, ms in calls:
        if kind not in seen:
            seen.add(kind)
            continue
        if t - t0 >= STEADY:
            by.setdefault(kind, []).append(ms)
    for kind, v in by.items():
        per.setdefault((arm, kind), []).append((run, statistics.median(v), len(v)))
print(f"| arm | call | per-run steady medians, ms (calls) | median of runs [min, max] |")
print("|---|---|---|---|")
for (arm, kind), rows in sorted(per.items(), key=lambda x: (x[0][1], x[0][0])):
    meds = [m for _, m, _ in rows]
    cells = ", ".join(f"{r}: {m:.3f} ({n})" for r, m, n in rows)
    print(f"| {arm} | {kind} | {cells} | {statistics.median(meds):.3f} [{min(meds):.3f}, {max(meds):.3f}] |")
