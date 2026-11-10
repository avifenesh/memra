#!/usr/bin/env python3
"""Calls of dsv41-tpbench runs by the time the card has been busy.

A card's clock steps down once its draw reaches its power limit (after 0.15 to 0.5 s of continuous load on
a 250 W card), and a call then costs up to 12% more. A served step is in that steady regime, so a reported
number is the steady one: the median over calls that start at least STEADY seconds (default 0.5) after the
run's first call. The burst figure (calls in the first 0.15 s) is printed beside it so the size of the step
is on record. A run's first call of each kind is left out of every column.

usage: steady.py DIR... [--steady S]
Each DIR holds run directories named runN-ARM with a cells.jsonl. A decode cell gives one call a step; a
prefill cell is one call (its time to first token). Output: a markdown table per DIR, a row per arm and
call kind (rows of a prefill call, c of a decode), with the calls in each regime.
"""
import json
import os
import statistics
import sys

args = [a for a in sys.argv[1:] if not a.startswith("--")]
steady = 0.5
if "--steady" in sys.argv:
    steady = float(sys.argv[sys.argv.index("--steady") + 1])
    args.remove(sys.argv[sys.argv.index("--steady") + 1])
BURST = 0.15


def calls_of(path):
    """[(start, kind, ms)] of a run, in time order."""
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


def med(v):
    return f"{statistics.median(v):.3f} ({len(v)})" if v else "-"


def kind_key(k):
    head = k.split()[0]
    return (0, int(head)) if head.isdigit() else (1, 0)


for d in args:
    if not os.path.isdir(d):
        continue
    runs = sorted(
        (x for x in os.listdir(d) if x.startswith("run") and os.path.exists(os.path.join(d, x, "cells.jsonl"))),
        key=lambda x: int(x[3:].split("-")[0]),
    )
    by = {}
    spans = []
    for run in runs:
        arm = run.split("-", 1)[1]
        calls = calls_of(os.path.join(d, run, "cells.jsonl"))
        if not calls:
            continue
        t0 = calls[0][0]
        spans.append(calls[-1][0] - t0)
        seen = set()
        for t, kind, ms in calls:
            slot = by.setdefault((arm, kind), {"burst": [], "mid": [], "steady": []})
            # a run's first call of a kind builds its lazy state (and under the graph door records): left out
            if kind not in seen:
                seen.add(kind)
                continue
            dt = t - t0
            slot["steady" if dt >= steady else "burst" if dt < BURST else "mid"].append(ms)
    if not by:
        continue
    print(f"\n### {os.path.basename(d.rstrip('/'))}: calls by how long the card has been busy ({len(runs)} runs of {min(spans):.2f} to {max(spans):.2f} s)")
    print(f"| arm | call | first {BURST} s: median ms (calls) | to {steady} s | from {steady} s: the steady figure | steady / burst |")
    print("|---|---|---|---|---|---|")
    for (arm, kind), s in sorted(by.items(), key=lambda x: (x[0][0], kind_key(x[0][1]))):
        ratio = (
            f"{statistics.median(s['steady']) / statistics.median(s['burst']):.3f}" if s["steady"] and s["burst"] else "-"
        )
        print(f"| {arm} | {kind} | {med(s['burst'])} | {med(s['mid'])} | {med(s['steady'])} | {ratio} |")
