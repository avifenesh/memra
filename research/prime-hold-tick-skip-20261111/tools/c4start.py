#!/usr/bin/env python3
"""How each c=4 phase of the normal cell starts (memra#1223 follow-up): the live count in the
first tick after the phase's first admission (1 = the first request was admitted and primed
alone, the others in the next tick; 2 = two and two), whether a pooled cohort engaged before all
four were primed, and the phase's TTFTs from normal.jsonl.
usage: c4start.py RUNDIR   (server-<arm>.log and normal.jsonl)"""
import json
import pathlib
import re
import statistics
import sys

d = pathlib.Path(sys.argv[1])
rows = [json.loads(l) for l in open(d / "normal.jsonl") if l.strip()]
by_arm = {}
for r in rows:
    by_arm.setdefault(r["arm"], []).append(r)
print("| boot | phase | first tick: live sessions | ticks until all 4 primed | TTFT p50 s | TTFTs sorted (first 4) |")
print("|---|---:|---:|---:|---:|---|")
for arm, phases in by_arm.items():
    log = (d / f"server-{arm}.log").read_text(errors="replace").splitlines()
    sk = [i for i, l in enumerate(log) if l.startswith('[spec-k] model="q"')]
    before = 0
    for n, r in enumerate(phases):
        k0, before = before, before + r["requests"]
        if r["c"] != 4:
            continue
        start = sk[k0] if k0 < len(sk) else None
        first_live = ticks_to_four = None
        if start is not None:
            ticks = [l for l in log[start:start + 400] if l.startswith("[tick]")]
            if ticks:
                first_live = int(re.search(r" spec=(\d+)", ticks[0]).group(1))
                for t, l in enumerate(ticks):
                    if int(re.search(r" spec=(\d+)", l).group(1)) >= 4:
                        ticks_to_four = t + 1
                        break
        tt = sorted(x["ttft_s"] for x in r["rows"] if x.get("ttft_s") is not None)
        print(f"| {arm} | {n} | {first_live} | {ticks_to_four} | {round(statistics.median(tt), 3) if tt else None} | {tt[:4]} |")
