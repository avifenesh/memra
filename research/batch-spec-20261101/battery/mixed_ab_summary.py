#!/usr/bin/env python3
"""Summarize an interleaved mixed_cell.py A/B.

usage: mixed_ab_summary.py LABEL_A=DIR[,DIR...] LABEL_B=DIR[,DIR...]

Per wave: each run's aggregate tok/s, the medians and the B/A ratio. Then the median end-to-end
seconds per request class over the runs: spec (r0, r1), riders (r2..r5: plain, device-sampled,
unconstrained) and phase (c) only (r6 host-sampled, r7 constrained).
"""
import json, os, statistics, sys

arms = []
for a in sys.argv[1:]:
    lab, dirs = a.split("=", 1)
    arms.append((lab, [json.load(open(os.path.join(d, "summary.json"))) for d in dirs.split(",")]))
waves = sorted({int(k.split("/")[0][1:]) for _, runs in arms for s in runs for k in s if "/agg" in k})
print("aggregate tok/s")
for c in waves:
    meds = []
    cells = []
    for lab, runs in arms:
        v = [s[f"c{c}/agg_tok_s"] for s in runs if f"c{c}/agg_tok_s" in s]
        meds.append(statistics.median(v))
        cells.append(f"{lab} {' / '.join(f'{x:.1f}' for x in v)} (median {meds[-1]:.1f})")
    print(f"  c={c}: " + "; ".join(cells) + f"; {arms[1][0]}/{arms[0][0]} {meds[1] / meds[0]:.3f}x")
classes = [("spec", (0, 1)), ("riders", (2, 3, 4, 5)), ("phase-c", (6, 7))]
print("median end-to-end s by request class")
for c in waves:
    row = []
    for name, js in classes:
        vals = []
        for lab, runs in arms:
            e = [s[f"c{c}/r{j}"]["e2e_s"] for s in runs for j in js
                 if f"c{c}/r{j}" in s and s[f"c{c}/r{j}"].get("e2e_s") is not None]
            vals.append(f"{lab} {statistics.median(e):.2f}" if e else f"{lab} -")
        if any(not v.endswith("-") for v in vals):
            row.append(f"{name}: " + ", ".join(vals))
    print(f"  c={c}: " + "; ".join(row))
