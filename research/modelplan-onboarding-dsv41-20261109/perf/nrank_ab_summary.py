#!/usr/bin/env python3
"""Summary of an interleaved N-rank decode A/B (dsv41-tpbench decode cells, the arm in each run's name).

usage: nrank_ab_summary.py RECEIPT_DIR LAYERS
"""
import glob
import json
import os
import statistics
import sys

R, layers = sys.argv[1], int(sys.argv[2])
arms = {}
shas = set()
print("| run | arm | step ms, median of each round | worst round p95 ms | us/layer | generated ids sha256 |")
print("|---|---|---|---|---|---|")
runs = [d for d in glob.glob(f"{R}/run*-*") if os.path.isdir(d)]
for d in sorted(runs, key=lambda p: int(os.path.basename(p)[3:].split("-")[0])):
    arm = d.rsplit("-", 1)[1]
    cells = [json.loads(l) for l in open(f"{d}/cells.jsonl") if l.strip()]
    med = [statistics.median(c["step_s"]) * 1e3 for c in cells]
    p95 = max(c["tpot_p95_s"] for c in cells) * 1e3
    sha = {s for c in cells for s in c["gen_sha256"]}
    shas |= sha
    arms.setdefault(arm, []).append(statistics.median(med))
    print(f"| {os.path.basename(d)} | {arm} | {', '.join(f'{x:.3f}' for x in med)} | {p95:.3f} | "
          f"{statistics.median(med) / layers * 1e3:.1f} | {', '.join(sorted(s[:12] for s in sha))} |")
print()
print("| arm | run medians, ms/step | arm median | us/layer | against A |")
print("|---|---|---|---|---|")
a = statistics.median(arms["A"])
for arm in sorted(arms):
    m = statistics.median(arms[arm])
    print(f"| {arm} | {', '.join(f'{x:.3f}' for x in arms[arm])} | {m:.3f} | {m / layers * 1e3:.1f} | {m / a:.3f} |")
print(f"\ngenerated ids identical across every run of every arm: {len(shas) == 1}")
