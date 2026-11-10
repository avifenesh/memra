#!/usr/bin/env python3
"""Table of an interleaved N-rank prefill A/B (perf_prefill_ab.sh): per run and prompt length the seconds to
the first token (the forward completed and the last row's logits read back), ms a layer, tokens a second,
gathers, copies and GB copied; then each arm's best and median. tpbench's forward_s is the host's issue time
(the forward returns before the cards finish): it is printed as "issue s", never as the result.
usage: prefill_ab_summary.py RECEIPT_DIR LAYERS"""
import glob
import json
import os
import statistics
import sys

r, layers = sys.argv[1], int(sys.argv[2])
rows = {}
print("| run | arm | tokens | round | first token s | ms/layer | tok/s | issue s | gathers | copies | GB copied | peak GiB a rank |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for d in sorted(glob.glob(os.path.join(r, "run*-*"))):
    if not os.path.isdir(d):
        continue
    run = os.path.basename(d)
    arm = run.split("-")[-1]
    p = os.path.join(d, "cells.jsonl")
    if not os.path.exists(p):
        print(f"| {run} | {arm} | no cells |")
        continue
    for line in open(p):
        if not line.strip():
            continue
        c = json.loads(line)
        if c.get("kind") != "prefill":
            continue
        n, f, issue = c["prompt_len"], c["ttft_s"], c["forward_s"]
        cm = c.get("comm", {})
        peak = max(c.get("peak_used", [0])) / 2**30
        rows.setdefault((arm, n), []).append(f)
        print(
            f"| {run} | {arm} | {n} | {c['round']} | {f:.4f} | {f / layers * 1e3:.2f} | {n / f:.0f} | {issue:.4f} | "
            f"{cm.get('collectives', 0)} | {cm.get('copies', 0)} | {cm.get('bytes', 0) / 1e9:.2f} | {peak:.2f} |"
        )
print()
print("| arm | tokens | runs | best ms | median ms | median ms/layer | median tok/s |")
print("|---|---|---|---|---|---|---|")
for (arm, n), v in sorted(rows.items(), key=lambda x: (x[0][1], x[0][0])):
    b = min(v)
    m = statistics.median(v)
    print(f"| {arm} | {n} | {len(v)} | {b * 1e3:.2f} | {m * 1e3:.2f} | {m / layers * 1e3:.3f} | {n / m:.0f} |")
print()
for n in sorted({k[1] for k in rows}):
    for arm in sorted({k[0] for k in rows} - {"A"}):
        if ("A", n) in rows and (arm, n) in rows:
            a, b = min(rows[("A", n)]), min(rows[(arm, n)])
            am, bm = statistics.median(rows[("A", n)]), statistics.median(rows[(arm, n)])
            print(f"{n} tokens: {arm}/A best {b / a:.3f} (A {a * 1e3:.2f} ms, {arm} {b * 1e3:.2f} ms), median {bm / am:.3f}")
