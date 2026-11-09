#!/usr/bin/env python3
"""Summarize an interleaved multi-arm `dsv41-bench gemm` run (perf_gemm_ab.sh: run<i>-<arm>.json).

Per GEMM shape: each arm's median over its runs of the run median (ms), its within-arm spread, the
ratio to the first arm, and whether every run of every arm wrote the same output hash (a schedule
change must not move the bytes).

usage: gemm_ab_summary.py DIR
"""
import glob
import json
import os
import re
import statistics
import sys


def main():
    d = sys.argv[1]
    arms = [l.split()[0] for l in open(os.path.join(d, "arms.txt")) if l.strip()]
    per = {}  # shape -> arm -> [run medians]
    sha = {}  # shape -> set
    order = []
    for f in sorted(glob.glob(os.path.join(d, "run*-*.json")), key=lambda p: int(re.search(r"run(\d+)-", p).group(1))):
        arm = re.search(r"run\d+-(.*)\.json$", os.path.basename(f)).group(1)
        for r in json.load(open(f))["rows"]:
            key = " ".join(f"{k}={r[k]}" for k in r if k not in ("median_ms", "ms", "sha"))
            if key not in per:
                order.append(key)
            per.setdefault(key, {}).setdefault(arm, []).append(r["median_ms"])
            sha.setdefault(key, set()).add(r["sha"])
    print("| shape | " + " | ".join(f"{a} ms (spread)" for a in arms) + " | " +
          " | ".join(f"{a} / {arms[0]}" for a in arms[1:]) + " | bytes |")
    print("|---|" + "---|" * (2 * len(arms)))
    for key in order:
        meds = {}
        cells = []
        for a in arms:
            v = per[key].get(a, [])
            if not v:
                cells.append("-")
                continue
            m = statistics.median(v)
            meds[a] = m
            cells.append(f"{m:.4f} ({100 * (max(v) - min(v)) / m:.1f}%, n={len(v)})")
        ratios = [f"{meds[a] / meds[arms[0]]:.3f}" if a in meds and arms[0] in meds else "-" for a in arms[1:]]
        print(f"| {key} | " + " | ".join(cells) + " | " + " | ".join(ratios) + " | " +
              ("same" if len(sha[key]) == 1 else f"{len(sha[key])} DISTINCT") + " |")


if __name__ == "__main__":
    main()
