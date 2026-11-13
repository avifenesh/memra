#!/usr/bin/env python3
"""Per-op GPU time of two binaries from interleaved nsys profiles (perf_prof_ab.sh: run<i>-<A|B>.ops.json,
order A B B A; each file a prof_attr.py report).

Per outer range (e.g. L20/decode): every op's GPU time in each run, the arm means, B / A. An op that
only one arm's change touches moves; the others show the run-to-run noise.

usage: prof_ab_summary.py DIR [RANGE ...]   (default ranges: every range in the first report)
"""
import glob
import json
import os
import re
import sys


def main():
    d = sys.argv[1]
    runs = []
    for f in sorted(glob.glob(os.path.join(d, "run*-?.ops.json")), key=lambda p: int(re.search(r"run(\d+)-", p).group(1))):
        runs.append((f[-10], json.load(open(f))))
    if not runs:
        sys.exit("no run*-?.ops.json")
    ranges = sys.argv[2:] or list(runs[0][1].keys())
    for rng in ranges:
        per = {"A": [], "B": []}
        for arm, js in runs:
            if rng in js:
                per[arm].append(js[rng])
        if not per["A"] or not per["B"]:
            print(f"{rng}: missing in one arm")
            continue

        def mean(arm, f):
            v = [f(r) for r in per[arm]]
            return sum(v) / len(v)

        print(f"### {rng}: GPU busy {mean('A', lambda r: r['gpu_busy_ms']):.3f} -> {mean('B', lambda r: r['gpu_busy_ms']):.3f} ms, "
              f"wall {mean('A', lambda r: r['wall_ms']):.3f} -> {mean('B', lambda r: r['wall_ms']):.3f} ms "
              f"(A n={len(per['A'])}, B n={len(per['B'])} profiles)\n")
        print("| op | A runs ms | B runs ms | A mean | B mean | B / A |")
        print("|---|---|---|---|---|---|")
        ops = set()
        for arm in per:
            for r in per[arm]:
                ops |= set(r["ops_ms"])
        ma = {op: mean("A", lambda r: r["ops_ms"].get(op, 0.0)) for op in ops}
        mb = {op: mean("B", lambda r: r["ops_ms"].get(op, 0.0)) for op in ops}
        for op in sorted(ops, key=lambda o: -ma[o]):
            if max(ma[op], mb[op]) < 0.002:
                continue
            ra = ", ".join(f"{r['ops_ms'].get(op, 0.0):.3f}" for r in per["A"])
            rb = ", ".join(f"{r['ops_ms'].get(op, 0.0):.3f}" for r in per["B"])
            ratio = f"{mb[op] / ma[op]:.2f}" if ma[op] > 0 else "-"
            print(f"| {op} | {ra} | {rb} | {ma[op]:.3f} | {mb[op]:.3f} | {ratio} |")
        print()


if __name__ == "__main__":
    main()
