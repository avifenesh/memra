#!/usr/bin/env python3
"""Per-op GPU time of two prof_attr.py reports side by side (one outer range, e.g. L21/decode).

usage: prof_compare.py BEFORE.ops.json AFTER.ops.json RANGE [RANGE ...]
"""
import json
import sys


def main():
    a = json.load(open(sys.argv[1]))
    b = json.load(open(sys.argv[2]))
    for rng in sys.argv[3:]:
        ra, rb = a.get(rng), b.get(rng)
        if not ra or not rb:
            print(f"{rng}: missing in one report")
            continue
        print(f"### {rng}: wall {ra['wall_ms']:.3f} -> {rb['wall_ms']:.3f} ms, GPU busy {ra['gpu_busy_ms']:.3f} -> {rb['gpu_busy_ms']:.3f} ms\n")
        print("| op | before ms | after ms | after / before |")
        print("|---|---|---|---|")
        ops = sorted(set(ra["ops_ms"]) | set(rb["ops_ms"]), key=lambda k: -ra["ops_ms"].get(k, 0))
        for op in ops:
            x, y = ra["ops_ms"].get(op, 0.0), rb["ops_ms"].get(op, 0.0)
            if max(x, y) < 0.002:
                continue
            ratio = f"{y / x:.2f}" if x > 0 else "-"
            print(f"| {op} | {x:.3f} | {y:.3f} | {ratio} |")
        print()


if __name__ == "__main__":
    main()
