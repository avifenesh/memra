#!/usr/bin/env python3
"""Crossover table of the grouped expert bodies from `dsv41-bench gemm` JSON runs (the small-row body vs CUTLASS's
normal form, A B B A inside each run): per shape the median of each body over all runs, the ratio, the launch's
block count (tiles x 64-column blocks x projections; tiles = experts with rows x ceil(rows / 8)) and whether the
bytes were equal in every run. usage: crossover.py RUN.json... > table.md"""
import json
import statistics
import sys

rows = {}
for f in sys.argv[1:]:
    doc = json.load(open(f))
    for r in doc["rows"]:
        if r.get("op") != "mxfp4_grouped" or "small_ms" not in r:
            continue
        key = (r.get("set", "full"), r["name"], r["groups"], r["n"], r["k"], r.get("projections", 1), r["experts"], r["rows"])
        e = rows.setdefault(key, {"s": [], "c": [], "eq": True})
        e["s"] += r["small_ms"]
        e["c"] += r["cutlass_ms"]
        e["eq"] &= bool(r["bytes_equal"])
print("| set | proj | groups | n | k | experts with rows | rows an expert | blocks | small ms | CUTLASS normal ms | small / CUTLASS | bytes equal |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for key, e in rows.items():
    st, name, g, n, k, p, act, rr = key
    tiles = act * -(-rr // 8)
    blocks = tiles * -(-n // 64) * p
    s, c = statistics.median(e["s"]), statistics.median(e["c"])
    print(f"| {st} | {name} | {g} | {n} | {k} | {act} | {rr} | {blocks} | {s:.4f} | {c:.4f} | {s / c:.3f} | {'yes' if e['eq'] else 'NO'} |")
