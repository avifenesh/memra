#!/usr/bin/env python3
"""Grouped expert GEMM medians: swapped forms (--form 0, by rows) vs the normal form (--form 1), A B B A.
usage: perf_form_summary.py DIR  (form0-0.json form1-1.json form1-2.json form0-3.json)"""
import json
import statistics
import sys

d = sys.argv[1]
runs = {0: ["form0-0.json", "form0-3.json"], 1: ["form1-1.json", "form1-2.json"]}
med = {}
for form, files in runs.items():
    for f in files:
        rows = json.load(open(f"{d}/{f}"))["rows"]
        for r in rows:
            if r.get("op") != "mxfp4_grouped":
                continue
            key = (r["name"], r["groups"], r["experts"], r["rows"])
            med.setdefault(key, {}).setdefault(form, []).append(r["median_ms"])
print(f"{'shape':44} {'swapped ms (2 runs)':>22} {'normal ms (2 runs)':>22} {'normal/swapped':>15}")
for key, v in med.items():
    a, b = v.get(0, []), v.get(1, [])
    if not a or not b:
        continue
    ratio = statistics.mean(b) / statistics.mean(a)
    name = f"{key[0]} groups {key[1]} with rows {key[2]} rows {key[3]}"
    print(f"{name:44} {' / '.join(f'{x:.4f}' for x in a):>22} {' / '.join(f'{x:.4f}' for x in b):>22} {ratio:15.3f}")
