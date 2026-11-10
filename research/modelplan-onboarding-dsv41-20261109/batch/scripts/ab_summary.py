#!/usr/bin/env python3
"""Medians of a served A/B directory by arm (rN-ARM; A = every call alone, B = batched; any other arm by its name).
usage: ab_summary.py DIR"""
import glob
import json
import statistics
import sys

arms = {}
for f in sorted(glob.glob(sys.argv[1] + "/r*/bench/summary.json")):
    arm = f.split("/")[-3].split("-")[-1]
    w = json.load(open(f))["workloads"]
    row = arms.setdefault(arm, {"c1": [], "conc8": [], "req": [], "mhz": []})
    if "decode_c1" in w:
        row["c1"].append(w["decode_c1"]["decode_tok_s"]["median"])
    if "conc8" in w:
        num = lambda v: v.get("median") if isinstance(v, dict) else v
        row["conc8"].append(num(w["conc8"]["agg_out_tok_s"]))
        row["req"].append(num(w["conc8"]["per_req_decode_tok_s_median"]))
        row["mhz"].append(num(w["conc8"].get("gpu_sm_mean_mhz")))
        row.setdefault("ttft", []).append(num(w["conc8"].get("ttft_max_s")))
for arm, r in sorted(arms.items()):
    name = {"A": "alone", "B": "batched"}.get(arm, arm)
    for k in ("c1", "conc8", "req"):
        if r[k]:
            print(f"{name:8s} {k:6s} {r[k]} median {statistics.median(r[k]):.1f}")
    print(f"{name:8s} conc8 clock MHz {r['mhz']} ttft max s {r.get('ttft')}")
if arms.get("A", {}).get("conc8") and arms.get("B", {}).get("conc8"):
    a, b = statistics.median(arms["A"]["conc8"]), statistics.median(arms["B"]["conc8"])
    print(f"conc8 batched/alone {b / a:.2f}x")
    a, b = statistics.median(arms["A"]["c1"]), statistics.median(arms["B"]["c1"])
    print(f"decode_c1 batched/alone {b / a:.3f}x")
