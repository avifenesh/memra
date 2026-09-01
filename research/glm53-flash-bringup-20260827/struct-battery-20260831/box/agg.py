#!/usr/bin/env python3
"""tp2-battery pricing aggregator: medians per (arm-run, tag) over rows.jsonl files.

usage: agg.py DIR [DIR..]      each DIR is one probe run dir containing rows.jsonl
Applies the 128-token floor BY NAME (the 3way short-sampled-row trap): rows with
out_tokens < 128 are listed as EXCLUDED, never silently dropped.
"""
import json, os, statistics, sys

runs = {}
excluded = []
for d in sys.argv[1:]:
    p = os.path.join(d, "rows.jsonl")
    if not os.path.exists(p):
        print(f"[agg] MISSING {p}")
        continue
    for line in open(p):
        r = json.loads(line)
        key = os.path.basename(d.rstrip("/"))
        if r["out_tokens"] < 128:
            excluded.append((key, r["tag"], r["arm"], r["out_tokens"], r["finish"]))
            continue
        runs.setdefault(key, []).append(r)

print("run\tn_rows\tmed_decode_tok_s\tmed_ttft_s\tmed_prime_s\ttags")
for key in sorted(runs):
    rows = [r for r in runs[key] if r["arm"] == "greedy"]
    if not rows:
        continue
    toks = statistics.median(r["decode_tok_s"] for r in rows)
    ttft = statistics.median(r["ttft_s"] for r in rows)
    prm = statistics.median(r["prime_s"] for r in rows)
    print(f"{key}\t{len(rows)}\t{toks:.3f}\t{ttft:.3f}\t{prm:.3f}\t{','.join(r['tag'] for r in rows)}")
    for r in runs[key]:
        if r["arm"] == "vendor":
            print(f"  vendor: {r['tag']} tok/s={r['decode_tok_s']:.3f} ttft={r['ttft_s']:.3f} out={r['out_tokens']} seed={r['seed']}")
if excluded:
    print("\nEXCLUDED by the 128-token floor (named, per the 3way trap guard):")
    for e in excluded:
        print(" ", e)
