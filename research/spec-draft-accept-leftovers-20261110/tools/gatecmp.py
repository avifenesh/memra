#!/usr/bin/env python3
"""Compare two seeded-invariance gate receipts row by row: text sha, completion tokens and
acceptance per (case, context, prompt). Inputs are runs.jsonl files or gate logs.
usage: gatecmp.py A B"""
import json, re, sys

def load(path):
    rows = {}
    if path.endswith(".jsonl"):
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                rows[(str(r["case"]), r["context"], r["prompt"])] = (r["sha"], r["completion_tokens"], r["acceptance"])
        return rows
    pat = re.compile(r"^\s*(\S+)\s+(\S+)\s+(\S+) tokens=(\d+) cached=\d+ acceptance=(\S+) sha=(\S+)")
    for line in open(path, errors="replace"):
        m = pat.match(line)
        if m:
            acc = None if m.group(5) == "None" else float(m.group(5))
            rows[(m.group(1), m.group(2), m.group(3))] = (m.group(6), int(m.group(4)), acc)
    return rows

a, b = load(sys.argv[1]), load(sys.argv[2])
shared = sorted(set(a) & set(b))
sha = sum(1 for k in shared if a[k][0] == b[k][0])
tok = sum(1 for k in shared if a[k][1] == b[k][1])
acc = sum(1 for k in shared if a[k][2] == b[k][2])
print(f"rows: {len(a)} and {len(b)}, shared {len(shared)}; same text sha {sha}, same completion tokens {tok}, same acceptance {acc}")
for k in shared:
    if a[k] != b[k]:
        print("DIFF", k, a[k], b[k])
only = sorted(set(a) ^ set(b))
if only:
    print("rows in one receipt only:", ", ".join("/".join(k) for k in only))
sys.exit(0 if shared and sha == tok == acc == len(shared) else 1)
