#!/usr/bin/env python3
"""Compare dsv41-spec runs of one plan: per-job hashes.txt byte for byte against the first arm,
then the timing fields side by side.  usage: cmp_arms.py OUT_DIR ARM [ARM...]"""
import filecmp
import json
import os
import sys

root, arms = sys.argv[1], sys.argv[2:]
jobs = {}
for a in arms:
    rows = {}
    p = os.path.join(root, a, "stdout.jsonl")
    for line in open(p):
        if line.startswith("{"):
            d = json.loads(line)
            rows[(d["job"], d["id"])] = d
    jobs[a] = rows
ref = arms[0]
bad = 0
for a in arms:
    r = jobs[a]
    print(f"{a}: jobs {len(r)} pass {sum(bool(d['pass']) for d in r.values())}")
for a in arms[1:]:
    same = diff = missing = 0
    for job, jid in jobs[ref]:
        if job not in ("verify-rows", "batch-rows"):
            continue
        fa = os.path.join(root, ref, jid, "hashes.txt")
        fb = os.path.join(root, a, jid, "hashes.txt")
        if not (os.path.exists(fa) and os.path.exists(fb)):
            missing += 1
            print("  missing", jid, os.path.exists(fa), os.path.exists(fb))
        elif filecmp.cmp(fa, fb, shallow=False):
            same += 1
        else:
            diff += 1
            print("  DIFFERENT", jid)
    lines = sum(
        sum(1 for _ in open(os.path.join(root, ref, jid, "hashes.txt")))
        for job, jid in jobs[ref]
        if job in ("verify-rows", "batch-rows") and os.path.exists(os.path.join(root, ref, jid, "hashes.txt"))
    )
    print(f"hashes {a} vs {ref}: same {same} different {diff} missing {missing} ({lines} hash lines)")
    bad += diff + missing
print()
print("rows-time (one-token ms median | rows 2..8 repeat ms median)")
for key in jobs[ref]:
    if key[0] != "rows-time":
        continue
    for a in arms:
        d = jobs[a].get(key)
        if d:
            r = " ".join(f"{x:.2f}" for x in d["rows_2_to_8_repeat_ms_median"])
            print(f"  {key[1]:12s} prompt {d['prompt']:6d} {a:14s} 1: {d['one_token_ms_median']:.2f} | {r}")
print("batch-time (batched ms 1/2/4/8 | each alone)")
for key in jobs[ref]:
    if key[0] != "batch-time":
        continue
    for a in arms:
        d = jobs[a].get(key)
        if d:
            b = " ".join(f"{d.get(f'batched_{w}_ms', float('nan')):.2f}" for w in (1, 2, 4, 8, 16, 32) if f"alone_{w}_ms" in d)
            al = " ".join(f"{d.get(f'alone_{w}_ms', float('nan')):.2f}" for w in (1, 2, 4, 8, 16, 32) if f"alone_{w}_ms" in d)
            print(f"  {key[1]:10s} {a:14s} {b} | {al} | experts/layer {d.get('distinct_experts_per_layer_mean', 0):.1f}")
            x = " ".join(f"{m} " + "/".join(f"{d.get(f'{m}_{w}_ms', float('nan')):.2f}" for w in (1, 2, 4, 8, 16, 32) if f"alone_{w}_ms" in d) for m in ("argmax", "fwd", "chunk8") if any(k.startswith(m + "_") for k in d))
            if x:
                print(f"    {x}")
            for k in sorted(d):
                if k.startswith("launches_") or k.startswith("launch_ms_"):
                    v = d[k]
                    print(f"    {k}: total {v['total']} ({v.get('total_ms', 0)} ms) per layer {v['per_layer']}")
            ks = sorted((k for k in d if k.startswith("launch_ms_")), key=lambda k: int(k.split("_")[2]))
            if len(ks) == 2:
                a1, a8 = d[ks[0]]["per_kernel"], d[ks[1]]["per_kernel"]
                print(f"    per kernel, two steps: launches and ms, {ks[0]} -> {ks[1]}")
                for name in sorted(set(a1) | set(a8), key=lambda n: -(a8.get(n, [0, 0])[1])):
                    c1, m1 = a1.get(name, [0, 0])
                    c8, m8 = a8.get(name, [0, 0])
                    print(f"      {name:28s} {c1:4d} {m1:7.2f} -> {c8:4d} {m8:7.2f}  (+{m8 - m1:.2f})")
sys.exit(1 if bad else 0)
