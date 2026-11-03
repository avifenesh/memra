#!/usr/bin/env python3
"""Per-part ms/step from nsys cuda_gpu_kern_sum CSVs + the bench tick (combo profile dir).
usage: parts2.py DIR TAG... (TAG = file stem, e.g. d12288-b8, d12288-b8-bf16)"""
import csv, os, re, sys

D = sys.argv[1]


def part(n):
    if n.startswith("fa_decode"):
        return "attention"
    if n.startswith("qmatvec"):
        return "matmul"
    if n.startswith("gdn_scan"):
        return "gdn"
    return "rest"


for tag in sys.argv[2:]:
    sums = {"matmul": 0.0, "attention": 0.0, "gdn": 0.0, "rest": 0.0}
    top = []
    fa_inst = 0
    for row in csv.DictReader(open(os.path.join(D, tag + "-kern.csv"))):
        ns = float(row["Total Time (ns)"])
        sums[part(row["Name"])] += ns
        top.append((ns, row["Name"], int(row["Instances"])))
        if row["Name"].startswith("fa_decode_vec"):
            fa_inst = int(row["Instances"])
    steps = fa_inst / 16 if fa_inst else 16
    m = re.search(r"\(([\d.]+) ms/tick", open(os.path.join(D, tag + ".log")).read())
    tick = float(m.group(1)) if m else float("nan")
    ms = {k: v / steps / 1e6 for k, v in sums.items()}
    gap = tick - sum(ms.values())
    print(f"{tag}: tick {tick:.1f} | matmul {ms['matmul']:.1f} | attention {ms['attention']:.1f} | "
          f"gdn {ms['gdn']:.1f} | rest {ms['rest']:.1f} | gap {gap:.1f}")
    for ns, name, inst in sorted(top, reverse=True)[:8]:
        print(f"    {ns / steps / 1e6:7.2f} ms/step  {inst // int(steps):4d}/step  {name}")
