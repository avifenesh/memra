#!/usr/bin/env python3
"""Per-part ms/step and effective GB/s from nsys cuda_gpu_kern_sum CSVs (decode-prof dir)."""
import csv, glob, os, re, sys

D = sys.argv[1]
STEPS = 16  # 8 measured steps x 2 passes; FA instances = 16 layers x STEPS
KV_TOK = 1856 * 16  # bytes per token over the 16 full-attention layers
GDN_SEQ = 48 * 48 * 128 * 128 * 4 * 2  # fp32 state read + write per session per step
W = 15.7e9

def part(name):
    if name.startswith("fa_decode"):
        return "fa"
    if name.startswith("qmatvec"):
        return "mm"
    if name.startswith("gdn_scan"):
        return "gdn"
    return "rest"

tick = {}
for log in glob.glob(os.path.join(D, "d*-b*.log")):
    if log.endswith("-nsys.out"):
        continue
    m = re.search(r"CHUNKSWEEP.*\(([\d.]+) ms/tick", open(log).read())
    if m:
        tick[os.path.basename(log)[:-4]] = float(m.group(1))

print("cell | tick ms | mm ms (GB/s) | fa ms (GB/s) | gdn ms (GB/s) | rest ms | gap ms")
for f in sorted(glob.glob(os.path.join(D, "d*-kern.csv")), key=lambda p: [int(x) for x in re.findall(r"\d+", os.path.basename(p))]):
    tag = os.path.basename(f)[:-9]
    depth, b = [int(x) for x in re.findall(r"\d+", tag)]
    sums = {"fa": 0, "mm": 0, "gdn": 0, "rest": 0}
    fa_inst = 0
    for row in csv.DictReader(open(f)):
        ns = float(row["Total Time (ns)"])
        p = part(row["Name"])
        sums[p] += ns
        if row["Name"].startswith("fa_decode_vec"):
            fa_inst = int(row["Instances"])
    passes = 2 if b > 16 else 1
    steps = fa_inst / 16 / passes if fa_inst else STEPS
    ms = {k: v / steps / 1e6 for k, v in sums.items()}
    rows = min(b, 16)
    t = tick.get(tag)
    # tick covers all passes; per-pass kernel sums are per step of one pass set
    mm_gbs = W * passes / (ms["mm"] / 1e3) / 1e9
    fa_gbs = b * depth * KV_TOK / (ms["fa"] / 1e3) / 1e9
    gdn_gbs = b * GDN_SEQ / (ms["gdn"] / 1e3) / 1e9
    tot = sum(ms.values())
    gap = (t - tot) if t else float("nan")
    print(f"{tag} | {t} | {ms['mm']:.1f} ({mm_gbs:.0f}) | {ms['fa']:.1f} ({fa_gbs:.0f}) | {ms['gdn']:.1f} ({gdn_gbs:.0f}) | {ms['rest']:.1f} | {gap:.1f}")
