#!/usr/bin/env python3
"""Apply the pre-registered acceptance rule to the gate re-run."""
import struct, sys

def load(p):
    hdr, logits = {}, []
    for line in open(p):
        f = line.rstrip("\n").split("\t")
        if f[0] == "logit":
            logits.append((int(f[1]), struct.unpack(">f", bytes.fromhex(f[2]))[0]))
        elif len(f) >= 2:
            hdr[f[0]] = f[1]
    logits.sort()
    return hdr, [v for _, v in logits]

rerun_p, banked_nvfp4_p, bf16_p, fp8_p = sys.argv[1:5]
_, rerun = load(rerun_p)
_, banked = load(banked_nvfp4_p)
_, bf16 = load(bf16_p)
_, fp8 = load(fp8_p)

raw_rerun = open(rerun_p, "rb").read()
raw_banked = open(banked_nvfp4_p, "rb").read()
print("PRIMARY BAR (bit identity vs banked nvfp4-oracle.tsv):",
      "PASS" if raw_rerun == raw_banked else "not met")
if raw_rerun != raw_banked:
    diff = sum(1 for a, b in zip(rerun, banked) if a != b)
    print(f"  differing logit entries: {diff} of {len(banked)}")

def cmp(name, a, b):
    am, bm = max(range(len(a)), key=a.__getitem__), max(range(len(b)), key=b.__getitem__)
    d = [abs(x - y) for x, y in zip(a, b)]
    ra = sorted(range(len(a)), key=a.__getitem__, reverse=True)[:8]
    rb = sorted(range(len(b)), key=b.__getitem__, reverse=True)[:8]
    k = 0
    while k < 8 and ra[k] == rb[k]:
        k += 1
    print(f"{name}: argmax {am} vs {bm} -> {'MATCH' if am == bm else 'DIFFER'}"
          f" | top-{k} rank-identical | max_abs {max(d):.3f} | mean_abs {sum(d)/len(d):.3f}")
    return am == bm, k, max(d), sum(d)/len(d)

print()
print("FALLBACK BAR (this artifact, re-run, vs its own BF16 source):")
ok, k, mx, mn = cmp("  rerun NVFP4 vs BF16 twin", rerun, bf16)
cmp("  rerun NVFP4 vs vendor FP8", rerun, fp8)
cmp("  banked NVFP4 vs BF16 twin (for reference)", banked, bf16)
cmp("  vendor FP8 vs BF16 twin (the calibration row)", fp8, bf16)
print()
print("VERDICT:", "PASS" if (raw_rerun == raw_banked) or (ok and k >= 3 and mx <= 3.489 * 1.05) else "REVIEW")
