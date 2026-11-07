#!/usr/bin/env python3
"""Rewrite the bf16 GDN decode-class code patch (main cf85a23ff5 -> 7456dac2e3) into the fp16 class.

Only added lines change: the CUDA bf16 intrinsics become their fp16 forms, the lane namespace
moves to research/gdn-state-fp16-20261107, and every bf16/BF16 token of the class becomes
fp16/FP16. Context and removed lines stay as they are, so the patch still applies to main.
usage: port_fp16.py IN.patch OUT.patch
"""
import re
import sys

src, dst = sys.argv[1], sys.argv[2]
subs = [
    ("__bfloat162float(", "__half2float("),
    ("__float2bfloat16(", "__float2half_rn("),
    ("__nv_bfloat16", "__half"),
    ("gdn-state-bf16-decode-20261107", "gdn-state-fp16-20261107"),
    ("GDN-STATE-BF16.md", "GDN-STATE-FP16.md"),
]
out = []
changed = 0
with open(src) as f:
    for line in f:
        if line.startswith("+") and not line.startswith("+++"):
            new = line
            for a, b in subs:
                new = new.replace(a, b)
            new = re.sub(r"bf16", "fp16", new)
            new = re.sub(r"BF16", "FP16", new)
            if new != line:
                changed += 1
            line = new
        out.append(line)
with open(dst, "w") as f:
    f.writelines(out)
print(f"rewrote {changed} added lines")
