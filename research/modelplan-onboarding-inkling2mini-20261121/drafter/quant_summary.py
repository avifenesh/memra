#!/usr/bin/env python3
"""Lane S: per-component NVFP4 drafter results from drafter_quant_study.py comp.
Per component and kind: simulated K=7 greedy acceptance length (steps along the sequence) for the
component and for BF16, mean a(P), drafter-logit KL(BF16 || component) and top-1 agreement."""
import json
import sys
from collections import defaultdict


def sim(acc, T0, n):
    P, steps, a = T0, 0, 0
    per = {T0 + i: x for i, x in enumerate(acc)}
    while P < n - 1:
        x = per.get(P, 0); steps += 1; a += x; P += x + 1
    return steps, a


agg = defaultdict(lambda: [0, 0, 0, 0, 0.0, 0.0, 0, 0, 0])
for l in open(sys.argv[1]):
    r = json.loads(l)
    n = r["first_anchor"] + len(r["accept"]) + 1
    s, a = sim(r["accept"], r["first_anchor"], n)
    sb, ab = sim(r["base_accept"], r["first_anchor"], n)
    for kind in (r["kind"], "all"):
        g = agg[(r["comp"], kind)]
        g[0] += s; g[1] += a; g[2] += sb; g[3] += ab
        g[4] += sum(r["kl"]); g[5] += sum(r["top1"]); g[6] += len(r["kl"]); g[7] += 1
        g[8] += sum(r["accept"]) - sum(r["base_accept"])
print("component\tkind\tprompts\tacc len\tBF16 acc len\tdelta\tKL\ttop1")
for (comp, kind), g in sorted(agg.items()):
    al, bl = g[1] / g[0], g[3] / g[2]
    print(f"{comp}\t{kind}\t{g[7]}\t{al:.3f}\t{bl:.3f}\t{al - bl:+.3f}\t{g[4] / g[6]:.2e}\t{g[5] / g[6]:.4f}")
