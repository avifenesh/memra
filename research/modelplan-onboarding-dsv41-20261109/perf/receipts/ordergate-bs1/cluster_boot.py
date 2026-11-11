#!/usr/bin/env python3
"""By-tape (cluster) bootstrap of the order gate's d: resample the cases with replacement, d = the step-weighted
mean of the cases' d means. Reads order_gate.py --json outputs. usage: cluster_boot.py JSON... [--boot N]"""
import json
import random
import sys

args = [a for a in sys.argv[1:] if not a.startswith("--")]
nb = int(sys.argv[sys.argv.index("--boot") + 1]) if "--boot" in sys.argv else 20000
if "--boot" in sys.argv:
    args.remove(str(nb))
cases = {}
for p in args:
    for c in json.load(open(p))["cases"]:
        cases[c["case"]] = c
rng = random.Random(1234)


def stat(cs):
    s = sum(c["steps"] for c in cs)
    return sum(c["steps"] * c["d_mean"] for c in cs) / s


for name, sel in [("chat", "chat"), ("long", "long"), ("all", None)]:
    cs = [c for c in cases.values() if sel is None or c["group"] == sel]
    if not cs:
        continue
    d = stat(cs)
    bs = sorted(stat([rng.choice(cs) for _ in cs]) for _ in range(nb))
    lo, hi = bs[int(0.025 * nb)], bs[int(0.975 * nb) - 1]
    print(f"{name}: {len(cs)} tapes, {sum(c['steps'] for c in cs)} steps, d {d:+.3e}, by-tape 95% [{lo:+.3e}, {hi:+.3e}]"
          f" (half-width {(hi - lo) / 2:.2e})")
