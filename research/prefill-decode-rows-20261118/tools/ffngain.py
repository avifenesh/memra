#!/usr/bin/env python3
"""ffngain.py <run> : per MoE layer, ffn-output rel diff over x1 rel diff (prefill vs decode),
split by whether the host top-k set is the same in both programs (from the run's log)."""
import array, math, re, statistics, sys

run = sys.argv[1]
log = open(f"{run}/log").read()


def load(prog, row):
    base = f"{run}/taps/{prog}-r{row}"
    data = array.array("f")
    data.frombytes(open(base + ".f32", "rb").read())
    out, off = {}, 0
    for line in open(base + ".idx"):
        name, layer, n = line.split("\t")
        n = int(n)
        out[(name, int(layer))] = data[off:off + n]
        off += n
    return out


def rel(a, b):
    num = sum((x - y) ** 2 for x, y in zip(a, b))
    den = sum(y * y for y in b)
    return math.sqrt(num / max(den, 1e-30))


gains = {"same": [], "DIFF": []}
nflip = []
for blk in log.split("=== row ")[1:]:
    row = int(blk.split()[0])
    flags = dict(
        (int(m.group(1)), m.group(2))
        for m in re.finditer(r"^\s+(\d+) (?:gdn|attn)\s+\S+\s+\S+\s+\S+\s+\S+\s+(same|DIFF)", blk, re.M)
    )
    P, D = load("prefill", row), load("decode", row)
    for il, f in flags.items():
        g = rel(P[("ffn", il)], D[("ffn", il)]) / max(rel(P[("x1", il)], D[("x1", il)]), 1e-12)
        gains[f].append(g)
    nflip.append(sum(1 for f in flags.values() if f == "DIFF"))
for k, v in gains.items():
    if v:
        v.sort()
        print(f"{k}: n={len(v)} ffn_rel/x1_rel median {statistics.median(v):.2f} p90 {v[int(0.9*len(v))]:.2f}")
print(f"rows={len(nflip)} layers with a different top-k set per row: {nflip}")
