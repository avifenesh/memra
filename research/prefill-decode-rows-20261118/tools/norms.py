#!/usr/bin/env python3
"""norms.py <run> <row>: per layer, residual norm and largest |element| (with its index) in both
programs, and the top-3 dims of |prefill - decode|."""
import array, math, sys

run, row = sys.argv[1], int(sys.argv[2])


def load(prog):
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


P, D = load("prefill"), load("decode")
nl = max(k[1] for k in P if k[0] == "mix") + 1
print("layer  |p| |d|  max_p@i max_d@i  top|p-d| dims")
for il in range(nl):
    po = [a + b for a, b in zip(P[("x1", il)], P[("ffn", il)])]
    do = [a + b for a, b in zip(D[("x1", il)], D[("ffn", il)])]
    np_, nd = math.sqrt(sum(x * x for x in po)), math.sqrt(sum(x * x for x in do))
    ip = max(range(len(po)), key=lambda i: abs(po[i]))
    idd = max(range(len(do)), key=lambda i: abs(do[i]))
    diff = sorted(range(len(po)), key=lambda i: -abs(po[i] - do[i]))[:3]
    dd = " ".join(f"{i}:{po[i]:.2f}/{do[i]:.2f}" for i in diff)
    print(f"{il:5} {np_:8.2f} {nd:8.2f}  {po[ip]:8.2f}@{ip} {do[idd]:8.2f}@{idd}  {dd}")
