#!/usr/bin/env python3
"""gain.py <run> <row>...: per layer, mixer gain rel(mix)/rel(in) and FFN gain rel(ffn)/rel(x1),
prefill against decode, where in = the residual entering the layer (out of the layer before)."""
import array, math, sys

run, rows = sys.argv[1], [int(r) for r in sys.argv[2:]]


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


def norm(a):
    return math.sqrt(sum(x * x for x in a))


res = {}
for row in rows:
    P, D = load("prefill", row), load("decode", row)
    nl = max(k[1] for k in P if k[0] == "mix") + 1
    lines = []
    for il in range(nl):
        if il == 0:
            rin = 0.0
        else:
            po = [a + b for a, b in zip(P[("x1", il - 1)], P[("ffn", il - 1)])]
            do = [a + b for a, b in zip(D[("x1", il - 1)], D[("ffn", il - 1)])]
            rin = rel(po, do)
        rm = rel(P[("mix", il)], D[("mix", il)])
        rx = rel(P[("x1", il)], D[("x1", il)])
        rf = rel(P[("ffn", il)], D[("ffn", il)])
        # size of each branch against the residual it is added to
        mfrac = norm(D[("mix", il)]) / max(norm(D[("x1", il)]), 1e-30)
        ffrac = norm(D[("ffn", il)]) / max(norm(D[("x1", il)]), 1e-30)
        lines.append((il, rin, rm, rx, rf, mfrac, ffrac))
    res[row] = lines
print("layer " + "".join(f"| row {r}: in mix(gain) x1 ffn(gain) |mix|/|x1| |ffn|/|x1| " for r in rows))
for il in range(len(res[rows[0]])):
    cells = []
    for r in rows:
        _, rin, rm, rx, rf, mf, ff = res[r][il]
        g = rm / rin if rin else float("nan")
        cells.append(f"| {rin:.3f} {rm:.3f}({g:4.1f}) {rx:.3f} {rf:.3f}({rf/rx:4.1f}) {mf:.2f} {ff:.2f} ")
    print(f"{il:5} " + "".join(cells))
