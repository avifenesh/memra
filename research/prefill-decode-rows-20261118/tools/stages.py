#!/usr/bin/env python3
"""stages.py <row> <layer,layer,..> <ref arm/prog> <arm/prog>...: rel L2 of each GDN stage tap
(and the layer's mix/x1/ffn) of every program against the reference program."""
import array, math, sys

row = int(sys.argv[1])
layers = [int(x) for x in sys.argv[2].split(",")]
ref, others = sys.argv[3], sys.argv[4:]
STAGES = ["la_qkv", "la_z", "la_beta_raw", "la_alpha", "la_q", "la_k", "la_v", "la_glog", "la_o", "mix", "x1", "ffn"]


def load(spec):
    arm, prog = spec.split("/")
    base = f"{arm}/taps/{prog}-r{row}"
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


R = load(ref)
O = [(s, load(s)) for s in others]
print(f"row {row}; rel L2 against {ref}")
print("layer stage        " + " ".join(f"{s:>28}" for s, _ in O))
for il in layers:
    for st in STAGES:
        if (st, il) not in R:
            continue
        cells = []
        for _, T in O:
            cells.append(f"{rel(T[(st, il)], R[(st, il)]):28.3e}" if (st, il) in T else f"{'-':>28}")
        print(f"{il:5} {st:12} " + " ".join(cells))
