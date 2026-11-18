#!/usr/bin/env python3
"""dim.py <row> <dim> <arm/prog>...: one hidden dim across programs: mix, x1, ffn and out per layer."""
import array, sys

row, dim, specs = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3:]


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


T = [(s, load(s)) for s in specs]
layers = [int(x) for x in "0 1 2 3 4 5 7 11 15 19 23 24 25 26 27".split()]
print(f"row {row} dim {dim}: mix / ffn / out per layer")
print("layer " + " ".join(f"{s:>34}" for s in specs))
for il in layers:
    cells = []
    for _, D in T:
        if ("mix", il) not in D:
            cells.append(f"{'-':>34}")
            continue
        m, x1, f = D[("mix", il)][dim], D[("x1", il)][dim], D[("ffn", il)][dim]
        cells.append(f"{m:10.4f} {f:10.4f} {x1 + f:11.4f}")
    print(f"{il:5} " + " ".join(cells))
