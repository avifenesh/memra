#!/usr/bin/env python3
"""xarm.py <runs-dir> <row> <ref-arm>/<prog> <arm>/<prog> ...: per-layer rel L2 distance of every
listed program's taps to the reference program's taps (stdlib only)."""
import array
import math
import sys


def load(runs, spec, row):
    arm, prog = spec.split("/")
    base = f"{runs}/{arm}/taps/{prog}-r{row}"
    data = array.array("f")
    with open(base + ".f32", "rb") as f:
        data.frombytes(f.read())
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


def main():
    runs, row, ref, *others = sys.argv[1:]
    row = int(row)
    R = load(runs, ref, row)
    O = [(s, load(runs, s, row)) for s in others]
    layers = sorted({k[1] for k in R if k[0] == "mix"})
    print(f"row {row}, reference {ref}; columns: mix / out (x1+ffn) rel L2 to reference")
    print("layer " + " ".join(f"{s:>30}" for s, _ in O))
    for il in layers:
        cells = []
        for _, T in O:
            if ("mix", il) not in T:
                cells.append(f"{'-':>30}")
                continue
            m = rel(T[("mix", il)], R[("mix", il)])
            ro = [a + b for a, b in zip(R[("x1", il)], R[("ffn", il)])]
            to = [a + b for a, b in zip(T[("x1", il)], T[("ffn", il)])]
            cells.append(f"{m:14.3e} {rel(to, ro):14.3e} ")
        print(f"{il:5} " + " ".join(cells))
    lr = R[("logits", -1)]
    for s, T in O:
        lt = T[("logits", -1)]
        mx = max(abs(a - b) for a, b in zip(lt, lr))
        print(f"logits {s}: rel {rel(lt, lr):.3e} maxabs {mx:.4f}")


main()
