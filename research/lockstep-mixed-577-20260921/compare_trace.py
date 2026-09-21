#!/usr/bin/env python3
"""First divergent (step, layer) between two MEMRA_MOE_INPUT_TRACE_DIR captures of stream 0.

usage: compare_trace.py TRACE_M1_DIR TRACE_MK_DIR PROMPT0_LEN
The M=1 capture holds one row per call (prompt tokens, then one per lockstep step); the M=k
capture holds k rows per lockstep call (stream 0 = row 0) after every stream's own prompt rows
(tokens == 1 entries). Compares stream 0's MoE input per layer per lockstep step, bitwise."""
import json
import os
import sys

import numpy as np


def entries(d):
    by_layer = {}
    with open(os.path.join(d, "index.jsonl")) as f:
        for line in f:
            e = json.loads(line)
            by_layer.setdefault(e["layer"], []).append(e)
    return by_layer


def row(d, e, r):
    n = e["hidden_size"]
    off = e["offset"] + r * n * 4
    with open(os.path.join(d, e["file"]), "rb") as f:
        f.seek(off)
        return np.frombuffer(f.read(n * 4), dtype="<f4")


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    d1, dk, p0 = sys.argv[1], sys.argv[2], int(sys.argv[3])
    e1, ek = entries(d1), entries(dk)
    layers = sorted(set(e1) & set(ek))
    first = None
    for il in layers:
        steps1 = [e for e in e1[il] if e["tokens"] == 1][p0:]
        stepsk = [e for e in ek[il] if e["tokens"] > 1]
        for s, (a, b) in enumerate(zip(steps1, stepsk)):
            ra, rb = row(d1, a, 0), row(dk, b, 0)
            if not np.array_equal(ra.view("<u4"), rb.view("<u4")):
                d = np.abs(ra.astype(np.float64) - rb.astype(np.float64))
                if first is None or (s, il) < first[:2]:
                    first = (s, il, float(d.max()), int((ra.view("<u4") != rb.view("<u4")).sum()))
                break
    if first is None:
        print(f"stream 0 MoE inputs identical over {len(layers)} layers")
        return 0
    s, il, mx, n = first
    print(f"first divergent stream-0 MoE input: lockstep step {s}, layer {il} (max|d| {mx:.3e}, {n} elements differ)")
    diverged0 = []
    for il in layers:
        a = [e for e in e1[il] if e["tokens"] == 1][p0:]
        b = [e for e in ek[il] if e["tokens"] > 1]
        if a and b and not np.array_equal(row(d1, a[0], 0).view("<u4"), row(dk, b[0], 0).view("<u4")):
            diverged0.append(il)
    print(f"layers whose lockstep step-0 input already differs: {diverged0[:6]}{'...' if len(diverged0) > 6 else ''} ({len(diverged0)} of {len(layers)})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
