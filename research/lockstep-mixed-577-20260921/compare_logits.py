#!/usr/bin/env python3
"""Diff two MEMRA_LOCKSTEP_LOGITS_DUMP files (stream 0, M=1 against M=k) step by step.

usage: compare_logits.py A.f32 B.f32 N_VOCAB [label]
Prints, per step: bit-identical or max |diff|, argmax of each side, top-2 margin of A.
Exit 0 when every step is bit-identical, 1 otherwise."""
import struct
import sys

import numpy as np


def load(path, n_vocab):
    raw = np.fromfile(path, dtype="<f4")
    if raw.size % n_vocab:
        sys.exit(f"{path}: {raw.size} floats is not a multiple of n_vocab={n_vocab}")
    return raw.reshape(-1, n_vocab)


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    a = load(sys.argv[1], int(sys.argv[3]))
    b = load(sys.argv[2], int(sys.argv[3]))
    label = sys.argv[4] if len(sys.argv) > 4 else f"{sys.argv[1]} vs {sys.argv[2]}"
    steps = min(len(a), len(b))
    first = None
    for i in range(steps):
        ra, rb = a[i], b[i]
        ia, ib = int(ra.argmax()), int(rb.argmax())
        srt = np.sort(ra)
        margin = float(srt[-1] - srt[-2])
        same = np.array_equal(ra.view("<u4"), rb.view("<u4"))
        if same:
            print(f"step {i:3d}: bit-identical  argmax {ia}  top2-margin {margin:.6e}")
        else:
            d = np.abs(ra.astype(np.float64) - rb.astype(np.float64))
            n_diff = int((ra.view("<u4") != rb.view("<u4")).sum())
            flag = "" if ia == ib else "  ARGMAX FLIP"
            print(
                f"step {i:3d}: DIFF max|d| {d.max():.3e} at {int(d.argmax())} ({n_diff} of {ra.size} differ)"
                f"  argmax {ia} vs {ib}  top2-margin(A) {margin:.6e}{flag}"
            )
            if first is None:
                first = i
    if len(a) != len(b):
        print(f"length mismatch: {len(a)} vs {len(b)} steps")
    if first is None:
        print(f"{label}: IDENTICAL over {steps} steps")
        return 0
    print(f"{label}: first differing step {first} of {steps}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
