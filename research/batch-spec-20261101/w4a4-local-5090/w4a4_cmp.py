#!/usr/bin/env python3
"""Compare two mmvq-b16-bench output dumps (little-endian f32): relative L2 error, max abs error
relative to the reference's max abs, and cosine. Usage: w4a4_cmp.py REF.bin ARM.bin"""
import array
import math
import sys

a = array.array("f"); a.frombytes(open(sys.argv[1], "rb").read())
b = array.array("f"); b.frombytes(open(sys.argv[2], "rb").read())
assert len(a) == len(b), (len(a), len(b))
num = sum((x - y) ** 2 for x, y in zip(a, b))
den = sum(x * x for x in a)
dot = sum(x * y for x, y in zip(a, b))
nb = sum(y * y for y in b)
mx = max(abs(x) for x in a)
md = max(abs(x - y) for x, y in zip(a, b))
nan = sum(1 for y in b if y != y)
print(f"n={len(a)} rel_l2={math.sqrt(num / den):.4e} max_abs_err/max_ref={md / mx:.4e} "
      f"cos={dot / math.sqrt(den * nb):.6f} nan={nan}")
