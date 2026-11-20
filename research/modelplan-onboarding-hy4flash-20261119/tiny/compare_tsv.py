"""Compare two tiny-oracle TSVs row by row: rows whose f32 bits differ, the largest absolute
difference and the row it sits on. usage: python3 compare_tsv.py A.tsv[.gz] B.tsv[.gz]"""
import gzip
import struct
import sys


def load(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        lines = f.read().split("\n")[1:]
    return [(tuple(l.split("\t")[:3]), l.split("\t")[3]) for l in lines if l]


a, b = load(sys.argv[1]), load(sys.argv[2])
assert len(a) == len(b), (len(a), len(b))
bits, worst = 0, (0.0, None)
for (ia, ba), (ib, bb) in zip(a, b):
    assert ia == ib, (ia, ib)
    if ba != bb:
        bits += 1
        va, vb = (struct.unpack(">f", bytes.fromhex(x))[0] for x in (ba, bb))
        if abs(va - vb) > worst[0]:
            worst = (abs(va - vb), (ia, va, vb))
print(f"rows {len(a)} differing_f32_bits {bits} max_abs {worst[0]:.3e} at {worst[1]}")
