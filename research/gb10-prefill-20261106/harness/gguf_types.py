#!/usr/bin/env python3
"""Print a census of GGUF tensor types by name class (stdlib only). usage: gguf_types.py FILE"""
import collections, re, struct, sys

f = open(sys.argv[1], "rb")
assert f.read(4) == b"GGUF"
ver, = struct.unpack("<I", f.read(4))
nt, nkv = struct.unpack("<QQ", f.read(16))


def rstr():
    n, = struct.unpack("<Q", f.read(8)); return f.read(n).decode("utf-8", "replace")


SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def rval(t):
    if t == 8: return rstr()
    if t == 9:
        et, = struct.unpack("<I", f.read(4)); n, = struct.unpack("<Q", f.read(8))
        return [rval(et) for _ in range(n)] if n < 64 else [rval(et) for _ in range(n)][:4]
    fmt = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}[t]
    return struct.unpack("<" + fmt, f.read(SZ[t]))[0]


for _ in range(nkv):
    k = rstr(); t, = struct.unpack("<I", f.read(4)); v = rval(t)
    if "file_type" in k or "quant" in k: print("kv", k, v)
c = collections.Counter(); ex = {}
for _ in range(nt):
    name = rstr(); nd, = struct.unpack("<I", f.read(4))
    dims = struct.unpack("<" + "Q" * nd, f.read(8 * nd)); typ, = struct.unpack("<I", f.read(4)); f.read(8)
    cls = re.sub(r"blk\.\d+\.", "blk.N.", name)
    c[(cls, typ)] += 1; ex[(cls, typ)] = dims
for (cls, typ), n in sorted(c.items()):
    print(f"{cls:40s} type={typ:3d} n={n:3d} dims={ex[(cls, typ)]}")
