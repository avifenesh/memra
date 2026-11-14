#!/usr/bin/env python3
"""Laptop: the cubins cubin_cmp.py extracted (DIR/x/<tag>/<file>/*.cubin), hashed again after nvcc's per-build-directory
internal-linkage prefix is masked (_INTERNAL_<8 hex>_ and _GLOBAL__N__<8 hex>_; nvcc derives the 8 hex from the
directory it compiles in, the dd_inthash.sh probe). Equal masked sets mean the device code is byte-identical apart
from those symbol-name bytes. usage: cubin_norm.py DIR
"""
import hashlib
import os
import re
import sys

MASK = re.compile(rb"(_INTERNAL_|_GLOBAL__N__)[0-9a-f]{8}_")


def masked(path):
    with open(path, "rb") as fh:
        raw = fh.read()
    return hashlib.sha256(MASK.sub(rb"\1xxxxxxxx_", raw)).hexdigest(), hashlib.sha256(raw).hexdigest()


def load(root, tag, name):
    d = os.path.join(root, "x", tag, name)
    return {f: masked(os.path.join(d, f)) for f in sorted(os.listdir(d)) if f.endswith(".cubin")}


def main():
    root = sys.argv[1]
    ref = {}
    for o in ("dsv41_gpu.o", "dsv41_cutlass.o", "dsv4_gpu.o"):
        for t in ("cfix2", "combo"):
            for m, r in load(root, t, o).values():
                ref.setdefault(m, set()).add(o)
    for name in ("dsv41_gpu.o", "dsv41_cutlass.o", "dsv4_gpu.o", "dsv41-gate", "memra-server"):
        a, b = load(root, "cfix2", name), load(root, "combo", name)
        ma, mb = {m for m, _ in a.values()}, {m for m, _ in b.values()}
        ra, rb = {r for _, r in a.values()}, {r for _, r in b.values()}
        print(f"{name}: raw {len(ra & rb)}/{len(ra)} equal, masked {len(ma & mb)}/{len(ma)} equal")
        for side, only in (("only cfix2", ma - mb), ("only combo", mb - ma)):
            for m in sorted(only):
                print(f"  {side} masked {m[:16]} ({','.join(sorted(ref.get(m, {'-'})))})")
    print("CUBIN-NORM-DONE")


if __name__ == "__main__":
    main()
