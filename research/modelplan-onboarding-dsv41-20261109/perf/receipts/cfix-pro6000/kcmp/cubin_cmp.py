#!/usr/bin/env python3
"""Laptop: compare the device code (cubins) of two dev desk builds with cuobjdump 13.1 (no build here).

usage: cubin_cmp.py DIR   with DIR/cfix2 and DIR/combo each holding dsv41_gpu.o, dsv41_cutlass.o, dsv4_gpu.o,
dsv41-gate and memra-server. Prints, per file, whether the cubin sets are equal; for the binaries, every cubin on one
side only, which object it came from (by hash) and its first kernel names.
"""
import hashlib
import os
import re
import subprocess
import sys

CUOBJDUMP = "/usr/local/cuda-13.1/bin/cuobjdump"
FILES = ["dsv41_gpu.o", "dsv41_cutlass.o", "dsv4_gpu.o", "dsv41-gate", "memra-server"]
OBJS = ["dsv41_gpu.o", "dsv41_cutlass.o", "dsv4_gpu.o"]


def cubins(root, tag, name):
    out = os.path.join(root, "x", tag, name)
    os.makedirs(out, exist_ok=True)
    subprocess.run([CUOBJDUMP, "-xelf", "all", os.path.join(root, tag, name)], cwd=out, check=True,
                   stdout=subprocess.DEVNULL)
    got = {}
    for f in sorted(os.listdir(out)):
        if f.endswith(".cubin"):
            with open(os.path.join(out, f), "rb") as fh:
                got.setdefault(hashlib.sha256(fh.read()).hexdigest(), os.path.join(out, f))
    return got


def kernels(path):
    txt = subprocess.run([CUOBJDUMP, "-elf", path], capture_output=True, text=True).stdout
    names = sorted(set(re.findall(r"\.text\.(\S+)", txt)))
    demangled = subprocess.run(["c++filt"], input="\n".join(names), capture_output=True, text=True).stdout.split("\n")
    return len(names), [d.split("(")[0] for d in demangled if d][:4]


def main():
    root = sys.argv[1]
    sets = {(t, f): cubins(root, t, f) for t in ("cfix2", "combo") for f in FILES}
    origin = {}
    for t in ("cfix2", "combo"):
        for o in OBJS:
            for h in sets[(t, o)]:
                origin.setdefault(h, set()).add(f"{t}:{o}")
    print("== cubin sets (cuobjdump -xelf all), cfix2 vs combo")
    for f in FILES:
        a, b = set(sets[("cfix2", f)]), set(sets[("combo", f)])
        state = "same  " if a == b else "DIFFER"
        print(f"{state} {f}: {len(a)} cubins vs {len(b)}, {len(a & b)} shared")
        if a != b and f in ("dsv41-gate", "memra-server"):
            for side, only, t in (("only cfix2", a - b, "cfix2"), ("only combo", b - a, "combo")):
                for h in sorted(only):
                    n, names = kernels(sets[(t, f)][h])
                    src = ",".join(sorted(origin.get(h, {"no compared object"})))
                    print(f"  {side} {h[:16]} from {src}: {n} kernels, e.g. {'; '.join(names)}")
    print("CUBIN-CMP-DONE")


if __name__ == "__main__":
    main()
