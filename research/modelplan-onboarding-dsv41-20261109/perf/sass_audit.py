#!/usr/bin/env python3
"""Per-kernel SASS census of a CUDA object: instruction count and the instructions that touch local
memory (LDL / STL: a thread's stack lives in global memory, so each one is a memory round trip).

A kernel that indexes a per-thread array with a runtime value, or takes a local's address in a way
the compiler cannot split, gets local-memory accesses in its inner loop; this lists them.

usage: cuobjdump -sass OBJECT | sass_audit.py [NAME_FILTER]
"""
import re
import sys

flt = sys.argv[1] if len(sys.argv) > 1 else ""
name = None
stats = {}
for line in sys.stdin:
    m = re.search(r"Function : (\S+)", line)
    if m:
        name = m.group(1)
        stats[name] = [0, 0, 0]
        continue
    if name is None:
        continue
    m = re.search(r"/\*[0-9a-f]{4,}\*/\s+([A-Z][A-Z0-9_.]*)", line)
    if not m:
        continue
    op = m.group(1)
    s = stats[name]
    s[0] += 1
    if op.startswith("LDL"):
        s[1] += 1
    elif op.startswith("STL"):
        s[2] += 1
print(f"{'instr':>7} {'LDL':>5} {'STL':>5}  kernel")
for n, (tot, ldl, stl) in sorted(stats.items(), key=lambda x: -(x[1][1] + x[1][2])):
    if flt and flt not in n:
        continue
    print(f"{tot:7d} {ldl:5d} {stl:5d}  {n[:150]}")
