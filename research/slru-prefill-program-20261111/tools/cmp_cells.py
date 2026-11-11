#!/usr/bin/env python3
"""Compare step-1 cells: probe table rows and the layer-0/1 MoE dumps (input and output bytes).

usage: cmp_cells.py <root> <cellA> <cellB> [...]
For each pair (first cell vs each other): rows whose prefill/decode columns differ, and for
every dumped (layer, t) whether the MoE INPUT bytes match and whether the OUTPUT bytes match,
with the count of differing f32 elements and the max abs difference.
"""
import pathlib
import re
import struct
import sys

root = pathlib.Path(sys.argv[1])
cells = sys.argv[2:]


def table(cell):
    rows = {}
    for line in (root / cell / "probe.log").read_text().splitlines():
        if re.match(r"^\d{4} ", line):
            f = line.split()
            rows[int(f[0])] = f[1:]
    return rows


def f32(path):
    b = path.read_bytes()
    return struct.unpack(f"<{len(b) // 4}f", b), b


def arms(cell):
    out = {}
    for line in (root / cell / "probe.log").read_text().splitlines():
        m = re.match(r"\[diag1217\] il=(\d+) t=(\d+) arm=(\S+)", line)
        if m:
            out[(int(m[1]), int(m[2]))] = m[3]
    return out


base = cells[0]
tb = table(base)
ab = arms(base)
for other in cells[1:]:
    to = table(other)
    ao = arms(other)
    pre = sum(1 for k in tb if k in to and tb[k][0:2] != to[k][0:2])
    dec = sum(1 for k in tb if k in to and tb[k][2:4] != to[k][2:4])
    print(f"## {base} vs {other}: rows={len(tb)} prefill-cols-differ={pre} decode-cols-differ={dec}")
    for key in sorted(ab):
        il, t = key
        tag = f"il{il}-t{t}"
        pa, po = root / base / "dump", root / other / "dump"
        try:
            ia, iab = f32(pa / f"{tag}-in.bin")
            io, iob = f32(po / f"{tag}-in.bin")
            oa, oab = f32(pa / f"{tag}-out.bin")
            oo, oob = f32(po / f"{tag}-out.bin")
        except FileNotFoundError:
            print(f"  {tag}: missing dump")
            continue
        ndiff = sum(1 for x, y in zip(oa, oo) if struct.pack("<f", x) != struct.pack("<f", y))
        mx = max((abs(x - y) for x, y in zip(oa, oo)), default=0.0)
        first = next((i for i, (x, y) in enumerate(zip(oa, oo)) if x != y), None)
        print(
            f"  {tag}: arms {ab[key]} / {ao.get(key)}  input bytes equal={iab == iob}  "
            f"output bytes equal={oab == oob}  differing={ndiff}/{len(oa)}  maxabs={mx:.3e}"
            + (f"  first@{first}: {oa[first]!r} vs {oo[first]!r}" if first is not None else "")
        )
