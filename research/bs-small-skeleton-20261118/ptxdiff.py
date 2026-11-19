#!/usr/bin/env python3
"""Diff one kernel's PTX between two files, with virtual register numbers masked.
usage: ptxdiff.py A.ptx B.ptx NAME_SUBSTR_A NAME_SUBSTR_B [maxlines]
"""
import difflib
import re
import sys


def entry(path, sub):
    txt = open(path).read()
    for m in re.finditer(r"\.visible \.entry (\S+)\(", txt):
        if sub in m.group(1):
            start = m.start()
            end = txt.find("\n}\n", start)
            return m.group(1), txt[start:end].split("\n")
    raise SystemExit(f"no entry with {sub} in {path}")


def norm(lines):
    out = []
    for l in lines:
        l = l.strip()
        if not l or l.startswith("//"):
            continue
        l = re.sub(r"%(r|rd|f|p|rs|fd|h|hh|b)\d+", r"%\1", l)
        l = re.sub(r"\$L__BB\d+_\d+", "$L", l)
        l = re.sub(r"_Z\w+_param_(\d+)", r"P\1", l)
        l = re.sub(r"\b(v41_ds_sm|v41_sx_sm|v41_sb_sm|_ZN2bs11bs_dense_smE|_ZN2bs9bs_moe_smE|_ZN2bs9bs_moe_smE)\b", "SM", l)
        out.append(l)
    return out


na, a = entry(sys.argv[1], sys.argv[3])
nb, b = entry(sys.argv[2], sys.argv[4])
mx = int(sys.argv[5]) if len(sys.argv) > 5 else 80
a, b = norm(a), norm(b)
d = list(difflib.unified_diff(a, b, lineterm="", n=2))
print(na, len(a), "|", nb, len(b), "| changed lines", sum(1 for l in d if l[:1] in "+-" and l[:3] not in ("+++", "---")))
print("\n".join(d[:mx]))
