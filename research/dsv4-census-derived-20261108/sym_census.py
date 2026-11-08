#!/usr/bin/env python3
"""Per-symbol kernel census of CUDA graph DOT dumps (parser copied from graph_census.py).

  sym_census.py DIR [DIR2]   -> per file: total kernel nodes and every symbol's count;
                                with DIR2, a column per dir and only differing symbols marked.
"""
import re
import sys
from collections import Counter
from pathlib import Path

ID = re.compile(r"\| \{ID \| \d+ \(topoId: \d+\) \| ([^\\<|}]+)")


def name(raw):
    raw = raw.strip()
    m = re.match(r"_Z(\d+)(\w+)", raw)
    if m:
        return m.group(2)[: int(m.group(1))]
    return raw


def census(path):
    c = Counter()
    lines = path.read_text(errors="replace").splitlines()
    for i, line in enumerate(lines):
        if 'label="{KERNEL' in line and i + 1 < len(lines):
            m = ID.search(lines[i + 1])
            c[name(m.group(1)) if m else "?"] += 1
    return c


dirs = [Path(d) for d in sys.argv[1:]]
files = sorted(p.name for p in dirs[0].glob("*.dot"))
tables = {f: [census(d / f) for d in dirs] for f in files}
syms = sorted({k for f in files for c in tables[f] for k in c})
print("symbol".ljust(52) + " ".join(f.replace("full-token-", "").replace(".dot", "").replace("rank", "r").replace("segment", "s").rjust(6) for f in files))
for d_i, d in enumerate(dirs):
    print(f"## {d}")
    print("TOTAL".ljust(52) + " ".join(str(sum(tables[f][d_i].values())).rjust(6) for f in files))
    for s in syms:
        print(s[:52].ljust(52) + " ".join(str(tables[f][d_i][s]).rjust(6) for f in files))
