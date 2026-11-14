#!/usr/bin/env python3
"""index_resolve.py FILE ROW_PREFIX: union-resolve research/INDEX.md conflict blocks (ours, then
theirs), then keep only the last line starting with ROW_PREFIX (the replayed commit's version of
this lane's row), in the position of the first."""
import sys
p, prefix = sys.argv[1], sys.argv[2]
out, mode = [], None
for l in open(p).read().split("\n"):
    if l.startswith("<<<<<<< "): mode = "ours"; continue
    if l.startswith("||||||| "): mode = "base"; continue
    if l.startswith("=======") and mode in ("ours", "base"): mode = "theirs"; continue
    if l.startswith(">>>>>>> "): mode = None; continue
    if mode == "base": continue
    out.append(l)
rows = [i for i, l in enumerate(out) if l.startswith(prefix)]
if len(rows) > 1:
    last = out[rows[-1]]
    out = [l for i, l in enumerate(out) if i not in rows[1:]]
    out[rows[0]] = last
open(p, "w").write("\n".join(out))
print("resolved", p, "rows", len(rows))
