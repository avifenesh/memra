#!/usr/bin/env python3
"""union_resolve.py FILE...: resolve diff3 conflict blocks by keeping both sides (ours, then
theirs), dropping the base. For append-only files (research/INDEX.md, .gitattributes)."""
import sys
for p in sys.argv[1:]:
    out, mode = [], None
    for l in open(p).read().split("\n"):
        if l.startswith("<<<<<<< "): mode = "ours"; continue
        if l.startswith("||||||| "): mode = "base"; continue
        if l.startswith("=======") and mode in ("ours", "base"): mode = "theirs"; continue
        if l.startswith(">>>>>>> "): mode = None; continue
        if mode == "base": continue
        out.append(l)
    open(p, "w").write("\n".join(out))
    print("resolved", p)
