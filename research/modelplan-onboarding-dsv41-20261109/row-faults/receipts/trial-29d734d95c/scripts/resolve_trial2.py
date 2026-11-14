#!/usr/bin/env python3
"""Resolve the conflicts of replaying memra#1315's review commit (the rowfault gate's --devices) onto the
trial tree, in the gate's usage text (diff3 markers). Both hunks are an edit of this change's own earlier
text, which the landing head's side carries inside a longer text:

- a doc hunk: the base's lines are replaced by this change's lines where they stand in the landing side;
- the one-line usage string: the base-to-change edit (what differs between their common prefix and
  suffix) is applied to the landing side's string, which must hold the base's edited span exactly once.
usage: resolve_trial2.py REPO_DIR
"""
import sys

root = sys.argv[1]
path = "crates/memra-engine/src/bin/dsv41_gate.rs"
lines = open(f"{root}/{path}", encoding="utf-8").read().split("\n")
out, i, n = [], 0, 0
while i < len(lines):
    if not lines[i].startswith("<<<<<<< "):
        out.append(lines[i])
        i += 1
        continue
    ours, base, theirs = [], [], []
    cur = ours
    i += 1
    while not lines[i].startswith(">>>>>>> "):
        if lines[i].startswith("||||||| "):
            cur = base
        elif lines[i] == "=======":
            cur = theirs
        else:
            cur.append(lines[i])
        i += 1
    i += 1
    n += 1
    if len(ours) == len(base) == len(theirs) == 1:
        o, b, t = ours[0], base[0], theirs[0]
        p = 0
        while p < min(len(b), len(t)) and b[p] == t[p]:
            p += 1
        s = 0
        while s < min(len(b), len(t)) - p and b[-1 - s] == t[-1 - s]:
            s += 1
        # widen the span with context so it is found once in the landing side
        ctx = 40
        old = b[max(0, p - ctx) : len(b) - s]
        new = t[max(0, p - ctx) : len(t) - s]
        if o.count(old) != 1:
            sys.exit(f"hunk {n}: the edited span is in the landing side {o.count(old)} times")
        out.append(o.replace(old, new))
    else:
        k = len(base)
        at = [j for j in range(len(ours) - k + 1) if ours[j : j + k] == base]
        if len(at) != 1:
            sys.exit(f"hunk {n}: the base's lines stand in the landing side {len(at)} times")
        out += ours[: at[0]] + theirs + ours[at[0] + k :]
open(f"{root}/{path}", "w", encoding="utf-8").write("\n".join(out))
print(f"{path}: {n} hunk(s) resolved")
