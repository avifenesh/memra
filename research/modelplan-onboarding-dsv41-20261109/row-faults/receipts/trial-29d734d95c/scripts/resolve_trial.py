#!/usr/bin/env python3
"""Resolve the conflicts of replaying memra#1315 onto the memra#1286 landing head (diff3 markers) in the
three places they meet: the gate's usage text (module doc and the usage string), docs/FLAGS.md and
docs/KERNELS.md.

- A table hunk (FLAGS, KERNELS) is merged row by row, keyed on the row's first cell: a row only one side
  changed takes that side, a row one side deleted and the other left alone stays deleted, a row only one
  side added is kept. A row both sides changed stops the script.
- The gate's doc hunk keeps both sides' added lines, the landing head's first.
- The usage string takes the landing head's and appends this change's subcommands (the text after the
  base's string in this change's line).
usage: resolve_trial.py REPO_DIR
"""
import re
import sys

root = sys.argv[1]


def hunks(text):
    """Split into ('text', str) and ('hunk', ours, base, theirs) parts."""
    out, lines, i = [], text.split("\n"), 0
    buf = []
    while i < len(lines):
        if lines[i].startswith("<<<<<<< "):
            out.append(("text", buf))
            buf = []
            ours, base, theirs, cur = [], [], [], None
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
            out.append(("hunk", ours, base, theirs))
        else:
            buf.append(lines[i])
        i += 1
    out.append(("text", buf))
    return out


def key(row):
    return row.split("|")[1].strip() if row.startswith("|") else row


def merge_rows(ours, base, theirs, where):
    b = {key(r): r for r in base}
    o = {key(r): r for r in ours}
    t = {key(r): r for r in theirs}
    res = []
    for r in ours:
        k = key(r)
        if k in b and k in t:
            if t[k] != b[k] and r != b[k] and t[k] != r:
                sys.exit(f"{where}: both sides changed the row {k[:80]!r}")
            res.append(t[k] if t[k] != b[k] else r)
        elif k in b and k not in t:
            if r != b[k]:
                sys.exit(f"{where}: this change deleted a row the landing head changed: {k[:80]!r}")
        else:
            res.append(r)  # the landing head's own new row (or one both added)
    # rows this change added: after the row that precedes them on its side
    for i, r in enumerate(theirs):
        k = key(r)
        if k in b or k in o:
            continue
        prev = key(theirs[i - 1]) if i else None
        at = next((j + 1 for j, x in enumerate(res) if key(x) == prev), len(res))
        res.insert(at, r)
    for k in b:
        if k not in o and k in t and t[k] != b[k]:
            sys.exit(f"{where}: the landing head deleted a row this change changed: {k[:80]!r}")
    return res


def resolve(path, fn):
    p = f"{root}/{path}"
    parts = hunks(open(p, encoding="utf-8").read())
    out, n = [], 0
    for part in parts:
        if part[0] == "text":
            out += part[1]
        else:
            n += 1
            out += fn(part[1], part[2], part[3], f"{path} hunk {n}")
    open(p, "w", encoding="utf-8").write("\n".join(out))
    print(f"{path}: {n} hunk(s) resolved")


def gate(ours, base, theirs, where):
    if all(x.startswith("//!") for x in ours + base + theirs):
        added = [x for x in theirs if x not in base]
        return ours + added
    if len(ours) == len(base) == len(theirs) == 1 and "usage: dsv41-gate" in ours[0]:
        end = '".into()),'
        b, t, o = base[0], theirs[0], ours[0]
        if not (b.endswith(end) and t.endswith(end) and o.endswith(end) and t.startswith(b[: -len(end)])):
            sys.exit(f"{where}: the usage strings are not base + appended subcommands")
        return [o[: -len(end)] + t[len(b) - len(end) :]]
    sys.exit(f"{where}: an unexpected conflict shape")


resolve("crates/memra-engine/src/bin/dsv41_gate.rs", gate)
resolve("docs/FLAGS.md", merge_rows)
resolve("docs/KERNELS.md", merge_rows)
