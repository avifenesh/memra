#!/usr/bin/env python3
"""Compare two row files of the same gate or cell, row by row (memra#1186).

Rows are matched on the key fields both files carry, of: case, context, prompt, id, mode,
class, kind. Per shared row: the text sha, and where both rows carry them the completion tokens, the
finish reason, the prompt and cached tokens and the speculative counts (acceptance, rounds,
drafted, accepted). The two
binaries of this lane must write the same texts; they do not round alike, so the speculative
counts are reported, not judged.

usage: rowcmp.py BASE.jsonl CAND.jsonl [--key case,id] [--same rounds,acceptance]
exit 0 when every shared row has the same sha (and the same --same fields) and no row is missing.
"""
import json
import sys

args = [a for a in sys.argv[1:] if not a.startswith("--")]
opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
KEYS = ("case", "context", "prompt", "id", "mode", "class", "kind")
FIELDS = ("sha", "completion_tokens", "prompt_tokens", "cached", "finish", "acceptance", "rounds", "drafted", "accepted")


def load(path):
    return [json.loads(line) for line in open(path) if line.strip()]


a, b = load(args[0]), load(args[1])
keys = opts["key"].split(",") if "key" in opts else [k for k in KEYS if all(k in r for r in a + b)]
must = ["sha"] + (opts["same"].split(",") if "same" in opts else [])


def index(rows):
    out = {}
    for r in rows:
        out.setdefault(tuple(str(r[k]) for k in keys), []).append(r)
    return out


ia, ib = index(a), index(b)
shared = sorted(set(ia) & set(ib))
counts = {f: [0, 0] for f in FIELDS}
bad = []
for k in shared:
    for ra, rb in zip(ia[k], ib[k]):
        for f in FIELDS:
            if f in ra and f in rb:
                counts[f][1] += 1
                if ra[f] == rb[f]:
                    counts[f][0] += 1
                elif f in must:
                    bad.append(f"{'/'.join(k)}: {f} {ra[f]} against {rb[f]}")
only = sorted(set(ia) ^ set(ib))
print(f"rows: {len(a)} and {len(b)}, matched on ({', '.join(keys)}): {len(shared)} keys")
for f in FIELDS:
    if counts[f][1]:
        print(f"  same {f}: {counts[f][0]} of {counts[f][1]}")
for line in bad:
    print("  DIFF " + line)
if only:
    print("  in one file only: " + ", ".join("/".join(k) for k in only))
sys.exit(1 if bad or only or not shared else 0)
