#!/usr/bin/env python3
"""Mint the own-gen-headed blend at an arbitrary width from the cached corpus counts.

The sweep in cell A2 is still RISING at 32,768 (81.88 -> 86.26 -> 87.12 -> 88.20 tok/s),
and the estimator puts the interior knee just past it (thinkon 1.052 at 32,768 vs 1.048 at
49,152). A width table that stops at the widest file it happens to have has inherited the
width, not chosen it — so this writes the wider classes the bracketing sweep needs. Same
blend, same rank law, no tokenization (counts come from rank_ranks.py's cache).

usage: mint_width.py <N> [<N>...]
"""
import gzip, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = os.path.dirname(HERE)
VOCAB_ROWS = 248320

counts = json.load(gzip.open(os.path.join(HERE, "counts-cache.json.gz"), "rt"))
og = {}
with gzip.open(os.path.join(SPEC, "mtp10", "ranks-owngen-big.txt.gz"), "rt") as f:
    for line in f:
        if line.startswith("#") or not line.strip():
            continue
        i, c = line.split("\t")[:2]
        og[int(i)] = int(c)

def norm(d):
    t = sum(d.values())
    return {int(i): c / t for i, c in d.items()}

fa, fp, fo = norm(counts["agentic"]), norm(counts["prose"]), norm(og)
mixed = {i: 0.5 * fa.get(i, 0.0) + 0.5 * fp.get(i, 0.0) for i in set(fa) | set(fp)}
blend = {i: 0.5 * fo.get(i, 0.0) + 0.5 * mixed.get(i, 0.0) for i in set(fo) | set(mixed)}
id_space = 248077
order = sorted(range(id_space), key=lambda i: (-blend.get(i, 0.0), i))
for n in (int(a) for a in sys.argv[1:]):
    ids = order[:n]
    assert len(ids) == n == len(set(ids)) and max(ids) < VOCAB_ROWS
    out = os.path.join(HERE, f"q4e-ranks-ogblend-{n}.txt")
    open(out, "w").write("".join(f"{i}\n" for i in ids))
    print(f"wrote {out} ({n} ids)")
