#!/usr/bin/env python3
"""Summary of a draft-length cell (pk_k9.py): one arm against another, per prompt.
usage: pk_k_summary.py DIR BASE_ARM CAND_ARM   (DIR holds runs.jsonl, groups.jsonl when two at once)"""
import collections
import json
import math
import pathlib
import statistics
import sys

D = pathlib.Path(sys.argv[1])
BASE, CAND = sys.argv[2], sys.argv[3]
rows = [json.loads(l) for l in open(D / "runs.jsonl") if l.strip()]
boots = {a: sorted({r["boot"] for r in rows if r["arm"] == a}) for a in (BASE, CAND)}
classes = []
for r in rows:
    if r["class"] not in classes:
        classes.append(r["class"])
print(f"boots: {BASE} {boots[BASE]}, {CAND} {boots[CAND]}; requests {len(rows)}, errors {sum(1 for r in rows if r.get('error'))}")


def per_boot(arm, cls, key):
    return [next(r[key] for r in rows if r["arm"] == arm and r["boot"] == b and r["class"] == cls) for b in boots[arm]]


def u_count(a, b):
    return sum(1.0 if x > y else 0.5 if x == y else 0.0 for x in a for y in b)


def geo(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


pairs = len(boots[BASE]) * len(boots[CAND])
print(f"| class | {BASE} tok/s (K) | {CAND} tok/s (K) | {CAND} over {BASE} | wins of {pairs} | acceptance {BASE} | {CAND} | TTFT {BASE} s | {CAND} s | texts equal |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
ratios = collections.defaultdict(list)
wins = collections.defaultdict(float)
for c in classes:
    v = {a: per_boot(a, c, "decode_tok_s") for a in (BASE, CAND)}
    m = {a: statistics.median(v[a]) for a in (BASE, CAND)}
    ks = {a: "/".join(sorted({(r["k"] or ["?"])[0] for r in rows if r["arm"] == a and r["class"] == c})) for a in (BASE, CAND)}
    acc = {a: statistics.median([x for x in per_boot(a, c, "acceptance") if x is not None] or [float("nan")]) for a in (BASE, CAND)}
    tt = {a: statistics.median(per_boot(a, c, "ttft_s")) for a in (BASE, CAND)}
    shas = {r["sha"] for r in rows if r["class"] == c}
    w = u_count(v[CAND], v[BASE])
    sampler = c.split("-")[0]
    ratios[sampler].append(m[CAND] / m[BASE])
    wins[sampler] += w
    print(f"| {c} | {m[BASE]:.2f} ({ks[BASE]}) | {m[CAND]:.2f} ({ks[CAND]}) | {m[CAND] / m[BASE]:.3f} | {w:g} | "
          f"{acc[BASE]:.3f} | {acc[CAND]:.3f} | {tt[BASE]:.3f} | {tt[CAND]:.3f} | {'yes' if len(shas) == 1 else 'no'} |")
for s, rs in ratios.items():
    print(f"{s}: geometric mean {geo(rs):.3f} over {len(rs)} prompts (per prompt {', '.join(f'{x:.3f}' for x in rs)}), "
          f"{wins[s]:g} of {pairs * len(rs)} boot pairs")
allr = [x for rs in ratios.values() for x in rs]
print(f"all {len(allr)} prompts: geometric mean {geo(allr):.3f}, {sum(1 for x in allr if x > 1)} prompts ahead, "
      f"{sum(wins.values()):g} of {pairs * len(allr)} boot pairs")
gpath = D / "groups.jsonl"
if gpath.exists():
    groups = [json.loads(l) for l in open(gpath) if l.strip()]
    print(f"| pair | {BASE} aggregate tok/s | {CAND} | {CAND} over {BASE} | wins of {pairs} |")
    print("|---|---:|---:|---:|---:|")
    gr = []
    gw = 0.0
    for gi in sorted({g["group"] for g in groups}):
        v = {a: [next(g["aggregate_tok_s"] for g in groups if g["arm"] == a and g["boot"] == b and g["group"] == gi) for b in boots[a]] for a in (BASE, CAND)}
        m = {a: statistics.median(v[a]) for a in (BASE, CAND)}
        w = u_count(v[CAND], v[BASE])
        gr.append(m[CAND] / m[BASE])
        gw += w
        name = "+".join(next(g["classes"] for g in groups if g["group"] == gi))
        print(f"| {name} | {m[BASE]:.2f} | {m[CAND]:.2f} | {m[CAND] / m[BASE]:.3f} | {w:g} |")
    print(f"pairs: geometric mean {geo(gr):.3f}, {gw:g} of {pairs * len(gr)} boot pairs")
