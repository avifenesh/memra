#!/usr/bin/env python3
"""Lane S: acceptance a(P) against the offset from the generation start, and the tokens around the
chat template boundary, from ink2-spec --per-position dumps. usage: boundary_probe.py CKPT PP_DIR"""
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(sys.argv[1])
pp = Path(sys.argv[2])
buckets = {}
for jf in sorted(pp.glob("*.json")):
    if jf.name.endswith(".oracle.json"):
        continue
    m = json.load(open(jf))
    seq, T0 = m["seq"], m["first_anchor"]
    rows = []
    for i, props in enumerate(m["proposals"]):
        P = T0 + i
        a = 0
        for j, d in enumerate(props):
            if P + 1 + j >= len(seq) or d != seq[P + 1 + j]:
                break
            a += 1
        rows.append(a)
        b = "0-3" if i < 4 else "4-15" if i < 16 else "16+"
        buckets.setdefault((m["kind"], b), []).append(a)
    if m["kind"] == "chat":
        print("==", m["id"], "prompt tail:", repr(tok.decode(seq[T0 - 6:T0])))
        for i in range(min(24, len(rows))):
            P = T0 + i
            print(f"  +{i:2d} anchor {repr(tok.decode([seq[P]])):24s} a={rows[i]} drafts {repr(tok.decode(m['proposals'][i][:4]))[:60]:60s} target {repr(tok.decode(seq[P + 1:P + 5]))[:60]}")
print("kind\toffset\tpositions\tmean a(P)")
for (k, b), v in sorted(buckets.items()):
    print(f"{k}\t{b}\t{len(v)}\t{sum(v) / len(v):.3f}")
