#!/usr/bin/env python3
"""Token-divergence stats of a non-exact arm against the exact default.
Usage: divergence.py coherence-base.jsonl coherence-arm.jsonl [dbb-base.log dbb-arm.log ...pairs]
- coherence probes (8 prompts at T=0 and T=1 seed 1234, 384 tokens): per record, whether the
  text is identical, and for greedy records the first differing character offset as a share
  of the base text;
- decode-batch-bench pairs: sessions whose 48-token greedy fingerprint matches (TOKFP)."""
import json
import sys


def load(p):
    return [json.loads(l) for l in open(p)]


base, arm = load(sys.argv[1]), load(sys.argv[2])
same = 0
for b, a in zip(base, arm):
    tb, ta = b.get("text", ""), a.get("text", "")
    ident = tb == ta
    same += ident
    k = next((i for i, (x, y) in enumerate(zip(tb, ta)) if x != y), min(len(tb), len(ta)))
    print(f"prompt {b.get('prompt')} {b.get('mode')}: identical={ident} first_diff_char={k} of {len(tb)}")
print(f"coherence records identical: {same}/{len(base)}")
args = sys.argv[3:]
for i in range(0, len(args), 2):
    fp = []
    for p in args[i:i + 2]:
        line = next(l for l in open(p) if l.startswith("TOKFP"))
        fp.append(line.split("[", 1)[1].rstrip("]\n").split())
    n = sum(1 for x, y in zip(*fp) if x == y)
    print(f"{args[i].rsplit('/', 1)[-1]} vs {args[i + 1].rsplit('/', 1)[-1]}: sessions with identical tokens {n}/{len(fp[0])}")
