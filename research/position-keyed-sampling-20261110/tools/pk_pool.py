#!/usr/bin/env python3
"""Pooled view of a pk_ab.py perf cell: sampled classes against the greedy control.
usage: pk_pool.py RUNS_JSONL [BASE CAND] [--drop-boots 1,11]"""
import json, math, statistics, sys
args = [a for a in sys.argv[1:] if not a.startswith("--drop-boots")]
drop = set()
for a in sys.argv[1:]:
    if a.startswith("--drop-boots="):
        drop = {int(x) for x in a.split("=")[1].split(",")}
rows = [json.loads(l) for l in open(args[0]) if l.strip()]
base, cand = (args[1], args[2]) if len(args) > 2 else ("base", "cand2")
rows = [r for r in rows if r["boot"] not in drop and not r["error"]]
def med(c, arm, key): return statistics.median(r[key] for r in rows if r["class"] == c and r["arm"] == arm)
def mean(c, arm, key): return statistics.mean(r[key] for r in rows if r["class"] == c and r["arm"] == arm)
classes = []
for r in rows:
    if r["class"] not in classes: classes.append(r["class"])
for group, name in ((("t1",), "T=1"), (("chat",), "T=0.7 top-p 0.95"), (("t1", "chat"), "both sampled"), (("greedy",), "greedy control")):
    cs = [c for c in classes if c.split("-")[0] in group]
    ratios = [med(c, cand, "decode_tok_s") / med(c, base, "decode_tok_s") for c in cs]
    gm = math.exp(statistics.mean(math.log(x) for x in ratios))
    ab = statistics.mean(mean(c, base, "acceptance") for c in cs)
    ak = statistics.mean(mean(c, cand, "acceptance") for c in cs)
    print(f"{name}: {len(cs)} prompts, decode ratio geometric mean {gm:.3f} (per prompt {', '.join(f'{x:.3f}' for x in ratios)}), acceptance {ab:.3f} -> {ak:.3f}")
