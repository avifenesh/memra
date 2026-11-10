#!/usr/bin/env python3
"""The acceptance cell (pk_ab2.py MODE=acc): per sampler, acceptance per drafted token and
decode rate over 16 prompts, base against candidate, paired by prompt.
usage: pk_acc.py RUNS_JSONL [BASE CAND]"""
import json, math, statistics, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
base, cand = (sys.argv[2], sys.argv[3]) if len(sys.argv) > 3 else ("base", "cand2")
print(f"requests: {len(rows)}, errors: {sum(1 for r in rows if r['error'])}")
print("| sampler | prompts | acceptance base | acceptance cand | paired difference, mean +- SE | decode base (tok/s, mean) | decode cand | decode ratio |")
print("|---|---:|---:|---:|---|---:|---:|---:|")
for cls, name in (("t1-acc", "T=1"), ("chat-acc", "T=0.7, top-p 0.95")):
    prompts = sorted({r["prompt"] for r in rows if r["class"] == cls})
    def per(arm, key):
        out = {}
        for p in prompts:
            v = [r[key] for r in rows if r["class"] == cls and r["arm"] == arm and r["prompt"] == p and r[key] is not None and not r["error"]]
            if v: out[p] = statistics.mean(v)
        return out
    ab, ak = per(base, "acceptance"), per(cand, "acceptance")
    both = [p for p in prompts if p in ab and p in ak]
    d = [ak[p] - ab[p] for p in both]
    se = statistics.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
    db, dk = per(base, "decode_tok_s"), per(cand, "decode_tok_s")
    mb, mk = statistics.mean(db[p] for p in both), statistics.mean(dk[p] for p in both)
    ks = sorted({(r["k"] or ["?"])[0] for r in rows if r["class"] == cls})
    print(f"| {name} (K served: {', '.join(ks)}) | {len(both)} | {statistics.mean(ab[p] for p in both):.3f} | {statistics.mean(ak[p] for p in both):.3f} | "
          f"{statistics.mean(d):+.3f} +- {se:.3f} | {mb:.2f} | {mk:.2f} | {mk / mb:.3f} |")
    same = sum(1 for p in both if len({r["sha"] for r in rows if r["class"] == cls and r["prompt"] == p and r["arm"] == cand}) == 1)
    print(f"|   texts: candidate writes one text per prompt across its boots on {same} of {len(both)} | | | | | | | |")
