#!/usr/bin/env python3
"""Table of the closed-loop cell (normal_cell.py rows) for memra#1227, and its text check.
usage: summ_norm.py RUNDIR [RUNDIR...]   (each holds normal.jsonl)
Arms are row["arm"] with the boot suffix removed. Per (c, arm): phases, the aggregate tok/s of
each phase and their mean, and over all requests of those phases the time to first token (p50,
p95, max) and the inter-token p99. Text check: every (phase index, prompt) has one sha across
all boots and arms. Also counts stall marks, demotions and admissions by draft length."""
import collections
import json
import pathlib
import statistics
import sys


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


rows = [json.loads(l) for d in sys.argv[1:] for l in open(pathlib.Path(d) / "normal.jsonl") if l.strip()]
g = collections.defaultdict(list)
for r in rows:
    g[(r["c"], r["arm"].rsplit("-", 1)[0])].append(r)
print("| c | arm | phases | aggregate tok/s per phase | mean | TTFT p50 s | TTFT p95 s | TTFT max s | inter-token p99 ms | admissions by K | stall marks | demotions |")
print("|---:|---|---:|---|---:|---:|---:|---:|---:|---|---:|---:|")
for (c, arm), rs in sorted(g.items()):
    tt = [x["ttft_s"] for r in rs for x in r["rows"] if x.get("ttft_s") is not None]
    itl = [v for r in rs for x in r["rows"] for v in x.get("itl_ms", [])]
    ks = collections.Counter()
    for r in rs:
        for k in r["spec_k"]:
            ks[f"K{k[0]}" if isinstance(k, (list, tuple)) else str(k)] += 1
    agg = [r["agg_tok_s"] for r in rs]
    print(f"| {c} | {arm} | {len(rs)} | {', '.join(str(a) for a in agg)} | {statistics.fmean(agg):.2f} | {pct(tt, .5):.3f} | {pct(tt, .95):.3f} | {max(tt):.3f} | "
          f"{pct(itl, .99):.1f} | {', '.join(f'{k} x{v}' for k, v in sorted(ks.items()))} | {sum(r['stall_marks'] for r in rs)} | {sum(r['demotions'] for r in rs)} |")
shas = collections.defaultdict(set)
n = 0
for r in rows:
    for x in r["rows"]:
        n += 1
        shas[(r["phase"], r["c"], x["prompt"])].add((x.get("sha"), x.get("completion_tokens")))
bad = {k: v for k, v in shas.items() if len(v) != 1}
errs = sum(r["errors"] for r in rows)
print(f"\ntext check: {n} requests, {len(shas)} (phase, prompt) groups, {len(bad)} with more than one text, {errs} request errors -> {'PASS' if not bad and not errs else 'FAIL'}")
for k, v in sorted(bad.items())[:10]:
    print("  differs:", k, sorted(v))
