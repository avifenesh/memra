#!/usr/bin/env python3
"""pipe_tab.py <served-root>: MEMRA_DSV4_SESSIONS A/B rows (r<i>-s<N>/cells.jsonl).
Per cell: per-row table, per-arm medians with N, delta s2 vs s1, hash identity across all rows
of both arms (each request's text sha, in request order), thermal regime."""
import csv, glob, json, os, statistics as st, sys
root = sys.argv[1]
rows = sorted((r for r in glob.glob(os.path.join(root, "r*-s*")) if os.path.isdir(r)
               and os.path.exists(os.path.join(r, "cells.jsonl"))),
              key=lambda r: int(os.path.basename(r)[1:].split("-")[0]))
def cells(r):
    return {json.loads(l)["label"]: json.loads(l) for l in open(os.path.join(r, "cells.jsonl"))}
def reqs(r):
    out = {}
    for l in open(os.path.join(r, "cells.jsonl.req.jsonl")):
        q = json.loads(l)
        out.setdefault(q["label"], []).append(q)
    return {k: tuple(x["sha"] for x in sorted(v, key=lambda q: q["idx"])) for k, v in out.items()}
labels = [l for l in cells(rows[0]) if l != "warmup"]
data = {r: cells(r) for r in rows}
hashes = {r: reqs(r) for r in rows}
for lab in labels:
    print(f"\n#### {lab}\n")
    print("| row | arm | agg tok/s | decode p50 tok/s | TPOT p50 / p95 / p99 ms | ITL p50 / p99 ms | TTFT p50 / p95 ms | E2E p50 / p95 ms | ok |")
    print("|---|---|---|---|---|---|---|---|---|")
    per = {}
    for r in rows:
        c = data[r][lab]; name = os.path.basename(r); arm = name.split("-")[1]
        print(f"| {name.split('-')[0]} | {arm} | {c['agg_tok_s']:.2f} | {c['decode_tok_s_p50']:.2f} | {c['tpot_ms_p50']:.2f} / {c['tpot_ms_p95']:.2f} / {c['tpot_ms_p99']:.2f} | {c['itl_ms_p50']:.2f} / {c['itl_ms_p99']:.2f} | {c['ttft_ms_p50']:,.0f} / {c['ttft_ms_p95']:,.0f} | {c['e2e_ms_p50']:,.0f} / {c['e2e_ms_p95']:,.0f} | {c['n_ok']}/{c['n_req']} |")
        per.setdefault(arm, []).append(c)
    print("\n| arm | N | agg median | decode p50 median | TPOT p50 median ms | TTFT p50 median ms |")
    print("|---|---|---|---|---|---|")
    med = {}
    for arm in sorted(per):
        v = per[arm]
        med[arm] = [st.median(x[k] for x in v) for k in ("agg_tok_s", "decode_tok_s_p50", "tpot_ms_p50", "ttft_ms_p50")]
        print(f"| {arm} | {len(v)} | {med[arm][0]:.2f} | {med[arm][1]:.2f} | {med[arm][2]:.2f} | {med[arm][3]:,.0f} |")
    if "s1" in med and "s2" in med:
        print(f"\nAggregate s2 vs s1: {100*(med['s2'][0]/med['s1'][0]-1):+.1f}%; per-request decode {100*(med['s2'][1]/med['s1'][1]-1):+.1f}%.")
    hs = {hashes[r].get(lab) for r in rows}
    print(f"\nHashes: {'identical across every row of both arms' if len(hs)==1 else 'DIFFER: '+str(len(hs))+' distinct tuples'}; {len(next(iter(hs)))} requests, first {' '.join(h[:8] for h in next(iter(hs)))}.")
th = {}
for r in rows:
    arm = os.path.basename(r).split("-")[1]
    p = os.path.join(r, "telemetry-250ms.csv")
    pw, clk, tmp = [], [], []
    with open(p) as f:
        rd = csv.reader(f); next(rd, None)
        for x in rd:
            try:
                if float(x[2].split()[0]) >= 20:
                    pw.append(float(x[4].split()[0])); clk.append(float(x[5].split()[0])); tmp.append(float(x[7]))
            except (ValueError, IndexError):
                pass
    if pw:
        th.setdefault(arm, []).append((st.median(pw), max(pw), min(clk), max(clk), max(tmp), len(pw)))
for arm, ts in sorted(th.items()):
    print(f"\nThermal {arm}: power median {min(t[0] for t in ts):.0f}..{max(t[0] for t in ts):.0f} W, peak {max(t[1] for t in ts):.0f} W, SM clock {min(t[2] for t in ts):.0f}..{max(t[3] for t in ts):.0f} MHz, max temp {max(t[4] for t in ts):.0f} C, {sum(t[5] for t in ts)} samples.")
