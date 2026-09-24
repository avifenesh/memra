#!/usr/bin/env python3
"""cells_tab.py <campaign-root> <lane-arm> [base-arm]

Markdown for one lane from a served campaign whose rows are <root>/<mode>/r<i>-<arm>/ with
cells.jsonl, cells.jsonl.req.jsonl and telemetry-250ms.csv. Works for any cell labels: per
mode and label, a per-row table (decode, TPOT, TTFT, E2E), per-arm medians with N and the
delta, per-label hash identity across every row of both arms (each request's text sha in
request order), and the thermal regime (samples at >= 20% utilization).
"""
import csv, glob, json, os, statistics as st, sys

root, lane = sys.argv[1], sys.argv[2]
base = sys.argv[3] if len(sys.argv) > 3 else "base"


def rows_of(mdir):
    rows = [r for r in glob.glob(os.path.join(mdir, "r*-*")) if os.path.isdir(r)
            and os.path.exists(os.path.join(r, "cells.jsonl"))]
    rows = [r for r in rows if os.path.basename(r).split("-", 1)[1] in (lane, base)]
    rows.sort(key=lambda r: int(os.path.basename(r)[1:].split("-")[0]))
    return rows


def cells(row):
    return {json.loads(l)["label"]: json.loads(l) for l in open(os.path.join(row, "cells.jsonl"))}


def shas(row):
    out = {}
    for l in open(os.path.join(row, "cells.jsonl.req.jsonl")):
        q = json.loads(l)
        out.setdefault(q["label"], []).append((q["idx"], q.get("sha"), q.get("finish")))
    return {k: tuple(x[1] for x in sorted(v)) for k, v in out.items()}


def thermal(row):
    pw, clk, tmp = [], [], []
    path = os.path.join(row, "telemetry-250ms.csv")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rd = csv.reader(f)
        next(rd, None)
        for r in rd:
            try:
                if float(r[2].split()[0]) >= 20:
                    pw.append(float(r[4].split()[0]))
                    clk.append(float(r[5].split()[0]))
                    tmp.append(float(r[7]))
            except (ValueError, IndexError):
                continue
    return (st.median(pw), max(pw), min(clk), max(clk), max(tmp), len(pw)) if pw else None


for mode in sorted(os.listdir(root)):
    mdir = os.path.join(root, mode)
    if not os.path.isdir(mdir):
        continue
    rows = rows_of(mdir)
    if not rows:
        continue
    data = {r: cells(r) for r in rows}
    hs = {r: shas(r) for r in rows}
    labels = [l for l in data[rows[0]] if l != "warmup"]
    print(f"\n### {mode}\n")
    for lab in labels:
        print(f"\n#### {lab}\n")
        print("| row | arm | decode p50 tok/s | TPOT p50 / p95 / p99 ms | TTFT p50 / p95 ms | E2E p50 ms | ok |")
        print("|---|---|---|---|---|---|---|")
        per = {}
        for r in rows:
            c = data[r].get(lab)
            if c is None:
                continue
            name = os.path.basename(r)
            arm = name.split("-", 1)[1]
            d = c.get("decode_tok_s_p50")
            print(f"| {name.split('-')[0]} | {arm} | {d:.2f} | {c['tpot_ms_p50']:.2f} / {c['tpot_ms_p95']:.2f} / {c['tpot_ms_p99']:.2f} | {c['ttft_ms_p50']:,.0f} / {c['ttft_ms_p95']:,.0f} | {c['e2e_ms_p50']:,.0f} | {c['n_ok']}/{c['n_req']} |")
            per.setdefault(arm, []).append(c)
        print("\n| arm | N | decode p50 median | TPOT p50 median ms | TTFT p50 median ms |")
        print("|---|---|---|---|---|")
        med = {}
        for arm in (lane, base):
            v = per.get(arm, [])
            if not v:
                continue
            med[arm] = [st.median(x[k] for x in v) for k in ("decode_tok_s_p50", "tpot_ms_p50", "ttft_ms_p50")]
            print(f"| {arm} | {len(v)} | {med[arm][0]:.2f} | {med[arm][1]:.2f} | {med[arm][2]:,.0f} |")
        if lane in med and base in med:
            print(f"\nDelta, lane vs base: decode {100*(med[lane][0]/med[base][0]-1):+.2f}%, TTFT {100*(med[lane][2]/med[base][2]-1):+.2f}%.")
        tuples = {hs[r].get(lab) for r in rows if lab in hs[r]}
        first = next(iter(tuples))
        print(f"\nHashes: {'identical across every row of both arms' if len(tuples) == 1 else 'DIFFER: ' + str(len(tuples)) + ' distinct tuples'}; {len(first)} requests, first {' '.join((h or '-')[:8] for h in first)}.")
    th = {}
    for r in rows:
        t = thermal(r)
        if t:
            th.setdefault(os.path.basename(r).split("-", 1)[1], []).append(t)
    for arm, ts in th.items():
        print(f"\nThermal {arm}: power median {min(t[0] for t in ts):.0f}..{max(t[0] for t in ts):.0f} W, peak {max(t[1] for t in ts):.0f} W, SM clock {min(t[2] for t in ts):.0f}..{max(t[3] for t in ts):.0f} MHz, max temp {max(t[4] for t in ts):.0f} C, {sum(t[5] for t in ts)} samples.")
