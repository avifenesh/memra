#!/usr/bin/env python3
"""lane_tab.py <campaign-root> <lane-arm> [base-arm]

Markdown for one lane from a shared-control served campaign: per-row table (lane and base rows
of each mode, campaign order), per-arm medians with N and deltas, hash identity per cell label,
and the thermal regime from the 250 ms telemetry (samples at >= 20% utilization).
Rows are <root>/<mode>/r<i>-<arm>/ with cells.jsonl, env.txt, telemetry-250ms.csv.
"""
import csv, glob, json, os, statistics as st, sys

root, lane = sys.argv[1], sys.argv[2]
base = sys.argv[3] if len(sys.argv) > 3 else "base"


def cells(row):
    out = {}
    for line in open(os.path.join(row, "cells.jsonl")):
        c = json.loads(line)
        out[c["label"]] = c
    return out


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
    if not pw:
        return None
    return st.median(pw), max(pw), min(clk), max(clk), max(tmp), len(pw)


for mode in sorted(os.listdir(root)):
    mdir = os.path.join(root, mode)
    if not os.path.isdir(mdir):
        continue
    rows = [r for r in glob.glob(os.path.join(mdir, "r*-*")) if os.path.isdir(r)]
    rows.sort(key=lambda r: int(os.path.basename(r)[1:].split("-")[0]))
    rows = [r for r in rows if os.path.basename(r).split("-", 1)[1] in (lane, base)
            and os.path.exists(os.path.join(r, "cells.jsonl"))]
    if not rows:
        continue
    print(f"\n### {mode}\n")
    print("| row | arm | greedy decode tok/s | TPOT p50 / p95 / p99 ms | ITL p50 / p99 ms | TTFT p50 / p95 ms | E2E p50 ms | agg tok/s | sampled decode tok/s | ignore-eos tok/s |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    per, shas, therm = {}, {}, {}
    for r in rows:
        name = os.path.basename(r)
        arm = name.split("-", 1)[1]
        c = cells(r)
        g, s, ie = c.get("greedy-c1"), c.get("sampled-c1"), c.get("greedy-c1-ignore-eos")
        print(f"| {name.split('-')[0]} | {arm} | {g['decode_tok_s_p50']:.2f} | {g['tpot_ms_p50']:.2f} / {g['tpot_ms_p95']:.2f} / {g['tpot_ms_p99']:.2f} | {g['itl_ms_p50']:.2f} / {g['itl_ms_p99']:.2f} | {g['ttft_ms_p50']:.0f} / {g['ttft_ms_p95']:.0f} | {g['e2e_ms_p50']:,.0f} | {g['agg_tok_s']:.2f} | {s['decode_tok_s_p50']:.2f} | {ie['decode_tok_s_p50']:.2f} |")
        per.setdefault(arm, []).append((g["decode_tok_s_p50"], s["decode_tok_s_p50"], ie["decode_tok_s_p50"], g["ttft_ms_p50"]))
        for label, cell in c.items():
            if label == "warmup":
                continue
            shas.setdefault(label, {}).setdefault(arm, set()).add(tuple(cell["shas"]))
        t = thermal(r)
        if t:
            therm.setdefault(arm, []).append(t)
    print(f"\n| arm | N | greedy median | sampled median | ignore-eos median | TTFT p50 median ms |")
    print("|---|---|---|---|---|---|")
    med = {}
    for arm in (lane, base):
        v = per.get(arm, [])
        if not v:
            continue
        med[arm] = [st.median(x[i] for x in v) for i in range(4)]
        print(f"| {arm} | {len(v)} | {med[arm][0]:.2f} | {med[arm][1]:.2f} | {med[arm][2]:.2f} | {med[arm][3]:.0f} |")
    if lane in med and base in med:
        d = [100 * (med[lane][i] / med[base][i] - 1) for i in range(3)]
        print(f"\nDelta, lane vs base: greedy {d[0]:+.2f}%, sampled {d[1]:+.2f}%, ignore-eos {d[2]:+.2f}%.")
    print("\nHash identity per cell (each arm's set of per-request sha tuples across its rows):\n")
    for label, arms in sorted(shas.items()):
        all_t = set().union(*arms.values())
        same = len(all_t) == 1
        first = next(iter(all_t))
        print(f"- `{label}`: {'identical across every row of both arms' if same else 'DIFFERS'}; {len(first)} requests, first hashes {' '.join(h[:8] for h in first[:8])}")
    for arm, ts in therm.items():
        print(f"\nThermal {arm}: power median {min(t[0] for t in ts):.0f}..{max(t[0] for t in ts):.0f} W, peak {max(t[1] for t in ts):.0f} W, SM clock {min(t[2] for t in ts):.0f}..{max(t[3] for t in ts):.0f} MHz, max temp {max(t[4] for t in ts):.0f} C, {sum(t[5] for t in ts)} samples.")
