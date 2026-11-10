#!/usr/bin/env python3
"""Tables from stall_cell.py rows (memra#1201).

usage: summarize.py RUNDIR [RUNDIR ...] [--ref RUNDIR]
Prints one markdown row per stall.jsonl row: arm, knobs, clients, think, the long request's
tokens in the window, its rate there, the longest gap, short requests served and their latency,
no-round bursts that emitted nothing, bound rounds, demotions. With --ref, each greedy row's
text is compared with the reference rows' text of the same (clients, think) cell, or of any
greedy row when the cell has none: equal over the common length (a stream is cut at a time,
so the shorter text must be a prefix of the longer).
"""
import json
import pathlib
import sys

args = sys.argv[1:]
ref = None
if "--ref" in args:
    i = args.index("--ref")
    ref = pathlib.Path(args[i + 1])
    del args[i:i + 2]


def rows(d):
    p = pathlib.Path(d) / "stall.jsonl"
    return [dict(json.loads(l), _dir=str(d)) for l in p.read_text().splitlines() if l.strip()]


def text(r):
    return (pathlib.Path(r["_dir"]) / r["text_file"]).read_text()


refs = rows(ref) if ref else []
print("| arm | knobs | clients | think ms | tokens in window | tok/s | longest gap s | short req | short p50 s | short p95 s | short mean s | silent no-round | bound rounds | flushes | demotions | text vs ref |")
print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
for d in args:
    for r in rows(d):
        cmp = ""
        if refs and r["class"] == "greedy":
            same = [x for x in refs if x["class"] == "greedy" and x["clients"] == r["clients"] and x["think_ms"] == r["think_ms"]] or \
                   [x for x in refs if x["class"] == "greedy"]
            t = text(r)
            ok = []
            for x in same:
                u = text(x)
                n = min(len(t), len(u))
                ok.append(t[:n] == u[:n])
            cmp = f"{sum(ok)}/{len(ok)} prefix-equal" if ok else "no ref"
        print(f"| {r['arm']} | {r.get('knobs') or 'off'} | {r['clients']} | {int(r['think_ms'])} | {r['tokens_during']} | "
              f"{r['rate_during']} | {r['longest_gap_in_window_s']} | {r['short_requests']} | {r['short_lat_p50_s']} | "
              f"{r['short_lat_p95_s']} | {r['short_lat_mean_s']} | {r['no_round_bursts_that_emitted_nothing']} | "
              f"{r['bound_rounds_in_window']} | {r['flushes_in_window']} | {r['demotions']} | {cmp} |")
