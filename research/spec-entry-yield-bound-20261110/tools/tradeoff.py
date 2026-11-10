#!/usr/bin/env python3
"""The arms' trade (memra#1201): long-stream tokens gained per second of added short latency.

usage: tradeoff.py STALL.jsonl
For every (clients, think) cell with an `off` row, each arm row gives: the long stream's tokens
in the window minus off's, the short requests' added latency in total seconds (short requests
served x (mean latency - off's mean latency)), and their ratio. Rows that demoted are marked:
their short-request means mix spec and plain ticks.
"""
import json
import pathlib
import statistics
import sys
from collections import defaultdict

rows = [json.loads(l) for l in pathlib.Path(sys.argv[1]).read_text().splitlines() if l.strip()]
cells = defaultdict(list)
for r in rows:
    cells[(r["clients"], int(r["think_ms"]))].append(r)
print("| clients | think ms | arm | long tokens gained | short seconds added | tokens per short second | demoted |")
print("|---:|---:|---|---:|---:|---:|---|")
for (c, t), rs in sorted(cells.items()):
    off = [r for r in rs if not r.get("knobs")]
    if not off:
        continue
    off_tok = statistics.fmean(r["tokens_during"] for r in off)
    off_mean = statistics.fmean(r["short_lat_mean_s"] for r in off)
    for r in rs:
        if not r.get("knobs"):
            continue
        gained = r["tokens_during"] - off_tok
        added = r["short_requests"] * (r["short_lat_mean_s"] - off_mean)
        ratio = f"{gained / added:.1f}" if added > 0 else "n/a (no added latency)"
        print(f"| {c} | {t} | {r['knobs']} | {gained:.0f} | {added:.1f} | {ratio} | {'yes' if r['demotions'] else 'no'} |")
