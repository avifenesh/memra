#!/usr/bin/env python3
"""Tables for the prime-hold fix A/B (memra#1223): stall rows (run/grace/stall.jsonl) and the
streamed normal census (run/grace-normal/normal.jsonl).
usage: summgrace.py STALL_JSONL NORMAL_JSONL"""
import json
import sys


def rows(path):
    try:
        return [json.loads(l) for l in open(path) if l.strip()]
    except FileNotFoundError:
        return []


print("| boot | cell | long tokens | long ITL p50 / p99 ms | short n | short p50 / p95 s | hold waits (ms total) |")
print("|---|---|---:|---|---:|---|---|")
for r in rows(sys.argv[1]):
    c = r.get("census_cell") or {}
    print(f"| {r['arm']} | {r['clients']}:{int(r['think_ms'])} | {r['tokens_during']} | "
          f"{r.get('long_itl_p50_ms')} / {r.get('long_itl_p99_ms')} | {r['short_requests']} | "
          f"{r['short_lat_p50_s']} / {r['short_lat_p95_s']} | {c.get('hold_waits', '')} ({c.get('hold_wait_ms', '')}) |")
print()
print("| boot | c | agg tok/s | TTFT p50 s | ITL p50 / p99 ms | K at admission | hold waits (ms total) | texts |")
print("|---|---:|---:|---:|---|---|---|---|")
ref = {}
for r in rows(sys.argv[2]):
    if r["arm"].startswith("main"):
        for q in r["rows"]:
            ref.setdefault((r["c"], q["prompt"]), q.get("sha"))
for r in rows(sys.argv[2]):
    ks = {}
    for k in r.get("spec_k", []):
        ks[k] = ks.get(k, 0) + 1
    diff = sum(1 for q in r["rows"] if ref.get((r["c"], q["prompt"])) not in (None, q.get("sha")))
    print(f"| {r['arm']} | {r['c']} | {r['agg_tok_s']} | {r.get('ttft_p50_s')} | {r.get('itl_p50_ms')} / {r.get('itl_p99_ms')} | "
          f"{' '.join(f'{k} x{v}' for k, v in sorted(ks.items()))} | {r.get('hold_waits')} ({r.get('hold_wait_ms')}) | "
          f"{len(r['rows']) - diff} of {len(r['rows'])} equal to main |")
