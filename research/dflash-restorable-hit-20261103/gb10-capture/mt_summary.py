#!/usr/bin/env python3
"""Summarize multiturn_client.py records: per arm run and turn, mean TTFT, mean decode tok/s,
mean prompt and cached tokens, total completion tokens.
usage: mt_summary.py turns.jsonl"""
import json
import sys
from collections import defaultdict

rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
g = defaultdict(list)
for r in rows:
    g[(r["tag"], r["turn"])].append(r)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


print("| run | turn | n | TTFT s | total s | decode tok/s | prompt tok | cached tok | completion tok |")
print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
for (tag, turn), rs in sorted(g.items(), key=lambda kv: (kv[0][0], ['x0', 't2', 't3'].index(kv[0][1]))):
    u = [r.get("usage") or {} for r in rs]
    print(f"| {tag} | {turn} | {len(rs)} | {mean([r['ttft_s'] for r in rs]):.2f} | {mean([r['total_s'] for r in rs]):.2f} | "
          f"{mean([r['decode_tps'] for r in rs]):.2f} | "
          f"{mean([x.get('prompt_tokens') for x in u]):.0f} | "
          f"{mean([(x.get('prompt_tokens_details') or {}).get('cached_tokens') for x in u]):.0f} | "
          f"{sum(r['completion_tokens'] or 0 for r in rs)} |")
