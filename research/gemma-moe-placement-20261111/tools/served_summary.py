#!/usr/bin/env python3
"""Summarize served cells: per group (cell name without its -N suffix) the text hashes, the
median times and the peak memory.

usage: served_summary.py <served dir>
"""
import json
import pathlib
import re
import statistics
import sys

root = pathlib.Path(sys.argv[1])
groups = {}
for res in sorted(root.glob("*/result.json")):
    name = res.parent.name
    r = json.loads(res.read_text())
    groups.setdefault(re.sub(r"-\d+$", "", name), []).append((name, r))


def med(vals):
    vals = [v for v in vals if v is not None]
    return (round(statistics.median(vals), 3), len(vals), round(min(vals), 3), round(max(vals), 3)) if vals else None


print("| group | N | text | chat | ttft s med (min-max) | e2e s med (min-max) | chat e2e s med | decode tok/s med (min-max) | peak MiB |")
print("|---|---|---|---|---|---|---|---|---|")
for g, cells in groups.items():
    rs = [r for _, r in cells]
    texts = sorted({r.get("text_sha16") for r in rs})
    chats = sorted({r.get("chat_sha16") for r in rs if r.get("chat_sha16")})
    ttft, e2e, ce2e, dec = (med([r.get(k) for r in rs]) for k in ("ttft_s", "e2e_s", "chat_e2e_s", "decode_tok_per_s"))
    fmt = lambda m: "-" if not m else f"{m[0]} ({m[2]}-{m[3]})"
    peak = sorted({r.get("peak_mem_mib") for r in rs})
    print(f"| {g} | {len(rs)} | {' '.join(t or '-' for t in texts)} | {' '.join(chats) or '-'} | {fmt(ttft)} | {fmt(e2e)} | {fmt(ce2e)} | {fmt(dec)} | {'/'.join(map(str, peak))} |")
