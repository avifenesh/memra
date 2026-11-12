#!/usr/bin/env python3
"""memra#1173, memra#1255: the normal-concurrency cells (normal_cell.py), one row per boot and c, in run order.
usage: cell-table.py CELLS_DIR"""
import json
import sys
from pathlib import Path
import gzip


def read_log(d, pattern):
    """A run's log, plain or gzipped (the research copy gzips logs over 20 kB)."""
    f = next(iter(sorted(d.glob(pattern))), None) or next(iter(sorted(d.glob(pattern + ".gz"))))
    data = gzip.decompress(f.read_bytes()) if f.suffix == ".gz" else f.read_bytes()
    return data.decode(errors="replace")

rows = []
for d in sorted(Path(sys.argv[1]).iterdir(), key=lambda p: p.name.split("-")[-1]):
    f = d / "normal.jsonl"
    if not f.exists():
        continue
    log = read_log(d, "server-*.log")
    for line in f.read_text().splitlines():
        r = json.loads(line)
        rows.append((d.name, r, log))
print("| boot | c | requests, errors | completion tokens | agg tok/s | TTFT p50 s | ITL p50 / p99 ms | text shas |")
print("|---|---|---|---|---|---|---|---|")
shas = {}
for name, r, log in rows:
    key = (r["c"],)
    sig = tuple(sorted((x["prompt"], x.get("sha")) for x in r["rows"]))
    shas.setdefault(key, set()).add(sig)
    print(f"| {name} | {r['c']} | {r['requests']}, {r['errors']} | {r['completion_tokens']} | {r['agg_tok_s']} | {r['ttft_p50_s']} | "
          f"{r['itl_p50_ms']} / {r['itl_p99_ms']} | {len(set(x.get('sha') for x in r['rows']))} |")
print()
for (c,), sigs in sorted(shas.items()):
    print(f"- c={c}: {'one' if len(sigs) == 1 else len(sigs)} set of (prompt, text sha) across the boots")
for name in sorted({n for n, _, _ in rows}):
    log = next(l for n, _, l in rows if n == name)
    print(f"- {name}: eager-arm lines {log.count('[admission] eager-arm admit:')}, drafter refused {log.count('[admission] drafter refused:')}, "
          f"VRAM defer lines {log.count('VRAM defer')}, settle wait lines {log.count('[admission] settle wait:')}, "
          f"out of memory {log.count('out of memory')}")
