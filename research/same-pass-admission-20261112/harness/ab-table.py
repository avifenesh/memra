#!/usr/bin/env python3
"""memra#1255: agentic replay boots, one row per boot in run order, plus text identity per turn.
usage: ab-table.py AGENTIC_DIR"""
import json
import re
import statistics
import sys
from pathlib import Path
import gzip


def read_log(d, pattern):
    """A run's log, plain or gzipped (the research copy gzips logs over 20 kB)."""
    f = next(iter(sorted(d.glob(pattern))), None) or next(iter(sorted(d.glob(pattern + ".gz"))))
    data = gzip.decompress(f.read_bytes()) if f.suffix == ".gz" else f.read_bytes()
    return data.decode(errors="replace")

RE_SETTLED = re.compile(r"\[admission\] settled: .* admitted after (\d+) settle wait\(s\), ([0-9.]+) ms")


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = max(0, min(len(xs) - 1, round(p / 100 * (len(xs) - 1))))
    return xs[k]


boots = [d for d in Path(sys.argv[1]).iterdir() if (d / "cells.log").exists()]
boots.sort(key=lambda p: p.name.split("-")[-1])
print("| boot | turns ok | tokens | wall s | agg gen tok/s | TTFT p50 / p95 s | settle waits (passes) | "
      "settled requests, wait ms sum / max | VRAM defer lines | drafter refused | step-OOM parks | out of memory |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
texts = {}
for d in boots:
    log = read_log(d, "serve.log")
    s = json.loads((d / "cells.log").read_text().strip().splitlines()[-1])
    m = json.loads((d / "metrics.json").read_text()) if (d / "metrics.json").exists() else {}
    reqs = [json.loads(l) for l in next(d.glob("*.requests.jsonl")).read_text().splitlines() if l.strip()]
    timed = [r for r in reqs if r.get("ok") and r.get("ttft_s") is not None and r.get("usage", {}).get("completion_tokens", 0) > 1]
    ttft = [r["ttft_s"] for r in timed]
    settled = [(int(a), float(b)) for a, b in RE_SETTLED.findall(log)]
    waits = [b for _, b in settled]
    texts[d.name] = {(r.get("session"), r.get("session_turn")): r.get("text_sha") for r in timed}
    print(f"| {d.name} | {s['ok']}/{s['turns']} | {s['tokens']:,} | {s['wall_s']} | {s['aggregate_gen_tps']} | "
          f"{pct(ttft, 50):.2f} / {pct(ttft, 95):.2f} | {m.get('admission_settle_waits', 0)} | "
          f"{len(settled)}, {sum(waits):.0f} / {max(waits, default=0):.0f} | {log.count('VRAM defer')} | "
          f"{log.count('drafter refused:')} | {m.get('step_oom_parks', 'n/a')} | {log.count('out of memory')} |")
if texts:
    ref_name = boots[0].name
    ref = texts[ref_name]
    print()
    print(f"Text identity per (session, turn) against {ref_name}:")
    for name, t in texts.items():
        keys = set(ref) & set(t)
        same = sum(1 for k in keys if ref[k] == t[k])
        print(f"- {name}: {same}/{len(keys)} identical")
