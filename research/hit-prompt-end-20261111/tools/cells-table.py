#!/usr/bin/env python3
"""memra#1233: one row per boot of the frozen mixed hit/cold c=4 cell, in run order.

Joins the cell's summary rows (hitcold-cell.py) with the per-class TTFT split (ttft-split.py):
hit rate, tail hits, hit TTFT p50/p95 over all hit-role requests and over the restored ones, the
restored hits' prime_wait and decode_wait, cold TTFT, output tok/s.

usage: cells-table.py RUN_DIR [--json]   (RUN_DIR holds cell.jsonl, summary.log, server-LABEL.log)
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
run = Path(sys.argv[1])
rows = []
for line in (run / "summary.log").read_text().splitlines():
    if not line.startswith("{"):
        continue
    s = json.loads(line)
    label = s["label"]
    split = json.loads(subprocess.run(
        [sys.executable, str(HERE / "ttft-split.py"), str(run / "cell.jsonl"), str(run / f"server-{label}.log"), label, "--json"],
        check=True, capture_output=True, text=True).stdout)
    m = s.get("metrics", {})
    whole, suffix, missed = (split.get(k, {"n": 0}) for k in ("hit-whole", "hit-suffix", "hit-missed"))
    rows.append({
        "label": label, "arm": label.rsplit("-", 1)[0], "hits": s["hits"],
        "restored": whole["n"] + suffix["n"], "whole": whole["n"], "suffix": suffix["n"], "missed": missed["n"],
        "hit_p50": s["hit_ttft_p50_ms"], "hit_p95": s["hit_ttft_p95_ms"],
        "whole_p50": whole.get("ttft_p50"), "whole_p95": whole.get("ttft_p95"),
        "suffix_p50": suffix.get("ttft_p50"), "suffix_p95": suffix.get("ttft_p95"),
        "suffix_prime_wait_p50": suffix.get("prime_wait_p50"), "suffix_prime_wait_p95": suffix.get("prime_wait_p95"),
        "suffix_prime_p50": suffix.get("prime_p50"), "suffix_decode_wait_p95": suffix.get("decode_wait_p95"),
        "whole_decode_wait_p50": whole.get("decode_wait_p50"), "whole_decode_wait_p95": whole.get("decode_wait_p95"),
        "cold_p50": s["cold_ttft_p50_ms"], "cold_p95": s["cold_ttft_p95_ms"], "tok_s": s["output_tok_s_median"],
        "not_exact": s["requests_not_exact"],
        "tail_attaches": m.get("prefix_cache_tail_attaches"), "tail_hits": m.get("prefix_cache_tail_hits"),
        "tail_skips": m.get("prefix_cache_tail_skips"), "tail_strips": m.get("prefix_cache_tail_strips"),
        "evictions": m.get("prefix_cache_evictions"), "entries": m.get("prefix_cache_entries"),
        "cache_bytes": m.get("prefix_cache_bytes"),
    })
if "--json" in sys.argv:
    for r in rows:
        print(json.dumps(r, sort_keys=True))
    sys.exit(0)


def f(v, nd=0):
    return "-" if v is None else f"{v:,.{nd}f}"


print("| boot | restored / 90 (whole + grid) | hit TTFT p50 / p95, all 90 | restored whole p50 / p95 | restored grid p50 / p95 | grid prime_wait p50 / p95 | cold TTFT p50 / p95 | out tok/s | tails att / hit / skip / strip |")
print("|---|---|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r['label']} | {r['restored']} ({r['whole']} + {r['suffix']}) | {f(r['hit_p50'])} / {f(r['hit_p95'])} | "
          f"{f(r['whole_p50'])} / {f(r['whole_p95'])} | {f(r['suffix_p50'])} / {f(r['suffix_p95'])} | "
          f"{f(r['suffix_prime_wait_p50'])} / {f(r['suffix_prime_wait_p95'])} | {f(r['cold_p50'])} / {f(r['cold_p95'])} | "
          f"{f(r['tok_s'], 1)} | {r['tail_attaches'] if r['tail_attaches'] is not None else '-'} / {r['tail_hits'] if r['tail_hits'] is not None else '-'} / "
          f"{r['tail_skips'] if r['tail_skips'] is not None else '-'} / {r['tail_strips'] if r['tail_strips'] is not None else '-'} |")
