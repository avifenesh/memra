#!/usr/bin/env python3
"""Medians of conc_bench.py runs: one row a (arm, workload, concurrency).

usage: conc_summary.py TAG=CONC_JSONL [TAG=CONC_JSONL ...] [--json OUT]

Per row, over the non-warm-up samples (N in the table): the wall aggregate output tokens/s, the
decode-phase aggregate (every request past its first token, none ended), the median request's
decode tokens/s, the decode step p50 and p95 (steps with no prefill rows), decode rows a step,
forward calls a decode step and decode rows a forward call (runs recorded before
`decode_rows_per_call` existed show none), and whether every request's text equals the same
request alone (identity).
"""
import json
import statistics
import sys

args = sys.argv[1:]
out_json = None
if "--json" in args:
    at = args.index("--json")
    out_json = args[at + 1]
    del args[at:at + 2]
rows = []
for spec in args:
    tag, path = spec.split("=", 1)
    data = [json.loads(line) for line in open(path) if line.strip()]
    keys = sorted({(r["workload"], r["concurrency"]) for r in data if "concurrency" in r},
                  key=lambda k: (k[0], k[1]))
    for w, c in keys:
        s = [r for r in data if r.get("workload") == w and r.get("concurrency") == c and not r["warmup"]]
        if not s:
            continue
        med = lambda xs: statistics.median(xs) if xs else None  # noqa: E731
        per = [statistics.median([q["decode_tok_s"] or 0 for q in r["requests"]]) for r in s]
        sv = [r["server"] for r in s]
        rows.append({
            "arm": tag, "workload": w, "c": c, "n": len(s),
            "agg_wall": med([r["aggregate_tok_s"] for r in s]),
            "agg_decode": med([r["decode_phase"]["agg_tok_s"] for r in s if r["decode_phase"]["agg_tok_s"]]),
            "per_req": med(per),
            "step_p50_ms": med([x["decode_step_s_p50"] * 1e3 for x in sv if x["decode_step_s_p50"]]),
            "step_p95_ms": med([x["decode_step_s_p95"] * 1e3 for x in sv if x["decode_step_s_p95"]]),
            "rows_step": med([x["decode_rows_per_step"] for x in sv if x["decode_rows_per_step"]]),
            "rows_call": med([x.get("decode_rows_per_call") for x in sv if x.get("decode_rows_per_call")]),
            "calls_step": med([x.get("decode_calls_per_step") for x in sv if x.get("decode_calls_per_step")]),
            "identity": all(q["identity"] for r in s for q in r["requests"]),
        })


def f(x, d=2):
    return "-" if x is None else f"{x:.{d}f}"


print("| arm | workload | c | N | agg tok/s (wall) | agg tok/s (decode phase) | per-request tok/s | "
      "step p50 ms | step p95 ms | rows/step | calls/step | rows/call | identity |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r['arm']} | {r['workload']} | {r['c']} | {r['n']} | {f(r['agg_wall'])} | {f(r['agg_decode'])} | "
          f"{f(r['per_req'])} | {f(r['step_p50_ms'], 1)} | {f(r['step_p95_ms'], 1)} | {f(r['rows_step'])} | "
          f"{f(r['calls_step'])} | {f(r['rows_call'])} | {'PASS' if r['identity'] else 'FAIL'} |")
if out_json:
    json.dump(rows, open(out_json, "w"), indent=1)
