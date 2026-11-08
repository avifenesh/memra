#!/usr/bin/env python3
"""#1090 mixed-shape table: per-run rows from summarize_mixed.py over several run dirs, then per-arm
aggregates (runs named <phase>-<n>-<A|B>; A = main, B = #1090).

  m12_table.py OFF_RUN_ABS DIR [DIR ...]
"""
import json
import statistics
import subprocess
import sys

H = "<harness>/summarize_mixed.py"
off, dirs = sys.argv[1], sys.argv[2:]
rows = []
for d in dirs:
    out = subprocess.run([sys.executable, H, d, off], capture_output=True, text=True)
    rows += [json.loads(l) for l in out.stdout.splitlines() if l.startswith("{")]
    if out.returncode:
        print(out.stderr[-500:])
for r in rows:
    print(r["run"], r["budget"], "hits", r["repeat_hits"], "cached", r["repeat_cached"], "ttft", r["repeat_ttft_ms"],
          "batch", r["batch_agg_tok_s"], r["batch_ttft_p50_ms"], "inserts", r["inserts"], "ev", r["evict_lines"],
          "reclaims", r["reclaims"], "free", r["free_min_mib"], "oom", r["oom"], "greedy", r["greedy_vs_off"],
          "repeat_diff", r["repeat_diff_vs_off"])
for arm in "AB":
    rs = [r for r in rows if r["run"].endswith("-" + arm) and "off" not in r["run"]]
    if not rs:
        continue
    allt = sorted(x for r in rs for x in r["repeat_ttft_ms"] if x is not None)
    print(f"{arm}: {len(rs)} runs; hits per run {[r['repeat_hits'] for r in rs]}; repeat TTFT over {len(allt)} repeats "
          f"median {statistics.median(allt):.1f} ms, mean {statistics.mean(allt):.1f}, min {allt[0]:.1f}, max {allt[-1]:.1f}; "
          f"per-run repeat medians {[round(r['repeat_ttft_median_ms'], 1) for r in rs]}; "
          f"batch tok/s median {statistics.median(r['batch_agg_tok_s'] for r in rs)}; "
          f"batch TTFT p50 median {statistics.median(r['batch_ttft_p50_ms'] for r in rs)}; "
          f"evictions {[r['evict_lines'] for r in rs]}; free min MiB {min(int(r['free_min_mib']) for r in rs)}")
