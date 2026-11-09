"""Table of the at-the-limit cell: per arm, the boot budget, what the defers flushed, the single-entry eviction
lines (snapshot preflight / LRU insert; no other kind appeared), OOM lines and each phase's hits, errors and TTFT.

  python3 limit_table.py LIMIT_DIR
"""
import json, re, sys
from pathlib import Path

root = Path(sys.argv[1])
print("| arm | budget MB | reclaim-on-defer flushes | entries / GB flushed | evictions preflight / LRU | OOM lines "
      "| pressure: errors, TTFT p50 / max s | repeats: hits, TTFT p50 s |")
print("|---|---|---|---|---|---|---|---|")
for d in sorted(p for p in root.glob("limit-*") if (p / "rows.json").is_file()):
    log = (d / "serve.log").read_text(errors="replace")
    budget = re.search(r"\[prefix-cache\] on: budget (\d+)MB", log)
    flushes = re.findall(r"reclaim-on-defer: evicted (\d+) prefix entries", log)
    settled = [int(x) for x in re.findall(r"reclaim settle \(reclaim-on-defer\): dev0 evicted_prefix_bytes=(\d+)", log)]
    oom = len(re.findall(r"out of memory|CUDA_ERROR_OUT_OF_MEMORY", log, re.I))
    pre = len(re.findall(r"\[prefix-cache\] evict \(snapshot preflight, LRU\)", log))
    lru = len(re.findall(r"\[prefix-cache\] evict \(LRU\)", log))
    s = json.loads((d / "rows.json").read_text())["summary"]
    p, r = s["pressure"], s["repeat"]
    print(f"| {d.name} | {budget.group(1) if budget else '?'} | {len(flushes)} | "
          f"{sum(int(x) for x in flushes)} / {sum(settled) / 1e9:.1f} | {pre} / {lru} | {oom} | "
          f"{p['errors']} of {p['rows']}, {p['ttft_ms_p50'] / 1000:.1f} / {p['ttft_ms_max'] / 1000:.1f} | "
          f"{r['hits']} of {r['rows']}, {r['ttft_ms_p50'] / 1000:.2f} |")
