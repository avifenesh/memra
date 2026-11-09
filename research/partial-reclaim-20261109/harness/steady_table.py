"""Table of the steady-traffic cell (harness/steady.py) from each arm's serve.log and rows.json.

  python3 steady_table.py STEADY_DIR

Per arm: reclaims (partial passes, still short), entries and GB freed, the booking's on/off lines and whether it was
still armed when the log ended, evictions, OOM lines, the steady stream's new prompts (rows, errors, TTFT p50 / p95)
and repeats (rows, hits, cached tokens, TTFT p50), the repeat hit rate per 300 s window, and greedy texts against the
first A arm (rows that completed on both; request tags are deterministic by issue order).
"""
import json, re, sys
from pathlib import Path

root = Path(sys.argv[1])


def order(p):
    m = re.search(r"-(\d+)-[AB]$", p.name)
    return (int(m.group(1)) if m else 0, p.name)


arms = sorted((p for p in root.iterdir() if (p / "rows.json").is_file() and re.search(r"-\d+-[AB]$", p.name)), key=order)
ref = next((p for p in arms if p.name.endswith("-A")), None)
ref_rows = {r["tag"]: r for r in json.loads((ref / "rows.json").read_text())["rows"]} if ref else {}

print("| arm | reclaims (partial, still short) | entries / GB freed (largest) | booking on / off, armed at end "
      "| evictions preflight / LRU | OOM lines | new prompts: rows, errors, TTFT p50 / p95 s "
      "| repeats: rows, hits, cached tokens, TTFT p50 s | repeat hits per 300 s | greedy texts = A |")
print("|---|---|---|---|---|---|---|---|---|---|")
for d in arms:
    log = (d / "serve.log").read_text(errors="replace")
    reclaims = [int(x) for x in re.findall(r"reclaim-on-defer: evicted (\d+) prefix entries", log)]
    settled = [int(x) for x in re.findall(r"reclaim settle \(reclaim-on-defer\): dev0 evicted_prefix_bytes=(\d+)", log)]
    partial = re.findall(r"reclaim-on-defer \(partial\): .*?; (covered|still short[^(]*)", log)
    short = sum(1 for p in partial if p.startswith("still short"))
    on = [m.start() for m in re.finditer(r"\[admit-oom\] reclaim booking on:", log)]
    off = [m.start() for m in re.finditer(r"\[admit-oom\] reclaim booking off:", log)]
    armed_end = bool(on) and (not off or on[-1] > off[-1])
    oom = len(re.findall(r"out of memory|CUDA_ERROR_OUT_OF_MEMORY", log, re.I))
    pre = len(re.findall(r"\[prefix-cache\] evict \(snapshot preflight, LRU\)", log))
    lru = len(re.findall(r"\[prefix-cache\] evict \(LRU\)", log))
    big = max(zip(reclaims, settled), key=lambda x: x[1], default=(0, 0))
    j = json.loads((d / "rows.json").read_text())
    s, rows = j["summary"], j["rows"]
    n, r = s["steady_new"], s["steady_repeat"]
    win = " / ".join(f"{w['hits']}/{w['repeats']}" for w in s["repeat_windows_300s"])
    both = [x for x in rows if x["tag"] in ref_rows and not x["error"] and not ref_rows[x["tag"]]["error"]]
    same = sum(1 for x in both if x["text"] == ref_rows[x["tag"]]["text"])
    f = lambda v: f"{v / 1000:.1f}" if v is not None else "-"
    print(f"| {d.name} | {len(reclaims)} ({len(partial)}, {short}) | {sum(reclaims)} / {sum(settled) / 1e9:.1f} "
          f"({big[0]} / {big[1] / 1e9:.1f}) | {len(on)} / {len(off)}, {'yes' if armed_end else 'no'} | {pre} / {lru} | {oom} | "
          f"{n['rows']}, {n['errors']}, {f(n['ttft_ms_p50'])} / {f(n['ttft_ms_p95'])} | "
          f"{r['rows']}, {r['hits']}, {r['cached_tokens']}, {f(r['ttft_ms_p50'])} | {win} | {same}/{len(both)} |")
