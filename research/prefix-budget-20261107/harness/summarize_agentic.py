"""Summarize agentic arms (measured pass): hits, cached tokens, TTFT, throughput, reclaims, device
free floor; with --exact OFF_RUN, greedy byte identity of every measured turn against it.

  python3 summarize_agentic.py RUNS_DIR [--exact OFF_RUN]
"""
import json, re, statistics, sys
from pathlib import Path

runs = Path(sys.argv[1])
off = None
if "--exact" in sys.argv:
    off = json.load(open(runs / sys.argv[sys.argv.index("--exact") + 1] / "rows.json"))["rows"]


def key(r):
    return (r["session"], r["turn"])


rows_out = []
for d in sorted(p for p in runs.iterdir() if p.is_dir() and (p / "rows.json").exists()):
    j = json.load(open(d / "rows.json"))
    c, rows = j["cell"], j["rows"]
    t = (d / "arm.txt").read_text()
    log = (d / "serve.log").read_text(errors="replace")
    g = lambda pat: (re.search(pat, t) or [None, None])[1]
    rec = re.findall(r"reclaim-on-defer: evicted (\d+) prefix entries", log)
    out = {"run": d.name, "budget": g(r"\[prefix-cache\] on: budget (\d+MB)"), "users": c["users"],
           "hits": f'{c["measured_hits"]}/{c["turns"]}', "cached_frac": c["measured_cached_frac"],
           "cached_tokens": c["measured_cached_tokens"], "prompt_tokens": c["measured_prompt_tokens"],
           "ttft_ms_p50": c["ttft_ms_p50"], "ttft_ms_p95": c["ttft_ms_p95"], "ttft_ms_mean": c["ttft_ms_mean"],
           "agg_tps": c["aggregate_gen_tps"], "per_user_tps": c["per_user_gen_tps_median"], "wall_s": c["wall_s"],
           "prime_wall_s": c["prime_wall_s"],
           "evict_lines": len(re.findall(r"\[prefix-cache\] evict \(", log)), "reclaims": len(rec),
           "reclaim_evicted": sum(map(int, rec)), "free_min_mib": g(r"device_free_min_mib (\d+)"),
           "oom": g(r"oom_lines=(\d+)")}
    if off is not None:
        ref = {key(r): r for r in off}
        same = sum(1 for r in rows if key(r) in ref and (r["content"], r["reasoning"], r["tool_calls"]) ==
                   (ref[key(r)]["content"], ref[key(r)]["reasoning"], ref[key(r)]["tool_calls"]))
        out["greedy_vs_off"] = f"{same}/{len(rows)}"
        out["diff_turns"] = [f"{r['session'][:8]}:{r['turn']}:cached={r['cached']}" for r in rows
                             if key(r) in ref and (r["content"], r["reasoning"], r["tool_calls"]) !=
                             (ref[key(r)]["content"], ref[key(r)]["reasoning"], ref[key(r)]["tool_calls"])]
    rows_out.append(out)
for o in rows_out:
    print(json.dumps(o))
