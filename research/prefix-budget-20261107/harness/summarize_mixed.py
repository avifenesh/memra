"""Summarize mixed-shape arms: hits, repeat TTFT, batch throughput, evictions, reclaims, device
free floor, and greedy byte identity against a cache-off reference run.

  python3 summarize_mixed.py RUNS_DIR OFF_RUN [OFF_RUN2 ...]
"""
import json, re, statistics, sys
from pathlib import Path

runs = Path(sys.argv[1])
offs = [json.load(open(runs / o / "rows.json"))["rows"] for o in sys.argv[2:]]
off_text = {}
for rows in offs:
    for r in rows:
        off_text.setdefault(r["tag"], r["text"])


def armtxt(d):
    t = (d / "arm.txt").read_text()
    g = lambda pat: (re.search(pat, t) or [None, None])[1]
    return {"free_min_mib": g(r"device_free_min_mib (\d+)"), "reclaims": g(r"reclaim_on_defer_lines=(\d+)"),
            "evict_lines": g(r"evict_lines=(\d+)"), "oom": g(r"oom_lines=(\d+)"),
            "budget": g(r"\[prefix-cache\] on: budget (\d+MB)")}


out = []
for d in sorted(p for p in runs.iterdir() if p.is_dir() and (p / "rows.json").exists()):
    j = json.load(open(d / "rows.json"))
    s, rows = j["summary"], j["rows"]
    log = (d / "serve.log").read_text(errors="replace")
    ev = len(re.findall(r"\[prefix-cache\] evict \(", log))
    rec = re.findall(r"reclaim-on-defer: evicted (\d+) prefix entries", log)
    ins = len(re.findall(r"\[prefix-cache\] insert \(", log))
    same = diff = 0
    diffs = []
    for r in rows:
        if r["tag"] in off_text:
            if r["text"] == off_text[r["tag"]]:
                same += 1
            else:
                diff += 1
                diffs.append(r["tag"])
    rep = sorted((r for r in rows if r["tag"].startswith("repeat-")), key=lambda r: r["tag"])
    rep_diff = [r["tag"] for r in rep if r["tag"] in off_text and r["text"] != off_text[r["tag"]]]
    a = armtxt(d)
    out.append({"run": d.name, "budget": a["budget"], "repeat_hits": s["repeat_hits"], "repeat_cached": s["repeat_cached"],
                "repeat_ttft_ms": s["repeat_ttft_ms"],
                "repeat_ttft_median_ms": statistics.median([x for x in s["repeat_ttft_ms"] if x is not None]),
                "batch_agg_tok_s": s["batch_agg_tok_s"], "batch_rows": s["batch_rows"],
                "batch_ttft_p50_ms": round(s["batch_ttft_ms_p50"], 1) if s["batch_ttft_ms_p50"] else None,
                "inserts": ins, "evict_lines": ev, "reclaims": len(rec), "reclaim_evicted_entries": sum(map(int, rec)),
                "free_min_mib": a["free_min_mib"], "oom": a["oom"],
                "greedy_vs_off": f"{same}/{same + diff}", "repeat_diff_vs_off": rep_diff, "diff_tags": diffs[:8]})
for o in out:
    print(json.dumps(o))
