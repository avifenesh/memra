#!/usr/bin/env python3
"""Summarize grow_ab.sh arms: per cell throughput, TTFT, refusals; per session turn the pool path
and TTFT; the pool counters and the fixed charge (memra#1093 follow-up). Stdlib only.

  python3 summarize.py OUT [ARM_DIR ...]      (default: every arm directory under OUT)
Prints one JSON object per arm and cell, then one per arm with the counters.
"""
import json
import pathlib
import re
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import trace_requests as tr  # noqa: E402

COUNTERS = ("dspark_pool_grid_park_rewinds", "dspark_pool_grid_park_skips", "dspark_pool_grid_rewind_resumes",
            "dspark_pool_grid_rewind_grows", "dspark_pool_grid_declines",
            "continuation_pool_grid_rewinds", "continuation_pool_grid_declines",
            "spec_pool_hits", "spec_pool_evictions", "served_dspark", "served_plain")


def med(xs):
    return round(statistics.median(xs), 3) if xs else None


def arm_summary(d):
    segs, _, parks = tr.segments(d / "serve.log")
    out = []
    for f in sorted(d.glob("*.requests.jsonl")):
        rows = [json.loads(x) for x in open(f)]
        cell = f.name.split(".")[0]
        by_turn = {}
        for r in rows:
            seg = segs.get(r["id"]) if r.get("id") else None
            path = tr.path_of(seg) if r["ok"] else f"http-{r['status']}"
            t = by_turn.setdefault(r["session_turn"], {"n": 0, "paths": {}, "ttft": [], "wall": []})
            t["n"] += 1
            t["paths"][path] = t["paths"].get(path, 0) + 1
            if r["ok"] and r["ttft_s"] is not None:
                t["ttft"].append(r["ttft_s"])
                t["wall"].append(r["wall_s"])
        out.append({"arm": d.name, "cell": cell,
                    "turns": {k: {"n": v["n"], "paths": v["paths"], "ttft_median_s": med(v["ttft"]),
                                  "ttft_sum_s": round(sum(v["ttft"]), 2), "e2e_sum_s": round(sum(v["wall"]), 2)}
                              for k, v in sorted(by_turn.items())},
                    "text_shas": [r["text_sha"] for r in rows]})
    cells = [json.loads(x) for x in open(d / "cells.jsonl")] if (d / "cells.jsonl").exists() else []
    m = json.loads((d / "metrics.json").read_text()) if (d / "metrics.json").exists() else {}
    counters = {k: m.get(k) for k in COUNTERS}
    text = tr.open_log(d / "serve.log").read()
    fixed = re.findall(r"fixed residual high-water: model=\"[^\"]+\" (\d+)MB", text)
    return out, cells, {"arm": d.name, "counters": counters, "parks": dict(parks),
                        "fixed_residual_mb_last": int(fixed[-1]) if fixed else None,
                        "oom_lines": len(re.findall(r"out of memory", text, re.I))}


def main():
    root = pathlib.Path(sys.argv[1])
    dirs = [pathlib.Path(x) for x in sys.argv[2:]] or sorted(
        p for p in root.iterdir() if (p / "serve.log").exists() or (p / "serve.log.gz").exists())
    for d in dirs:
        out, cells, tail = arm_summary(d)
        for c in cells:
            print(json.dumps({"arm": d.name, **{k: c[k] for k in c if k != "arm"}}))
        for o in out:
            print(json.dumps({k: o[k] for k in o if k != "text_shas"}))
        print(json.dumps(tail))


if __name__ == "__main__":
    main()
