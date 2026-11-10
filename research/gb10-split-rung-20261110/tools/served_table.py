#!/usr/bin/env python3
"""The served B/T table of this lane: per campaign and request, the median round time of each arm
with its spread (rounds_by_req.py's method: every generating `[tick-spec]` burst but the
request's first and last), the change, whether every T cell is under every B cell, the TTFT and
decode rate after the first token (completion tokens over total minus TTFT) medians, the tokens a
round over all of the request's rounds, and the text identity (one sha per arm, and whether the
arms match). One markdown row per (campaign, request).
usage: served_table.py LABEL=CAMPAIGN_DIR ..."""
import json, pathlib, re, statistics, sys

ts = re.compile(r"(\d+\.\d+) \[tick-spec\].* wall_ms=([\d.]+) generated=(\d+)->(\d+) rounds=(\d+)")
print("| context (unit, B / T) | request | round ms B (spread) | T (spread) | T - B | every T under every B | TTFT s B / T | tok/s after the first token B / T | tokens a round B / T | text |")
print("|---|---|---:|---:|---:|---|---|---|---|---|")
for arg in sys.argv[1:]:
    label, d = arg.split("=", 1)
    d = pathlib.Path(d)
    res = {}
    for c in sorted(p for p in d.iterdir() if p.is_dir() and re.match(r"\d\d-[BT]$", p.name)):
        log = c / ("server.log" if (c / "server.log").exists() else "trace.txt")
        ticks = [(float(m.group(1)), float(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)))
                 for l in log.read_text(errors="replace").splitlines() if (m := ts.match(l))]
        for r in (json.loads(l) for l in (c / "ab.jsonl").read_text().splitlines() if l.strip()):
            t0, t1 = r["send_unix"], r["send_unix"] + r["total_s"] + 0.05
            g = [t for t in ticks if t0 <= t[0] - t[1] / 1000 + 0.001 and t[0] <= t1 and t[3] > t[2]]
            mid = g[1:-1]
            res.setdefault(r["req"], {}).setdefault(c.name[-1], []).append({
                "ms": sum(t[1] for t in mid) / sum(t[4] for t in mid) if mid else None,
                "ttft": r["ttft_s"], "rate": r["completion_tokens"] / max(r["total_s"] - r["ttft_s"], 1e-9),
                "tpr": r["completion_tokens"] / max(sum(t[4] for t in g), 1), "sha": r["text_sha256"][:10]})
    for req, arms in res.items():
        B, T = arms.get("B", []), arms.get("T", [])
        med = lambda xs, k: statistics.median(x[k] for x in xs)
        spr = lambda xs: max(x["ms"] for x in xs) - min(x["ms"] for x in xs)
        mb, mt = med(B, "ms"), med(T, "ms")
        under = max(x["ms"] for x in T) < min(x["ms"] for x in B)
        shb, sht = {x["sha"] for x in B}, {x["sha"] for x in T}
        if len(shb) == 1 and shb == sht:
            text = f"one text ({next(iter(shb))})"
        else:
            text = f"B {'/'.join(sorted(shb))}, T {'/'.join(sorted(sht))}"
        print(f"| {label} | {req} | {mb:.2f} ({spr(B):.2f}) | {mt:.2f} ({spr(T):.2f}) | {(mt - mb) / mb * 100:+.1f}% | {'yes' if under else 'no'} | "
              f"{med(B, 'ttft'):.2f} / {med(T, 'ttft'):.2f} | {med(B, 'rate'):.2f} / {med(T, 'rate'):.2f} | {med(B, 'tpr'):.2f} / {med(T, 'tpr'):.2f} | {text} |")
