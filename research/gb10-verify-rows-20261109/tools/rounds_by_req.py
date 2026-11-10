#!/usr/bin/env python3
"""rounds_by_req.py CAMPAIGN_DIR: the cost of one speculative round per request of a served B/T.

Per cell (NN-B / NN-T with ab.jsonl and a MEMRA_TICK_TRACE=1 server log or its trace.txt) and per
request: wall over rounds of every generating `[tick-spec]` burst except the request's first (which
holds the prime) and its last (cut by the stop). Then per request and arm the median over the
cells with its spread (max - min), the T - B change, whether every T cell is under every B cell,
TTFT and total medians, and whether every cell served the same text."""
import json
import pathlib
import re
import statistics
import sys

d = pathlib.Path(sys.argv[1])
ts = re.compile(r"(\d+\.\d+) \[tick-spec\].* wall_ms=([\d.]+) generated=(\d+)->(\d+) rounds=(\d+)")
res = {}
for c in sorted(p for p in d.iterdir() if p.is_dir() and re.match(r"\d\d-[BT]$", p.name)):
    log = c / ("server.log" if (c / "server.log").exists() else "trace.txt")
    ticks = [(float(m.group(1)), float(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)))
             for l in log.read_text(errors="replace").splitlines() if (m := ts.match(l))]
    for r in (json.loads(l) for l in (c / "ab.jsonl").read_text().splitlines() if l.strip()):
        t0, t1 = r["send_unix"], r["send_unix"] + r["total_s"] + 0.05
        g = [t for t in ticks if t0 <= t[0] - t[1] / 1000 + 0.001 and t[0] <= t1 and t[3] > t[2]]
        mid = g[1:-1]
        row = res.setdefault(r["req"], {}).setdefault(c.name[-1], [])
        row.append({"cell": c.name, "ms": sum(t[1] for t in mid) / sum(t[4] for t in mid) if mid else None,
                    "rounds": sum(t[4] for t in mid), "ttft": r["ttft_s"], "total": r["total_s"],
                    "sha": r["text_sha256"][:12], "tok": r["completion_tokens"]})
for req, arms in res.items():
    print(f"== {req}")
    med = {}
    for arm in "BT":
        rows = arms.get(arm, [])
        for x in rows:
            ms = f"{x['ms']:.3f}" if x["ms"] is not None else "n/a"
            print(f"  {x['cell']}: {ms} ms per round over {x['rounds']} rounds, TTFT {x['ttft']:.3f} s, "
                  f"total {x['total']:.3f} s, {x['tok']} tok, text {x['sha']}")
        v = [x["ms"] for x in rows if x["ms"] is not None]
        if v:
            med[arm] = (statistics.median(v), min(v), max(v))
            print(f"  {arm}: {med[arm][0]:.3f} ms per round (spread {med[arm][2] - med[arm][1]:.3f}), "
                  f"TTFT {statistics.median(x['ttft'] for x in rows):.3f} s, "
                  f"total {statistics.median(x['total'] for x in rows):.3f} s, N={len(v)}")
    if len(med) == 2:
        sep = med["T"][2] < med["B"][1]
        print(f"  T-B: {med['T'][0] - med['B'][0]:+.3f} ms per round ({(med['T'][0] / med['B'][0] - 1) * 100:+.2f}%), "
              f"every T cell under every B cell: {sep}")
    shas = {x["sha"] for rows in arms.values() for x in rows}
    print(f"  same text in every cell: {len(shas) == 1}")
