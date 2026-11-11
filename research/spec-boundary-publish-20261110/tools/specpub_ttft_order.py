#!/usr/bin/env python3
"""specpub_ttft_order.py CAMPAIGN_DIR: per cell, where the spec-boundary publication sits against
the first token of the context's request (the last row of ab.jsonl). The first token is sent at
send_unix + ttft_s (client clock = server clock: one host). The publication is the
`[prefix-cache] insert (spec-boundary)` line after the request was sent, and burst 1 is the
`[tick-spec]` line just before it (its wall_ms covers burst 1's rounds). Printed per cell: the
first token's time, burst 1's start (its line minus its wall), the insert line, and the insert
line minus the first token (positive: the publication ran after the first token went out)."""
import json
import pathlib
import re
import sys

camp = pathlib.Path(sys.argv[1])
for cell in sorted(p for p in camp.iterdir() if p.is_dir() and re.match(r"\d\d-[A-Z]$", p.name)):
    rows = [json.loads(l) for l in (cell / "ab.jsonl").read_text().splitlines() if l.startswith("{")]
    r = rows[-1]
    first = r["send_unix"] + r["ttft_s"]
    lines = (cell / "trace.txt").read_text(errors="replace").splitlines()
    prev_tick, ins, b1 = None, None, None
    for l in lines:
        t = float(l.split(" ", 1)[0])
        if t < r["send_unix"]:
            continue
        if "[tick-spec]" in l and ins is None:
            prev_tick = (t, float(re.search(r"wall_ms=([\d.]+)", l).group(1)))
        if "insert (spec-boundary)" in l:
            ins, b1 = t, prev_tick
            break
    if ins is None or b1 is None:
        print(f"{cell.name}: no spec-boundary insert after the request")
        continue
    b1_start = b1[0] - b1[1] / 1e3
    print(f"{cell.name}: ttft {r['ttft_s']:.3f} s; burst 1 starts {1e3 * (b1_start - first):+.1f} ms from the first token, "
          f"its line {1e3 * (b1[0] - first):+.1f} ms, the insert line {1e3 * (ins - first):+.1f} ms")
