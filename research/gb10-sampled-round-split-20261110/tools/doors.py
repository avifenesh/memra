#!/usr/bin/env python3
"""doors.py CELL_DIR: the engine's own clocks for one boot of the six requests, no profiler.

Reads server.log and reqs.jsonl of a cell that ran with MEMRA_SPEC_PHASE=1 (one `[spec-phase]`
line per burst: the round loop's wall split into draft, verify-issue, verify-wait, commit-host)
or with MEMRA_SPEC_SETUP_TRACE=1 (one `[spec-setup]` line per burst: init feed, capture, fill,
the round loop, the tail). Requests are the `[spec-k]` lines after the server starts listening,
in the driver's order. Per request it prints the sums per round and what the client's decode
time (total less TTFT) leaves outside them.
"""
import json
import pathlib
import re
import sys

d = pathlib.Path(sys.argv[1])
reqs = [json.loads(l) for l in open(d / "reqs.jsonl") if l.strip()]
log = (d / "server.log").read_text(errors="replace").split("[server] listening", 1)[1].splitlines()
blocks = []
for line in log:
    if line.startswith("[spec-k] model="):
        blocks.append([])
    elif blocks:
        blocks[-1].append(line)
assert len(blocks) == len(reqs), (len(blocks), len(reqs))
PH = re.compile(r"\[spec-phase\] draft=([\d.]+)ms .* verify-issue=([\d.]+)ms .* verify-wait=([\d.]+)ms .* commit-host=([\d.]+)ms .* rounds=(\d+)")
ST = re.compile(r"\[spec-setup\] init=([\d.]+)ms cap=([\d.]+)ms fill=([\d.]+)ms rounds=([\d.]+)ms tail=([\d.]+)ms total=([\d.]+)ms out=(\d+) cont=(\w+)")
phase = any(PH.search(l) for b in blocks for l in b)
setup = any(ST.search(l) for b in blocks for l in b)
if phase:
    print("MEMRA_SPEC_PHASE=1: the round loop on the engine's clock, ms per round; outside = client decode time less the round loops")
    print(f"{'class':9s} {'bursts':>6s} {'rounds':>6s} {'draft':>8s} {'verify':>8s} {'commit-host':>11s} {'in the round loop':>18s} {'client decode':>14s} {'outside':>8s} {'outside per burst':>18s}")
    for q, b in zip(reqs, blocks):
        rows = [tuple(float(x) for x in m.groups()) for m in (PH.search(l) for l in b) if m]
        n = int(sum(r[4] for r in rows))
        dr, vi, vw, ch = (sum(r[i] for r in rows) for i in range(4))
        dec = (q["total_s"] - q["ttft_s"]) * 1e3
        inside = dr + vi + vw + ch
        print(f"{q['class']:9s} {len(rows):6d} {n:6d} {dr / n:8.2f} {(vi + vw) / n:8.2f} {ch / n:11.2f} {inside / n:18.2f} "
              f"{dec / n:14.2f} {(dec - inside) / n:8.2f} {(dec - inside) / len(rows):18.1f}")
if setup:
    print("MEMRA_SPEC_SETUP_TRACE=1: each burst on the engine's clock; ms per burst (mean), then ms per round of the request")
    print(f"{'class':9s} {'bursts':>6s} {'rounds':>6s} {'init feed':>10s} {'capture':>8s} {'fill':>6s} {'round loop':>11s} {'tail':>7s}   per round: {'init':>6s} {'capture':>8s} {'fill':>6s} {'loop':>8s} {'tail':>6s} {'sum':>8s} {'client decode':>14s}")
    for q, b in zip(reqs, blocks):
        rows = [m.groups() for m in (ST.search(l) for l in b) if m]
        v = [[float(x) for x in r[:6]] for r in rows]
        n = q["rounds"]
        s = [sum(r[i] for r in v) for i in range(6)]
        nb = len(v)
        dec = (q["total_s"] - q["ttft_s"]) * 1e3
        # the first burst's init holds the prime: leave the cold burst's init out of the means
        cont = [r for r, raw in zip(v, rows) if raw[7] == "true"]
        mean = lambda i, rr: sum(r[i] for r in rr) / max(1, len(rr))
        print(f"{q['class']:9s} {nb:6d} {n:6d} {mean(0, cont):10.2f} {mean(1, v):8.2f} {mean(2, v):6.2f} {mean(3, v):11.1f} {mean(4, v):7.2f}"
              f"              {sum(r[0] for r in cont) / n:6.2f} {s[1] / n:8.2f} {s[2] / n:6.2f} {s[3] / n:8.2f} {s[4] / n:6.2f} "
              f"{(sum(r[0] for r in cont) + s[1] + s[2] + s[3] + s[4]) / n:8.2f} {dec / n:14.2f}")
    print("init feed: continuation bursts only (the cold burst's init holds the prime). The door adds one stream synchronize per burst, at its tail.")
