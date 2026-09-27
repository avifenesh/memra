#!/usr/bin/env python3
"""Day-34 BOX4 reading (added after the sitting, log only, no clause): where each promote's first poll ends
against its tick top, per run. The poll stamp is taken after the settle returns (worker.rs, `poll {} at`), so
`end - top` is that poll's owner time plus whatever precedes it in the tick. Usage: poll1-reading.py RUN_DIR..."""
import pathlib
import re
import statistics as st
import sys

P1 = re.compile(r"poll 1 at \+([0-9.]+)ms \(tick \d+, its top \+([0-9.]+)ms\) pending")
P2 = re.compile(r"poll 2 at \+([0-9.]+)ms \(tick \d+, its top \+([0-9.]+)ms\) complete")


def stats(xs):
    if not xs:
        return "N=0"
    return f"N={len(xs)} median={st.median(xs):.2f} min={min(xs):.2f} max={max(xs):.2f}"


for run in sys.argv[1:]:
    p1, p2 = [], []
    for log in sorted(pathlib.Path(run).rglob("server.log")):
        for line in log.read_text(errors="replace").splitlines():
            if "promote published off the tick" not in line:
                continue
            m = P1.search(line)
            if m:
                p1.append(float(m.group(1)) - float(m.group(2)))
            m = P2.search(line)
            if m:
                p2.append(float(m.group(1)) - float(m.group(2)))
    print(f"DAY34 POLL READING run={run} poll1-end-minus-its-top {stats(p1)} | poll2-end-minus-its-top {stats(p2)}")
