#!/usr/bin/env python3
"""memra#1258: one line per fresh-boot margin row (b-fresh.sh): the gate's line, the pool's reserve
at the prime's two calls and whether the row hit an out-of-memory line, per arm.
usage: fresh-table.py LEN MARGIN ARM...   (reads sweep/cap-ARM-fresh-LEN-mMARGIN/sSEED/serve.log)"""
import re
import sys
from pathlib import Path

B = Path.home() / ".cache/memra-1258/sweep"
length, margin, arms = sys.argv[1], sys.argv[2], sys.argv[3:]
RE_PASS = re.compile(r"gate-pass id=(\S+) prompt=(\d+) path=(\w+) .*required=(\d+)MB .*limiting_free=(\d+)MB")
RE_SNAP = re.compile(r"\[diag1255\] (dspark-prime-\S+) id=(\S+)(?: rows=\d+)?: driver_free=(\d+)MB pool_res=(\d+)MB pool_used=(\d+)MB pool_unused=(\d+)MB")
for arm in arms:
    fails = rows = 0
    for d in sorted((B / f"cap-{arm}-fresh-{length}-m{margin}").glob("s*")):
        text = (d / "serve.log").read_text(errors="replace")
        lines = text.splitlines()
        gate = [m for l in lines if (m := RE_PASS.search(l)) and int(m.group(2)) > 1000 and m.group(3) == "spec"]
        if len(gate) < 2:
            print(f"{arm} {d.name}: no placed row")
            continue
        g = gate[1]  # the first is the unplaced prime that learns `required`
        rid = g.group(1)
        snaps = [(m.group(1), int(m.group(3)), int(m.group(4)), int(m.group(5)), int(m.group(6)))
                 for l in lines if (m := RE_SNAP.search(l)) and m.group(2) == rid]
        oom = [l for l in lines if "out of memory" in l]
        rows += 1
        fails += bool(oom)
        seq = " -> ".join(f"{name.replace('dspark-prime-', '')} res {res} free {free}" for name, free, res, _used, _unused in snaps)
        where = ""
        if oom:
            m = re.search(r"\[dev0 free (\d+)MB pool-res (\d+)MB pool-used (\d+)MB\]", oom[0])
            done = any(name == "dspark-prime-done" for name, *_ in snaps)
            where = f" | OOM {'after the prime' if done else 'in call ' + str(sum(1 for n, *_ in snaps if n == 'dspark-prime-chunk-before'))}: free {m.group(1)} res {m.group(2)} used {m.group(3)}" if m else " | OOM"
        print(f"{arm} {d.name}: prompt {g.group(2)} required {g.group(4)} eff_free {g.group(5)} seen {int(g.group(5)) - int(g.group(4))} | {seq}{where}")
    print(f"{arm}: {fails} of {rows} rows with an out-of-memory line")
