#!/usr/bin/env python3
"""rounds_tab.py CAMPAIGN_DIR: per cell (NN-B / NN-T with ab.jsonl and a MEMRA_TICK_TRACE=1 server log)
the decode cost per spec round inside full bursts: every generating `[tick-spec]` burst of a request
except its first (which holds the prime) and its last (cut by the stop), wall over rounds. Then per arm
the median of the cells' values with its spread, the T-B change, and whether the cells served the same
texts."""
import json, pathlib, re, statistics, sys

d = pathlib.Path(sys.argv[1])
ts = re.compile(r"(\d+\.\d+) \[tick-spec\].* wall_ms=([\d.]+) generated=(\d+)->(\d+) rounds=(\d+)")
cells = {}
for c in sorted(p for p in d.iterdir() if p.is_dir() and re.match(r"\d\d-[BT]$", p.name)):
    f = c / "ab.jsonl"
    if not f.exists():
        continue
    ticks = [(float(m.group(1)), float(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)))
             for l in (c / "server.log").read_text().splitlines() if (m := ts.match(l))]
    wall = rounds = 0
    shas = []
    for r in (json.loads(l) for l in f.read_text().splitlines() if l.strip()):
        t0, t1 = r["send_unix"], r["send_unix"] + r["total_s"] + 0.05
        # a burst belongs to the request whose window holds its whole wall (its end stamp alone can tie
        # with the next request's send time)
        g = [t for t in ticks if t0 <= t[0] - t[1] / 1000 + 0.001 and t[0] <= t1 and t[3] > t[2]]
        for t in g[1:-1]:
            wall += t[1]; rounds += t[4]
        shas.append(r["text_sha256"][:12])
    cells[c.name] = (wall / rounds if rounds else None, rounds, tuple(shas))
    print(f"{c.name}: {cells[c.name][0]:.3f} ms per round over {rounds} rounds in full bursts, texts {' '.join(shas)}")
med = {}
for arm in "BT":
    v = [x[0] for k, x in cells.items() if k.endswith(arm) and x[0]]
    if v:
        med[arm] = statistics.median(v)
        print(f"{arm}: median {med[arm]:.3f} ms per round (spread {max(v) - min(v):.3f}), N={len(v)}")
if len(med) == 2:
    print(f"T-B: {med['T'] - med['B']:+.3f} ms per round ({(med['T'] / med['B'] - 1) * 100:+.2f}%)")
print("same texts in every cell:", len({x[2] for x in cells.values()}) == 1)
