#!/usr/bin/env python3
"""r6_ktab.py CELL_DIR...: per served request of a probe cell (pf.jsonl, the timestamped server log with
MEMRA_TICK_TRACE=1): rounds over the whole answer, tokens per round, ms per round over every burst but
the first (which holds the prime), decode tok/s after the first token, and the text's sha256 head."""
import json, pathlib, re, sys
ts = re.compile(r"(\d+\.\d+) \[tick-spec\].* wall_ms=([\d.]+) generated=(\d+)->(\d+) rounds=(\d+) drafted=(\d+) accepted=(\d+)")
for a in sys.argv[1:]:
    d = pathlib.Path(a)
    ticks = [tuple(float(x) for x in m.groups()) for l in (d / "server.log").read_text(errors="replace").splitlines() if (m := ts.match(l))]
    for l in (d / "pf.jsonl").read_text().splitlines():
        r = json.loads(l)
        g = [t for t in ticks if r["send_unix"] <= t[0] <= r["send_unix"] + r["total_s"] + 0.5 and t[3] > t[2]]
        rounds = sum(t[4] for t in g)
        rest = g[1:]
        mpr = sum(t[1] for t in rest) / max(1, sum(t[4] for t in rest))
        toks = r["completion_tokens"] - 1
        print(f"{d.name:10s} n={r['prompt_tokens']:6d} rounds {int(rounds):3d} drafted {int(sum(t[5] for t in g)):3d} accepted {int(sum(t[6] for t in g)):3d} "
              f"tok/round {r['completion_tokens'] / rounds:.2f} ms/round {mpr:7.2f} decode {toks / (r['total_s'] - r['ttft_s']):6.2f} tok/s "
              f"ttft {r['ttft_s']:.2f} s sha {r['text_sha256'][:12]}")
