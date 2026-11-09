#!/usr/bin/env python3
"""r5_tab.py RUN_DIR: per served request of a round-5 decode cell (pf.jsonl, the timestamped server log
with MEMRA_TICK_TRACE=1): prompt tokens, TTFT, decode tok/s after the first token, and from the
generating `[tick-spec]` lines the rounds, drafted and accepted counts, the acceptance per drafted
token, tokens per round and the tick wall per round (the last tick, cut by the stop, listed apart)."""
import json, pathlib, re, sys

d = pathlib.Path(sys.argv[1])
reqs = [json.loads(l) for l in (d / "pf.jsonl").read_text().splitlines() if l.strip()]
ts = re.compile(r"(\d+\.\d+) \[tick-spec\].* wall_ms=([\d.]+) generated=(\d+)->(\d+) rounds=(\d+) drafted=(\d+) accepted=(\d+)")
ticks = [tuple(float(x) for x in m.groups()) for l in (d / "server.log").read_text().splitlines() if (m := ts.match(l))]
for r in reqs:
    t0, t1 = r["send_unix"], r["send_unix"] + r["total_s"] + 0.5
    g = [t for t in ticks if t0 <= t[0] <= t1 and t[3] > t[2]]
    dec = r["total_s"] - r["ttft_s"]
    toks = r["completion_tokens"] - 1
    full = g[:-1]
    R = sum(t[4] for t in full); Dr = sum(t[5] for t in full); A = sum(t[6] for t in full); W = sum(t[1] for t in full)
    gen = sum(t[3] - t[2] for t in full)
    last = g[-1] if g else None
    print(f"n={r['prompt_tokens']}: ttft {r['ttft_s']:.3f} s, decode {toks} tok in {dec:.3f} s = {toks / dec:.2f} tok/s; "
          f"full ticks: {int(R)} rounds, acceptance {A / Dr:.3f} per drafted token, {gen / R:.2f} tok/round, "
          f"{W / R:.2f} ms/round (first tick holds the prime)"
          + (f"; last tick {last[1]:.0f} ms, {int(last[4])} rounds for {int(last[3] - last[2])} kept tokens, {last[1] / last[4]:.2f} ms/round" if last else ""))
    for t in g:
        print(f"   tick {int(t[2])}->{int(t[3])} wall {t[1]:.1f} ms rounds {int(t[4])} drafted {int(t[5])} accepted {int(t[6])}: {t[1] / t[4]:.2f} ms/round")
