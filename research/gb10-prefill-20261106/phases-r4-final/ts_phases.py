#!/usr/bin/env python3
"""ts_phases.py RUN_DIR: the phases of each served request from the timestamped server log
(tsrun.py prefixes unix time; MEMRA_TICK_TRACE=1 prints `[prime-chunk] phase=... rows=N wall_ms=W`
and `[tick-spec] ... generated=a->b`). Per request (pf.jsonl, in order): the trunk calls' wall sum and
count, the draft fill's, the time from the request's arrival to its first prime call's start, from
the trunk's end to the fill's start, from the fill's end to the first tick that generated a token,
and that tick's wall; against the measured TTFT."""
import json, pathlib, re, sys

d = pathlib.Path(sys.argv[1])
reqs = [json.loads(l) for l in (d / "pf.jsonl").read_text().splitlines() if l.strip()]
lines = []
for l in (d / "server.log").read_text().splitlines():
    m = re.match(r"(\d+\.\d+) (.*)", l)
    if m:
        lines.append((float(m.group(1)), m.group(2)))
pc = re.compile(r"\[prime-chunk\] phase=(\S+) rows=(\d+) wall_ms=([\d.]+)")
ts = re.compile(r"\[tick-spec\].* wall_ms=([\d.]+) generated=(\d+)->(\d+)")
for r in reqs:
    t_send = r["send_unix"]
    t_tok = t_send + r["ttft_s"]
    win = [(t, s) for t, s in lines if t_send <= t <= t_tok + 0.5]
    trunk = [(t, int(m.group(2)), float(m.group(3))) for t, s in win if (m := pc.search(s)) and m.group(1) == "mtp-trunk"]
    fill = [(t, int(m.group(2)), float(m.group(3))) for t, s in win if (m := pc.search(s)) and m.group(1) == "mtp-draft-fill"]
    gen = [(t, float(m.group(1)), int(m.group(2)), int(m.group(3))) for t, s in win if (m := ts.search(s)) and int(m.group(3)) > int(m.group(2))]
    if not trunk:
        print(f"n={r['prompt_tokens']}: no trunk lines in the window")
        continue
    t_first = trunk[0][0] - trunk[0][2] / 1e3
    t_trunk_end = trunk[-1][0]
    out = {
        "n": r["prompt_tokens"], "ttft": r["ttft_s"],
        "host_before_first_call": t_first - t_send,
        "trunk_calls": len(trunk), "trunk_wall": sum(w for _, _, w in trunk) / 1e3,
        "trunk_span": t_trunk_end - t_first,
        "fill_calls": len(fill), "fill_wall": sum(w for _, _, w in fill) / 1e3,
    }
    if fill:
        out["trunk_end_to_fill"] = fill[0][0] - fill[0][2] / 1e3 - t_trunk_end
        out["fill_span"] = fill[-1][0] - (fill[0][0] - fill[0][2] / 1e3)
        t_last = fill[-1][0]
    else:
        t_last = t_trunk_end
    if gen:
        out["to_first_generating_tick_end"] = gen[0][0] - t_last
        out["first_generating_tick_wall"] = gen[0][1] / 1e3
    out["first_tick_end_to_token"] = t_tok - (gen[0][0] if gen else t_last)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}))
