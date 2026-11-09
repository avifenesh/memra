#!/usr/bin/env python3
"""eos_tab.py CAMPAIGN_DIR: round 5 lever 1's served A/B. Per cell and request (eos_ab.py rows): total
time, decode time after the first token, the text sha, and from the timestamped `[tick-spec]` lines in
the request's window the rounds of its last burst and the tokens that burst kept. Then per request
and arm the median total and decode time with their spread (max - min), the T-B change, and whether
every cell of the request served the same text."""
import json, pathlib, re, statistics, sys

d = pathlib.Path(sys.argv[1])
ts = re.compile(r"(\d+\.\d+) \[tick-spec\].* wall_ms=([\d.]+) generated=(\d+)->(\d+) rounds=(\d+)")
rows = {}
for c in sorted(p for p in d.iterdir() if p.is_dir() and re.match(r"\d\d-[BT]$", p.name)):
    f = c / "ab.jsonl"
    if not f.exists():
        continue
    ticks = [(float(m.group(1)), float(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)))
             for l in (c / "server.log").read_text().splitlines() if (m := ts.match(l))]
    for r in (json.loads(l) for l in f.read_text().splitlines() if l.strip()):
        t0, t1 = r["send_unix"], r["send_unix"] + r["total_s"] + 0.05
        g = [t for t in ticks if t0 <= t[0] <= t1 and t[3] > t[2]]
        last = g[-1] if g else None
        dec = r["total_s"] - r["ttft_s"]
        rows.setdefault(r["req"], []).append((c.name, r["total_s"], dec, r["text_sha256"][:12], r["completion_tokens"],
                                              last[4] if last else None, (last[3] - last[2]) if last else None,
                                              last[1] if last else None))
for req, v in rows.items():
    print(f"== {req}")
    for name, tot, dec, sha, n, lr, lk, lw in v:
        print(f"  {name}: total {tot:.3f} s, decode {dec:.3f} s, {n} tok, text {sha}, last burst {lr} rounds for {lk} kept tokens ({lw} ms)")
    med = {}
    for arm in "BT":
        a = [x for x in v if x[0].endswith(arm)]
        if a:
            t = [x[1] for x in a]; dd = [x[2] for x in a]
            med[arm] = (statistics.median(t), statistics.median(dd))
            print(f"  {arm}: total median {med[arm][0]:.3f} s (spread {max(t) - min(t):.3f}), decode median {med[arm][1]:.3f} s (spread {max(dd) - min(dd):.3f}), N={len(a)}")
    if len(med) == 2:
        print(f"  T-B: total {med['T'][0] - med['B'][0]:+.3f} s ({(med['T'][0] / med['B'][0] - 1) * 100:+.1f}%), decode {med['T'][1] - med['B'][1]:+.3f} s ({(med['T'][1] / med['B'][1] - 1) * 100:+.1f}%)")
    print(f"  same text in every cell: {len({x[3] for x in v}) == 1}")
