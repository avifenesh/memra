#!/usr/bin/env python3
"""q45_tab.py CAMPAIGN_DIR: the q45k MMQ_X probe's cells: per cell the draft fill's wall (the sum of
`[prime-chunk] phase=mtp-draft-fill` calls), the trunk's, TTFT, the answer's sha and the
speculative trace (each `[tick-spec]` line's drafted/accepted counts) after the prime; then per arm
the medians and spreads, and whether every cell's answer and trace are the same."""
import json, pathlib, re, statistics, sys

d = pathlib.Path(sys.argv[1])
pc = re.compile(r"\[prime-chunk\] phase=(\S+) rows=(\d+) wall_ms=([\d.]+)")
sp = re.compile(r"\[tick-spec\].* generated=(\d+)->(\d+) rounds=(\d+) drafted=(\d+) accepted=(\d+)")
cells = {}
for c in sorted(p for p in d.iterdir() if p.is_dir() and re.match(r"\d\d-[BT]$", p.name)):
    pf = c / "pf.jsonl"
    if not pf.exists() or not pf.read_text().strip():
        continue
    r = [json.loads(l) for l in pf.read_text().splitlines() if l.strip()][-1]
    log = (c / "server.log").read_text()
    fill = sum(float(m.group(3)) for m in pc.finditer(log) if m.group(1) == "mtp-draft-fill") / 1e3
    trunk = sum(float(m.group(3)) for m in pc.finditer(log) if m.group(1) == "mtp-trunk") / 1e3
    trace = tuple((int(m.group(4)), int(m.group(5))) for m in sp.finditer(log) if int(m.group(2)) > int(m.group(1)))
    cells[c.name] = (fill, trunk, r["ttft_s"], r["text_sha256"][:12], trace)
    print(f"{c.name}: fill {fill:.4f} s, trunk {trunk:.3f} s, ttft {r['ttft_s']:.3f} s, answer {r['text_sha256'][:12]}, {len(trace)} spec ticks")
for arm in "BT":
    v = [x for k, x in cells.items() if k.endswith(arm)]
    if v:
        f = [x[0] for x in v]; t = [x[2] for x in v]
        print(f"{arm}: fill median {statistics.median(f):.4f} s (spread {max(f) - min(f):.4f}), ttft median {statistics.median(t):.3f} s (spread {max(t) - min(t):.3f}), N={len(v)}")
print("answers identical:", len({x[3] for x in cells.values()}) == 1, "; spec traces identical:", len({x[4] for x in cells.values()}) == 1)
