#!/usr/bin/env python3
"""tf_phases.py CAMPAIGN_DIR: lever 8's primary signal from the per-call trace of each served cell
(MEMRA_TICK_TRACE=1: `[prime-chunk] phase=mtp-trunk rows=N wall_ms=W`). Per request, the trunk's
calls are taken from the end of the log back until their rows reach the prompt length (pf.jsonl's
prompt_tokens);
`end` is the wall time of the trunk calls after its last full 4,096-row call: on B the boundary
call plus the separate tail call, on T the one folded call. Prints per cell, then per arm and
prompt the medians, the spread (max - min) and T - B."""
import json, pathlib, re, statistics, sys

d = pathlib.Path(sys.argv[1])
pat = re.compile(r"\[prime-chunk\] phase=mtp-trunk rows=(\d+) wall_ms=([\d.]+)")
res = {}
for cell in sorted(p for p in d.iterdir() if p.is_dir() and re.match(r"\d\d-[BT]$", p.name)):
    arm = cell.name[-1]
    if not (cell / "pf.jsonl").exists():
        continue
    reqs = [json.loads(l) for l in (cell / "pf.jsonl").read_text().splitlines() if l.strip()]
    calls = [(int(m.group(1)), float(m.group(2))) for m in pat.finditer((cell / "server.log").read_text())]
    # From the end: boot-time warm-up primes, if any, come before the first request.
    i = len(calls)
    seqs = []
    for r in reversed(reqs):
        n, rows, seq = r["prompt_tokens"], 0, []
        while i > 0 and rows < n:
            i -= 1; seq.insert(0, calls[i]); rows += calls[i][0]
        seqs.insert(0, (r, n, rows, seq))
    for r, n, rows, seq in seqs:
        if rows != n:
            print(f"{cell.name}: trunk calls cover {rows} of {n} rows; skipped")
            continue
        last_full = max((k for k, (c, _) in enumerate(seq) if c == 4096), default=-1)
        end = seq[last_full + 1:]
        end_s = sum(w for _, w in end) / 1e3
        trunk_s = sum(w for _, w in seq) / 1e3
        res.setdefault((n, arm), []).append((cell.name, end_s, trunk_s, r["ttft_s"], [c for c, _ in end]))
        print(f"{cell.name} n={n}: end calls {[c for c, _ in end]} {end_s:.3f} s, trunk {trunk_s:.3f} s, ttft {r['ttft_s']:.3f} s")
print()
for n in sorted({k[0] for k in res}):
    row = {}
    for arm in ("B", "T"):
        v = res.get((n, arm), [])
        if not v:
            continue
        e = [x[1] for x in v]; tr = [x[2] for x in v]; tt = [x[3] for x in v]
        row[arm] = (statistics.median(e), max(e) - min(e), statistics.median(tr), statistics.median(tt), max(tt) - min(tt))
        print(f"n={n} {arm}: end {statistics.median(e):.3f} s (spread {max(e) - min(e):.3f}), trunk {statistics.median(tr):.3f} s, "
              f"ttft {statistics.median(tt):.3f} s (spread {max(tt) - min(tt):.3f}), N={len(v)}")
    if "B" in row and "T" in row:
        b, t = row["B"], row["T"]
        print(f"n={n} T - B: end {t[0] - b[0]:+.3f} s (spreads {b[1]:.3f} / {t[1]:.3f}), "
              f"ttft {t[3] - b[3]:+.3f} s ({100 * (t[3] / b[3] - 1):+.2f}%, spreads {b[4]:.3f} / {t[4]:.3f})")
