#!/usr/bin/env python3
"""specpub_tab.py CAMPAIGN_DIR [REQ]: the spec boundary publication's served B/T, one context.

Per cell (NN-B, NN-T, NN-U with ab.jsonl and a MEMRA_TICK_TRACE=1 server log with a unix time on
every line, or its trace.txt) the cold request REQ (default: the last row of ab.jsonl) is read as
its generating `[tick-spec]` bursts. The publication runs between the first burst and the second:
the server prints burst 1's line, publishes (`[prefix-cache] insert (spec-boundary)`), then runs
burst 2 and prints its line with its own wall time.

  pub     insert line - burst 1 line: the publisher's time on the scheduler thread
  gap     (burst 2 line - burst 1 line) - burst 2's wall: everything between the two bursts
  b2      burst 2's wall over its rounds
  rest    wall over rounds of every burst after the second (the last ends on EOS)
  stall   (burst 2 line - burst 1 line) - burst 2's rounds x rest: the time the first burst
          boundary costs over the rounds it holds, whether the publisher spends it on the
          scheduler thread or its copies delay burst 2's first kernels
  round   the lane's ms per round: every burst but the first and the last (burst 2 included)

Then per arm the median of each over the cells with its spread (max - min), every other arm
against B (and U against T), whether every cell of one arm is under every cell of the other, and
whether every cell served the same text and the same bursts."""
import json
import pathlib
import re
import statistics
import sys

d = pathlib.Path(sys.argv[1])
want = sys.argv[2] if len(sys.argv) > 2 else None
ts = re.compile(r"(\d+\.\d+) \[tick-spec\].* wall_ms=([\d.]+) generated=(\d+)->(\d+) rounds=(\d+) drafted=(\d+) accepted=(\d+)")
ins = re.compile(r"(\d+\.\d+) \[prefix-cache\] insert \(spec-boundary\): (\d+) tokens, ([\d.]+)MB")
cells = {}
for c in sorted(p for p in d.iterdir() if p.is_dir() and re.match(r"\d\d-[A-Z]$", p.name)):
    log = c / ("server.log" if (c / "server.log").exists() else "trace.txt")
    if not log.exists() or not (c / "ab.jsonl").exists():
        print(f"{c.name}: no log or no ab.jsonl yet; skipped")
        continue
    lines = log.read_text(errors="replace").splitlines()
    ticks = [(float(m.group(1)), float(m.group(2)), *(int(x) for x in m.groups()[2:]))
             for l in lines if (m := ts.match(l))]
    inserts = [(float(m.group(1)), int(m.group(2)), float(m.group(3))) for l in lines if (m := ins.match(l))]
    rows = [json.loads(l) for l in (c / "ab.jsonl").read_text().splitlines() if l.strip()]
    r = next((x for x in rows if x["req"] == want), None) if want else rows[-1]
    if r is None:
        print(f"{c.name}: no {want} row yet; skipped")
        continue
    t0, t1 = r["send_unix"], r["send_unix"] + r["total_s"] + 0.05
    g = [t for t in ticks if t0 <= t[0] - t[1] / 1000 + 0.001 and t[0] <= t1 and t[3] > t[2]]
    if len(g) < 3:
        print(f"{c.name}: {len(g)} generating bursts, need 3; skipped")
        continue
    b1, b2 = g[0], g[1]
    pub = [i for i in inserts if b1[0] <= i[0] <= b2[0]]
    rest = g[2:]
    rest_ms = sum(t[1] for t in rest) / sum(t[4] for t in rest)
    mid = g[1:-1]
    interval = (b2[0] - b1[0]) * 1000
    cells[c.name] = {
        "arm": c.name[-1], "req": r["req"], "prompt": r["prompt_tokens"], "tok": r["completion_tokens"],
        "ttft": r["ttft_s"], "total": r["total_s"], "sha": r["text_sha256"], "finish": r["finish"],
        "entry_tokens": pub[0][1] if pub else None, "entry_mb": pub[0][2] if pub else None,
        "pub": (pub[0][0] - b1[0]) * 1000 if pub else None,
        "gap": interval - b2[1], "interval": interval,
        "b2": b2[1] / b2[4], "b2_rounds": b2[4], "rest": rest_ms, "rest_rounds": sum(t[4] for t in rest),
        "stall": interval - b2[4] * rest_ms,
        "round": sum(t[1] for t in mid) / sum(t[4] for t in mid),
        "bursts": [(t[2], t[3], t[4], t[5], t[6]) for t in g],
    }

KEYS = [("stall", "first burst boundary over its rounds, ms"), ("pub", "publisher on the scheduler thread, ms"),
        ("gap", "between burst 1 and burst 2, ms"), ("interval", "burst 1 line to burst 2 line, ms"),
        ("b2", "burst 2, ms per round"), ("rest", "later bursts, ms per round"),
        ("round", "ms per round, every burst but the first and last"), ("ttft", "TTFT, s"), ("total", "total, s")]
for name, x in cells.items():
    print(f"{name}: {x['req']} prompt {x['prompt']}, entry {x['entry_tokens']} tokens {x['entry_mb']} MB; "
          f"stall {x['stall']:.1f} pub {x['pub']:.1f} gap {x['gap']:.1f} interval {x['interval']:.1f} ms; "
          f"burst 2 {x['b2']:.2f} ms per round over {x['b2_rounds']}, later {x['rest']:.2f} over {x['rest_rounds']}, "
          f"round {x['round']:.2f}; TTFT {x['ttft']:.3f} s, total {x['total']:.3f} s, {x['tok']} tok {x['finish']}, "
          f"text {x['sha'][:12]}")
arms = sorted({x["arm"] for x in cells.values()})


def fmt(k):
    return "{:.3f}" if k in ("ttft", "total") else "{:.2f}" if k in ("b2", "rest", "round") else "{:.1f}"


def vals(k, a):
    return [x[k] for x in cells.values() if x["arm"] == a and x[k] is not None]


print()
print("| | " + " | ".join(f"{a} median (spread)" for a in arms) + " |")
print("|---|" + "---:|" * len(arms))
for k, label in KEYS:
    f = fmt(k)
    row = []
    for a in arms:
        v = vals(k, a)
        row.append(f"{f.format(statistics.median(v))} ({f.format(max(v) - min(v))})" if v else "")
    print(f"| {label} | " + " | ".join(row) + " |")
pairs = [(a, "B") for a in arms if a != "B" and "B" in arms]
pairs += [(a, r) for a, r in (("U", "T"), ("W", "T"), ("W", "U")) if a in arms and r in arms]
for a, ref in pairs:
    print()
    print(f"| {a} against {ref} | {a} - {ref} | every {a} cell under every {ref} cell |")
    print("|---|---:|---|")
    for k, label in KEYS:
        va, vr = vals(k, a), vals(k, ref)
        if not va or not vr:
            continue
        f = fmt(k)
        ma, mr = statistics.median(va), statistics.median(vr)
        pct = f" ({(ma - mr) / mr * 100:+.1f}%)" if mr else ""
        under = "yes" if max(va) < min(vr) else (f"no, every {a} over every {ref}" if min(va) > max(vr) else "no, ranges overlap")
        print(f"| {label} | {f.format(ma - mr)}{pct} | {under} |")
n = ", ".join(f"{sum(1 for x in cells.values() if x['arm'] == a)} {a}" for a in arms)
shas = {x["sha"] for x in cells.values()}
bursts = {json.dumps(x["bursts"]) for x in cells.values()}
entries = {(x["entry_tokens"], x["entry_mb"]) for x in cells.values()}
print(f"\ncells: {n}; texts: {len(shas)} distinct sha256 ({', '.join(sorted(s[:12] for s in shas))}); "
      f"burst structure (generated, rounds, drafted, accepted): {len(bursts)} distinct; "
      f"published entry (tokens, MB): {sorted(entries)}")
