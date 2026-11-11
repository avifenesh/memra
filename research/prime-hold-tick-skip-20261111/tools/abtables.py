#!/usr/bin/env python3
"""Tables for the #1223 A/B on option A (memra#1223): stall rows and server logs in STALL_DIR,
the streamed normal census in NORMAL_DIR. Hold waits and the ticks to prime come from the tick
trace (`form_waits`, or `hold_waits` in the first form), not from the cell census.
usage: abtables.py STALL_DIR NORMAL_DIR"""
import collections
import json
import pathlib
import re
import sys

SPECK = re.compile(r"\[spec-k\] model=\"q\" .* K=(\d+) source=\S+ prompt=(\d+)")
WAITS = re.compile(r"(?:hold|form)_waits=(\d+) (?:hold|form)_wait_ms=([0-9.]+)")


def trace_cells(path, whole=False):
    lines = path.read_text(errors="replace").splitlines()
    cells, cur = [], []
    for line in lines:
        m = SPECK.search(line)
        if m and int(m.group(2)) > 40 and not whole:
            cur = []
            cells.append(cur)
            continue
        cur.append(line)
    if whole:
        cells = [lines]
    out = []
    for cell in cells:
        waits, pend, n_wait, ms = collections.Counter(), [], 0, 0.0
        for line in cell:
            m = SPECK.search(line)
            if m:
                if m.group(1) == "0":
                    pend.append(0)
                continue
            if not line.startswith("[tick]"):
                continue
            w = WAITS.search(line)
            if w:
                n_wait += int(w.group(1))
                ms += float(w.group(2))
            single = int(re.search(r"prefill_single_calls=(\d+)", line).group(1))
            batch = int(re.search(r"prefill_batch_calls=(\d+)", line).group(1))
            pend = [p + 1 for p in pend]
            k = single if batch == 0 else len(pend)
            for _ in range(min(k, len(pend))):
                waits[pend.pop(0)] += 1
        out.append((dict(sorted(waits.items())), n_wait, round(ms, 1)))
    return out


sd, nd = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
rows = [json.loads(l) for l in open(sd / "stall.jsonl") if l.strip()]
by_arm = collections.defaultdict(list)
for r in rows:
    by_arm[r["arm"]].append(r)
print("| boot | cell | long tokens | long ITL p50 / p99 ms | short n | short p50 / p95 s | plain primes: ticks to prime | waits (ms) |")
print("|---|---|---:|---|---:|---|---|---|")
for arm, rs in by_arm.items():
    tc = trace_cells(sd / f"server-{arm}.log")
    for r, t in zip(rs, tc):
        cell = f"{r['clients']}:{int(r['think_ms'])}"
        print(f"| {arm} | {cell} | {r['tokens_during']} | {r.get('long_itl_p50_ms')} / {r.get('long_itl_p99_ms')} | "
              f"{r['short_requests']} | {r['short_lat_p50_s']} / {r['short_lat_p95_s']} | {t[0] or '-'} | {t[1]} ({t[2]}) |")
print()
nrows = [json.loads(l) for l in open(nd / "normal.jsonl") if l.strip()]
ref = {}
for r in nrows:
    if r["arm"].startswith("main"):
        for q in r["rows"]:
            ref.setdefault((r["c"], q["prompt"]), q.get("sha"))
print("| boot | c | agg tok/s | TTFT p50 s | ITL p50 / p99 ms | K at admission | texts equal to main |")
print("|---|---:|---:|---:|---|---|---|")
for r in nrows:
    ks = collections.Counter(r.get("spec_k", []))
    eq = sum(1 for q in r["rows"] if ref.get((r["c"], q["prompt"])) == q.get("sha"))
    log = nd / f"server-{r['arm']}.log"
    print(f"| {r['arm']} | {r['c']} | {r['agg_tok_s']} | {r.get('ttft_p50_s')} | {r.get('itl_p50_ms')} / {r.get('itl_p99_ms')} | "
          f"{' '.join(f'{k} x{v}' for k, v in sorted(ks.items()))} | {eq} of {len(r['rows'])} |")
waits_per_boot = {}
for log in sorted(nd.glob("server-*.log")):
    t = trace_cells(log, whole=True)[0]
    waits_per_boot[log.stem.replace("server-", "")] = t
print()
print("normal census, whole boot from the trace (ticks to prime for plain admissions, waits):")
for k, v in waits_per_boot.items():
    print(f"- {k}: {v[0]}, waits {v[1]} ({v[2]} ms)")
