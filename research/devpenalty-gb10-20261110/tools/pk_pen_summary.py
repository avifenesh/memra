#!/usr/bin/env python3
"""Summary of a device penalty arm A/B (pk_pen.py): off against on.
usage: pk_pen_summary.py DIR   (DIR holds runs.jsonl and groups.jsonl)"""
import json
import pathlib
import statistics
import sys

D = pathlib.Path(sys.argv[1])
runs = [json.loads(l) for l in open(D / "runs.jsonl") if l.strip()]
groups = [json.loads(l) for l in open(D / "groups.jsonl") if l.strip()]
boots = sorted({r["boot"] for r in runs})
arms = {b: next(r["arm"] for r in runs if r["boot"] == b) for b in boots}
print(f"boots: {', '.join(f'{b} {arms[b]}' for b in boots)}; requests {len(runs)}, errors {sum(1 for r in runs if r['error'])}")


def u_count(a, b):
    return sum(1.0 if x > y else 0.5 if x == y else 0.0 for x in a for y in b)


def per_boot(cls, key, src):
    out = {"off": [], "on": []}
    for b in boots:
        v = [r[key] for r in src if r["boot"] == b and r["class"] == cls and r.get(key) is not None]
        if v:
            out[arms[b]].append(statistics.median(v))
    return out


print("| class | measure | off (median of per-boot medians, range) | on | on over off | U |")
print("|---|---|---|---|---:|---:|")
def is_short(r):
    return r["prompt"].startswith("s") or (r["prompt"].startswith("g") and r["prompt"][-1] in "01")


short_runs = [r for r in runs if is_short(r)]
long_runs = [r for r in runs if not is_short(r)]
for cls, key, src, name in (("c1", "decode_tok_s", runs, "decode tok/s per request"),
                            ("c1", "ttft_s", short_runs, "TTFT, 34-token prompt (s)"), ("c1", "ttft_s", long_runs, "TTFT, 1.2k-token prompts (s)"),
                            ("c1", "aggregate_tok_s", groups, "tokens per second over the request"),
                            ("c4", "aggregate_tok_s", groups, "aggregate tok/s, four at once"),
                            ("c4", "decode_tok_s", runs, "decode tok/s per request"),
                            ("c4", "ttft_s", short_runs, "TTFT, 34-token prompts (s)"), ("c4", "ttft_s", long_runs, "TTFT, 1.2k-token prompts (s)"),
                            ("c4", "total_s", runs, "request time (s)")):
    pb = per_boot(cls, key, src)
    if not pb["off"] or not pb["on"]:
        continue
    mo, mn = statistics.median(pb["off"]), statistics.median(pb["on"])
    fmt = (lambda x: f"{x:.3f}") if key == "ttft_s" else (lambda x: f"{x:.2f}")
    print(f"| {cls} | {name} | {fmt(mo)}, {fmt(min(pb['off']))}-{fmt(max(pb['off']))} | {fmt(mn)}, {fmt(min(pb['on']))}-{fmt(max(pb['on']))} | "
          f"{mn / mo:.3f} | {u_count(pb['on'], pb['off']):g}/{len(pb['on']) * len(pb['off'])} |")
ks = {}
for r in runs:
    k = (r["arm"], r["class"], (r["k"] or ["?"])[0], (r["k"] or ["?", "?"])[1])
    ks[k] = ks.get(k, 0) + 1
print("K served:", ", ".join(f"{a} {c} K={k} ({s}) x{n}" for (a, c, k, s), n in sorted(ks.items())))
texts = {}
for r in runs:
    texts.setdefault((r["arm"], r["prompt"]), set()).add(r["sha"])
print("texts per prompt within an arm:", ", ".join(f"{a} {p}: {len(s)}" for (a, p), s in sorted(texts.items())))


def k_of(r):
    return (r["k"] or ["?"])[0]


print("groups of four, in boot order (K at arrival, + = the request reported an acceptance):")
seen = {}
for g in groups:
    if g["class"] != "c4":
        continue
    idx = seen.get(g["boot"], 0)
    seen[g["boot"]] = idx + 1
    mix = sorted(k_of(r) + ("+" if r["acceptance"] is not None else "") for r in runs
                 if r["boot"] == g["boot"] and r["class"] == "c4" and r["prompt"].startswith(f"g{idx}"))
    print(f"  boot {g['boot']} {g['arm']} group {idx + 1}: K {' '.join(mix)}; aggregate {g['aggregate_tok_s']} tok/s")
for arm in ("off", "on"):
    by = {}
    for r in runs:
        if r["arm"] == arm and r["class"] == "c4":
            by.setdefault((k_of(r), r["acceptance"] is not None), []).append(r["decode_tok_s"])
    print(f"per-request decode tok/s four at once, {arm}: " + ", ".join(
        f"K={k}{' speculating' if s else ' plain'} x{len(v)} median {statistics.median(v):.2f}" for (k, s), v in sorted(by.items())))


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))]


print("time per output token after the first (ms), percentiles over requests (nearest rank):")
for cls in ("c1", "c4"):
    for arm in ("off", "on"):
        v = [1000 * (r["total_s"] - r["ttft_s"]) / (r["completion_tokens"] - 1) for r in runs
             if r["arm"] == arm and r["class"] == cls and r["completion_tokens"] and r["completion_tokens"] > 1]
        print(f"  {cls} {arm}: n {len(v)}, p50 {pct(v, .5):.1f}, p95 {pct(v, .95):.1f}, p99 {pct(v, .99):.1f}")
print("first token four at once (s), by group position in the boot, sorted:")
for arm in ("off", "on"):
    for gi in ("g0", "g1"):
        s = sorted(round(r["ttft_s"], 2) for r in runs if r["arm"] == arm and r["class"] == "c4" and r["prompt"].startswith(gi) and is_short(r))
        l = sorted(round(r["ttft_s"], 2) for r in runs if r["arm"] == arm and r["class"] == "c4" and r["prompt"].startswith(gi) and not is_short(r))
        print(f"  {arm} group {int(gi[1]) + 1}: 34-token {s} mean {statistics.mean(s):.2f}; 1.2k-token {l} mean {statistics.mean(l):.2f}")
    a = [r["ttft_s"] for r in runs if r["arm"] == arm and r["class"] == "c4"]
    s = [r["ttft_s"] for r in runs if r["arm"] == arm and r["class"] == "c4" and is_short(r)]
    print(f"  {arm}: mean over the 34-token requests {statistics.mean(s):.2f}, over all {len(a)} requests {statistics.mean(a):.2f}")

import re

print("the spec cost model's realized rates and choices, per boot (server log):")
for b in boots:
    log = D / f"server-{b}-{arms[b]}.log"
    if not log.exists():
        continue
    text = log.read_text(errors="replace")
    real = re.findall(r"realized n=(\d+) K=(\d+): ([0-9.]+) tok/s", text)
    choice = re.findall(r"\[spec-cost\] model=\"[^\"]*\" n=(\d+) rung=(\d+) -> K=(\d+) \(predicted ([0-9.]+) vs plain ([0-9.]+)", text)
    dem = re.findall(r"demoted session to batched decode: (\d+) active.*?generated (\d+),", text)
    print(f"  boot {b} {arms[b]}: realized " + ", ".join(f"n={n} K={k} {v}" for n, k, v in real if int(n) >= 3)
          + "; choices " + ", ".join(f"n={n} rung {g} -> K={k} ({p} vs plain {q})" for n, g, k, p, q in choice if int(n) >= 3)
          + "; demotions at generated " + " ".join(g for _, g in dem))
import csv

print("thermal, per boot (250 ms samples while the GPU is busy):")
for b in boots:
    tel = D / f"telemetry-{b}-{arms[b]}.csv"
    if not tel.exists():
        continue
    rows = list(csv.reader(open(tel)))[1:]
    busy = [r for r in rows if len(r) >= 7 and r[5].strip().split()[0].isdigit() and int(r[5].strip().split()[0]) > 0]
    if not busy:
        continue
    temps = [int(r[1]) for r in busy]
    clk = [int(r[2].strip().split()[0]) for r in busy]
    pw = [float(r[4].strip().split()[0]) for r in busy if r[4].strip().split()[0].replace(".", "").isdigit()]
    flags = {}
    for r in busy:
        flags[r[6].strip()] = flags.get(r[6].strip(), 0) + 1
    print(f"  boot {b} {arms[b]}: {len(busy)} busy samples, {min(temps)}-{max(temps)} C, SM clock median {statistics.median(clk):.0f} MHz, "
          f"power median {statistics.median(pw):.0f} W, clock event reasons " + ", ".join(f"{k} x{v}" for k, v in sorted(flags.items())))
