#!/usr/bin/env python3
"""normal_table.py CAMPAIGN_DIR: memra#1266. The normal-concurrency cells (normal_cell.py), one row
per boot and c in boot order, the text-set check per c across boots, and each boot's summary.txt
(OOM, capped forwards, pre-cohort reclaims, cohort engagement). Then per c the arm medians."""
import json
import re
import pathlib
import statistics
import sys

camp = pathlib.Path(sys.argv[1])


def load_limits(d):
    """The platform profile a boot ran under and the enforced power limit its telemetry shows under
    load (>= 50 % utilization): median (min to max). The guard's profile= (cell start) and
    profile_end= (cell end) readings when it has them, "mixed" when they differ;
    for earlier boots, the telemetry's span against the owner's switch from balanced to
    performance (the first 175 W sample, 2026/11/13 06:41:49). "n/a" without such telemetry."""
    import gzip
    import statistics
    guard = d / "guard.log"
    prof = None
    if guard.exists():
        gt = guard.read_text()
        m = re.search(r"profile=(\S+)", gt)
        e = re.search(r"profile_end=(\S+)", gt)
        if m and e:
            prof = m.group(1) if m.group(1) == e.group(1) else f"mixed ({m.group(1)} at start, {e.group(1)} at end)"
        elif m:
            prof = f"{m.group(1)} at start (end not read)"
    f = d / "telemetry.csv.gz"
    g = d / "telemetry.csv"
    if f.exists():
        text = gzip.open(f, "rt").read()
    elif g.exists():
        text = g.read_text()
    else:
        return prof or "n/a"
    lims, stamps = [], []
    for line in text.splitlines():
        p = [x.strip() for x in line.split(",")]
        # agentic cells: timestamp, temp, clock, draw, limit, util, mem; normal cells: timestamp,
        # clock, draw, limit, temp, util, mem.
        li = 3 if len(p) > 1 and p[1].endswith("MHz") else 4
        if len(p) < 6 or not p[5].endswith("%") or not p[li].endswith("W"):
            continue
        try:
            util = float(p[5].rstrip(" %"))
            lim = float(p[li].rstrip(" W"))
        except ValueError:
            continue
        stamps.append(p[0][:19])
        if util >= 50:
            lims.append(lim)
    if not stamps:
        return prof or "n/a"
    if prof is None:
        switch = "2026/11/13 06:41:49"
        prof = "balanced" if stamps[-1] < switch else "performance" if stamps[0] >= switch else "balanced, then performance from 06:41:49"
    out = prof if not lims else f"{prof}; {statistics.median(lims):.0f} ({min(lims):.0f} to {max(lims):.0f})"
    tj = d / "take.json"
    if tj.exists():
        # The owner's timed-take rule (take_check.py): this take counts, or is the banked best of 4.
        t = json.loads(tj.read_text())
        out += (f"; take {t['take']} {'counts' if t['counts'] else 'banked best of 4'}: sw thermal "
                f"{t['sw_thermal_load_share']:.1%} of samples under load, start {t['start_temp_c']:.0f} C")
    return out
rows = []
for d in sorted(p for p in camp.iterdir() if p.is_dir() and (p / "normal.jsonl").exists()):
    for line in (d / "normal.jsonl").read_text().splitlines():
        rows.append((d.name, json.loads(line)))
print("| boot | c | requests, errors | completion tokens | agg tok/s | TTFT p50 s | ITL p50 / p99 ms |")
print("|---|---|---|---|---|---|---|")
sets = {}
for name, r in rows:
    sets.setdefault(r["c"], set()).add(tuple(sorted((x["prompt"], x.get("sha")) for x in r["rows"])))
    print(f"| {name} | {r['c']} | {r['requests']}, {r['errors']} | {r['completion_tokens']} | {r['agg_tok_s']} | "
          f"{r['ttft_p50_s']} | {r['itl_p50_ms']} / {r['itl_p99_ms']} |")
print()
for c, s in sorted(sets.items()):
    print(f"- c={c}: {'one' if len(s) == 1 else len(s)} set of (prompt, text sha) across the boots")
for d in sorted(p for p in camp.iterdir() if p.is_dir() and (p / "summary.txt").exists()):
    print(f"- {d.name}: " + "; ".join(l.strip() for l in (d / "summary.txt").read_text().splitlines())
          + f"; profile; limit under load W: {load_limits(d)}")
print()
for c in sorted(sets):
    by = {}
    for name, r in rows:
        if r["c"] == c:
            by.setdefault(name.split("-")[-1], []).append(r["agg_tok_s"])
    print(f"- c={c} agg tok/s: " + ", ".join(f"{arm} {statistics.median(v):.2f} ({', '.join(map(str, v))})" for arm, v in sorted(by.items())))
