#!/usr/bin/env python3
"""agentic_table.py CAMPAIGN_DIR [REF_DIR ...]: memra#1266. One row per boot of the agentic replay
(cells.jsonl + summary.txt + metrics.json in each NAME/ dir), then the text comparison: every
request k's text sha across the boots (only answered requests), against the first fix boot, and
against the reference boots (REF_DIR: other boots' directories, e.g. memra#1221 part 2's head).
Rows carry no prompt or generated text, only counts and hashes."""
import json
import re
import pathlib
import sys

camp = pathlib.Path(sys.argv[1])
refs = [pathlib.Path(p) for p in sys.argv[2:]]


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


def boot(d):
    cell = json.loads((d / "cells.jsonl").read_text().splitlines()[-1])
    reqs = [json.loads(l) for l in next(d.glob("*.requests.jsonl")).read_text().splitlines() if l.strip()]
    summ = {}
    s = d / "summary.txt"
    if s.exists():
        for line in s.read_text().splitlines():
            if "=" in line:
                k, v = line.rsplit("=", 1)
                summ[k] = v
    m = {}
    if (d / "metrics.json").exists():
        try:
            m = json.loads((d / "metrics.json").read_text())
        except json.JSONDecodeError:
            m = {}
    return cell, {r["k"]: r for r in reqs}, summ, m


boots = sorted(p for p in camp.iterdir() if p.is_dir() and (p / "cells.jsonl").exists())
data = {b.name: boot(b) for b in boots}
print("| boot | ok | failed | tokens | agg gen tok/s | per-user median tok/s | TTFT p50 s | OOM lines | not parked | "
      "capped forwards | pre-cohort reclaims | graph suspended | profile; limit under load W |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for name, (cell, reqs, summ, m) in data.items():
    failed = sorted(k for k, r in reqs.items() if not r["ok"])
    print(f"| {name} | {cell['ok']} | {','.join(map(str, failed)) or '-'} | {cell['tokens']} | {cell['aggregate_gen_tps']} | "
          f"{cell['per_user_gen_tps_median']} | {cell['ttft_median_s']} | {summ.get('out of memory', '?')} | "
          f"{summ.get('NOT parked', '?')} | {m.get('spec_pool_capped_forwards_total', 'n/a')} | "
          f"{m.get('spec_cohort_reclaims_total', 'n/a')} | {summ.get('graph replay suspended', '?')} | {load_limits(camp / name)} |")
print()
first_fix = next((n for n in data if n.endswith("-F")), None)
ref = data[first_fix][1] if first_fix else None
for name, (cell, reqs, summ, m) in data.items():
    if ref is None:
        break
    same = sum(1 for k, r in reqs.items() if r["ok"] and k in ref and ref[k]["ok"] and r["text_sha"] == ref[k]["text_sha"])
    diff = sorted(k for k, r in reqs.items() if r["ok"] and k in ref and ref[k]["ok"] and r["text_sha"] != ref[k]["text_sha"])
    print(f"- {name}: {same} answered texts equal {first_fix}'s; differing k: {diff or 'none'}")
for rd in refs:
    if rd.is_file():
        # A reference file of hash rows: {"boot", "k", "ok", "text_sha"} per line.
        rows = [json.loads(l) for l in rd.read_text().splitlines() if l.strip()]
        groups = {}
        for r in rows:
            groups.setdefault(r["boot"], {})[r["k"]] = r
        for gname, rreqs in groups.items():
            for name, (cell, reqs, summ, m) in data.items():
                same = sum(1 for k, r in rreqs.items() if r["ok"] and k in reqs and reqs[k]["ok"] and r["text_sha"] == reqs[k]["text_sha"])
                both = sum(1 for k, r in rreqs.items() if r["ok"] and k in reqs and reqs[k]["ok"])
                picks = {k: rreqs[k]["text_sha"] == reqs[k]["text_sha"] for k in (19, 21) if k in rreqs and k in reqs and reqs[k]["ok"]}
                print(f"- {name} against {gname}: {same} of {both} texts equal; k=19/21 equal: {picks}")
        continue
    try:
        _, rreqs, _, _ = boot(rd)
    except (StopIteration, FileNotFoundError):
        print(f"- reference {rd}: unreadable")
        continue
    same = sum(1 for k, r in rreqs.items() if r["ok"] and k in ref and ref[k]["ok"] and r["text_sha"] == ref[k]["text_sha"])
    diff = sorted(k for k, r in rreqs.items() if r["ok"] and k in ref and ref[k]["ok"] and r["text_sha"] != ref[k]["text_sha"])
    picks = {k: (rreqs[k]["text_sha"] == ref[k]["text_sha"]) for k in (19, 21) if k in rreqs and k in ref and rreqs[k]["ok"] and ref[k]["ok"]}
    print(f"- reference {rd.parent.name}/{rd.name}: {same} answered texts equal {first_fix}'s; differing k: {diff or 'none'}; "
          f"k=19/21 equal: {picks}")
