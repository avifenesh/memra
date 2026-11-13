#!/usr/bin/env python3
"""agentic_table.py CAMPAIGN_DIR [REF_DIR ...]: memra#1266. One row per boot of the agentic replay
(cells.jsonl + summary.txt + metrics.json in each NAME/ dir), then the text comparison: every
request k's text sha across the boots (only answered requests), against the first fix boot, and
against the reference boots (REF_DIR: other boots' directories, e.g. memra#1221 part 2's head).
Rows carry no prompt or generated text, only counts and hashes."""
import json
import pathlib
import sys

camp = pathlib.Path(sys.argv[1])
refs = [pathlib.Path(p) for p in sys.argv[2:]]


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
      "capped forwards | pre-cohort reclaims | graph suspended |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for name, (cell, reqs, summ, m) in data.items():
    failed = sorted(k for k, r in reqs.items() if not r["ok"])
    print(f"| {name} | {cell['ok']} | {','.join(map(str, failed)) or '-'} | {cell['tokens']} | {cell['aggregate_gen_tps']} | "
          f"{cell['per_user_gen_tps_median']} | {cell['ttft_median_s']} | {summ.get('out of memory', '?')} | "
          f"{summ.get('NOT parked', '?')} | {m.get('spec_pool_capped_forwards_total', 'n/a')} | "
          f"{m.get('spec_cohort_reclaims_total', 'n/a')} | {summ.get('graph replay suspended', '?')} |")
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
