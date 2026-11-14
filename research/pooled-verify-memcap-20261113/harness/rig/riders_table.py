#!/usr/bin/env python3
"""riders_table.py CAMPAIGN_DIR: memra#1266 review 2. The riders cell (riders_cell_1266.py), one
row per boot and cell: streams, riders, their errors, and the boot's memory-cap counters from its
/metrics dump (capped forwards, capped rider rows, pre-cohort reclaims and short passes). Then the
text check: every rider's text sha (the cell's plain requests and their solo re-runs) across the
boots, and the long streams' texts as prefixes of one another (they are hung up at different
points). Correctness cell: no timing column."""
import json
import pathlib
import sys

camp = pathlib.Path(sys.argv[1])
boots = sorted(d for d in camp.iterdir() if d.is_dir() and (d / "riders.jsonl").exists())
print("| boot | cell | streams + riders | rider errors | capped forwards | capped rider rows | pre-cohort reclaims | short passes | OOM lines |")
print("|---|---|---|---|---|---|---|---|---|")
texts = {}
streams = {}
for d in boots:
    m = {}
    mf = next(iter(d.glob("metrics-*.json")), None)
    if mf:
        try:
            m = json.loads(mf.read_text())
        except json.JSONDecodeError:
            m = {}
    summ = dict(l.rsplit("=", 1) for l in (d / "summary.txt").read_text().splitlines() if "=" in l) if (d / "summary.txt").exists() else {}
    for line in (d / "riders.jsonl").read_text().splitlines():
        r = json.loads(line)
        print(f"| {d.name} | {r['cell']} | {r['longs']} + {r['riders']} {r['plain_class']} | {r['plain_errors']} | "
              f"{m.get('spec_pool_capped_forwards_total', 'n/a')} | {m.get('spec_pool_capped_rider_rows_total', 'n/a')} | "
              f"{m.get('spec_cohort_reclaims_total', 'n/a')} | {m.get('spec_cohort_short_total', 'n/a')} | "
              f"{summ.get('out of memory', '?')} |")
        for kind in ("news", "solos"):
            for x in r[kind]:
                texts.setdefault((r["cell"], kind, x["n"]), {})[d.name] = x.get("sha")
        for s in r["long_streams"]:
            t = (d / s["text_file"]).read_text() if (d / s["text_file"]).exists() else ""
            streams.setdefault((r["cell"], s["stream"]), {})[d.name] = t
print()
same = sum(1 for v in texts.values() if len(set(v.values())) == 1 and len(v) == len(boots))
diff = [k for k, v in texts.items() if len(set(v.values())) != 1]
print(f"- rider and solo texts: {same} of {len(texts)} identical across the {len(boots)} boots; differing: {diff or 'none'}")
pref = 0
for k, v in streams.items():
    vals = sorted(v.values(), key=len)
    if all(vals[i + 1].startswith(vals[i]) for i in range(len(vals) - 1)):
        pref += 1
print(f"- long streams: {pref} of {len(streams)} prefix-equal across the boots")
