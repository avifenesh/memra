#!/usr/bin/env python3
"""summarize.py CAMPAIGN_DIR: per-arm TTFT per target (cells in run order), medians, deltas
against the first arm, and the served fingerprint check (text sha256 per (target, rep) must be
identical across every arm and cell)."""
import collections, json, pathlib, statistics, sys

d = pathlib.Path(sys.argv[1])
rows = []
for cell in sorted(p for p in d.iterdir() if p.is_dir()):
    arm = cell.name.split("-", 1)[1]
    f = cell / "pf.jsonl"
    if not f.exists():
        continue
    for line in f.read_text().splitlines():
        r = json.loads(line)
        r["arm"], r["cell"] = arm, cell.name
        rows.append(r)
arms = list(dict.fromkeys(r["arm"] for r in rows))
targets = sorted({r["target"] for r in rows})
print(f"campaign {d.name}: {len(rows)} requests, arms {arms}")
med = {}
for t in targets:
    print(f"\ntarget {t}")
    for a in arms:
        v = [r["ttft_s"] for r in rows if r["arm"] == a and r["target"] == t and not r.get("error")]
        cells = [r["cell"] for r in rows if r["arm"] == a and r["target"] == t]
        if not v:
            continue
        med[(a, t)] = statistics.median(v)
        print(f"  {a:8s} ttft {', '.join(f'{x:.3f}' for x in v):40s} median {med[(a, t)]:.3f}  cells {cells}")
    base = med.get((arms[0], t))
    for a in arms[1:]:
        if (a, t) in med and base:
            print(f"  {a} vs {arms[0]}: {100 * (med[(a, t)] / base - 1):+.2f}%")
print("\nfingerprints (text sha256 per target/rep across all cells)")
fp = collections.defaultdict(set)
for r in rows:
    fp[(r["target"], r["rep"])].add(r["text_sha256"])
ok = True
for k, v in sorted(fp.items()):
    print(f"  target {k[0]} rep {k[1]}: {len(v)} distinct {'IDENTICAL' if len(v) == 1 else 'DIFFER'}")
    ok &= len(v) == 1
errs = [r for r in rows if r.get("error") or not r.get("needles_ok")]
print(f"errors/needle misses: {len(errs)}")
print("FINGERPRINTS", "IDENTICAL" if ok else "DIFFER")
