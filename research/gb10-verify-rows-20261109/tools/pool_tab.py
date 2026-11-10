#!/usr/bin/env python3
"""pool_tab.py CAMPAIGN_DIR [DEST]: a pooled-cohort campaign (pool-<ARM>/summary.json per boot, arms
named B.. and T..): per boot the aggregate rate of each wave and whether its streams equal the
first boot's, then per wave the B and T medians with ranges and T/B. With DEST, also copies each
summary.json there as summary-<ARM>.json (paths written from the home directory down) and writes
runs.txt."""
import json
import pathlib
import re
import statistics
import sys

src = pathlib.Path(sys.argv[1])
dst = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else None
boots = sorted((p for p in src.glob("pool-*") if (p / "summary.json").exists()),
               key=lambda p: (p / "summary.json").stat().st_mtime)
ref, rates, lines = None, {"B": [], "T": []}, []
for p in boots:
    arm = p.name[5:]
    s = json.loads((p / "summary.json").read_text())
    ref = ref or s
    keys = sorted(k for k in ref if "/r" in k)
    r = [s.get(f"c{c}/agg_tok_s") for c in (2, 4, 8)]
    rates[arm[0]].append(r)
    same = all(s.get(k, {}).get("sha256") == ref[k]["sha256"] for k in keys)
    lines.append(f"{arm}: c2 {r[0]} tok/s, c4 {r[1]}, c8 {r[2]}; {len(keys)} streams, identical to {boots[0].name[5:]}: {same}")
    if dst:
        dst.mkdir(parents=True, exist_ok=True)
        (dst / f"summary-{arm}.json").write_text(re.sub(r"/home/[A-Za-z0-9_.-]+", "~", json.dumps(s, indent=1)))
for i, c in enumerate((2, 4, 8)):
    b, t = [x[i] for x in rates["B"]], [x[i] for x in rates["T"]]
    if b and t:
        lines.append(f"c{c}: B median {statistics.median(b):.2f} ({min(b):.2f}-{max(b):.2f}), "
                     f"T median {statistics.median(t):.2f} ({min(t):.2f}-{max(t):.2f}), T/B {statistics.median(t) / statistics.median(b):.4f}")
print("\n".join(lines))
if dst:
    (dst / "runs.txt").write_text("\n".join(lines) + "\n")
