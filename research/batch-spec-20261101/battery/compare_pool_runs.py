#!/usr/bin/env python3
"""Compare pool_cell.py runs stream by stream against a reference run.

usage: compare_pool_runs.py REF_DIR LABEL=DIR [LABEL=DIR ...]
Prints one row per (c, request) with each run's sha256 prefix, SAME/DIFF against REF, the
aggregate tok/s per wave and run, and each run's [spec-pool] engagement lines.
"""
import json, os, sys

ref = json.load(open(os.path.join(sys.argv[1], "summary.json")))
runs = [(a.split("=", 1)[0], json.load(open(os.path.join(a.split("=", 1)[1], "summary.json"))))
        for a in sys.argv[2:]]
keys = [k for k in ref if "/r" in k]
ok = True
for k in keys:
    rh = ref[k]["sha256"][:16]
    same = all(r.get(k, {}).get("sha256") == ref[k]["sha256"] for _, r in runs)
    ok &= same
    cells = " ".join(f"{lab} {r.get(k, {}).get('sha256', 'MISSING')[:16]}" for lab, r in runs)
    print(f"{'SAME' if same else 'DIFF'}  {k:<6} serial-ref {rh} {cells} "
          f"{ref[k].get('completion_tokens', '?')} tok")
for c in ("c2", "c4", "c8"):
    vals = " ".join(f"{lab} {r.get(c + '/agg_tok_s')}" for lab, r in runs)
    print(f"{c}: serial-ref {ref.get(c + '/agg_tok_s')} tok/s, {vals}")
for lab, r in runs:
    lines = r.get("spec_pool_lines", [])
    print(f"{lab}: {len(lines)} [spec-pool] lines")
    for l in lines:
        print(f"    {l}")
print("VERDICT:", f"ALL {len(keys)} STREAMS IDENTICAL" if ok else "STREAMS DIFFER")
