#!/usr/bin/env python3
"""Compare served-stream cell runs (pool_cell.py / dspark_pool_cell.py summaries) stream by
stream. usage: compare_cells.py LABEL=OUTDIR [LABEL=OUTDIR ...]

Prints SAME/DIFF per (wave, request) across every run, the per-wave aggregate tok/s of each run
and the pooled cohort lines each run logged. Exit 1 on any DIFF or missing stream."""
import json, os, sys

runs = []
for arg in sys.argv[1:]:
    label, d = arg.split("=", 1)
    runs.append((label, json.load(open(os.path.join(d, "summary.json")))))
keys = sorted({k for _, s in runs for k in s if "/r" in k},
              key=lambda k: (int(k.split("/")[0][1:]), int(k.split("/r")[1])))
ok = True
for k in keys:
    shas = [s.get(k, {}).get("sha256") for _, s in runs]
    same = all(shas) and len(set(shas)) == 1
    ok &= same
    toks = runs[0][1].get(k, {}).get("completion_tokens")
    cells = " ".join(f"{label} {(sha or 'MISSING')[:16]}" for (label, _), sha in zip(runs, shas))
    spec = "/".join(str(s.get(k, {}).get("spec_rounds", "-")) for _, s in runs)
    print(f"{'SAME' if same else 'DIFF'}  {k:8s} {cells} {toks} tok spec_rounds {spec}")
waves = sorted({k.split("/")[0] for k in keys}, key=lambda w: int(w[1:]))
for w in waves:
    print(f"{w}: " + ", ".join(f"{label} {s.get(w + '/agg_tok_s')} tok/s" for label, s in runs))
for label, s in runs:
    print(f"{label}: pooled cohort lines {len(s.get('spec_pool_lines', []))}"
          + "".join(f"\n    {l}" for l in s.get("spec_pool_lines", [])))
print("VERDICT:", "BYTE-IDENTICAL" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
