#!/usr/bin/env python3
"""Aggregate memra-ep-map-v1 mint stats into one row per map (plain mean over layers,
matching struct-battery's mint-stats-summary.txt method)."""
import json, sys
from pathlib import Path

hdr = f"{'map':<44}{'layers':>7}{'peer_touch mean':>17}{'single-rank':>13}{'worst-layer pt':>16}{'exp_max_touch':>15}{'even_max':>10}{'tol':>8}"
print(hdr)
rows = []
for p in sorted(sys.argv[1:]):
    doc = json.loads(Path(p).read_text())
    L = doc["layers"]
    n = len(L)
    pt = [r["stats"]["peer_touch_fraction"] for r in L]
    em = [r["stats"]["expected_max_rank_touch"] for r in L]
    ev = [r["stats"]["even_baseline_expected_max_rank_touch"] for r in L]
    intra = [r["stats"]["intra_rank_coactivation_fraction"] for r in L]
    tol = doc.get("params", {}).get("balance_tolerance")
    # per-rank expert COUNT loads (the thing tolerance actually bounds)
    loads = []
    for r in L:
        a = r["assignment"]
        loads.append([a.count(k) for k in range(doc["ranks"])])
    mx = max(max(x) for x in loads); mn = min(min(x) for x in loads)
    row = dict(name=Path(p).name, n=n,
               pt=sum(pt)/n, sr=1.0-sum(pt)/n, wpt=max(pt),
               em=sum(em)/n, ev=sum(ev)/n, tol=tol,
               intra=sum(intra)/n, load_max=mx, load_min=mn,
               em_worst=max(em))
    rows.append(row)
    print(f"{row['name']:<44}{n:>7}{row['pt']:>17.4f}{row['sr']:>13.4f}"
          f"{row['wpt']:>16.4f}{row['em']:>15.3f}{row['ev']:>10.3f}{str(tol):>8}")
print()
print(f"{'map':<44}{'intra_coact_frac':>18}{'load_max':>10}{'load_min':>10}{'em_worst_layer':>16}{'em_over_even':>14}")
for r in rows:
    print(f"{r['name']:<44}{r['intra']:>18.4f}{r['load_max']:>10}{r['load_min']:>10}"
          f"{r['em_worst']:>16.3f}{r['em']/r['ev']:>14.4f}")
