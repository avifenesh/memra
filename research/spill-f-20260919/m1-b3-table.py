#!/usr/bin/env python3
"""Per-arm table of one B3 regime from its visit records (no new measurement).

Usage: m1-b3-table.py <regime visits dir>   prints a markdown table and the verdict lines.
Decode tok/s is run-gen's gen-only rate. Device GB and residency come from the sampler and
mincore; the stage columns are the pool's whole-visit totals (prefill included) per visit.
"""
import json
from pathlib import Path
import statistics
import sys


def main():
    d = Path(sys.argv[1])
    visits = [json.loads(p.read_text()) for p in sorted(d.glob("r*/visit.json"))]
    summary = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else {}
    arms = []
    for v in visits:
        if v["arm"] not in arms:
            arms.append(v["arm"])
    print("| Arm | Scored | Decode tok/s median (min to max) | Device read GB per visit | Artifact resident at end | Read s (worker or blocking) | Owner wait s | Verdict vs worker16 |")
    print("|---|---|---|---|---|---|---|---|")
    for arm in arms:
        vs = [v for v in visits if v["arm"] == arm]
        sc = [v for v in vs if v.get("scored")]
        tps = [v["tok_s"] for v in sc if v.get("tok_s")]
        dev = [v["contamination"]["device_read_bytes"] / 1e9 for v in sc if v.get("contamination")]
        res = [v["residency_end"][0] / v["residency_end"][1] for v in sc if v["residency_end"][1]]
        drops = [v["parsed"]["drop"] for v in sc if v["parsed"].get("drop")]
        wr = [(int(x[8]) or int(x[9])) / 1e9 for x in drops]  # worker read, or blocking demand read
        wt = [int(x[10]) / 1e9 for x in drops]
        verdict = (summary.get("arms", {}).get(arm) or {})
        vtxt = "baseline" if arm == "worker16" else (
            f"{verdict.get('verdict')} ({verdict.get('median_ratio'):.3f})" if verdict.get("median_ratio") else str(verdict.get("verdict", "refused" if arm in summary.get("refused_arms", []) else "n/a")))
        fmt = lambda xs, n=2: f"{statistics.median(xs):.{n}f}" if xs else "n/a"
        print(f"| `{arm}` | {len(sc)}/{len(vs)} | {fmt(tps)} ({min(tps):.2f} to {max(tps):.2f}) | {fmt(dev, 1)} | "
              f"{fmt(res, 3)} | {fmt(wr)} | {fmt(wt)} | {vtxt} |" if tps else
              f"| `{arm}` | {len(sc)}/{len(vs)} | n/a | n/a | n/a | n/a | n/a | {vtxt} |")
    for arm, s in (summary.get("arms") or {}).items():
        print(f"M1-VERDICT regime={summary.get('regime')} arm={arm} vs worker16: {s['verdict']} "
              f"median_ratio={s['median_ratio']} pairs={s['n_pairs']}")
    print(f"regime_scored={summary.get('regime_scored')} refused={summary.get('refused_arms')} "
          f"contaminated={summary.get('contaminated_visits')}")


if __name__ == "__main__":
    main()
