#!/usr/bin/env python3
"""OWED 18 verdict (M1-PREREG.md section E): paired direct / buffered handoff cycles.

Usage: m1-handoff-pairs.py <cell dir>  -> handoff-pairs.json

Reads every `cycle-NN/cycle.json` of a 5090 regime dir (`round-KK/visits/cycle-01..02`, one pair
per round) or of one PRO sitting cell (`visits/cycle-01..20`, consecutive cycles form a pair).
A pair is forward when its first cycle is buffered. Every cycle must have passed (probes restored,
text identical to cold, entries imported equal exported, the `io=` lines); a cell with any failed
cycle gets no verdict. Per metric (export ms, its write and fsync parts, import s), the ratio is
direct / buffered within the pair: winner at median <= 0.95 with at least 4 of 5 pairs below 1 in
each order, loser at median >= 1.05 with at least 4 of 5 above 1 in each order, otherwise flat.
"""
import json
from pathlib import Path
import statistics
import sys

METRICS = {"export_ms": ("export", "ms"), "write_ms": ("export", "write_ms"),
           "fsync_ms": ("export", "fsync_ms"), "import_s": ("import", "done", "seconds")}


def value(c, path):
    x = c
    for key in path:
        x = x.get(key) if isinstance(x, dict) else None
        if x is None:
            return None
    return float(x)


def pairs_of(d):
    rounds = sorted(p for p in d.glob("round-[0-9][0-9]") if (p / "visits").is_dir())
    groups = []
    if rounds:
        for r in rounds:
            groups.append([json.loads(p.read_text()) for p in sorted((r / "visits").glob("cycle-*/cycle.json"))])
    else:
        cycles = [json.loads(p.read_text()) for p in sorted((d / "visits").glob("cycle-*/cycle.json"))]
        groups = [cycles[i:i + 2] for i in range(0, len(cycles), 2)]
    return groups


def main():
    d = Path(sys.argv[1])
    groups = pairs_of(d)
    failed = [c["cycle"] for g in groups for c in g if not c.get("passed")]
    bad_pairs = [i + 1 for i, g in enumerate(groups) if len(g) != 2 or {c.get("io") for c in g} != {"buffered", "direct"}]
    out = {"pairs": len(groups), "failed_cycles": failed, "malformed_pairs": bad_pairs, "metrics": {}}
    if not failed and not bad_pairs:
        for name, path in METRICS.items():
            ratios = {"forward": [], "reverse": []}
            for g in groups:
                by = {c["io"]: c for c in g}
                b, x = value(by["buffered"], path), value(by["direct"], path)
                if b and x is not None:
                    ratios["forward" if g[0]["io"] == "buffered" else "reverse"].append(x / b)
            allr = ratios["forward"] + ratios["reverse"]
            if min(len(ratios["forward"]), len(ratios["reverse"])) < 5:
                verdict = "insufficient"
            else:
                med = statistics.median(allr)
                below = [sum(r < 1 for r in ratios[o]) for o in ("forward", "reverse")]
                above = [sum(r > 1 for r in ratios[o]) for o in ("forward", "reverse")]
                verdict = ("winner" if med <= 0.95 and min(below) >= 4 else
                           "loser" if med >= 1.05 and min(above) >= 4 else "flat")
            out["metrics"][name] = {"verdict": verdict, "median_ratio": statistics.median(allr) if allr else None,
                                    "n_pairs": len(allr), **ratios}
    (d / "handoff-pairs.json").write_text(json.dumps(out, indent=1) + "\n")
    for name, m in out["metrics"].items():
        print(f"M1-HANDOFF-VERDICT {d.name} {name} direct/buffered: {m['verdict']} median_ratio={m['median_ratio']} pairs={m['n_pairs']}")
    print(f"pairs={out['pairs']} failed_cycles={failed} malformed_pairs={bad_pairs}")
    return 0 if not failed and not bad_pairs else 3


if __name__ == "__main__":
    sys.exit(main())
