#!/usr/bin/env python3
"""pairs_tab.py VERDICTS.jsonl [BASE_VERDICTS.jsonl]: one row per warm == cold pair case of
research/spec-pool-grid-20261109/spec_pool_pair_cells.py: whether the warm turns 2 and 3 are
byte-equal to their cold twins, whether every resume stayed on the prime grid, and the resume path
the warm boot took. With BASE_VERDICTS (the same cases on another binary) its path is the last
column, so a path the change moved shows."""
import json
import sys

NAMES = {"spec_parked_end": "parked end", "spec_affinity": "affinity", "spec_grid_rewind": "exact, grid rewind",
         "spec_grid_decline": "exact, declined", "spec_affinity_grid_decline": "affinity declined",
         "prefix_hit": None, "spec_restore": "prefix restore", "spec_park_skip": None}


def load(path):
    out = {}
    for l in open(path):
        if l.startswith('{"case"'):
            r = json.loads(l)
            out[r["case"]] = r
    return out


def path(r):
    parts = []
    for k, v in r["warm"].items():
        name = NAMES.get(k, k)
        if not name or not v or v == [[]]:
            continue
        parts.append(name + " " + ", ".join("->".join(str(x) for x in h) if h else "" for h in v))
    return "; ".join(parts) or "cold"


new = load(sys.argv[1])
base = load(sys.argv[2]) if len(sys.argv) > 2 else {}
head = "| case | turn 2 / turn 3 equal to cold | on the grid | warm path |" + (" base path |" if base else "")
print(head)
print("|---|---|---|---|" + ("---|" if base else ""))
ok = 0
for name, r in new.items():
    eq = " / ".join("=" if r["turns"][t].get("equal") else "DIFF" for t in ("2", "3"))
    ok += bool(r["bytes"] and r["path_ok"])
    row = f"| {name} | {eq} | {'yes' if r['path_ok'] else 'NO'} | {path(r)} |"
    if base:
        row += f" {path(base[name]) if name in base else ''} |"
    print(row)
print(f"\n{ok} of {len(new)} cases byte-equal to cold on a grid path")
