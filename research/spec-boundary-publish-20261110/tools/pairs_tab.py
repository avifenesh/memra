#!/usr/bin/env python3
"""pairs_tab.py CELLS_DIR: one row per warm == cold pair case (specpub_pairs.py's verdicts under
CELLS_DIR/pairs-*/out/verdicts.jsonl): the arm (the case name's last letter), whether each later
warm turn is byte-equal to its cold twin, the restores (tokens restored of the prompt, each from
an `insert (spec-boundary)` entry with its draft plane), the answers' sha256 prefixes per turn and
the published entries' state digests when the case ran with the trace. Then, per shape, whether
the arms served the same answers, took the same bursts and published the same digests."""
import json
import pathlib
import sys

rows = {}
for f in sorted(pathlib.Path(sys.argv[1]).glob("pairs-*/out/verdicts.jsonl")):
    lines = [l for l in f.read_text().splitlines() if l.strip()]
    if lines:
        v = json.loads(lines[-1])
        rows[v["case"]] = v
print("| case | arm | later turns equal to cold | restored from a spec-published entry, with its draft plane | answers (sha256, per turn) | entries published (tokens) |")
print("|---|---|---|---|---|---|")
shapes = {}
for name, v in rows.items():
    shape, arm = name[6:].rsplit("-", 1)  # pairs-<shape>-<arm>; an arm may carry its main (B11f3)
    later = [t for t in v["turns"] if t != "1"]
    eq = " / ".join("=" if v["turns"][t]["equal"] else "DIFF" for t in later)
    w = v["warm"]
    rest = ", ".join(f"{n:,} of {m:,}" for n, m in w["draft_restore"]) if v["restored_from_spec_entry"] else "NO"
    shas = " / ".join(v["turns"][t]["warm_sha"][:8] for t in sorted(v["turns"]))
    ents = ", ".join(f"{n:,}" for why, n, _ in w["insert"] if why == "spec-boundary")
    print(f"| {shape} | {arm} | {eq} | {rest} | {shas} | {ents} |")
    shapes.setdefault(shape, {})[arm] = v
print()
for shape, arms in shapes.items():
    if len(arms) < 2:
        print(f"{shape}: one arm ({', '.join(arms)})")
        continue
    ans = {json.dumps({t: x["warm_sha"] for t, x in v["turns"].items()}, sort_keys=True) for v in arms.values()}
    bursts = {json.dumps(v["warm"]["bursts"]) for v in arms.values()}
    states = {json.dumps([v["warm"]["state"], v["cold_state"]], sort_keys=True) for v in arms.values()}
    has_state = any(v["warm"]["state"] for v in arms.values())
    dig = ("published-entry digests " + ("equal" if len(states) == 1 else "DIFFER")) if has_state else "no digest trace"
    print(f"{shape}: arms {', '.join(sorted(arms))}: answers {'equal' if len(ans) == 1 else 'DIFFER'}, "
          f"warm bursts (generated, rounds, drafted, accepted, K) {'equal' if len(bursts) == 1 else 'DIFFER'}, {dig}")
