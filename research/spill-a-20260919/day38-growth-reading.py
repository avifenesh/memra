#!/usr/bin/env python3
"""WP-A day 38 section 12's growth reader, written before it runs. Input: ROOT/bNN-{x1,x2}/demote/receipt.json
(stall_cell.py's demote arm). Per boot: the tenant's pre-fire inter-token latency median of every demote run (its first
22 ITLs, before the intruder fires at token 24), in run order; EARLY = the median over runs 2 to 6, LATE = over the last
five runs, GROWTH = LATE minus EARLY. Per arm: the median of its boots' GROWTH. An arm GROWS if that median exceeds
0.10 ms. Verdict: `x1 grows, x2 flat -> the separate receipt stream`; `both grow -> elsewhere in G''`; `x1 flat ->
not reproduced`. Usage: day38-growth-reading.py ROOT"""
import glob, json, os, statistics as st, sys

root = sys.argv[1]
by = {"x1": [], "x2": []}
for f in sorted(glob.glob(os.path.join(root, "b*-x*", "demote", "receipt.json"))):
    arm = f.split(os.sep)[-3].split("-")[1]
    j = json.load(open(f))
    rs = [r for r in j["runs"] if r.get("arm") == "demote" and r.get("itl_ms")]
    itl = [st.median(r["itl_ms"][:22]) for r in rs]
    fired = [r["intruder"]["fired_at_ms"] for r in rs if "fired_at_ms" in r.get("intruder", {})]
    if len(itl) < 10:
        print(f"GROWTH boot={f.split(os.sep)[-3]} runs={len(itl)} too few -> INCOMPLETE")
        continue
    early, late = st.median(itl[1:6]), st.median(itl[-5:])
    by[arm].append(late - early)
    print(f"GROWTH boot={f.split(os.sep)[-3]} runs={len(itl)} early={early:.3f} late={late:.3f} growth={late - early:+.3f} "
          f"itl={[round(x, 2) for x in itl]} fired_first={fired[:2]} fired_last={fired[-2:]}")
grow = {a: (st.median(v) if v else float("nan")) for a, v in by.items()}
g = {a: (grow[a] > 0.10) for a in grow}
print(f"GROWTH arm=x1 boots={len(by['x1'])} median-growth={grow['x1']:+.3f} grows={g['x1']}")
print(f"GROWTH arm=x2 boots={len(by['x2'])} median-growth={grow['x2']:+.3f} grows={g['x2']}")
if not by["x1"] or not by["x2"]:
    print("GROWTH VERDICT incomplete")
elif g["x1"] and not g["x2"]:
    print("GROWTH VERDICT x1 grows, x2 flat -> the separate receipt stream")
elif g["x1"] and g["x2"]:
    print("GROWTH VERDICT both grow -> elsewhere in G''")
else:
    print("GROWTH VERDICT x1 flat -> not reproduced")
