#!/usr/bin/env python3
"""Day 54 reader for cell `slices` (research/spill-c-20260919/DAY54.md section 1, registered before any slice binary
was built or booted). Per binary, over its ten boots' promote-arm runs (lane A's harness receipts, `server_log_lines`
per run): the promote intruder's inline demote landing (`demote published off the tick: ... Xms from submission to
completion`), the parks per run (`promote submitted off the tick`, `restore submitted off the tick`), the tenant's
stall. Then the endpoint check and the slices where the landing class or the park count changes.

usage: day54-slice-reading.py <cell-dir>
"""
import json
import re
import statistics
import sys
from pathlib import Path

LABELS = ["e0", "s1", "s2", "s3", "s4", "s5", "s6", "s7"]
COMMITS = {"e0": "0713c1a79", "s1": "ff64e7f5d", "s2": "da1f59bf6", "s3": "226abab0e", "s4": "5df11152f",
           "s5": "58b814abe", "s6": "f661406e4", "s7": "269ef2cec"}
LANDING = re.compile(r"\[prefix-host\] demote published off the tick: .*? ([0-9.]+)ms from submission to completion")


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def quart(xs):
    if len(xs) < 4:
        return float("nan"), float("nan")
    q = statistics.quantiles(xs, n=4)
    return q[0], q[2]


def main():
    ev = Path(sys.argv[1]) / "ev"
    replays = (ev / "replays.log").read_text(errors="replace") if (ev / "replays.log").exists() else ""
    stats = {l: {"boots": 0, "landing": [], "missing": 0, "restore": [], "promote": [], "stall": [], "errors": 0,
                 "texts": set()} for l in LABELS}
    receipts = 0
    for rec in sorted(ev.glob("o[12]/b*-*/promote/receipt.json")):
        label = rec.parent.parent.name.split("-", 1)[1]
        r = json.loads(rec.read_text())
        receipts += 1
        s = stats[label]
        s["boots"] += 1
        stalls = []
        for run in r["runs"]:
            s["errors"] += len(run.get("errors") or [])
            s["texts"].add(run.get("tenant_text_sha"))
            if run["arm"] != "promote":
                continue
            lines = run.get("server_log_lines") or []
            hits = [float(m.group(1)) for m in map(LANDING.search, lines) if m]
            if hits:
                s["landing"].extend(hits)
            else:
                s["missing"] += 1
            s["promote"].append(sum("promote submitted off the tick" in x for x in lines))
            s["restore"].append(sum("restore submitted off the tick" in x for x in lines))
            stalls.append(run["stall_ms"])
        s["stall"].append(med(stalls))
    passes = replays.count("STALL REPLAY: PASS")
    admissible = receipts == 80 and passes >= receipts and all(
        s["boots"] == 10 and s["errors"] == 0 and len(s["texts"]) == 1 for s in stats.values())
    print(f"DAY54 CHECKS receipts={receipts} replays_pass={passes} admissible={admissible} "
          + " ".join(f"{l}:boots={s['boots']},errors={s['errors']},texts={len(s['texts'])}" for l, s in stats.items()))
    L = {}
    R = {}
    for label in LABELS:
        s = stats[label]
        L[label] = med(s["landing"])
        R[label] = med(s["restore"])
        p25, p75 = quart(s["landing"])
        print(f"DAY54 SLICE label={label} commit={COMMITS[label]} boots={s['boots']} promote_runs={len(s['promote'])}"
              f" landing_ms median={L[label]:.1f} p25={p25:.1f} p75={p75:.1f} n={len(s['landing'])}"
              f" landing_missing={s['missing']} promote_parks_per_run median={med(s['promote'])}"
              f" restore_parks_per_run median={R[label]} stall_median_of_boots={med(s['stall']):.1f}")
    reproduced = (L["e0"] > 2 * L["s7"]) or (R["e0"] != R["s7"])
    print(f"DAY54 ENDPOINTS e0 landing={L['e0']:.1f} restore={R['e0']} s7 landing={L['s7']:.1f} restore={R['s7']}"
          f" rule (e0 landing > 2 x s7 landing) or (restore parks differ) -> "
          f"{'reproduced' if reproduced else 'not_reproduced'}")
    mid = (L["e0"] + L["s7"]) / 2
    cls = {l: ("late" if L[l] > mid else "early") for l in LABELS}
    moves = []
    for a, b in zip(LABELS, LABELS[1:]):
        what = []
        if cls[a] != cls[b]:
            what.append(f"landing {cls[a]}->{cls[b]} ({L[a]:.1f}->{L[b]:.1f} ms)")
        if R[a] != R[b]:
            what.append(f"restore parks {R[a]}->{R[b]}")
        if what:
            moves.append(f"{b}={COMMITS[b]}: " + ", ".join(what))
    print("DAY54 MOVES midpoint_ms=%.1f " % mid + ("; ".join(moves) if moves else "none"))
    if not admissible:
        verdict = "void"
    elif not reproduced:
        verdict = "not_reproduced"
    else:
        verdict = "moved_at " + ",".join(m.split(":")[0] for m in moves) if moves else "no_single_slice"
    print(f"DAY54 VERDICT -> {verdict}")
    return 0 if admissible else 1


if __name__ == "__main__":
    sys.exit(main())
