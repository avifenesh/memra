#!/usr/bin/env python3
"""Day 30 reading of the capture-share cell (DAY30.md pre-registration, fixed before the run).

Replays the harness's rule line from every receipt.json (`day28_stall_cell.py --replay`), checks the
pre-registered admissibility per arm (including the server-log clauses that prove the refused arms
reached the seed insert and were refused before any copy), then computes per pass:

  base            = stall_median(refused-off)                    [the prime with the seed boundary and NO copy]
  share(arm)      = stall_median(capture-arm) - base             [the capture arm's own on-tick share]
  unc(arm)        = sqrt(IQR(capture-arm)^2 + IQR(refused-off)^2)
  door_idle       = stall_median(refused-on) - stall_median(refused-off)   [a control: nothing to route]
  on_minus_off    = stall_median(capture-on) - stall_median(capture-off)   [day 28's clean quantity, repeated]

`stall_median` and IQR are over the boot's 10 arm runs (N=5 per arm per order, both orders); a quantity is
`isolated` when |value| > unc, otherwise `under_resolution`. No threshold is tuned here. One `DAY30
CAPTURE-SHARE` line per quantity, one summary line. `admissible=False` decides nothing and says so.

    day30-stall-reading.py <ev_dir>    # ev_dir holds pass1/ pass2/, each with <cell>/{run/receipt.json,server.log}
"""
import json
import math
import os
import re
import subprocess
import sys

HARNESS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "day28_stall_cell.py")
CELLS = ("refused-off", "refused-on", "capture-off", "capture-on")
REFUSED_BUDGET_BYTES = 128 * 1024 * 1024
REFUSAL = re.compile(r"\[prefix-cache\] insert refused: entry (\d+) exceeds budget (\d+) \(snapshot preflight")


def pct(xs, q):
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def load(ev, p, cell):
    d = os.path.join(ev, f"pass{p}", cell)
    path = os.path.join(d, "run", "receipt.json")
    rec = json.load(open(path))
    rep = subprocess.run([sys.executable, HARNESS, "--replay", path], capture_output=True, text=True)
    log = open(os.path.join(d, "server.log"), errors="replace").read()
    return rec, "STALL REPLAY: PASS" in rep.stdout, log


def count(log, needle):
    return sum(1 for ln in log.splitlines() if needle in ln)


def admissible(cell, rec, log):
    s = rec["summary"]
    ok = not s["errors"] and len(s["tenant_text_shas"]) == 1
    why = []
    if s["errors"]:
        why.append(f"errors={len(s['errors'])}")
    if len(s["tenant_text_shas"]) != 1:
        why.append("tenant text differs")
    if s["server_demote_ms"] or s["server_promote_ms"]:
        ok = False
        why.append(f"demote/promote inside the cell (demotes={len(s['server_demote_ms'])} promotes={len(s['server_promote_ms'])})")
    arm = [r for r in rec["runs"] if r["arm"] != "idle"]
    seeds_5088 = count(log, "insert (seed): 5088 tokens")
    grid_refusals = count(log, "seed REFUSED (grid)")
    captures = count(log, "capture submitted off the tick")
    disabled = count(log, "DISABLED") + count(log, "capture refused (contracts door)")
    if grid_refusals:
        ok = False
        why.append(f"{grid_refusals} `seed REFUSED (grid)` line(s): the prime did not stop at the boundary")
    if disabled:
        ok = False
        why.append(f"{disabled} refused/DISABLED route line(s)")
    if cell.startswith("refused"):
        refusals = [(int(m.group(1)), int(m.group(2))) for m in REFUSAL.finditer(log)]
        if not refusals:
            ok = False
            why.append("no typed oversize refusal line")
        elif any(b != REFUSED_BUDGET_BYTES or e <= b for e, b in refusals):
            ok = False
            why.append(f"refusal line not the 128 MiB oversize shape: {refusals[:2]}")
        if seeds_5088 or captures:
            ok = False
            why.append(f"an entry was published or routed (seed inserts={seeds_5088} captures={captures})")
        if count(log, "hit: 5088 of"):
            ok = False
            why.append("a 5088-token hit exists in a refused boot")
    if cell.startswith("capture"):
        bad = [r["run_id"] for r in arm if not r["intruder"] or "repost_cached_tokens" not in r["intruder"]
               or r["intruder"]["repost_cached_tokens"] is None
               or r["intruder"]["repost_cached_tokens"] < r["intruder"]["repost_prompt_tokens"] - 64]
        if bad:
            ok = False
            why.append(f"re-post not a grid hit in runs {bad}")
        if seeds_5088 < 10:
            ok = False
            why.append(f"fewer than ten 5088-token seed inserts ({seeds_5088})")
        if cell == "capture-on" and captures < 10:
            ok = False
            why.append(f"fewer than ten capture submissions in the ON arm ({captures})")
        if cell == "capture-off" and captures:
            ok = False
            why.append(f"{captures} capture submission(s) in the OFF arm")
    return ok, "; ".join(why) or "errors=0, tenant text identical, every server-log clause of the arm holds"


def stalls(rec):
    return [r["stall_ms"] for r in rec["runs"] if r["arm"] != "idle" and "stall_ms" in r]


def med_iqr(xs):
    return pct(xs, .5), pct(xs, .75) - pct(xs, .25)


def classify(v, u):
    return "isolated" if abs(v) > u else "under_resolution"


def main():
    ev = sys.argv[1]
    recs, adm, logs = {}, {}, {}
    for p in (1, 2):
        for cell in CELLS:
            rec, replay, log = load(ev, p, cell)
            recs[(p, cell)] = rec
            logs[(p, cell)] = log
            ok, why = admissible(cell, rec, log)
            adm[(p, cell)] = ok and replay
            print(rec["rule_line"])
            print(f"  pass{p}/{cell}: replay={'PASS' if replay else 'FAIL'} admissible={ok} ({why}); "
                  f"server-log: seed_inserts_5088={count(log, 'insert (seed): 5088 tokens')} "
                  f"oversize_refusals={len(REFUSAL.findall(log))} capture_submitted={count(log, 'capture submitted off the tick')}")
    verdict = []
    for p in (1, 2):
        m = {cell: med_iqr(stalls(recs[(p, cell)])) for cell in CELLS}
        bm, bu = m["refused-off"]
        s = recs[(p, "refused-off")]["summary"]
        print(f"DAY30 CAPTURE-SHARE pass={p} base=refused-off stall_median={bm:.1f} iqr={bu:.1f} "
              f"arm_p99={s['arm']['p99']:.1f} arm_max={s['arm']['max']:.1f} n_per_order=5 pooled=10")
        dm, du = m["refused-on"]
        v, u = dm - bm, math.sqrt(du ** 2 + bu ** 2)
        a = adm[(p, "refused-on")] and adm[(p, "refused-off")]
        c = classify(v, u) if a else "inadmissible"
        print(f"DAY30 CAPTURE-SHARE pass={p} control refused on_minus_off={v:+.1f} unc={u:.1f} "
              f"(refused-on {dm:.1f} iqr {du:.1f}) -> {c}")
        verdict.append(f"pass{p} refused on-off {v:+.1f} (unc {u:.1f}) {c}")
        for arm in ("off", "on"):
            cm, cu = m[f"capture-{arm}"]
            share, unc = cm - bm, math.sqrt(cu ** 2 + bu ** 2)
            s = recs[(p, f"capture-{arm}")]["summary"]
            a = adm[(p, f"capture-{arm}")] and adm[(p, "refused-off")]
            c = classify(share, unc) if a else "inadmissible"
            print(f"DAY30 CAPTURE-SHARE pass={p} class=capture arm={arm} stall_median={cm:.1f} iqr={cu:.1f} "
                  f"share={share:+.1f} unc={unc:.1f} arm_p99={s['arm']['p99']:.1f} arm_max={s['arm']['max']:.1f} -> {c}")
            verdict.append(f"pass{p} share {arm} {share:+.1f} (unc {unc:.1f}) {c}")
        om, ou = m["capture-on"]
        fm, fu = m["capture-off"]
        v, u = om - fm, math.sqrt(ou ** 2 + fu ** 2)
        a = adm[(p, "capture-on")] and adm[(p, "capture-off")]
        c = classify(v, u) if a else "inadmissible"
        print(f"DAY30 CAPTURE-SHARE pass={p} class=capture on_minus_off={v:+.1f} unc={u:.1f} -> {c}")
        verdict.append(f"pass{p} capture on-off {v:+.1f} (unc {u:.1f}) {c}")
    print("DAY30 CAPTURE-SHARE VERDICT: " + "; ".join(verdict) + f"; admissible={all(adm.values())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
