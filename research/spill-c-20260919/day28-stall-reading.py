#!/usr/bin/env python3
"""Day 28 reading of the isolating stall cell (DAY28.md pre-registration, fixed before the run).

Replays the harness's rule line from every receipt.json (`day28_stall_cell.py --replay`), checks the
pre-registered admissibility per arm, then computes per pass:

  capture class   share(arm) = stall_median(capture-arm) - stall_median(prime)      [the prime cancels]
                  unc(arm)   = sqrt(IQR(capture-arm)^2 + IQR(prime)^2)             [IQR of the 10 per-run stalls]
                  moved      = stall_median(capture-on) - stall_median(capture-off), unc in quadrature
  restore class   share(arm) = stall_median(exact-arm)      [no subtraction: the intruder has no prime of its own]
                  unc(arm)   = IQR(exact-arm)
                  moved      = stall_median(exact-on) - stall_median(exact-off), unc in quadrature

A share or a moved figure is `isolated` when |value| > unc, otherwise `under_resolution`. No threshold is
tuned here; the IQR is the cell's own spread and nothing else. One `DAY28 ISOLATION` line per quantity,
one summary line. `admissible=False` decides nothing and says so.

    day28-stall-reading.py <ev_dir>    # ev_dir holds pass1/ pass2/, each with <cell>/run/receipt.json
"""
import json
import math
import os
import subprocess
import sys

HARNESS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "day28_stall_cell.py")
CELLS = ("prime", "capture-off", "capture-on", "exact-off", "exact-on")


def pct(xs, q):
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def load(ev, p, cell):
    path = os.path.join(ev, f"pass{p}", cell, "run", "receipt.json")
    rec = json.load(open(path))
    rep = subprocess.run([sys.executable, HARNESS, "--replay", path], capture_output=True, text=True)
    return rec, "STALL REPLAY: PASS" in rep.stdout


def admissible(cell, rec):
    s = rec["summary"]
    ok = not s["errors"] and len(s["tenant_text_shas"]) == 1
    why = []
    if s["errors"]:
        why.append(f"errors={len(s['errors'])}")
    if len(s["tenant_text_shas"]) != 1:
        why.append("tenant text differs")
    arm = [r for r in rec["runs"] if r["arm"] != "idle"]
    if cell != "prime" and (s["server_demote_ms"] or s["server_promote_ms"]):
        ok = False
        why.append(f"demote/promote inside the cell (demotes={len(s['server_demote_ms'])} promotes={len(s['server_promote_ms'])})")
    if cell.startswith("capture"):
        bad = [r["run_id"] for r in arm if not r["intruder"] or "repost_cached_tokens" not in r["intruder"]
               or r["intruder"]["repost_cached_tokens"] is None
               or r["intruder"]["repost_cached_tokens"] < r["intruder"]["repost_prompt_tokens"] - 64]
        if bad:
            ok = False
            why.append(f"re-post not a grid hit in runs {bad}")
    if cell.startswith("exact"):
        bad = [r["run_id"] for r in arm if not r["intruder"] or r["intruder"].get("cached_tokens") is None
               or r["intruder"]["cached_tokens"] != r["intruder"]["prompt_tokens"]]
        if bad:
            ok = False
            why.append(f"not a whole-entry zero-suffix hit in runs {bad}")
        seed = [x for x in rec["setup"] if x.get("seed")]
        if not seed or seed[0]["ongrid_tokens"] % 32 != 0:
            ok = False
            why.append("seed prompt off the grid")
    return ok, "; ".join(why) or "errors=0, tenant text identical, every intruder the pre-registered hit shape"


def stalls(rec):
    return [r["stall_ms"] for r in rec["runs"] if r["arm"] != "idle" and "stall_ms" in r]


def med_iqr(xs):
    return pct(xs, .5), pct(xs, .75) - pct(xs, .25)


def classify(v, u):
    return "isolated" if abs(v) > u else "under_resolution"


def main():
    ev = sys.argv[1]
    recs, adm = {}, {}
    for p in (1, 2):
        for cell in CELLS:
            rec, replay = load(ev, p, cell)
            recs[(p, cell)] = rec
            ok, why = admissible(cell, rec)
            adm[(p, cell)] = ok and replay
            print(rec["rule_line"])
            print(f"  pass{p}/{cell}: replay={'PASS' if replay else 'FAIL'} admissible={ok} ({why})")
    verdict = []
    for p in (1, 2):
        pm, pu = med_iqr(stalls(recs[(p, "prime")]))
        print(f"DAY28 ISOLATION pass={p} prime stall_median={pm:.1f} iqr={pu:.1f} n_per_order=5 pooled=10")
        shares = {}
        for arm in ("off", "on"):
            cm, cu = med_iqr(stalls(recs[(p, f"capture-{arm}")]))
            share, unc = cm - pm, math.sqrt(cu ** 2 + pu ** 2)
            shares[arm] = (cm, cu)
            a = adm[(p, f"capture-{arm}")] and adm[(p, "prime")]
            print(f"DAY28 ISOLATION pass={p} class=capture arm={arm} stall_median={cm:.1f} iqr={cu:.1f} "
                  f"share={share:+.1f} unc={unc:.1f} -> {classify(share, unc) if a else 'inadmissible'}")
        moved = shares["on"][0] - shares["off"][0]
        munc = math.sqrt(shares["on"][1] ** 2 + shares["off"][1] ** 2)
        a = adm[(p, "capture-on")] and adm[(p, "capture-off")]
        c = classify(moved, munc) if a else "inadmissible"
        print(f"DAY28 ISOLATION pass={p} class=capture on_minus_off={moved:+.1f} unc={munc:.1f} -> {c}")
        verdict.append(f"capture pass{p} on-off {moved:+.1f} (unc {munc:.1f}) {c}")
        ex = {}
        for arm in ("off", "on"):
            em, eu = med_iqr(stalls(recs[(p, f"exact-{arm}")]))
            ex[arm] = (em, eu)
            s = recs[(p, f"exact-{arm}")]["summary"]
            a = adm[(p, f"exact-{arm}")]
            print(f"DAY28 ISOLATION pass={p} class=restore arm={arm} stall_median={em:.1f} iqr={eu:.1f} "
                  f"share={em:+.1f} unc={eu:.1f} arm_p99={s['arm']['p99']:.1f} arm_max={s['arm']['max']:.1f} "
                  f"server_restore_ms={[round(x, 1) for x in s['server_restore_ms']]} -> {classify(em, eu) if a else 'inadmissible'}")
        moved = ex["on"][0] - ex["off"][0]
        munc = math.sqrt(ex["on"][1] ** 2 + ex["off"][1] ** 2)
        a = adm[(p, "exact-on")] and adm[(p, "exact-off")]
        c = classify(moved, munc) if a else "inadmissible"
        print(f"DAY28 ISOLATION pass={p} class=restore on_minus_off={moved:+.1f} unc={munc:.1f} -> {c}")
        verdict.append(f"restore pass{p} on-off {moved:+.1f} (unc {munc:.1f}) {c}")
    print("DAY28 ISOLATION VERDICT: " + "; ".join(verdict) + f"; admissible={all(adm.values())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
