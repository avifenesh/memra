#!/usr/bin/env python3
"""Day 25 reading of the retire-seam settle cell (DAY25.md pre-registration, fixed before the run).

Replays the harness's rule line from every receipt.json (`day25-retire-cell.py --replay`), checks
admissibility, then applies the pre-registered clauses below and prints one `DAY25 RETIRE READING`
line per clause per arm per pass plus one verdict line. Thresholds are constants here; nothing is
tuned after the run. The four boots of one collector hold are pass 1 = (off1, on1) and pass 2 =
(on2, off2); each boot carries two arms, `plain` (the shape as briefed) and `coincide` (the seam
exercised by a victim's abort in the seed's iteration).

The claim under test (registered before the run): the retire-seam settle adds at most one copy's
time (under 3.0 ms at the shape: the door-moved term is the 5184-token entry's KV planes, 154 MB on
the 27B, 1.5 ms even at 100 GB/s, an order of magnitude below the card's memory bandwidth class) to
the retiring request's own tick, and nothing to the tenant's stall median. Operationalized, one-sided,
per arm, per pass, ON minus OFF in ms:

- R1 the retiring request's own tick: median intruder wall (its hit restore, 64-row prime, seed and
  retire) ON minus OFF <= 3.0.
- R2 the tenant's stall: the harness's `stall_median` (max ITL minus p50) ON minus OFF <= 3.0. In the
  `coincide` arm the block sits in the seed's own iteration, which is the tenant's stall tick, so R2
  is the settle's clause there.
- R3 the tick after the stall tick: `two_tick_stall_median` ON minus OFF <= 3.0.
- R4, a reading with no rule: the ON arm's `settled_by` counts, `copy_ms` (submission to completion of
  every capture) list and median, and the retire-settled captures' `copy_ms` (an upper bound on the
  block's wait: the block starts after the submission and after the tick program's work between them).
- R5 the seam exercised (`coincide` only, admissibility of the seam claim): each ON receipt reads
  `settled_by_retire >= 8` of its 10 captures (an iteration-boundary straddle is a miss, recorded);
  under 8 the arm's clauses are still printed but the seam is `not_exercised` and decides nothing
  about the settle.

Admissibility (a receipt that fails decides nothing and is reported as such): replay PASS on every
receipt; `errors=0`; `tenant_text_identical=True` inside every receipt AND one tenant text sha across
all eight receipts; every intruder `prompt_tokens == base + deepen` and `cached_tokens == base`; ON:
`captures_published == 2N` and `seed_inserts == 2N` (the capture's publish prints the insert); OFF:
`captures_published == 0` and `seed_inserts == 2N`; `coincide`: `victim_aborts == 2N`; no
`[prefix-host] demote:` line in any timed run of any boot.

    day25-retire-reading.py <ev_dir>    # ev_dir holds off1/ on1/ on2/ off2/, each with plain/ and coincide/ receipt.json
"""
import json
import os
import subprocess
import sys

BOUND_MS = 3.0
SEAM_MIN = 8
HARNESS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "day25-retire-cell.py")
BOOTS = ("off1", "on1", "on2", "off2")
ARMS = ("plain", "coincide")
PASSES = {"pass1": ("off1", "on1"), "pass2": ("off2", "on2")}


def load(ev, boot, arm):
    p = os.path.join(ev, boot, arm, "receipt.json")
    rec = json.load(open(p))
    rep = subprocess.run([sys.executable, HARNESS, "--replay", p], capture_output=True, text=True)
    return rec, "RETIRE REPLAY: PASS" in rep.stdout


def main():
    ev = sys.argv[1]
    recs, admissible, why = {}, True, []
    shas = set()
    for boot in BOOTS:
        for arm in ARMS:
            rec, ok = load(ev, boot, arm)
            recs[(boot, arm)] = rec
            s = rec["summary"]
            n2 = 2 * rec["n_per_order"]
            base, deep = rec["base_tokens"], rec["deepen"]
            on = boot.startswith("on")
            print(rec["rule_line"])
            checks = {
                "replay": ok,
                "errors=0": not s["errors"],
                "tenant_text_identical": len(s["tenant_text_shas"]) == 1,
                "intruder_prompt_tokens": all(t == base + deep for t in s["intruder_prompt_tokens"]) and len(s["intruder_prompt_tokens"]) == n2,
                "intruder_cached_tokens": all(t == base for t in s["intruder_cached_tokens"]),
                "publish_route": (s["captures_published"] == (n2 if on else 0)) and s["seed_inserts"] == n2,
                "victim_aborts": s["victim_aborts"] == (n2 if arm == "coincide" else 0),
                "no_demote_in_timed_runs": not s["server_demote_ms"],
            }
            shas |= set(s["tenant_text_shas"])
            bad = [k for k, v in checks.items() if not v]
            print(f"  {boot}/{arm}: " + " ".join(f"{k}={'ok' if v else 'FAIL'}" for k, v in checks.items()))
            if bad:
                admissible = False
                why.append(f"{boot}/{arm}:{','.join(bad)}")
    if len(shas) != 1:
        admissible = False
        why.append(f"tenant_text_across_boots:{sorted(shas)}")
    print(f"  tenant text sha across the eight receipts: {sorted(shas)} admissible={admissible}" + (f" ({'; '.join(why)})" if why else ""))
    out, seam = {}, {}
    for arm in ARMS:
        for pname, (off, on) in PASSES.items():
            so, sn = recs[(off, arm)]["summary"], recs[(on, arm)]["summary"]
            for clause, key in (("R1", ("intruder_wall_ms", "median")), ("R2", ("stall_ms", "median")),
                                ("R3", ("two_tick_stall_ms", "median"))):
                vo, vn = so[key[0]][key[1]], sn[key[0]][key[1]]
                d = None if vo is None or vn is None else vn - vo
                verdict = "na" if d is None else ("within_bound" if d <= BOUND_MS else "on_over")
                out[f"{arm}-{clause}-{pname}"] = verdict
                print(f"DAY25 RETIRE READING {arm} {clause} {pname} ({key[0]} ON minus OFF, bound <= {BOUND_MS}): {verdict} "
                      f"(on={vn if vn is None else round(vn, 1)} off={vo if vo is None else round(vo, 1)} delta={d if d is None else round(d, 1)})")
            print(f"DAY25 RETIRE READING {arm} R4 {pname} (reading, no rule): ON settled_by_retire={sn['settled_by_retire']} "
                  f"settled_by_poll={sn['settled_by_poll']} settled_by_other={sn['settled_by_other']} "
                  f"copy_ms={[round(x, 1) for x in sn['copy_ms']['per_capture']]} copy_ms_median={sn['copy_ms']['median'] and round(sn['copy_ms']['median'], 1)} "
                  f"copy_ms_retire_settled={[round(x, 1) for x in sn['copy_ms_retire_settled']]} "
                  f"stall_ms_retire_settled={[round(x, 1) for x in sn['stall_ms_retire_settled']]} "
                  f"capture_polls={sn['capture_polls']} captures_submitted_mb={sn['captures_submitted_mb'][:1]}")
            if arm == "coincide":
                seam[pname] = "exercised" if sn["settled_by_retire"] >= SEAM_MIN else "not_exercised"
                print(f"DAY25 RETIRE READING coincide R5 {pname} (seam exercised, >= {SEAM_MIN} of 10 retire-settled): {seam[pname]} "
                      f"(settled_by_retire={sn['settled_by_retire']})")
    if not admissible:
        verdict = "UNDECIDED (a receipt failed admissibility; decides nothing)"
    elif any(v != "within_bound" for v in out.values()):
        verdict = "FAILS (" + ", ".join(k for k, v in out.items() if v != "within_bound") + " over the bound)"
    elif all(v == "exercised" for v in seam.values()):
        verdict = "HOLDS (R1, R2, R3 within 3.0 ms in both arms and both passes; the seam exercised in both passes)"
    else:
        verdict = "HOLDS AT THE SHAPE, SEAM NOT EXERCISED (R1, R2, R3 within 3.0 ms; coincide R5 " + \
            ", ".join(f"{k}={v}" for k, v in seam.items()) + ")"
    print(f"DAY25 RETIRE VERDICT: admissible={admissible} " + " ".join(f"{k}={v}" for k, v in out.items())
          + " " + " ".join(f"seam-{k}={v}" for k, v in seam.items()) + f" -> {verdict}")
    return 0 if admissible else 1


if __name__ == "__main__":
    sys.exit(main())
