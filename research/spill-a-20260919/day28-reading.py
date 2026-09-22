#!/usr/bin/env python3
"""WP-A day 28 reading of the double-park cell against option (a)'s pre-registered acceptance gate (DAY28.md
section 1, fixed before the run). Reuses day 25's per-run decomposition and admissibility rules.

Clause 1a (stall): per order, stall_median(ON) <= stall_median(OFF) + 2.0, from the DAY25 `stall` per-arm medians.
Clause 1b (e2e): per order, the request's end-to-end `on_minus_off <= +20.0`.
Clause 1c (in - completion on the owner thread): per ON run, the ledger line's `owner in-completion I ms`
(= pre-submit + hashing polls + take-back, bind and publish), median over all ON runs <= 12.0. On a tree without the
ledger line (the day-27 baseline) the figure is the wall `demote_in - demote_completion`, which is the owner-thread
figure there by construction (nothing between completion and publication ran elsewhere). On a tree WITH the helper a
run without a ledger line has no figure and the clause FAILS for it (a wall reading is never substituted).
Reported, not clauses: the wall `in - completion`, the helper's `hashed in H ms`, the count of hashing settles by mode,
`demote_completion`, the tenant's two largest gaps.

    day28-reading.py <ev_dir>
"""
import importlib.util
import json
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("d25", os.path.join(HERE, "day25-double-park-reading.py"))
d25 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d25)

LEDGER = re.compile(r"demote digests landed off the tick: ticket seq=(\d+), (\d+) payloads \(([\d.]+)MB\) hashed in "
                    r"([\d.]+)ms on the hash helper, landed after (\d+) poll\(s\) \(([^)]*)\); .*owner in-completion "
                    r"([\d.]+)ms; wall ([\d.]+)ms t0 to publication")


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def ledger(run):
    out = []
    for ln in run.get("server_log_lines", []):
        m = LEDGER.search(ln)
        if m:
            out.append({"seq": int(m.group(1)), "payloads": int(m.group(2)), "mb": float(m.group(3)),
                        "helper_ms": float(m.group(4)), "hash_polls": int(m.group(5)), "mode": m.group(6),
                        "owner_ic": float(m.group(7)), "wall": float(m.group(8))})
    return out


def main():
    ev = sys.argv[1]
    boots = []  # (order, arm, receipt)
    for order in ("o1", "o2"):
        d = os.path.join(ev, order)
        for b in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            rec = os.path.join(d, b, "promote", "receipt.json")
            if not os.path.exists(rec):
                print(f"  {order}/{b}: NO RECEIPT")
                continue
            boots.append((order, b.rsplit("-", 1)[-1], json.load(open(rec))))

    def promote_runs(r):
        return [x for x in r["runs"] if x["arm"] == "promote"]

    fails = 0
    for order in ("o1", "o2"):
        on = [r for o, a, r in boots if o == order and a == "on"]
        off = [r for o, a, r in boots if o == order and a == "off"]
        s_on = med([r["summary"]["stall_ms"]["median"] for r in on])
        s_off = med([r["summary"]["stall_ms"]["median"] for r in off])
        ok = s_on <= s_off + 2.0
        fails += 0 if ok else 1
        print(f"DAY28 CLAUSE 1a stall order={order} N_boots_on={len(on)} N_boots_off={len(off)} on_cell_median={s_on:.1f} "
              f"off_cell_median={s_off:.1f} rule on<=off+2.0 -> {'PASS' if ok else 'FAIL'}")
        e_on = [x["intruder"]["wall_ms"] for r in on for x in promote_runs(r) if x.get("intruder") and "wall_ms" in x["intruder"]]
        e_off = [x["intruder"]["wall_ms"] for r in off for x in promote_runs(r) if x.get("intruder") and "wall_ms" in x["intruder"]]
        dlt = med(e_on) - med(e_off)
        ok = dlt <= 20.0
        fails += 0 if ok else 1
        print(f"DAY28 CLAUSE 1b e2e order={order} N_runs_on={len(e_on)} N_runs_off={len(e_off)} on={med(e_on):.1f} "
              f"off={med(e_off):.1f} on_minus_off={dlt:+.1f} rule <=+20.0 -> {'PASS' if ok else 'FAIL'}")
    on_runs = [x for o, a, r in boots if a == "on" for x in promote_runs(r)]
    decs = [d25.decompose(x) for x in on_runs]
    ledgers = [l for x in on_runs for l in ledger(x)]
    wall_ic = [x["demote_in"] - x["demote_completion"] for x in decs
               if x.get("demote_in") is not None and x.get("demote_completion") is not None]
    completions = [x["demote_completion"] for x in decs if x.get("demote_completion") is not None]
    top1 = [x["top2"][0] for x in decs if len(x["top2"]) > 0]
    top2 = [x["top2"][1] for x in decs if len(x["top2"]) > 1]
    if ledgers:
        owner_ic = [l["owner_ic"] for l in ledgers]
        runs_with_demote = sum(1 for x in decs if x.get("demote_in") is not None)
        missing = runs_with_demote - len(ledgers)
        ok = med(owner_ic) <= 12.0 and missing == 0
        fails += 0 if ok else 1
        modes = {}
        for l in ledgers:
            modes[l["mode"]] = modes.get(l["mode"], 0) + 1
        print(f"DAY28 CLAUSE 1c owner in-completion N={len(owner_ic)} median={med(owner_ic):.2f} min={min(owner_ic):.2f} "
              f"max={max(owner_ic):.2f} runs_with_demote_without_ledger={missing} rule <=12.0 -> {'PASS' if ok else 'FAIL'}")
        helper = [l["helper_ms"] for l in ledgers]
        print(f"DAY28 REPORTED wall in-completion median={med(wall_ic):.1f} (N={len(wall_ic)}); demote_completion median="
              f"{med(completions):.1f}; helper hashed_in_ms median={med(helper):.1f} min={min(helper):.1f} max={max(helper):.1f}; "
              f"payloads={sorted(set(l['payloads'] for l in ledgers))} mb={sorted(set(l['mb'] for l in ledgers))}; "
              f"hash_polls median={med([l['hash_polls'] for l in ledgers]):.0f} max={max(l['hash_polls'] for l in ledgers)}; "
              f"settle modes={modes}; tenant top gaps: largest median={med(top1):.1f} second median={med(top2):.1f}")
    else:
        ok = med(wall_ic) <= 12.0
        fails += 0 if ok else 1
        print(f"DAY28 CLAUSE 1c owner in-completion (no ledger line on this tree: the wall figure IS the owner figure by "
              f"construction) N={len(wall_ic)} median={med(wall_ic):.1f} rule <=12.0 -> {'PASS' if ok else 'FAIL'}")
        print(f"DAY28 REPORTED demote_completion median={med(completions):.1f}; tenant top gaps: largest median={med(top1):.1f} "
              f"second median={med(top2):.1f}")
    print(f"DAY28 VERDICT clauses_failed={fails} -> {'ALL PASS' if fails == 0 else 'FAIL'}")


if __name__ == "__main__":
    main()
