#!/usr/bin/env python3
"""WP-A day 49 reader (DAY49.md section 1: OWED items 7 and 8), written before its cell runs.

usage: day49-reading.py ROOT
Input: ROOT/{nofree,free,long}/bNN/server.log (the attribution binary's `demote helper split` and `demote pre-submit
split` lines) and .../receipt.json (stall_cell.py). Steady demotes: the 4th on for `free`, the 2nd on for `nofree` and
`long`, per boot in log order, pooled per arm. Complete: every boot with a receipt, `errors` empty, at least five steady
helper-split lines per `nofree` and `free` boot, and a pre-submit split line for every helper split line.
Item 8 (pages = copy bytes / 4096): H attributed when nofree's copy minflt median >= 0.5 x pages, free's <= 0.25 x
pages, and nofree's copy ms median exceeds free's by >= 5 ms; H refuted when the two minflt medians are within a factor
of 2; otherwise not placed. Item 7: attributed exactly when item 8 is; otherwise not placed; the pre-submit split is
printed per arm either way (the long arm a reading).
"""
import glob
import json
import os
import re
import statistics
import sys

HELPER = re.compile(r"demote helper split: ticket seq=(\d+) copy ([\d.]+) ms over ([\d.]+) MB \(minflt \+(-?\d+)\), "
                    r"hash ([\d.]+) ms \(helper ([\d.]+) ms\)")
PRE = re.compile(r"demote pre-submit split: ticket seq=(\d+) leases ([\d.]+) ms \((\d+) pinned, ([\d.]+) MB, minflt "
                 r"\+(-?\d+)\), register ([\d.]+) ms, spans ([\d.]+) ms, other ([\d.]+) ms \(pre-submit ([\d.]+) ms\)")
STEADY_FROM = {"nofree": 1, "free": 3, "long": 1}


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def boot(d, arm):
    out = {"ok": True, "why": [], "helper": [], "pre": []}
    recs = glob.glob(os.path.join(d, "*", "receipt.json"))
    if not recs:
        out["ok"] = False
        out["why"].append("no receipt")
    for r in recs:
        j = json.load(open(r))
        if j.get("summary", {}).get("errors"):
            out["ok"] = False
            out["why"].append(f"errors={len(j['summary']['errors'])}")
    log = os.path.join(d, "server.log")
    helper, pre = [], []
    for ln in open(log, errors="replace") if os.path.exists(log) else []:
        m = HELPER.search(ln)
        if m:
            helper.append({"copy_ms": float(m.group(2)), "mb": float(m.group(3)), "minflt": int(m.group(4)),
                           "hash_ms": float(m.group(5)), "helper_ms": float(m.group(6))})
        m = PRE.search(ln)
        if m:
            pre.append({"leases_ms": float(m.group(2)), "leases": int(m.group(3)), "lease_mb": float(m.group(4)),
                        "lease_minflt": int(m.group(5)), "register_ms": float(m.group(6)),
                        "spans_ms": float(m.group(7)), "other_ms": float(m.group(8)), "pre_ms": float(m.group(9))})
    k = STEADY_FROM[arm]
    out["helper"], out["pre"] = helper[k:], pre[k:]
    if arm != "long" and len(out["helper"]) < 5:
        out["ok"] = False
        out["why"].append(f"steady helper lines={len(out['helper'])}")
    if len(pre) < len(helper):
        out["ok"] = False
        out["why"].append(f"pre-submit lines={len(pre)} for {len(helper)} helper lines")
    return out


def main():
    root = sys.argv[1]
    pools, complete, notes = {}, True, []
    for arm in ("nofree", "free", "long"):
        bs = [boot(d, arm) for d in sorted(glob.glob(os.path.join(root, arm, "b*")))]
        if not bs:
            complete = False
            notes.append(f"{arm} no boots")
        for i, b in enumerate(bs):
            if not b["ok"]:
                complete = False
                notes.append(f"{arm}/b{i + 1:02d} {';'.join(b['why'])}")
        pools[arm] = {"helper": [x for b in bs for x in b["helper"]], "pre": [x for b in bs for x in b["pre"]]}
    for arm in ("nofree", "free", "long"):
        h, p = pools[arm]["helper"], pools[arm]["pre"]
        f = lambda xs, key: med([x[key] for x in xs])
        print(f"DAY49 READING arm={arm} helper N={len(h)} copy_ms={f(h, 'copy_ms'):.2f} mb={f(h, 'mb'):.1f} "
              f"minflt={f(h, 'minflt'):.0f} hash_ms={f(h, 'hash_ms'):.2f} helper_ms={f(h, 'helper_ms'):.2f} | "
              f"pre-submit N={len(p)} leases_ms={f(p, 'leases_ms'):.2f} leases={f(p, 'leases'):.0f} "
              f"lease_mb={f(p, 'lease_mb'):.1f} lease_minflt={f(p, 'lease_minflt'):.0f} "
              f"register_ms={f(p, 'register_ms'):.2f} spans_ms={f(p, 'spans_ms'):.2f} other_ms={f(p, 'other_ms'):.2f} "
              f"pre_ms={f(p, 'pre_ms'):.2f}")
    if not complete:
        print(f"DAY49 INCOMPLETE ({'; '.join(notes)}) -> nothing is read; the cell repeats whole once")
        sys.exit(2)
    nf, fr = pools["nofree"]["helper"], pools["free"]["helper"]
    pages = med([x["mb"] for x in nf]) * 1e6 / 4096
    nf_flt, fr_flt = med([x["minflt"] for x in nf]), med([x["minflt"] for x in fr])
    nf_ms, fr_ms = med([x["copy_ms"] for x in nf]), med([x["copy_ms"] for x in fr])
    attributed = nf_flt >= 0.5 * pages and fr_flt <= 0.25 * pages and nf_ms - fr_ms >= 5.0
    lo, hi = sorted([max(nf_flt, 1.0), max(fr_flt, 1.0)])
    refuted = hi / lo <= 2.0
    item8 = "H attributed" if attributed else ("H refuted" if refuted else "not placed")
    print(f"DAY49 ITEM8 pages={pages:.0f} nofree_minflt={nf_flt:.0f} (rule >= {0.5 * pages:.0f}) free_minflt={fr_flt:.0f} "
          f"(rule <= {0.25 * pages:.0f}) copy_ms nofree={nf_ms:.2f} free={fr_ms:.2f} diff={nf_ms - fr_ms:+.2f} (rule >= +5.0) "
          f"-> {item8}")
    print(f"DAY49 ITEM7 -> {'b1 step attributed to H (the heap first touch)' if attributed else 'not placed'} "
          f"(the current pre-submit split above)")


if __name__ == "__main__":
    main()
