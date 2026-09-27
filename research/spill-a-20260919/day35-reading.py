#!/usr/bin/env python3
"""WP-A day 35 reader (DAY35.md section 1: the F decision cell), written before the cell runs.

Input: ROOT/ab/o{1,2}/bNN-{hk,fk,off}/server.log and .../promote/receipt.json (stall_cell.py's promote arm), and
ROOT/ab/replays.log.
Complete: 30 boots, five per arm per order, every boot with a receipt, every receipt `errors` empty and every intruder
without an error, 30 `STALL REPLAY: PASS` lines. Nothing is read from an incomplete cell.
E2E: `wall_ms` of every promote-arm intruder run, pooled per arm per order (median), and the median of each boot's runs.
PIN: the `[prefix-host] promote: .. in Y ms` lines of each boot, steady = the second and later, pooled per arm per
order (median), and each boot's median.
Pair noise of a metric in an order: max(range of hk's five boot medians, range of fk's five boot medians).
KEEP iff in BOTH orders, for E2E AND PIN: median(hk) - median(fk) > the pair noise. Otherwise DELETE.
Readings: each ON arm's DAY28 1b `on_minus_off` against the order's off boots (rule <=+20.0), poll counts, the
helper's checksum time, the promote published timeline's first-poll offset.
"""
import glob
import json
import os
import re
import statistics
import sys

PROMOTE = re.compile(r"\[prefix-host\] promote: \d+ tokens, [\d.]+MB in ([\d.]+)ms")
PUB = re.compile(r"promote published off the tick: ticket complete after (\d+) poll\(s\)")
HELPER = re.compile(r"KV checksums on the hash helper \(([\d.]+)MB in ([\d.]+)ms\)")
ARMS = ("hk", "fk", "off")


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def stat(name, xs):
    if not xs:
        return f"{name} N=0"
    return f"{name} N={len(xs)} median={med(xs):.2f} min={min(xs):.2f} max={max(xs):.2f}"


def boot(d):
    out = {"e2e": [], "pin": [], "polls": [], "helper": [], "ok": True, "why": []}
    rec = os.path.join(d, "promote", "receipt.json")
    if not os.path.exists(rec):
        out["ok"] = False
        out["why"].append("no receipt")
    else:
        j = json.load(open(rec))
        if j.get("summary", {}).get("errors"):
            out["ok"] = False
            out["why"].append(f"errors={len(j['summary']['errors'])}")
        for r in j["runs"]:
            if r.get("arm") != "promote":
                continue
            i = r.get("intruder") or {}
            if "error" in i or "wall_ms" not in i:
                out["ok"] = False
                out["why"].append("intruder error")
                continue
            out["e2e"].append(i["wall_ms"])
    log = os.path.join(d, "server.log")
    if os.path.exists(log):
        k = 0
        for ln in open(log, errors="replace"):
            m = PROMOTE.search(ln)
            if m:
                k += 1
                if k >= 2:
                    out["pin"].append(float(m.group(1)))
            m = PUB.search(ln)
            if m:
                out["polls"].append(int(m.group(1)))
            m = HELPER.search(ln)
            if m:
                out["helper"].append(float(m.group(2)))
    return out


def main():
    root = sys.argv[1]
    cells = {}
    complete = True
    notes = []
    for order in ("o1", "o2"):
        for arm in ARMS:
            ds = sorted(glob.glob(os.path.join(root, "ab", order, f"b*-{arm}")))
            bs = [boot(d) for d in ds]
            if len(bs) != 5:
                complete = False
                notes.append(f"{order} {arm} boots={len(bs)}")
            for d, b in zip(ds, bs):
                if not b["ok"]:
                    complete = False
                    notes.append(f"{order}/{os.path.basename(d)} {';'.join(b['why'])}")
            cells[(order, arm)] = bs
    replays = os.path.join(root, "ab", "replays.log")
    passes = open(replays, errors="replace").read().count("STALL REPLAY: PASS") if os.path.exists(replays) else 0
    if passes != 30:
        complete = False
        notes.append(f"replays PASS={passes} of 30")
    for order in ("o1", "o2"):
        for arm in ARMS:
            bs = cells[(order, arm)]
            e2e = [x for b in bs for x in b["e2e"]]
            pin = [x for b in bs for x in b["pin"]]
            polls = [x for b in bs for x in b["polls"]]
            helper = [x for b in bs for x in b["helper"]]
            bm_e = [round(med(b["e2e"]), 2) for b in bs if b["e2e"]]
            bm_p = [round(med(b["pin"]), 2) for b in bs if b["pin"]]
            keys = sorted(set(polls))
            print(f"DAY35 READING order={order} arm={arm} boots={len(bs)} {stat('e2e', e2e)} boot-medians {bm_e} | "
                  f"{stat('pin-steady', pin)} boot-medians {bm_p} | polls {keys} (counts "
                  f"{[polls.count(p) for p in keys]}) | {stat('helper-ms', helper)}")
    if not complete:
        print(f"DAY35 F DECISION INCOMPLETE ({'; '.join(notes)}) -> nothing is read; the cell repeats whole once")
        sys.exit(2)
    verdict = True
    for order in ("o1", "o2"):
        off = [x for b in cells[(order, "off")] for x in b["e2e"]]
        for arm in ("hk", "fk"):
            on = [x for b in cells[(order, arm)] for x in b["e2e"]]
            d = med(on) - med(off)
            print(f"DAY35 1b READING order={order} arm={arm} on={med(on):.1f} off={med(off):.1f} on_minus_off={d:+.1f} "
                  f"rule <=+20.0 -> {'PASS' if d <= 20.0 else 'FAIL'}")
        for metric in ("e2e", "pin"):
            hk = [x for b in cells[(order, "hk")] for x in b[metric]]
            fk = [x for b in cells[(order, "fk")] for x in b[metric]]
            rng = []
            for arm in ("hk", "fk"):
                bm = [med(b[metric]) for b in cells[(order, arm)] if b[metric]]
                rng.append(max(bm) - min(bm))
            noise = max(rng)
            margin = med(hk) - med(fk)
            ok = margin > noise
            verdict &= ok
            print(f"DAY35 F CLAUSE order={order} metric={metric} hk={med(hk):.2f} fk={med(fk):.2f} hk-minus-fk="
                  f"{margin:+.2f} pair-noise={noise:.2f} (hk range {rng[0]:.2f}, fk range {rng[1]:.2f}) rule "
                  f"hk-minus-fk>pair-noise -> {'CLEARS' if ok else 'DOES NOT CLEAR'}")
    print(f"DAY35 F DECISION -> {'KEEP' if verdict else 'DELETE'}")


if __name__ == "__main__":
    main()
