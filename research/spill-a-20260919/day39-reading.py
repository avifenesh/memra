#!/usr/bin/env python3
"""WP-A day 39 reader (DAY39.md section 5: design T's cells), written before either cell runs.

usage: day39-reading.py <root> target|5090
Input: ROOT/ab/o{1,2}/bNN-<arm>/server.log and .../promote/receipt.json (stall_cell.py's promote arm), ROOT/ab/replays.log.
target: arms hk ft f1 off, 40 boots (five per arm per order); 5090: arms ft f1 off, 30 boots; both `--n 5` per boot
(stall_cell's n per order: ten promote runs per boot, nine steady; section 5a).
Complete: every boot present with a receipt, every receipt `errors` empty, every intruder without an error, one
`STALL REPLAY: PASS` per boot. Nothing is read from an incomplete cell.
E2E: `wall_ms` of every promote-arm intruder run, pooled per arm per order (median), and each boot's median.
PIN: the `[prefix-host] promote: .. in Y ms` lines of each boot, steady = the second and later, pooled and per boot.
POLLS: the `promote published off the tick: ticket complete after N poll(s)` lines, steady = the second and later of a
boot, pooled per arm over both orders.
Pair noise of a metric in an order for arms (a, b): max(range of a's five boot medians, range of b's).
target (a): ft's steady polls == 1 on at least 80 of its 90. (b): in BOTH orders, for E2E AND PIN,
median(hk) - median(ft) > pair noise (hk, ft). 5090 (e): in BOTH orders, for E2E AND PIN, median(ft) - median(f1) <=
pair noise (ft, f1). Readings: f1 against ft on the target, every ON arm's DAY28 1b on_minus_off (rule <=+20.0), polls of
every arm, the helper's checksum time.
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


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def stat(name, xs):
    if not xs:
        return f"{name} N=0"
    return f"{name} N={len(xs)} median={med(xs):.2f} min={min(xs):.2f} max={max(xs):.2f}"


def boot(d, n):
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
        if len(out["e2e"]) != n:
            out["ok"] = False
            out["why"].append(f"promote runs={len(out['e2e'])} of {n}")
    log = os.path.join(d, "server.log")
    if os.path.exists(log):
        k = q = 0
        for ln in open(log, errors="replace"):
            m = PROMOTE.search(ln)
            if m:
                k += 1
                if k >= 2:
                    out["pin"].append(float(m.group(1)))
            m = PUB.search(ln)
            if m:
                q += 1
                if q >= 2:
                    out["polls"].append(int(m.group(1)))
            m = HELPER.search(ln)
            if m:
                out["helper"].append(float(m.group(2)))
    return out


def noise(cells, order, metric, arms):
    rng = []
    for arm in arms:
        bm = [med(b[metric]) for b in cells[(order, arm)] if b[metric]]
        rng.append(max(bm) - min(bm))
    return max(rng), rng


def pooled(cells, order, arm, metric):
    return [x for b in cells[(order, arm)] for x in b[metric]]


def main():
    root, mode = sys.argv[1], sys.argv[2]
    arms = ("hk", "ft", "f1", "off") if mode == "target" else ("ft", "f1", "off")
    n = 10  # promote runs per boot under `--n 5`
    boots_expected = 10 * len(arms)
    cells, complete, notes = {}, True, []
    for order in ("o1", "o2"):
        for arm in arms:
            ds = sorted(glob.glob(os.path.join(root, "ab", order, f"b*-{arm}")))
            bs = [boot(d, n) for d in ds]
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
    if passes != boots_expected:
        complete = False
        notes.append(f"replays PASS={passes} of {boots_expected}")
    for order in ("o1", "o2"):
        for arm in arms:
            bs = cells[(order, arm)]
            polls = pooled(cells, order, arm, "polls")
            keys = sorted(set(polls))
            print(f"DAY39 READING mode={mode} order={order} arm={arm} boots={len(bs)} "
                  f"{stat('e2e', pooled(cells, order, arm, 'e2e'))} boot-medians "
                  f"{[round(med(b['e2e']), 2) for b in bs if b['e2e']]} | {stat('pin-steady', pooled(cells, order, arm, 'pin'))} "
                  f"boot-medians {[round(med(b['pin']), 2) for b in bs if b['pin']]} | steady polls {keys} (counts "
                  f"{[polls.count(p) for p in keys]}) | {stat('helper-ms', pooled(cells, order, arm, 'helper'))}")
    if not complete:
        print(f"DAY39 T {mode.upper()} INCOMPLETE ({'; '.join(notes)}) -> nothing is read; the cell repeats whole once")
        sys.exit(2)
    for order in ("o1", "o2"):
        off = pooled(cells, order, "off", "e2e")
        for arm in arms:
            if arm == "off":
                continue
            on = pooled(cells, order, arm, "e2e")
            d = med(on) - med(off)
            print(f"DAY39 1b READING order={order} arm={arm} on={med(on):.1f} off={med(off):.1f} on_minus_off={d:+.1f} "
                  f"rule <=+20.0 -> {'PASS' if d <= 20.0 else 'FAIL'}")
    ok_all = True
    if mode == "target":
        ft_polls = [x for o in ("o1", "o2") for x in pooled(cells, o, "ft", "polls")]
        ones = ft_polls.count(1)
        a = len(ft_polls) == 90 and ones >= 80
        ok_all &= a
        print(f"DAY39 T CLAUSE (a) ft steady promotes N={len(ft_polls)} polls==1 {ones} rule N=90 and >=80 -> "
              f"{'PASS' if a else 'FAIL'}")
        for order in ("o1", "o2"):
            for metric in ("e2e", "pin"):
                hk, ft = med(pooled(cells, order, "hk", metric)), med(pooled(cells, order, "ft", metric))
                nz, rng = noise(cells, order, metric, ("hk", "ft"))
                b = hk - ft > nz
                ok_all &= b
                print(f"DAY39 T CLAUSE (b) order={order} metric={metric} hk={hk:.2f} ft={ft:.2f} hk-minus-ft={hk - ft:+.2f} "
                      f"pair-noise={nz:.2f} (hk range {rng[0]:.2f}, ft range {rng[1]:.2f}) rule hk-minus-ft>pair-noise -> "
                      f"{'CLEARS' if b else 'DOES NOT CLEAR'}")
                f1 = med(pooled(cells, order, "f1", metric))
                nz1, _ = noise(cells, order, metric, ("f1", "ft"))
                print(f"DAY39 T READING order={order} metric={metric} f1={f1:.2f} ft={ft:.2f} f1-minus-ft={f1 - ft:+.2f} "
                      f"pair-noise(f1, ft)={nz1:.2f}")
        print(f"DAY39 T TARGET (a) and (b) -> {'PASS' if ok_all else 'FAIL'} (clause (c) is the gates and the unit cells)")
    else:
        for order in ("o1", "o2"):
            for metric in ("e2e", "pin"):
                ft, f1 = med(pooled(cells, order, "ft", metric)), med(pooled(cells, order, "f1", metric))
                nz, rng = noise(cells, order, metric, ("ft", "f1"))
                e = ft - f1 <= nz
                ok_all &= e
                print(f"DAY39 T CLAUSE (e) order={order} metric={metric} ft={ft:.2f} f1={f1:.2f} ft-minus-f1={ft - f1:+.2f} "
                      f"pair-noise={nz:.2f} (ft range {rng[0]:.2f}, f1 range {rng[1]:.2f}) rule ft-minus-f1<=pair-noise -> "
                      f"{'PASS' if e else 'FAIL'}")
        print(f"DAY39 T 5090 (e) -> {'PASS' if ok_all else 'FAIL'} (the gates and the unit cells are read from their logs)")


if __name__ == "__main__":
    main()
