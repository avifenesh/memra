#!/usr/bin/env python3
"""WP-A day 34 reader (DAY34.md section 2: acceptance (c) and (d) on the 5090 A/B), written before the A/B runs.

Input: ROOT/ab/o{1,2}/bNN-{d33,d34}/server.log and .../promote/receipt.json (stall_cell.py's promote arm).
(c)  On the d34 boots: the completing poll's owner hold, P minus T of the `poll k at +P ms (tick K, its top +T ms)
     complete` term of each `promote published off the tick` line's timeline (an upper bound: it includes the tick top's
     demote poll). Steady = the second and later promote of each boot. Rule: median <= 1.5 ms and max <= 3.0 ms over
     N >= 20; every d34 H2D receipt names `KV checksums on the hash helper`.
(d)  Per order: the intruder's e2e (`wall_ms` of every promote-arm run) median on d34 at most d33's plus 1.0 ms.
Readings (not clauses): the same hold on the d33 boots, `promote_in`, poll counts, the helper's checksum time.
--pro D33_EV D34_EV (DAY34 section 4, the target card): over the ON boots (`o*/b*-on/`) of the day-26 double-park runs
of the day-33 and day-34 binaries in one hold: (c) the day-34 run's landing-poll hold, the same rule; (d) its ON
intruder e2e median at most the day-33 run's plus 1.0 ms per order (DAY28 1b is read by `day28-reading.py`).
"""
import glob
import json
import os
import re
import statistics
import sys

PUB = re.compile(r"promote published off the tick: ticket complete after (\d+) poll\(s\).*timeline from t0: (.*)$")
POLL = re.compile(r"poll \d+ at \+([\d.]+)ms \(tick \d+, its top ([+-][\d.]+)ms\) complete")
PROMOTE = re.compile(r"\[prefix-host\] promote: \d+ tokens, [\d.]+MB in ([\d.]+)ms")
HELPER = re.compile(r"KV checksums on the hash helper \(([\d.]+)MB in ([\d.]+)ms\)")
RECEIPT = "contracts door H2D receipt: ticket issuer="


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def stat(name, xs):
    if not xs:
        return f"{name} N=0"
    return f"{name} N={len(xs)} median={med(xs):.2f} min={min(xs):.2f} max={max(xs):.2f}"


def boot(d):
    out = {"hold": [], "polls": [], "promote_in": [], "helper_ms": [], "receipts": 0, "receipts_helper": 0, "e2e": []}
    log = os.path.join(d, "server.log")
    if os.path.exists(log):
        k = pk = 0
        for ln in open(log, errors="replace"):
            if RECEIPT in ln:
                out["receipts"] += 1
                m = HELPER.search(ln)
                if m:
                    out["receipts_helper"] += 1
                    out["helper_ms"].append(float(m.group(2)))
            m = PUB.search(ln)
            if m:
                k += 1
                p = POLL.search(m.group(2))
                if k >= 2 and p:
                    out["hold"].append(float(p.group(1)) - float(p.group(2)))
                    out["polls"].append(int(m.group(1)))
            m = PROMOTE.search(ln)
            if m:
                pk += 1
                if pk >= 2:
                    out["promote_in"].append(float(m.group(1)))
    rec = os.path.join(d, "promote", "receipt.json")
    if os.path.exists(rec):
        for r in json.load(open(rec))["runs"]:
            if r.get("arm") == "promote" and r.get("intruder") and "wall_ms" in r["intruder"]:
                out["e2e"].append(r["intruder"]["wall_ms"])
    return out


def pro(d33_ev, d34_ev):
    rc = 0
    runs = {}
    for name, ev in (("d33", d33_ev), ("d34", d34_ev)):
        for d in sorted(glob.glob(os.path.join(ev, "o*", "b*-on"))):
            order = os.path.basename(os.path.dirname(d))
            b = boot(d)
            r = runs.setdefault((name, order), {"hold": [], "e2e": [], "promote_in": [], "receipts": 0,
                                                "receipts_helper": 0, "helper_ms": []})
            for k in ("hold", "e2e", "promote_in", "helper_ms"):
                r[k].extend(b[k])
            r["receipts"] += b["receipts"]
            r["receipts_helper"] += b["receipts_helper"]
    for name in ("d33", "d34"):
        hold = [x for (n, o), v in runs.items() if n == name for x in v["hold"]]
        pin = [x for (n, o), v in runs.items() if n == name for x in v["promote_in"]]
        rec = sum(v["receipts"] for (n, o), v in runs.items() if n == name)
        rech = sum(v["receipts_helper"] for (n, o), v in runs.items() if n == name)
        hms = [x for (n, o), v in runs.items() if n == name for x in v["helper_ms"]]
        line = (f"run={name} steady {stat('landing-poll-hold', hold)} | {stat('promote-in', pin)} | receipts={rec} "
                f"on-helper={rech} {stat('helper-ms', hms)}")
        if name == "d34":
            ok = len(hold) >= 20 and med(hold) <= 1.5 and max(hold, default=99) <= 3.0 and rec > 0 and rech == rec
            print(f"DAY34 PRO C {line} rule N>=20 median<=1.5 max<=3.0 every-receipt-on-helper -> "
                  f"{'PASS' if ok else 'FAIL'}")
            rc |= 0 if ok else 1
        else:
            print(f"DAY34 PRO READING {line}")
    for order in ("o1", "o2"):
        e33 = runs.get(("d33", order), {}).get("e2e", [])
        e34 = runs.get(("d34", order), {}).get("e2e", [])
        ok = bool(e33) and bool(e34) and med(e34) <= med(e33) + 1.0
        print(f"DAY34 PRO D order={order} {stat('ON e2e d33', e33)} {stat('ON e2e d34', e34)} d34-minus-d33 "
              f"{med(e34) - med(e33):+.2f} rule <=+1.0 -> {'PASS' if ok else 'FAIL'}")
        rc |= 0 if ok else 1
    return rc


def main():
    if sys.argv[1] == "--pro":
        sys.exit(pro(sys.argv[2], sys.argv[3]))
    root = sys.argv[1]
    arms = {}
    for d in sorted(glob.glob(os.path.join(root, "ab", "o*", "b*-*"))):
        order = os.path.basename(os.path.dirname(d))
        arm = os.path.basename(d).split("-", 1)[1]
        b = boot(d)
        a = arms.setdefault((order, arm), {k: [] for k in b} | {"boots": 0})
        a["boots"] += 1
        for k, v in b.items():
            if isinstance(v, list):
                a[k].extend(v)
            else:
                a[k] = (a[k] if isinstance(a[k], int) else 0) + v
    rc = 0
    for arm in ("d33", "d34"):
        hold = [x for (o, a), v in arms.items() if a == arm for x in v["hold"]]
        polls = [x for (o, a), v in arms.items() if a == arm for x in v["polls"]]
        pin = [x for (o, a), v in arms.items() if a == arm for x in v["promote_in"]]
        hms = [x for (o, a), v in arms.items() if a == arm for x in v["helper_ms"]]
        rec = sum(v["receipts"] for (o, a), v in arms.items() if a == arm)
        rech = sum(v["receipts_helper"] for (o, a), v in arms.items() if a == arm)
        boots = sum(v["boots"] for (o, a), v in arms.items() if a == arm)
        keys = sorted(set(polls))
        line = (f"arm={arm} boots={boots} steady {stat('landing-poll-hold', hold)} polls {keys} "
                f"(counts {[polls.count(p) for p in keys]}) | {stat('promote-in', pin)} | receipts={rec} "
                f"on-helper={rech} {stat('helper-ms', hms)}")
        if arm == "d34":
            ok = len(hold) >= 20 and med(hold) <= 1.5 and max(hold, default=99) <= 3.0 and rec > 0 and rech == rec
            print(f"DAY34 C {line} rule N>=20 median<=1.5 max<=3.0 every-receipt-on-helper -> {'PASS' if ok else 'FAIL'}")
            rc |= 0 if ok else 1
        else:
            print(f"DAY34 READING {line}")
    for order in ("o1", "o2"):
        e33 = arms.get((order, "d33"), {}).get("e2e", [])
        e34 = arms.get((order, "d34"), {}).get("e2e", [])
        ok = bool(e33) and bool(e34) and med(e34) <= med(e33) + 1.0
        print(f"DAY34 D order={order} {stat('e2e d33', e33)} {stat('e2e d34', e34)} d34-minus-d33 "
              f"{med(e34) - med(e33):+.2f} rule <=+1.0 -> {'PASS' if ok else 'FAIL'}")
        rc |= 0 if ok else 1
    sys.exit(rc)


if __name__ == "__main__":
    main()
