#!/usr/bin/env python3
"""WP-A day 33 reader (DAY33.md section 2: acceptance (b) and (d), the readings), written before any day-33 run.

(b)  --b2 BEFORE_EV AFTER_EV: the promote's owner segment (the `owner segment X ms` field of the `promote submitted off
     the tick` line) over the ON boots (`o*/b*-on/server.log`) of two runs of the day-26 double-park cell in one hold:
     BEFORE the day-32 H2D binary, AFTER the day-33 binary. Steady = the second and later promote of each ON boot.
     Rule, unchanged from day 32: AFTER median <= 1.5 ms and max <= 3.0 ms over N >= 20 steady promotes, and every
     AFTER submission carries its spans (`f32 spans filled on the copy stream`).
(d)  (printed with --b2): AFTER steady owner segment median <= 0.90 ms and max <= 1.50 ms. The census half of (d) is the
     server's CPU cell `day33_the_promote_submits_its_filled_spans_in_the_probes_tick`.
Readings (not clauses): `promote published off the tick: ticket complete after P poll(s), X ms from submission to
     completion` (the poll count and the time, before and after), `[prefix-host] promote: .. in X ms` (before and after).
--stall ROOT: the 5090 development reading over `ROOT/b*-<arm>/server.log` and each boot's stall receipt.
DAY28 clauses 1a to 1c are read by `day28-reading.py`, unchanged.
"""
import argparse
import glob
import json
import os
import re
import statistics
import sys

SUB = re.compile(r"promote submitted off the tick: .*; owner segment ([\d.]+)ms")
FILLED = "f32 spans filled on the copy stream"
PROMOTE = re.compile(r"\[prefix-host\] promote: \d+ tokens, [\d.]+MB in ([\d.]+)ms")
PUBLISHED = re.compile(r"promote published off the tick: ticket complete after (\d+) poll\(s\), ([\d.]+)ms from submission")


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def stat(name, xs):
    if not xs:
        return f"{name} N=0"
    return f"{name} N={len(xs)} median={med(xs):.2f} min={min(xs):.2f} max={max(xs):.2f}"


def read_logs(logs):
    out = {"owner": [], "owner_first": [], "promote": [], "promote_first": [], "published": [], "polls": [],
           "subs": 0, "subs_filled": 0, "logs": 0}
    for log in logs:
        out["logs"] += 1
        k = pk = qk = 0
        for ln in open(log, errors="replace"):
            m = SUB.search(ln)
            if m:
                k += 1
                out["subs"] += 1
                out["subs_filled"] += FILLED in ln
                (out["owner"] if k >= 2 else out["owner_first"]).append(float(m.group(1)))
                continue
            m = PUBLISHED.search(ln)
            if m:
                qk += 1
                if qk >= 2:
                    out["polls"].append(int(m.group(1)))
                    out["published"].append(float(m.group(2)))
            m = PROMOTE.search(ln)
            if m:
                pk += 1
                (out["promote"] if pk >= 2 else out["promote_first"]).append(float(m.group(1)))
    return out


def polls(xs):
    keys = sorted(set(xs))
    return f"polls {keys} (counts {[xs.count(p) for p in keys]})"


def b2(before_ev, after_ev):
    b = read_logs(sorted(glob.glob(os.path.join(before_ev, "o*", "b*-on", "server.log"))))
    a = read_logs(sorted(glob.glob(os.path.join(after_ev, "o*", "b*-on", "server.log"))))
    print(f"DAY33 B2 before (day-32 H2D binary) steady {stat('owner-segment', b['owner'])} boots_on={b['logs']}")
    ok_b = (len(a["owner"]) >= 20 and med(a["owner"]) <= 1.5 and max(a["owner"], default=99) <= 3.0
            and a["subs"] > 0 and a["subs_filled"] == a["subs"])
    print(f"DAY33 B2 after (day-33 binary) steady {stat('owner-segment', a['owner'])} boots_on={a['logs']} "
          f"submissions={a['subs']} filled={a['subs_filled']} rule N>=20 median<=1.5 max<=3.0 "
          f"every-submission-filled -> {'PASS' if ok_b else 'FAIL'}")
    ok_d = len(a["owner"]) >= 20 and med(a["owner"]) <= 0.90 and max(a["owner"], default=99) <= 1.50
    print(f"DAY33 D after steady {stat('owner-segment', a['owner'])} rule median<=0.90 max<=1.50 "
          f"-> {'PASS' if ok_d else 'FAIL'} (the census half: day33_the_promote_submits_its_filled_spans_in_the_probes_tick)")
    print(f"DAY33 READING submission-to-completion steady before {stat('ms', b['published'])} {polls(b['polls'])}; "
          f"after {stat('ms', a['published'])} {polls(a['polls'])}")
    print(f"DAY33 READING promote in-ms steady before {stat('in', b['promote'])}; after {stat('in', a['promote'])}; "
          f"after-minus-before median {med(a['promote']) - med(b['promote']):+.2f}")
    print(f"DAY33 READING first of a boot: owner before {stat('ms', b['owner_first'])}, after "
          f"{stat('ms', a['owner_first'])}; promote in before {stat('ms', b['promote_first'])}, after "
          f"{stat('ms', a['promote_first'])}")
    return 0 if ok_b and ok_d else 1


def stall(root):
    arms = {}
    for d in sorted(glob.glob(os.path.join(root, "b*-*"))):
        arm = os.path.basename(d).split("-", 1)[1]
        arms.setdefault(arm, []).append(d)
    for arm, dirs in sorted(arms.items()):
        r = read_logs([os.path.join(d, "server.log") for d in dirs if os.path.exists(os.path.join(d, "server.log"))])
        walls, stalls = [], []
        for d in dirs:
            p = os.path.join(d, "promote", "receipt.json")
            if not os.path.exists(p):
                continue
            rec = json.load(open(p))
            for run in rec["runs"]:
                if run.get("arm") == "promote":
                    if run.get("intruder") and "wall_ms" in run["intruder"]:
                        walls.append(run["intruder"]["wall_ms"])
                    if "stall_ms" in run:
                        stalls.append(run["stall_ms"])
        print(f"DAY33 5090 READING arm={arm} boots={len(dirs)} {stat('owner-segment', r['owner'] + r['owner_first'])} "
              f"submissions={r['subs']} filled={r['subs_filled']} | completion {stat('ms', r['published'])} "
              f"{polls(r['polls'])} | {stat('promote-in', r['promote'])} | {stat('intruder-e2e', walls)} | "
              f"{stat('stall', stalls)}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b2", nargs=2, metavar=("BEFORE_EV", "AFTER_EV"))
    ap.add_argument("--stall")
    a = ap.parse_args()
    rc = 0
    if a.b2:
        rc |= b2(*a.b2)
    if a.stall:
        rc |= stall(a.stall)
    sys.exit(rc)


if __name__ == "__main__":
    main()
