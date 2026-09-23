#!/usr/bin/env python3
"""WP-A day 32 reader (DAY31.md section 2, acceptance B2 to B4 of the H2D half), written before the sitting.

B2  --b2 BEFORE_EV AFTER_EV: the promote's owner segment, the `owner segment X ms` field of the `promote submitted off
    the tick` line, over the ON boots (`o*/b*-on/server.log`) of two runs of the day-26 double-park cell in one hold:
    BEFORE on the pre-H2D binary (the log-only field, t0 to the line), AFTER on the H2D binary (the owner thread's held
    time by segment up to the line). Steady = the second and later promote of each ON boot. Rule, stated before any
    run: AFTER median <= 1.5 ms and max <= 3.0 ms over N >= 20 steady promotes; BEFORE is a reading against its
    prediction (5 to 8 ms steady). Every AFTER submission must carry `f32 spans from the staging fill` (the H2D program
    ran), or the verdict is FAIL.
B3  (printed with --b2, a reading, not a clause): `[prefix-host] promote: .. in X ms` of the same steady promotes,
    before and after; the helper's fill time and the fill-to-submission poll count from the AFTER submission lines;
    `promote published off the tick: .. X ms from submission to completion`, before and after.
B4  --b4 ROOT --spans N --kv A,B: every `contracts door H2D receipt: ticket issuer=..` line under ROOT names N f32 spans
    (`; N f32 spans landed under the ticket and taken back before the retire`) and keeps `items=` in {A, B} with
    `items=` equal to twice the KV plane count (plus the draft pair). The flip-demote note line (`.. host bytes
    differ ..`) is not a receipt. `--exclude NAME` skips every directory of that name (added after the first run over
    the day-32 box root, which walked into `pre/`, the pre-H2D binary's run, whose receipts carry no span term by
    construction; the clause is unchanged).
"""
import argparse
import glob
import os
import re
import statistics
import sys

SUB = re.compile(r"promote submitted off the tick: .*; owner segment ([\d.]+)ms")
SPANS = re.compile(r"f32 spans from the staging fill \(fill seq=\d+, [\d.]+MB filled by the hash helper in ([\d.]+)ms, "
                   r"landed at poll (\d+)\)")
PROMOTE = re.compile(r"\[prefix-host\] promote: \d+ tokens, [\d.]+MB in ([\d.]+)ms")
PUBLISHED = re.compile(r"promote published off the tick: ticket complete after \d+ poll\(s\), ([\d.]+)ms from submission")
RECEIPT = re.compile(r"contracts door H2D receipt: ticket issuer=\d+ seq=\d+ .* items=(\d+) \((\d+) KV planes(, draft)?\)"
                     r".* published retired acknowledged(; (\d+) f32 spans landed under the ticket and taken back before "
                     r"the retire)?")


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def stat(name, xs):
    if not xs:
        return f"{name} N=0"
    return f"{name} N={len(xs)} median={med(xs):.2f} min={min(xs):.2f} max={max(xs):.2f}"


def boots(ev):
    return sorted(glob.glob(os.path.join(ev, "o*", "b*-on", "server.log")))


def read_arm(ev):
    out = {"owner": [], "owner_first": [], "promote": [], "promote_first": [], "published": [], "fill": [],
           "fill_polls": [], "subs": 0, "subs_with_spans": 0, "boots": 0}
    for log in boots(ev):
        out["boots"] += 1
        k = 0
        pk = 0
        for ln in open(log, errors="replace"):
            m = SUB.search(ln)
            if m:
                k += 1
                out["subs"] += 1
                s = SPANS.search(ln)
                if s:
                    out["subs_with_spans"] += 1
                (out["owner"] if k >= 2 else out["owner_first"]).append(float(m.group(1)))
                if s and k >= 2:
                    out["fill"].append(float(s.group(1)))
                    out["fill_polls"].append(int(s.group(2)))
                continue
            m = PUBLISHED.search(ln)
            if m and k >= 2:
                out["published"].append(float(m.group(1)))
            m = PROMOTE.search(ln)
            if m:
                pk += 1
                (out["promote"] if pk >= 2 else out["promote_first"]).append(float(m.group(1)))
    return out


def b2(before_ev, after_ev):
    b, a = read_arm(before_ev), read_arm(after_ev)
    print(f"DAY32 B2 before (pre-H2D binary) steady {stat('owner-segment', b['owner'])} boots_on={b['boots']} "
          f"first-of-boot {stat('owner-segment', b['owner_first'])} (predicted 5 to 8 ms steady: "
          f"{'held' if b['owner'] and 5.0 <= med(b['owner']) <= 8.0 else 'did not hold'})")
    ok = (len(a["owner"]) >= 20 and med(a["owner"]) <= 1.5 and max(a["owner"], default=99) <= 3.0
          and a["subs"] > 0 and a["subs_with_spans"] == a["subs"])
    print(f"DAY32 B2 after (H2D binary) steady {stat('owner-segment', a['owner'])} boots_on={a['boots']} "
          f"submissions={a['subs']} with_spans={a['subs_with_spans']} "
          f"rule N>=20 median<=1.5 max<=3.0 every-submission-with-spans -> {'PASS' if ok else 'FAIL'}")
    print(f"DAY32 B2 READING after first-of-boot {stat('owner-segment', a['owner_first'])}")
    print(f"DAY32 B3 READING promote in-ms steady before {stat('in', b['promote'])}; after {stat('in', a['promote'])}; "
          f"after-minus-before median {med(a['promote']) - med(b['promote']):+.2f}")
    print(f"DAY32 B3 READING first promote of a boot before {stat('in', b['promote_first'])}; "
          f"after {stat('in', a['promote_first'])}")
    print(f"DAY32 B3 READING submission-to-completion steady before {stat('ms', b['published'])}; "
          f"after {stat('ms', a['published'])}")
    print(f"DAY32 B3 READING helper fill steady {stat('ms', a['fill'])}; fill landed at poll "
          f"{sorted(set(a['fill_polls']))} (counts {[a['fill_polls'].count(p) for p in sorted(set(a['fill_polls']))]})")
    return 0 if ok else 1


def b4(root, spans, kv, exclude=()):
    n = bad = 0
    tally = {}
    for log in sorted(glob.glob(os.path.join(root, "**", "*.log"), recursive=True)):
        if any(part in exclude for part in os.path.relpath(log, root).split(os.sep)[:-1]):
            continue
        for ln in open(log, errors="replace"):
            m = RECEIPT.search(ln)
            if not m:
                continue
            n += 1
            items, planes, draft, _, s = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
            items, planes = int(items), int(planes)
            want_items = 2 * planes + (2 if draft else 0)
            got = int(s) if s else 0
            key = (items, got)
            tally[key] = tally.get(key, 0) + 1
            if items not in kv or items != want_items or got != spans:
                bad += 1
                print(f"DAY32 B4 BAD {log}: items={items} planes={planes} draft={bool(draft)} spans={got}")
    ok = n > 0 and bad == 0
    print(f"DAY32 B4 receipts={n} bad={bad} by (items, spans)={sorted(tally.items())} rule spans=={spans}, items in "
          f"{sorted(kv)} and == 2 x planes (+2 with the draft) -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b2", nargs=2, metavar=("BEFORE_EV", "AFTER_EV"))
    ap.add_argument("--b4")
    ap.add_argument("--spans", type=int, default=96)
    ap.add_argument("--kv", default="32,34")
    ap.add_argument("--exclude", action="append", default=[])
    a = ap.parse_args()
    rc = 0
    if a.b2:
        rc |= b2(*a.b2)
    if a.b4:
        rc |= b4(a.b4, a.spans, {int(x) for x in a.kv.split(",")}, tuple(a.exclude))
    sys.exit(rc)


if __name__ == "__main__":
    main()
