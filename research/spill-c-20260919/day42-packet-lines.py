#!/usr/bin/env python3
"""Day 42 check (DAY42.md section 1, registered before the packet edit): every line the day-42 packet update quotes
is present verbatim in the receipt file named beside it. Paths are relative to research/.

usage: day42-packet-lines.py [research-dir]
"""
import sys
from pathlib import Path

A = "spill-a-20260919"
LINES = [
    # A day 31, target card (BOX3), the double-park cell on the day-31 tree.
    (f"{A}/pro-single-day31/box/reading-day30-doublepark.log",
     "DAY30 A2 pre-submit steady N=80 median=0.63 min=0.60 max=0.66 boots_on=10 demotes_per_boot=[11] rule N>=80 median<=1.5 max<=3.0 -> PASS"),
    (f"{A}/pro-single-day31/box/reading-day30-doublepark.log",
     "DAY30 READING pre-submit demote 1 of its boot N=10 median=29.71 min=29.50 max=29.89"),
    (f"{A}/pro-single-day31/box/reading-day30-doublepark.log",
     "DAY30 READING helper hashed_in_ms N=110 median=79.8 min=79.4 max=108.8; copy submission-to-completion N=110 median=92.6"),
    (f"{A}/pro-single-day31/box/reading-day28.log",
     "DAY28 CLAUSE 1b e2e order=o1 N_runs_on=50 N_runs_off=50 on=127.4 off=115.6 on_minus_off=+11.8 rule <=+20.0 -> PASS"),
    (f"{A}/pro-single-day31/box/reading-day28.log",
     "DAY28 CLAUSE 1c owner in-completion N=100 median=2.02 min=1.96 max=2.66 runs_with_demote_without_ledger=0 rule <=12.0 -> PASS"),
    (f"{A}/pro-single-day31/box/reading-day28.log", "DAY28 VERDICT clauses_failed=0 -> ALL PASS"),
    (f"{A}/pro-single-day31/box/reading-day25.log",
     "DAY25 DOUBLE-PARK stall order=o1 on_minus_off=-8.4 unc=0.1 -> isolated (on 76.9, off 85.4)"),
    (f"{A}/pro-single-day31/box/reading-day25.log",
     "DAY25 DOUBLE-PARK stall order=o2 on_minus_off=-8.5 unc=0.2 -> isolated (on 76.9, off 85.4)"),
    # A day 32, target card (BOX3), the pre-H2D and H2D binaries in one hold.
    (f"{A}/pro-single-day32/box/reading-day32-b2.log",
     "DAY32 B2 after (H2D binary) steady owner-segment N=90 median=0.70 min=0.65 max=1.22 boots_on=10 submissions=100 with_spans=100 rule N>=20 median<=1.5 max<=3.0 every-submission-with-spans -> PASS"),
    (f"{A}/pro-single-day32/box/reading-day32-b2.log",
     "DAY32 B3 READING promote in-ms steady before in N=90 median=20.70 min=20.50 max=22.00; after in N=90 median=31.00 min=30.90 max=32.30; after-minus-before median +10.30"),
    (f"{A}/pro-single-day32/box/reading-day32-b2.log",
     "DAY32 B3 READING helper fill steady ms N=90 median=6.40 min=6.20 max=6.50; fill landed at poll [1] (counts [90])"),
    (f"{A}/pro-single-day32/box/reading-day32-b4.log",
     "DAY32 B4 receipts=109 bad=0 by (items, spans)=[((32, 96), 101), ((34, 96), 8)] rule spans==96, items in [32, 34] and == 2 x planes (+2 with the draft) -> PASS"),
    (f"{A}/pro-single-day32/box/reading-day28.log",
     "DAY28 CLAUSE 1b e2e order=o1 N_runs_on=50 N_runs_off=50 on=137.8 off=115.3 on_minus_off=+22.5 rule <=+20.0 -> FAIL"),
    (f"{A}/pro-single-day32/box/reading-day28.log",
     "DAY28 CLAUSE 1b e2e order=o2 N_runs_on=50 N_runs_off=50 on=137.7 off=115.5 on_minus_off=+22.2 rule <=+20.0 -> FAIL"),
    (f"{A}/pro-single-day32/box/reading-day28.log", "DAY28 VERDICT clauses_failed=2 -> FAIL"),
    # A day 34, target card (BOX4), the d32, d33 and d34 binaries in one hold.
    (f"{A}/pro-single-day34/box/reading-day34-pro.log",
     "DAY34 PRO C run=d34 steady landing-poll-hold N=90 median=0.35 min=0.33 max=0.37 | promote-in N=90 median=27.40 min=27.30 max=28.90 | receipts=100 on-helper=100 helper-ms N=100 median=1.40 min=1.10 max=1.70 rule N>=20 median<=1.5 max<=3.0 every-receipt-on-helper -> PASS"),
    (f"{A}/pro-single-day34/box/reading-day34-pro.log",
     "DAY34 PRO D order=o1 ON e2e d33 N=50 median=133.42 min=132.76 max=135.81 ON e2e d34 N=50 median=132.36 min=131.79 max=134.66 d34-minus-d33 -1.07 rule <=+1.0 -> PASS"),
    (f"{A}/pro-single-day34/box/reading-day34-pro.log",
     "DAY34 PRO D order=o2 ON e2e d33 N=50 median=133.37 min=132.65 max=135.67 ON e2e d34 N=50 median=132.31 min=131.71 max=134.75 d34-minus-d33 -1.05 rule <=+1.0 -> PASS"),
    (f"{A}/pro-single-day34/box/reading-day28.log",
     "DAY28 CLAUSE 1b e2e order=o1 N_runs_on=50 N_runs_off=50 on=132.4 off=120.3 on_minus_off=+12.0 rule <=+20.0 -> PASS"),
    (f"{A}/pro-single-day34/box/reading-day28.log",
     "DAY28 CLAUSE 1a stall order=o1 N_boots_on=5 N_boots_off=5 on_cell_median=76.8 off_cell_median=93.1 rule on<=off+2.0 -> PASS"),
    # A day 36, target card (BOX5): the D2D restore price cell and M'.
    (f"{A}/pro-single-day36/box/reading-day36-target.log",
     "DAY36 PRICE VERDICT (target card) owner-stream median=0.290 host median=0.190 rule each < 0.5 ms per restore -> CLOSES"),
    (f"{A}/pro-single-day36/box/reading-day35m2-target.log",
     "DAY35 M2 C take-back N=80 median=0.11 min=0.10 max=0.11 rule N>=20 median<=1.5 max<=3.0 (copy-settle reading: copy-settle N=80 median=0.70 min=0.64 max=0.82) -> PASS"),
    (f"{A}/pro-single-day36/box/reading-day35m2-target.log",
     "DAY35 M2 D order=o1 wall base=114.70 m=114.30 m-minus-base=-0.40 rule <=+17.0 | e2e base=176.48 m=176.57 m-minus-base=+0.09 rule <=+1.0 -> PASS"),
    (f"{A}/pro-single-day36/box/reading-day35m2-target.log",
     "DAY35 M2 D order=o2 wall base=114.70 m=114.20 m-minus-base=-0.50 rule <=+17.0 | e2e base=176.52 m=176.54 m-minus-base=+0.01 rule <=+1.0 -> PASS"),
    (f"{A}/pro-single-day36/box/reading-day35m2-target.log",
     "DAY35 M READING arm=m steady copy-settle N=80 median=0.70 min=0.64 max=0.82 | take-back N=80 median=0.11 min=0.10 max=0.11 | owner-held N=80 median=2.27 min=2.12 max=2.86 | helper-hash N=80 median=97.60 min=96.90 max=98.70 | modes [('tick-top poll', 80)] | landed=90 receipts-lines=0 lease-lines=90 receipts-ms N=0"),
    (f"{A}/pro-single-day36/box/reading-day35m2-target.log",
     "DAY35 M READING arm=base steady copy-settle N=80 median=0.70 min=0.63 max=0.83 | take-back N=80 median=0.54 min=0.53 max=0.57 | owner-held N=80 median=2.67 min=2.54 max=2.92 | helper-hash N=80 median=97.00 min=96.60 max=98.70 | modes [('tick-top poll', 80)] | landed=90 receipts-lines=0 lease-lines=0 receipts-ms N=0"),
    # RTX 5090, A days 33 to 36.
    (f"{A}/rtx5090-day34/reading-day34.log",
     "DAY34 D order=o1 e2e d33 N=50 median=80.49 min=76.65 max=87.27 e2e d34 N=50 median=80.70 min=76.48 max=102.32 d34-minus-d33 +0.21 rule <=+1.0 -> PASS"),
    (f"{A}/rtx5090-day34/reading-day34.log",
     "DAY34 D order=o2 e2e d33 N=50 median=81.19 min=76.61 max=86.76 e2e d34 N=50 median=81.70 min=76.43 max=87.03 d34-minus-d33 +0.50 rule <=+1.0 -> PASS"),
    (f"{A}/rtx5090-day35/reading-day35.log",
     "DAY35 F CLAUSE order=o1 metric=e2e hk=86.75 fk=78.95 hk-minus-fk=+7.81 pair-noise=5.57 (hk range 5.57, fk range 3.32) rule hk-minus-fk>pair-noise -> CLEARS"),
    (f"{A}/rtx5090-day35/reading-day35.log",
     "DAY35 F CLAUSE order=o2 metric=e2e hk=88.26 fk=80.82 hk-minus-fk=+7.44 pair-noise=5.03 (hk range 5.03, fk range 3.84) rule hk-minus-fk>pair-noise -> CLEARS"),
    (f"{A}/rtx5090-day35/reading-day35.log", "DAY35 F DECISION -> KEEP"),
    (f"{A}/rtx5090-day35/m2/reading-day35m2.log",
     "DAY35 M2 C take-back N=80 median=0.06 min=0.06 max=0.09 rule N>=20 median<=1.5 max<=3.0 (copy-settle reading: copy-settle N=80 median=8.33 min=8.15 max=9.08) -> PASS"),
    (f"{A}/rtx5090-day35/m2/reading-day35m2.log",
     "DAY35 M2 D order=o1 wall base=67.75 m=59.60 m-minus-base=-8.15 rule <=+17.0 | e2e base=119.20 m=111.52 m-minus-base=-7.68 rule <=+1.0 -> PASS"),
    (f"{A}/rtx5090-day35/m2/reading-day35m2.log",
     "DAY35 M2 D order=o2 wall base=67.95 m=59.55 m-minus-base=-8.40 rule <=+17.0 | e2e base=119.85 m=111.81 m-minus-base=-8.04 rule <=+1.0 -> PASS"),
    (f"{A}/rtx5090-day36/reading-day36.log",
     "DAY36 PRICE READING (5090, not the rule's card) host median=0.130 owner-stream median=0.170 (the rule's bound 0.5 each, read on the target card only)"),
    # Added with the rows (substrings of longer lines are checked the same way).
    (f"{A}/pro-single-day32/box/reading-day32-b2.log",
     "DAY32 B2 before (pre-H2D binary) steady owner-segment N=90 median=4.58 min=4.46 max=5.11"),
    (f"{A}/pro-single-day31/box/reading-day28.log",
     "DAY28 CLAUSE 1b e2e order=o2 N_runs_on=50 N_runs_off=50 on=127.3 off=115.5 on_minus_off=+11.8 rule <=+20.0 -> PASS"),
    (f"{A}/rtx5090-day33/reading-day33-stall.log",
     "DAY33 5090 READING arm=d32-on boots=2 owner-segment N=20 median=0.41 min=0.38 max=0.51 submissions=20 filled=0"),
    (f"{A}/rtx5090-day33/reading-day33-stall.log", "promote-in N=18 median=16.90 min=16.70 max=19.30 | intruder-e2e N=20 median=78.04"),
    (f"{A}/rtx5090-day33/reading-day33-stall.log",
     "DAY33 5090 READING arm=d33-on boots=2 owner-segment N=20 median=0.41 min=0.37 max=0.67 submissions=20 filled=20"),
    (f"{A}/rtx5090-day33/reading-day33-stall.log", "promote-in N=18 median=17.10 min=16.50 max=19.60 | intruder-e2e N=20 median=78.53"),
    (f"{A}/rtx5090-day33/reading-day33-stall.log", "promote-in N=9 median=10.40 min=9.60 max=26.50 | intruder-e2e N=10 median=62.11"),
    (f"{A}/rtx5090-day34/reading-day34.log", "DAY34 READING arm=d33 boots=10 steady landing-poll-hold N=90 median=8.50"),
    (f"{A}/rtx5090-day34/reading-day34.log", "DAY34 C arm=d34 boots=10 steady landing-poll-hold N=90 median=0.12"),
    (f"{A}/rtx5090-day34/reading-day34.log", "helper-ms N=100 median=8.80"),
    (f"{A}/rtx5090-day35/reading-day35.log", "DAY35 READING order=o1 arm=off boots=5 e2e N=50 median=64.69"),
    (f"{A}/rtx5090-day35/reading-day35.log", "DAY35 READING order=o2 arm=off boots=5 e2e N=50 median=65.92"),
    (f"{A}/rtx5090-day35/m/reading-day35m.log",
     "DAY35 M READING arm=m steady copy-settle N=80 median=0.15 min=0.12 max=0.20 | take-back N=80 median=0.07"),
    (f"{A}/rtx5090-day35/m2/reading-day35m2.log",
     "DAY35 M READING arm=base steady copy-settle N=80 median=8.31 min=8.11 max=8.58 | take-back N=80 median=8.25 min=8.10 max=9.78 | owner-held N=80 median=17.98"),
    (f"{A}/rtx5090-day35/m2/reading-day35m2.log",
     "DAY35 M READING arm=m steady copy-settle N=80 median=8.33 min=8.15 max=9.08 | take-back N=80 median=0.06 min=0.06 max=0.09 | owner-held N=80 median=9.55"),
    (f"{A}/pro-single-day36/box/reading-day35m2-target.log",
     "DAY35 M READING order=o1 arm=base wall N=40 median=114.70 min=114.30 max=115.00 | e2e N=50 median=176.48"),
    (f"{A}/pro-single-day36/box/reading-day35m2-target.log",
     "DAY35 M READING order=o2 arm=base wall N=40 median=114.70 min=114.30 max=115.00 | e2e N=50 median=176.52"),
]


# Section 3's gate row: each verdict line's count over a sitting's gate logs (`*.log` under the directory).
GATES = {
    "KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)": 4,
    "KV-HOST-SPILL FAILURE GATE: ALL GREEN": 2,
    "SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)": 2,
}
FAULT = "KV-HOST-CONTRACT-FAULT GATE: ALL GREEN"
SITTINGS = {f"{A}/pro-single-day31/box/gates": 1, f"{A}/pro-single-day32/box/gates": 1,
            f"{A}/pro-single-day34/box/gates": 1, f"{A}/pro-single-day36/box/gates": 2}


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
    missing = []
    for rel, line in LINES:
        path = root / rel
        text = path.read_text(errors="replace") if path.exists() else ""
        if line not in text:
            missing.append(rel)
            print(f"MISSING in {rel}: {line[:120]}")
    for rel, faults in SITTINGS.items():
        logs = "\n".join(p.read_text(errors="replace") for p in sorted((root / rel).rglob("*.log")))
        for line, want in {**GATES, FAULT: faults}.items():
            got = sum(1 for row in logs.splitlines() if row.strip() == line)
            ok = got == want
            print(f"DAY42 GATES {rel} {line!r} lines={got} expected={want} {'ok' if ok else 'MISMATCH'}")
            if not ok:
                missing.append(rel)
    checked = len(LINES) + len(SITTINGS) * (len(GATES) + 1)
    print(f"DAY42 PACKET LINES checked={checked} missing={len(missing)} -> {'PASS' if not missing else 'FAIL'}")
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
