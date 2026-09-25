#!/usr/bin/env python3
"""Day 62 check (DAY62.md section 1, registered before the packet edit): every line the day-62 packet update quotes is
present in the file named beside it. A `.log` or `.txt` receipt: a whole line (a line's text after a timestamp tab
counts as the line). Lane A's own `.md` record: a substring with the file's line breaks read as spaces. A count: the
number of lines of the file that contain the pattern. Paths are relative to research/.

usage: day62-packet-lines.py [research-dir]
"""
import re
import sys
from pathlib import Path

A = "spill-a-20260919"
G4 = f"{A}/pro-single-g4/box"
LINES = [
    # A day 37, finding 5: the target card's all arm (BOX7, in the day-38 sitting's unit cells).
    (f"{A}/pro-single-day38/box/unit/all-arm/run.log", "DAY37 FINDING5 TARGET all-arm green=20 of 20 rule 20 of 20 -> PASS"),
    # A day 38, G4 on BOX7.
    (f"{G4}/g/reading-day38-target.log",
     "DAY38 READING arm=base steady copy-settle N=80 median=1.56 min=1.48 max=1.64 | take-back N=80 median=0.21 min=0.21 max=0.23 | owner-held N=80 median=4.05 min=3.81 max=4.28 | helper-hash N=80 median=160.30 min=158.30 max=164.70 | modes [('tick-top poll', 80)] | landed=90 copy-stream-receipt-lines=0 receipt-kernel-ms N=0"),
    (f"{G4}/g/reading-day38-target.log",
     "DAY38 READING arm=g steady copy-settle N=80 median=0.53 min=0.44 max=0.61 | take-back N=80 median=0.21 min=0.19 max=0.22 | owner-held N=80 median=2.97 min=2.75 max=3.25 | helper-hash N=80 median=160.00 min=158.90 max=165.70 | modes [('tick-top poll', 80)] | landed=90 copy-stream-receipt-lines=90 receipt-kernel-ms N=90 median=3.40 min=2.49 max=3.45"),
    (f"{G4}/g/reading-day38-target.log",
     "DAY38 G C copy-settle N=80 median=0.53 min=0.44 max=0.61 rule N>=20 median<=1.5 max<=3.0 -> PASS"),
    (f"{G4}/g/reading-day38-target.log",
     "DAY38 G D order=o1 wall base=182.65 g=181.40 g-minus-base=-1.25 rule <=+5.0 | e2e base=207.55 g=206.37 g-minus-base=-1.17 rule <=+1.0 -> PASS"),
    (f"{G4}/g/reading-day38-target.log",
     "DAY38 G D order=o2 wall base=182.70 g=181.50 g-minus-base=-1.20 rule <=+5.0 | e2e base=207.52 g=206.42 g-minus-base=-1.10 rule <=+1.0 -> PASS"),
    (f"{G4}/hump/reading-hump.log", "HUMP arm=xg4 boots=2 median-hump=+0.035 humps=False"),
    (f"{G4}/hump/reading-hump.log", "HUMP arm=xgpp boots=2 median-hump=+0.576 humps=True"),
    (f"{G4}/gates/identity-default-on.log", "KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)"),
    (f"{G4}/gates/contract-fault.log", "KV-HOST-CONTRACT-FAULT GATE: ALL GREEN"),
    (f"{G4}/gates/contract-fault-plain.log", "KV-HOST-CONTRACT-FAULT GATE: ALL GREEN"),
    (f"{G4}/gates/hitgate-on.log", "SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)"),
    (f"{G4}/gates/failure-on.log", "KV-HOST-SPILL FAILURE GATE: ALL GREEN"),
    # A day 38, G'' on BOX7 (the failed form), first run and re-run.
    (f"{A}/pro-single-day38/box/g/reading-day38-target.log",
     "DAY38 G D order=o1 wall base=182.80 g=185.40 g-minus-base=+2.60 rule <=+5.0 | e2e base=207.66 g=209.08 g-minus-base=+1.42 rule <=+1.0 -> FAIL"),
    (f"{A}/pro-single-day38/box/g/reading-day38-target.log",
     "DAY38 G D order=o2 wall base=182.70 g=185.30 g-minus-base=+2.60 rule <=+5.0 | e2e base=207.64 g=209.01 g-minus-base=+1.36 rule <=+1.0 -> FAIL"),
    (f"{A}/pro-single-day38/box/g-rerun/reading-day38-target.log",
     "DAY38 G D order=o1 wall base=182.70 g=185.25 g-minus-base=+2.55 rule <=+5.0 | e2e base=207.61 g=209.04 g-minus-base=+1.43 rule <=+1.0 -> FAIL"),
    (f"{A}/pro-single-day38/box/g-rerun/reading-day38-target.log",
     "DAY38 G D order=o2 wall base=182.70 g=185.35 g-minus-base=+2.65 rule <=+5.0 | e2e base=207.59 g=209.07 g-minus-base=+1.48 rule <=+1.0 -> FAIL"),
    # A day 38, G4 on the RTX 5090, and section 20's base-controlled hump cell.
    (f"{A}/rtx5090-day38/g4/reading-day38.log",
     "DAY38 READING arm=base steady copy-settle N=80 median=8.34 min=8.11 max=9.51 | take-back N=80 median=0.07 min=0.06 max=0.08 | owner-held N=80 median=9.59 min=8.64 max=13.58 | helper-hash N=80 median=37.75 min=34.40 max=45.20 | modes [('tick-top poll', 80)] | landed=90 copy-stream-receipt-lines=0 receipt-kernel-ms N=0"),
    (f"{A}/rtx5090-day38/g4/reading-day38.log",
     "DAY38 READING arm=g steady copy-settle N=80 median=0.15 min=0.12 max=0.24 | take-back N=80 median=0.07 min=0.06 max=0.09 | owner-held N=80 median=1.11 min=0.60 max=5.04 | helper-hash N=80 median=36.95 min=34.70 max=39.50 | modes [('tick-top poll', 80)] | landed=90 copy-stream-receipt-lines=90 receipt-kernel-ms N=90 median=3.52 min=2.14 max=3.98"),
    (f"{A}/rtx5090-day38/g4/reading-day38.log",
     "DAY38 G C copy-settle N=80 median=0.15 min=0.12 max=0.24 rule N>=20 median<=1.5 max<=3.0 -> PASS"),
    (f"{A}/rtx5090-day38/g4/reading-day38.log",
     "DAY38 G D order=o1 wall base=60.05 g=51.95 g-minus-base=-8.10 rule <=+5.0 | e2e base=111.43 g=103.96 g-minus-base=-7.47 rule <=+1.0 -> PASS"),
    (f"{A}/rtx5090-day38/g4/reading-day38.log",
     "DAY38 G D order=o2 wall base=60.10 g=53.00 g-minus-base=-7.10 rule <=+5.0 | e2e base=112.31 g=105.19 g-minus-base=-7.12 rule <=+1.0 -> PASS"),
    (f"{A}/rtx5090-day38/g4/hump/reading-hump.log", "HUMP arm=xg4 boots=2 median-hump=+0.299 humps=True"),
    (f"{A}/rtx5090-day38/g4/hump/reading-hump.log", "HUMP arm=xgpp boots=2 median-hump=+0.601 humps=True"),
    (f"{A}/rtx5090-day38/g34/hump/reading-hump.log", "HUMP arm=xbase boots=2 median-hump=+0.072 humps=False"),
    (f"{A}/rtx5090-day38/g34/hump/reading-hump.log", "HUMP arm=xg3 boots=2 median-hump=+0.055 humps=False"),
    (f"{A}/rtx5090-day38/g34/hump/reading-hump.log", "HUMP arm=xg4 boots=2 median-hump=+0.032 humps=False"),
    (f"{A}/rtx5090-day38/g34/hump/reading-hump.log", "HUMP arm=xgpp boots=2 median-hump=+0.334 humps=True"),
    # A day 39, T: BOX7's survey and clauses, the 5090's clause (e).
    (f"{A}/pro-single-day38/box/fill/survey.log",
     "FILL shape=27B bytes=156893184 threads=1 N=5 ms median=10.985 min=10.867 max=11.071 gbps=14.28 bitwise=true"),
    (f"{G4}/item3/reading-day39-target.log", "DAY39 T CLAUSE (a) ft steady promotes N=90 polls==1 90 rule N=90 and >=80 -> PASS"),
    (f"{G4}/item3/reading-day39-target.log",
     "DAY39 T CLAUSE (b) order=o1 metric=e2e hk=131.24 ft=118.61 hk-minus-ft=+12.64 pair-noise=0.20 (hk range 0.16, ft range 0.20) rule hk-minus-ft>pair-noise -> CLEARS"),
    (f"{G4}/item3/reading-day39-target.log",
     "DAY39 T CLAUSE (b) order=o1 metric=pin hk=27.40 ft=14.70 hk-minus-ft=+12.70 pair-noise=0.10 (hk range 0.00, ft range 0.10) rule hk-minus-ft>pair-noise -> CLEARS"),
    (f"{G4}/item3/reading-day39-target.log",
     "DAY39 T CLAUSE (b) order=o2 metric=e2e hk=131.15 ft=118.59 hk-minus-ft=+12.55 pair-noise=0.23 (hk range 0.23, ft range 0.17) rule hk-minus-ft>pair-noise -> CLEARS"),
    (f"{G4}/item3/reading-day39-target.log", "DAY39 1b READING order=o1 arm=ft on=118.6 off=120.2 on_minus_off=-1.6 rule <=+20.0 -> PASS"),
    (f"{G4}/item3/reading-day39-target.log", "DAY39 1b READING order=o2 arm=ft on=118.6 off=120.3 on_minus_off=-1.7 rule <=+20.0 -> PASS"),
    (f"{G4}/item3/reading-day39-target.log", "DAY39 T TARGET (a) and (b) -> PASS (clause (c) is the gates and the unit cells)"),
    (f"{A}/rtx5090-day39/t/reading-day39-5090.log",
     "DAY39 T CLAUSE (e) order=o1 metric=e2e ft=70.47 f1=70.47 ft-minus-f1=+0.00 pair-noise=2.27 (ft range 2.27, f1 range 0.39) rule ft-minus-f1<=pair-noise -> PASS"),
    (f"{A}/rtx5090-day39/t/reading-day39-5090.log", "DAY39 T 5090 (e) -> PASS (the gates and the unit cells are read from their logs)"),
    # A day 40, S on the RTX 5090, refuted; its target sitting cancelled.
    (f"{A}/rtx5090-day40/s/demote/reading-day40-demote.log",
     "DAY40 S C order=o1 wall g4=51.70 s=91.95 s-minus-g4=+40.25 rule <=+8.0 | e2e g4=103.27 s=104.75 s-minus-g4=+1.47 rule <=+1.0 -> FAIL"),
    (f"{A}/rtx5090-day40/s/demote/reading-day40-demote.log",
     "DAY40 S C order=o2 wall g4=52.20 s=91.50 s-minus-g4=+39.30 rule <=+8.0 | e2e g4=104.47 s=105.63 s-minus-g4=+1.16 rule <=+1.0 -> FAIL"),
    (f"{A}/pro-single-s/box/CANCELLED.txt", "cancelled 2026-09-25: S failed its 5090 price clauses (DAY40 section 5); the S sitting does not run"),
    # A day 41, the Sources fault cells on the 5090.
    (f"{A}/rtx5090-day41/fault-default/gate.log", "KV-HOST-CONTRACT-FAULT GATE: ALL GREEN"),
    (f"{A}/rtx5090-day41/fault-plain/gate.log", "KV-HOST-CONTRACT-FAULT GATE: ALL GREEN"),
]
MD = [
    (f"{A}/DAY37.md",
     "DAY37 FINDING5 cause=cuMemFreeHost/cuMemFree/module-load hold every other owner thread of one context (same context only; across contexts only context create/destroy) fix=one pool context per native cell, created before any cell body, never destroyed in-process pair=100/100 all=100/100 serial=3/3 red-arm=FAIL as required -> 5090 PASS; target card all-arm 20/20 owed"),
    (f"{A}/DAY38.md", "the hump cell at 87 to 88 C with SM clocks 1995 falling to 1830 to 1970 MHz at 150 to 164 W"),
    (f"{A}/DAY40.md", "**the 48 source digests 0.61 ms** (0.47 busy)"),
    (f"{A}/DAY40.md", "**the 48 landed digests 2.49 ms** (2.26 busy"),
    (f"{A}/DAY40.md", "**S adds about 3.1 ms of copy-stream time on this card**"),
    (f"{A}/DAY41.md",
     "`DAY41 K-ARMS (5090) default ALL GREEN, plain ALL GREEN, the three arms each latch typed with no publication -> 5090 PASS; target card owed`"),
    ("spill-lead-20260919/INTEGRATION-DAY12.md",
     "G4 is the door's form of hash 1 off the owner thread. It passes (a) to (f) on the target card. On the 5090, (f) FAILS as registered, and that FAIL stands in the record."),
    ("spill-lead-20260919/INTEGRATION-DAY12.md", "T is the door's fill program. The 9950X-class reading is owed."),
    ("spill-lead-20260919/INTEGRATION-DAY12.md", "S is refuted and reverted; item 4 stays open under DAY40 section 7's revision."),
    ("spill-lead-20260919/INTEGRATION-DAY12.md",
     "Owed: item 4, items 6 to 15, the 9950X-class fill reading, and the 5090 hump replicate."),
    ("spill-lead-20260919/INTEGRATION-DAY12.md",
     "G''' against G4 at long entries: item 15's 4096-token cell carries both arms on the next target card."),
]
COUNTS = [
    (f"{G4}/gates/contract-fault.log", "ok:", 229),
    (f"{G4}/gates/contract-fault-plain.log", "ok:", 229),
    (f"{G4}/gates/contract-fault.log", "sources-", 45),
    (f"{A}/rtx5090-day41/fault-default/gate.log", "ok:", 229),
    (f"{A}/rtx5090-day41/fault-plain/gate.log", "ok:", 229),
    (f"{A}/rtx5090-day38/g4/fault-default/gate.log", "ok:", 229),
    (f"{G4}/gates/identity-default-on.log", "ok:", 12),
    (f"{G4}/gates/hitgate-on.log", "ok:", 68),
]


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
    missing = []
    for rel, line in LINES:
        path = root / rel
        found = path.is_file() and any(
            raw.strip() == line or raw.partition("\t")[2].strip() == line
            for raw in path.read_text(errors="replace").splitlines())
        if not found:
            missing.append(f"line in {rel}: {line[:90]}")
    for rel, text in MD:
        path = root / rel
        flat = re.sub(r"\s+", " ", path.read_text(errors="replace")) if path.is_file() else ""
        if text not in flat:
            missing.append(f"text in {rel}: {text[:90]}")
    for rel, pattern, n in COUNTS:
        path = root / rel
        got = sum(pattern in raw for raw in path.read_text(errors="replace").splitlines()) if path.is_file() else -1
        if got != n:
            missing.append(f"count in {rel}: {pattern!r} {got} (expected {n})")
    checked = len(LINES) + len(MD) + len(COUNTS)
    for m in missing:
        print("MISSING", m)
    print(f"DAY62 PACKET LINES checked={checked} missing={len(missing)} -> {'PASS' if not missing else 'FAIL'}")
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
