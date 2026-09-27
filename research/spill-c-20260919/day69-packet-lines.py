#!/usr/bin/env python3
"""Day 69 check (DAY69.md section 1, registered before the packet edit): every line the day-69 packet update quotes is
present in the file named beside it. A `.log` or `.txt` receipt: a whole line (a line's text after a timestamp tab
counts as the line). Lane A's own `.md` record: a substring with the file's line breaks read as spaces. A count: the
number of lines of the file that contain the pattern. Paths are relative to research/.

usage: day69-packet-lines.py [research-dir]
"""
import re
import sys
from pathlib import Path

A = "spill-a-20260919"
LINES = [
    # A day 42 / 46 / 48, item 4 on BOX10: S2 and S3 fail their price clause, S4 passes.
    (f"{A}/pro-single-s2/box-design-s2/demote/reading-day42-demote.log",
     "DAY42 S2 C order=o1 wall g4=101.45 s2=104.60 s2-minus-g4=+3.15 rule <=+8.0 | e2e g4=176.08 s2=179.25 s2-minus-g4=+3.16 rule <=+1.0 -> FAIL"),
    (f"{A}/pro-single-s2/box-design-s2/demote/reading-day42-demote.log", "DAY42 S2 DEMOTE -> FAIL"),
    (f"{A}/pro-single-s2/box-design-s3/demote/reading-day42-demote.log",
     "DAY42 S2 C order=o1 wall g4=101.40 s2=104.70 s2-minus-g4=+3.30 rule <=+8.0 | e2e g4=176.02 s2=179.36 s2-minus-g4=+3.34 rule <=+1.0 -> FAIL"),
    (f"{A}/pro-single-s2/box-design-s4/demote/reading-day42-demote.log",
     "DAY42 S2 C order=o1 wall g4=101.40 s2=101.80 s2-minus-g4=+0.40 rule <=+8.0 | e2e g4=176.06 s2=176.46 s2-minus-g4=+0.40 rule <=+1.0 -> PASS"),
    (f"{A}/pro-single-s2/box-design-s4/demote/reading-day42-demote.log",
     "DAY42 S2 C order=o2 wall g4=101.50 s2=101.80 s2-minus-g4=+0.30 rule <=+8.0 | e2e g4=176.21 s2=176.58 s2-minus-g4=+0.37 rule <=+1.0 -> PASS"),
    (f"{A}/pro-single-s2/box-design-s4/demote/reading-day42-demote.log", "DAY42 S2 DEMOTE -> PASS"),
    (f"{A}/pro-single-s2/box-design-s4/promote/reading-day42-promote.log",
     "DAY42 S2 D order=o1 pin g4=13.50 s2=13.60 s2-minus-g4=+0.10 rule <=+1.0 | e2e g4=102.23 s2=102.53 s2-minus-g4=+0.29 rule <=+1.0 -> PASS"),
    (f"{A}/pro-single-s2/box-design-s4/promote/reading-day42-promote.log", "DAY42 S2 PROMOTE -> PASS"),
    (f"{A}/pro-single-s2/box-design-s4/hump/reading-hump.log", "HUMP arm=xs2 boots=2 median-hump=+0.012 humps=False"),
    (f"{A}/pro-single-s2/box-design-s4/hump/reading-hump.log", "HUMP arm=xgpp boots=2 median-hump=+0.507 humps=True"),
    # A day 47, design V, and the pause gate.
    (f"{A}/pro-single-v/box/pause/reading-day47.log",
     "DAY47 V C order=o1 stall base=202.88 v=3.17 rule v<=base/4=50.72 tenant_texts=1 -> PASS | D released=[20] -> PASS"),
    (f"{A}/pro-single-v/box/pause/reading-day47.log",
     "DAY47 V C order=o2 stall base=202.90 v=3.21 rule v<=base/4=50.72 tenant_texts=1 -> PASS | D released=[20] -> PASS"),
    (f"{A}/pro-single-v/box/pause/reading-day47.log", "DAY47 V -> PASS"),
    (f"{A}/pro-single-v/box/gates/pause-demote-rerun.log", "KV-HOST-PAUSE-DEMOTE GATE: ALL GREEN"),
    (f"{A}/pro-single-v/box/gates/contract-fault.log", "KV-HOST-CONTRACT-FAULT GATE: ALL GREEN"),
    (f"{A}/pro-single-v/box/gates/contract-fault-plain.log", "KV-HOST-CONTRACT-FAULT GATE: ALL GREEN"),
    (f"{A}/pro-single-v/box/gates/identity-default-on.log", "KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)"),
    # Item 15.
    (f"{A}/pro-single-i15/box/reading-item15-corrected.log", "ITEM15 (4) hump xg3 median=+0.022 rule <=0.15 -> HOLDS"),
    (f"{A}/pro-single-i15/box/reading-item15-corrected.log", "ITEM15 -> G4 STAYS the single placement"),
    # Item 3 on the 9950X3D2 host.
    (f"{A}/pro-single-t9950/box/item3/reading-day39-target.log",
     "DAY39 T CLAUSE (b) order=o2 metric=pin hk=25.70 ft=13.50 hk-minus-ft=+12.20 pair-noise=0.10 (hk range 0.10, ft range 0.10) rule hk-minus-ft>pair-noise -> CLEARS"),
    (f"{A}/pro-single-t9950/box/item3/reading-day39-target.log", "DAY39 T TARGET (a) and (b) -> PASS (clause (c) is the gates and the unit cells)"),
    (f"{A}/pro-single-t9950/box/host-shape.txt", "Model name:                              AMD Ryzen 9 9950X3D2 16-Core Processor"),
]
# Lines that carry a prefix before the part quoted: checked as substrings of one line of the file.
PREFIXED = [
    (f"{A}/pro-single-i15/box/reading-item15-corrected.log",
     "ITEM15 order=o1 (1) chain g4-g3=-0.11 rule >=+2.0 | (2) copy g4-g3=+5.40 rule >=+2.0"),
    (f"{A}/pro-single-i15/box/reading-item15-corrected.log",
     "ITEM15 order=o2 (1) chain g4-g3=-0.12 rule >=+2.0 | (2) copy g4-g3=+5.40 rule >=+2.0"),
]
MD = [
    ("spill-lead-20260919/INTEGRATION-DAY12.md", "Item 4 is S4 on the target card; S2 and S3 are refuted and reverted. Item 6 is V. Items 3 and 15 close."),
    ("spill-lead-20260919/INTEGRATION-DAY12.md",
     "The deferred admission reclaim flush (moving it off the tick means deferring an arrival until the reclaim lands) sits inside `MEMRA_ADMIT_BY_MEMORY`, so it is lane B's owed item, with V's receipts as its source."),
    ("spill-lead-20260919/INTEGRATION-DAY12.md", "Owed by A: the 5090 halves of S4 and V and item 16 (the card needs its reset), then items 7 to 14."),
    ("spill-lead-20260919/INTEGRATION-DAY12.md", "The capture settle now holds 0.16 ms on both arms."),
    ("spill-lead-20260919/INTEGRATION-DAY12.md", "The receipt kernel costs 104.9 ms per 32 x 4 MiB on this card."),
]
COUNTS = [
    (f"{A}/pro-single-v/box/gates/contract-fault.log", "ok:", 255),
    (f"{A}/pro-single-v/box/gates/contract-fault-plain.log", "ok:", 255),
    (f"{A}/pro-single-v/box/gates/pause-demote-rerun.log", "ok:", 40),
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
    for rel, text in PREFIXED:
        path = root / rel
        if not (path.is_file() and any(text in raw for raw in path.read_text(errors="replace").splitlines())):
            missing.append(f"prefix in {rel}: {text[:90]}")
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
    checked = len(LINES) + len(PREFIXED) + len(MD) + len(COUNTS)
    for m in missing:
        print("MISSING", m)
    print(f"DAY69 PACKET LINES checked={checked} missing={len(missing)} -> {'PASS' if not missing else 'FAIL'}")
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
