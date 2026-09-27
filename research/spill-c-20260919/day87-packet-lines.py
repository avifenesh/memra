#!/usr/bin/env python3
"""Day 87 check (DAY87.md section 1, registered before the packet edit): every line the day-87 packet update quotes is
present in the file named beside it, read through `git show <tree>:research/<path>` (default tree 359e850d0). A `.log`
or `.txt` receipt: a whole line (a line's text after a timestamp tab counts as the line). A leading part of one
receipt line: a substring of one line. An `.md` record: a substring with the file's line breaks read as spaces (and
runs of spaces as one).

usage: day87-packet-lines.py [tree]
"""
import re
import subprocess
import sys

TREE = sys.argv[1] if len(sys.argv) > 1 else "359e850d0"
A = "spill-a-20260919"
L = "spill-lead-20260919/INTEGRATION-DAY12.md"
LINES = [
    (f"{A}/pro-single-day49/box/cell/reading-day49.log",
     "DAY49 ITEM8 pages=38306 nofree_minflt=38306 (rule >= 19153) free_minflt=0 (rule <= 9576) copy_ms nofree=23.73 free=7.64 diff=+16.09 (rule >= +5.0) -> H attributed"),
    (f"{A}/pro-single-day49/box/cell/reading-day49.log",
     "DAY49 ITEM7 -> b1 step attributed to H (the heap first touch) (the current pre-submit split above)"),
    (f"{A}/pro-single-p/box/reading-day51.log",
     "DAY51 P (g) cell=chain order=o1 chain p-minus-base=+1.40 rule <=+1.00 | first p-minus-base=+0.23 rule <=+1.00 -> FAIL"),
    (f"{A}/pro-single-p/box/reading-day51.log", "DAY51 P -> FAIL"),
    (f"{A}/pro-single-p2/box/reading-day52.log",
     "DAY52 P2 (g) cell=chain order=o2 chain p2-minus-base=+1.15 rule <=+1.00 | first p2-minus-base=+1.11 rule <=+1.00 -> FAIL"),
    (f"{A}/pro-single-p2/box/reading-day52.log", "DAY52 P2 -> FAIL"),
    (f"{A}/pro-single-p2l2/box/reading-day52.log",
     "DAY52 P2 (g) cell=chain order=o1 chain p2-minus-base=-0.06 rule <=+1.00 | first p2-minus-base=-0.00 rule <=+1.00 -> PASS"),
    (f"{A}/pro-single-p2l2/box/reading-day52.log",
     "DAY52 P2 (g) cell=chain order=o2 chain p2-minus-base=-0.03 rule <=+1.00 | first p2-minus-base=-0.02 rule <=+1.00 -> PASS"),
    (f"{A}/pro-single-p2l2/box/reading-day52.log",
     "DAY52 P2 (b) cell=demote order=o1 minflt p2-median (0.25 x pages)=+0.00 rule <=+9576.42 | copy p2-minus-base=-15.88 rule <=-8.00 | hits fraction of steady demotes=+1.00 rule >=+0.90 -> PASS"),
    (f"{A}/pro-single-p2l2/box/reading-day52.log",
     "DAY52 P2 (c) cell=demote order=o1 wall p2-minus-base=-12.40 rule <=-8.00 | e2e p2-minus-base=+0.34 rule <=+1.00 -> PASS"),
    (f"{A}/pro-single-p2l2/unit-rerun/box/run.log",
     "unit-cells parallel=3/3 engine-serial-rc=0 door-rc=0 cpu-rc=0 engine-census-rc=0 tier-rc=0"),
    (f"{A}/pro-single-day54/reading-day54-corrected.log",
     "DAY54 PRICE fanout (short) stall fanout-minus-prime o1=+6.50 o2=+6.50 ms rule >1.0 both -> DESIGN NEXT"),
    (f"{A}/pro-single-day54/reading-day54-corrected.log",
     "DAY54 PRICE fanout (long) stall fanout-minus-prime o1=+896.82 o2=+895.60 ms rule >1.0 both -> DESIGN NEXT"),
    (f"{A}/pro-single-day54/reading-day54-corrected.log",
     "DAY54 PRICE pause park snapshot owner N=30 median=0.73 ms (pause stall median 64.00 ms, N=30) rule >1.0 -> CLOSED AS PRICED"),
    (f"{A}/pro-single-day59/box/reading-day59.log", "DAY59 SELECT -> DESIGN B1 (batched copies)"),
    (f"{A}/pro-single-b1/box/reading-b1.log",
     "B1 READING order=o1 N_own=45 own base=1.36 b1=0.25 ms | fanout-minus-prime base=+4.77 b1=+3.74 (gain +1.03) ms | members wall base=95.9 b1=94.9 ms"),
    (f"{A}/pro-single-b1/box/reading-b1.log",
     "B1 READING order=o2 N_own=45 own base=1.34 b1=0.25 ms | fanout-minus-prime base=+4.90 b1=+3.75 (gain +1.15) ms | members wall base=96.0 b1=94.8 ms"),
    (f"{A}/pro-single-b1/box/reading-b1.log", "B1 VERDICT -> ADOPT (B1 is the naked program)"),
    (f"{A}/pro-single-day60/box/reading-day60.log",
     "DAY29 CELL(i) CLAUSE class=demote stall_median(second stream)=63.8 idle_p99_sitting=12.9 -> clause_not_met"),
    (f"{A}/pro-single-day60/box/reading-day60.log",
     "DAY29 CELL(i) CLAUSE class=promote stall_median(second stream)=62.9 idle_p99_sitting=12.9 -> clause_not_met"),
    (f"{A}/pro-single-day60/box/reading-day60.log",
     "DAY29 CELL(i) CLAUSE: NOT MET (demote=False promote=False admissible=True); executed-not-qualified"),
    (f"{A}/pro-single-w/box/reading-w.log", "W (c) FAIL [True, False, True, False]"),
    (f"{A}/pro-single-w/box/reading-w.log", "W VERDICT card=target -> FAIL (a, c)"),
    (f"{A}/rtx5090-w/cell/reading-w.log", "W VERDICT card=5090 -> FAIL (a, d)"),
    (f"{A}/pro-single-r2/box/reading-r2.log", "R2 (c) FAIL [True, False, True, False]"),
    (f"{A}/pro-single-r2/box/reading-r2.log",
     "R2 VERDICT -> REVERT ((a) passed; failed c): recorded as read, reverted in one commit"),
    (f"{A}/pro-single-r1/box/reading-r1.log", "R1 VERDICT -> ADOPT (R1 is the naked program)"),
    (f"{A}/pro-single-l/box/reading-l.log", "L VERDICT -> REFUTED ((a) failed): revert in one commit, red receipts banked"),
    (f"{A}/pro-single-l2/box/reading-l2.log",
     "L READING cell=chain order=o2 twin kv base=9.87 l=0.05 ms (N=95) | long leases base=26.27 l=0.04 ms (N=95, pooled median 32) | stall base=95.05 l=68.36 | e2e base=151.0 l=124.3 | chain base=282.4 l=246.1 ms"),
    (f"{A}/pro-single-l2/box/reading-l2.log",
     "L2 GATE RED ARM rc=1 staging-fill FAILs=2 of 2 marker=True -> caught (as required)"),
    (f"{A}/pro-single-l2/box/reading-l2.log", "L2 VERDICT -> ADOPT (L' is the naked program; the gate change stands)"),
    (f"{A}/pro-single-f/box/reading-f.log", "F READING late pooled base=86/90 f=79/90"),
    (f"{A}/pro-single-f/box/reading-f.log",
     "F VERDICT -> REVERT ((a) passed; failed b, c): recorded as read, reverted in one commit"),
    (f"{A}/pro-single-th/box/reading-th.log", "TH (b) FAIL [False, False]"),
    (f"{A}/pro-single-th/box/reading-th.log",
     "TH VERDICT -> REVERT ((a) passed; failed b): recorded as read, reverted in one commit"),
]
PREFIXED = [
    (f"{A}/pro-single-r1/box/reading-r1.log",
     "R1 READING order=o2 mode=retire-seam-nosource | no-source settles base N=45 median=12.54 r1 N=0 skips r1=45"),
    (f"{A}/pro-single-l/box/reading-l.log", "L (a) gates {'contract-fault-plain': '1', 'contract-fault': '1',"),
]
MD = [
    (L, "Days 49 to 51 are read as registered. Items 7 and 8 are attributed to H; item 9 closes. Item 17's design P is refuted on (g) and reverted; its revision is owed on top of item 14."),
    (L, "Days 52 to 55 are read as registered. P2 is refuted and reverted; item 17 sits on item 14. Item 21 closes as F1. Item 10's fanout publisher is the next design; the pause park closes as priced. Item 22 closes; T-d stays unchanged."),
    (L, "Days 56 to 58 are read as registered. Items 23, 24 and 25 close."),
    (L, "B1 is adopted as the naked prefix snapshot and restore program on both cards (a per-hardware check, both read ADOPT). The fanout publisher's owed items (the DFlash, GLM-5 and latent publishers) stay with A. W (item 12) waits on its 5090 half; R2 (item 13) is refuted and reverted on the target card, R1 selected (not in this merge)."),
    (L, "L' and R1 are the naked program on the target card (items 13, 14 and 19 close); their 5090 halves are owed by A."),
    (L, "P2 is the naked program on the target card (its 5090 half owed by A, as L' and R1)."),
    (L, "since L', a tenant purge's dropped host leases parked in the lease pool with the revoked tenant's q8/q5 KV bytes still in them (`put` does not scrub, `take` does not refill), up to one host budget, until a same-class reuse, the latch or the engine drop."),
    (L, "Fix `4f297e7bd` (DAY69 design P): `LeasePool::drain()` frees every idle backing and releases its pool charge (the pool stays open) and advances an epoch;"),
    (L, "Two latent hazards are owed items, not defects here: the API would let a caller read a fresh pooled lease before its copy lands (no production caller does), and the GLM-5 TP startup arena returns released regions unscrubbed."),
    (L, "L1.2 charged each lease its size class (the next power of two to 1 MiB, then whole MiB), while the host LRU keeps residents at or under one budget of actual bytes."),
    (L, "L' refused a 16-plane short demote at its 12th lease with `Capacity`, and main admits all 16 at 81 of its 128 MiB. It was worse than a lost demote: the contract route maps a lease refusal to `Alloc`, and the caller latches the tier off."),
    (L, "Fix `a57f85897` (DAY70 design Q): a lease is charged its length, main's charge;"),
    (L, "so a lease is refused only where main's ledger refuses it, and a pool reserve that does not fit never errors."),
    (L, "engine cells `18 passed` (the new `day69_a_drained_backing_is_never_the_next_lease` among them); worker span cells `19 passed` (the new `option_b_purge_drains_the_pool_and_zeroes_the_staging_set` among them);"),
    (L, "GPU battery on BOX43 (a Ryzen 9 9950X with one RTX PRO 6000 WS; `integ70-pro-run1/`, 572 receipts mirrored and checked), tree `1d317fcb7`, binary `60f4359d`,"),
    (L, "identity, fault default and plain, hit OFF and ON, admit-mem burst, spec-ctx-edge and the pause gate `ALL GREEN`; `tier-transfer-gate` conformance 13 PASS lines and six byte-exact roundtrips; all seven `kv-tier-gate` fault arms `FAULT-ARM PASS` with the 27B."),
    (f"{A}/DAY60.md", "Candidate, for the owner's decision: a cell that isolates each class from its intruder's prime"),
    (f"{A}/DAY67.md", "Item 17 closes as P2."),
    (f"{A}/DAY61.md", "a 5090 half that fails (a), (b) or (d) reverts W"),
    (f"{A}/DAY68.md", "the owed 5090 halves of R1 and L', registered before they run"),
    (f"{A}/DAY67.md", "F2 (DAY64 section 5, recorded for after this verdict) is now due, as item 18's next design."),
    (f"{A}/DAY67.md", "Run by the lead on the same BOX31 card"),
    (f"{A}/DAY49.md", "BOX10 (one RTX PRO 6000 Blackwell Workstation Edition"),
]


def show(path):
    return subprocess.run(["git", "show", f"{TREE}:research/{path}"], capture_output=True, text=True,
                          errors="replace").stdout


def lines_of(text):
    return [line.partition("\t")[2] if "\t" in line else line for line in text.splitlines()]


def main():
    missing = []
    for path, line in LINES:
        if line not in lines_of(show(path)):
            missing.append(f"{path}: {line[:90]}")
    for path, part in PREFIXED:
        if not any(part in ln for ln in lines_of(show(path))):
            missing.append(f"{path} (part): {part[:90]}")
    for path, text in MD:
        flat = re.sub(r"\s+", " ", show(path))
        if re.sub(r"\s+", " ", text) not in flat:
            missing.append(f"{path} (md): {text[:90]}")
    for m in missing:
        print("MISSING", m)
    n = len(LINES) + len(PREFIXED) + len(MD)
    print(f"DAY87 PACKET LINES tree={TREE} checked={n} missing={len(missing)} -> {'PASS' if not missing else 'FAIL'}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
