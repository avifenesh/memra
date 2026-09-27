# WP-A day 68: the owed 5090 halves of R1 and L', registered before they run

R1 (DAY62 section 10) and L' (DAY63 section 6) were adopted on the target card and go to main through integ69. Each
changes the naked program on every card, so by the per-hardware rule each owes a reading on the local RTX 5090 Laptop
GPU. Both registrations said "the 5090 half follows"; this file fixes how, before any build or cell.

## 1. The shape both halves share

- **Trees.** Each half runs on its target sitting's own pair, so the 5090 reads the program the target read:
  - R1: r1 at `15d7ed351`, base at `15d7ed351` with the crates of `40821db01` (R1's parent).
  - L': l at `21984b527`, base at `21984b527` with the crates of `217ace3fd`, red and redgate at l plus the two
    red-arm patches in that tree (`pro-single-l2/red-arm.patch`, `pro-single-l2/gate-red-arm.patch`).
- **Scripts.** The target sitting's own `build.sh`, `gates.sh`, `ab.sh`, `unit-cells.sh` and `gates-red.sh`, derived
  for this rig by exact text replacements only (`rtx5090-derive.py`, which refuses to write a script if a replacement
  does not match). The replacements are:
  - the receipt root and the tree path become arguments;
  - `/tmp/memra-gpu.lock` becomes `/tmp/memra-5090.lock`, the rig's lock;
  - the box's `nice -n 5 cargo` becomes the rig's CPU cap (`systemd-run --user --scope -q -p CPUQuota=1200% -p
    MemoryMax=20G`), with the half's own target dir;
  - the box's clone, fetch and named-branch checkout become a detached scratch worktree at the tip.
  - No cell, mode, environment, boot count, order or bound changes.
- **Frozen.** Each half's tip is a detached worktree under `/home/avifenesh/spill-a-cells/`, outside `/tmp`. Its
  builds run before the hold, and the cells run the tip tree's own tools, `stall_cell.py` and reader, as the target
  did. The derived scripts are copied beside the receipts and run from that copy, never from this worktree.
- **The hold.** One bounded hold of `/tmp/memra-5090.lock` per half: up to 180 x 120 s behind the other lanes; then
  no compute app and at least 20000 MiB free, up to 15 x 60 s. Another project's process that lands on the card is
  waited out, never touched. The children use the hold's fd 9 (`--external-lock 9`, `tier-lock-proof.py --fd 9`). R1
  runs first, then L', and the lock is released between the two so the other lanes can interleave.
- **Conditions recorded:** each boot's start temperature, SM clock and power (the target scripts' `BOOT.txt`), the host
  load and compute apps at each cell's start, a 5 s host-load log through the hold, and 250 ms telemetry. Each cell
  keeps the collector's timeout from the target driver. None of this lane's builds runs inside either hold.
- **Driver:** `rtx5090-half.sh prepare|card|clean <half>`.
- **The model:** the 27B the target read (`Qwen3.8-27B-NVFP4-Q5K-mtp.gguf`, sha256 `1facf36c..`, the file W's and B1's
  5090 halves read), hashed outside the hold.

## 2. What each half reads, and what it decides

- **R1:** the 11 gates on r1, then the 60-boot paired cell (`ab.sh seam retire-seam-nosource retire-seam prime 448`),
  then `r1-reading.py`: DAY62 section 8's (a) to (d) with their bounds unchanged.
- **L':** the unit cells (green and red), the 11 gates on l, the gate change's red arm on redgate, the chain and demote
  cells (20 boots each), then `l2-reading.py`: DAY63 section 4's clauses with their bounds unchanged.
- **What a result decides.**
  - A verdict of ADOPT closes the half: the design is the naked program on both cards.
  - A failed (a) is a correctness finding, not a per-card choice. It is placed before anything else and reported at
    once, since integ69 carries both designs.
  - A failed timing clause with (a) green makes the design a per-card default question under its own registration,
    keyed on the device, as B1's registration said. It is not a revert on the target, and no bound is relaxed.
- Receipts go to `rtx5090-r1/cell/` and `rtx5090-l2/cell/`, with the executables recorded by hash and kept outside the
  repo until the lane closes.
- **Budget:** 0.2 agent-day. Card time is about 2.5 h for R1 and 2 h for L', plus the queue.

## 3. Item 16 (DAY45) runs as registered, from its registration tree

- DAY45 section 1 registered item 16's cell and its scripts (`rtx5090-day45/build.sh`, `hot-hump-run.sh`) in `7b849a817`,
  to wait for the 5090's reset. The card is back, so the cell runs as registered, with nothing changed.
- **Frozen:** a detached worktree at `7b849a817` under `/home/avifenesh/spill-a-cells/i16/`. Its `build.sh` builds the
  four registered servers (base `80039a8de`, g4 `26676c037`, g3 `9ab5c1265`, gpp `358749c9f`), and `hot-hump-run.sh`
  runs from that tree with that tree's `stall_cell.py` and readers. This worktree's later edits cannot reach it.
- **The model:** the 9B of DAY38 sections 19 and 20 (`Qwen3.5-9B-NVFP4-MTP-GGUF.gguf`, sha256 `52c9cceb..`), which is
  the day-35 5090 environment DAY45 names.
- It queues after the two halves, in its own hold, with DAY45's bounded wait. DAY45's regime check and placing rule
  read it.

## 4. S4's and V's owed 5090 halves (OWED items 4 and 6), registered before they run

- DAY48 and DAY47 left both halves to wait for the card's reset. They run the same way as section 1.
- **S4:** its target sitting's tip `a0f9968e3` (s2 arm), g4 `b4816eda8`, and gpp `358749c9f` (the hump control), with
  `pro-single-s2/`'s own scripts from that tip. The target's `run-all-4.sh` ran exactly these.
- **V:** tree `ccfd26af0`, base `bbd2535b6` (S4's program), with `pro-single-v/`'s scripts from that tree.
  - `ccfd26af0` is V's target tip `a324503df` plus the revised pause gate of DAY47 section 3a, and its crates are
    byte-identical to `a324503df`'s (`git diff --stat a324503df ccfd26af0 -- crates tools` names only
    `tools/kv-host-pause-demote-gate.sh`).
  - So the gates cell reads the day-36 set and the revised pause gate in one pass. That matches the target's accepted
    (a) and (b): the day-36 set from `a324503df`, and the pause gate from its section 3b re-run.
- **Derivation:** `rtx5090-derive.py`, with the same exact replacements as section 1. It adds one replacement: the unit
  cells' `cargo test` runs under the rig's CPU cap (the box ran the prebuilt test binaries through cargo under its hold;
  the build step builds them here the same way). The derived R1 and L' scripts are unchanged by the added halves.
- **The holds follow the target drivers' own split** (`rtx5090-half-sv.sh`). The hit gate takes its own flock, so:
  - hold A runs the cells before it (S4: `ab-demote`, `ab-promote`, `hump-cell`, `gates`; V: `ab-pause`, `gates`);
  - the hold is released for `hitgate.sh`, which takes the rig's lock itself (its own 15 x 120 s retry);
  - hold B runs the cells after it (both: `unit-cell`; S4: `trace-cell`, Nsight Systems being on this host).
  - Each hold uses section 1's bounded wait and idle rule.
- **What they read.** Each target cell script's own reader runs inside it: `day42-reading.py` demote and promote,
  `day38-hump-reading.py`, `day42-trace-reading.py`, `day47-reading.py`. Each gate's `.exit` and the unit cells' line
  complete the reading. The bounds are unchanged: S4's (a) to (e) (DAY48 section 1 over DAY42's), and V's (a) to (d)
  (DAY47 section 1).
- **What a result decides:** as section 2.
- **Order and card time:** after item 16, S4 (about 1.5 h) then V (about 1 h). All builds finish before the chain
  starts, so none of this lane's builds runs inside any of its holds.

## 5. The chain, started

- Every build finished first: R1's, L''s, item 16's, S4's and V's, each `rc=0`. The markers of R1, L', S4 and V equal
  their target sittings' `markers.txt` byte for byte. All four trees are clean at their tips.
- `chain-day68.sh` (the frozen copy of `rtx5090-chain-day68.sh` at `33624ff76`) started at 19:42:30Z. R1's half queued
  on the rig's lock at 19:43:00Z behind another lane's hold.

## 6. A recorded overlap: this lane's CPU builds ran inside R1's gates cell, not its timed cell

- R1's half took the hold at 20:43:01Z (`gates-cell start: host load 12.05 9.54 7.13`, no compute app).
- DAY69's fix and P2's integ branch were built and tested under the rig's CPU cap from 20:32Z to 21:02:51Z. So they
  overlapped the gates cell's first 20 minutes. The gates read exit codes only, with no timing clause.
- No build of this lane runs from 21:02:51Z while the chain's timed cells run. The 5 s host-load log
  (`host-load-5s.log`) records the whole hold.

## 7. A second recorded overlap: DAY70's builds inside R1's timed cell

- DAY70's fix was built and tested under the rig's CPU cap, lowered to 800% for this, inside R1's `ab-r1-cell`. That
  cell started at 21:11:45Z; the lead asked for the fix before integ69 merges, with the queue kept running.
- The windows:
  - about 22:17Z to 22:20:11Z (the placement cell);
  - 22:25:13Z to 22:27:48Z;
  - 22:28:55Z to 22:44:17Z.
  - The 5 s host-load log reads 4.5 to 5.2 before 22:25Z and 5.8 to 10.3 from 22:25Z to 22:43Z.
- The boots they overlapped:
  - o1's last three (`r1 retire-seam` 22:19:54Z, `base prime` 22:22:19Z, `r1 prime` 22:24:45Z), under the lighter
    first window;
  - o2's first eight (22:27:14Z to 22:44Z), both arms interleaved.
- Per the lead's W ruling (DAY61 section 4): if any R1 clause's verdict lands in those boots, the cell repeats once in
  a hold without builds. The reading names the verdict per order, so o2 is the order to check.

## 8. The first chain, read as registered (19:42Z on 2026-09-26 to 14:41Z on 2026-09-27)

- Chain log `rtx5090-r1/chain-day68.log`. Receipts, the executables recorded by hash only:
  `rtx5090-r1/cell/`, `rtx5090-l2/cell/`, `rtx5090-day45/cell-20260927/`, and `rtx5090-{s4,v}/cell-not-run/`.
- **L''s 5090 half: ADOPT.** Verbatim (`rtx5090-l2/cell/reading-l2.log`):

      L READING cell=chain order=o1 twin kv base=13.98 l=0.05 ms (N=95) | long leases base=43.02 l=0.04 ms (N=95, pooled median 32) | stall base=178.70 l=141.57 | e2e base=264.1 l=228.2 | chain base=394.0 l=345.7 ms
      L READING cell=chain order=o2 twin kv base=15.00 l=0.05 ms (N=95) | long leases base=44.14 l=0.04 ms (N=95, pooled median 32) | stall base=177.10 l=137.90 | e2e base=260.9 l=222.3 | chain base=379.5 l=331.6 ms
      L (b) PASS [True, True]   L (c) PASS [True, True]   L (d) PASS [True, True, True, True]
      L2 GATE RED ARM rc=1 staging-fill FAILs=2 of 2 marker=True -> caught (as required)
      L2 VERDICT -> ADOPT (L' is the naked program; the gate change stands)

  - L' is the naked program on both cards. On the 5090 the chain falls by 48 ms per order, and the twin's kv free
    and the long leases fall from 14 to 15 and 43 to 44 ms to 0.05 ms.
- **R1's 5090 half:** (a) passed; the reader's verdict line reads REVERT (b, c, d). Verbatim
  (`rtx5090-r1/cell/reading-r1.log`): `R1 READING order=o1 mode=retire-seam-nosource | no-source settles base N=0
  median=nan r1 N=0 skips r1=0 | ..`, the same N=0 in o2, and `R1 (b) FAIL [False, False]`, `R1 (c) FAIL [True, True,
  True, True, False, True]`, `R1 (d) FAIL [True, True, False, True]`.
  - Read:
    - **The shape R1 changes never occurred on this card.** In `retire-seam-nosource`, base has no no-source settle
      in either order (N=0), and r1 no skip. On the 5090 the seed capture settles at its source's own retire
      (`source-retire` 45 = 45 in both arms), so no capture is pending when another session retires. R1's branch runs
      only then, so it ran in neither arm, and both arms made the same settles with the same counts.
    - (b) fails on that absence (its medians are nan), not on a measured hold.
    - (c) and (d) fail in one cell, o2 `retire-seam`: the long request's e2e is +18.2 ms against +1.0, and the
      source-retire hold is +1.91 ms against +1.0. Its per-boot e2e medians spread from 5.4 to 6.4 s in both arms
      (base 5607, 6142, 6372, 5722, 5756; r1 5405, 5832, 6252, 5710, 5830 ms).
    - That cell's first boot per arm (22:32Z and 22:34Z) lay inside DAY70's build window (section 7).
  - **By the lead's W ruling (a clause landed in overlapped boots), the timed cell repeats once in a hold without
    builds** (section 9). By section 2 a timing clause failed with (a) green is a per-card question, not a revert on the
    target. So R1 stays on main while it is read.
  - The repeat cannot read (b) either, because the shape does not occur on this card. That, and whether the 5090 half
    closes as "R1 unexercised on the 5090" (the same settle program in both arms), is the lead's to rule. It is stated
    now, before the repeat's numbers.
- **Item 16 (DAY45): REGIME NOT REPRODUCED.**
  - Every hump boot started at 80 to 82 C, against the registered 85 C or above. The SM clock medians read 1665 to
    1792 MHz, under 2000 as required.
  - Verbatim: `ITEM16 HUMPS {'xbase': (2, 0.639), 'xg3': (2, 1.299), 'xg4': (2, 0.598), 'xgpp': (2, 0.672)}`, then
    `ITEM16 REGIME NOT REPRODUCED -> nothing is placed; the cell repeats once with the warm-up doubled`.
  - Nothing is placed. Recorded as it read: every arm humps in this regime, base included (+0.639 ms).
- **S4's and V's halves: NOT RUN.** Each hold's 180 x 120 s wait expired behind other lanes' holds (08:40:56Z and
  14:41:25Z; `NOT RUN (hold A): the 5090 lock stayed busy`). Their builds stand.

## 9. The second chain, registered before it runs

- `rtx5090-chain-day68b.sh`, run as the frozen copy `spill-a-cells/chain-day68b.sh`, each part in its own bounded hold
  (180 x 120 s, the idle rule):
  1. **R1's timed cell once more** (`ab.sh seam retire-seam-nosource retire-seam prime 448`, 60 boots), with no build
     of this lane running.
     - Receipts in `r1/receipts-repeat/`; the binaries and the gates (which stand from the first run) are linked from
       there.
     - `r1-reading.py` reads it. The first run's reading stays on record beside it.
  2. **Item 16 with the warm-up doubled,** as DAY45 section 1 registered: "the promote A/B of the same arms added, 40
     boots".
     - `rtx5090-day45/hot-hump-run-2x.sh` is `7b849a817`'s script verbatim, plus that promote A/B (o1 base g4 x5, o2
       g4 base x5, `--mode promote --n 5`) after the demote warm-up. It runs from the frozen `7b849a817` tree.
     - A second miss is recorded and goes to the lead with both holds' telemetry, as DAY45 registered.
  3. **S4's and V's halves,** as section 4 registered them.
- **Owed after it:** P2's 5090 half, registered next.

## 10. The lead's ruling on R1's 5090 half (2026-09-27)

- **Closed as "R1 unexercised on the 5090". That is not a PASS.**
  - Zero no-source settles occurred in either arm and either order, so the branch R1 changes never ran. (b) can be
    neither met nor failed on this card.
  - R1 stays the naked program everywhere. On the 5090 it runs the same settles as base.
- **The queued repeat still runs** (section 9, part 1: the whole timed cell, of which o2 `retire-seam` is the
  question). An e2e of +18.2 ms with identical settles in both arms comes either from this lane's build window
  (section 7) or from something R1-independent, and the repeat tells which.
  - If o2 `retire-seam` reads inside the bound with no builds, the first run's failure is placed on the build window.
  - If it repeats outside the bound with no builds, it is placed as its own finding, not R1's, and registered then.
- Item 16's `REGIME NOT REPRODUCED` and S4's and V's `NOT RUN` stand as they read (section 8), and section 9's chain
  runs as queued.
