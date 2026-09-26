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
