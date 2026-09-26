# WP-A resumable state (2026-09-26, NEED TARGET CARD: L's sitting; R1's queued on BOX31; W's 5090 half waiting)

- Lane `lane/spill-a-20260919`; worktree `wt-spill-a`. integ65 took `9ab479d9c` (DAY66); integ67 takes B1 after integ66.
- Items 21 to 25 closed (DAY53 to DAY58); item 23's addendum F2b (`b4d6f95c2`) accepted.
- Item 10 (the fanout publisher, DAY59). The target twin selected **DESIGN B1** (calls 0.83 / 0.82 of 1.34 ms, DAY59
  section 6). Revuto's split defect on #731 did not move it: all 100 split lines carry the clean counts (section 8).
  The 5090 re-run was NOT RUN, because a foreign app held the card (section 5).
  - **B1 ADOPTED** (DAY59 section 10, code `e522a9417`, for integ67 after integ66): own time 0.25 ms against
    1.36 / 1.34, stall gain 1.03 / 1.15 ms, every gate green.
  - integ67's asks (DAY59 section 11): the gridDim.y refusal `231fba087` (with its red arm); B1's 5090 half
    `rtx5090-b1/` is building, then runs in one hold (receipts `rtx5090-b1/cell/`, untracked; executables in
    `/home/avifenesh/spill-a-cells/b1-bins`, removed when it closes). W's 5090 half repeats whole after it.
- DAY66 (revuto on #731): the fanout's split is scoped to its snapshot (`prefix_copy_scoped`), with its census, cell
  and red arm (`day66/`), server lib 935 passed. No card.
- Item 11 (DAY60) is closed with its reading: CLAUSE NOT MET as stated before it ran, and y_minus_x is +86.3 (demote)
  and +76.0 (promote) in both orders. The class-isolating candidate cell is registered as text for the owner (section
  4).
- Item 12 (DAY61, design W): built (`34a348fd2`, CPU-green), with two sittings (DAY61 section 2).
  - The target: `pro-single-w/build.sh 9d0143dbf 457321806`, then `driver.sh`, receipts in
    `/root/spill-receipts/a-w`, last line `W VERDICT card=target -> ..`.
  - The 5090: `rtx5090-w/build-local.sh` then `card-run.sh`, running in the background into `rtx5090-w/cell/`
    (untracked; bank it without `bins/`, and remove `bins/` when the cell closes). It holds the 5090 lock under the
    idle rule.
  - The target read `FAIL (a, c)`: (a) was a harness defect, now fixed; (c) is real, promote at 1.25 / 1.33 of base
    (DAY61 section 3). W is not the target's program. If the 5090 half passes, W becomes a per-card write-combined arm
    under a new pre-registration; otherwise it is reverted. Its 5090 run died in the 07:28Z reboot, then was
    stopped by me at 07:36Z (after (a), before its timed cell) so B1's half runs first; it repeats whole after.
  - Scratch to remove when the cells close: `rtx5090-w/cell/bins/`, `/home/avifenesh/spill-a-cells/`, and the four
    lines added to the shared `/home/avifenesh/projects/memra/.git/info/exclude`.
- Item 13 (DAY62): the price selected R2; R2 was built, read `REVERT (c)` and reverted (section 7). The stall is
  the long prime's own segments, and the wait moved to the next capture's second-capture settle. R1's corrected cell
  selected R1 (no-source hold 12.53 / 12.54 ms). R1 is built (`04554e99f`); its sitting is `pro-single-r1/`
  (`build.sh <tip> <R1's parent>`, then `driver.sh`, last line `R1 VERDICT -> ..`).
- B1's 5090 half read ADOPT (DAY59 section 12); B1 goes to integ67.
- Items 14 and 19 (DAY63, design L): built (`f6dfe303a`, CPU-green, section 3). Its sitting is `pro-single-l/`
  (`build.sh <tip> <L's parent>`, then `driver.sh`, last line `L VERDICT -> ..`). Item 17 is re-read on top after L's
  verdict.
- Pre-registered and waiting in order: item 13's design (DAY62, the retire seam: lines, a price, R1
  or R2), items 14 and 19 (DAY63, design L; item 17 re-read on top), item 18 (DAY64), item 20 (DAY65, design T-H).
- The owed 5090 cells (S4's half, V's half, item 16, and B1's (a1) and (a2) as a compatibility reading) queue on the
  5090.
- The local 5090: another project's process (a Python venv, no rig lock) keeps landing on the card during lane
  cells; lane C's i15 rerun went void on it. The idle rule stays: a GPU cell here waits for an idle card, or reports.
  The lead has told the owner.
