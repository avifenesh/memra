# WP-A resumable state (2026-09-26, stopped at NEED TARGET CARD; the 5090 cell still queued)

- Lane `lane/spill-a-20260919`; worktree `wt-spill-a`. integ65 takes `071e1126a` (F2b included).
- Items 21 to 25 closed (DAY53 to DAY58); item 23's addendum F2b (`b4d6f95c2`) accepted.
- Item 10 (the fanout publisher, DAY59): the owner-time split is built (`8b5e5e213`, log only). The attribution cell
  selects design B1 or B2 by a pre-registered rule.
  - The 5090 re-run took the lock but read a foreign compute app on the card for all 15 idle checks and stopped
    NOT RUN (DAY59 section 5, `rtx5090-day59/cell-not-run/`); the first attempt is banked as `cell-cancelled/`.
  - **The cell reads owner-thread time, so it wants a quiet host.** The 5090 rig's CPUs are shared with B and C; a
    target-card twin is prepared (`pro-single-day59/`: `build.sh <tip>`, then `driver.sh`), and if both run, the target
    card's selection decides (DAY59 section 4).
- Pre-registered and waiting in order (DAY60 to DAY65): item 11 (DAY60, its sitting `pro-single-day60/` prepared: C's
  day-29 cell over the tip and the day-16 tree), item 12 (DAY61, design W), item 13 (DAY62, the retire seam: lines,
  a price, R1 or R2), items 14 and 19 (DAY63, design L, the pinned pool and the staging set at boot; item 17 re-read on
  top), item 18 (DAY64, the late promotes placed first), item 20 (DAY65, design T-H).
- Target sittings ready for one box (any class with one RTX PRO 6000 Blackwell and the 27B at
  `/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf`): DAY59's twin (about 20 minutes) and DAY60 (about 40 minutes).
- The owed 5090 cells (S4's half, V's half, item 16) queue after DAY59's on the 5090.
- Local scratch: `target/day59/memra-server` (the 5090 cell's binary; removed when the cell closes).
