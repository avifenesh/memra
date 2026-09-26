# WP-F resumable state (2026-09-26 about 11:25Z, integrable milestone; 5090 timing queue running)

- Lane `lane/spill-f-20260919`, worktree `wt-spill-f`; origin/main merged at a233f6fe5 (integ65);
  integ66 (#744) merged ef5640af8.
- Resync 07:34Z after the requested 07:28Z reboot: recorded in `M1-PREREG.md` (D resync
  amendment); interrupted and refused cells kept under `rtx5090/` as `interrupted-*`, `refused-*`.
- Done: BOX27 (`box27/RESULTS.md`, with the worker2 fallback correction); 5090 capped regime,
  unscored (`rtx5090/RESULTS.md`); OWED 17 and 18 implemented, default off, correctness green
  (`owed17/RESULTS.md`, `owed18/RESULTS.md`); OWED 20 candidates (`ITEM20-CANDIDATES.md`, owner
  pick); OWED 26 flagged (worker demand reads fall back to mmap when the ring is busy).
- Running, detached, idle-gated on `/tmp/memra-5090.lock` (lanes B and C hold the card most of the
  time, so cells trickle): `m1-5090-queue.py --receipts ~/spill-f-5090/receipts --from handoff-1g
  --bounded-max 7864223232`, steps handoff-1g (round 1 done), bounded, g2, f17, handoff-8g.
  Progress: `~/spill-f-5090/receipts/QUEUE.jsonl` and each step's `waits.jsonl`. Hold it between
  cells with `touch ~/spill-f-5090/PAUSE`; stop it by killing the queue and rounds PIDs (my own
  processes); resume with `--from <step>` (and `--capped-rounds`, `--bounded-max`). A failing
  step stops the queue. Attribution sampler `m1-io-attribution.py` (nice 19, 12 h timeout) writes
  `~/spill-f-5090/receipts/io-attribution.jsonl`; it folds reaped children into parents.
- After each step: mirror `~/spill-f-5090/receipts/<step>` into `rtx5090/`, `owed17/5090/` or
  `owed18/5090/`; pool bounded with `m1-b3-pool.py <dir> --fallback-unclean`, f17 with
  `--bypass-check --fallback-unclean`; the handoff pairs from each cycle.json (section E rule).
- Scratch to remove when the queue ends: `~/spill-f-5090/`, `/data/cache/spill-f-b2/`,
  `/data/cache/spill-f-5090-proof/`, `target/handoff-io-tests/` if present.
- PRO 6000 halves of OWED 17 and 18: NEED TARGET CARD.
