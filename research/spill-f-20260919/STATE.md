# WP-F resumable state (2026-09-26, 5090 half queued; items 17 and 18 implemented)

- Lane `lane/spill-f-20260919`, worktree `wt-spill-f`; origin/main merged at a233f6fe5 (integ65).
- BOX27 complete and destroyed; receipts and verdicts in `box27/` and `box27/RESULTS.md`.
- 5090 half (OWED 19, 23; `M1-PREREG.md` section D and its bounded amendment): frozen binaries
  built at the pre-merge tip, engine source equal to BOX27's (`rtx5090/build/`), in
  `~/spill-f-5090/bin`. Fresh `/data` proof PASS (`rtx5090/proof/`). Capped smoke done: all six
  arms correct, 128 tokens equal across arms; ru_maxrss counts the mapped file, so bounded is sized
  by `m1-anon-peak.py` (amendment); foreign device bytes 1.5% to 35% from desktop co-tenants.
- OWED 18 (section E): `MEMRA_KV_HOST_HANDOFF_IO=direct`, `crates/memra-server/src/handoff_io.rs`,
  unit cells green (byte-identical files, cross-readable, truncation). Frozen gate binaries in
  `~/spill-f-5090/bin18` (`owed18/build/`).
- OWED 17 (section F): `MEMRA_MOE_COLD_BYPASS=staged|mapped` in `moe_cache.rs` and
  `spill_pread.rs`, CPU cells green, GPU ownership cell queued. Frozen run-gen in
  `~/spill-f-5090/bin17` (`owed17/build/`). Arms lock `m1-prereg/f17-arms.lock.json`.
- OWED 20: candidates written in `ITEM20-CANDIDATES.md`; owner pick.
- Resync 2026-09-26 07:34Z after the requested 07:28Z rig reboot: tip and origin matched
  (`03c9f645d` plus one unpushed data commit), binaries and receipts under `~/spill-f-5090` intact
  and hash-verified, the queue gone. Capped rounds 1 to 4 complete and mirrored; round 5 died
  mid visit 5, banked as `capped/interrupted-round-05-reboot` (never scored); queue relaunched
  with `--capped-rounds 5-10`.
- Running: `m1-5090-queue.py --receipts ~/spill-f-5090/receipts` (detached), order capped,
  anonpeak, bounded, g2, mapped-gpu-cell, f17-smoke, f17-smoke-gate, f17, handoff-1g, handoff-8g.
  Each round waits for an idle card and a free `/tmp/memra-5090.lock` (waits recorded);
  `touch ~/spill-f-5090/PAUSE` holds it between cells for this lane's compiles. Resume a stopped
  queue with `--from <step>`. Progress: `~/spill-f-5090/receipts/QUEUE.jsonl`.
- After each regime: `m1-b3-pool.py <dir>` (f17: `--bypass-check`), mirror the receipts into
  `rtx5090/` (then `owed17/`, `owed18/`), record verdicts in OWED and a 5090 RESULTS file.
- Local scratch to remove at the end: `~/spill-f-5090/`, `/data/cache/spill-f-b2/`.
