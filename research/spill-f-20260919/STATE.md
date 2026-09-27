# WP-F resumable state (2026-09-27 about 20:20Z; integrable; OWED 18's 5090 8 GiB rerun is the open measurement)

- Lane `lane/spill-f-20260919`, worktree `wt-spill-f`; tip on origin; main 21ce97836 (integ72)
  fast-forwarded in. Everything through d58f4bfd8 (OWED 26, 17 and 18 code, BOX36) is in main.
- Resync 2026-09-27 17:00Z (many hours since the last check): the 5090 f17 cell had finished at
  11:05Z and handoff-8g was running; re-derived from QUEUE.jsonl and git before acting.
- OWED 17 closed: door deleted (`31247ab134`), both rigs measured (`owed17/RESULTS.md`).
- OWED 20 closed by owner ruling 2026-09-27 (option 3: BOX27's balloon regime stands; no Step
  sitting); the pin, census and sitting text stay as the record.
- Scratch loss (~19:20Z): `~/spill-f-5090` vanished, cause unknown (`rtx5090/RESULTS.md`); the
  handoff-8g cell's first run (rounds 1 to 8) was lost. Rerun from round 1 on a rebuild of
  5b001e125 (`owed18/build-rebuild/`, section E rebuild amendment), started 19:42Z; round 1 passed
  and is mirrored; a nice-19 watcher (`~/spill-f-5090/mirror-8g.sh`, 12 h timeout) mirrors each
  later round into `owed18/5090/handoff-8g/` (drop the `.mirrored` marker files before committing).
  The driver now mirrors cells itself (`--mirror`, from the next launch).
- Post-deletion pool GPU cells: 6 of 6 on the 5090 (`owed17/deletion/poolcells-after-deletion`).
- After handoff-8g: `m1-handoff-pairs.py owed18/5090/handoff-8g`, then the OWED 18 decision
  (section E, both rigs).
- Builds: nice 19, CPUQuota=600%, MemoryMax=12G. Pre-push (owner, 2026-09-27: stop
  overcomplicating local CI): the affected crates' tests only; effort goes to the measurements
  that decide open items.
- Scratch at the end: `~/spill-f-5090/`, `/data/cache/spill-f-b2/`, `/data/cache/spill-f-5090-proof/`,
  the local backup ref.
