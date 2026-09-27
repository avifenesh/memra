# WP-F resumable state (2026-09-27 about 20:00Z; NEED TARGET CARD for OWED 20; 5090 handoff 8 GiB rerunning)

- Lane `lane/spill-f-20260919`, worktree `wt-spill-f`; tip on origin; main 21ce97836 (integ72)
  fast-forwarded in. Everything through d58f4bfd8 (OWED 26, 17 and 18 code, BOX36) is in main.
- Resync 2026-09-27 17:00Z (many hours since the last check): the 5090 f17 cell had finished at
  11:05Z and handoff-8g was running; re-derived from QUEUE.jsonl and git before acting.
- OWED 17 closed: door deleted (`31247ab134`), both rigs measured (`owed17/RESULTS.md`).
- OWED 20 (owner pick, candidate 3): registered (`M1-PREREG.md` H), pinned, census, sitting ready:
  `owed20/SITTING.md` (box shape), `owed20/run-sitting.sh` (commands), `owed20/m1-step-runner.py`
  (split-artifact, two-card wrapper; unit cells `owed20/test-m1-step-runner.py`). The GGUF heads
  for the census are private (`~/.local/share/memra-lane-f-private/owed20/raw/`, hashes in
  `owed20/raw/HEADS.sha256`): the 16 MiB binaries tripped the public-boundary patterns, and the
  unpushed commits carrying them were rebuilt without them (backup ref
  `backup/spill-f-pre-heads-fix`, local only; delete after integration).
- Scratch loss (~19:20Z): `~/spill-f-5090` vanished, cause unknown (`rtx5090/RESULTS.md`); the
  handoff-8g cell's first run (rounds 1 to 8) was lost. Rerun from round 1 on a rebuild of
  5b001e125 (`owed18/build-rebuild/`, section E rebuild amendment), started 19:42Z; round 1 passed
  and is mirrored; a nice-19 watcher (`~/spill-f-5090/mirror-8g.sh`, 12 h timeout) mirrors each
  later round into `owed18/5090/handoff-8g/` (drop the `.mirrored` marker files before committing).
  The driver now mirrors cells itself (`--mirror`, from the next launch).
- Post-deletion pool GPU cells: 6 of 6 on the 5090 (`owed17/deletion/poolcells-after-deletion`).
- After handoff-8g: `m1-handoff-pairs.py owed18/5090/handoff-8g`, then the OWED 18 decision
  (section E, both rigs).
- Builds: nice 19, CPUQuota=600%, MemoryMax=12G (lead, four lanes at once). Pre-push list:
  fmt; both battery clippy forms; tier and kv tests; touched suites; check-flags; an edit to
  `moe_cache.rs` re-pins the SLRU fixture after the SLRU-statement check.
- Scratch at the end: `~/spill-f-5090/`, `/data/cache/spill-f-b2/`, `/data/cache/spill-f-5090-proof/`,
  the local backup ref.
