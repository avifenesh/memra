# WP-F resumable state (2026-09-27 about 18:40Z; NEED TARGET CARD for OWED 20; 5090 handoff 8 GiB finishing)

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
- 5090 queue: handoff-8g rounds 9 and 10 remain (idle-gated, 300 s yield after every cell); a
  one-off `poolcells-after-deletion` step runs the pool's GPU cells on the post-deletion build.
  After handoff-8g: `m1-handoff-pairs.py ~/spill-f-5090/receipts/handoff-8g`, then the OWED 18
  decision (section E rules, both rigs).
- Builds: nice 19, CPUQuota=600%, MemoryMax=12G (lead, four lanes at once). Pre-push list:
  fmt; both battery clippy forms; tier and kv tests; touched suites; check-flags; an edit to
  `moe_cache.rs` re-pins the SLRU fixture after the SLRU-statement check.
- Scratch at the end: `~/spill-f-5090/`, `/data/cache/spill-f-b2/`, `/data/cache/spill-f-5090-proof/`,
  the local backup ref.
