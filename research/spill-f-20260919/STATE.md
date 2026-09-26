# WP-F resumable state (2026-09-26 about 15:25Z; BOX36 sitting read; main ff53e3e50 merged; 5090 queue running)

- Lane `lane/spill-f-20260919`, worktree `wt-spill-f`; integ68 takes 290fbbc1e after integ67 (#762).
- Done since 290fbbc1e: OWED 26 routed to F and registered (`M1-PREREG.md` G): G1 visibility in
  every arm's gate and per-visit GPU telemetry; G2 fix `e5d899500` (red arm red, green green); its
  serving-shape check FAILED (a free buffer still waited 30 s; 0.65 tok/s); correction `50e1cf3f6`,
  both red arms red and green, serving-shape check PASS 6 of 6 with zero fallbacks; G3 census `owed26/CENSUS.md`. PRO sitting prepared: `pro-sitting/SITTING.md`
  (host needs) and `pro-sitting/run-sitting.sh` (commands). Old closed OWED 26 renumbered 28.
- Frozen binaries (`~/spill-f-5090`): `bin` (B3 build, BOX27 engine), `bin17` (OWED 17 door,
  pre-fix), `bin18` (handoff door), `bin-g2` (G2 probe), `bin26-attempt1` (first fix, failed the
  serving check), `bin26` (corrected fix: the OWED 17 cell and PRO sittings use this),
  `bin26-tests` (red, green, red2, green2 lib test binaries). Hashes under each `build*/` record.
- Queue (detached, idle-gated on `/tmp/memra-5090.lock`, lanes B and C hold the card for long
  stretches): `m1-5090-queue.py --receipts ~/spill-f-5090/receipts --from owed26-cells --bounded-max
  7864223232 --step-rounds handoff-1g=3-10 --from f17-spec-staged`; remaining steps f17-spec-staged,
  f17-spec-mapped, handoff-1g rounds 3 to 10, bounded, g2, f17, handoff-8g.
  `touch ~/spill-f-5090/PAUSE` holds it between cells; a failing step stops it.
- BOX36 (the lead ran `run-sitting.sh` at 114ef768d, 13:08Z to 14:43Z): raw mirror private in
  `~/.local/share/memra-lane-f-private/box36-f-pro/`, sanitized export and `box36/RESULTS.md` in the
  lane. OWED 17 and 18 PRO rows done; their decisions follow the 5090 rows.
- Merged origin/main ff53e3e50 (integ68); SLRU fixture re-pinned to the merged `moe_cache.rs`
  (3a532fbf...), post-merge battery green. Bounded's first cell OOMed on the runner's whole-file
  hash; fixed (streamed), queue resumes `--from bounded`.
- Mirror each finished step into `rtx5090/`, `owed17/5090/`, `owed18/5090/`, `owed26/5090/`; pool
  bounded with `--fallback-unclean`, f17 with `--bypass-check --fallback-unclean`; handoff with
  `m1-handoff-pairs.py`.
- Scratch to remove at the end: `~/spill-f-5090/` (the red-arm worktree and its build are already removed), `/data/cache/spill-f-b2/`, `/data/cache/spill-f-5090-proof/`,
  `target/handoff-io-tests/` if present.
- Engine or server pushes from this lane now run, before the push: `cargo fmt --all -- --check`;
  `cargo clippy -p memra-engine -p memra-server -p memra-tier --offline --all-targets -- -D warnings`;
  `DOCS_RS=1 cargo clippy -p memra-engine -p memra-server -p memra-tier -p memra-kv -p memra-gguf
  --offline --target x86_64-unknown-linux-gnu --all-targets -- -D warnings`;
  `cargo test -p memra-tier -p memra-kv --offline --no-fail-fast`; the touched crates' suites;
  `bash tools/check-flags.sh`. An edit to `moe_cache.rs` re-pins
  `research/spill-c-20260919/fixtures/slru-synthetic.json` after checking no SLRU statement changed.
- After integ68 lands: merge origin/main (its clippy fix is cherry-picked here as 93214be84, same
  bytes), re-pin the SLRU fixture to the merged `moe_cache.rs`, rerun the list above.
- Card sharing (coordinator, 2026-09-26 ~18:10Z): the 5090 driver now releases the lock after every
  registered cell and sits out a recorded 300 s `yield` (longer than lane B's 240 s) before its idle
  check; each cell stays one continuous hold. Recorded in `~/spill-f-5090/receipts/QUEUE.jsonl`
  (`queue-change`) and in bounded's `waits.jsonl`. Queue relaunched 18:12Z `--from bounded
  --step-rounds bounded=8-10`; bounded rounds 1 to 7 done under the old back-to-back holds.

