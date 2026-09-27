# integ10 self-review (lead, 2026-09-21)

Read in full: D day 12 `kv_tier_gate/fault.rs`, `fault_contract.rs` diffs (arms through the seams, REFUSED branch removed) and the reclaim/contract replay tests; B day 13 `crates/memra-server/src/worker.rs` diff (155 lines: `PoolReading`, `pool_readings`,
`reclaim_settle_keep`, `settle_reclaimed_prefix_bytes`, the reclaim-on-defer call site, unit test) and
`tools/prefix-evict-reclaim-gate.py` (697 lines, skimmed for the four verdict clauses); A day 11 `crates/memra-kv/src/lib.rs`
(`SuspendedLayers`, `ContinuationRefused`, `SuspendError`, `suspend_layer`/`resume_layer`, `ensure_usable`),
`crates/memra-engine/src/tier_transfer.rs` (`recover_source`, `holds_cancelled_source`, the `retire`/`retire_source`
guards), `crates/memra-tier/src/conformance/recovery.rs`, `contracts.rs`, the transfer and resident-binding tests.

## Findings
1. **B, the fix is accounting, not a program change.** `settle_reclaimed_prefix_bytes` runs only inside the
   reclaim-on-defer tick after an eviction: it fences the model-owned streams, trims each pool to
   `used_after + cached_before` (so only the eviction's gain leaves the pool; `reclaim_settle_keep` returns `None` when
   the cache did not grow), re-reads driver free, and prints the retained bytes when the pool cannot release a chunk
   that shares a live neighbour. Tokens are byte-identical across base and fix on the target card (digest in the
   gate line). The synchronize in that tick is a stall for every session during an OOM-reclaim event, which is already
   the exceptional path; it is the same shape #536 lists (copy engine off the tick) and belongs to that issue, not
   this fix.
2. **B, gate is red on base and green on fix** on the same card, artifact and prompts, N=1, with the driver free delta
   quoted (`driver_free_delta_bytes=none` versus `1610612736`). `pool_retained_bytes` is printed and counted as
   pool-cached headroom, never as driver free. Two refused calibration cells and one over-tight first V3 clause are
   kept in the receipts, not relabelled.
3. **A, rule 1 seam.** `recover_source` requires an observed producer and an idle source side, is once-only
   (`source_retired`), refuses on retired or published tickets, and `retire`/`retire_source` answer `Busy` while a
   cancelled H2D still holds its source, so the source can never be consumed and then cancelled. Frozen schedule
   blobs are byte-identical (hashes in A's DAY11.md); the new schedules live in `conformance/recovery.rs`; the CPU
   fake with `legacy = true` is the red arm. Native bindings exist in `tier-transfer-gate` but were not run on a card
   today: pending D day 12, stated in the record.
4. **A, rule 2 seam.** `SuspendedLayers` is a typed register; `suspend_layer` takes the layer and clears the cached
   graphs (`glm5_decode_graph`, `glm5_tp_sym_graph`, `qwen_prime_graph`), `resume_layer` refuses an occupied or
   non-suspended slot, and `ensure_usable` returns `ContinuationRefused { path, layers }` while the register is
   non-empty; the signature is unchanged, so all 30 callers compile and no server code moved. Today nothing in the
   server fills the register (kv-tier-gate still uses raw `take()`), so serving behaviour is unchanged; the contract
   is real once D's day-12 arms use the seam.
5. **Build script hygiene (lead, this PR).** `crates/memra-engine/build.rs` reads `DOCS_RS` but did not declare
   `cargo:rerun-if-env-changed=DOCS_RS`; a clippy pass with `DOCS_RS=1` left stub artifacts that the next
   `cargo test -p memra-server` linked against (`undefined symbol: memra_dsv4_c4_recent_write` and friends) until a
   `cargo clean -p memra-engine`. One line added; the two failed attempts stay in the battery dir.
6. **Nits (not blocking).** The collector pytest suite takes the real rig lock path, so it fails with `EAGAIN` while a
   serving job holds `/tmp/memra-5090.lock` (attempt 1 here); it should take a private lock path under test. B's
   gate is a Python script under `tools/` with its own verdict clauses; a `--validate` receipt exists but the gate is
   N=1 and says so.

7. **D day 12 closes the gap revuto found.** A's `retire`/`retire_source` `Busy` guard on a cancelled H2D with an
   unrecovered source contradicted the day-11 `cancel-restore` sequence still shipped in `fault.rs` (`retire` expected
   `Ok(())`). D's day 12, merged into this PR, rewrites that arm through `recover_source` and the `require-resident`
   arm through `suspend_layer`/`resume_layer`; both print `FAULT-ARM PASS` natively, and A's two rule lines PASS
   natively for the first time. The REFUSED branch of the arm vocabulary is removed: a backend without the seam is now
   a failed cell, which is stricter.

## Verification this review relied on
integ10 CPU battery (`integration-day12/integ10-cpu-battery/`): fmt, tier+kv+gguf 600 tests, memra-server 734 tests
(after the engine rebuild), clippy `-D warnings`, censuses, collector pytest 85 (rerun with the lock free), perf board,
`git diff --check`, all rc=0. Local RTX 5090 `tools/serve-smoke.sh` on the merged tree: `serve-smoke: 0 failed`.
B's target-card gate pair and serve-smoke on the fix; A's tier+kv 303 tests and BOX3 release builds. This rig cannot
run the model gates; the argmax and K=1..8 evidence stays the lanes' target-card cells.
