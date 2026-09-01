# cx-slrucache progress ledger

Date: 2026-08-13
Branch: `lane/cx-slrucache`
Base: `v0.81.3` (`7cf5fd842ebc76f6e8a82910a8e6d4b864b6b42d`)
Rig: thermally capped local RTX 5090 Laptop GPU (210-1200 MHz; clocks stay locked)

## Objective

Replace the cross-request prefix cache's global timestamp LRU with a byte-budgeted
probation/protected SLRU. New snapshots enter probation and earn protected residency only on a
real reuse. Preserve namespace isolation, leases/pins, emergency flush, the global byte ceiling,
and refusal of an entry larger than that ceiling.

## Required source inventory — read before lane edits

The merged eviction-churn evidence is newer than this lane's mandated tag, so it was read without
rebasing from `main` commit `7f1788be6`:

- `research/evictchurn-20260813/RESULTS.md` and `PROGRESS.md`: fixed 782 MiB / 12-of-40 baseline,
  policy inventory, exactness receipts, and the three scored outcomes.
- `research/evictchurn-20260813/run-local5090.sh` and `evict_churn.py`: direct `prompt_ids`, four
  stable tenant salts, serial per-request metric attribution, schedules, refusal parsing,
  byte-identity checks, TTFT reduction, fresh server per arm, and tee-first raw receipts.
- `crates/memra-server/src/worker.rs:2186-2600`: worker-owned `PrefixEntry`/`PrefixCache`, global
  evictable timestamp index, lookup/pin/unpin, exact-key dedupe, insertion refusal, eviction loop,
  and emergency flush. The restore hit/pin path is at `worker.rs:6133-6190`; source tests are at
  `worker.rs:9643-10087`.
- `crates/memra-engine/src/moe_cache.rs:1-13,153-193,522-538`: existing expert-cache SLRU
  precedent (probation admission, reuse promotion, protected demotion, probation-first victim,
  80% protected default).

The public contract being preserved is `docs/SERVING.md:918-982`; the existing knob row is
`docs/FLAGS.md:518`.

Current external cross-checks (retrieved 2026-08-13): vLLM documents reference-count-aware LRU
block eviction and `cache_salt` hashing, while S3-FIFO uses a small probation FIFO plus a metadata
ghost queue and a main queue. Sources:

- <https://docs.vllm.ai/en/v0.14.1/design/prefix_caching/>
- <https://www.pdl.cmu.edu/ftp/Storage/FIFOqueues-SOSP23_abs.shtml>
- <https://s3fifo.com/blog/2023/08/01/fifo-queues-are-all-you-need-for-cache-eviction/>

## Predeclared design

- Default protected target: 80% of `MEMRA_PREFIX_CACHE_MB`, configurable as an integer percent;
  probation receives the remaining target.
- Segment occupancy and limits are bytes, never entry counts. Probation may borrow currently
  unused protected capacity so the global budget remains useful and an individually fitting Q27
  snapshot is not rejected merely because it exceeds the nominal probation slice.
- A successful reuse promotes probation to protected MRU. Protected overflow demotes protected
  LRU to probation; capacity pressure evicts probation LRU before protected LRU. Pinned entries
  remain unevictable and may temporarily defer segment rebalancing until release.
- Emergency flush still removes every unpinned entry and preserves every pinned entry. Exact
  `(model, cache_salt)` lookup visibility and the global process budget do not change.

The 80/20 split is a declared starting point, not selected from public evaluation scores. It
matches the already-shipped expert-cache split and leaves about two 68,313,600-byte baseline
entries in nominal probation at 782 MiB while the eight-entry hot subset fits protected.

## Predeclared finite-trace prediction

The 80/20 hot-set should remove or nearly remove its 13 scan-polluted hot misses. The sequential
one-pass scan should remain 0 hits and 0 evicted-before-reuse.

A strict first-reuse SLRU cannot create a hit in evictchurn's two-cycle 40-key round robin at
12-entry capacity: no key is reused until 28 earlier keys have already forced it out, so no entry
can earn promotion. S3-FIFO's ghost queue likewise recognizes such a key only on the second miss;
that request is not retroactively a hit. The unchanged trace is still being run and reported. No
admission refusal, ghost-hit promotion, schedule change, or selective reporting will be added to
manufacture the requested round-robin improvement.

## Coordination

`lane/cx-cachesize` had not landed when this lane started. Its in-progress box1 receipts report
sold-shape entry bytes of 301,215,744 (Q27) and 110,964,480 (Q35). The policy therefore treats the
split as byte targets with unused-share borrowing; its budget recommendation can compose later
without changing the eviction semantics.

## Checklist

- [x] Verify clean dedicated worktree and exact `v0.81.3` base.
- [x] Read and cite evictchurn evidence, harness, contention driver, cache implementation, and
  expert SLRU precedent before editing.
- [x] Check current external S3-FIFO and vLLM policy documentation.
- [x] Check `cx-cachesize` state and record the available entry-size receipts.
- [x] Checkpoint this progress ledger (`7d63fb296`).
- [x] Implement byte-budgeted SLRU, focused host tests, and documented configuration.
- [x] Replay the three unchanged eviction-churn schedules on the final binary under the shared
  5090 lock; the scored `raw/run-20260812T235813Z/` manifest verifies.
- [x] Run final prefix exactness, contention byte identity, focused host tests, and
  `cargo test --workspace` on the post-audit implementation.
- [x] Complete the post-audit full GPU battery. The authoritative receipt is
  `raw/postaudit-battery-complete/`; the two swept attempts remain preserved as incomplete history.
- [x] Finalize `RESULTS.md`, retain raw logs/manifests, review, and commit the completed lane.

## Guardrails

- Raw stdout/stderr is tee'd before parsing; causes are quoted only from captured text.
- Every timed local GPU run holds `/tmp/memra-5090.lock`; the thermal cap is not changed.
- No live serve box, merge, tag, push, generated board, or release operation in this lane.

## 2026-08-13 — implementation checkpoint

- Replaced the one global evictable LRU index with probation/protected indexes, per-entry segment
  identity, exact segment-byte accounting, an 80% protected byte target, and probation-first
  capacity victims. Protected overflow demotes protected LRU back to probation.
- Successful restored hits promote while acquiring their existing lease. Same-window fanout with
  multiple participants counts as immediate demonstrated reuse; all existing lease refcounts and
  last-release behavior remain intact.
- `MEMRA_PREFIX_CACHE_PROTECTED_PCT` accepts 1..99 and defaults/falls back to 80. The startup log,
  `docs/FLAGS.md`, and `docs/SERVING.md` describe the policy and unused-share borrowing.
- Focused host receipt before the checkpoint: `cargo test -p memra-server prefix_cache` passed
  11/11 (212 filtered out), including unequal byte sizes, cross-tenant scan protection,
  protected demotion, oversized refusal, pins, fanout, emergency flush, and the 10,000-entry
  index smoke.

## 2026-08-13 — host and harness checkpoint

- `cargo test --workspace` exited 0; the retained raw log is
  `raw/cargo-test-workspace.log`. The server crate passed 223/223 tests, including all 11 focused
  prefix-cache cases, and its dependent workspace suites and doc tests were green.
- A lane-local replay wrapper now freezes evictchurn's schedules, seeds, prompt shape, 40-prefix
  working set, four tenants, 12-entry-equivalent byte budget, and fresh-server isolation while
  using the required `/tmp/memra-5090.lock`. `bash -n` passed.
- The release server was rebuilt after the final warning-only annotation. No GPU measurement has
  started while the neighboring `cx-fa3softmax` kernel-check is resident on the 5090.

## 2026-08-13 — unchanged evictchurn replay checkpoint

- Scored run: `raw/run-20260812T232540Z/`; all five phase exits are zero, both manifests verify,
  and the terminal marker is `SLRUCACHE_LOCAL_PASS 2026-08-12T23:27:38Z`.
- The calibration reproduced 68,313,600 bytes per entry and the fixed 782 MiB budget held 12/40
  entries. Across 357 thermal samples the observed maximum was 1,200 MHz and 62 C; clocks were
  not changed.
- Round-robin remained 0/40 reuse hits, 40 evicted-before-reuse, 68 evictions, zero refusals.
  This does not meet the predeclared acceptance target: with a 40-reference reuse distance and
  capacity 12, no entry receives the hit required for SLRU promotion before LRU probation eviction.
- Hot-set reuse improved from 107/120 to 115/120; broad driver thrash fell from 13 to 5 and
  evictions from 41 to 33, with zero refusals. All five misses occurred before that key's first
  cache hit; after promotion, no protected hot key missed again.
- Sequential scan stayed at 0 hits, 0 thrash, 28 evictions, and zero refusals. Prefix exactness
  passed repeated 3/3 and shared-prefix 3/3; all 115 contention hits were byte-identical.
- This is a faithful strict one-hit SLRU result, not a tuned acceptance-table result. The round-
  robin target cannot be met by the requested hit-triggered promotion rule without a separate
  admission/refusal policy or retaining more than the fixed byte budget.

## 2026-08-13 — final validation

- `tools/local-ci.sh` ran from a fresh release build under `/tmp/memra-5090.lock` and exited 0:
  kernel-check 106 green / one optional capture skip; prime-gate green; run-spec K=1..8 8/8;
  31B and 12B run-gen MATCH; verify, decode-batch, graph-warmup, serve-smoke, c=64 stress, and
  served-spec acceptance all green. `serve-smoke` reported `0 failed`.
- Final focused host cache tests passed 11/11, and the earlier complete workspace run remains
  green. The post-battery GPU snapshot was idle with no compute applications.
- `RESULTS.md` records a NO-GO against the literal acceptance table. It separates driver-defined
  hot thrash (5) from post-promotion protected thrash (0) and reports the unchanged round-robin
  result without a parameter sweep.
- No live serve box, merge, tag, push, generated board, clock change, or hook bypass occurred.

## 2026-08-13 — final source-audit checkpoint

- The post-battery source audit found a generic victim fallback that could select protected LRU
  when every probation entry was pinned, even while protected was within its share. The scored
  serial traces never reach this corner, but it contradicted the literal policy contract.
- Capacity victims are now probation-only. Pinned retention preflights bytes available from
  existing probation plus protected LRU entries that its own promotion would demote; if that is
  insufficient, the snapshot is not retained instead of crossing the protected share. Fanout
  participants keep serving from their private session copies, so pin ownership remains exact.
- A focused regression test covers the protected-under-share refusal. Focused cache tests pass
  12/12 and a fresh `cargo test --workspace` passes with the server crate at 224/224.
- Because this is a post-battery code change, the behavior replay and full GPU battery will be
  repeated from a newly built final binary; the earlier complete receipts remain preserved rather
  than overwritten.

## 2026-08-13 — orchestrator-checkpoint recovery audit

- The post-audit behavior replay had completed before the second sweep. All five phase exits in
  `raw/run-20260812T235813Z/` are zero, its output manifest verifies, and the retained server hash
  matches the current `target/release/memra-server` built from `a5ddc3da3`.
- The post-audit full battery had not completed. `raw/local-ci-final.log` ends immediately after
  `run-gen argmax: MATCH (31B)`; the complete `raw/local-ci.log` predates `a5ddc3da3` and therefore
  remains supporting evidence only.
- Resume only the missing full battery under `/tmp/memra-5090.lock`, preserving both earlier logs.
  No box1 work is planned; if that changes, scored GPU work there must hold
  `/tmp/memra-gpu.lock` across both PRO 6000 cards because they share a PIX path.

## 2026-08-13 — post-audit battery completion

- A fresh `tools/local-ci.sh` correctness battery ran to completion under one exclusive
  `/tmp/memra-5090.lock` hold and exited 0. Its source receipt is `65067371b`; that commit has no
  crate, tool, or documentation changes relative to audited implementation commit `a5ddc3da3`.
- The authoritative raw directory is `raw/postaudit-battery-complete/`. It retains the full
  tee-first log, exit code, source id, server binary hash, prime/run-spec/server/stress/accept
  sublogs, and before/after GPU state. Both compute-application snapshots are empty.
- Kernel-check was 106 green with one optional capture skip; prime-gate was green; run-spec was
  8/8; both run-gen arms matched; both K=7 verify arms, 31B 64/64 stream agreement, four batched
  decode arms, graph-warmup plus canary, serve-smoke, c=64 stress, and served-spec acceptance all
  passed. `serve-smoke` reported 0 failed.
- The raw completion receipt was checkpointed as `ae7fd1ff7`. No box1 work, live-serving change,
  merge, tag, push, board edit, clock change, `cargo fmt`, or hook bypass occurred.
