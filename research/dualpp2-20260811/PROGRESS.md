# Dual-active PP-2 admission re-gate and promotion

Date: 2026-08-11

Lane: `lane/cx-dualpp2`

Original base: `110376f85b864833e5758ec14ea4a1a0a3245b14` (`lane/cx-dualpp1`)

Merged base: `4fa5a26679453c4b93cfd39840e822cb1585e612` (`origin/main`, fetched 2026-08-11)

## Status

The four ordered PRE-RE-GATE blockers and the post-review local 5090 exactness battery are complete.
Dual-active remains default OFF; no promotion decision is valid until the c=64 admission re-gate
and x100 cross-device event-ordering soak pass on a PRO pair.

The remediation started from `efa72bb6` and each blocker is isolated in its own commit. Blocker #4
merged the fetched `origin/main` snapshot above. The remote advanced during the work (`6bfe89a3`
at exactness start and `49f5002d` at the final audit), so those later refs are recorded as observed
rather than falsely claimed as merged. The blocker goal is satisfied on the tested source: the
peerprobe boot gate is present, and neither later ref alters the three files resolved for the merge.

No merge of this lane into `main`, tag, push, formatting run, or generated performance-board
change is in scope for this lane.

## PRE-RE-GATE blockers

### Blocker #4 — peerprobe-bearing base

- Merged fetched `origin/main` at `4fa5a26679453c4b93cfd39840e822cb1585e612` into the lane.
- The only textual conflict was the shared `pp.rs` host-bounce test module. Its resolution keeps
  the dual-active split, eligibility, and timing assertions together with peerprobe's corrupted
  readback fail-closed assertion. `worker.rs` and `main.rs` auto-merged and retain the dual-active
  scheduler/metrics paths alongside the peerprobe-bearing engine initialization path.
- `git grep run_peer_probe crates/memra-engine/src/pp.rs` finds all three probe call sites on the
  merged tree. Large-rung sufficiency remains blocker #1 below and is not inferred from presence
  of the boot gate alone.
- `cargo test -p memra-engine host_bounce_tests --lib` passes all 8 merged dual/peerprobe cells,
  and `cargo check -p memra-server` passes the auto-merged worker/server integration.

### Blocker #6 — dual admission re-derivation on merged base

The pre-merge one-boundary verdict below is superseded. The old term does not prove the c=64
two-wave peak:

- A combined dual tick runs `ceil(chunk/2)` on stage 1 while stage 0 runs
  `floor(chunk/2)`. Each PP device therefore owns one cap-bounded live walker at the overlap peak;
  neither distinct device owns two complete wave scratch sets (`decode_batch.rs`, dual stage-0 and
  stage-1 bodies).
- The waves do not allocate or duplicate KV. Each admitted session already owns one cache, and
  `Cache::new_ppn` places each layer's allocation on its stage device. Those live allocations are
  visible in that device's free-memory reading.
- Admission nevertheless reads only the primary/head engine (`effective_free_bytes(&engine)`),
  while its analytic `cost` is the aggregate KV geometry for all stages. An aggregate cost checked
  against primary free is conservative for the primary, but says nothing about a tighter remote
  stage. The primary check therefore cannot prove that the next stage-0 KV allocation plus its
  simultaneous wave transient fits.
- Both boundary slots are persistent receiver allocations. Serial PP with overlap enabled also
  alternates through both slots, so describing one slot as dual's complete incremental peak was
  insufficient. Dual's first tick does eagerly prepare both slots before either wave runs.

Verdict: under-counted. The dual-only admission plan must check every distinct PP device using its
exact stage-local incoming KV geometry, a conservative fixed residual plus the existing transient
reserve per simultaneous stage walker, and both prepared f32 boundary slots on the receiving
stage. Multiple stages on one physical device must aggregate their requirements. The
`MEMRA_DUAL_PP=0` serial policy must keep the existing primary-device `cost + reserve` equation
unchanged. The PRO-pair c=64 and teeth cells remain the runtime proof; this section establishes the
off-GPU ownership bound only.

Implementation:

- `memra-kv` now exposes the allocator's existing bytes/token and ring-capped geometry by layer
  range. The aggregate helpers delegate to those range helpers, so PP admission and cache
  allocation cannot drift onto independent dtype/layout math (`memra-kv/src/lib.rs:353-397`).
- The worker partitions the incoming cache at the resolved PP fence, assigns trailing MTP/NextN
  and speculative scratch to the last stage exactly as `Cache::new_ppn` does, and asserts that the
  two stage context allocations sum to the aggregate context allocation (`worker.rs:1944-1994`).
- A dual plan charges each stage its local context allocation, the conservative fixed residual,
  and one existing transient reserve; the receiving stage also charges both f32 boundary slots.
  Requirements are combined when both stages map to one device (`worker.rs:1878-2014`).
- Admission reads `free + pool_cached` and pool occupancy from every distinct stage engine and
  defers if any device misses its own requirement. Parked-session reclaim refreshes the complete
  device set after every eviction (`worker.rs:3463-3609,5211-5302`). Serial policy builds no stage
  plan and retains the prior primary-only `cost + reserve` arithmetic.
- Focused off-GPU checks pass: `cargo test -p memra-server dual_pp_` (6/6),
  `cargo test -p memra-server admission_` (9/9), and `cargo test -p memra-kv` (4/4 plus doc tests).
  Receipts: `raw/local/cargo-server-dual-admission-rerederive.log`,
  `raw/local/cargo-server-admission-per-device.log`, and `raw/local/cargo-kv-stage-partition.log`
  with matching `.exit` files.

### Blocker #5 — host-bounce resolves to serial before dispatch

- `resolve_decode_chunk_policy` now takes the active transport into account. Even with dual,
  overlap, and PP-2 explicitly ready, `MEMRA_PP_HOST_BOUNCE=1` resolves `dual=false` at policy
  construction (`worker.rs:6924-6934,6975-6989`).
- The resolved policy keeps the one-wave tick cap and produces `wave_mid=None`; the worker calls
  serial `decode_step_batch_ppn` and never enters the dual body's host-bounce refusal. Live rows
  therefore degrade to serial instead of being retired through the batch `Event::Error` path.
- `cargo test -p memra-server dual_pp_scheduler_` passes 3/3, including an explicit bounce-host
  cell that pins the serial policy, serial tick cap, original row order, and absent midpoint.
  Receipt: `raw/local/cargo-server-host-bounce-policy.log` and matching `.exit`.

### Blocker #1 — production-scale boot peerprobe rung

- The merged production probe already runs before weight upload through the real stream-ordered
  `BoundarySlot` TX/RX path in both directions, pre-poisons the destination, reads the result back,
  and compares every byte. Native mismatch refuses startup; host-bounce diagnostics tear temporary
  peer/pool access down before fallback serving proceeds (`pp.rs:1560-1848`).
- Its ladder now ends at `PRIME_CHUNK_MAX_TOKENS` instead of 16 rows:
  `[1, 8, 16, 4096]`. For Step-3.7 (`n_embd=4096`) the final transfer is exactly 67,108,864 bytes
  (64 MiB), the same maximum payload geometry as a production prime handoff. Boot logs derive the
  ladder from the constant and retain `largest_clean_payload_bytes`.
- `cargo test -p memra-engine host_bounce_tests --lib` passes 8/8. The merged corruption test now
  pins the full ladder, the 64 MiB Step geometry, and the >=1 MiB floor while retaining the native
  fail-closed / host-bounce-proceed decision. Receipt:
  `raw/local/cargo-engine-peerprobe-large-rung.log` and matching `.exit`.
- This is off-GPU source/geometry proof, not a claim that the PRO pair's peer link is clean. The
  orchestrator boot must show a PASS in both directions with
  `largest_clean_payload_bytes=67108864` before enabling dual on the Step-3.7 shape.

## Post-review local 5090 exactness

Tested source: `a7e5cd00ae304f9a81e51ed322c529345f543261`

One `/tmp/memra-gpu.lock` hold covered fresh release builds, all focused tests, and both model
exactness gates on the local NVIDIA GeForce RTX 5090 Laptop GPU. This is development exactness,
not PRO-pair promotion or performance evidence.

| Gate | Result |
|---|---|
| Fresh release `memra-server` build | PASS |
| Fresh release `run-gen` / `run-spec` build | PASS |
| `cargo test -p memra-server dual_pp_` | 7 passed, 0 failed |
| `cargo test -p memra-server admission_` | 9 passed, 0 failed |
| `cargo test -p memra-engine dual_pp_ --lib` | 4 passed, 0 failed |
| `run-gen`, 64 generated tokens | prefill/decode argmax MATCH; batched-prime/tokenwise argmax MATCH |
| `run-spec`, 32 generated tokens | K=1..8 self-consistency PASS (8/8) |

Primary receipts are under `raw/local/exactness-post-review/`: source/base metadata, release binary
and pinned model hashes, before/after GPU state, raw build/test/model logs, matching `.exit` files,
and `verdict.txt`. `raw/local/exactness-post-review-attempt1/` retains an incomplete harness
attempt: the server build passed, then a zsh/Bash receipt-helper mismatch stopped the harness before
any test or model gate; `STATUS.txt` quotes the failure and points to the complete rerun.

## Historical pre-merge admission-residual analysis (base `110376f8`, superseded)

### What admission currently proves

- `AdmissionCostModel` charges the request's exact context-linear cache geometry plus a learned,
  monotonic fixed high-water residual (`crates/memra-server/src/worker.rs:608-681`).
- For a non-first session, admission reads effective free bytes and requires
  `free >= cost + reserve`; otherwise it evicts reclaimable parked entries and then queues the
  request FIFO (`crates/memra-server/src/worker.rs:3301-3405`).
- The reserve is the full 1.5 GiB floor for spec-capable admission and `min(cost, floor)` for the
  plain path (`crates/memra-server/src/worker.rs:1863-1876`). This rule was calibrated for the
  serial, single-wave transient class; it has no decode-schedule input.
- The F5 right-size ladder shrinks a speculative allocation toward the request-owned `need` and
  probes `SPEC_SHRINK_RESERVE` before accepting a newly learned landing size
  (`crates/memra-server/src/worker.rs:5660-5755`).
- The step-OOM recovery path parks and requeues only an un-emitted speculative session, with a
  bounded retry count (`crates/memra-server/src/worker.rs:3968-4005`). The plain batched decode
  loop reports one batch-step error to every row and retires them (`crates/memra-server/src/worker.rs:4376-4380`),
  so admission must prevent a dual-active plain-decode OOM rather than rely on park/requeue.

### What dual-active changes

- The worker treats the exact numeric width as a per-wave cap and can combine two such waves in
  one tick (`crates/memra-server/src/worker.rs:6584-6603`). The arm is currently selected only
  when dual PP, overlap, and a ready PP-2 path are all explicit (`crates/memra-server/src/worker.rs:6671-6682`).
- The engine keeps each wave at or below that same cap, prepares both boundary slots, then overlaps
  stage 0 of wave B with stage 1 of wave A (`crates/memra-engine/src/decode_batch.rs:737-762,796-833`).
- Boundary buffers are persistent receiver-side allocations, and the shared boundary atomic gives
  the two concurrent callers distinct slots (`crates/memra-engine/src/pp.rs:609-629,1134-1154,1182-1189`).
- KV ownership and allocation do not double: every request already owns its cache, and PP-N places
  each layer's cache on its owning stage (`crates/memra-engine/src/pp.rs:1409-1440`). Likewise,
  per-device compute scratch does not become two full waves: during the overlap, stage 0(B) and
  stage 1(A) run on different stage devices and each remains bounded by `wave_cap`.

### Verdict

The current bound is insufficient for promotion, but the missing term is not a second session and
not a second 1.5 GiB transient floor. Relative to the serial single-slot walk at the same exact
wave width, dual-active's additional durable peak is one receiver-side f32 boundary residual:

```text
dual_extra_bytes = wave_cap * n_embd * sizeof(f32)
required          = cost + reserve + dual_extra_bytes
```

The existing reserve continues to cover one cap-bounded wave's activation/scratch class. Charging
another whole reserve would conflate cross-device concurrency with same-device memory pressure and
would reduce admission without matching the allocation lifetime. The extra term must be zero under
the serial rollback seam and nonzero only when the model's resolved decode policy is dual-active.

This is a source-level bound. The PRO-pair c=64 run must still validate the allocator/runtime
assumption against real high-water behavior and compare step-OOM/deferral receipts with the serial
baseline.

## Historical pre-merge admission-bound implementation (superseded)

Source commit: `2d88a6ad3545aa0a717d2b502eda9efb1b86c878`

- `dual_pp_admission_residual` computes one overflow-safe
  `wave_cap * n_embd * sizeof(f32)` receiver slot and returns exactly zero for a resolved serial
  policy (`crates/memra-server/src/worker.rs:1883-1890`).
- `admission_required` saturating-adds the request cost, existing transient reserve, and the dual
  residual (`crates/memra-server/src/worker.rs:1892-1894`).
- Admission derives the term from the already-resolved per-model decode policy, logs its geometry
  only for dual mode, and uses it in the defer threshold (`crates/memra-server/src/worker.rs:3249-3288,3378-3385`).
  Serial mode retains both its old arithmetic and its old defer diagnostic.
- Focused tests pin the f32 slot geometry, serial-zero behavior, exact old serial equation, dual
  equation, and overflow saturation (`crates/memra-server/src/worker.rs:7830-7853`).

## Historical pre-remediation local 5090 exactness (source `2d88a6ad`)

One `/tmp/memra-gpu.lock` hold covered the complete evidence block on the local NVIDIA GeForce RTX
5090 Laptop GPU. The source, release binaries, model, drafter, prompt, before/after GPU state, raw
stdout/stderr, and exit receipts are under `raw/local/`.

| Gate | Result |
|---|---|
| Fresh release `memra-server` build | PASS |
| Fresh release `run-gen` / `run-spec` build | PASS |
| `cargo test -p memra-server dual_pp_` | 5 passed, 0 failed |
| `cargo test -p memra-server admission_` | 8 passed, 0 failed |
| `cargo test -p memra-engine dual_pp_` | 4 passed, 0 failed |
| `run-gen`, 64 generated tokens | prefill/decode argmax MATCH; batched-prime/tokenwise argmax MATCH |
| `run-spec`, 32 generated tokens | K=1..8 self-consistency PASS (8/8) |

Primary receipts:

- `raw/local/build/memra-server.log` and `raw/local/build/exactness-binaries.log`
- `raw/local/exactness/source.txt` and `raw/local/exactness/SHA256SUMS`
- `raw/local/exactness/cargo-server-dual.log`, `cargo-server-admission.log`, and
  `cargo-engine-dual.log` with matching `.exit` receipts
- `raw/local/exactness/run-gen.log` and `run-spec.log` with matching `.exit` receipts
- `raw/local/exactness/nvidia-smi-before.log`, `nvidia-smi-after.log`, and `verdict.txt`

These are development exactness receipts, not PRO-pair promotion evidence and not performance
measurements.

## Remaining PRO-pair increment

1. From this committed source, run serial and explicit-dual c=64 `tools/serve-stress-gate.sh`
   cells on the PRO pair, including the teeth control, with raw server/client/metrics/thermal logs.
   Dual must complete cleanly and must not add step-OOM park/requeue thrash over the serial count.
2. Re-run PP split bit identity plus the live one-hash matrix at c=1..8.
3. Run the x100 cross-device event-ordering soak; require balanced slot use, zero collisions, and
   identical completion hashes.
4. Only if every gate above is green, flip eligible Step PP-2 dual-active decode to default ON,
   retain `MEMRA_DUAL_PP=0` as the serial rollback, then run the N=5 interleaved naked-default vs
   rollback A/B on the same lock hold. A failed re-gate leaves the lane at HOLD with no flip.
