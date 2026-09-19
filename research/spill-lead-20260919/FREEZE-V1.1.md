# Generic-spill freeze revision v1.1 — lead draft

Date: 2026-09-19. Repository **avifenesh/memra**.
Branch: `lane/spill-lead2-20260919`, worktree `../wt-spill-lead2`.
Base: `98e558dc` (frozen v1 plus A/B/C/D day-2 integration).
Checked source: **`60ceca2e52cd8cfadb93f25fa32d4f86184a70eb`**.

**Local committed draft; not main-integrated, pushed, tagged, deployed or GPU-qualified.**
The full CPU test run is intentionally **RED: two lane fix items below**. Neither is
ignored or marked `should_panic`. This is a conformance delivery, not a claim that the
implementation backlog or native G0–G7 qualification is complete. The worktree stays
open for the lead handoff, not abandoned.

## Decision and unchanged semantics

Add reusable schedules, test-only factory/observation hooks, and additive lane bindings.
No runtime source was edited. `contracts.rs`, the canonical fixture bytes, root manifests
and lockfile are byte-unchanged from `98e558dc`; the receipt checks this directly.
No persisted struct changed: **WIRE_VERSION remains 1**, and the eight payload plus
three canonical-wire pins still match. “v1.1” names the conformance revision only.

No new external dependencies, env reads, numerical programs, kernel/backend defaults,
model code or GPU claims. No main/WP worktree edits; no push/PR/tag. Worktree creation
succeeded without an index-lock repair. The test reference governor's loader cap is now
initialized alongside its other configured fake caps; this is not a production policy.

Commits:

- `50b6b829`: `test(tier): add v1.1 generic spill conformance schedules`
- `60ceca2e`: `test(tier): bind spill lanes to stricter lifecycle schedules`
- Subsequent documentation/receipt commit adds no Rust changes; use `git log -1` for
  the final handoff tip. The exact tested Rust commit is pinned above and in commands.json.

## Shared schedules and cases

Owning source: `crates/memra-tier/tests/contracts/revision_v11.rs`, re-exported by the
existing `conformance.rs`. Old six schedules remain available unchanged. Hooks are
fixture control/observation APIs, not additions to the frozen runtime traits.

| New schedule(s) | Cases and actual binding |
|---|---|
| `pinned_lease` | Uninitialized bytes refuse; metadata/alignment shape; two slots with one demand reserve; optional/full refusal; one backing charge; producer-to-consumer ownership move; Busy backing release; explicit slice release then pool close and zero charge. A pool. |
| `pinned_quarantine` | Slot never returns to free list; Busy while quarantine exists; no quota credit from unknown last-pool shutdown. **Fails A at shutdown**, not at ordinary slot retention. |
| `budget_governor` | Shared trait-generic headroom/mandatory reserve; refusal atomicity; all 12 scalar/device-vector ceiling cells; overlapping physical/peer/replica accounting; pins do not double-charge; foreign/double release; InUse/Quarantined refusal; explicit retired release to zero. Reference Governor and **B Governor**. |
| `governor_priority` | Five priorities enqueued out of order; assert actual dispatched IDs in frozen order; release each capability. **B enqueue/dispatch**, not a sorted fixture vector. |
| `peer_capacity` | Non-peer occupancy in the SAME injected governor; device/peer counters; undercharge/alignment/stale/overcapacity refusal without charge; foreign release; two Busy retries on one borrowed handle; graph-capability pin; old-state release after pins retire; double release and exact zero. D fake. |
| `peer_directed_grants` | Denied forward context and pool grants must not deny a configured-good reverse route; downgrade refuses forward without requiring reverse physical health; restoration admits again; no charge on denial. Complete reference fake passes all arms; **D's first directed-context cell fails**. |
| `object_publish_release` | Post-commit cancel is AlreadyPublished; root survives; lease/read/Busy eviction; explicit and double release; released read refuses. Reference plus A actual filesystem. Existing `object_cancel` covers put-complete/root-unpublished cancel. |
| `transfer_complete_cancel`, `bank_complete_cancel`, `rows_complete_cancel`, `peer_complete_cancel` | Explicit hook then observed producer completion before cancel; cancelled publication refuses. Transfer/peer validate Completion against expected bytes/checksums; bank/rows observe their actual nonempty completion vectors. A, C, D, reference fixtures. Existing `tier_cancel` and `tier_identity` explicitly reach Phase::Ready before cancellation on B. |
| `transfer_lifetime`, `peer_lifetime` | Foreign issuer, unknown, producer completion without retirement, recovery, delayed consumer, delayed graph, failed acknowledgement retains retry; explicit ack makes ticket Unknown. Reference transfer and D peer. Usage remains charged across partial retirement. |
| `bank_lifetime`, `rows_lifetime` | Generic unknown/publication refusal and delayed retirement hooks. Reference control-flow fixtures only: their stage reserves no backing before publication, so their zero-ledger assertion is **not** a measured charge-retention proof. C's synchronous host implementation has no native unknown-event/graph hook. |
| `tier_identity`, `tier_unknown`, `tier_miss` | Every one of ten ProgramIdentity digests independently rejected at admission/ready; parent/high-water/missing plane refusals; independent stale ready epochs; Ready-before-cancel; unknown retains actual B charges through recovery/retirement; advisory absent lookup vs failed mandatory admission. B Hierarchy. |
| `acceptance` | Original three indices; accepted/short/rejected with per-segment counts/checksum/epochs; reject 0 and 1 separately; test original last item, not compacted accepted indices. Reference transfer and D byte fake. |
| `transfer_zero_accept`, `peer_zero_accept`, `peer_submit_epochs` | Empty/all-rejected return owned inputs; actual host/source ownership checked; each stale state/src/dst at peer submit; heterogeneous triples refuse without entries; separate homogeneous submissions succeed with distinct tickets. Reference transfer and D. A's existing mixed/partial ownership tests still run. |
| `bank_identity`, `rows_identity`, `bank_uniform` | Artifact/tensor/original-ID/layout refusal; stale publication epochs; actual C catalog source class; Uniform acceptance and single homogeneous PerRecord subset refusal with returned IDs/ownership; delayed release and explicit retirement. C. |
| `rows_bytes`, `bank_failed_sibling`, `rows_failed_sibling` | Actual opaque row bytes, duplicate capability IDs, stale publication, explicit post-retirement release; 51 logical rows/48 unique at 264 bytes, 4KiB slots, four reads/three straddles, output larger than slot; corrupt bank siblings and failed second physical row read refuse all publication. C. Scale-bearing bank layouts retain C's macro-scale planes; existing scale/missing-plane tests still run. |
| `bundle_planes` | Every required sibling removed/corrupted independently, including the shared all/trailing/alias envelope and B native q8_0/q5_1 byte bundle. Shared wire fixture and B. |
| `bundle_integrity` | Reusable StateBundleAdapter missing-plane/high-water restore refusal. **Not bound/run**: no implementation of that trait exists in this integrated tree. This is an open adapter gate, not a pass. |

The reference directional fake is deliberately distinct from D's implementation: it
proves all route-fault branches of the schedule can run; it cannot qualify D's capacity.
Likewise fake graph booleans prove ordering/assertions, never CUDA address stability.

## Per-lane findings and required fixes

### A — ordinary pool/object/transfer bindings pass; quarantine shutdown FAIL

Binding: `crates/memra-tier/tests/storage/day2.rs`, test
`revision_v11_pinned_quarantine`.
Assertion: **`crates/memra-tier/tests/contracts/revision_v11.rs:215`**:

```text
assertion `left == right` failed
  left: Ok(())
 right: Err(Busy)
```

Sequence: acquire → quarantine → verify slot unavailable and charge Busy → release
other idle slice → drop last pool → explicit backing charge release unexpectedly succeeds.
Implementation: `crates/memra-tier/src/pool/mod.rs:206–215` stores quarantined slots
inside `Inner`; `Inner` (lines 34–38) has no unknown-shutdown retention/drain guard.
Dropping the last pool drops the backing and its LeasePin. This violates the frozen
unknown-retirement requirement, not ordinary idle slice Drop behavior.

**A fix item:** retain/quarantine actual backing and its governor pin through unknown
pool shutdown; release only following a genuine all-use retirement/recovery proof.
Do not weaken the schedule or simply mark the charge retired while the use is unknown.
Native pinned/DMA implementation remains a separate qualification surface.

### B — PASS in CPU scope

`crates/memra-kv/src/tiered/tests.rs`: new generic governor, actual dispatch priority,
complete-before-cancel identity/planes, and unknown/missing-backing bindings all pass.
The existing 40 tests remain; 4 added = **44 passed**. No B implementation fix observed.
Service fairness, dirty backlog under load, real graph addresses and native adapter
semantics are not inferred from these CPU cells.

### C — PASS in CPU scope

`crates/memra-tier/tests/bank/main.rs`: new complete-before-cancel bank/row,
identity/uniform, row-byte/dedup/bounded-straddle, corrupt-sibling and localized read-fault
bindings pass. Existing 27 plus 4 new = **31 passed**. No C implementation fix observed.
No fake GPU completion was added to C's host-only backend.

### D — ordinary capacity/lifetime/acceptance/epoch bindings PASS; directed capacity FAIL

Binding: `crates/memra-tier/tests/peer/fake.rs`, test
`revision_v11_directed_capacity_lane_fix_required`.
Failure: **`crates/memra-tier/tests/contracts/revision_v11.rs:567`**:

```text
denying one directed route must not deny its reverse: Unsupported
```

D's existing fixture has only `grants: bool` at
`crates/memra-tier/tests/peer/fake.rs:28`; `PeerCapacity::reserve` checks it globally
at lines 131–133. A forward context denial therefore denies the reverse too. The fake
also has no separate pool-grant/link-health injection. Its separate topology tests pass,
but that does not bind those route decisions to actual capacity admission.

**D fix item:** connect directed context/pool/live-route state to PeerCapacity admission;
add route-specific fixture controls for each fault. No host-bounce substitution. Retain
one injected governor and Busy retry ownership. This is a CPU fake/admission-binding
finding, not evidence that any actual P2P route failed.

## Disposition of D's nine proposed additions

1. **Capacity:** added factory/hooks and actual D governor binding; strict directional
   failure retained. Physical retention/pin/old-state/zero accounting pass.
2. **Async timing:** added explicit complete-before-cancel plus unknown/delayed-retirement
   schedules. Existing pending/publish-before-cancel tests remain. Bank/row native pending,
   real consumer/graph, and A unknown async driver hooks stay unrun until those backends
   exist. StateBundleAdapter returns a ticket but has **no cancel/poll/publish API**;
   adding cancellation there would change the frozen trait and remains a lead decision.
3. **Acceptance:** generic three-index matrix added and run on reference+D, including
   reject-first/reject-middle and returned ownership. Existing A ownership cells retained.
   Actual short I/O from a native async driver is a hardware/adapter gate, not manufactured.
4. **Epochs:** generic stale-at-publication, D stale-submit/full-return/heterogeneous/split
   cases, foreign issuer and acknowledgement added. Bank/row stage has only one Epochs
   argument, no external current-generation authority; an arbitrary new triple may be
   legitimate. Requiring rejection of every changed stage epoch would invent semantics.
   Native source/destination authority binding remains open, rather than imposing it on
   source-only host banks. No heterogeneous bank batch can be expressed by the v1 type.
5. **Identity:** all ten program fields, parent/high-water/planes and bank/row namespaces
   are exercised. Byte-only peer has no invented program schema.
6. **Bundle/siblings/misses:** all/trailing removal and corrupt siblings promoted to
   shared helpers; B/C bindings pass. `tier_miss` tests advisory miss vs failed mandatory
   admission, **not** deliberate loss of live sole-active backing. Its recovery/retention
   policy needs native source lifetime instrumentation; no fallback/drop policy is added.
7. **Rows:** generic bytes/ordered duplicates/charge identity, bounded straddles,
   complete-before-publish cancellation, failed physical read and explicit release run.
   Exact physical I/O counts come from C's actual reader fixture, not a guessed quota.
8. **Uniform:** generic proof acceptance/refusal runs on C's actual Catalog/BankService;
   macro-scale source completeness and engine-shaped adapter tests remain in C's suite.
   Existing shared compile-fail tests retained, no copy of their API expectation.
9. **Fairness/dirty/graphs:** actual B priority dispatch added; existing concurrent B quota,
   fairness and dirty-bound unit tests remain. Timed service fairness, dirty backlog under
   serving load and physical graph-address stability remain unrun native qualification
   items. A trait-generic reserve/release API has no queue instrumentation; test hooks do
   not silently add queue methods to BudgetGovernor.

## Verification receipt

Reproducer: `python3 research/spill-lead-20260919/verify-v11.py`.
Raw outputs: `v1.1/*.log.gz` (lossless stdout+stderr before parsing).
`v1.1/commands.json` records exact argv/exit/raw SHA-256 plus all tier/KV Rust/schema
source hashes at checked source `60ceca2e52cd8cfadb93f25fa32d4f86184a70eb`.
Before baseline is `v1.1/before.log.gz`, run directly on `98e558dc` before edits.
Development red logs are retained separately; Clippy's needless-range-loop was repaired,
not suppressed. No credential or env file was read or copied into these receipts.

| Command | Actually ran | Result |
|---|---|---|
| `cargo fmt --all -- --check` | Yes | PASS |
| `cargo check -p memra-tier -p memra-kv --offline --all-targets` | Yes | PASS on macOS |
| Same check + `--target x86_64-unknown-linux-gnu` | Yes | PASS, cross-compile check only (no Linux execution/link qualification) |
| `cargo test -p memra-tier -p memra-kv --offline` | Yes | exit 101; D known failure stops later targets |
| Same test + `--no-fail-fast` | Yes | exit 101; all targets executed, **172 pass / 2 fail / 0 ignored**, including 4 doctests |
| `cargo clippy -p memra-tier -p memra-kv --offline --all-targets --no-deps -- -D warnings` | Yes | PASS; pre-existing dependency-only macOS unused AsRawFd warning in memra-gguf remains |
| `git diff --check` | Yes | PASS |
| `bash tools/check-flags.sh` | Yes | PASS, 864 runtime literal reads, none uncovered |
| `python3 crates/memra-tier/tests/contracts/fixture_reference.py --check` | Yes | PASS, all 11 pins unchanged |
| `git diff --exit-code 98e558dc --` contracts/fixtures/root manifests/lock | Yes | PASS, no persisted/runtime schema changes |
| Linux runtime/O_DIRECT, engine/server/nvcc, CUDA/graphs, model/serving/G0–G7 | **No** | No GPU/nvcc here; not substituted with CPU conformance |

| Test target | Before passed | After passed | After failed |
|---|---:|---:|---:|
| memra-kv | 40 | 44 | 0 |
| bank | 27 | 31 | 0 |
| contracts | 36 | 44 | 0 |
| peer (includes topology) | 10 | 14 | 1 |
| placement | 6 | 6 | 0 |
| storage | 26 | 29 | 1 |
| tier compile-fail doctests | 4 | 4 | 0 |
| **Total** | **149** | **172** | **2** |

25 tests added, none removed or ignored. CPU crate library target with zero tests and
KV zero doctests are unchanged. No GPU time. Effort approximately **0.9 agent-hours**
for reading, implementation, iterative CPU checks and this receipt; wall-time estimate,
not an asserted service/performance measurement.

## Lead handoff / remaining decisions

Keep the two failing lane tests strict and dispatch A/D fixes in their own lanes. Do not
call this branch integration-ready until they pass and the lead independently repeats
the integrated battery. `bundle_integrity` remains an unbound adapter schedule; freeze
revision does not create a competing native StateBundle adapter or add new methods.
All native qualifiers and the listed trait-authority questions remain explicit; no
wire bump, runtime policy change, model-work authorization or support-state promotion
is needed or granted by this draft.
