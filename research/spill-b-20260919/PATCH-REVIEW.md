# HostPrefix patch v2 — hunk review, day 4

Repository **avifenesh/memra**, branch `lane/spill-b-20260919`. Exact base:
`020d20479cd686835c0fb7743040947d0fc2723b` (integration through day 3).
**UNAPPLIED, NOT COMPILED as engine/server, NOT enabled.** `tier=None` remains
unchanged off-path behavior. No bootstrap, numerical program, environment flag,
CUDA shim or dependency was added. Source review is by B, not independent approval.

`revise-patch-v2.py` reconstructs v1 from Git `40f34e47`, applies it only to deleted
scratch copies of the three base files, edits v2, runs rustfmt, and emits the diff.
`git apply --check` checks applicability only. V1 and its review remain in Git history.

## Every hunk (base file:line → review)

All `worker.rs` below are `crates/memra-server/src/worker.rs`.

| Hunk | Reviewed invariant / change |
|---|---|
| `worker.rs:6121` | Device PrefixEntry guard stays last, so synchronous payload destruction precedes quota credit. V2 guard carries full ProgramIdentity as well as the shared-governor lease. |
| `worker.rs:7391` | Host metadata vector, its separate pageable guard, and image residency guard stay after payload fields. Metadata lease is not mistaken for a DMA payload lease. |
| `worker.rs:7399` | Optional HostTierContext is owner-injected only, contains the SAME governor and canonical per-PoolKey program + model-generation Arc. No prefix-only allocator or auto-enable. |
| `worker.rs:7671` | V2 insert requires guard's complete ProgramIdentity to equal the context and validates the sidecar generation. Legacy unbound handoff refuses only when tier context is injected. Existing identity/version, quota, tenant-delta replacement and LRU flow remains. |
| `worker.rs:7789` | Helpers: source Backup and destination AdmittedRestore charge existing byte estimates through shared governor; fixed arenas refuse until already-charged backing handoff exists. V2 helper passes explicit ProgramIdentity to the CPU-tested guard, which checks tenant before reserving. V2 binding additionally checks existing host `model_generation` Arc equals injected context before sealing identity. Captures native q8_0 K/q5_1 V, recurrent, logits, hidden and presence/order/length transaction metadata; no codec/substitution. Metadata capacity is charged before identity publication (scratch/allocator overhead audit remains pending). |
| `worker.rs:8002` | Successful existing D2H constructor initializes metadata and guards empty. Identity not published before native capture completes. |
| `worker.rs:8191` | Enabled path rejects uncovered TP/latent/draft shapes, computes pinned K/V vs pageable remainder with checked arithmetic, reserves BEFORE existing D2H. Off path performs no new accounting. Failure leaves caller-owned live source. |
| `worker.rs:8204` | Completed immutable host copy receives its guard, binds complete identity, then enters existing insertion. Failure drops payload before credit. Source prefix lifetime remains with its owner. |
| `worker.rs:8494` | Reconstructed device PrefixEntry starts uncharged; promotion assigns guard only on successful completion. |
| `worker.rs:8576` | Existing native generation check preserved. Acquire exact candidate identity; reserve local device plus pageable token/logit/hidden bytes before copy. Synchronous borrow retains actual source; host twin remains independently charged. |
| `worker.rs:8612` | Recheck full identity/model generation before publication and attach the program-bound destination guard. Digest validation and recency-before-insertion discipline unchanged; no stale host index retained through demotions. |
| `worker.rs:9632` | Legacy handoff constructor stays unbound/uncharged, format unchanged. Tier-enabled insertion refuses this legacy import. |
| `worker.rs:10127` | Ordinary committed PrefixEntry producer sets guard None; does not fabricate provenance or claim global accounting covers all producers. |
| `worker.rs:11058` | Second ordinary producer likewise None; draft/auxiliary numerical code untouched. |
| `worker.rs:30793` | CPU PrefixEntry test constructor initializes guard None. |
| `worker.rs:33250` | Budget/LRU test constructor initializes guard None. |
| `worker.rs:33289` | Host test constructor initializes guard and metadata fields; existing test inputs preserved. |
| `crates/memra-server/src/admit_memory.rs:721` | New helper requires explicit ProgramIdentity and calls `ResidentCharge::reserve_for_program`. Existing advisory memory verdict/retry/timeout arithmetic unchanged; advisory approval is not a lease. |
| `crates/memra-server/src/worker/host_glm.rs:533` | Cross-file test constructor gets required guard None field. No GLM binding enabled. |

## CPU proof for v2 helper

`program_residency_refuses_cross_tenant_before_charging_and_retains_full_identity`
checks mismatch before quota changes, identity retention and credit on synchronous
payload-owner release. This does NOT compile worker/admit_memory/host_glm or prove
native allocator byte accounting. Existing source/destination/host-twin guard tests
remain in the memra-kv suite.

## Runner order and actual verification boundary

`rig-cells-b.sh` → `rig-cells-b.py`, checked by `test-rig-cells-b.py`:

1. Scratch worktree from committed exact lane tip; `git apply --check`.
2. Baseline release engine/server/run-gen build; full server lib tests; existing
   prefix identity → teeth → failure gates.
3. Apply patch in scratch; patched release build/server tests; same existing gates.
4. Baseline then patched **8k and 32k fitting** raw-token run-gen cells, using their
   separately built binaries. Immutable artifact/tokens and source-bound fit envelope
   required. Missing/insufficient envelope refuses without format fallback.
5. Native active/prefix full-state gate remains absent; runner exits nonzero for live
   qualification rather than calling legacy-prefix success an active-tier result.

Raw stdout/stderr is teed to a file BEFORE JSONL/hash parsing. Build is outside the
GPU lock; probes use `/tmp/memra-5090.lock`; legacy gates own that lock internally
(no nested flock). Existing box-global server-kill teardown makes exclusive approved
non-serving authorization mandatory. 250ms telemetry and failure compute-app capture
remain. CPU dry-run tests assert the new exact order, raw hashes, and injected exit 7.
No live command in this recipe ran here.

## Blocking review items before runtime application/enablement

1. Lead's canonical artifact/plan/numeric/stream/tokenizer/template/tenant bootstrap
   and per-model generation lifecycle are absent; applying patch alone does not engage.
2. Full native shape agreement, allocator capacity/metadata and bind-time scratch
   accounting need audit. Fixed-arena, GLM, TP and draft handoffs still refuse. This is
   not a generic persistent wire migration or a complete global-allocation conversion.
3. Synchronous guard is not asynchronous retirement. CUDA producer/consumer/graph
   events, native allocation/error behavior and complete continuation need rig tests.
4. Generic active scheduler/materializer server binding, host/NVMe engagement,
   long contexts and serving numeric crossings remain unimplemented/unrun here.
5. Native engine/server build requires nvcc; rustfmt/apply checks cannot rule out
   compiler/API/borrow defects. Independent lead review plus target gates are required.
