# HostPrefixCache extension — day-2 CPU boundary

Repository `avifenesh/memra`, `lane/spill-b-20260919`. Frozen contracts inherited from
`259ff819f1efab8032695bf4e6b5313f86258ef0`. This is a concrete patch plan plus an
inactive identity sidecar, **not** online active/prefix NVMe serving qualification.

## First slice actually implemented

`crates/memra-kv/src/tiered/hostprefix.rs` supplies `IdentitySlot`/`IdentityLease`.
`crates/memra-server/src/worker.rs::HostPrefixEntry` holds one default-unbound slot,
through the existing `memra_engine::cache` re-export. Both production constructors
(`host_entry_from_device`, `host_entry_from_owned`) and the CPU test constructor
initialize it unbound. There is no new dependency in the server manifest.

An unbound slot cannot issue a generic tier identity lease. Binding requires a frozen
immutable-prefix bundle, exact tokens, full ProgramIdentity, expected native layout,
committed high-water and the same model-generation `Arc`. Rebinding/drop invalidates
old leases. A new model-generation Arc refuses old metadata even if token ids agree.
Two CPU tests exercise legacy refusal, token/salt mismatch, model replacement,
rebinding, removal/purge-style drop and untrusted handoff. They compile in memra-kv;
they are not a substitute for compiling/running the complete GPU-dependent server.

No runtime caller invokes bind yet. The slot's empty default performs no allocation,
copy, env read, policy branch, payload change or persistence-version change. Existing
host lookup/promotion remains the old synchronous API. A metadata lease does NOT
retain host payload/DMA memory; the next adapter MUST add the actual host-image lease
and shared-governor pin before it can submit asynchronous work. Do not mistake this
sidecar or a passing identity test for the full HostPrefixCache implementation.

## Exact next patch map (baseline worker.rs line numbers before the six-line sidecar)

| Existing owner/function | Next change / invariant |
|---|---|
| `HostPrefixCache::lookup` (7597), `key_index` (7613), `host_promote_candidate` (8538) | Keep PoolKey `(model, PC-ISO namespace)` and exact-token longest-prefix selection. Attach a fully checked bundle to a selected entry only after complete native capture. Lookup stays advisory; never cache vector indices across ticks/swap_remove. |
| `host_entry_from_device` (7862), `host_demote_prefix_entry` (8069), `host_demote_prefix_ref` (8135) | Capture all existing image planes and counters first. Build group-specific layout and checksum actual valid bytes. Physical immutable copy precedes `seal` (epoch 0); then bind sidecar. A failed capture leaves existing demotion/failure-latch behavior, not a partial root. |
| `HostPrefixCache::insert` (7670), `remove_at` (7640), `purge_tenant` (7763) | Preserve tenant delta-cap check before exact-key replacement, LRU index repair, byte accounting and allocation-failure latch. Pin source images by stable id/generation, not position. Invalidate identity before removal becomes visible. Purge cancels pending tickets immediately but holds payload/governor pins until physical retirement. The current sidecar Drop handles metadata revocation only. |
| `HostPrefixCache::reserve_image` (7490), `take` (7796), `worker/host_glm.rs::HostGlmState` | Use A's already-charged fixed arena backing and per-operation slot/pin reservations; don't charge fixed backing twice. GLM/rank/window/recurrent images need their own complete mappings; no automatic generic v1 import. |
| `host_promote_prefix_hit` (8558), `device_entry_from_host` (8407) | Preserve existing generation check, byte copies, digest comparison, native PrefixEntry construction and retained host twin. Move scheduling around these operations only after tickets and native consumer fences are wired; do not change their numerical program. |
| `admit_memory.rs::Tiers`, `decide` (246), `demote_budget_bytes` (285) | Keep current live free-memory/reclaim verdict and bounded demote amount. Add governor reservation after an advisory verdict; a verdict is not a capacity lease. Maintain mandatory active headroom and release deferred/cancelled admission only after all physical uses. Do not add a second allocation formula. |
| Admission/decode tick hooks (`worker.rs` 14359, 14624, 15310, 19252–19268, 19931–19940, 20395) | Store `TierAdmissionPlan` with reservation, frozen policy and current owner epochs. Advance nonblocking before suffix prime/attention. Requeue WAIT; timeout may choose cold only before admission with same-program proof. Mandatory state is never converted into a prefix miss. |
| `host_handoff_export` (9662), `host_handoff_import_step` (9823), `host_entry_from_owned` (9549) | Keep the existing whole-image handoff format/version and model-stat validation unchanged. Imported legacy entries stay identity-unbound. A future generic restart format needs explicit new wire identity and full auxiliary mapping, not reinterpretation of old bytes. |

## Lookup → admit → prefetch → load → fence → ready

1. **Lookup:** existing cache selection plus full frozen identity/layout compatibility;
   search local GPU, eligible directed peer, pinned host, local NVMe. No quota credit.
2. **Admit:** `Hierarchy::admit` reserves all physical/quota dimensions through the
   single injected `BudgetGovernor`, then `KvBacking::acquire` atomically retains the
   actual selected source. Source identity/layout/epoch/tier are rechecked. Whole
   native operand capacity must fit; one bounded staging slot may be smaller.
3. **Prefetch:** native `KvBacking` creates A's shared `TransferOp` descriptors in
   layout order. Homogeneous `{state,src_gen,dst_gen}` per ticket. Rejected descriptors
   return to native owner cleanup; accepted subsets remain owned and cannot publish.
4. **Load:** after all host bytes/checksums complete, prepare exact H2D/P2P descriptors
   for stable existing attention planes. No chunk-wise softmax, requantization, or
   alternate activation/KV program. HostReady is explicitly not device-ready.
5. **Fence:** `Completion::require` checks every ordered item/segment; on consumption
   `TransferEngine::ready_view` additionally proves destination owner/context/generation.
   A partial ready-view failure exposes no operands and quarantines the entry. No
   successful sibling may be used as a partial promised restore.
6. **Ready:** `Hierarchy::ready` returns a borrowed BlockLease retaining SOURCE
   provenance/charge; the actual target permission is a sealed transfer ReadyView.
   Existing native restore/suffix-prime or active attention may run only after complete
   auxiliary state and current program/epochs are checked. Publication is distinct from
   last-use retirement. Cancel after publication does not revoke an active consumer.
7. **Retire:** disk/DMA/consumer/graph completion must all be observed. Unknowns keep
   source, destination, governor and backend owners quarantined, including shutdown.
   Explicit release acknowledges tickets and credits charges; Busy retains retry handles.

`KvBacking<T>` is the B-owned native mapping seam using frozen descriptors, NOT a
second ObjectStore or TransferEngine. The day-2 adapter is exercised with CPU fake
movement. Mapping A ObjectStore roots to real host images, stable source leases and
native CUDA events is pending. Local/peer direct movement remains a permitted native
adapter optimization, not a implemented/readiness claim from these tests.

## Native materializer boundary

`materializer.rs::NativeGeometry::from_layer` reads the EXISTING `KvLayer` geometry:
`kv_dim_k/v`, `k_tok_bytes/v_tok_bytes`, `len`, and `ring`. Actual binding of
`k`, `v`, and stable `len_d/base_d` counters is still native owner work.
It enforces q8_0 K at 34 bytes/32 elements and q5_1 V at 24 bytes/32 elements.
The CPU `QwenMaterializer` implements the frozen trait and resolves a retained
`NativeKvImage` through DeviceOwner, checks full bundle/program/epochs, and returns
non-clonable operands retaining the same allocation. Capture concatenates native
role pages in exact token/head/dim order, with no math or codec. Tests cover one
record and multi-page three-token history with odd final page, padding, wrong
program/format, wrong order, capacity and foreign materializer retirement.

This is an **attention-layer operand adapter**, not a complete Qwen continuation.
Qwen recurrent/conv/SSM/logits/hidden/draft state must stay resident or join the full
HostPrefixCache bundle; it cannot disappear because the K/V fixture passes. Ring,
trailing-only, alternate encoding and partial dense history refuse. CUDA binding
must use KvDev/native copy operations into the existing KvLayer planes, preserve
stable len/base addresses, and register the actual last-use events. No CUDA binding
or engine module export was added on the Mac; real active reload during generation
and graph-stable addresses remain blocking B2/B3 cells.

## Recompute/load policy and fixture table

Pure policy only, no runtime default. The old linear diagnostic is retained as a
non-calibrated primitive. `calibrated_recompute_vs_load` uses measured
`(prompt_tokens,reused_tokens,cold_prefill_ns,suffix_prefill_ns,read_ns,copy_ns,
materialize_ns)`. Load only when `suffix + read + copy + materialize < cold`.
Equivalently, positive I/O budget is `cold - suffix - copy - materialize`; tie
recomputes. Costs must bind binary/plan/device/artifact/tier before runtime use.

Synthetic ns fixtures (NOT measurements):

| cold | suffix | read | copy | materialize | State / proof | Expected |
|---:|---:|---:|---:|---:|---|---|
| 1000 | 200 | 300 | 100 | 100 | optional, before admission, same program | Load (700 < 1000) |
| 1000 | 200 | 300 | 100 | 400 | same | Recompute (tie) |
| 1000 | 1100 | 300 | 100 | 100 | same | Recompute (negative saved-work budget) |
| any | any | any | any | any | mandatory active | RequireState, including timeout |
| 1000 | 200 | 300 | 100 | 100 | optional but already admitted | Busy refusal; do not discard reserved state |
| any | any | any | any | any | no same-program cold proof | Load; never invent a cold fallback |
| 1000 | 200 | u64::MAX | 100 | 100 | optional pre-admission | Overflow refusal |

Chunk sizes 64/128/256/512, real long-context costs and route preference changes remain
unmeasured. Nothing in this CPU fixture table selects a shipping policy.
