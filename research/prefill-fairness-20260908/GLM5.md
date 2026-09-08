# GLM5 prime walker: phase 1 mechanism and design

2026-09-08. Static inspection only. No cargo, gate, server, GPU, rental, or box access.
Code baseline: `72aa777c3763a21281fb9c1c55b2499f2c14d1ee`, freshly fetched
`origin/main`. Every Memra source citation below refers to that exact commit.
Paths are repository-relative; `worker.rs` means `crates/memra-server/src/worker.rs`,
`hybrid_forward.rs`, `glm_spec.rs`, `glm5_tp.rs`, `glm5_tp_sym_graph.rs`, `tp_ar.rs`,
and `mla_ffi.rs` mean those files under `crates/memra-engine/src/`.

Authority: owner `GLM5-HANDOFF.md`, read in full from the local Relay receipt
`prefill-fairness-20260908`. Shared design read from
`lane/prefill-fairness-20260908:research/prefill-fairness-20260908/MECHANISM.md`
in that lane's worktree, also citing this baseline. The seam commit and exact trait
signature are pending orchestrator handoff. This document proposes one GLM5 adapter;
it does not define a competing scheduler or trait.

Door: shared `MEMRA_PRIME_YIELD`, default OFF, decide-by: 2026-09-22.
OFF drives the walker to completion. ON returns after one existing chunk to the
shared drain/admit, bounded rotating peer-service, resume policy. Both execute the
same frozen tape. No global prefill budget, decode-first reorder, or concat-prime
partition. The shared FLAGS row belongs to the seam owner; later this route adds
its receipt pointer. Rollback is unset plus restart. This document does not flip it.

## 1. Prime paths and current boundaries

**Cold speculative prime.** `step_glm5_spec` drains `prefill_queue`, calls
`glm5_spec_session_new`, then installs the completed session, accepts queued tokens
into the worker sampler and sets `prefill_done` (`worker.rs:23662-23734`). The
constructor allocates the stage-owned cache, selects native MTP or DFlash2 taps,
and calls `prime_cache` once for the whole prompt (`glm_spec.rs:2665-2777`).
`prime_cache_hyper` freezes its ranges, then loops all of them synchronously:
PP at `hybrid_forward.rs:2271-2363`, unsplit/TP at `:2373-2397`. The odometer
`note_prime_rows` is progress reporting, not a worker yield. No worker yield
exists in either loop. `worker.rs:15375-15377` calls the step synchronously.

Chunk width is `MEMRA_PRIME_CHUNK` with a 4096 fallback, bounded by ring and CUDA
geometry; automatic PP geometry can choose smaller/dynamic ranges
(`hybrid_forward.rs:1300-1317`, `:1384-1431`, `:1490-1509`;
`crates/memra-kv/src/lib.rs:137`). `hyper_prime_ranges` delegates to that schedule
and documents why changing chunk size is NOT a bit-identity operation: mHC
cuBLASLt shape selection can change reduction order (`hybrid_forward.rs:1513-1578`).
KDA here is the sequential scan, with no GDN WY fold-grid analogue; preserve the
actual plan's `gdn_prime_grid_on()` result rather than inventing a KDA grid
(`:1530-1537`). CUDA grid.y/z max is 65,535; the ring-off launch cap reserves room
for folding a short tail (`:1289-1298`). Yielding cannot recompute this schedule.

**Restored speculative prime.** Admission reconstructs DFlash KV from the prefix
entry tail, or under SPEC_WARM creates an empty drafter at the restored absolute
position (`worker.rs:19620-19708`). It proves the restored carrier at
`:20686-20714`, retains the cache explicitly for the first spec tick at
`:20790-20797`, and stores `glm5_restored_dkv` at `:21073`.
`step_glm5_spec` takes that cache and calls `glm5_spec_session_from_restored`
(`:23685-23705`). The constructor orders the head engine behind the import,
uses entry logits directly on full cover, otherwise primes the suffix once and
captures the new boundary (`glm_spec.rs:3434-3478`). An empty suffix has zero
trunk chunks, but still has once-only anchor/finalization work.

**HYPER_SUFFIX_PRIME and admission: exact-main correction.** There is no direct
GLM5 suffix `prime_cache` call inside admission on this SHA. Admission uses
`carried_suffix_primes(..., hyper_suffix_prime_on())` to permit prime-provenance
recapture (`worker.rs:18995-19027`), and installs the carrier above. The actual
plain carried prime runs in `prefill_tick` (`:21640-21736`), whose take can still
contain several internal engine ranges. The GLM5 spec suffix runs at
`glm_spec.rs:3454`. The admission call `worker.rs:19397` is
`spec_session_from_restored`, the generic MTP route cited by the shared mechanism,
not `glm5_spec_session_from_restored`. Preserve that distinction in review.

Required change: every GLM5 cold/restored constructor entry becomes a walker
constructor, with no hidden prime-to-completion in admission. The plain hyper
suffix entry also needs an owned continuation of its already-decided prefill
segment, driven by the same hyper chunk machinery and shared policy. Its existing
snapshot and seq_end boundaries remain intact. The generic MTP admission bypass
belongs to the seam lane. A future refactor moving GLM5 suffix work into admission
must install a pending walker, never execute a nested synchronous prime there.

**DSA at 256k and 1M.** Each MLA middle appends latent rows, updates the indexer,
and selects before attention (`hybrid_forward.rs:9882-9926`). The multi-CTA
`MEMRA_B200_DSA_SELECT` is default ON on sm_100a builds, but its actual predicate
is width-sensitive (`mla_ffi.rs:511-519`, `:563-593`, `:2499-2524`):

| Query rows | Pool floor | Consequence with pool size 4 |
|---|---:|---|
| 1 | 65,536 | Starts at exactly 262,144 tokens, not a loosely named 256k prompt |
| 2 through 8 | 262,144 | Starts at 1,048,576 tokens |
| More than 8 | Ineligible | A normal 4096-row prime chunk uses the ordinary selector |

A 258,626-token trace is below even the width-1 floor. A 1,001,928-token prompt
is above the decode floor but below the width-2..8 floor. Short tails can cross
these conditions; freeze their per-chunk geometry too. Do not label multi-CTA
select as the normal long-prime chunk path merely because decode uses it.
The TP query-row split's per-rank selected width and exchange are also part of the
tape (`hybrid_forward.rs:11231-11245`, `:11389-11421`).

**TP-2 prime on the symmetric configuration.** `MEMRA_GLM5_TP=all@0,1` names the
loaded rank group (`glm5_tp.rs:92-111`, `:168-197`). Symmetric decode is not a
separate symmetric prime loop. Prime still uses the root-orchestrated hyper
range, calling sharded MLA and KDA (`hybrid_forward.rs:4882-4896`;
PP range twin at `:2675-2694`), and the split
expert grouped prime at `:15400`. `moe_ffn_glm5_tp_split_grouped_prime` runs all
routed pairs against each rank's half-width slabs, then returns/adds rank partials
(`:16266-16329`, `:16575-16626`). MLA tensor-core prefill admits 32-head shards
through the geometry-based chain (`:9952-9980`). Freeze EXPERT_SPLIT, the grouped
prime predicates and host-diet level; do not replace this combine with a new
collective in a fairness lane.

The one-shot and symmetric graph flags describe the decode/peer-service tape.
`glm5_tp_sym_graph.rs:1-20` explains per-rank graph pieces and live device sequence
words. TP speculative composition is separately default-OFF and requires
`MEMRA_GLM5_SPEC_TP=1` plus batched verify (`glm_spec.rs:2511-2538`). A plain TP
receipt cannot qualify this adapter's speculative composition.

**PP-2 prime.** Serial PP uses the same range list and passes
`queued_after + (t - end)` to preserve request-absolute seq_end
(`hybrid_forward.rs:2341-2354`). Inside each range, stage 0 sends expanded mHC
rows, the last stage receives and computes the tail, and all stages publish back
to the caller (`:4475-4548`). With B200_PRIME_V2 and its other conditions, the
pipeline overlaps stage 0(k+1) with stage 1(k), on two scoped host threads
(`:2290-2336`, `:4640-4755`). That entire loop also has no worker yield today.

## 2. State that survives each chunk

| State | Current owner and required lifetime |
|---|---|
| Tokens, absolute base, range cursor, overlay | Constructor/prime locals. Own the original token vector, full range list, absolute base and queued_after; preserve overlay windows and offsets (`hybrid_forward.rs:2258-2264`, `:2341-2354`, `:4834-4862`). |
| Hidden stack, last logits and seed | Whole-prompt/suffix hidden allocation plus last chunk result in the hyper loop; copy each range to its original row offset (`hybrid_forward.rs:2339-2363`, `:2382-2397`). Keep allocation identity/device ownership; do not truncate to a boundary row while MTP or captures need earlier rows. The seed copy occurs after logit readback (`:4928-4941`), so logit visibility alone is not a complete chunk fence. |
| Tap sink and drafter state | HcTapSink base/origin, host rows or device ring, tap layers, staging buffers, ingest cursor and DflashKv. Device mode owns Glm5DraftPrimeInflight (`glm_spec.rs:268-287`, `:2691-2755`); host mode preserves whole-prompt rows and ingest placement. |
| First token and sampler twin | Anchor, anchor_emitted, pending native `(token, hidden)` pairs or DFlash tap rows, sampling config, penalty history, worker fed/generated/emitted cursors. Cold capture is before anchor draw (`glm_spec.rs:2778-2805`); native warm follows that draw (`:2860-2877`). Worker accepts prompt tokens only after constructor completion (`worker.rs:23728-23734`). |
| Philox | Session `sctr` for device sampling, `uctr` for host accept uniforms, both persisted across bursts (`glm_spec.rs:4835-4839`). Cold/restored anchor initializes and advances sctr exactly once (`:2795-2805`, `:3483-3499`); no sampling at an intermediate trunk boundary. |
| CUDA graph captures | These are not the prefix snapshot. Existing VerifyGraphPool owns stage-range/row-count captures and private scratch (`glm_spec.rs:724-763`, `:4840-4845`). SymGraphState owns stable root/peer buffers, graph pieces, warm/phase/failed state (`glm5_tp_sym_graph.rs:102-118`), with two KDA ping-pong captures (`:29-35`). Cold prime has no live verify capture yet. Keep any carried cache graph state and all peer sessions' graph state with their owning session; never yield inside capture or between rank graph launches. |
| PP cursors and transfers | Owned per-stage caches, logical completed range, stage-0 lookahead cursor, next slot and payload. Existing PrimeCacheStages borrows its parent (`hybrid_forward.rs:223-227`); dropping it moves planes back, takes the minimum stage position and taints an uncommitted parent (`:315-347`). A resumable owner must eliminate that self-borrow, not drop/recreate the splitter on yield. |
| TP KV and recurrence | Canonical cache.pos plus each rank's KDA conv/SSM state and peer latent planes. TP prefix snapshots clone the rank planes on their own devices and reject torn peer lengths (`hybrid_forward.rs:3008-3066`). Indexer split refuses mismatched lengths before work, restores planes on errors, advances len and len_d only on success (`:11231-11245`, `:11389-11421`). |
| DSA indexer | Latent rows and len, index tail ring, final pool keys, index_pools_ready, pool size and device length mirror. Final pool keys are append-only; incomplete tail is not a pool (`crates/memra-kv/src/lib.rs:505-593`). Preserve partial tail through non-divisible chunks; never rebuild it from only the completed pools. |
| Prefix capture and publication | Cold constructor snapshots after complete trunk prime and before anchor (`glm_spec.rs:2778-2787`); restored suffix snapshots the new boundary (`:3458-3470`). Worker drains the capture only from a completed session, exports draft tail and inserts it (`worker.rs:11661-11704`, sweep at `:15056`). No intermediate yielded chunk becomes a published prefix by accident. |
| Lease and admission accounting | Prefix source pin, restored carrier reservation, active-session memory and cancellation/error status must remain owned while suspended. A yielded prime is active work, never a parked reuse entry. |

FIRST_TOKEN_EAGER defaults ON on this route (`worker.rs:24091-24099`). It does
not draw an eager token during the trunk loop. The completed constructor already
knows the anchor; the first streamed burst emits it once before the first round
(`glm_spec.rs:3727-3735`). Worker ON installs the callback and OFF uses the same
burst without it (`worker.rs:23740-23814`). Preserve both dimensions in tests:
PRIME_YIELD OFF/ON with FIRST_TOKEN_EAGER=1, then the FIRST_TOKEN_EAGER=0 twin.
The one-chunk peer turn must finish preparation and emit its anchor in that turn.
Long draft preparation may have later chunks; never emit the anchor earlier than
the existing lazy/eager draft-ingest setting permits.

Native MTP has a second preparation loop: `glm5_mtp_plane_fill` walks 512-row
chunks of successor-token/hidden pairs after the anchor draw (`glm_spec.rs:2415-2441`,
`:2860-2877`). Host DFlash eager ingest also runs before session return
(`:2883-2904`). These need owned preparation phases, not an unbounded `finish()`.
For the restored DFlash arm, pending suffix taps deliberately remain for the
burst (`:3517-3535`); preserve that existing ordering.

Device-staged taps default OFF with two explicitly documented defects: PP-2
pipeline lacks range hooks and the ring can be read across unordered streams
(`glm_spec.rs:248-265`). Do not switch that flag on as part of this adapter or
claim it is already safe. Its composition needs its own fixes and red gates if
included later. The ordinary host-tap configuration remains the design baseline.

**#343 source-pin release.** `worker.rs:21121-21153` releases a source lease only
after `prefix_restore_fence` succeeds and only for a plain carrier: it excludes
spec, gemma spec, dspark and GLM5. `hybrid_forward.rs:3185-3196` drains the caller
and TP peer streams because copies being enqueued is insufficient. The GLM5
adapter must retain its current spec lease; do not generalize the plain early
release without proving the drafter/tail import and every source read complete.
Fence failure keeps the lease and suppresses optional publication.

**#345/#350 reclaim-on-defer.** Current admission first pins the incoming prompt's
donor when headroom is short, then sheds borrowed cache, evicts unleased prefixes,
and evicts globally oldest parked plain/spec/dspark sessions while rereading
headroom (`worker.rs:14320-14394`). Keep this exact path when admission is invoked
between chunks. The yielded carrier must count toward active memory, its leases
must survive the sweep, and no retry may reconstruct it cold after a token has
been emitted (`worker.rs:23791-23794`). Exercise defer, cancel and allocation
failure with two requests sharing the donor. This is separate from the plain
source-release rule; do not confuse a source lease with a session reservation.

## 3. Rendezvous decision and proof

**Decision: one host-side completion barrier owned by the walker, before yield
return. Do not add a chunk word to MemraArSignal.** The worker and the
root-orchestrated TP prime already issue both ranks from one host control path.
Submit all operations for the chosen quantum on both ranks first, then fence
all participating stage and engine streams, collect rank completion/error status,
and return one request-level boundary `(request_generation, phase, chunk_index)`.
Never synchronize rank 0 between issuing rank 0 and rank 1 of a collective.
No peer work can start until both ranks acknowledge the same logical boundary.
On resume restore the owning contexts/streams and publish reverse dependencies
before scratch reuse. Host bookkeeping alone does not prove device completion.

TP prime rendezvous is after the entire hyper range, rank partial returns/adds,
all rank KV/indexer cursor updates, tap ingest, hidden/seed copies and temporary
buffer retirements, just before the new equivalent of the loop backedge at
`hybrid_forward.rs:2359` or `:2394`. Require all trunk-layer rank lengths to match
base+end; exclude the separately advanced MTP plane from that assertion.
The split grouped prime uses bulk-return/add (`:16608-16624`), not one-shot AR.
Symmetric decode peer service must nevertheless retain its complete quantum:
`glm5_tp_sym_graph.rs:604-614` launches rank 0 and rank 1 consecutively. Yielding
between those launches leaves the first rank waiting for its peer.

The exact one-shot failure is `cu/tp_ar.cu:148-158` MemraArSignal: per-block
`start`, `end`, `seq`, plus fused `grid_go`, `grid_done`, `hp_seq`. The start
barrier waits for the same flag from both ranks (`:174-197`), then reduces and
waits at the end barrier (`:215-230`). One-rank continuation would be a collective
stall, bounded today into error 40043/40044, not an infinite spin by design.
Read `ArLink::barrier_errors` (`tp_ar.rs:622-630`) before success. Reusing these
signals for chunk scheduling would couple chunk indices to per-block collective
sequence and graph replay; it would not protect PP transfer slots anyway.
Do not reset the link's global sequence between sessions. The shared link is
serialized across peer service, while request chunk progress is request-local.

PP serial rendezvous is after last-stage tail and `publish_all_to`
(`hybrid_forward.rs:4539-4548`), plus caller/seed-copy completion. A prematurely
continued receiver waits on `ev_tx` in `pp.rs:3161-3165`; if it observes an old
slot/event generation instead, it can consume the wrong payload. This is a
transfer-generation divergence hazard as well as a possible wait, not a TP
all-reduce. Never allow another request to overwrite a suspended transfer slot.

PP pipelined rendezvous must preserve the existing one-chunk lookahead. After
`stage0.join` and the stage-1 tail for chunk k (`hybrid_forward.rs:4705-4714`),
both stages park at logical completed chunk k, but physically stage 0 can be at
end(k+1) and stage 1 at end(k) (`:4656`, `:4686`, `:4757-4758`). Own both cursors,
the next-slot generation and the already-computed payload. Copy that payload to
request-owned storage or reserve the slot with a lease spanning peer work before
returning; runtime boundary slots are shared. Resume consumes it exactly once
without repeating stage 0(k+1). This preserves per-rank math and the frozen PP
overlap schedule. Equal physical stage cursors at every yield would require
removing or draining lookahead and would change that schedule. The utility must
accept a common logical watermark plus route-owned physical frontier, not demand
equal cursor values. At finish both physical cursors equal request end.

**Test to prove the decision, proposed `glm5_prime_yield_gpu` rendezvous cells.**
Run on the real B200 pair for TP `all@0,1` with EXPERT_SPLIT, TP_AR_1STAGE,
TP_SYMMETRIC and SYM_GRAPH armed, and separately PP-2 serial and pipelined.
Use at least three non-divisible ranges and inject a bounded delay on rank/stage
1 just before the boundary acknowledgment at early, middle and last chunks.
Record issued/completed chunk indices, absolute rank lengths, slot generations,
AR error words, signal sequence deltas and first-token event order. Assert:
no peer dispatch before both acknowledgments; no next unscheduled chunk on one
rank; original long prime resumes after one bounded peer sweep; logits/captures
and request token ids equal the unyielded arm. PP pipeline additionally asserts
stage-0 lookahead is executed once, its payload hash survives a peer prime that
uses the same transfer slots, and both cursors converge at finish.

Red arms: missing rank-1 acknowledgment must refuse before peer service; stale
chunk generation must refuse; overwriting the retained PP slot must break the
capture oracle; skipping a rank graph launch must produce the existing bounded
AR error rather than pass with root-only output. Use a harness timeout and only
owned PIDs. Record errors before cleanup; no unbounded test spin. Inject
cancel/error at the same boundaries and verify cache ownership, leases and no
partial prefix publication. Shared CPU fake-executor tests prove policy; these
real-device gates prove CUDA ordering and multi-rank state preservation.

## 4. Frozen before chunk one

1. Original tokens, model/artifact/plan identity, dtype and placement, cold versus
   restored versus full cover, prefix cut, captured boundary logits and drafter
   tail provenance. A 49,152-token restored prefix is an input cut, not a new
   hardcoded split. Capture points and any plain prefill snapshot/segment limits
   are frozen before building their range lists (`worker.rs:19009-19027`,
   `:20686-20698`; `glm_spec.rs:3440-3470`).
2. Each segment's entire range list, dynamic/fixed choice, tail folding, launch
   cap, ring posture and effective chunk width (`hybrid_forward.rs:1306-1317`,
   `:1390-1431`, `:1490-1509`). Re-entering `prime_cache` with a smaller suffix
   would rederive geometry; the adapter calls the extracted inner operation.
3. Request-absolute `seq_end = initial_base + segment_tokens + queued_after`,
   even after suspension. PP's remaining-token adjustment stays equivalent
   (`hybrid_forward.rs:2248-2250`, `:2348`, `:2370-2374`).
4. Overlay offsets/windows and their storage, absolute tap origin/base, hidden
   copy offsets and all per-rank stage cuts (`hybrid_forward.rs:2258-2264`,
   `:2342-2354`, `:4850-4862`).
5. Kernel selection policy and shape-derived decisions per frozen chunk:
   mHC/GEMM algorithm shapes, grouped prime versus sequential, host-diet level,
   MLA TC eligibility, TP expert split and indexer query split, KDA scan class,
   PP pipeline eligibility and graph configuration. Do not let peer arrivals
   or changed environment reads pick another prime program (`:1530-1555`,
   `:2294-2302`, `:9952-9980`, `:15385-15409`).
6. Grid law and DSA predicate schedule, including per-chunk absolute pool count,
   query width and threshold crossing (`mla_ffi.rs:563-593`). Visibility still
   follows each absolute query row; freezing policy does not expose future keys.
7. Sampler/penalty seed, draft source, lazy/device/host ingest placement and
   FIRST_TOKEN_EAGER emission posture (`glm_spec.rs:2600-2605`, `:2795-2805`,
   `:2883-2904`; `worker.rs:23740-23756`). The existing concurrency/K policy
   may act between complete service quanta; prime yields cannot resample anchors.

## 5. Chunk-wall evidence and missing cell

Read-only external receipt: private sibling `darklanes` at
`004c2cfddb81a130126805dded49060e09090bb6`,
`research/glm5-b200-mint-20260904/LANE.md`. The following are historical receipts
with their own engine pins, not measurements of this main or a future tune pair.

| Receipt and file lines | What it actually measures | Early/middle/late complete chunk wall? |
|---|---|---|
| tpprime2, LANE.md:2963-2975 | Whole primes: TP 7.08/57.7 s at 3,766/29,961 tokens; PP 1.73/7.36 s | No |
| tpprime3, :2980-2993 | Whole-prime grouped-split OFF/ON, 57.7/31.5 s at 29,961 | No |
| tpprime4, :3007-3019 | Explicit MLA receipt `t=4096, t_kv=4096, nh=32`; TC shard port | Establishes chunk shape, not chunk wall |
| tpprime5, :3023-3033 | 44.7 s at 128,847 tokens; 782 s at 1,001,928 | No |
| tptrace8, :3056-3075 | 258,626-token prime, 106.6 s; 704 DSA scorer launches = 11 layers x 64 chunks, 45-46 ms/launch, 32.6/31.9 s rank totals | Kernel distribution, not indexed chunk walls |
| tptrace9, :3101-3126 | Decode at 128k, 64 generated tokens; width-1 select 77 us/layer | No, decode |
| tptrace10, :3128-3146 | Decode at 1M, 48 generated tokens; score 139 us/layer, hist/tie selection timings | No, decode |

Target first wall cell uses explicit `MEMRA_PRIME_CHUNK=4096`, matching the cited
receipt shape, on the future non-production B200 tune pair. Confirm the handed-off
recipe's width before launching; if different, measure that width too. No locally
available indexed chunk-wall series was found under that lane's receipt names.
Do not reconstruct C from TTFT divided by chunks or from summed average kernels.

Measure every existing range with request-id, boot nonce, mode, phase, range index,
start/end, rank frontiers, initial/restored cut, host start and completion wall,
and per-device events. The wall closes after the host rendezvous, copies and tap
consumer; enqueue duration is insufficient. Retain raw JSONL, then report first,
middle, final full chunk, short-tail chunk and maximum C separately for cold and
restored 256k/1M, PP-2 and TP-2. Measure setup, capture, draft preparation and
finalization separately, including Q for a one-chunk peer's first-token turn.
For pipeline mode record the lookahead as well as completed chunk; C bounds the
actual service quantum, not stage 1 alone. Three fresh boots per arm, interleaved.

Shared bound: added long-prime work at most `(m-1)*(N-1)*Q` plus scheduler
overhead. One admitted long prime and one one-chunk small request: small TTFT is
residual C plus its idle first-token work. Admission/VRAM waiting and setup or
finalization holds are separately reported. No latency SLO is established yet.

## 6. Adapter shape and estimated patch

Proposed names only, pending the seam owner's actual trait signature:

```rust
struct Glm5PrimeWalker {
    input: Glm5PrimeInput,       // cold, restored, or plain carried segment
    frozen: Glm5PrimeTape,       // tokens, ranges, seq_end, cuts, numeric choices
    phase: Glm5PrimePhase,      // trunk, anchor/setup, draft preparation, ready
    next_range: usize,
    cache: Glm5PrimeCacheOwner,  // whole cache OR owned PP stage caches
    hidden_stack: HiddenRows,
    last_boundary: Option<BoundaryRows>,
    taps_and_draft: Glm5DraftPreparation,
    pending_anchor: Option<u32>,
    sampling: Glm5PrimeSampling, // sctr, uctr, config, penalty history
    prefix_capture: Option<SpecBoundaryCapture>,
    rendezvous: Glm5PrimeFrontier, // request generation, phase, ranks, PP lookahead
}
```

The server's pending wrapper owns the source lease, accounting, token-emission
state and request cancellation handle. Engine code does not own PrefixCache or
worker scheduling. Caches own graph state, not a second graph pool in the walker.
Names above are design types, not claimed existing Rust declarations.

`advance one chunk`: execute exactly the next frozen existing range, including
per-range tap consumer and hidden copy; complete the host rendezvous; update the
cursor once. Setup, anchor draw and draft preparation follow their original
ordering. Draft preparation advances its existing fill/ingest chunks as additional
phases. Never insert another call to the outer `prime_cache` loop. PP pipeline
retains the original ahead-of-consumer frontier and owned transfer payload.

`remaining`: report remaining executable preparation chunks, including later draft
phases. Expose terminal-finalization readiness separately if the shared signature
cannot represent a zero-trunk/full-cover preparation. No inference from mutable
`prefill_queue.len()` after the vector has been moved.

`finish`: consume a fully prepared walker exactly once into Glm5SpecSession or a
plain carried-prime result. Transfer the capture, sampler counters and pending
anchor, update worker fed/sampler/prefill_done once, then allow existing emission.
It must not conceal a whole-prompt draft loop. OFF is the shared drive-to-completion
loop over these same calls. It preserves numerical order; the necessary boundary
fence cost must appear in the OFF-refactor control receipt as well as ON/OFF.

Estimated implementation, excluding the generic seam: 700-1,100 moved/edited lines
in `hybrid_forward.rs` (owned hyper cursor and PP frontier), 350-650 in
`glm_spec.rs` (cold/restored preparation phases), 120-220 in `worker.rs` (pending
state and plain carried segment integration), 30-100 in `pp.rs`/`glm5_tp.rs`
(completion and transfer ownership accessors), 400-700 for focused real-device
and HTTP gate drivers. Roughly 1,600-2,800 changed lines, much of it extraction.
No CUDA kernel edit is planned. `docs/FLAGS.md` only gains this route's pointer
in the shared row when implementation/receipts land. Phase 1 changes this file only.

## 7. Gate contract copied verbatim from owner handoff

4. Exactness gates (per-request determinism; cross-timing sampled identity is NOT the gate,
   because route/K choice under concurrency is existing policy):
   - c=1 greedy 4-turn chain OFF vs ON byte-identical, cold and restored.
   - c=2 greedy pair where the yield provably fires (log line with yield count > 0, long prime +
     small request), spec guard pinned so both keep K>0 (`MEMRA_SPEC_GATE_LOW/HIGH` raised for
     the cell, say so); each request byte-identical to its own c=1 greedy output.
   - resumed prime boundary logits / capture identical to the unyielded prime (route's own
     oracle; add one if none exists).
   - PP-2/TP-2: the same three gates on the pair, plus rank-rendezvous test.
5. Latency receipt, interleaved arms, 3 boots per arm, boot nonce identity, vendor-default
   sampling (no sampling params) with K>0 receipts from the server log:
   - one long cold prime at the served context (use 256k and 1M shapes) + small requests at
     0.20 req/s for 100 s: small p95 TTFT OFF vs ON, long TTFT and total OFF vs ON, zero
     OOM/retries, spec route counts.
   - chunk wall at the served chunk size (sets C).
   - 8-turn cache-on twin unchanged.

## 8. Concrete future cells and execution contract

Execute only after the orchestrator hands off the non-production B200 tune pair,
its private access details and recipe. Never Texas production boxes, never the
local rig. No box has been allocated or contacted in phase 1. A different hardware
shape is a separate qualification. PP-2 and TP-2 are separate boots, never co-armed
on the same load. Keep provider identifiers, credentials and endpoint routing in
the private handoff, not in this document.

The following commands define the phase-2 driver interface to implement under
`research/prefill-fairness-20260908/glm5/`. `run_cells.py` and
`glm5_prime_yield_gpu` do not exist on the baseline; these commands are a concrete
implementation/gate plan, not executed or passing tests. Existing seed oracles are
`tests/glm5_chunked_prime_gpu.rs`, `tests/glm5_spec_session_gpu.rs`,
`tests/glm5_dflash_session_gpu.rs`, `glm5-hyper-ppn-gate` and `glm5-tp-gate`.
Reuse their fixture/capture mechanisms, but add a bitwise same-range resumed
boundary oracle: the existing changed-chunk-size tolerance gate is insufficient.
The HTTP driver can reuse SSE parsing and cache checks from
`research/glm5-prefix-latent2-20260901/battery2.py`; its old PP3 recipe is not this
pair's recipe and must not be copied as a PP2 config.

On the assigned pair only, after staging the exact merged seam+adapter commit:

```sh
export MEMRA_CUDA_ARCH=100a
export MEMRA_GPU_LOCK=/tmp/memra-gpu.lock
# TUNE_RECEIPTS and TUNE_RECIPE are absolute paths supplied in the private handoff.
# The recipe pins model revision, artifact, plan, drafter, capacity, binary and topology.
flock "$MEMRA_GPU_LOCK" bash -euc '
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
  test -z "$apps"
  cargo test --release -p memra-engine --test glm5_prime_yield_gpu --no-run
  cargo build --release -p memra-server
  python3 research/prefill-fairness-20260908/glm5/run_cells.py \
    --recipe "$TUNE_RECIPE" --out "$TUNE_RECEIPTS" \
    --cell exactness --placements pp2,tp2 --arms 0,1 \
    --turns 4 --concurrency 1,2 --restores cold,suffix,full-cover \
    --first-token-eager 1,0 --gate-low 64 --gate-high 128 --require-yields \
    --sampling greedy --boundary-captures bitwise
  cargo test --release -p memra-engine --test glm5_prime_yield_gpu \
    -- --ignored --nocapture --test-threads=1
'
```

Export TUNE_RECEIPTS and TUNE_RECIPE into the child environment before this block.
The driver must hold the same outer lock throughout all boots and must repeat the
empty compute-app check before EACH server or gate launch. After each boot it
stops only its recorded child PIDs and waits for exit and empty compute-app list.
No build command in this block is authorized on the rig. Exactness boots set
`MEMRA_PRIME_YIELD=0/1`, `MEMRA_PRIME_CHUNK=4096`,
`MEMRA_SPEC_FIRST_TOKEN_EAGER=1/0`, `MEMRA_SPEC_GATE_LOW=64`,
`MEMRA_SPEC_GATE_HIGH=128`. Record these raised guard values explicitly; ensure
session limit >=2. Cold cache namespaces are fresh, restored ones are deliberately
seeded and require positive cached tokens plus the GLM5 restored-route log.
Compare every request against its own same-placement, same-restore c=1 oracle,
including full four-turn chains. The c=2 trigger waits until the long prime has
begun a measured chunk, then injects the small request. Count actual nonterminal
yields and require K>0 per request, not merely an armed boot flag.

TP recipe: `MEMRA_PP_STAGES=1`, `MEMRA_GLM5_TP=all@0,1`,
`MEMRA_GLM5_TP_EXPERT_SPLIT=1`, `MEMRA_TP_AR_1STAGE=1`,
`MEMRA_GLM5_TP_SYMMETRIC=1`, `MEMRA_GLM5_TP_SYM_GRAPH=1`,
`MEMRA_GLM5_SPEC_TP=1`, batched verify enabled, and the recipe's TP prefix door.
PP recipe: TP unset, `MEMRA_PP_STAGES=2`, devices 0,1 and a pinned model-valid
PP split supplied by the tune recipe; separate B200_PRIME_V2 serial/pipeline
cells. In both, GLM5 speculation and its pinned drafter must actually engage.
Record HYPER_SUFFIX_PRIME, PREFIX_LATENT, GLM5_SPEC_PREFIX, full-cover settings
and cache capacity. A recipe with SERVE_SPEC=0 or a SPEC_MAX_PROMPT cap excluding
the long shape cannot satisfy these gates; correct the experimental recipe before
claiming a fairness result, with no changes to any serving recipe.

Latency and wall cells, under the same lock/empty-card/owned-PID protocol:

```sh
python3 research/prefill-fairness-20260908/glm5/run_cells.py   --recipe "$TUNE_RECIPE" --out "$TUNE_RECEIPTS" --cell chunk-walls   --placements pp2,tp2 --contexts 262144,1048576 --prime-chunk 4096   --restores cold,suffix --capture-all-chunks --boots-per-arm 3   --boot-order 0,1,1,0,0,1 --sampling vendor-default
python3 research/prefill-fairness-20260908/glm5/run_cells.py   --recipe "$TUNE_RECIPE" --out "$TUNE_RECEIPTS" --cell mixed-latency   --placements pp2,tp2 --contexts 262144,1048576 --prime-chunk 4096   --small-rps 0.20 --arrival-seconds 100 --drain-all-requests   --boots-per-arm 3 --boot-order 0,1,1,0,0,1 --sampling vendor-default   --require-spec-engagement --cache-on-twin-turns 8
```

Contexts name capacities, not a demand to stuff the full capacity with prompt
ids and leave no generation room. Fix actual tokenized prompts and output budget
in a hashed manifest, record exact input/remaining capacity, and respect the
constructor's prompt+anchor+round reserve (`glm_spec.rs:2606-2613`). Include a
separate threshold fixture at exact DSA pool boundaries. The 1M cell's long
request must finish even if it runs beyond the 100-second small-arrival window.
Use identical arrival offsets and corpus between arms. Vendor-default HTTP bodies
omit temperature, top_p, top_k and other sampling overrides. No sampled
cross-timing byte identity gate. Record each boot nonce, full env, engine and
binary hashes, artifact/plan hashes, requests/responses, token counts, per-request
route/K and `[glm5-acc]` logs (`worker.rs:23819-23836`), all failures/retries and
OOMs. K>0 must be evidenced during each scored request window; if ordinary policy
demotes, the required speculative cell is not satisfied and needs a separately
reported pinned-policy cell, not silent removal of the failed condition.

Report small p95 TTFT, long TTFT and total, setup/finalization walls, chunk C,
peer Q, yields, route counts, retries/OOM, cold/restored splits and all three boot
rows per arm. Exclude greedy loops from performance aggregation; greedy remains
correctness only. The eight-turn cache-on twin must show continued cache reuse
and the same gate behavior as OFF, not just one first hit. Add a pre-refactor main
versus walker-OFF control so extraction/fencing cost cannot hide inside both arms.
Remote CPU checks cover shared fake-executor policy, worker ownership, fmt and
hosted CI in phase 2. No local checks that invoke cargo, no push now.

## 9. Questions for the shared seam owner

1. Exact trait lifetimes and completion/error types: can the engine walker own its
   cache without borrowing a Session/active vector, and how does finish consume
   it exactly once? Send the seam SHA and signature through the orchestrator.
2. Does remaining count multiple preparation phases and support zero-trunk
   full-cover readiness? GLM5 native warm and host DFlash ingest must not hide in
   an unbounded finish, while one-chunk peers must reach first-token emission.
3. Can the utility represent a common logical chunk watermark with route-owned
   PP lookahead frontiers? Require a route completion hook before service; equal
   physical PP cursors would force a different pipeline schedule.
4. Provide generic ownership/cancellation and scoped scratch/transfer reservation
   hooks. A peer prime must not steal a suspended PP transfer slot or invalidate
   live graph/hidden/tap storage. Do not put GLM5-specific buffers in the policy.
5. How does admission install pending walkers and account their live/reserved
   bytes? Include plain carried hyper segments and exclude suspended active
   walkers from parked-session reclaim. The generic admission-MTP bypass at
   worker.rs:19397 remains the seam lane's change.
6. Shared telemetry schema should carry request generation, phase, chunk index,
   completed/remaining counts, yields, boot nonce and timing scope. GLM5 extends
   it with rank frontiers/AR errors/PP slot generation. Confirm schema and fake
   executor hooks before implementation; do not create duplicate policy tests.

Phase 1 result: design only. Implementation, compilation, exactness, rendezvous
and latency gates are all pending the seam and tune-pair handoff. The branch is
kept for the stacked phase-2 continuation. The pre-commit hook invokes
`cargo fmt --all -- --check` (`tools/hooks/pre-commit:6`), so the authorized
doc-only commit uses `git commit --no-verify`. Nothing is pushed, released or deployed.

## Phase 2: seam integration checkpoint

2026-09-09. Rebased the design commit onto seam head
`f604518ca`, containing shared utility `534040262`. Static template reads:
`crates/memra-engine/src/prime_walker.rs`, `spec/prime.rs`, `dflash.rs`,
`crates/memra-server/src/prime_fairness.rs` and their production worker call sites.
No shared utility or policy changes are owned by this lane.

Section-9 answers from that code:

| Question | Answer at f604518ca | Remaining dependency |
|---|---|---|
| Lifetimes and ownership | PrimeWalker has associated Output; advance/remaining borrow, finish consumes. DsparkPrimeWalker and MtpPrimeWalker bind borrowed engines to owned pending state only during a call. | None for GLM5 state ownership. |
| Multiple phases/full cover | advance_prime requires remaining to decrease by exactly one. finish_prime permits zero chunks and forbids incomplete finish. MTP counts trunk and draft fill together. | Route must precompute all preparation chunks and preserve the anchor position. |
| Rank watermark | No rank assumptions in PrimeChunk, only phase/rows. Adapter must fence before return. | Route owns logical/physical frontiers, PP lookahead and completion acknowledgments. |
| Scratch/cancel | Trait requires no shared-scratch borrow across advance. PrimeService keeps pending true through a failed advance/finalize; worker skips pending demotion and retirement reuse. | PP runtime slots need route-owned copied payloads; no generic reservation API exists. |
| Admission | MTP restore can defer its suffix to the walker. GLM5 already carries restored cache/dkv to its first tick, so that tick must construct its walker. | Plain hyper admission is outside spec_order; generic scheduling of plain pending primes requires seam-owner design, not a local scheduler fork. |
| Receipts/tests | trace_chunk records phase/rows/wall; PrimeService counts yields. Shared fake tests exercise drain/advance, errors and finish; policy tests rotate peers. | Request/boot identity, rank frontier and phase-specific capture diagnostics are not in shared trace. Route receipts must correlate them; request correlation in the generic trace is an orchestrator request. |

Additional code fact: `glm5_spec_session_from_restored` still explicitly refuses
TP-sharded models. Preserve that admission rule in the adapter. The queued TP
restored gate cannot be reported passing without the separately owned TP spec
restore support; record it as a prerequisite, never silently test plain instead.

Builds/tests use a lane-owned directory and target on the assigned single B200 tune
host, nice 19, at most 16 jobs, serialized with other lanes via
`/tmp/memra-gpu.lock`. Pair gates are queued, not executed on this single-card host.

### Stage 2 implementation and validation

The route now owns Glm5PrimeState and Glm5PlainPrimeState, bound temporarily to
PrimeWalker implementations. Glm5TrunkPrime keeps the frozen token ranges,
request-absolute seq_end, hidden stack, last logits, PP stage-0 lookahead payload
and logical rank acknowledgments. Each advance issues all ranks before waiting,
checks physical MLA frontiers and retires all participating streams before return.
PP rx already returns an owned copy; the adapter retains that copy across peer
work so the shared runtime slot can be reused safely. No MemraArSignal layout or
shared PrimeWalker/PrimePolicy code changed.

Cold and restored speculative constructors drain these same walkers on OFF.
The worker keeps queues intact until successful finish and protects pending GLM5
sessions from both plain prefill phases. Anchor sampling stays after the last
trunk range; native warm uses its existing 512-row ranges and host DFlash ingest
its existing 256-row ranges. A single short warm is folded into a one-chunk cold
request's final service quantum, preserving first-token delivery in that turn.
No prompt-length loop is hidden in finish. Existing prefix leases remain owned
by the worker; optional captures publish only after a completed speculative step.

Text-only plain hyper segments, including HYPER_SUFFIX_PRIME, also use the saved
trunk. Their original prefill_tick segment size, queued_after and snapshot stops
are frozen, and later budgets cannot resize an in-flight segment. Vision and
prompt-capture paths keep their existing synchronous overlay program. This
answers the earlier plain-admission open item without a scheduler change: the
existing plain phase can advance the same pending segment once per tick.

Frozen numeric settings are checked on every advance/finish against the initial
MEMRA environment snapshot. A gate changing settings in-process must finish the
current walker first; a mutation during suspension refuses instead of silently
selecting another numerical program. Model/artifact/placement references remain
immutable; overlay is absent in the admitted adapter shape.

Remote validation r5 on one B200, own directory/target, nice 19 and jobs=16 under
the shared GPU lock: strict library clippy passed; 33 engine prime CPU tests passed;
639 server tests passed, zero failed, one pre-existing ignored. Server and pair
probe builds passed. The PP-1 miniature DFlash GPU gate passed four-turn cold and
restored chains OFF/ON, including interleaved peer work, exact boundary logits and
actual recurrent/latent-tail captures. Cold ON arms each yielded 17 times; restored
ON arms each yielded twice. These are fixture correctness receipts, not model-scale
HTTP or pair qualification. The HTTP cells and full pair gate remain separate.

Remaining seam-owner requests are generic request/boot correlation in prime
trace records and review of the shared phase-order fairness bound for plain
pending segments. No new generic utility API is required for the implemented
ownership, multi-phase count, zero-trunk full-cover or host barrier.
At f604518ca the worker still refuses TP speculation at boot and the engine
refuses TP spec restores. The separate TP spec lane must remove those refusals
with its own evidence before this lane's TP HTTP cold/restored queue can pass.
The adapter does not remove or bypass those admission laws.

### Stage 3 gate artifacts and pair queue

The reproducible remote command is `glm5/run-validation.sh`; run it under
nohup/setsid with a complete log. It takes the shared lock, uses a lane-owned
CARGO_TARGET_DIR, nice 19 and jobs=16, records source/binary hashes and verifies
empty compute applications before/after GPU work. Final exit status is 0 in
`glm5/receipts/final/validation.exit`. The exact source manifest was checked
against the local files after remote formatting.

Final fixture/CPU result: strict library clippy and builds passed; 33 engine prime
CPU tests passed; 639 server tests passed with one existing ignored; both new
GLM5 GPU tests passed, including the plain restored segment with queued_after=9;
all 10 existing native MTP session GPU tests passed. Raw output is retained in
`glm5/receipts/validation-final.log`, with source/binary hashes alongside it.
Generated test output is retained verbatim. The standalone plain test compares
both OFF and yielded suffix logits/hidden rows to the old hyper loop while a
peer primes between ranges.

Pair script: `glm5/run-pair-cell.sh`. Supply PAIR_PROFILE, PAIR_METADATA,
PAIR_RECEIPTS, PAIR_BINARY and its SHA256, PAIR_GATE_BINARY (the same-source
`glm5-tp2-box-probe`) and its SHA256, PAIR_GATE_PROMPTS (multichunk text), and
PAIR_PROMPT_256K / PAIR_PROMPT_1M (pinned request JSON). The gate probe's new
BOXP_MODE=prime-walker compares the complete hidden stack and boundary logits
with the independent old hyper loop while another cache primes and decodes
between saved chunks. It requires nonzero yields and exercises reuse of the PP
transfer slots and decode graph scratch. CPU rank acknowledgments have stale,
missing and repeated-rank red cases. Actual delayed-device and one-rank collective
fault injection remain pair qualification work; this script does not claim those
faults were exercised by PP-1 fixtures.

Run separate pinned PP-2 serial, PP-2 pipeline and TP-2 configurations. The HTTP
runner then executes c1 four-turn cold/restored chains and a c2 long/small pair
against each request's c1 oracle. Exactness boots pin K=3 and raise LOW/HIGH to
64/128. Each response requires complete SSE and per-request drafted counts; ON
requires a real yield, and every boot requires server-side GLM5 engagement.
The 256k and 1M latency cells use OFF/ON/ON/OFF/OFF/ON boots, 20 small requests
at five-second offsets over 100 seconds, full drain, vendor-default sampling and
an eight-turn cache-on twin. Input JSON must reserve enough context for all eight
continuations, not just the initial generation. Every actual phase/chunk wall is
retained in server logs; no TTFT-per-chunk estimate is produced.

The single-card HTTP run uses a separate bounded context profile and validates
only PP-1. Its first attempt refused MEMRA_REQUEST_LEDGER at boot because the
public engine binary carries no deployment accounting. The harness now omits
that deployment-only variable; client requests/events/results remain the receipt.
No serving process or shared checkout was modified. Pair cells remain queued and
default OFF remains intentional; no latency improvement is claimed by this lane.

Local commit/push hooks are disabled per invocation to obey the no-rig-gates
instruction. Pushes set MEMRA_SKIP_PERF_CI=1. The complete remote checks above and
hosted PR CI remain the validation path. No release or deployment is part of this PR.
