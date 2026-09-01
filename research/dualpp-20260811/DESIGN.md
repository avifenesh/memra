# Dual-active PP-2 decode — design study

Date: 2026-08-11. Lane: `lane/dualpp` (read-only research; no GPU work, no code edits).
SOL audit candidate #3 (research/solgap-20260811/REPORT.md).

## Headline

Plain batched decode on PP-2 runs strictly serial stages (5.557/6.179 ms balanced but one card
always idle). The dual-active proposal: split the batch into TWO waves so stage0(wave B) runs
while stage1(wave A) runs — chunk pipelining across the PP boundary. Expected aggregate: up to
+80-90% at c>=16 (bounded by the m2-pp8 1.88x receipt), realistic first cut +40-60%.

## (a) Shape

### Per-wave state

- **KV positions**: each wave's requests own their position snapshots (`caches[i].pos`) at wave
  issue time; waves do NOT share position buffers.
- **Boundary slots**: double-buffered slots already exist (pp.rs:466 `BoundarySlot` x2); the
  dual-active arm alternates slot assignment per wave (wave A → slot 0, wave B → slot 1, then
  wave A → slot 0 again). The existing `tx_pipelined`/`prepare_overlap_slots` machinery from
  opti0 (RESULTS.md:23-60) proved the seam primitives.
- **Streams/events**: each stage already owns its stream (pp.rs:446 `StageRt`); boundary events
  (ev_tx/ev_rx) order wave A vs wave B cross-stage writes. No new event needed (opti0 used the
  existing pair).
- **Residuals**: each wave materializes its own `[b_n, n_embd]` boundary buffer per stage; waves
  do NOT share residual storage.

### Shared state

- **Weights**: all layers' weights remain shared across waves (single load, no duplication). Under
  MEMRA_PP_DEVICES each stage's weights live on that stage's device (pp.rs:1333 `layer_engine`),
  unchanged.
- **KV cache**: each request's KV belongs to that request's cache exclusively (pp.rs:45-46);
  waving does NOT touch cache ownership. Per-layer KV is already per-stage-owned (pp.rs:1260
  `new_cache`), unchanged.
- **Boundary buffers**: the two persistent slots (pp.rs:458) are already double-buffered and
  GROW-ONLY (pp.rs:1010 tx comment); dual-active inherits this high-water design — no new
  allocation strategy needed.

### The schedule

At c>=2, the worker splits `caches` into wave A (first half) and wave B (second half), then:

1. issue stage0(A), await boundary TX(A) → slot 0
2. issue stage0(B) in parallel with stage1(A):
   - stage0(B) runs on stage-0's stream
   - stage1(A) runs on stage-1's stream
   - boundary RX(A, slot 0) orders stage1(A) after TX(A)
   - TX(B) → slot 1
3. issue stage1(B), RX(B, slot 1)

The overlap window is the intersection of stage0(B) and stage1(A); opti0 measured 10.429 ms
(RESULTS.md:82) for the spec path at this boundary. With balanced stages (5.557/6.179 ms,
specpp2-20260810), the overlap hides most of stage0(B)'s 5.557 ms — the incremental cost is
max(stage0(B), stage1(A)) - stage1(A) ≈ 0 when stage0 < stage1, bounded by the longer stage.

### Wave-split arithmetic

- c=1: serial fallback (no second wave)
- c=2: wave A = caches[0], wave B = caches[1] (one session per wave)
- c=3: wave A = caches[0..2], wave B = caches[2] (A gets ceil(3/2)=2, B gets floor(3/2)=1)
- c=4: wave A = caches[0..2], wave B = caches[2..4] (equal split)
- c=8: wave A = caches[0..4], wave B = caches[4..8] (equal split)
- c=16: wave A = caches[0..8], wave B = caches[8..16] (equal split)

At serving tick c=64 (research/solgap-20260811: 8 serial B=8 chunks/tick), the worker issues 8
pairs of waves, each wave processing 4 requests. Today's serial walk: 8 x (stage0 + stage1) = 8 x
11.736 ms ≈ 93.9 ms. Dual-active: stage0(A) + 7 x max(stage0(B), stage1(A)) + stage1(last-B) ≈
5.557 + 7 x 6.179 + 6.179 ≈ 55 ms ideal, ~60-65 ms realistic (scheduler overhead, unbalanced
waves at odd c). Expected aggregate: 129.7 tok/s serial → 210-245 tok/s dual-active (+62-89%).

At c=1 (single session, no second wave): the arm MUST fall back to serial (one stage0, one
stage1, identical to today's walk). Do not fake a "dual-active c=1" by splitting a single
request across two waves — that changes the request's arithmetic and breaks exactness.

## (b) Exactness

### Invariant

Each request's byte stream must be identical to the serial PP-2 arm: same logits, same sampled
tokens, same KV append order. The one-numeric-class contract (decode_batch.rs:787 step35, :802
generic exclusion) already holds at the batched level; dual-active preserves it by NOT changing
any request's forward pass — waves run the SAME trunk code, just scheduled differently.

Waving does NOT change:
- per-request embed gather (decode_batch.rs:818)
- per-request layer walks (decode_batch.rs:1446-1481 per-row FA loop; step35_decode_batch_layers
  for the step35 path)
- per-request KV append (decode_batch.rs:1456-1462)
- per-request logits (decode_batch.rs:1536+ epilogue)

Waving DOES change:
- issue order: stage0(A), then stage0(B) || stage1(A), then stage1(B) — but each stage's stream
  orders ITS OWN work, so wave B's stage0 cannot see wave A's KV and vice versa.
- boundary residual transport: wave A uses slot 0, wave B uses slot 1; the TX/RX fences
  (pp.rs:1042 ev_rx wait, :1099 ev_tx record, :1111 ev_tx wait, :1132 ev_rx record) prevent
  slot reuse before consumption.

### Gate

Extend the existing one-hash decode-batch matrix (the bit-identity gate that caught the step35
geometric corruption, research/step-sku-20260807):

- **serial arm**: today's `decode_step_batch_ppn` at c=1/2/3/4/5/6/7/8 (A/B seam: MEMRA_DUAL_PP=0)
- **waved arm**: dual-active at c=2/3/4/5/6/7/8 (MEMRA_DUAL_PP=1, default OFF initially)
- **assertion**: EVERY request's completion bytes match serial, per session. One hash per request,
  all requests in one invocation must match.

The gate MUST pass at all widths c=1..8 before promotion. c=1 proves the serial fallback; c=2..8
prove the wave split at even/odd widths. If a single request diverges, the mechanism is wrong —
do NOT ship a "mostly correct" wave scheduler.

Gate structure (mirror pp2-batch's `--mode pp` pattern from research/pp-leverb-20260807):

1. load model with PP door OPEN (sharded, cross-device — the production config)
2. serial arm: MEMRA_DUAL_PP=0, record hashes per request at each c
3. waved arm: MEMRA_DUAL_PP=1, replay same token sequence, compare every hash
4. pass iff all hashes match AND liveness counter advanced (see below)

One additional liveness counter (pp.rs:321 pattern: PRIME_SPLIT_CHUNKS, PRIME_PIPE_OVERLAPS):
`DUAL_PP_OVERLAPS` (AtomicUsize, relaxed ordering) bumped once per wave-pair that actually
overlapped (transition from 1 active stage to 2 active stages, measured via a second atomic
`DUAL_PP_ACTIVE_STAGES` with fetch_add/fetch_sub guards like pp.rs:345). The gate asserts this
counter advanced during the waved arm — a bit-identical walk that never overlapped is vacuous
(the slot-alternation code exists but never fired).

## (c) Event ordering

### Hazards

1. **Stage1(A) reading boundary slot while stage0(B) writes**: CLOSED by slot alternation (A uses
   slot 0, B uses slot 1) plus ev_rx → ev_tx fences. TX waits ev_rx (pp.rs:1042), so wave B's TX
   to slot 1 cannot start until wave A's RX from slot 1 (the PREVIOUS tick's wave B) completes.
2. **Same-slot reuse within one tick**: CLOSED by the wave sequence — wave B's stage0 finishes
   before wave A's next stage0 (next tick); the boundary step counter (pp.rs:473
   `AtomicUsize::fetch_add`) increments per TX, so slot assignment cycles A→B→A across ticks.
3. **KV cache cross-contamination**: CLOSED by per-session cache ownership (pp.rs:45-46 comment)
   — wave A's request 0 and wave B's request 1 each own distinct `caches[i]`, and the per-layer
   KV append (decode_batch.rs:1456-1462) writes to `kvl = cache.kv[il]`, not a shared buffer.
4. **Host-bounce interaction**: if MEMRA_PP_HOST_BOUNCE=1, boundary transport goes D2H(TX) →
   H2D(RX) through pinned staging (pp.rs:1074-1078 TX, :1118-1123 RX). The same ev_tx/ev_rx
   fences order D2H completion before H2D issue, so waves cannot race the staging slot. The
   existing two-slot-per-boundary design (pp.rs:565 `HostBounceRt::new`) already supports
   dual-active without new allocations.

### The invariant sequence

For wave A (slot 0) and wave B (slot 1):

```
tick N:
  A.stage0 enqueue
  A.TX wait ev_rx[0], write slot[0], record ev_tx[0]
  B.stage0 enqueue || A.RX wait ev_tx[0], read slot[0], record ev_rx[0]
  B.TX wait ev_rx[1], write slot[1], record ev_tx[1]
  B.RX wait ev_tx[1], read slot[1], record ev_rx[1]

tick N+1:
  A.TX waits ev_rx[0] — which was recorded AFTER A.RX read slot[0] in tick N
  B.TX waits ev_rx[1] — which was recorded AFTER B.RX read slot[1] in tick N
  → no write-after-write or read-after-write hazard across ticks
```

The spec-verify seam (opti0) proved this sequence correct at 220/220 rounds (RESULTS.md:72-82)
with zero slot collisions.

## (d) Increment plan

### Code seams (implementation guidance)

The dual-active fork point is `decode_step_batch_ppn` (decode_batch.rs:689). Today's serial walk:

1. Stage 0 scope (:812-829): `rt.enter(0)`, `pos_d` upload, embed, `decode_batch_layers` or
   `step35_decode_batch_layers` (range [fence[0], fence[1]]), TX → slot (alternating 0/1 under
   MEMRA_PP_OVERLAP, fixed slot 0 otherwise)
2. Middle stages loop (:831-841): for s=1..n_st-1: `rt.enter(s)`, `pos_d`, RX, range walk, TX
3. Last stage (:843-859): `rt.enter(n_st-1)`, `pos_d`, RX, range walk, output_norm + lm_head,
   `decode_batch_epilogue` (logits readback + sampling + pos bump), return host tokens

Dual-active transforms this into:

1. **Split caches** into wave A (`caches[0..mid]`) and wave B (`caches[mid..]`), where
   `mid = (b_n + 1) / 2` (ceil division). At c=1, skip to serial fallback.
2. **Wave A, stage 0**: `rt.enter(0)`, walk wave-A caches only through [fence[0], fence[1]], TX →
   slot_A (alternating 0/1 per tick)
3. **Wave B stage 0 || wave A middle stages**: on the host thread, issue wave B's stage 0, then
   immediately (no wait) issue wave A's middle stages. The streams order: stage-0 stream queues
   wave B, stage-s streams queue wave A's RX+walk+TX. Mark overlap (bump `DUAL_PP_ACTIVE_STAGES`,
   if went 1→2 bump `DUAL_PP_OVERLAPS`).
4. **Wave B middle stages + last**: after wave A finishes its last-stage TX, issue wave B's middle
   + last stages serially.
5. **Epilogue**: both waves' logits are now host-side; concatenate wave-A + wave-B results, return.

The EXACT-16 scope (decode_batch.rs:735-748) must wrap BOTH waves' walks — set once at fn entry,
applies to all stages per Engine (the per-Engine `AtomicBool`).

The `fence_stages_behind` call (:730) stays at fn entry — it orders stage streams behind the
caller before ANY wave issues, so no wave-A allocation can reuse a prior tick's pool block whose
primary-stream consumer is still queued.

The `publish_to` call (today's :858) stays at last-stage exit — wave B's last stage finishes
after wave A, so one `rt.publish_to(n_st-1, &e.stream())` after wave B's epilogue orders the
caller's stream behind all stage work.

### Increment 0: prototype at fixed c=8/16
- Fork `decode_step_batch_ppn` into `decode_step_batch_dual` (rollback seam:
  MEMRA_DUAL_PP=0/1, default OFF).
- Split `caches` into two waves (first half / second half); assert `caches.len() >= 2` else fall
  back to serial.
- Issue stage0(A), TX(A, slot 0), then parallel stage0(B) || stage1(A) (RX(A, slot 0)), TX(B,
  slot 1), then stage1(B) (RX(B, slot 1)).
- Measure N=5 interleaved A/B at c=8/16 on box1: dual vs serial aggregate tok/s. Gate: one-hash
  PASS, liveness counter advanced. **Kill if**: aggregate gain <+15% at c=16 — below that
  threshold the L-class complexity (double-buffering + scheduling) is not paid by serving wins.

### Increment 1: wave scheduler integration
- Generalize c=8/16 prototype to ARBITRARY even width c>=2: split into two equal waves. Odd c:
  assign the extra session to wave A (A gets ceil(c/2), B gets floor(c/2)).
- At c=1: REFUSE dual-active, fall back to serial. Do NOT split a single session — that breaks
  the numeric class.
- Wire the scheduler's chunk-fill path (worker.rs) to detect c>=2 and enable dual-active (still
  gated behind MEMRA_DUAL_PP).
- Gate: same one-hash matrix extended to c=1/2/3/4/5/6/7/8, with c=1 proving the serial fallback.

### Increment 2: interaction with prefill ticks
- Prefill (grouped prime) already owns the prefill path (pp.rs:302 `prime_pp_on`, :310
  `prime_pipe_on` for the chunk pipeline); dual-active decode does NOT touch prime.
- At serving tick transitions (prime → decode, decode → prime): ensure no wave A/B state persists
  across the boundary. Each decode tick resets wave split fresh from `caches`.
- Gate: mixed-tick serving run (prime 4096 solo, then decode c=8 x64 tokens, repeat 10x) with
  one-hash PASS.

### Kill criteria (increment 0)
- **<+15% at c=16**: the floor. The m2-pp8 1.88x receipt (at N=2/4 sharded, not the PP-2 serving
  config) bounds the ceiling; if first-cut dual-active on the box1 pair lands below +15%, the
  event-ordering complexity + double-buffer cost + wave-scheduler integration debt is not worth
  the serving gain. Kill, document the attempt, and move to candidate #1 (device-side sigmoid
  router, research/solgap-20260811 ranked higher by mechanism simplicity).

### Kill criteria (increment 1+)
- **Exactness break**: if ANY request in the extended one-hash matrix diverges, kill. Do not ship
  a statistically-correct wave scheduler — the contract is bit-identity.
- **Event-ordering flake**: if the c>=2 soak (x100 cross-device, the pp2-hardening precedent)
  fails ONCE with a slot collision or nondeterministic output, kill. The opti0 seam passed 220/220
  rounds clean, but that was spec verify (smaller state); batched decode moves more boundary bytes
  per wave. One flake = unfixed hazard.

## (e) What it does NOT touch

### Spec path (K=0 on PP-2)
The speculative verify path is already HOLD on PP-2 (research/specpp2-20260810: K=1 -18.8% c=1,
-42.8% c=2). Dual-active decode is plain-only; spec stays OFF on the pair until its own mechanism
fixes the c=2 cliff.

### Single-session c=1
At c=1 there is NO second wave to fill. The dual-active arm MUST fall back to serial (one stage0,
one stage1, identical to today's decode_step_batch_ppn at c=1). This is not a deficiency — it is
the honest contract: dual-active exploits batch parallelism, and a single session has no batch to
split.

The serving story: c=1 throughput (85-99 tok/s on box1, research/newbox-bench-20260811) remains
serial. The dual-active win accrues at c>=2 (119.9+ tok/s serial at c=2, 165.6+ at c=8); aggregate
climbs as concurrent load fills the batch. At c=64 full-window serving, today's 129.7 tok/s agg
projects to ~245 (+89%) if dual-active hits the 1.88x bound, or ~210 (+62%) at a conservative +60%
first cut.

Short TTFT (0.133 s, 4k 5.227 s on box1) is unchanged — prefill is separate, and its own chunk
pipeline (research/pp-prefill-20260807) already closed the prefill gap.

### Prefill/grouped prime
Grouped prefill (research/grouped-serve-20260810: 692.7 tok/s pp4096 solo, +62.3% over ungrouped)
and its PP-2 chunk pipeline (lane/cx-pipeline-prime, pp.rs:310 `prime_pipe_on`) are separate
mechanisms. Dual-active decode does NOT change the prefill path — each prefill tick materializes
its own per-chunk stage walks, and decode ticks do the same for their waves.

The only interaction: at tick transitions (prime → decode, decode → prime), the worker must ensure
no wave A/B staging persists. Increment 2's mixed-tick gate proves this clean.

### Serving integration (beyond increment 2)

After increment 2 gates pass, the serving path is:

- **Scheduler**: worker.rs chunk-fill produces `caches: Vec<&mut Cache>` per tick. If
  `caches.len() >= 2` and MEMRA_DUAL_PP=1, route to `decode_step_batch_dual`; else serial.
- **c=1 honest story**: single-session ticks remain serial (85-99 tok/s on box1,
  research/newbox-bench-20260811). The dual-active win accrues at c>=2 where batch fill provides
  two waves. Aggregate climbs as load fills: c=2 119.9 → ~170 (+42%), c=8 165.6 → ~275 (+66%),
  c=64 129.7 → ~210-245 (+62-89%).
- **Memory**: no new per-session allocation (KV unchanged, boundary slots are shared process-wide
  and already exist). Per-wave residuals are scratch (allocated + freed on stage streams per
  wave), not held state.
- **Default OFF initially**: behind MEMRA_DUAL_PP (default unset/0) until x100 cross-device soak
  passes (the pp2-hardening precedent). After soak: default ON, with MEMRA_DUAL_PP=0 rollback.

The short TTFT story (0.133 s, 4k 5.227 s on box1) is unchanged — prefill is separate, and its
chunk pipeline already closed that gap.

---

## Design confidence

- **Seam precedent**: opti0 (research/opti0-20260810) proved the TX-release boundary + slot
  alternation correct at 220/220 rounds, +13.94% c=2 end-to-end. The dual-active walk is the same
  event sequence, extended from 2-session verify to N-session batched decode.
- **Receipt bound clarification**: m2-pp8 (research/m2-pp8-20260802) measured 1.87-1.88x via
  **deferred readback** (tokens in flight across ticks, window 3), NOT chunk pipelining. That arm
  used `decode_step_h_ppn_deferred` keeping later tokens enqueuable while token t drains
  (m2-pp8:147-148). Dual-active targets CHUNK pipelining within ONE tick — split batch into two
  waves, overlap stage0(B) with stage1(A). The m2-pp8 1.88x is an upper bound (two balanced
  stages, full overlap), not a direct receipt — dual-active's realistic first-cut is +40-60%.
- **Prefill precedent**: research/pp-leverb-20260807/PROGRESS.md:68 notes prime chunk pipelining
  as "step 6" (explicitly deferred to a follow-up). That lane merged the per-stage prime walker
  without the pipeline; dual-active is the decode twin of that deferred step.
- **Effort class**: L (solgap ranks it L; the TX/RX seam exists, double-buffering exists, the
  per-row FA loop is unchanged). The risk is event ordering — hence the x100 soak gate before
  promotion.

## References

- research/solgap-20260811/REPORT.md — candidate #3, expected +40-90% agg at c>=16
- research/opti0-20260810/RESULTS.md — TX-release seam, 10.429 ms overlap receipted
- research/m2-pp8-20260802/RESULTS.md — 1.87-1.88x pipelined engine arm (8-stage, not PP-2)
- crates/memra-engine/src/pp.rs — boundary slots (:466), TX (:1014), RX (:1107), events (:458-465)
- crates/memra-engine/src/decode_batch.rs — serial walk (:787 step35, :1446-1481 per-row FA)
- research/specpp2-20260810 — spec HOLD on PP-2 (K=1 -18.8% c=1, -42.8% c=2)
- research/pp-prefill-20260807 — prefill chunk pipeline (separate mechanism)

## Skeleton commit

This design doc commits as a skeleton within 10 minutes of lane start (CLAUDE.md write-first
doctrine), then improves in committed increments. Next: read m2-pp8 RESULTS.md for the 1.88x arm
details, refine the wave-split arithmetic, and expand the gate matrix.

## AMENDMENT 2026-08-11 (post-review, binding on implementation)

**Hard requirement — TX path selection is fail-closed, not env-dependent.** Live
`PpNRt::tx()` alternates slots only under `MEMRA_PP_OVERLAP=1`; otherwise every TX pins
slot 0. An implementor reusing the serial `decode_step_batch_ppn` TX path without OVERLAP
would enqueue both waves into ONE residual buffer and silently corrupt PP-2 logits under
dual-active. Therefore:

1. `decode_step_batch_dual` calls ONLY `tx_pipelined` / `prepare_overlap_slots`
   (pp.rs:1014-1034 — always-alternating primitives). It must never route through the
   env-conditional `tx()`.
2. Boot check: if the PP boundary is single-slot (overlap slots not prepared), dual-active
   REFUSES with a quoted reason — it does not fall back to serial silently and does not run.
3. Gate matrix gains a NEGATIVE cell: dual-ON + single-slot boundary must refuse (exit with
   the refusal line), never produce output. A run that produces tokens in that cell is a
   gate FAILURE even if the bytes look right.
