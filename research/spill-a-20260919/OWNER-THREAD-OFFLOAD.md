# Taking copies off the CUDA owner thread's tick: what the current contracts allow (memra#536, lane A day 16)

Design note, no code beyond day 16's cancellation point. Input: `OWNER-THREAD-CENSUS.md` (the class table
at tree `21307b636`), the door's `TransferEngine` path (`crates/memra-engine/src/tier_transfer.rs`,
`worker.rs host_kv_planes_through_contract` and `host_kv_planes_from_contract`), the one-program law
(`CLAUDE.md` "One numeric program per request") and the capture law (memra#602: an entry is published on a
grid-aligned prime boundary or not at all). Every cell named here is pre-registered by shape only; none has
run. The stall cell of the census (`DAY16.md`) is the baseline they compare against.

## What the door already moved, and what still waits on the tick

Under `MEMRA_KV_HOST_CONTRACTS=1` the D2H demote (Option B) and the H2D promote (Option C) are one
`TransferEngine` batch each. What changed hands: the planes are registered leases with generations, the
copies are `CopyOp`s validated by the engine, completion carries a checksum receipt, the unwind is typed
(`Refused`, `SourceQuarantined`, `ReceiptMismatch`, `Latched`, `TicketLeaked`), and every fence is an
event the engine records. What did not change: the stream. `CudaTransfers::new(owner, governor)` takes
the worker's owner stream and stores it as its only stream (`tier_transfer.rs:355`, `stream: owner`);
`check_thread` refuses any other thread; `record_producer` records its fence on that stream; `submit_batch`
issues `memcpy_dtoh`/`memcpy_htod` on it; `synchronize(&ticket)` blocks the host on the per-item events
and the callers then drain the owner stream once more before `retire`. So on the ON side the copy is still
serialized behind the tick's kernels (same stream) and the tick is still blocked until the copy lands
(host waits). The day-15 pair measured the door's price for that shape on this card: demote 82 against
6.1 ms steady (the two hashes over cacheable memory and the ticket lifecycle), promote's own share 5.8
against 4.4 ms. The census cell (`DAY16.md`) measures what the tenant's tick absorbs in each case.

So the honest split is:

| class | leaves the tick today? | what holds it |
|---|---|---|
| D2H demote, OFF | no | `dtoh_u8_into_pinned` synchronizes the owner stream per plane |
| D2H demote, ON | no | same stream; `synchronize(&ticket)` + owner drain before publication |
| H2D promote, OFF | the copy is asynchronous with respect to the host (pinned source, `memcpy_htod`, no wait) but on the owner stream, so the restore and the next step queue behind it | stream order, not a host wait |
| H2D promote, ON | no | `synchronize(&ticket)` + owner drain; `require` against the D2H receipt before `ready_view` |
| D2D capture | on the owner stream, no host wait | stream order |
| D2D restore | on the owner stream, no host wait | stream order |
| prime | no (the owner thread computes it) | not a copy; a cancellation question, day 16 |
| trim | no | two owner-stream synchronizes plus `pool_trim_to_zero` |

The one class where the door's contract is already the right shape for an off-tick move is the pair it
covers: the `TransferEngine` API is stream-agnostic on its face (fences are events, completion is an
event per item, publication is `ready_view` after the consumer fence), and the only reason the copy sits
on the owner stream is the constructor's argument.

## Move 1: the door's D2H and H2D on a second stream (per device)

**What it needs, in the current contracts.**

1. `CudaTransfers::new_on(copy_stream, owner_stream, governor)`: the engine keeps two streams. The
   producer fence stays on the OWNER stream (`record_producer` records the event where the planes were
   last written); `submit_batch` makes the COPY stream wait on that event (`cuStreamWaitEvent`) and issues
   the copies there; the per-item completion events are recorded on the copy stream. `check_thread` stays:
   the worker thread still issues everything; nothing runs on another host thread, so the CUDA context
   ownership rule of the census is untouched. This is one constructor and two `stream.wait(event)` lines,
   not a new program.
2. The waits move from `synchronize` to `poll`. Today `host_kv_planes_through_contract` calls
   `t.synchronize(&ticket)` (host blocks) then `poll`. Off the tick the demote is submitted in tick N and
   `poll`ed at the top of tick N+1, N+2, ... until `Completion` is ready; `take_plane`, `bind_tier_image`,
   `retire`, `acknowledge` run then. The entry is in a new state between the two: `Demoting` (planes taken
   out of the device entry, not yet on the host list). The LRU must not serve it (it has no device planes)
   and must not evict-again or promote it; the ledger's in-flight dimension already refuses a second
   ticket for the same planes (day 15's `Capacity` refusal is the safety net, not the mechanism).
3. The consumer fence is what protects the SOURCE. For the D2H the source planes are device memory that
   the pool may reuse after the entry is dropped; today the entry drops after `retire_source`. Off the
   tick, `retire_source` runs when the completion is observed, so the planes' device leases live inside
   the engine's registry for one or more ticks: the ledger's device dimension (sized at three prefix
   budgets on day 16 of lane C) must cover a demote in flight plus the residents; the sizing rule needs
   one more term, "one entry in flight per direction", stated in `HOSTPREFIX-DOOR.md`.
4. The H2D promote already publishes through `ready_view` per item after `require` against the D2H
   receipt. Off the tick, `require` waits for the completion (a `poll` per tick), and the RESTORE (D2D
   into the session cache) must not start before the promoted planes are ready: the restore is on the
   owner stream, so `owner_stream.wait(consumer_event)` before the first `copy_u8_into` is the whole
   ordering; the request that hit stays in admission (`AdmissionRestoreRoute`) for the ticks the copy
   takes instead of blocking the worker. That is a scheduler state, "promoting", with the same shape as
   the step-OOM park (`requeue_oom` at the front of the queue).
5. What must not change: the bytes. The copy program is the same driver call on a different stream; the
   receipts (`Completion::require` against the demote's checksums) already prove the bytes crossed
   intact, and the promote's `MEMRA_KV_HOST_VERIFY` digest stays as the OFF-path check. The one-program
   law is not touched: no token is produced by a copy.

**Cells that decide it (pre-registered by shape).** (i) The census stall cell repeated with the second
stream: the tenant's ITL max during the demote and promote arms against the day-16 baseline, N=5 per arm
per order, both orders, door ON in both (the OFF arm is not affected by the move). Decision clause:
`stall_median(second stream) <= idle p99` of the same sitting for both classes, and the identity clauses
of C's `wc-pair.py` (`WC PAIR REPLAY: PASS`) unchanged. (ii) The fault gate `kv-host-contract-fault-gate.sh`
on the second stream: every one-shot fault (`contract-presubmit`, `contract-postpublish`, the four promote
faults) must produce the same typed outcome; the new state `Demoting`/`promoting` adds two faults, "the
completion is never observed" (the entry must latch off, `Latched`, never serve) and "the hit arrives
while the entry is `Demoting`" (a cold prime, never a partial restore). (iii) The identity gate
(`kv-host-spill-identity-gate.sh`) OFF and ON: byte-identical texts.

## Move 2: the D2D capture and restore off the tick

These are not under the door and have no engine contract today: `prefix_snapshot` allocates and copies
per layer on the owner stream inside the capture publish (inside `prefill_tick`), and `prefix_restore_at`
copies into the session cache at admission. They do not block the host, but they occupy the owner
stream between the tick's kernels, and the capture's allocation (`alloc_u8` per plane) is a pool call on
the worker thread.

**What a move needs.**

1. A second stream is not enough by itself: a D2D copy on a copy stream must wait on the producer (the
   prime that wrote the planes: an event recorded on the owner stream after the boundary chunk) and the
   consumer (the next step that may overwrite the source region, if the capture copies from a live
   session cache) must wait on the copy's event. For the CAPTURE the source is the live cache of a
   session that keeps decoding: the copy must complete before the session's next `decode_step` writes
   position `boundary + 1`... which it does not: the snapshot copies the prefix rows `0..boundary`, and
   the decode appends rows past the boundary; K/V planes are append-only per position, so the source
   rows are stable and only the recurrent states (`conv_state`, `ssm_state`, copied by `clone_dtod`)
   are overwritten by the next step. So: the recurrent-state copies stay on the owner stream at the
   boundary (they are small and must be exact at that position), and the KV plane copies (the bulk) may
   go to the copy stream with a producer event only. That split is the capture law's requirement
   restated as stream ordering: the entry is exactly the grid boundary's state or it is nothing.
2. Event-ordered publication: the entry enters the LRU as `Capturing` and becomes servable only when the
   copy stream's event has completed (polled at the tick top, the same `poll` shape as Move 1). A hit on
   a `Capturing` entry is a miss (cold prime), never a wait and never a partial restore: the capture law
   forbids the partial entry, and a wait would put the copy back on the tick.
3. For the RESTORE the destination is a fresh session cache and the source is a published entry, both
   stable: the copy can run on the copy stream and the session's first step waits on its event
   (`owner_stream.wait(event)`); the session sits in a `restoring` admission state for the ticks it
   takes, like Move 1's promote. Entry eviction must be refused while a restore reads it (a reader
   count on the entry, or the engine's `retain_device` twin as in Option C).
4. The one-program law is the whole risk here and it is not about bytes moved: it is about WHICH program
   produces the first token after a restore. Today the restore is synchronous, so the request's first
   step always runs on a complete cache. With an asynchronous restore the first step must never start
   early; the wait is an event on the owner stream, so the ordering is the driver's, but the gate has to
   prove it: the hit gate's identity clause (restored bytes equal cold bytes, `spec-on-cache-hit-gate.sh`
   plain and spec arms) and the continuation gate are the proof, run with the copy stream forced slow
   (a fault door that queues a delay kernel on the copy stream ahead of the restore, so the first step
   would run on an incomplete cache if the ordering were wrong; the digests must still match).

**Cells that decide it.** (i) The capture: a tenant's ITL during another request's grid-aligned seed
publish (a 4096-token prompt whose seed captures 4096 rows) against idle, N=5 per arm per order, both
orders, before and after; the entry must then hit (`cached_tokens = 4096`) with the cold digest. (ii) The
restore: the tenant's ITL during another request's 4096-token hit restore, same shape. (iii) The delayed
copy stream fault: hit gate and continuation gate green with the delay armed; any digest difference is
a FAIL of the ordering, never a tolerance.

## What stays on the tick, and why

- **Prime**: it is compute, not a copy; the owner thread runs it by design. What day 16 changed is that
  a gone client stops it at the next internal chunk (`progress::prime_cancel_point`) instead of at the
  end of the take. The pipelined walks (PP-2 split, ppN waves) keep the tick-top sweep as their
  cancellation point: returning mid-wave leaves another stage's work in flight against a cache the caller
  drops, and draining it is a wider seam than one check (the tainted-cache contract exists for exactly
  that shape). `prime_cache_batch` (several sessions in one call) has no per-session cancel.
- **Trim**: it exists to return memory to the driver at a quiescent point; the two synchronizes are its
  correctness. Off-tick trim is a different mechanism (a `cuMemPoolTrimTo` after evictions settle on the
  copy stream, the issue's item 4) and depends on Move 1's copy stream existing; not designed here.
- **Decode**: the tick.

## Move 1, first slice: what landed on day 17 (`DAY17.md`)

The D2H side of Move 1 is in, under the door: `CudaTransfers::new_with_copy_stream` (the second stream,
the D2H issued behind the producer fence's event with NO owner wait at submit), the route split into
`host_kv_planes_submit_contract` and `host_kv_planes_settle_contract` (`Block` is the day-16 program,
`Poll` hands the ticket back), the worker's one `Demoting` entry (`HostPrefixCache::demoting`), the
tick-top poll as the run loop's first statement (idle wait capped at 2 ms while pending), settle-first at
the hook, the promote hook and the tenant purge, publication through the shared `host_demote_publish`.
Items 1, 2 and 3 of the Move 1 list above are done for the demote in this shape; item 3's ledger term
("one entry in flight per direction") is not needed while every other route settles the pending demote
first, so the ledger's in-flight dimension stays one batch. Item 5 holds: the copy program is the same
driver call on a different stream and the receipts prove the bytes (the identity and fault gates).

What Move 1 still owes, in order:

1. **The promote (item 4).** `host_kv_planes_from_contract` keeps the owner stream and the blocking
   `synchronize`. Its turn: an H2D on the copy stream needs the OWNER stream to wait on the copy's
   event before the D2D restore reads the promoted planes (the one place the submit-time owner wait is
   the correct ordering, so the engine keeps that install for an H2D and moves only the issue stream);
   the request that hit stays in admission (`AdmissionRestoreRoute`) for the ticks the copy takes, a
   `Promoting` state with the step-OOM park's shape (`requeue_oom` at the front of the queue); the
   restore's first step must never start early, proven by the hit gate's identity clause and the
   continuation gate with the copy stream forced slow (the Move 2 delay fault applies here first).
2. **The receipt hashes.** Two checksums over the entry's pinned bytes stay on the owner thread inside
   the tick: `progress`'s completion checksum (at the poll, on the items that completed since the last
   poll) and `bind_tier_image`'s bundle checksum (at publication). Lane C's day-18 reading (77.9 ms per
   160 MB of cacheable pinned memory on the target host) makes them the remaining demote cost of the ON
   arm once the copy is off the tick. Decision cell: hash on a helper thread against the completion event
   (the frozen contract's `Completion.checksum` is a value, not a place), or a GPU-side digest of the
   source bytes recorded on the copy stream; either needs the tier crate's conformance to speak first.
3. **The by-reference routes.** The admission reclaim flush (`evict_all_demoting`), the pause sweep and
   the handoff keep the blocking program; the sink was the slice because its source is already evicted.
   The pause sweep's park half holds live device state and needs the `Demoting` state to protect the
   park until publication (the offload note's "source stays until the host copy publishes" rule).
4. **The decision cell (i) as pre-registered above** with both classes on the second stream, once the
   promote has moved; day 17 ran the demote arm alone against the day-16 receipt (`DAY17.md`).

## Move 1, second slice: the promote half (day 18, `DAY18.md`)

The H2D side of Move 1 is in, under the door: the engine issues every copy on the copy stream when one
exists and keeps the submit-time owner wait for an H2D (its destination's consumer is the owner stream;
a D2H still installs none); the door's promote decision moved from the admission body to the admission
loop before `admit(..)` (`host_promote_park_probe`, sharing the body's predicates), a host hit submits
(`host_kv_planes_submit_promote`) and the request PARKS on the requeue; the worker's one `Promoting` entry
(`HostPrefixCache::promoting`) is polled at the tick top (`host_promote_settle_pending`,
`host_kv_planes_settle_promote` under `Poll`) and published through the shared day-16 tail
(`host_promote_finish`) with the insertion pin held one tick; the parked request re-admits to a device hit.
Every other path settles it first (the hook, a demote of any route, a tenant purge; contract-only where
no device cache is in hand, the purged tenant's dropped); failures keep the day-16 typed outcomes and set a
one-tick memo so the request serves cold once; the fail-closed arm mirrors the lead's #622 ruling. Item 4
of the Move 1 list is done in this shape. Item 5 holds. The gates and the stall cell are in `DAY18.md`:
identity, failure, fault, hit and twin gates green on the target card in both door arms (two runs), and the
promote arm of the stall cell `stall_median=81.9` against day 16's ON `162.8` and OFF `85.0` (`at_off`, 3.1 ms
under OFF) and day 17's `86.4` (`promote_half_flat` by the pre-fixed six-ms threshold; moved 4.5 ms). Run 1
before the hook fix read `157.8` (`flat`): the hook settled a pending demote on every admission, so the parked
request's re-admission paid the demote copy synchronously; the settle-first now runs only before the hook's
own submission, the day-17 pre-registered rule.

What Move 1 still owes, in order:

1. **The settle-time owner wait for an H2D.** The engine keeps the submit-time `owner.wait(item event)`
   for an H2D, so kernels the tenant submits after the promote's submit queue behind the copy's landing
   (bounded by the copy time, about 6 ms per 160 MB on the target card). An engine method that installs
   the wait after `event_done` at the settle (the consumer fence recorded after it) would remove that
   bound; it changes the engine's `consumer_fenced` semantics for a copy-stream H2D and needs the tier
   crate's conformance to speak first. Decided by the day-18 stall reading of the promote arm.
2. **The receipt hashes** (unchanged from the day-17 list: `progress`'s completion checksum at the poll,
   `bind_tier_image`'s bundle checksum at a demote's publication, both on the owner thread inside the tick).
3. **The by-reference demote routes** (unchanged: the admission reclaim flush, the pause sweep, the handoff
   keep the blocking program; the pause sweep's park half needs the `Demoting` state to protect the park).
4. **The decision cell (i) as pre-registered above, both classes, same window.** Day 17 ran the demote arm
   and day 18 the promote arm, each against the day-16 receipt on the same box across sittings; the
   same-window interleaved A/B of both classes (door ON with the copy stream against door ON on the owner
   stream) has no arm today because the owner-stream program left with the slices; its shape is the
   day-16 script's `on` boot against a build of the day-16 tree, N=5 per arm per order, both orders.

Move 2 (the D2D capture and restore) is next in line: it reuses the copy stream and the tick-top poll shape
both halves of Move 1 now have, and its new work is the capture and restore contract itself (the
`Capturing` and `restoring` states, the delayed-copy-stream fault, the event-ordered publication).

## Order of work, priced

Move 1 is the bounded one: the contract already has fences, tickets and typed unwinds, and the copy
stream is a constructor argument plus two `wait(event)` calls; the new work is the `Demoting`/`promoting`
states in the LRU and admission, the ledger's in-flight term, two fault cells and the decision cell.
About two agent-days including the cells on the target card. Move 2 is a new contract (capture and
restore have none today) and its correctness surface is the restore ordering under the one-program law;
about four agent-days, after Move 1, because it reuses Move 1's `poll`-at-tick-top shape and its copy
stream. Neither move is started by this lane without the lead's ruling; the census and the stall cell
are the input.

## Move 1, the first owed item, day 19: rule 3 in the tier crate (`DAY19.md`)

The settle-time owner wait for an H2D now has its conformance: `crates/memra-tier/src/conformance/reader_fence.rs`
(rule 3, unversioned beside the frozen schedules, the day-11 shape). `consumer_fenced` is the installed wait on the
reader's stream, never a flag and never the copy's landing; the schedule `h2d_reader_fence` is the same under both
installs (`ReaderWaitInstall::AtSubmit`, the engine's day-18 program; `AtSettle`, the owed program) and
`h2d_reader_issued_before_its_wait_is_unordered` is the forbidden order. CPU bindings
(`tests/contracts/reader_fence_bindings.rs`): the at-submit install passes, the at-settle install with a wait on the
reader stream passes, and the red arm (a settle-time flag with no reader wait) fails the schedule. The engine change
(the wait installed at the settle, after `event_done`, before `ready_view`, the consumer fence recorded after it)
is allowed by the schedule and lands only with the identity gate OFF and ON on the target card; `DAY19.md` records
whether it did.

## Move 2 pre-registration (day 19, committed before any Move 2 code): the D2D capture and restore off the tick

Input: the census (`OWNER-THREAD-CENSUS.md` rows "prefix snapshot capture (D2D)" and "hit restore (D2D)"), the
day-16 stall cell (`stall_cell.py`, its prime arm), the twin gate's fitted entry shape on the 27B
(`rtx5090-day18/twin27-off/shape.json`: `fit_fixed_bytes=157.3 MB`, `fit_bytes_per_token=29,564`), lane C's day-22
observation (a pool-full demote under `TENANT_PCT=100` runs the whole contract copy before `insert` refuses it: a
Move 2 capture must ask the budget BEFORE it copies, never after), and rule 3 above.

### What a capture copies, what a restore copies, where each runs today

**Capture** (`prefix_snapshot`, `worker.rs`, called from the capture publish inside `prefill_tick`: the LCP-split
boundary, the grid seed through `maybe_prefix_seed`, and the DFlash/glm5 drains). Per layer with a KV plane: two
`engine.alloc_u8` (a pool call on the worker thread) and two `engine.copy_u8_into` (K and V, `len * k_tok_bytes` and
`len * v_tok_bytes`, D2D on the owner stream); per recurrent layer: two `engine.clone_dtod` (`conv_state`,
`ssm_state`, f32, an alloc plus a copy each); the MTP draft plane is one more KV slot of the same shape; latent
(MLA) planes and glm5 TP shards only under their own flags (out of scope here: a Move 2 entry that carries them
keeps the tick program, refused by name as the door refuses them at demote). No host wait: the copies are
stream-ordered behind the prime that wrote the source; the insert that follows may evict, and an eviction into the
host tier is the demote class (Move 1). On the 27B (65 layers, the served artifact of the day-16 cell): the 64-token
grid seed is 159.8 MB (measured, `DAY16.md`), of which about 157.3 MB is the recurrent state (fixed per entry) and
about 1.9 MB the KV rows; a 4096-token entry is about 278 MB (157.3 fixed plus 4096 x 29,564); the day-16 prime
arm's 5120-token seed about 309 MB. The recurrent state dominates every capture on this model: it is a fixed 157 MB
of `clone_dtod` per entry, and it is the part that MUST be exact at the boundary (the next decode step overwrites
it), so it is the part that cannot leave the boundary's stream position.

**Restore** (`prefix_restore_at`, called from admission for a device hit, `AdmissionRestoreRoute::Native`, and after
a host promote publishes). Into the request's fresh session cache, per layer: `engine.copy_u8_into` of K and V for
`restore_len` rows, `set_i32_one(len_d)`; per recurrent layer `engine.copy_into` of `conv_state` and `ssm_state`;
`cache.pos = restore_len`; then the request's first prime chunk on the same stream. Same byte counts as the
capture at the restored length (a 64-token hit restores 159.8 MB, a 4096-token hit about 278 MB). No host wait;
stream-ordered; the source entry is leased for the restore (`source lease released after restore fence`).

**Today** both run on the owner stream, synchronously with respect to the tick's kernel order, inside the tick
(capture) or inside admission on the worker thread (restore). Neither blocks the host; both occupy the owner stream
between the tenant's kernels, so the tenant's next step queues behind up to 160 to 310 MB of D2D per event. The
day-16 prime arm (`stall_median=301.5` ms) is the prime plus its capture and cannot separate them; cell (i) below
does.

### The one-program law under an event-ordered publication

No token is produced by a copy, so the law is about WHICH program produces the first token after a restore and
WHAT an entry holds when it is served. Two clauses, both pre-registered as gates, not tolerances:

1. **A captured entry is published only after its copy's event.** The entry enters the LRU as `Capturing` and
   becomes servable only when the copy stream's completion event has been observed complete at the tick top (the
   Move 1 poll shape). The KV plane copies may run on the copy stream behind a producer event recorded on the owner
   stream after the boundary chunk; the recurrent-state copies (`clone_dtod` of `conv_state` and `ssm_state`) STAY
   on the owner stream at the boundary, because the next decode step of the captured session overwrites them and a
   copy-stream read would race it. The K/V rows `0..boundary` are append-only per position and stable. So the
   capture law (memra#602, a grid-aligned boundary's state or nothing) holds by stream position for the recurrent
   state and by event for the rows.
2. **A restore's destination is readable only after its event, before the request's first prime chunk.** The
   restore copies run on the copy stream behind a producer event (the source entry's publication, already
   complete); the OWNER stream waits on the restore's completion event before the first prime chunk is issued
   (rule 3's reader fence: the wait installed on the reader's stream, then the consumer fence recorded after the
   prime chunk). The request sits in a `Restoring` admission state for the ticks the copy takes (the step-OOM park's
   shape, as `Promoting` on day 18); it never blocks the worker and its first step never runs on an incomplete
   cache. The proof is the hit gate's identity clause and the continuation gate, run with the copy stream forced
   slow (the delayed-copy-stream fault below): any digest difference is a FAIL of the ordering, never a tolerance.

### States and what every path does on meeting them

`Capturing` (one per worker, the entry's planes allocated and being written by the copy stream, not in the LRU's
servable set) and `Restoring` (one per worker, a request parked with a fresh cache whose planes the copy stream is
writing):

| path | on a `Capturing` entry | on a `Restoring` request/entry |
|---|---|---|
| hit lookup | a MISS (cold prime); never a wait, never a partial restore | the source entry is leased by the restore: servable to others (reads only), not evictable |
| eviction (capacity, LRU) | not a candidate (not in the servable set); if room is needed, the LRU evicts servable entries or refuses the capture at publication (the capture is dropped, its planes freed after its event) | the source entry is leased: skipped, as any leased entry is today |
| admission reclaim (`reclaim-on-defer`) | settles the capture first (a host wait on its event, then publish or drop), then reclaims: the reclaim's accounting must see a published or a freed entry, never a half-written one | the parked request holds its booked KV bytes; the reclaim may not take its fresh cache; the source entry is leased |
| trim (`pool_trim_to_zero`) | settles first (trim needs a quiescent pool; the capture's allocations are live) | settles first (the restore's destination is live) |
| demote (any route, Move 1) | a `Capturing` entry is not demotable (not published); the demote takes the next LRU candidate | the source entry is leased: not demotable while the restore reads it (Option C's `retain_device` twin or a reader count) |
| tenant purge | the purged tenant's `Capturing` entry is dropped unpublished after a host wait on its event (planes freed; never freed under a running copy); another tenant's stays | the purged tenant's parked request is shed with a host wait on its restore event before its cache is freed; the source entry, if the tenant's, is dropped after the lease releases |
| shutdown / model unload | a host wait on the event, then drop; no publication after a stop | the same; the parked request is failed with a typed line |
| a second capture while one is pending | settles the pending one first (never two in flight; the Move 1 rule) | n/a |
| a second hit on the same entry while a restore is pending | serves from the published source (a second restore may queue behind the first, one per worker, else cold) | n/a |
| tier latched off (Move 1's `Latched`) | unaffected (the capture is device-only) | unaffected |

### Failure paths

Typed, one line each, never silent: a copy-stream allocation refusal at capture (`capture refused: device alloc of
N B failed`, the entry is not created, the tick continues); a completion never observed (an event that never
completes or a quarantined poll: the entry is dropped unpublished after a bounded host wait and the capture path
latches to the tick program for the boot, `[prefix-cache] CAPTURE OFF-TICK DISABLED`); a restore whose event is
never observed (the parked request is failed with a typed line and its cache freed after a host wait; the restore
path latches likewise); a restore source evicted under a pending restore (impossible by the lease; asserted by a
census and a fault); a publication attempted before the event (a debug assertion and a gate teeth cell). Every
failure keeps the OFF program's line where one exists and adds none where none exists.

### The contract in `memra-tier` terms

A D2D contract beside the D2H and H2D ones, additive: `TransferOp::D2d(ContiguousCopy)` on ONE device (the
existing `ContiguousCopy` shape: source and destination `DeviceLease`s, spans, epochs, a producer fence; `p2p` is
its two-device sibling and stays as is). Ticket: one per capture or restore batch (all planes of the entry, one
`Epochs`). Producer fence: an event recorded on the owner stream after the boundary chunk (capture) or after the
source entry's publication (restore). Consumer fence: for a capture, none on the owner stream (the destination's
consumer is the LRU, whose "read" is the publication after `event_done`, like a D2H's host consumer); for a restore,
rule 3's reader fence (the owner stream waits on the item event before the first prime chunk; the consumer fence is
recorded after it). Receipt: byte count, epochs, completion event, and a checksum term. The checksum is the open
design point: `progress` hashes HOST bytes today and a D2D has none. Two candidates, to be decided by a CPU
schedule before any GPU code: (a) a device-side digest recorded on the copy stream after the copy (a Memra-native
reduction kernel over the destination bytes, its CPU oracle the same function over a D2H readback in the gate;
one kernel, not a numeric program for tokens), compared against the same digest of the source; (b) no checksum for
D2D: `Completion::require` grows an additive `Checksum::Unwitnessed` arm for a D2D item whose bytes crossed a
single device's memory under one driver call, with the identity binding (`IdentitySlot::bind`) carrying the entry's
identity. (a) proves the bytes, (b) proves nothing about them and is the OFF program's level of evidence. The
pre-registered choice is (a) unless the digest's cost on the copy stream exceeds the copy's own time on the
target card (measured, N>=5, both orders); (b) then stays a diagnostic. The frozen schedule: `d2d_capture_publish`
(a captured item is `NotReady` until its event; `ready_view` requires the receipt; a publish before the event is a
schedule failure) and `d2d_restore_reader_fence` (rule 3 re-bound to a D2D: the reader stream's wait, the early read
unordered), both with CPU bindings and a red arm each, in `conformance/` beside rule 3.

### Decision cells (pre-registered by shape; none has run)

All on the target card class (one RTX PRO 6000 Blackwell, 600 W), through the collector, `N=5` per arm per order,
both orders, replay `PASS`, errors 0, the tenant's text byte-identical, the day-16 harness with two new arms:

- **(i) capture arm.** The tenant streams its 160 tokens; the intruder is a 4096-token grid-aligned seed whose
  capture publishes about 278 MB (recorded as the tokenizer made it). `stall_median` before (the tick program) and
  after (the copy stream); decision clause `stall_median(after) <= idle p99` of the same sitting; the entry must then
  hit (`cached_tokens` equal to the published length) with the cold digest.
- **(ii) restore arm.** The same, the intruder a 4096-token hit whose restore copies about 278 MB; the same clause;
  the request's text equal to its cold text.
- **(iii) the delayed copy stream.** A fault door (`MEMRA_KV_HOST_FAULT=d2d-delay`, one-shot, its FLAGS.md row and
  decide-by in the same commit) queues a delay kernel on the copy stream ahead of the restore; the hit gate
  (`spec-on-cache-hit-gate.sh`, plain and spec arms) and the continuation gate must stay `ALL GREEN`: the first step
  would read an incomplete cache if the ordering were wrong, and any digest difference is a FAIL.
- **(iv) identity, captured and restored entries, OFF versus ON.** `kv-host-spill-identity-gate.sh` default and plain
  with the Move 2 door OFF and ON, plus the twin gate on the 27B; every line `ALL GREEN (teeth=0)` and `-> PASS`.
- **(v) the capture digest's price** (only under receipt choice (a)): the digest kernel's time on the copy stream
  against the copy's own time, same window, both orders.

### Slices and their price

1. **Slice 1, the capture on the copy stream with an event-ordered publication** (the first slice, named): the
   D2D contract's CPU schedule `d2d_capture_publish` and its bindings; `TransferOp::D2d` in the tier crate;
   `CudaTransfers` issuing a same-device copy on the copy stream behind the producer event; `prefix_snapshot`
   splitting the recurrent-state `clone_dtod` (owner stream, boundary) from the KV plane copies (copy stream) under a
   door; the `Capturing` state and its tick-top settle; cells (i) and (iv). About 1.5 agent-days including the
   target-card cells.
2. **Slice 2, the restore behind rule 3's reader fence**: `Restoring`, the parked request, the reader wait before
   the first prime chunk, the source lease, cells (ii), (iii) and (iv). About 1.5 agent-days.
3. **Slice 3, the receipt**: the digest kernel or the `Unwitnessed` arm by the pre-registered rule, cell (v), the
   fault gate's D2D cells. About 1 agent-day.

Total about four agent-days, as priced on day 16. Slice 1 is first because its publication rule (an entry is
servable only after its event) is the smaller correctness surface and its state (`Capturing`) is the one every
other path already knows how to meet (a miss); slice 2's `Restoring` touches admission and the first-token program
and rides on rule 3, which now exists.
