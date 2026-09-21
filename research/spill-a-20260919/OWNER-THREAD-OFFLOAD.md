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

## Order of work, priced

Move 1 is the bounded one: the contract already has fences, tickets and typed unwinds, and the copy
stream is a constructor argument plus two `wait(event)` calls; the new work is the `Demoting`/`promoting`
states in the LRU and admission, the ledger's in-flight term, two fault cells and the decision cell.
About two agent-days including the cells on the target card. Move 2 is a new contract (capture and
restore have none today) and its correctness surface is the restore ordering under the one-program law;
about four agent-days, after Move 1, because it reuses Move 1's `poll`-at-tick-top shape and its copy
stream. Neither move is started by this lane without the lead's ruling; the census and the stall cell
are the input.
