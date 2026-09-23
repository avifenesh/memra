# WP-A day 33: the promote's staging fill lands inside the tick that hands it (the day-32 lever)

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`. Rig: the local RTX 5090 Laptop GPU now; the target card (one RTX PRO
6000 Blackwell) when the lead restores it. No cross-card comparison. Every cell `executed-not-qualified`. Every push in
the announced `MEMRA_RELEASE_QUALIFICATION_MODE=development` mode. Scope: the lead's day-33 brief (ruling 47 on DAY32):
bring DAY28 clause 1b back under its registered bound on the H2D binary without giving back B2.

## 0. Resync

`git fetch`: `origin/main` `649d96210` (integ52, PR #684, still open); the lane tip `e009df6d3` already carries it
(merged as `17be62a14`). The worktree was clean. The target card was reclaimed after the day-32 sitting; no connection
is made to its old address.

## 1. The read, before any design (the day-32 receipts, `pro-single-day32/box/`)

Where the day-32 promote's ten extra milliseconds come from, read off the ON boots' lines, not guessed:

- **Pre-H2D binary**: the probe submits the KV batch (the ticket's submission instant), then uploads the 96 recurrent
  planes synchronously on the owner stream (4.58 ms owner segment), then the tick's decode step runs; the next tick-top
  poll finds the ticket complete: `promote published off the tick: ticket complete after 1 poll(s), 19.7ms from
  submission to completion`. The entry publishes at tick N+1; `promote_in median=20.7`.
- **H2D binary (day 32)**: the probe hands the fill to the helper and parks (0.01 ms); the helper fills in 6.40 ms while
  tick N's decode step runs; the tick N+1 top lands the fill and submits the KV batch and the spans (0.62 to 0.67 ms);
  the copy (about 3 ms for 157 MB pinned, DAY13 `2.98` per 160 MiB) is not done at that instant, so the ticket completes
  at the tick N+2 top: `ticket complete after 2 poll(s), 16.0ms from submission to completion`. The entry publishes at
  tick N+2; `promote_in median=31.1`. The tenant's tick in this cell is `idle_p50(tick)=13.47`.

So the extra tick is not the fill's 6.4 ms: it is the SUBMISSION moving from tick N to tick N+1. The copy stream is
idle between the probe and the tick N+1 top. Landing the promote in one tick again means submitting at the probe, which
means the copy stream, not the owner thread, must wait for the fill.

## 2. Pre-registration (committed before any day-33 code)

**Options, priced by arithmetic.**

- W (the owner waits for the fill at the probe, bounded): the owner holds the fill's 6.4 ms (plus the 0.7 ms submission)
  inside tick N. B2's owner segment becomes about 7 ms: (a) and (b) cannot both hold by construction. Refuted.
- C (the owner waits for the copy after the tick N+1 submission, bounded): publishes at N+1 again, but the owner holds
  about 3 ms more per promote (the copy); B2's field would not see it (it ends at the submitted line), so it would pass
  while the owner gives most of the day-32 win back. Refuted as gaming B2.
- P (a mid-step poll: submit from inside the decode step's host wait): touches every decode path's blocking wait, and
  the producer fence recorded mid-step orders the copy behind the step's kernels, so it still lands after N+1's top.
  Refuted.
- G (a host-written flag the copy stream waits on, `cuStreamWaitValue32`): the owner submits at the probe; the helper
  flips the flag after its fill. Needs host-mapped flag memory, a buffer shared across threads, and a release path for
  a stream blocked on a flag that a gone helper never writes. Priced and not taken: more unsafe surface and a new
  wedge class for the whole copy stream.
- **F (taken): the fill is a host function ON THE COPY STREAM, ahead of the span copies.** Below.

**Design F.**

1. At the admission probe (tick N), on the off-tick contract route: one staging buffer per resident recurrent plane
   from the context's set (charged when fresh, the day-31 rule), then the unchanged submission
   (`device_entry_from_host_parts`, `OffTick`): the KV batch, then the spans through a new engine call,
   `CudaTransfers::submit_h2d_spans_filled(ticket, spans, fills)`. It runs the same rule-1 admission as
   `submit_h2d_spans` (except that a source need not be written yet; each fill must be exactly its source's length),
   the owner-stream event and the copy stream's wait, then ONE `cuLaunchHostFunc` on the copy stream whose function
   copies each resident plane (an owned `Arc<Vec<f32>>` clone the function holds) into its span's staging buffer, then
   one `cuMemcpyHtoDAsync` and one event per span. Stream order puts every span copy behind the fill; the span's event
   observed complete implies the fill ran. `take_h2d_spans` marks each source written. The request parks once, as on
   the pre-H2D binary; the next tick-top poll finds the ticket complete if the copy stream's work (KV items, the fill,
   the span copies) took less than the rest of tick N.
2. Removed with it (door hygiene): the day-32 `Filling` phase (`PendingFill`, `host_promote_fill_handoff`,
   `host_promote_fill_step`, the helper's `Fill` job and its reply channel, the `promote staging fill off the tick`
   line). The hash helper goes back to one job kind. The resident form `Arc<Vec<f32>>` stays.
3. The owner thread waits for nothing new: no host wait on the fill, the copy or any event is added on any promote path;
   the tick-top settle stays a `Poll`.
4. `unsafe`: the day-31 design item 2 said "no `unsafe`" for the helper's fill; F departs by name. The host function is
   an `extern "C"` trampoline (it catches any unwind) over a boxed task that owns the plane `Arc`s and the staging
   buffers' raw pointers; its safety contract is the span's: the engine owns each staging buffer from the attach until
   `take_h2d_spans`, which runs only after the span's event, which stream order puts after the host function; no one
   else reads or writes the buffer in between. Confined to `pinned_host.rs` and `tier_transfer.rs`.
5. Fail closed: a staging refusal, a destination alloc refusal or a rule-1 refusal hands every span back (staging to the
   set, destinations dropped, nothing enqueued) and unwinds the KV ticket typed (`tier H2D spans refused: ..`, the tier
   on); a host-function launch error before any copy is a rule-1 refusal (the task comes back and drops); an enqueue or
   event error after the first span copy quarantines the ticket (the engine keeps every span; the settle latches). The
   host function cannot block and cannot fail once launched (lengths checked at the attach).
6. The line: `promote submitted off the tick: .. N items on the contracts door's copy stream, 96 f32 spans filled on
   the copy stream (156.9MB); owner segment X.XXms; request parked` (the field before `request parked`, as day 32). The
   H2D receipt line is unchanged (`; 96 f32 spans landed under the ticket and taken back before the retire`).
7. Conformance first: `conformance/h2d_span.rs` gains rule 6, the fill (additive, unversioned, `WIRE_VERSION` stays 1):
   a filled span's copy is ordered behind its fill; the red arm is a binding that issues a span's copy before its fill
   (the copy reads unfilled bytes). CPU binding and the native cell.
8. The fault gate's `promote-span-refusal` cell keeps every assertion except the day-32 `the staging fill ran on the
   helper before the refused submission and before the next one`, which names the removed phase; it becomes `the next
   promote's submission carries its spans filled on the copy stream` (one such submission after the refusal). Stated
   here, before any run.

**Acceptance, stated before running (target card, 27B, the day-26 double-park cell's shape, both binaries in one hold:
the day-32 H2D binary and the day-33 binary).**

- (a) DAY28 clauses 1a, 1b and 1c exactly as registered on day 28 (`day28-reading.py`, unchanged): 1a stall ON at most
  OFF + 2.0 per order; **1b e2e ON minus OFF at most +20.0 per order**; 1c owner in-completion median at most 12.0.
- (b) B2's rule unchanged: the promote's owner segment (the `owner segment` field) median at most 1.5 ms and max at
  most 3.0 ms over at least 20 steady promotes (the second and later of each ON boot), and every steady submission
  carries its spans (`f32 spans filled on the copy stream`).
- (c) B1, B4 and B5 unchanged: B1 the gate set (identity x4, failure x2, the fault gate every cell, twin x2, hit OFF/ON
  armed with the day-24 census, the unit cells) ALL GREEN, and on the 5090 identity default ON, fault default and plain,
  hit OFF/ON; B4 every H2D receipt names 96 spans (48 on the 9B) with `items=` 32 / 34 (16 / 18); B5 no new numeric
  program, `verify ok: promoted state digest matches demote digest` on the identity ON arms, the native cell reading the
  promoted planes bitwise equal to the resident bytes.
- (d) **The owner thread's added wait per promote is zero**: a census proves no host wait (no `synchronize`, no
  `recv_timeout`, no event wait) on the promote's owner path between t0 and the submitted line and that the tick-top
  settle stays `ContractWait::Poll`; and measured: the owner segment's steady median is at most 0.90 ms (day 32's 0.70
  plus at most 0.20 ms for moving the submission to the probe and the one host-function launch) and its max at most
  1.50 ms.

**Predictions, stated before running.** Copy-stream work after the probe: KV items about 0.1 ms, the fill about 6.4
ms (the day-32 helper's figure; the driver's callback thread does the same memcpy), the span copies about 3.0 ms, so
about 9.5 ms, inside the rest of tick N (about 13.5 ms). Hence: `ticket complete after 1 poll(s)` on at least 90 percent
of steady promotes, about 13 to 16 ms from submission to completion; `promote_in` steady about 16 to 18 ms (day 32:
31.1; pre-H2D: 20.7); **1b about +6 to +12 ms per order** (day 32: +22.5 / +22.2; pre-H2D: +12.2 / +12.1); 1a stall ON
about 75 to 77 ms against OFF about 85; 1c about 2.0 ms; the owner segment steady median about 0.7 to 0.9 ms. (a) and
(b) can both hold by construction: nothing moves onto the owner thread. If the copy stream's work outlasts tick N on
some promotes (a slower fill), those take two polls and 1b rises toward the day-32 figure; that is a result, reported
as it falls.

**The 5090 before the sitting (development readings, not clauses).** The gate set above on the 9B, and a promote-mode
stall reading (`stall_cell.py --mode promote`, the gate cache of 64 MB so the 54 MB entries evict) on the day-32 binary
and the day-33 binary, reading the poll count, submission to completion and `promote_in`. No 5090 number is compared to
the target card's.

**Budget.** 0.5 agent-day plus card time: the conformance rule and binding, the engine call and its native cell, the
server's probe submission and the removal, the gate cell's one assertion, the 5090 half; the target-card sitting waits
for the restore. If budget remains after the 5090 half: the D2D half's pre-registration only (no code).
