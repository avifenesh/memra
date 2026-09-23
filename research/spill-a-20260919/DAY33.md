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

## 3. What landed, in the pre-registered order, one census each

1. **Tier rule 6** (`f814af107`): `conformance/h2d_span.rs`
   gains `H2dFillFixture` and `h2d_span_fill_ordered_before_its_copy` (additive, unversioned, `WIRE_VERSION` stays 1);
   the CPU binding's red arm is a binding whose copies run when offered, before the fill (`h2d_span_red_arm_a_copy_before_its_fill_fails`).
   Tier contracts `h2d_span` 5 passed.
2. **The engine call** (`82472ff12`): both H2D span attaches share one body (`attach_h2d_spans`);
   `submit_h2d_spans_filled` runs the rule-1 admission (a source need not be written; each fill exactly its source's
   length), the owner-stream event and the copy stream's wait, ONE `cuLaunchHostFunc` whose task copies every resident
   plane into its staging buffer, then one copy and one event per span; a launch error hands every span and fill back;
   `take_h2d_spans` marks each source written. `unsafe` is confined to `pinned_host.rs` (`fill_target`,
   `enqueue_to_device_f32_after_fill`) and the task; the trampoline never unwinds across the FFI. Census
   `h2d_span_rules_are_as_stated` (rule 6's order, one launch site); native cell
   `h2d_span_filled_batch_fills_on_the_copy_stream_before_its_copies` (a wrong-length fill refused whole; a 300 ms
   copy-stream hold keeps the fill and copies queued while the KV item lands; after the reader wait each staging source
   reads its plane's bytes and each destination the plane, bit for bit; the attach under 100 ms; the injected
   second-enqueue fault quarantines).
3. **The server** (`045ab57d4`): the probe takes the staging (`host_promote_stage`) and submits in its own step through
   the unchanged route, the spans through `submit_h2d_spans_filled`; the submitted line reads `.. 48 f32 spans filled on
   the copy stream (52.7MB); owner segment 0.47ms; request parked` (9B). Removed with it: the day-32 Filling phase, the
   helper's Fill job and reply channel, `HostHelperJob`, the `promote staging fill off the tick` line and the two day-32
   Filling CPU cells. The fault gate's `promote-span-refusal` cell swaps its one day-32 assertion for `the next promote's
   submission carries its spans filled on the copy stream` (DAY33 item 8; the helper run red and green,
   `rtx5090-day33/gatecheck/helper-red-green.txt`). Census `day33_the_promote_submits_its_filled_spans_in_the_probes_tick`
   (the probe's order; no `synchronize(`, `recv_timeout(`, `reply_within(` or `ContractWait::Block` on the path from t0
   to the line, the census half of (d); nothing of the Filling phase left). Server lib 876 passed; clippy `-D warnings`
   clean on tier, engine and server.
4. **The owner-hold cell** (`aad1e1797`): `h2d_fill_host_function_does_not_hold_the_owner_thread`, written after the
   5090 stall reading (section 4), queued for the card's next free window.

## 4. The 5090 half (`rtx5090-day33/`, binary `1f890107872d21722cf7ed596ee831b719e2bb7d3f6a53d1ce99659d1e7bde1c` at `045ab57d4`)

Unit cells under `/tmp/memra-5090.lock`, serial: the door's GPU cells `test result: ok. 16 passed` (with
`option_c_spans_ride_the_promote_ticket_and_land_bitwise`: the promoted planes bitwise equal to the resident bytes
through the copy-stream fill); the engine's span cells `ok. 3 passed` (`d2h_span_batch_..`, `h2d_span_batch_..`,
`h2d_span_filled_batch_..`). A lane B `memra-server` was resident on the card during the engine cells (the lock was free;
`unit/card-apps-at-engine-cells.txt`); correctness cells only, recorded.

The battery (`battery.log`, 16:00:33Z to 16:05:29Z, each gate under its own `flock`, the 9B NVFP4 MTP artifact):
identity default ON `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` (12 ok; `verify ok: promoted state digest
matches demote digest`, the H2D receipt `checksums_sha256=ed284403..` equal to day 32's); fault default and plain
`KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` (160 ok each; the new assertion `refusal at 74, filled span submissions 1,
after the refusal 1`, `byte-unequal request(s): none`); hit OFF `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` (61 ok), hit
ON the same (68 ok). `DAY32 B4 receipts=13 bad=0 by (items, spans)=[((16, 48), 6), ((18, 48), 7)] ... -> PASS`;
`DAY30 A3 ... receipts_paired=88 ... -> PASS`. **(c) on the 5090: B1 and B4 hold; B5's native cell holds.**

The promote-mode stall reading (`stall/`, one hold 16:07:14Z to 16:09:50Z, five boots, `STALL REPLAY: PASS` 5 of 5;
`reading-day33-stall.log`), verbatim:

- `DAY33 5090 READING arm=d32-on boots=2 owner-segment N=20 median=0.41 ... | completion ms N=18 median=8.85 min=8.70
  max=9.90 polls [1] (counts [18]) | promote-in N=18 median=16.90 ... | intruder-e2e N=20 median=78.04 ...`
- `DAY33 5090 READING arm=d33-on boots=2 owner-segment N=20 median=0.41 min=0.37 max=0.67 submissions=20 filled=20 |
  completion ms N=18 median=16.40 min=16.00 max=18.50 polls [1] (counts [18]) | promote-in N=18 median=17.10 ... |
  intruder-e2e N=20 median=78.53 ...`
- `DAY33 5090 READING arm=off ... promote-in N=9 median=10.40 ... | intruder-e2e N=10 median=62.11 ...`

**The lever did not move on the 5090.** `promote_in` 16.90 (day 32) against 17.10 ms (day 33), the intruder's e2e
78.04 against 78.53 ms. The owner segment is the same 0.41 ms. What changed is where the time sits: on the day-33
binary the ticket is complete at the FIRST tick-top poll after the submission, but that poll comes 16.4 ms after it
(one poll, not two), while the tenant's tick here is about 7.4 ms. The owner loop took about two ticks to come back
around after the probe that submitted. The prediction of section 2 assumed the copy stream starts at the probe and the
next tick top comes one tick later; the reading says the tick that carries the submission is about twice as long.
Section 5 tests the one mechanism the code can hold responsible (the host function holding the owner thread's CUDA
work) before any target-card time is spent.

## 5. The owner-hold cell, and what it rules out

`h2d_fill_host_function_does_not_hold_the_owner_thread` on the local RTX 5090 (18:21:43Z, one hold of
`/tmp/memra-5090.lock`, no compute app on the card; `rtx5090-day33/owner-hold/owner-not-held.log`), verbatim:
`owner-thread work while the copy stream's host function sleeps: 3.55 ms; the copy behind the host function still
pending: true`, `test result: ok. 1 passed`. A 200 ms host function on the copy stream does not hold the owner thread's
CUDA work in the same context (an allocation, a 4 MB pageable H2D and D2H, an event, a stream sync). **The hypothesis
that the host function holds the owner thread is refuted; design F is not refuted by it.** The second tick of section 4
stays unplaced.

The two Nsight Systems boots queued with the cell did not run: `Failed to create directory "/tmp/nvidia/nsight_systems":
Permission denied` (its temp directory; TMPDIR was not set). The job then held the lock, waiting for a server that
could not come up, until it was stopped at about 18:27:30Z (`owner-hold/gap.log`); the lead's integ53 battery took the
lock at 18:27.

## 6. The D2D half of Move 2 owed item 1, pre-registered (no code)

**The read (file:line at `045ab57d4`).** The capture route copies each layer's recurrent state on the OWNER stream at
the boundary: `prefix_capture_off_tick`, `engine.clone_dtod(&r.conv_state)` / `(&r.ssm_state)` (`worker.rs:15233`),
96 planes, 156.9 MB on the 27B; the capture law is why: the next decode writes the live state. The spec-boundary
publisher takes the recurrent state from the spec engine at the prime stop (already owned; no copy at publish). The
restore route copies the entry's recurrent planes into the request's fresh cache on the owner stream before the KV
batch: `host_restore_submit`, `copy_into(&mut dst.conv_state, ..)` (`worker.rs:16749`), the line's own words
`recurrent state copied on the owner stream; request parked`. The tick programs (`prefix_snapshot` `:18755`,
`prefix_restore_at` `:19070`) do the same with the door off.

**Priced by arithmetic, before any design.**

- Host time cannot leave the owner thread in either class: every CUDA call is the owner thread's (`check_thread`),
  so a copy-stream placement issues the same 96 allocations and copies plus one event and one copy-stream wait.
- The capture: the next decode WRITES the planes the copy reads, so a copy-stream capture needs the owner stream to wait
  on every copy event before that decode (a write-after-read fence). The owner stream's critical path keeps the copy's
  GPU time (about 0.2 ms: 2 x 156.9 MB at about 1.5 TB/s on the target card). **No owner time of either kind can
  leave by construction: the capture half is refuted as a design.**
- The restore: the reader is the parked request's prime at its re-admission, one tick later; a copy-stream restore of
  the recurrent planes would overlap that tick's decode step and free at most the copy's owner-stream GPU time (about
  0.2 ms per restore on the target card), no host time.

**The one cell owed before any restore design (log only, pre-registered).** A field on the `restore submitted off the
tick` line: `recurrent copy H.HHms host, G.GGms owner stream` (host time around the copy loop; owner-stream GPU time
between two events around it), read on the existing restore arm (`stall_cell.py --mode restore`, day 21) on the target
card and the 5090. **Decision rule, stated now:** if on the target card the owner-stream GPU median is under 0.5 ms and
the host median under 0.5 ms per restore, the D2D half closes as not worth a door (the verdict and the receipt in
`OWNER-THREAD-OFFLOAD.md` and the verdicts ledger; no code). Otherwise a restore-recurrent design is pre-registered
with its own acceptance (its ordering rule is the day-21 reader fence, extended over the recurrent planes, as day 32
extended it over the H2D spans). The strong-form receipt stays owed separately.
