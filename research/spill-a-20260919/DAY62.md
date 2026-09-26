# WP-A day 62: OWED item 13, the capture retire seam's Block settle (a price first, then a design)

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`. Behind `MEMRA_KV_HOST_CONTRACTS` (default OFF). Every cell
`executed-not-qualified`. Pre-registered while DAY59's cells wait for a card; its code follows in order.

## 1. Pre-registration (committed before any code)

**The item** (DAY38 section 4): `capture published off the tick (seed): .. settled synchronously by a session retire;
the settle held the owner thread 2607.12ms` under the `d2h-delay` arm; day 25 priced the seam at 0.4 ms on a copy that
had landed. Read from the code (`worker.rs`, the retire pass): when any session finishes while a capture is pending, the
worker settles the capture with `ContractWait::Block` before any session leaves `active`, because the capture reads a
live session's KV planes on the copy stream with no source retained. So the owner waits for the capture's copy and for
everything queued ahead of it on the one side stream (a demote's copies, its 105 ms receipt kernel at long entries, a
promote's fill and spans), and it waits even when the retiring session is not the capture's source.

**Step 1, the lines (log only).** The retire settle's line gains whether any retiring session is the capture's source
(`source retiring: yes|no`) and the copy-stream work queued ahead of the capture at its submission (the tickets in
flight on the copy stream, by kind).

**Step 2, the price** (one RTX PRO 6000, the 27B, door ON; a new `stall_cell.py` mode `retire-seam`): the tenant
streams; at the fire, a 4096-token prompt demotes the resident long entry (its copies and receipt kernel queue on the
copy stream) and, at once, a short fresh prompt (`max_tokens 1`) seeds a capture behind them and retires; and a second
shape where the retiring request is not the capture's source (a third short request retires while the capture is
pending). N=5 per order, both orders against the plain `prime` control; the owner hold of each retire settle read from
its line.

**The designs, selected by the price** (stated now):

- R1 (no-source retires do not settle): the retire pass settles the pending capture only when a retiring session is its
  source. Selected when the no-source shape's median hold is above 1.0 ms.
- R2 (the source's planes held by the capture): a retiring source session's KV planes move into the pending capture,
  which drops them (or parks the session's reuse entry) only after its landing; the retire never blocks. Selected when
  the source shape's median hold is above 1.0 ms.
- Both when both are above; neither when both are at or below 1.0 ms (the seam closes as priced).

**Each design's acceptance, stated now:** the identity gate door ON default and plain, the fault gate default and plain
(its capture cells included), the hit gate door ON; the selected shape's median hold at most 0.5 ms per order; the
tenant's stall and the intruders' e2e at most the tip's plus 1.0 ms; for R2 a census that no path drops, rewrites or
parks a capture source before its landing.

**What each card decides.** The target card (the long demote's receipt kernel is the queue the seam waits behind).

**Budget.** 0.5 agent-day: the lines 0.05, the mode and the sitting 0.1, the card 0.1, the design(s) 0.25.

## 2. Step 1 as built (`17a1c8076`), and the price sitting prepared

- The lines (log only):
  - At submission, `PendingCapture` records the source cache's identity (its layer vector's heap address, which
    stays stable while the cache moves between owners) and the copy-stream tickets the worker had in flight, by kind
    (`host_copy_stream_in_flight`: demote, promote, restore, or `none`).
  - The retire pass computes whether a retiring session's cache is the source, and the settle's why reads `a session
    retire (source retiring: yes|no)`. The settle is unchanged: `Block`, before any session leaves `active`.
  - The publish line gains `copy stream in flight at submission: ..`.
  - The census `day62_the_retire_seam_lines_are_log_only` checks: the two new fields are read only by the why and
    the line, and the `Block` settle still precedes the removal. The older retire census now finds the why by its
    new literal.
- CPU: server lib `936 passed; 0 failed; 26 ignored` (`day62/server-lib.log`); clippy `-D warnings`; fmt.
- The modes `retire-seam` and `retire-seam-other` in `stall_cell.py`:
  - Both share one untimed long seed in setup.
  - Each timed intruder posts a fresh long prompt (max_tokens 1), whose insert demotes the resident long entry.
  - The moment it returns, `retire-seam` posts one fresh 72-word prompt at max_tokens 1: the source shape, retiring
    with its own capture pending.
  - `retire-seam-other` posts two at once: a 72-word source at max_tokens 32, still decoding, and a second 72-word
    prompt at max_tokens 1 that retires meanwhile: the no-source shape.
  - The earlier modes are unchanged.
- **The sitting** `pro-single-day62/`, receipts `/root/spill-receipts/a-d62`:
  - `build.sh <tip>` builds the tip's server and checks both line wordings in `markers.txt`.
  - `driver.sh` runs one collector hold: `prime`, `retire-seam`, `retire-seam-other`, o1 in that order x5 and o2
    in reverse x5 (30 boots), with the prefix cache at 448 MB (the chain cell's long-entry shape),
    `MEMRA_MAX_SESSIONS=4`, `MEMRA_KV_HOST_MB=8192`, door ON, then `day62-reading.py`.
  - Its last line is `DAY62 SELECT -> ..`, by section 1's rule: the median hold above 1.0 ms in both orders selects
    R1 for the no-source shape and R2 for the source shape. A shape that forms no line in an order is named and
    selects nothing.
  - About 35 minutes of card time.

## 3. The price, read as registered: SELECT R2

- Run by the lead on one RTX PRO 6000 Blackwell Workstation card (a 16-core host), `build.sh c8ab2f609` then
  `driver.sh`, 07:19Z to 07:52Z. Mirror `pro-single-day62/box/`: 144 receipts, sha256-checked against the box manifest
  (144 OK). The tip server is recorded by hash; both line wordings are present (`markers.txt`). 30 boots with start
  temperatures of 39 C to 69 C.
- Verbatim (`box/reading-day62.log`):

      DAY62 READING order=o1 stall prime=280.20 retire-seam=280.36 retire-seam-other=259.02 ms | holds source N=45 median=86.69 no-source N=0 median=nan ms
      DAY62 READING order=o2 stall prime=280.40 retire-seam=280.53 retire-seam-other=259.07 ms | holds source N=45 median=86.85 no-source N=0 median=nan ms
      DAY62 IN-FLIGHT shape=source {'demote': 90}
      DAY62 IN-FLIGHT shape=no-source {}
      DAY62 OTHER retire settles (not a shape's): {('prime', 'yes'): 90}
      DAY62 SHAPE MISSING (o1 no-source, o2 no-source) -> that shape selects nothing
      DAY62 SELECT -> R2 (the source's planes held by the capture)

- **Read from the receipts** (the route of every `capture published` line per mode; the medians per mode, order and
  route):

      prime o1 a session retire (source retiring: yes) inflight=demote N=45 median=86.73
      prime o2 a session retire (source retiring: yes) inflight=demote N=45 median=86.84
      retire-seam o1 a session retire (source retiring: yes) inflight=demote N=45 median=86.69
      retire-seam o2 a session retire (source retiring: yes) inflight=demote N=45 median=86.85
      retire-seam-other o1 a second capture inflight=demote N=50 median=0.06
      retire-seam-other o2 a second capture inflight=demote N=50 median=0.06

  - **Whose capture is settled.** Every source settle belongs to the long intruder's own 5088-token seed capture:
    `retire-seam` 90 of 90 and `prime` 90 of 90, never the short's. The short's 64-token captures all landed at the
    tick top.
  - The long prompt's insert evicts the resident long entry, so its demote (310 MB and the D2H receipt kernel) is on
    the copy stream first. The capture is submitted behind it; the settle is entered about 104 ms after the
    submission and then holds the owner thread 86.7 / 86.8 ms waiting for the copy stream.
  - So the source shape formed, but as the long request retiring with its own capture pending, not the short.
    **The same 86.7 ms hold sits in the `prime` control**: every fresh long prime in this shape pays it at its retire.
    That is why `prime` and `retire-seam` read the same stall (280.2 / 280.4 against 280.4 / 280.5). R2 therefore
    targets the long-prime class itself, not only the seam's cell.
- **Why the no-source shape never formed** (so R1's question is unanswered, not closed):
  - In `retire-seam-other` the two shorts' seed captures (64 tokens each) evict entries of their own. The device
    cache then holds short entries, so the long prompt's insert evicts a short entry, and that demote lands fast.
  - The long's own capture then lands at the tick top: median 12.7 ms from submission, 100 of 100 by tick-top poll.
    Nothing long is queued ahead of any capture when a non-source session retires.
  - The shorts' two captures meet each other instead: the second's submission settles the first synchronously (`a
    second capture`, 100 settles, median 0.06 ms, the first already landed).
  - The mode was designed on a wrong model of the timing: I expected the long demote to stay ahead of the shorts'
    captures, and the cache composition changed which entry the long evicts.
- **R1 needs its own corrected cell** (section 4). R2 is built per the rule (section 5), after B1's and W's 5090
  halves.

## 4. R1's corrected cell, pre-registered (to run with R2's sitting)

- **The mode `retire-seam-nosource`.** The shared long seed as in section 2. Each timed intruder streams a fresh long
  prompt (max_tokens 32), so its insert demotes the resident long entry and its own capture queues behind that demote
  while it keeps decoding. At the long request's first streamed token, a short fresh prompt of 30 words (below the
  64-token grid, so no seed capture and no insert) at max_tokens 1 is posted. It primes, retires, and meets the long
  capture pending: a non-source retire. `wall_ms` is the long request's; the short's wall is recorded.
- **Reading the rule** (section 1's, unchanged): the no-source shape's holds are the `source retiring: no` lines of
  this mode. A median above 1.0 ms in both orders selects R1; at or below, R1's seam closes as priced. A mode that
  again forms no line is recorded and revised under a new registration. It is paired against `prime` and
  `retire-seam` in one hold, 30 boots, on the base tree (R2's parent), so the R2 sitting reads it on the same card.

## 5. Design R2, pre-registered (committed before any code)

**The program.** When the retire pass meets a pending capture whose source is a retiring session, that session
leaves `active` into the pending capture (`PendingCapture.held`) instead of being retired now. The retire pass does
not settle, and no one drops, parks, rewrites or hands out the source cache while its copies may still read it. The
session's response is already complete; what waits is its retire tail (metrics and the reuse park).

- (R2.1) The source test is step 1's identity (the cache's layer-vector address), now a decision. A capture with no
  retiring source keeps today's `Block` settle (R1 is not selected).
- (R2.2) Release, exactly once, after the copies are observed complete (or after a host wait on them):
  - every exit of `PendingCapture` hands the held session to `hpx.retire_ready`: the tick-top landing (published),
    every `Block` settle (a second capture, a tenant purge, a demote or promote that settles captures), a dropped
    or refused capture, and the latch;
  - the worker drains `retire_ready` into the retire pass of the same tick, the unchanged retire body (the park
    included);
  - at shutdown the held session drops after the shutdown settle.
- (R2.3) A line `[prefix-cache] retire deferred: the capture's source session is held until its landing (.. ms)` at
  the deferral, and `retire released: .. after .. ms (<route>)` at the release, under the door. The deferral's own
  owner time is the new hold.

**Acceptance** (section 1's, with the cells named):

- (a) The identity gate door ON default and plain; the fault gate default and plain (its capture cells included); the
  hit gate door ON; the pause gate. CPU cells:
  - a retiring source with a pending capture is held and not parked, and released at the landing into the retire
    pass;
  - each release route (landing, second capture, purge, drop, latch) releases exactly once;
  - a retire with a pending capture whose source is not retiring still settles `Block`.
- The census: no path drops, parks, rewrites or hands out a held source before its release; `held` is written only at
  the deferral.
- (b) The source shape's median hold (the deferral's line) at most 0.5 ms per order, in `retire-seam` and `prime`,
  base against R2.
- (c) The tenant's stall and each intruder's e2e at most base's plus 1.0 ms. The stall is expected to fall by about
  the 86.7 ms hold; that is a reading, not a bound.
- (d) R1's cell (section 4) runs in the same sitting on base and is read by its own rule.

**What each card decides.** The target card (the long demote is the queue). The 5090 half follows, per the
per-hardware rule, since the program changes on every card.

**Budget.** 0.4 agent-day: the hold and releases 0.2, the cells and census 0.1, the sitting 0.1.

## 6. Design R2 as built (`00168e79a`), and its sitting prepared

- (R2.1) `r2_defer_retire(finished, source_retiring, ticket, held)`: the retire pass defers only when the one
  retiring session is the pending capture's source, the capture has a ticket in flight, and no source is already
  held. Otherwise it runs today's `Block` settle with step 1's why.
  - On a deferral, the session leaves `active` into `hpx.held_source` (with the ticket sequence) and prints
    `[prefix-cache] retire deferred: .. (ticket seq=S; X ms)`.
  - The session stays booked: the admission book retires it once, at the retire loop's single removal, after its
    release.
- (R2.2) Release and the other exits:
  - The settle's `Done` records `capture_landed_seq`. The retire pass then decides with `r2_held_release`:
    Release (the landing observed: the session re-enters `active` and this tick's retire loop, the park included,
    `retire released: .. (the landing observed)`), Keep (still pending), or Quarantine (the holding capture ended
    without an observed landing: never parked or dropped, `retire QUARANTINED`).
  - The tenant purge drops a held source of the purged tenant only after its landing (never parked), and
    quarantines it otherwise.
  - Shutdown (inside `host_capture_drain_at_shutdown`) drops a landed held source and leaks one that never landed,
    plus every quarantined source.
  - The idle block and its 2 ms cap also wait while a source is held, so a landing seen at the tick top reaches the
    retire pass in the same iteration.
- Read while building, stated for the sitting: while a source is held it is not in `active`, so the session-count
  cap sees one fewer session for about the capture's remaining copy time (its bytes stay booked). A released
  session retiring while a newer capture is pending is a non-source retire and settles `Block` (R1's territory).
- Cells: `day62_r2_defers_only_the_lone_source_and_releases_only_after_the_landing` (the two decisions' truth
  tables) and the census `day62_r2_no_path_releases_a_held_source_before_its_landing`. Three older censuses were
  updated for the new shape:
  - step 1's census now counts the source test as R2's decision;
  - the admission book's census finds the retire loop's own removal;
  - the idle guard's census literal gains `held_source`.
- The red arm (`day62-r2/red-arm.patch`: a pending capture releases its source, with a marker) fails the decision
  cell (`left: Release, right: Keep`, `day62-r2/red-arm.log`).
- Server lib `938 passed; 0 failed; 26 ignored` (`day62-r2/server-lib.log`); clippy `-D warnings`; fmt.
- R1's mode `retire-seam-nosource` is in `stall_cell.py` (section 4), and `day62-reading.py` takes the cell and
  the no-source mode as arguments (defaults: section 2's).
- **The sitting** `pro-single-r2/`, receipts `/root/spill-receipts/a-r2`: `build.sh <tip> <R2's parent>` (r2 and
  base from one clone), then `driver.sh`, each step under one collector hold:
  - `gates.sh`: 11 gates on r2 (the identity gate default and plain door OFF and ON, the fault and contract fault
    gates default and plain, the pause gate, the hit gate OFF and ON);
  - `ab.sh seam prime retire-seam 448`: base against r2, 40 boots, `MEMRA_MAX_SESSIONS=4`;
  - `ab-r1.sh`: R1's cell on base, 30 boots;
  - then `r2-reading.py` (`R2 VERDICT -> ..`) and `day62-reading.py <root> seam-r1 retire-seam-nosource`
    (`DAY62 SELECT -> ..` for R1).
  - About 2 hours of card time.

## 7. R2's sitting, read as registered: REVERT (c); what the stall waits on

- Run by the lead on one RTX PRO 6000 Blackwell Workstation card (a 16-core host), `build.sh f5e9cf798 3641f1f1c`
  then `driver.sh`, 08:25Z to 10:03Z. Mirror `pro-single-r2/box/`: 819 receipts, sha256-checked against the box
  manifest (0 mismatches); the two servers are recorded by hash. Start temperatures 47 C to 69 C.
- Verbatim (`box/reading-r2.log`):

      R2 READING order=o1 mode=prime base hold=86.78 ms (N=45) | r2 deferral=0.002 ms (N=45), r2 retire settles=0 | stall base=281.79 r2=282.02 (+0.23) | e2e base=1455.8 r2=1456.3 ms
      R2 READING order=o1 mode=retire-seam base hold=86.88 ms (N=45) | r2 deferral=0.002 ms (N=45), r2 retire settles=0 | stall base=281.71 r2=281.94 (+0.23) | e2e base=1787.2 r2=1806.6 ms
      R2 READING order=o2 mode=prime base hold=86.84 ms (N=45) | r2 deferral=0.002 ms (N=45), r2 retire settles=0 | stall base=282.29 r2=282.28 (-0.01) | e2e base=1457.2 r2=1457.1 ms
      R2 READING order=o2 mode=retire-seam base hold=86.92 ms (N=45) | r2 deferral=0.002 ms (N=45), r2 retire settles=0 | stall base=282.02 r2=282.12 (+0.10) | e2e base=1786.9 r2=1807.7 ms
      R2 (b) PASS [True, True, True, True]
      R2 (c) FAIL [True, False, True, False]
      R2 VERDICT -> REVERT ((a) passed; failed c): recorded as read, reverted in one commit

  (a) held: all 11 gates 0, and no quarantine line.
- **What the 282 ms stall waits on.** The stall cell reads the tenant's single largest gap minus its p50. The tenant's
  four largest gaps in both arms are 293.4 / 268.4 / 265.4 / 262.1 ms (medians over the runs, `retire-seam`; `prime`
  reads the same): the long 5088-token intruder's own prime segments on the owner thread, one tenant step each. The
  86.8 ms retire hold was a smaller gap below those. Removing it could not move the largest one, and R2's stall
  moved +0.23 / -0.01 ms as read. The prime segments are the prime class's price (the day-16 `prime` arm, OWED item
  4's class), not the seam's.
- **Why `retire-seam`'s e2e grew 20 ms: the wait moved to the next capture.**
  - Parts (medians): long 1457.4 ms on base against 1457.7 on r2; short 329.1 against 349.7. The long is unchanged;
    the short grew.
  - Routes: on base the long's capture settles at its own retire (`source retiring: yes`, 86.9 ms). On r2 it is
    still pending when the short's 64-token seed capture is submitted, and the one-capture rule settles it there
    (`settled synchronously by a second capture`, N=90, median 22.46 ms), inside the short request's own path.
  - R2 removed the retire's wait and the next capture paid its remainder: the second-capture settle is the same seam
    one step later. Recorded as a finding for any later design on this seam: removing a `Block` settle must name
    where the wait goes.
- **Reverted** as registered, in one commit (`revert(spill-a): design R2 ..`). The code returns to R2's parent's,
  and R2's receipts stay (`day62-r2/`). The step-1 lines stay.
- R1's corrected cell ran in the same sitting (section 8).

## 8. R1's cell, read: SELECT R1; R1 pre-registered (committed before any code)

- Verbatim (`box/reading-r1.log`, `retire-seam-nosource` on the base binary, 30 boots):

      DAY62 READING order=o1 stall prime=282.32 retire-seam=282.36 retire-seam-nosource=282.18 ms | holds source N=45 median=86.84 no-source N=45 median=12.53 ms
      DAY62 READING order=o2 stall prime=282.33 retire-seam=282.26 retire-seam-nosource=282.21 ms | holds source N=45 median=86.87 no-source N=45 median=12.54 ms
      DAY62 IN-FLIGHT shape=source {'demote': 90}
      DAY62 IN-FLIGHT shape=no-source {'demote': 90}
      DAY62 SELECT -> R1 (no-source retires do not settle) and R2 (the source's planes held by the capture)

- Read:
  - The no-source shape formed (45 per order). A short session unrelated to the capture waits 12.53 / 12.54 ms for
    it. That is above the 1.0 ms rule, so R1 is selected.
  - The tenant's largest gap is again the long prime's segment (282.2 ms in all three modes), so the stall cannot
    read R1 either. A reading over the tenant's mid gaps (the excess over p50 of every gap between p50 + 5 ms and 150
    ms, summed per run; medians `prime` 161.9, `retire-seam` 355.8, `retire-seam-nosource` 151.5 ms) is where a
    retire's wait shows.
  - The short's e2e reads 101.8 ms.
- **The design, R1.** The retire pass settles the pending capture only when a retiring session is its source; any
  other retire goes ahead with no settle. The capture copies only its source's planes (the trunk KV rows, and on the
  spec-boundary route the source's draft scratch), and the recurrent state was cloned at submission. So a non-source
  session's cache may drop or park while the copy runs.
  - (R1.1) The source test becomes a decision, so it must never miss. The capture records its source session's
    request id at submission, threaded through both capture routes (`prefix_insert_from_session`, and
    `prefix_insert_from_spec_boundary` from all three publishers).
  - The pass settles when any retiring session has that request id, **or** when any of the retiring session's
    caches (the plain cache, the MTP spec cache, the DFlash cache) has the recorded layer-vector address. Either
    match settles.
  - Every capture's source is an active session at submission (the seed, the lcp split and the spec boundaries all
    run on live sessions), so its request id is known.
  - (R1.2) The settle's why stays `a session retire (source retiring: yes)`. A skipped settle prints `[prefix-cache]
    retire with a capture pending: no retiring session is its source (ticket seq=S); no settle`.
  - (R1.3) Nothing else changes. The one-capture rule, the second-capture settle, the trim, purge and shutdown
    settles, and the source's own retire keep today's `Block`.
- **Acceptance** (section 1's, with R2's lesson folded in):
  - (a) The identity gate door ON default and plain, door OFF too; the fault and contract fault gates default and
    plain; the pause gate; the hit gate OFF and ON.
  - (a) CPU cells: the decision (source by id or by any cache address settles; neither skips).
  - (a) The census: the settle runs whenever either identity matches; the request id is recorded at every capture
    submission; no other settle site changes.
  - (b) In `retire-seam-nosource`, r1's no-source retire hold at most 0.5 ms per order. There is no settle line; the
    skip line's count must equal base's no-source settle count within 10%.
  - (c) The tenant's stall, and the long and the short request's e2e separately, each at most base's plus 1.0 ms,
    per order.
  - (d) Where the wait goes (R2's lesson), a reading with a bound: r1's second-capture settle count and median hold
    and its source-retire settle count and median, against base's. A second-capture or source-retire hold that grows
    by more than 1.0 ms per order fails (d).
  - Readings: the tenant's mid-gap excess sum, base against r1 per order; the `retire-seam` and `prime` modes in the
    same hold as regression controls, their stall and e2e under (c).
  - Every clause read in both orders.
- **What each card decides.** The target card; the 5090 half follows under the per-hardware rule.
- **Budget.** 0.3 agent-day: the identity threading and decision 0.15, the cells 0.05, the sitting 0.1.
