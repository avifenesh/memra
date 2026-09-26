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
