# WP-A day 42: OWED item 4 again, design S2 (S's revision, DAY40 section 7)

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`. Resync (the lead's resume message): `git fetch`; `origin/main`
`5d653e851` (integ59, #723, merged this lane's tip `e9c9c062c`) merged into the lane (a fast-forward). Ruling 54 read
(`research/spill-lead-20260919/INTEGRATION-DAY12.md`): S refuted and reverted; item 4 open under DAY40 section 7's
revision; order of work item 4, item 15 (both arms), the 9950X-class fill reading, the 5090 hump replicate, items 6 to 14.
Every cell `executed-not-qualified`. Behind `MEMRA_KV_HOST_CONTRACTS` (default OFF).

## 1. Pre-registration (committed before any S2 code)

**What S taught** (DAY40 sections 5 and 7). Its correctness held on the 5090 (every span digest bitwise, both red arms
witnessed by exactly their span, identity, failure and hit gates green). Its price did not: about 3.1 ms of copy-stream
work on the demote's landing path (48 source digests 0.61 ms, 48 landed digests 2.49 ms, 96 launches) pushed the landing
past the first tick-top poll on 73 of 90 demotes (wall +40.25 / +39.30 ms), and the kernels beside the intruder's last
steps cost its e2e +1.16 to +1.53 ms.

**Design S2** (on G4's one side stream; everything below is on the copy stream):

1. **One batched kernel.** `span_receipt_digests(SpanItems items, u64* lanes)` in `cu/tier_receipt.cu`: the four-lane
   program of `d2d_receipt_digest` (the CPU oracle `memra_tier::conformance::receipt_digest`), one item per `blockIdx.y`
   (up to 64 items per launch, their device addresses and byte lengths passed by value), a grid-stride loop over the
   item's words in `blockIdx.x`, one atomic per lane per block into the item's 32-byte lanes. Order-independent wrapping
   sums, so the device answer is the oracle's per item. A `docs/KERNELS.md` row.
2. **The source digests stay before the copies, in one launch.** `submit_d2h_spans`, behind its producer fence and the
   lanes' zero-fill: one launch (per 64 spans) over every span's device source, then the span copies and their events
   exactly as G4. **The batch lands with its copies** (G4's landing; nothing of the receipt is on it after this launch,
   about 0.15 ms for the 9B's 52.7 MB and 0.6 ms for the 27B's 157 MB of device memory).
3. **The landed digests leave the landing path.** `take_d2h_spans` (the landing observed) enqueues, on the copy stream,
   the `span-flip-landed` flip under its fault, then one launch (per 64) over every span's pinned staging through its
   device address, the lanes' D2H into a pooled twin and the span receipt's event, and returns the spans with a span
   receipt id. `d2h_span_receipt(id)`: `Ok(None)` while its event is pending, `Ok(Some(pairs))` once observed (each
   span's `(source, landed)` digests; the entry leaves the engine, its twin back to the pool); `d2h_span_receipt_wait(id)`
   for a `Block` settle; `d2h_span_receipt_abandon(id)` for a demote that ends unpublished (the entry is reaped when its
   event is observed, at the next span receipt call; the engine's drop leaks what is left, as it leaks any in-flight
   input).
4. **Why the sources and the staging may leave at the take** (stated so its census can hold it): the sources were
   digested before their copies, so the source digest is complete when the batch lands; every writer of a staging
   buffer is on the copy stream (the demote's span copies, the promote's fill host function), so a staging buffer the
   set hands out again is written only behind the landed digest in stream order. The hash helper's CPU read of the
   staging runs beside the landed digest's device read: two readers, no writer.
5. **The H2D destination digests in one launch, on the landing.** `attach_h2d_spans`: after the span copies, one launch
   (per 64) over every destination, the lanes' D2H and the receipt event; the H2D batch lands with them (as S; S's (d)
   read PIN +0.50 / +0.30 ms there with 48 launches).

**The server.**

6. **The demote's publication waits for the span receipt, its landing does not.** The settle takes the spans back at
   the landing (the staging to the helper as today) and keeps the span receipt id (with the spans' slots) on the
   `Hashing` phase. Each `Hashing` poll first polls the span receipt: pending, the entry stays `Hashing` (the same
   10 s deadline; past it, or under a `Block` settle after `d2h_span_receipt_wait`, a receipt that never landed latches
   the tier typed: `tier span receipt never landed: ticket seq=S ..`); observed, the pairs are kept and the poll goes on
   to the helper's reply. On the reply, BEFORE any staging goes back to the set: a span whose pair differs refuses the
   demote typed (`demote failed (tier image <slot> span landed bytes differ from their device source); nothing
   published`), the staging returned, the tier on; equal pairs keep each source digest with the published entry
   (`HostPrefixEntry::span_digests`, S's field). Every other exit of a demote holding a span receipt abandons it.
7. **The promote**, as S: the destination digests against the entry's kept source digests before the reader wait's
   publication; a difference takes the KV receipt mismatch's path (the entry dropped, the cold path serves); an entry
   without a kept digest (a handoff import) keeps the weak span receipt.
8. **Red arms** on the existing `MEMRA_KV_HOST_FAULT` row: `span-flip-landed` (the flip at the take, before the landed
   digest) and `span-flip-resident` (S's); fault gate cells `span-flip-landed` and `span-flip-resident` (S's cells, with
   the settle helper taking the cell's refusal words).
9. **The tier rule** `span_receipt`, revised for the split landing: a D2H span batch lands with its copies; the demote
   publishes only after its span receipt is observed and only if every pair agrees; an H2D span batch lands with its
   destination digests and the promote publishes only if they equal the kept sources. Red arms: a caller that publishes
   before the span receipt is observed, and one that publishes a differing pair. CPU binding.
10. **Censuses**: the kernel's one launch per 64 spans in each place; the D2H order (fence, zero-fill wait, source
    launch, copies, events) and the take's order (flip, landed launch, lanes, event); the Hashing step's order (span
    receipt poll before the reply; the pair check before the first `staging_put`); every exit that abandons.

**Acceptance, stated before any code** (each card its own; the bounds are S's, unchanged):

- (a) Semantics: the tier rule and its red arms; native cells (the batched kernel bitwise the CPU oracle per item on
  1 B to 3 MiB + 3 at offsets 0 to 7, 1 to 70 items, device and pinned host memory; the D2H span cell's pairs and its
  flip arm, span 0 alone; the H2D span cells' destination digests); the fault gate's two cells green.
- (b) The gate set ALL GREEN (identity x4, failure OFF and ON, the fault gate default and plain with every cell, twin
  OFF and ON on the target card, hit OFF and ON) and the unit cells.
- (c) Demote price: S2 against G4 (DAY38 section 3's demote A/B, 20 boots): per order the steady demotes' wall median at
  most +8.0 ms and the demoting intruder's e2e median at most +1.0 ms.
- (d) Promote price: S2 against G4 (the promote A/B, 20 boots, `--n 5`): per order PIN median at most +1.0 ms and the
  promoting intruder's e2e median at most +1.0 ms.
- (e) The hump clause on S2 (two S2 boots, the G'' control beside them, 16 demote runs each): median HUMP at most 0.15
  ms; each boot's start temperature and SM clock and the hold's 250 ms telemetry recorded (ruling 54's regime record).
- Reading, no clause: the copy stream's side work per demote from one traced boot of each arm (DAY40 section 6's
  reader, its grouping fixed to S2's order).

**Predictions.** (c): the copy phase lands at the first poll as G4's (8.4 ms) with the source launch's 0.15 ms added;
the wall within +2 ms; the e2e within +0.5 ms (the landed launch runs about 8 ms later, at the copy-settle poll). (d):
PIN about +0.2 ms. (e): flat, as G4.

**What each card decides.** Each card its own (a) to (e).

**Budget.** 1.5 agent-days: the kernel and the engine 0.4, the server and the tier rule 0.4, the 5090 cells 0.4, the
target sitting 0.3.
