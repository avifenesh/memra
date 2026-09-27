# WP-A day 64: OWED item 18, the promote's one-tick-late publications under S4 (a placing first)

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`. Behind `MEMRA_KV_HOST_CONTRACTS` (default OFF). Every cell
`executed-not-qualified`. Pre-registered while DAY59's cells wait for a card; its code follows in order.

## 1. Pre-registration (committed before any code)

**The item** (DAY50 sections 2 and 3): on S4's tree 9 of 90 steady promotes publish one tick after the next tick top (0
of 90 on G4), and 2 of 27 on the tip; S4's H2D destination digests are queued on the copy stream after the span copies
(DAY42 section 1 step 5), and the promote publishes only when both have landed. The delay did not reach S4's (d) (PIN
+0.10 ms per order).

**What is not known.** Whether the late promotes wait on the destination digests (the receipt event lands after the
tick top that saw the copies land), on the copies themselves (a long fill ahead of them), or on the helper's `Sources`
reply (design K's checksums of the source leases, which the publication also requires).

**Step 1, the lines (log only).** The promote's published line gains, per poll, which of its three requirements was
still pending (`copies`, `receipt`, `sources`), and the tick-top poll at which each was first seen complete.

**Step 2, the cell.** The promote cell (`stall_cell.py --mode promote --n 5`, the 27B, door ON), 20 boots on the target
card; `day50-reading.py` plus the new per-poll fields. **The placing rule:** the requirement pending at the extra poll
in at least half of the late promotes is named the place; otherwise not placed.

**The designs, by the place (stated now):**

- `receipt`: the destination digests launched ahead of the fill-and-copy tail they cover is impossible (they read the
  landed destinations), so the design is a second, non-blocking poll of a pending promote after the tick's decode
  launch (an event query on the owner thread, no wait), publishing a promote whose requirements landed during the
  tick instead of at the next tick top; the parked request primes at the same tick either way, and the publication's
  requirements are unchanged.
- `sources`: T-H's split of the `Sources` job (DAY65) is the design, and item 18 closes on DAY65's reading.
- `copies`: the late promotes wait on their own copies; recorded as the copy stream's price, no design.

**Acceptance of the `receipt` design:** 0 of 27 steady promotes published after the next tick top (the tip read 2), the
promote's PIN and e2e at most the tip's plus 1.0 ms, the tenant's stall at most the tip's plus 1.0 ms, the fault and
identity gates door ON.

**Budget.** 0.3 agent-day.

## 2. Step 1 as built (`193f2634d`), and the placing sitting prepared

- The lines (log only):
  - Each `Pending` answer of the promote's settle now carries what it still waits on, and the timeline's poll entry
    reads `pending on X` (`sources`, `copies`, `receipt`, joined by `+`). Before, it read `pending`.
  - The sources label comes from the helper's reply not yet landing. The copies and receipt labels come from the
    engine's new read-only `CudaTransfers::h2d_landing_parts`: the items' and spans' copy events, and the spans'
    destination-digest receipt event, queried without changing any state.
  - "The tick-top poll at which each was first seen complete" is read from the same entries: the first poll that no
    longer names a requirement.
  - The census `day64_the_promote_waiting_labels_are_log_only` checks that the query is read only by the labeller,
    and the label only by the timeline.
  - Server lib `941 passed; 0 failed; 26 ignored`; clippy `-D warnings` (engine and server); fmt (`day64/`).
- Read while writing the reader: W's target base receipts already show a late promote (submitted at tick 9859,
  published at tick 9861, one poll `pending` at 9860). The shape exists on the current tree, so the cell should place
  it.
- **The sitting** `pro-single-day64/`, receipts `/root/spill-receipts/a-d64`: `build.sh <tip>`, then `driver.sh`, in
  one collector hold:
  - the promote mode, 20 boots at `--n 5`, `MEMRA_MAX_SESSIONS=4`, the prefix cache at 256 MB, door ON;
  - then `day64-reading.py`, whose last line is `DAY64 PLACE -> ..`. It reads section 1's rule: a promote is late
    when it publishes after the next tick top, and the label at that next tick top's poll names the requirement.
    Named at least half the time, it selects the design.
  - About 25 minutes of card time.

## 3. The placing cell, read as registered: PLACE receipt

- Run by the lead on one RTX PRO 6000 Blackwell Workstation card (a 16-core host), `build.sh c537abdb5` then
  `driver.sh`, to 13:22Z. Mirror `pro-single-day64/box/`, sha256-checked against the box manifest (0 mismatches);
  the server is recorded by hash. Start temperatures 40 C to 67 C.
- Verbatim (`box/reading-day64.log`):

      DAY64 READING steady promotes=180 late=176 labels at the extra poll={'receipt': 176}
      DAY64 PLACE -> receipt (176 late of 180)

- Read:
  - On this tree nearly every steady promote (176 of 180) publishes one tick after the next tick top. At that next
    tick top, every one of them had its copies landed and its sources' checksums in, and waited only on the spans'
    destination-digest receipt (S4's H2D span receipt).
  - From the timelines (medians over the late promotes), the poll at the next tick top runs at +13.1 ms from
    submission and reads `pending on receipt`; the next one, at +25.4 ms, reads `complete`. The copy's own
    submission-to-completion reads 25.0 to 25.2 ms.
  - So the receipt lands between one and two ticks after submission. The parked request re-admits and primes one
    tick (about 11.5 ms on this card) later than its copies would allow.
- Item 18 is placed at `receipt`. Section 1 named a design for it; section 4 revises that design before any code, and
  says why.

## 4. The receipt's design, pre-registered (committed before any code)

- **Why section 1's design is set aside.** Section 1 proposed a second non-blocking poll after the tick's decode
  launch, publishing a promote whose requirements landed during the tick.
  - A publication mid-tick does not move the parked request's prime: admission runs at the tick top, so the request
    re-admits at the next tick top either way. The late count would fall to 0 while the request's e2e stayed the
    same.
  - Section 1 said so itself ("the parked request primes at the same tick either way"). A design whose only effect is
    the counted line is not the improvement item 18 is for, so it is not built.
- **Step 1, the lines (log only).** Three timing events on the copy stream, created with timing and read only at the
  settle, once complete (never waited on):
  - after the fill host function;
  - after the last span copy;
  - after the digests and the lanes' D2H, the receipt event itself.
  - The H2D receipt line gains `span receipt: fill X ms, copies Y ms, digests Z ms`, the elapsed times between them.
  - A census checks that the events decide nothing.
- **Step 2, the design, selected by step 1's split** on the target card (the promote cell, 20 boots, the same shape
  as section 2's):
  - **Digests at least half of the time from the last copy to the receipt, in the median: D1, the overlapped span
    digests.** The destination digests run on a second stream of the same context. Each group of spans (S2's
    64-span launch) waits on its own group's copy event and digests as soon as that group lands, overlapping the
    remaining copies. The lanes' D2H and the receipt event follow the last group, on that stream. The receipt then
    lands about one group's digest after the last copy, instead of every digest after it.
    - The digest program is unchanged: `span_receipt_digests` over the same device planes, the same lanes, the same
      `receipt_digest_from_lanes` fold.
    - The batch still lands only with its receipt (`h2d_span_batch` rule 2).
    - The second stream is created with the copy stream, and a creation failure is a construction refusal.
  - **Otherwise** (the fill or the lanes' D2H dominate): that term is named with its numbers and designed next under
    its own registration.
- **D1's acceptance:**
  - (a) Correctness:
    - the engine's span digest cells;
    - the fault gate default and plain, whose span cells include `span-flip-landed` and `span-flip-resident`: a
      flipped span must still be refused by the receipt;
    - the identity gate default and plain door OFF and ON;
    - the hit gate OFF and ON;
    - the pause gate.
  - (b) The late count: at most 9 of 180 steady promotes publish after the next tick top (5%), against section 3's 176
    of 180.
  - (c) The promote cell's intruder e2e at most base's minus 5.0 ms per order, the parked request re-admitting one
    tick earlier. The PIN at most base's plus 1.0 ms.
  - (d) The tenant's stall at most base's plus 1.0 ms. The hump (day 38's HUMP, the digests now overlapping the
    copies on a second stream) at most base's plus 0.15 ms.
  - Every per-order clause in both orders.
- **What each card decides.** The target card: step 1's split and (a) to (d). The 5090 half follows under the
  per-hardware rule.
- **Budget.** 0.4 agent-day: the lines 0.05, their sitting 0.05, D1 0.2, its sitting 0.1.
- **Timing of the work.** This lane builds nothing while W's 5090 cell holds the card (DAY61 section 4). Step 1's
  code follows its reading.
