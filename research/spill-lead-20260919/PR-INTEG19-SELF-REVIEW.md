# integ19 self-review (lead, 2026-09-21)

Read in full: `crates/memra-engine/src/tier_transfer.rs` (`retain_host`, 24 lines), the `crates/memra-server/src/worker.rs`
day-16 diff (`host_kv_planes_from_contract`, `HostPromoteFailure`, `PromotePlane`, the promote abort helpers,
`device_entry_from_host` change, the six GPU cells and the door-only test), `worker/host_glm.rs`, the fault gate's
promote arms, `WC-DESTINATIONS.md`, `wc-pair.py`, `verify-day16.py`; receipts spot-checked.

## Findings
1. **One engine addition, minimal and typed.** `retain_host` clones the `Rc` on a pinned allocation after the owner-thread
   and owner-context checks, refusing a released allocation; both handles own the charge and the last drop releases
   both; while a twin lives, `write` and a D2H into either refuse `Busy`. It mirrors `retain_device` and adds no copy
   program; the engine already accepted a shared H2D source.
2. **Publication only after `require`.** The H2D batch's completion is checked against the D2H receipts before
   `ready_view` and before the planes become the live entry; a receipt mismatch cancels, recovers the source per rule 1
   with pointer identity checked, and the caller drops the entry (typed). The verify digest stays as the OFF-path check.
3. **Day-15 lessons applied.** Originals dropped before the unwind, no discarded `retire`/`acknowledge`/`release_producer`
   result, drain before retire, `TicketLeaked`-style latch; promote-side one-shot faults drive the fault gate to
   `ALL GREEN` on four cells.
4. **OFF untouched.** `plane_up` and the OFF promote path are unchanged; `Pinned` entries keep the OFF path under ON
   (stated); serve-smoke and the identity gate are equal OFF/ON.
5. **The WC pair is honest.** N=5 per arm per order, both orders within 0.5 ms, one lock hold, regime stated; reported as
   the first decide-by cell, not a verdict; ruling 20 routes the allocation flag to lane A as a typed seam with an A/B.
6. **Battery note.** The first full memra-server run on this tree failed one test
   (`same_effort_value_resolves_identically_on_every_surface`: a 429 where 200 was expected); it passed 3 of 3 alone and
   on the rerun of the full suite; earlier integ batteries (integ12, integ14) show it passing. Order-dependent flake in a
   test unrelated to this PR's files; recorded, receipts kept, an issue follows if it recurs.

7. **Revuto round, both fixed on the lane (`4467131f6`).** Partial acceptance: `sources` carries `(item, lease
   pointer)` per op and the unwind hands `recover_source` only the accepted items, so a rejected slot is never asked
   and the abort ends `Refused` with the ticket retired and acknowledged and every fresh destination released. Published
   state: the abort no longer infers `published` from the item index; it asks the engine through `cancel`
   (`PublicationRevoked` recovers the accepted sources per rule 1, `AlreadyPublished` records the consumer fence, drains
   and retires the sources), so the classification is the engine's. Two new one-shot faults (`contract-promote-reject`,
   a last op mis-sized by one byte and rejected by the engine's own validation; `contract-promote-readyview`, the first
   `ready_view` published by the engine with the route told otherwise) drive two new fault-gate cells and two GPU unit
   cells; `ALL GREEN` on six cells on the card, the aborted ticket consumed, no `TIER DISABLED`, drop, `Capacity`,
   `leaked` or `already published` line; identity gate ALL GREEN OFF and ON with one H2D receipt per promote. A
   source-text cell pins the cancel-then-recover-then-consumer order and the absence of `published: bool`.
8. **Push regime changed under the PR.** Main gained #589 (release qualification): the perf-ci arm is retired and an
   engine-source push is `UNQUALIFIED` unless announced as development; this tree pushes in the announced, logged
   development mode and claims no qualification.

## Verification this review relied on
integ19 CPU batteries (`integration-day12/integ19-cpu-battery/` with the server-suite rerun, `-2/` after the review
round, plus the local serve-smoke), full `tools/local-ci.sh --perf` on the pre-review tree (`integ19-local-ci-perf/`,
attempt 3 green after a settled tripwire), C's target-card gates and the WC pair. This rig cannot run the model
gates.
