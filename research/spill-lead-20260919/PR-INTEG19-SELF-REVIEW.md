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

## Verification this review relied on
integ19 CPU battery (`integration-day12/integ19-cpu-battery/`, with the server-suite rerun), full `tools/local-ci.sh
--perf` on this tree (`integ19-local-ci-perf/`), C's target-card gates and the WC pair. This rig cannot run the model
gates.
