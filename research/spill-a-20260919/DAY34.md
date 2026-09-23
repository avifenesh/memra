# WP-A day 34: the H2D completion checksum on the hash helper (the lead's lever 2 from DAY33 section 5a)

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`. Rig: the local RTX 5090 Laptop GPU (the lever's proving card); the
target card with the day-33 sitting when the lead restores it. No cross-card comparison. Every cell
`executed-not-qualified`. Every push in the announced `MEMRA_RELEASE_QUALIFICATION_MODE=development` mode. Behind
`MEMRA_KV_HOST_CONTRACTS` (default OFF), off-tick contract promotes only.

## 1. The read and the census (file:line at `d91bead02`)

Every SHA-256 the owner thread runs over a pinned KV lease under the door:

| site | class | when | on the 5090 (write-combined leases) |
|---|---|---|---|
| `tier_transfer.rs:2193`, `CudaTransfers::progress` | the H2D item's completion checksum (its HOST source, the entry's KV lease) | the promote's landing poll, one per item | 16 items on the 9B: 16 steps of about 0.6 ms, 8.4 ms in all (DAY33 section 5a) |
| the same line | the D2H item's completion checksum (hash 1, the landed lease) | the demote's landing poll | the same class and count |
| `worker.rs:9490`, `bind_tier_image` | the demote's bind re-hash of each KV lease (hash 2), compared with hash 1 | the demote's publication | the same class and count |
| `worker.rs` `host_roundtrip_digest` (`MEMRA_KV_HOST_VERIFY`) | the verify arm's digest | diagnostic arm only | out of scope |

**Today's promote rule, restated.** The settle requires each H2D item's completion checksum (the engine's SHA-256 of the
host source after the copy) equal to that plane's D2H receipt recorded at the demote (`completion.require(.., receipts,
true)`), before the reader wait's publication and before any read; a difference refuses typed (`ReceiptMismatch`, the
host entry dropped, the cold path serves), except under the `flip-demote` diagnostic, which names the injected
difference and lets the verify arm catch it at promote. An unobservable completion latches.

**What moves, what stays.** The H2D checksum of the promote moves (below). The demote's two KV hashes stay on the owner
thread today: hash 2 exists to catch a change BETWEEN the landing (hash 1) and the bind (the `flip-demote` fault writes
in that window by design), so moving them is a two-hash ordering rule of its own, not the same rule; they are owed,
with their 5090 cost (DAY30's `owner in-completion` 10 to 13 ms on this card) as the price.

## 2. Pre-registration (committed before any day-34 code)

**Design K (the promote's H2D checksums on the hash helper).**

1. Engine, conformance first. A new rule `conformance/h2d_deferred_checksum.rs` (additive, unversioned, `WIRE_VERSION`
   stays 1): an H2D ticket whose completion checksums are DEFERRED to the caller lands only with every accepted item's
   checksum supplied (`producer_done` false until then, whatever the copies did); the supplied checksums are the items'
   completion checksums and `Completion::require` against the demote-time receipts gates publication exactly as before
   (a supplied mismatch is `Corrupt`); each item's host source stays owned by the engine while a view of it is out
   (`retire_source`, `recover_source` and `retire` are `Busy`); every view comes back exactly once. The red arm: a
   binding that lands a deferred item on its copy alone (checksum absent) must fail the schedule. CPU binding; native
   cell.
2. Engine call: `CudaTransfers::defer_h2d_checksums(&ticket) -> Vec<H2dSourceView>` right after the batch's
   submission (one read-only view per accepted H2D item: its index and its host source's bytes), and
   `supply_h2d_checksums(&ticket, Vec<(H2dSourceView, Digest)>)` (each view back, its digest the item's completion
   checksum and expectation, the same assignment `progress` makes today). `H2dSourceView::digest` is the same
   `checksum` program over the same bytes; `Send`, its `unsafe` (a raw view of the pinned bytes) confined to the engine,
   sound because the engine keeps the source until every view is back and nothing writes the source meanwhile (the
   entry's own handle cannot write while a twin lives).
3. Server: on the off-tick route (`device_entry_from_host_parts`, `OffTick`), right after the KV batch's submission, the
   views go to the hash helper as ONE `Sources` job (the helper's second job kind, its own reply channel); the helper
   hashes them while the copy runs. The settle, before its completion step, takes the helper's reply (`Poll`: if it has
   not landed the ticket is `Pending`, as a running copy is; `Block`: a bounded wait within the helper's deadline) and
   supplies the digests; then everything runs as today, `require` included. Fail closed: a helper gone, a reply that
   does not describe the job, or digests past the 10 s deadline latch the tier (the engine keeps the views' sources:
   a leak, never a free). The on-tick route (`OnTick`, the body's hook and the GPU unit cells) keeps the engine's own
   checksum.
4. The timeline (day 33) and the H2D receipt line are unchanged in form; the receipt line gains `; checksums on the
   hash helper` so every receipt names where its checksums ran. No new `MEMRA_*` name, no new numeric program.

**Acceptance, stated before running.**

- (a) Verification semantics unchanged: every H2D item is still checksummed against its demote-time checksum before the
  promoted entry is published or read, and a mismatch still refuses or latches as today. Cells: the conformance rule
  and its red arm; a native cell that supplies a wrong digest and reads `Corrupt` from `require` with nothing
  published; and the failure gate's `digest` cell on the door ON arm (`MEMRA_KV_HOST_FAULT=flip-demote`, verify on):
  `KV-HOST-SPILL FAILURE GATE: ALL GREEN` with its `VERIFY FAILED` caught and the cold path serving, and the settle's
  line `contracts door H2D receipt: .. plane host bytes differ from the D2H receipt as injected` present, which is the
  helper-supplied digest seeing the flipped byte.
- (b) On the 5090: identity x4 (default OFF and ON, plain OFF and ON), the fault gate default and plain (every cell),
  hit OFF/ON armed with the day-24 census, and the unit cells (the door's GPU cells, the engine's span and deferred cells,
  the CPU censuses): ALL GREEN.
- (c) The landing poll's owner hold on the 5090, read from the timeline (`poll k at +P ms (tick K, its top +T ms)
  complete`, P minus T on the completing poll, an upper bound that includes the tick top's demote poll): median at most
  1.5 ms and max at most 3.0 ms over at least 20 steady promotes (day 33: about 8.5 ms).
- (d) The added promote end to end: the intruder's e2e median on the day-34 binary at most the day-33 binary's plus 1.0
  ms, per order, in the same hold.

**The 5090 A/B (pre-registered cell).** The stall harness's promote arm (`stall_cell.py --mode promote --n 5`, byte for
byte), the day-33 binary against the day-34 binary, interleaved: order o1 = d33 d34 d33 d34 d33 d34 d33 d34 d33 d34, order
o2 = d34 d33 d34 d33 d34 d33 d34 d33 d34 d33 (N=5 boots per arm per order), all twenty boots in ONE bounded hold of
`/tmp/memra-5090.lock` (never inside another lane's), the 9B, the day-33 stall-pair boot environment (prefix budget 64
MB), raw logs and receipts banked. (c) is read on the day-34 boots; (d) per order.

**Predictions.** The helper's SHA-256 over the 9B's write-combined KV leases runs about 8 ms, concurrent with the copy
stream's 6.3 ms, both from the probe; the tick is about 8.4 ms. Most steady promotes should find the digests landed at
the next tick top: then the landing poll's hold falls to about 0.5 ms and `promote_in` to about 9 to 10 ms (day 33: about
17), the intruder's e2e by about 7 ms. If the digests miss that top, the promote takes one more tick (about 17 ms, as
day 33) and (d) is where it shows. On the target card (cached leases) the helper's hash is about 1 ms and the change is
about 1 ms of owner time per promote, read at the restore's sitting.

**Budget.** 0.5 agent-day plus card time. The 5090 is held by lane B day 34's O2 order until about 20:45Z; the A/B and
the gates queue behind it with bounded waits.

## 3. What landed, in the pre-registered order, one census each

1. **The tier rule** (`88b734bc9`):
   `h2d_deferred_checksum_lands_with_its_digests`, `h2d_deferred_checksum_mismatch_is_corrupt` and the red arm
   `h2d_deferred_checksum_copy_alone_is_not_landed` (additive, unversioned, `WIRE_VERSION` stays 1); CPU binding 3
   passed (the red arm: a binding that lands a deferred item on its copy alone fails the schedule).
2. **The engine** (`e3424be24`): `defer_h2d_checksums` (one `H2dSourceView` per accepted H2D item, taken WITHOUT the
   tracking event's host wait: `bytes()` would wait for the item's own copy, whose event is recorded after it; the view
   and the copy only read) and `supply_h2d_checksums` (every view checked before any digest is taken; each digest
   becomes its item's completion checksum and expectation, the assignment `progress` makes for an undeferred item);
   `progress` lands a deferred item only with its digest; `recover_source` refuses while a view is out, and the
   source's retire and the ticket's wait on the landing. `H2dSourceView::digest` is the same `checksum` program over
   the same bytes; `Send`, its `unsafe` confined to the engine. Census `h2d_deferred_checksum_rules_are_as_stated`;
   native cell `h2d_deferred_checksum_lands_with_the_supplied_digests`.
3. **The server** (`100214477`): the off-tick route's last submission step (after the spans) hands the KV items' views
   to the hash helper as one `Sources` job (the helper's second job kind, its own reply channel); the settle takes the
   reply and supplies the digests before the completion step (under `Poll` an unlanded reply hands the ticket back;
   helper gone, a reply that does not describe the ticket, or the deadline latch); the H2D receipt line gains `; N KV
   checksums on the hash helper (X MB in Y ms)`. Census `day34_the_promote_checksums_ride_the_hash_helper_in_the_stated_order`;
   GPU cell `option_c_off_tick_checksums_ride_the_hash_helper_and_a_corrupt_lease_is_refused`. The day-33 census now
   names `HostHelperJob::Fill` (gone) rather than the enum (back with the `Sources` kind). Server lib 877 passed.
4. `docs/TESTING.md` line (`201b48617`). No new `MEMRA_*` name, no new numeric program.

Checks on `201b48617`: tier contracts 98 passed, engine lib 547 passed, server lib 877 passed; clippy `-D warnings`
all targets on tier, engine and server `clippy_rc=0`; the `DOCS_RS=1` pass `docsrs_rc=0`; `check-flags` clean.

## 4. The target-card sitting, day 33 and day 34 together (pre-registered before it runs; BOX4)

The target card is now BOX4, one RTX PRO 6000 Blackwell Workstation Edition (600 W). BOX3's numbers are not a baseline
here: every comparison below is within one hold on BOX4. Queue: lane B's #680 boots first (marker `LANE-B-680-DONE`),
then this sitting; the build runs after that marker (no build beside another lane's cells).

**The hold.** One collector hold of `/tmp/memra-gpu.lock`: the day-26 double-park cell (`pro-single-day26/double-park.sh`,
byte for byte; twenty boots, N=5 per arm per order, both orders) three times, on the day-32 H2D binary (`5ae139734`), the
day-33 binary (`d7d54c7a9`: design F plus the log-only timeline) and the day-34 binary (the lane tip at the sitting:
design F, the timeline and design K), in that order; then the gate set on the day-34 binary under the collector
(identity x4, failure x2, the fault gate with all twelve cells, twin x2), the hit gate OFF and ON under its own `flock`,
and the unit cells under the collector (the door's GPU cells, the engine's `d2d_`, `d2h_span` and `h2d_` cells serially,
the CPU censuses and the tier bindings).

**Day 33's acceptance, unchanged** (DAY33 section 2), read on the day-33 run against the day-32 run of the same hold:
(a) DAY28 1a, 1b (at most +20.0 per order) and 1c on the day-33 run; (b) B2's rule on the day-33 run's owner segment
(median at most 1.5 ms, max at most 3.0 ms, N at least 20, every submission filled); (c) B1, B4, B5 on the gate set
(the gates run on the day-34 binary, which carries design F unchanged); (d) the day-33 run's owner segment median at
most 0.90 ms and max at most 1.50 ms, and its census.

**Day 34's target-card clauses** (DAY34 section 2's (a) and (b) as written; (c) and (d) restated for this card):
(a) the failure gate's `digest` cell on the door ON arm, with the `plane host bytes differ from the D2H receipt as
injected` line, and the GPU cell `option_c_off_tick_checksums_ride_the_hash_helper_and_a_corrupt_lease_is_refused`;
(b) the gate set and the unit cells ALL GREEN; (c) on the day-34 run, the landing poll's hold (`day34-reading.py`'s
measure) median at most 1.5 ms and max at most 3.0 ms over at least 20 steady promotes; (d) on the day-34 run, DAY28 1b
at most +20.0 per order, and its ON intruder e2e median at most the day-33 run's plus 1.0 ms per order.

**Predictions.** The day-33 run: the copy stream's work (about 9.5 ms) inside the probe's tick (about 13.5 ms), one
poll, `promote_in` about 16 to 18 ms, 1b about +6 to +12. The day-34 run: the helper's SHA-256 over cached leases about
1 ms, the landing poll's hold about 0.5 ms, e2e within 1 ms of day 33's. If the day-33 run's copy misses the probe's tick,
1b stays near the day-32 run's and that is the result.

## 5. The BOX4 sitting, as it ran

- 20:35:51Z: the lead's go (lane B's #680 part A done 20:21:58Z); the lock free, no compute app. `build.sh 5ae139734
  d7d54c7a9 afd58bbde` in this lane's own clone `/root/wt-a`: three release servers, `rc=0` by 20:46Z.
- **Attempt 1 refused at the port guard.** Every boot of all three double-park runs refused before any server
  started: `cannot prove port 18132 is free. An unobservable port is not a free port: a foreign responder would be
  measured as the model under test. Install iproute2 (ss) or lsof.` (neither was on BOX4); the gates and the hit gate
  refused the same way. The unit-cell collector ran (its cells boot no server). The driver was stopped; iproute2 was
  installed (`apt-iproute2.log`); attempt 1's boot-dependent receipts are kept under `attempt1-no-ss/`.
- `driver-rerun.sh` (`fee93513d`): the same `doubleparks.sh`, `gates.sh` and `hitgate.sh` on the same three binaries under
  the same collector. No script, clause, threshold or cell changed.
