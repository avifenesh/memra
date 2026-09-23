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

## 6. The 5090 half (`rtx5090-day34/`)

One bounded hold of `/tmp/memra-5090.lock`, 20:55:55Z to 21:12:08Z (`run.log`; it waited 47 bounded attempts behind lane
B's O2 order, never inside it), no compute app at the start. Binaries: the day-33 binary with the timeline
`a4f00c5656d35b4a1290565ad060af4b265ffc45614bb938827b18d29eebfee8` (`d7d54c7a9`), the day-34 binary
`3d3b96bf4652e9a2a36a5b88084c35344e7debff2d929ac7b46697232c347cc0` (`100214477` code), the 9B NVFP4 MTP artifact.

- Unit cells: the door's GPU cells `test result: ok. 17 passed` (with
  `option_c_off_tick_checksums_ride_the_hash_helper_and_a_corrupt_lease_is_refused`); the engine's `h2d_` and
  `d2h_span` cells `ok. 5 passed` (with `h2d_deferred_checksum_lands_with_the_supplied_digests` and the day-33 owner-hold
  cell).
- The A/B, twenty boots, `STALL REPLAY: PASS` 20 of 20 (`reading-day34.log`), verbatim:
  - `DAY34 READING arm=d33 boots=10 steady landing-poll-hold N=90 median=8.50 min=8.20 max=9.54 polls [1] (counts [90]) |
    promote-in N=90 median=17.10 ...`
  - `DAY34 C arm=d34 boots=10 steady landing-poll-hold N=90 median=0.12 min=0.10 max=0.40 polls [1, 2, 3] (counts [1,
    88, 1]) | promote-in N=90 median=16.50 min=9.50 max=31.90 | receipts=100 on-helper=100 helper-ms N=100 median=8.80
    min=8.70 max=18.50 rule N>=20 median<=1.5 max<=3.0 every-receipt-on-helper -> PASS`
  - `DAY34 D order=o1 e2e d33 N=50 median=80.49 ... e2e d34 N=50 median=80.70 ... d34-minus-d33 +0.21 rule <=+1.0 ->
    PASS`; `order=o2 ... median=81.19 ... median=81.70 ... +0.50 rule <=+1.0 -> PASS`
- **(c) PASSES**: the landing poll's owner hold 8.50 ms to 0.12 ms (N=90 each, max 0.40). **(d) PASSES** in both orders.
  The helper's hash over the write-combined leases takes 8.8 ms, just longer than the 8 ms to the next tick top, so 88
  of 90 promotes land at the second poll: the owner thread is free, and the request waits one more tick than it would
  if the digests had made the first (the predicted "miss" branch): `promote_in` 17.10 to 16.50.
- Gates on the day-34 binary through `--external-lock 9`: identity x4 `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)`
  (12 ok each); failure ON `KV-HOST-SPILL FAILURE GATE: ALL GREEN` (15 ok), with, in its `digest` cell, `contracts door H2D
  receipt: Kv(3) K plane host bytes differ from the D2H receipt as injected (MEMRA_KV_HOST_FAULT=flip-demote); the
  verify arm catches it at promote` and `VERIFY FAILED: promoted digest .. != demote digest ..`, the receipt naming `KV
  checksums on the hash helper (1.1MB in 9.9ms)`: **(a)'s served-path cell holds** (the helper's digest saw the flipped
  byte); fault default and plain `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` (160 ok each).
- The hit gates did not run in that hold: the gate's own `stop` addresses its server under an external lock only while
  `/proc/<pid>/comm` reads `memra-server` (C day 28's rule), and the binary was named `memra-server-d34`; its spec-on
  server outlived the stop and the spec-off twin refused the port (`hit-off/gate.log`). The orphan (this lane's own
  binary on port 18099, its environment read from `/proc`) was stopped by pid. The rerun under the name `memra-server`,
  the same bytes (`hit-rerun.sh`; `binary 3d3b96bf4652e9a2`), took the hold at 21:26:25Z after four bounded busy
  attempts: `gate hit-off-rerun rc=0 SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` (61 ok) and `gate hit-on-rerun rc=0
  SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` (68 ok, with the day-24 census `capture_submitted=12 capture_published=12
  restore_submitted=13 restore_landed=13`); released 21:27:15Z.

## 7. The BOX4 sitting's verdicts (`pro-single-day34/box/`)

**Resync note.** The session stopped on a server-side API error during the sitting (about 21:55Z) and resumed at
22:07Z. Post-block resync: `git fetch` and `git log` (the lane tip `22363f29d` equal to origin; nothing lost), DAY33 and
DAY34 re-read (design K is DAY34 section 2's registered design, committed at `a40ade417` before any day-34 code; its
tier, engine and server code were in before the stop), the running driver (`driver-rerun.sh`) left to finish; no second
driver. `origin/main` `25bbb91f5` (#692, integ53) merged as `37c08c06b` after the sitting: its `worker.rs` hunks are the
admission's memory bound, not the host-tier paths; server lib 885 passed, clippy clean on the merge.

**The hold** (collector, `/tmp/memra-gpu.lock`, zero lock retries, every `CELL.jsonl` closing row
`executed-not-qualified False 0`): the three double-park runs 20:54:42Z to 21:53:50Z (`STALL REPLAY: PASS` 20 of 20 in
each, `errors=0` on all 180 receipt lines, `compute_apps: []` before and after; 45 to 76 C, 15.4 to 348.8 W), the gates 21:53:50Z to 22:06:34Z,
the hit gate to 22:08:25Z; the unit cells ran in attempt 1 (20:49:30Z to 20:54:35Z). Binaries: day-32
`62766088afd996b6..` (`5ae139734`), day-33 `81d3db48c97a8f0d..` (`d7d54c7a9`), day-34
`ded36fa2c5d50ac8e1256139c1dafb46da0c071dca316bc2a860fd07c8a5d1ef` (`afd58bbde`; the gates' binary).

**Day 33's acceptance on BOX4, verbatim** (`reading-day33-d32-vs-d33.log`, `d33/reading-day28.log`):

- (a) `DAY28 CLAUSE 1a stall order=o1 .. on_cell_median=76.8 off_cell_median=93.1 rule on<=off+2.0 -> PASS` (o2 the
  same); `DAY28 CLAUSE 1b e2e order=o1 N_runs_on=50 N_runs_off=50 on=133.4 off=120.4 on_minus_off=+13.0 rule <=+20.0 ->
  PASS` (o2 `+13.0`); `DAY28 CLAUSE 1c owner in-completion N=100 median=2.30 .. rule <=12.0 -> PASS`; `DAY28 VERDICT
  clauses_failed=0 -> ALL PASS`.
- (b) `DAY33 B2 after (day-33 binary) steady owner-segment N=90 median=0.76 min=0.70 max=0.82 boots_on=10
  submissions=100 filled=100 rule N>=20 median<=1.5 max<=3.0 every-submission-filled -> PASS`.
- (c) B1: the gate set on the day-34 binary (design F unchanged in it) ALL GREEN (below); B4 `DAY32 B4 receipts=209 bad=0
  by (items, spans)=[((32, 96), 201), ((34, 96), 8)] .. -> PASS`; B5 identity x4 `teeth=0`, the `verify ok` round trips,
  the native cells.
- (d) `DAY33 D after steady owner-segment N=90 median=0.76 min=0.70 max=0.82 rule median<=0.90 max<=1.50 -> PASS`, with its
  census.
- Section 4's prediction for the day-33 run (one poll, `promote_in` 16 to 18 ms) missed; its stated miss branch ("1b
  stays near the day-32 run's") is what ran: two polls, `promote_in` 28.40, 1b +13.0 against the day-32 run's +12.8.
  The day-34 predictions: helper about 1 ms (read 1.40), landing poll about 0.5 ms (read 0.35), e2e within 1 ms of day
  33's (read 1.07 and 1.05 lower).
- **Day 33's acceptance holds on BOX4, and BOX4 cannot credit the lever with it.** The day-32 binary in the same hold also
  meets 1b: `on_minus_off=+12.8` / `+12.7 -> PASS` (`d32/reading-day28.log`). On this box design F's copy misses the
  probe's tick: `DAY33 READING submission-to-completion steady before ms N=90 median=14.70 .. polls [2] (counts [90]);
  after ms N=90 median=27.30 .. polls [2] (counts [90])`, `promote in-ms steady before in N=90 median=28.40; after in
  N=90 median=28.40; after-minus-before median +0.00`. The timeline says why: `submitted +0.36ms (tick 6503, its top
  -0.64ms); poll 1 at +14.54ms (tick 6504, its top +13.45ms) pending`. On this CPU the day-32 binary's helper fill of the
  same bytes reads `156.9MB filled by the hash helper in 11.4ms` (the median of 200), so on the copy stream the fill
  plus the 96 spans outlast the 13.1 ms from submission to the next tick top, and the ticket lands one tick later.
  Nothing here is compared against BOX3.

**Day 34's target-card clauses, verbatim** (`reading-day34-pro.log`, `reading-day28.log`):

- (a) the failure gate's `digest` cell on the door ON arm: `KV-HOST-SPILL FAILURE GATE: ALL GREEN` (15 ok), with
  `contracts door H2D receipt: Kv(3) K plane host bytes differ from the D2H receipt as injected (MEMRA_KV_HOST_FAULT=flip-demote);
  the verify arm catches it at promote`, the receipt naming `KV checksums on the hash helper (2.0MB in 1.1ms)`, and `VERIFY
  FAILED: promoted digest .. != demote digest ..`; the GPU cell
  `option_c_off_tick_checksums_ride_the_hash_helper_and_a_corrupt_lease_is_refused ... ok`. **PASS.**
- (b) identity x4 `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` (12 ok each); failure x2 `ALL GREEN` (15 ok each);
  `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` (160 ok, twelve cells); twin x2 `PREFIX-NEWEST-TURN-FITS: .. -> PASS`; hit OFF
  `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` (61 ok) and ON (68 ok) with the day-24 census (`capture_submitted=12
  capture_published=12 restore_submitted=13 restore_landed=13 ..`, `30 route submission(s)`); unit: the door's GPU cells
  `ok. 17 passed`, the engine's `d2d_`, `d2h_span` and `h2d_` cells `ok. 10 passed` (the owner-hold cell: `owner-thread
  work while the copy stream's host function sleeps: 0.42 ms; the copy behind the host function still pending: true`),
  the engine censuses `ok. 6 passed`, the CPU cells `ok. 16 passed`, the tier bindings `ok. 13 passed`. **PASS.**
- (c) `DAY34 PRO C run=d34 steady landing-poll-hold N=90 median=0.35 min=0.33 max=0.37 | promote-in N=90 median=27.40 ..
  | receipts=100 on-helper=100 helper-ms N=100 median=1.40 min=1.10 max=1.70 rule N>=20 median<=1.5 max<=3.0
  every-receipt-on-helper -> PASS`.
- (d) `DAY28 CLAUSE 1b e2e order=o1 .. on=132.4 off=120.3 on_minus_off=+12.0 rule <=+20.0 -> PASS` (o2 `+12.0`); `DAY34 PRO
  D order=o1 ON e2e d33 N=50 median=133.42 .. ON e2e d34 N=50 median=132.36 .. d34-minus-d33 -1.07 rule <=+1.0 -> PASS`
  (o2 `-1.05`). **PASS.**
- Where day 34's millisecond comes from on this card (`reading-poll1.log`, `poll1-reading.py`, log only, added after
  the sitting, no clause): `run=d33/double-park/ev poll1-end-minus-its-top N=100 median=1.10 min=1.08 max=1.13 |
  poll2-end-minus-its-top N=100 median=0.35` and `run=double-park/ev poll1-end-minus-its-top N=100 median=0.03 min=0.02
  max=0.04 | poll2-end-minus-its-top N=100 median=0.35`. The poll stamp is taken after the settle returns. On the
  day-33 binary the pending first poll checksums the items already landed on the owner thread (about 1.1 ms on this
  card's cached leases); on day 34 those checksums ride the helper (`32 KV checksums on the hash helper (1.9MB in
  1.1ms)`). The completing poll reads 0.35 on both, which is why PRO C's landing-poll hold did not move here while
  `promote_in` went 28.40 to 27.40 and the e2e -1.07 / -1.05. On the 5090 the checksums fell in the landing poll
  (write-combined reads, 8.50 to 0.12 ms).

## 8. Findings

1. **Design K works on both cards as registered.** The 5090's landing poll 8.50 to 0.12 ms (write-combined leases); the
   target card's owner checksum time about 1.1 ms off the tick and the promote about 1 ms earlier end to end.
2. **Design F's one-tick landing is box-dependent and did not happen on either card measured.** On the 5090 the
   copy lands in the probe's tick, but the landing poll's write-combined checksums (now moved by K) cost what the tick
   saved; on BOX4 the 11.4 ms single-thread fill plus the spans outlast the 13.1 ms to the next tick top, and the copy
   lands a tick later. No card where the fill fits inside the tick has been measured on the day-33 binary.
3. **On BOX4 the day-32 binary already meets DAY28 1b**: `on=133.5 off=120.7 on_minus_off=+12.8` (o1) and `on=133.6
   off=120.9 on_minus_off=+12.7` (o2). The day-32 BOX3 FAIL stands as that box's result; no cross-box reading is drawn
   from the pair.
4. The hit gate's own `stop` addresses its server only under the name `memra-server` (a 5090 run with a renamed binary
   left its spec-on server up; section 6). Named for any cell that renames binaries.

## 9. What is owed, and integrability

- **Integrable: yes** for day 34 (design K) and for day 33's code (design F): every pre-registered clause holds on BOX4
  and on the 5090. Day 33's lever is integrable as correct code whose one-tick landing is unproven on the cards measured
  (finding 2); the lead decides whether F stays or the day-32 helper fill comes back (F removed the day-32 Filling phase;
  both pass every clause on BOX4).
- Owed: the demote's two KV hashes on the owner thread (DAY34 section 1: hash 1 in `progress`, hash 2 in the bind,
  about 10 to 13 ms of owner time per demote on the 5090, their own ordering rule); the fill's speed on slower CPUs if
  the one-tick landing is wanted (a multi-threaded or chunked fill on the copy stream, not pre-registered); the D2D half
  (DAY33 section 6: the restore's price cell decides; the capture refuted by construction); the strong-form receipt.

## 10. Checks, budget, cleanup

- Checks on the merged tree `37c08c06b`: `cargo fmt --all -- --check` clean; clippy `-D warnings` all targets on tier,
  engine and server `clippy_rc=0`; server lib 885 passed; on `201b48617`: tier contracts 98, engine lib 547, the
  `DOCS_RS=1` pass `docsrs_rc=0`; `check-flags` and `check-conflict-markers: OK`; `git diff --check` clean;
  `.gitattributes` in `rtx5090-day34/` and `pro-single-day34/box/`; no em dashes in this lane's lines; no new `MEMRA_*`
  name.
- Budget: about 3 agent-hours of work for day 34 (19:00Z to 22:30Z wall, the stop included), plus about 1.5 h queued
  behind other lanes on both cards.
- Cleanup: BOX4 `/root/wt-a` is this lane's own clone at `afd58bbde`, left for later sittings; `/root/spill-receipts/a-day34/`
  mirrored to `pro-single-day34/box/` (binaries excluded; their sha256 in each run's `binary.sha256` and
  `gates/binary.sha256`), the mirror checked file for file (903 of 903), then removed from the box; iproute2 installed there (attempt 1); no process of this lane on the box after 22:08:25Z;
  `LANE-A-BOX4-DONE` written. Local: the 5090 carries no process of this lane; `/tmp/wt-a-d33` removed at close.
