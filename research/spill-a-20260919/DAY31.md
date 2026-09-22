# WP-A day 31: DAY30's owed small items closed (1a to 1d), the H2D half pre-registered

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`. Rig: the target card (BOX3, one RTX PRO 6000 Blackwell, 96 GB,
600 W, `research/spill-lead-20260919/BOX-ACCESS.md`) and the local RTX 5090 rig where its lock allows; no cross-card
comparison. Every cell `executed-not-qualified`. Every push in the announced `MEMRA_RELEASE_QUALIFICATION_MODE=development`
mode (logged; no qualification claimed). Budget: 5 agent-hours.

## 0. First action

- Fast-forward merge of `origin/lane/spill-integ47-20260923` (tip `160929a92`: A day 30 plus C day 38), pushed. No
  conflict, no local change beyond it.

## 1. What landed (`33b1285e0`), one census each

**1a. The staging goes back on every post-take refusal.** `host_kv_planes_settle_contract` now builds a `StagedSpans`
guard before `take_d2h_spans`; every landed `D2hSpan` is split into its source (back to the slot) and its staging
buffer (into the guard). The guard's `Drop` puts every buffer back into the context's set; the only way out with the
buffers is `StagedSpans::landed`, called once, on the `Ok(ContractSettle::Done(..))` return. The driver's two early
exits of the `Done` arm (`!host.armed()` and the flip-demote fault) put the landed buffers back too. Census
(`day31_the_staging_goes_back_on_every_exit_and_is_charged_once`, CPU): the guard sits before the take, the take before
the first push, the push before the span-count check and the first home; exactly 11 `return Err(` after the take, all
inside the guard's scope; `staged.bufs` 3 times; `landed` once; no `mem::forget`; the `Drop` body puts buffers back;
the driver's two `unstage(host, staged);` calls in order. GPU cell `option_b_span_postpublish_refusal_returns_the_staging_to_the_set`
(a post-take refusal: the whole set back, the charge unchanged, the next demote reuses it).

**1b. The staging is charged to the tier governor's pinned ledger.** A fresh staging buffer is charged on allocation
(`ResidentCharge::reserve`, pinned = its length, `Priority::Backup`, the staging tenant `digest("host-tier-staging",
..)`, distinct from every pool tenant), released only at the latch's `clear()` (the set closes: later takes refuse
`the staging set is closed (the tier latched off)`, later puts drop the buffer). A reused buffer is not charged again.
A refused charge refuses the span attach through the existing typed arm (`tier D2H spans refused: a N-byte staging
buffer: the governor's pinned ledger refused N bytes (Capacity) (k f32 spans handed back)`): sources back, staging back,
the KV ticket unwound, the tier stays on. A fresh fill prints one line, `[prefix-host] tier span staging: F fresh pinned
buffer(s), B bytes charged to the governor's pinned ledger; the set holds S bytes charged`. Census: the take's order
latched, reuse, charge, allocate, kept; `PinnedHostBuf::new_unwritten(` once in production. GPU cells: the two day-30
span cells extended (charged equals the span bytes, the ledger reads `(span_bytes, 0, 0)` after the settle and `(0, 0,
0)` after the latch), `option_b_span_staging_charge_refusal_refuses_the_attach_and_keeps_the_tier_on` (a hog charge
leaves room for exactly one span; the second is refused with the typed line above, `49172` bytes and `1 f32 spans`;
the next demote charges the rest).

Byte accounting of one 27B demote (64 tokens, 96 recurrent planes, 157.9 MB of span bytes), before and after:

| term | before (day 30) | after (day 31) |
|---|---|---|
| KV destinations (pinned leases, `alloc_host_kind`) | charged pinned, about 1.9 MB | unchanged |
| the image's pageable residency (`HostPrefixCache`, `prefix_host_bytes`) | 159.8 MB | unchanged |
| the context's span staging set | 157.9 MB pinned, uncharged | 157.9 MB pinned, charged once at the first demote, steady after reuse, released at the latch |

The governor's pinned capacity is twice the host budget (`host_tier_governor`), so the staging (at most one image's
recurrent bytes per distinct length set) fits beside the KV leases whenever the image fits the budget. Limits, stated:
after the latch the charges release even when a quarantined ticket or a detached helper still holds buffers (the memory
is freed when they drop, not before); a multi-model context's set grows with each distinct plane length.

**1c. A span-refusal cell in the fault gate.** A new value of the existing door, `MEMRA_KV_HOST_FAULT=contract-spans`
(no new `MEMRA_*` name; the `docs/FLAGS.md` row updated in the same commit), one-shot: after every recurrent plane of
the evicted entry was built into a span, the attach refuses before `submit_d2h_spans`. The gate's `span-refusal` cell
(`tools/kv-host-contract-fault-gate.sh`), two boots: door ON with the fault, then door OFF (`MEMRA_KV_HOST_CONTRACTS=0`)
as the byte reference, the same four requests (r1 P_A, r2 P_B evicts E_A into the refused demote, r3 P_C evicts E_B
into a completing demote, r4 P_B hits E_B on the host). Asserts: four served; exactly one `demote failed (tier D2H spans
refused: injected failure (MEMRA_KV_HOST_FAULT=contract-spans) (N f32 spans handed back)); nothing demoted`, N equal to
the next copy-complete line's span count and N >= 1; the next D2H receipt says `N f32 spans landed`; the refused ticket
consumed one sequence (next receipt seq 2 plus captures); the next demote publishes; one `tier span staging:` line in
the boot with N fresh buffers (every later demote reused the set); r4 promotes; no latch, no quarantine, no leak, no
other refusal; the OFF boot serves four with the door off and r4 promoting; r1 to r4 byte-equal across the boots. The
helpers were run red and green against a synthetic log before commit. GPU cell `option_b_span_attach_fault_hands_every_span_back`
is the same arm on a card.

**1d. The per-slot-class byte tally.** The `demote copy complete off the tick` line ends with `by slot class: conv C (B
B), ssm S (B B), hidden H (B B), logits L (B B)`, each payload's bytes read as its landed staging length or its heap
length times 4. Log only, no behavior change. Census `day31_the_copy_complete_line_tallies_the_payload_bytes_by_class`
(a mixed fixture and the zero case). This is the separating line lane C named (C DAY38 section 3).

Checks on `33b1285e0`: server lib 838 passed; clippy `-D warnings` on tier, engine and server all targets plus the
`DOCS_RS=1` pass clean; `cargo fmt --all -- --check`, `tools/check-flags.sh`, `tools/check-conflict-markers.sh`,
`git diff --check` clean; local RTX 5090 `option_b_` 7 passed.

## 2. Pre-registration: the H2D half of Move 2 owed item 1 (before any H2D code)

**The read (file:line at `33b1285e0`).** The off-tick promote (`host_promote_park_probe`, `worker.rs:14420`) calls
`device_entry_from_host_parts` with `ContractH2d::OffTick`: the KV planes go to the copy stream first
(`host_kv_planes_submit_promote`, `:11541`, the ticket's submission instant), then every `conv` and `ssm` plane is
uploaded by `engine.htod` (`:13869`, `:13876`), a `clone_htod` on the owner stream from the resident `HostF32::Heap`
`Vec` (pageable). On the target card the second promote of a boot reads `7.6` to `7.7 ms from submission to completion`
and `9.2` to `9.4 ms` end to end (day-30 fault-gate logs, `pro-single-day30/box/gates/contract-fault/promote-*`), and
pinned H2D DMA is `2.98 ms` per 160 MiB (DAY13); so by subtraction (an estimate, not a measurement) about 5 ms of the 7.7 is
the owner thread pushing 157.9 MB of pageable bytes before the submitted line. The first promote of a boot reads 230 to 268 ms on both door arms (a first
touch not attributed here).

**Design options, priced by arithmetic before any code.**

- P (owner memcpy into the staging set, then spans): the owner copies 157.9 MB heap to cached pinned (about 8 to 16 ms
  at one core's 10 to 20 GB/s) before the async H2D. More owner time than today's pageable htod. Refuted.
- R (the resident form stays pinned): the D2H staging becomes the host image's recurrent planes (`HostF32::Pinned`).
  157.9 MB pinned per resident entry over the whole host budget; a residency and ledger change the door does not own.
  Not taken.
- H (the helper fills, the owner submits): taken. Below.

**Design H.**

1. Item class. A typed f32 H2D span, `H2dSpan { source: PinnedHostBuf, destination: CudaSlice<f32> }`, submitted with
   the promote's KV batch on one ticket (`CudaTransfers::submit_h2d_spans` right after the batch). The copy stream
   enqueues one `cuMemcpyHtoDAsync` per span and records one event each; the ticket is `producer_done` only when every
   KV item and every span event is observed; `take_h2d_spans` refuses `NotReady` before that; `retire` refuses `Busy`
   while a span is untaken. The destination is readable on the owner stream only after the consumer wait the promote's
   `ready_view` already installs (rule 3's reader fence), extended over the span events. An enqueue or event error
   quarantines the ticket. Conformance `conformance/h2d_span.rs` (additive, unversioned, `WIRE_VERSION` stays 1) with a
   red arm (an owner read before the wait), a CPU binding, and a native cell.
2. The fill. The resident planes become `HostF32::Heap(Arc<[f32]>)` (a representation change, same bytes, same hash
   input); the promote hands the entry's `Arc`s and staging buffers from the context's set to the hash helper as a
   `Fill` job; the helper copies each plane into its staging buffer and replies; the request stays parked. At the tick
   top the owner submits the KV batch and the spans together (one ticket), then `Promoting` as today. No `unsafe`, no
   borrowed heap across threads.
3. The settle takes the spans back after the landing and before the retire: each destination into the promoted entry's
   `conv` / `ssm` slot, each staging buffer back to the set (the day-31 guard pattern, so every refusal after the take
   returns it).
4. Fail closed. A fill that fails or a helper gone: the promote refuses typed, the host entry is intact, the request is
   served cold, the staging returns (or drops with the helper), the tier stays on unless the helper is gone (the day-28
   latch). A span refused before enqueue: every destination released, every staging buffer back, the KV ticket
   unwound through `host_promote_contract_abort`, `promote refused (contracts door): tier H2D spans refused: ...`. A new
   one-shot value of the existing door, `MEMRA_KV_HOST_FAULT=contract-promote-spans`, and a fault-gate cell for it
   (byte-compared with a door-OFF boot, as day 31's `span-refusal`).

**The receipt term, before and after.** Before: a promoted recurrent plane carries no transfer receipt (a synchronous
owner-stream `clone_htod`'s success); the KV items carry the H2D receipt with `require` against the D2H receipt's
checksums. After: KV unchanged; each recurrent plane carries its span completion under the promote's ticket (event
observed, `valid_bytes` equal to the submission, taken back exactly once before the retire, the ticket's quarantine on
any error), and the H2D receipt line gains `; N f32 spans landed under the ticket and taken back before the retire`
(96 on the 27B, 48 on the 9B) with `items=` still the KV count (so the partial-reject cell's `1 of M items` is
unchanged). The strong form (the helper's SHA-256 of the staged bytes required equal to the plane's share recorded at
the demote, about +73 ms on the helper per promote on the target card) is named and not built.

**Acceptance, stated before running (target card, 27B, the day-26 double-park cell's shape).**

- B1. Identity x4, failure x2, the fault gate every cell (the day-31 set plus the promote-spans cell), twin x2, hit
  OFF/ON armed with the day-24 census, the unit cells (the door's GPU cells, the engine's `d2d_`, `d2h_span` and new
  `h2d_span` cells, the CPU censuses): ALL GREEN; on the 5090, identity default ON, fault default and plain, hit OFF/ON.
- B2. The promote's owner segment, a new field on the `promote submitted off the tick` line (t0 to the line, landed
  first as a log-only commit and read on the pre-H2D binary in the same hold): before, predicted 5 to 8 ms steady;
  after, median at most 1.5 ms and max at most 3.0 ms over at least 20 steady promotes (the second and later of each
  ON boot).
- B3. The promote end to end (`[prefix-host] promote: ... in X ms`), a reading, not a clause: grows by the helper's
  fill (predicted 8 to 16 ms) plus one tick-top poll; the hit request's TTFT pays it, the peers' ticks do not.
- B4. On every contract promote of an ON arm the H2D receipt line names 96 spans (27B) and keeps `items=` at 34 / 32.
- B5. No new numeric program: identity byte digests unchanged in kind; `verify ok: promoted state digest matches
  demote digest` on the identity ON arms; a native cell reads the promoted planes bitwise equal to the resident bytes.
  No new `MEMRA_*` name.

**Budget and today's call.** The D2H half took one lane day (day 30: the engine class, the conformance rule, the worker
arm, twelve cells, one sitting). The H2D half is the same size plus the `Arc` representation change and the parked
`Fill` state. It does not fit the 5 agent-hours beside 1a to 1d and their one-sitting gate run, so it does not land
today. Owed, in order: the log-only owner-segment field and its baseline read; `conformance/h2d_span.rs` and its
binding; `submit_h2d_spans` / `take_h2d_spans` with the native cell; the `Arc` resident form; the `Fill` job and the
parked state; the settle's take-back; the fault value and its gate cell; then the sitting against B1 to B5.

## 3. The sitting (target card, BOX3, one RTX PRO 6000 Blackwell, 600 W; `pro-single-day31/box/`)

Every number in sections 3 to 6 comes from a command over the receipt files, banked beside it: `bash
pro-single-day31/counts.sh <root>` (`pro-single-day31/box/counts.log`, `rtx5090-day31/counts.log`), the day-30 reader
(`python3 day30-reading.py --double-park pro-single-day31/box/double-park/ev --spans 96 --kv 32,34`,
`box/reading-day30-doublepark.log`; `--spans 96 --kv 32,34 --a3-root pro-single-day31/box`, `box/reading-day30-a3.log`;
`--spans 48 --kv 16,18 --a3-root rtx5090-day31`, `rtx5090-day31/reading-day30-a3.log`) and the earlier readers
unchanged (`day28-reading.py`, `day25-double-park-reading.py` over `box/double-park/ev`, `day26-reading.py` over `box`;
`box/reading-day28.log`, `box/reading-day25.log`, `box/reading-day26.log`).

One binary for every cell, `ebfac35382330b023a445c8282e968a9a1945c353125245667fe147dbc6dfd48` (`bins/memra-server.sha256`),
built on `/root/wt-a` at `6d940a97c` (`tree.sha`, `unit/tree.sha`; the code is `33b1285e0`, the later commits move no
`crates/`, `tools/` or `docs/` file). Lane C's scored cell held `/tmp/memra-gpu.lock` first: the driver's bounded retry
logged 27 waits of 120 s, 22:00:36Z to 22:52:37Z (`lock-retries.log`), and nothing of this lane ran beside it. Then
three collector holds and the hit gate's own `flock` in one sitting, 22:54:37Z to 23:27:39Z; no compute app before or
after (`compute apps before/after (data rows): 0/0`); every `CELL.jsonl` closing row `executed-not-qualified False 0`.
Telemetry at 250 ms (`command.gpu.csv`):

| cell | window (Z) | samples | temp | power | memory |
|---|---|---|---|---|---|
| double-park | 22:54:37 to 23:13:59 | 4,635 | 35 to 50 C | 33.5 to 355.6 W | 0 to 17,109 MiB |
| gates | 23:13:59 to 23:23:19 | 2,238 | 37 to 60 C | 87.3 to 507.6 W | 0 to 21,939 MiB |
| hit gate (own flock) | 23:23:20 to 23:24:50 | (no collector) | | | |
| unit-cell | 23:24:50 to 23:27:39 | 673 | 32 to 40 C | 31.5 to 96.3 W | 0 to 627 MiB |

The double-park cell is day 26's script byte for byte (one hold, twenty boots, N=5 per arm per order, both orders):
`STALL REPLAY: PASS` 20 of 20, `DAY26 DOUBLE-PARK ADMISSIBLE all_receipts=True`.

## 4. Verdicts, verbatim

Target card (`box/gates/`, `box/unit/`, `box/double-park/`):

- Identity x4 (default OFF, default ON, plain OFF, plain ON): `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)`, 12 ok each.
- Failure x2 (OFF, ON): `KV-HOST-SPILL FAILURE GATE: ALL GREEN`, 15 ok each (the two `FAIL` substrings per log are the
  gate's own `ok: the promote caught it: VERIFY FAILED, loud and named` lines).
- Contract fault, eleven cells with the new `span-refusal`: `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN`, 142 ok, 0 FAIL.
  The span-refusal cell: `refusal handed back 96 span(s); the next copy-complete carries 96`, `the next D2H receipt
  landed 96 span(s), 96 were handed back`, `receipt seq=5 expected 2 + 3 capture ticket(s) submitted before it = 5`,
  `staging fill line(s) [96], refusal N 96`, `byte-unequal request(s): none`, `ok: span-refusal: r1..r4 byte-equal to
  the door-OFF boot`; the served line `[prefix-host] demote failed (tier D2H spans refused: injected failure
  (MEMRA_KV_HOST_FAULT=contract-spans) (96 f32 spans handed back)); nothing demoted`; r4's `promote published off the
  tick: ticket complete after 1 poll(s), 7.8ms from submission to completion (tick-top poll)`.
- Twin x2 (OFF, ON): `PREFIX-NEWEST-TURN-FITS: budget_bytes=1073741824 cohort_bytes=736755712 turns=8 ... V1=ok V2=ok
  V3=ok V4=ok V5=ok V6=ok -> PASS`.
- Hit OFF/ON, ON armed: `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)`, 61 / 68 ok; the day-24 census: `DAY26 CLAUSE 5
  hitgate-on counts against day 24: spec_on_census_equal=True spec_off_census_equal=True route_submissions=30 (day 24:
  30) spec_boundary_captures_with_draft_plane=11 (day 24: 11) not_routed_lines=0 (must be 0: the gate has no
  promote-then-hit shape) -> PASS`.
- Unit cells: the door's GPU cells `test result: ok. 13 passed` (the day-30 seven `option_b_` and six `option_c_`,
  including `option_b_span_attach_fault_hands_every_span_back`, `option_b_span_postpublish_refusal_returns_the_staging_to_the_set`,
  `option_b_span_staging_charge_refusal_refuses_the_attach_and_keeps_the_tier_on`, `option_b_span_refusal_returns_every_plane_and_keeps_the_tier_on`
  and `option_b_spans_ride_the_ticket_and_land_bitwise`); the engine's `test result: ok. 6 passed` (the four `d2d_`
  cells, the price reading, and `d2h_span_batch_lands_with_its_ticket_on_the_copy_stream ... ok`); the CPU censuses
  `test result: ok. 13 passed` (with `day31_the_staging_goes_back_on_every_exit_and_is_charged_once` and
  `day31_the_copy_complete_line_tallies_the_payload_bytes_by_class`); the tier bindings `test result: ok. 3 passed`.
- Double park (day 28's cell): `DAY30 A2 pre-submit steady N=80 median=0.63 min=0.60 max=0.66 boots_on=10
  demotes_per_boot=[11] rule N>=80 median<=1.5 max<=3.0 -> PASS`; `DAY30 A3 logs=115 copy_complete_lines=134
  receipts_paired=134 receipts_without_copy_line(on-tick)=0 bad=0 ... -> PASS`; `DAY28 CLAUSE 1a stall order=o1 ...
  on_cell_median=76.9 off_cell_median=85.4 rule on<=off+2.0 -> PASS` (o2 the same), `DAY28 CLAUSE 1b e2e order=o1 ...
  on_minus_off=+11.8 rule <=+20.0 -> PASS` (o2 `+11.8`), `DAY28 CLAUSE 1c owner in-completion N=100 median=2.02 min=1.96
  max=2.66 runs_with_demote_without_ledger=0 rule <=12.0 -> PASS`, `DAY28 VERDICT clauses_failed=0 -> ALL PASS`;
  `DAY25 DOUBLE-PARK stall order=o1 on_minus_off=-8.4 unc=0.1 -> isolated`, `order=o2 on_minus_off=-8.5 unc=0.2 ->
  isolated`, e2e `+11.8 unc=0.9` / `+11.8 unc=0.8 -> isolated`.
- Day 26's reader (its own clauses, not today's acceptance): clause 1 `-> PASS`; clause 2 `|d-expected|=3.9` / `4.0 ->
  FAIL` against its day-26 expectation of +15.8 (day 30 read 3.8 / 4.1 the same way); clause 3 `-> FINDING` both orders;
  clause 4 `restore-arm: NO RECEIPT -> FAIL` (the restore arm is not part of this cell); clause 5 `-> PASS` (the hit
  gate now sits under `gates/`).

Local RTX 5090 (`rtx5090-day31/`, binary `0427e65c30c85917cadc3297bbe6cce69cb65c0f2d835b101a9b2ca461cf7b6a`, the same
`6d940a97c`, each gate under its own `flock` on `/tmp/memra-5090.lock`, 22:01:21Z to 22:05:38Z, no wait logged):
identity default ON `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` (12 ok); fault default and plain
`KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` (142 ok each; the span-refusal cell `refusal handed back 48 span(s); the next
copy-complete carries 48`, `byte-unequal request(s): none`); hit OFF `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` (61 ok),
hit ON the same (68 ok); `DAY30 A3 logs=39 copy_complete_lines=38 receipts_paired=38 ... -> PASS`.

No arm is red.

## 5. What the sitting shows for 1a to 1d

- **1a on the served path.** The `postpublish` cell refuses a landed receipt after the take. Day 30: the boot's next
  demote paid a fresh staging allocation, `pre-submit 28.71`. Day 31: `pre-submit 0.93`, and the boot has one
  `tier span staging:` line (before the refusal). DAY30 finding 4 is closed.
- **1b.** Every server log that carried a demote has exactly one fill line: 22 of 22 on the target card (`tier span
  staging: 96 fresh pinned buffer(s), 156893184 bytes charged to the governor's pinned ledger; the set holds 156893184
  bytes charged`), 19 of 19 on the 5090 (48 buffers, 52690944 bytes). No `Capacity` refusal anywhere: the one span
  refusal per card is the injected one. The charge moved no double-park reading (same box, cross-sitting, not a
  same-window pair): pre-submit steady 0.63 (day 30 0.62), demote 1 of a boot 29.71 (29.69), demotes 2 and 3 1.25 /
  1.23 (1.26 / 1.27), helper 79.8 (79.8), owner in-completion 2.02 (1.96), stall ON against OFF -8.4 / -8.5 (-8.3 /
  -8.6).
- **1c.** Section 4; the cell is green on both cards and both environments of the 5090.
- **1d.** The byte split by slot class, read off the copy-complete lines (`counts.log`, `day 31` section):

| image | conv | ssm | hidden | logits | total |
|---|---|---|---|---|---|
| 27B plain (112 lines) | 48, 5,898,240 B (3.7 %) | 48, 150,994,944 B (95.6 %) | 1, 0 B | 1, 993,280 B (0.63 %) | 157,886,464 B |
| 27B draft-bearing (22 lines) | as above | as above | 1, 20,480 B | as above | 157,906,944 B |
| 9B plain (18 lines) | 24, 2,359,296 B (4.4 %) | 24, 50,331,648 B (93.8 %) | 1, 0 B | 1, 993,280 B (1.85 %) | 53,684,224 B |
| 9B draft-bearing (20 lines) | as above | as above | 1, 16,384 B | as above | 53,700,608 B |

## 6. Findings

1. The SSM state is the image: 95.6 percent of the 27B's heap payload bytes and 93.7 to 93.8 percent of the 9B's. Conv
   is 3.7 / 4.4 percent, logits under 2 percent, hidden 0 or one row. This is lane C's separating line (C DAY38 section
   3), now a log term on every demote.
2. A plain entry carries a hidden payload of 0 bytes (counted in the 98 / 50 payloads, empty); a draft-bearing entry
   carries one row (5120 f32 on the 27B, 4096 on the 9B). Log only; no behavior read into it.
3. The staging charge costs nothing measurable on the served path and never refused at these budgets (the governor's
   pinned capacity is twice the host budget). The refusal arm is exercised by the GPU cell only (a hog charge); no served
   cell drives the ledger to Capacity.
4. The box was held 54 minutes by lane C's scored cell before this sitting; the bounded retry held, no overlap.

## 7. What is owed, and integrability

- **Integrable.** 1a to 1d are complete, with their censuses, GPU cells and the fault gate's new cell, and every gate
  is green on both cards on `6d940a97c` in one sitting per card. The lead integrates from `origin/lane/spill-a-20260919`.
- **Not landed: the H2D half** (section 2: design H, acceptance B1 to B5, the ordered list). Nothing of it is in the
  tree; no H2D claim is made.
- Still owed after it: the D2D half, the strong-form receipt; Move 1 items 1, 3, 4 unchanged. The 1b limits in section 1
  (charges release at the latch while a quarantined ticket or a detached helper may still hold buffers; a multi-model
  context's set grows per distinct plane length) stand.

## 8. Checks, budget, cleanup

- Checks on the final tree (the records commit): `cargo fmt --all -- --check` clean; clippy `-D warnings` all targets on
  tier, engine and server `clippy_rc=0`; the `DOCS_RS=1 --target x86_64-unknown-linux-gnu` pass `docsrs_rc=0`;
  `check-flags: every runtime MEMRA_* name resolves against 'docs/FLAGS.md' (no grandfather list)`;
  `check-conflict-markers: OK`; `git diff --check` clean; `.gitattributes` with `*.log -whitespace` in
  `pro-single-day31/box/` and `rtx5090-day31/`; zero em dashes in this lane's lines. No new `MEMRA_*` name.
- Budget: about 2.5 agent-hours of 5 (21:15Z to about 23:45Z wall), of which 54 minutes waited on lane C's lock.
- Cleanup: BOX3 `/root/a31.bundle` removed; `/root/wt-a` at `6d940a97c` on `lane-a-day31`, clean; `/root/spill-receipts/a-day31/`
  mirrored to `pro-single-day31/box/` (the binary excluded, its sha256 kept); no process of mine on the box; the lock
  free; nothing of other lanes touched. Local: `/tmp/wt-a-d31` removed; the 5090 carries no process of mine.
