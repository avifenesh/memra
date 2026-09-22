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
