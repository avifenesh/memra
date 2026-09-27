# WP-A day 32: the H2D half of Move 2 owed item 1 (design H), landed and sat against B1 to B5

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`. Rig: the target card (BOX3, one RTX PRO 6000 Blackwell, 96 GB,
600 W, `research/spill-lead-20260919/BOX-ACCESS.md`) and the local RTX 5090 Laptop GPU; no cross-card comparison.
Every cell `executed-not-qualified`. Every push in the announced `MEMRA_RELEASE_QUALIFICATION_MODE=development` mode
(logged; no qualification claimed). Scope: DAY31 section 2 exactly (design H, acceptance B1 to B5, its ordered list).

## 0. Resync and the merge

- Post-block resync: `git fetch`; the worktree was clean at `979881aa0` (day 31, on `main` through #665), no stash, no
  day-32 file. `origin/main` was `7029cd67c` (#668 spec round guards and `tools/spec-ctx-edge-gate.sh`, #671 lane B day
  31, the dsv4 PRs); `979881aa0` is its ancestor, so the merge was a fast-forward (no conflict, no behavior to resolve).
- `origin/main` then moved to `db5929017` (#677, memra#641: the fresh batched prime runs the per-seq solo attention
  core; `hybrid_forward.rs` `prime_cache_batch_inner`, no host-tier path). Merged as `71d21057c` before the first
  engine push; no conflict (#677 touches no file this lane touches).
- After the records, `origin/main` moved to `649d96210` (#681, integ51: lane B day 32, plus #678 dsv4 latency kernels).
  Merged as `17be62a14`, no conflict (both sides' `research/INDEX.md` rows present, the `docs/FLAGS.md` row with
  `contract-promote-spans` intact, `docs/TESTING.md` carries both additions); #681 moves no file of the host-tier path
  (the dsv4 kernels and their docs), so the sitting's verdicts stand on the code they name. `cargo check -p
  memra-server --tests`, `cargo fmt --all -- --check`, `check-flags` and `check-conflict-markers` clean on the merge.

## 1. What landed, in DAY31 section 2's order, one census each

1. **The owner-segment field, log only, first** (`3cabdd4a3`). The `promote submitted off the tick` line gains
   `owner segment X.XXms` (t0 to the line) before `request parked`, so every reader keyed on the line's end (lane C's
   day-35, 37 and 39 stall readers) still matches. Census: `every_path_that_meets_a_promoting_entry_settles_it_first`
   pins the field and its read before the line. The pre-H2D binary B2 reads is built at `71d21057c` (this field, the
   tier conformance of item 2, which no runtime path calls, and #677; none of the H2D code).
2. **`conformance/h2d_span.rs`** (`e30bd5cbf`, additive, unversioned, `WIRE_VERSION` stays 1). Rules 1 to 5 of the
   H2D span batch: refusal before enqueue is whole; one landing over items and spans; a destination is handed out
   only behind the reader wait over every span event (rule 3 of `reader_fence`, extended: landing is not a fence);
   back exactly once before retirement, `valid_bytes` equal to the attach; an enqueue or event error quarantines.
   Three schedules (`h2d_span_batch`, `h2d_span_enqueue_failure_quarantines`, and the red arm
   `h2d_span_read_before_its_wait_is_unordered`: a binding that hands the spans back on the landing alone lets the
   reader stream read a destination before its wait). CPU binding `tests/contracts/h2d_span_bindings.rs`: 3 passed,
   the red arm's one read recorded unordered.
3. **The engine class** (`148dc4a29`). `H2dSpan { source: PinnedHostBuf, destination: CudaSlice<f32> }`;
   `CudaTransfers::submit_h2d_spans` (admission first: copy stream, a live uncancelled unquarantined H2D ticket
   with no spans of either class, a fully written source of the destination's byte length, a destination on the
   owner stream; then an owner-stream event, the copy stream's wait, one `cuMemcpyHtoDAsync` and one event per
   span, `PinnedHostBuf::enqueue_to_device_f32`, no host wait) and `take_h2d_spans` (`NotReady` until the batch
   landed AND `install_consumer_wait` put the owner stream's wait on every span event). `progress` folds the span
   events into `producer_done`; `retire` is `Busy` while a span is untaken; an unretired entry's drop forgets its
   spans. The D2H span code is unchanged. Census `h2d_span_rules_are_as_stated`; native cell
   `h2d_span_batch_lands_with_its_ticket_on_the_copy_stream` (three spans of 3, 5 and 4 MiB; a length mismatch and
   an unwritten source refused whole; a 300 ms copy-stream hold so the KV item lands first and the take is
   `NotReady`, a second attach `Busy`; after the host wait the take is STILL `NotReady` until the reader wait; the
   owner stream reads each destination bit for bit; the injected second-enqueue fault quarantines).
4. **The resident form** (`441ef6607`): `HostF32::Heap(Arc<Vec<f32>>)`, same bytes, same hash input, same bind
   program. Named departure from the pre-registered `Arc<[f32]>`: `Arc<[f32]>::from(Vec<f32>)` copies the bytes
   into a fresh allocation (the reference-count header sits in front of the data), which would put a second full
   copy of every demoted recurrent plane on the door-OFF demote's owner thread (`host_glm::read_f32` returns the
   `Vec`); `Arc::new(vec)` moves it. The sharing and the immutability are the same. Server lib 875 passed.
5. **The `Fill` job and the parked state** and 6. **the settle's take-back** (`6461a9106`, one commit: the fill
   without the take-back has no settle to land in). At the admission probe, on the off-tick contract route with a
   copy stream and resident heap recurrent planes (`host_promote_fill_handoff`): one staging buffer per plane from
   the context's set (charged when fresh, the day-31 rule, the same `tier span staging:` line), the planes as owned
   `Arc` clones, ONE `Fill` job to the hash helper (`HostHelperJob::Fill`, its own reply channel); the request parks
   and the new line `promote staging fill off the tick: N tokens, 96 f32 planes (156.9MB) handed to the hash helper
   (fill seq=K); the request waits for the fill` prints (no `request parked` in it: that phrase is the submission
   line's and three readers count it). The helper fills each buffer with `copy_from_slice` (a checked full write,
   no `unsafe`). The tick-top poll that lands the fill (`host_promote_fill_step`) finds the entry again by id and
   submits through the unchanged route (`device_entry_from_host_parts`, `OffTick`, the staging attached): the KV
   batch, then `host_h2d_spans_submit` (one fresh uninitialized owner-stream destination per plane,
   `Engine::alloc_f32_uninit`, then `submit_h2d_spans`); the shell leaves the spanned slots empty; then the
   submitted line with the owner segment by part. The settle (`host_kv_planes_settle_promote`, step 6c) takes the
   spans back after the landing and the reader wait and before the receipt check, publication and the retire: each
   destination into its slot of the shell, each staging buffer into a `HostStagingBack` guard whose `Drop` returns
   it to the set on every exit, success or refusal. The H2D receipt line keeps `items=` as the KV count and gains
   `; N f32 spans landed under the ticket and taken back before the retire`. Fail closed: a staging refusal or a
   failed fill refuses typed with the tier on and the staging back; a helper gone or a fill past the 10 s deadline
   latches; a reply that does not describe the fill latches; an entry that left refuses typed; the tenant purge
   (no engine in hand) drops a Filling entry typed once its fill lands; a demote meeting a Filling entry lands its
   fill and submits with its own engine first. Census `day32_the_h2d_spans_ride_the_promote_ticket_in_the_stated_order`
   (the probe hands off before any submission; the step's reply, id lookup, submission and line in that order; the
   KV batch before the spans, the destinations before the attach, the abort after it; the settle's reader wait,
   guard, take, receipt, publication and retire in that order; the fill arm has no `unsafe`; every settle path runs
   the Filling step first). CPU cells `day32_a_filling_promote_waits_and_drops_typed_without_the_engine` and
   `day32_a_failed_or_lost_staging_fill_is_typed_and_the_lost_ones_latch`; GPU cells
   `option_c_spans_ride_the_promote_ticket_and_land_bitwise` (B5's native cell: through the real helper's fill, the
   promoted planes read bitwise equal to the resident bytes and to the device planes the demote read, every KV plane
   whole, the host twin intact and sole owner, every staging buffer back charged once, the latch clears the ledger)
   and `option_c_span_postpublish_refusal_returns_the_staging_to_the_set`.
7. **The fault value and its gate cell** (`4d1335cd6`). `MEMRA_KV_HOST_FAULT=contract-promote-spans`, a one-shot
   value of the existing door taken by the promote route only (the `docs/FLAGS.md` row updated in the same commit, no
   new `MEMRA_*` name): after every span was built from the filled staging, the attach refuses before
   `submit_h2d_spans`; typed `promote refused (contracts door): tier H2D spans refused: injected failure
   (MEMRA_KV_HOST_FAULT=contract-promote-spans) (N f32 spans handed back); serving without the host entry`. Gate cell
   `promote-span-refusal` (two boots, door ON with the fault then door OFF, the promote cells' four requests): one
   typed refusal, N equal to the next H2D receipt's span term, the fill before the refused and the next submission,
   one staging fill in the boot, the next promote landing and publishing, no latch, quarantine, leak or other
   refusal, r1..r4 byte-equal to the door-OFF boot. The three new helpers were run red and green against synthetic
   logs before commit (`rtx5090-day32/gatecheck/helpers-red-green.txt`). GPU cell
   `option_c_span_attach_fault_hands_every_span_back`. `docs/TESTING.md` gains the span cells' line (`5ae139734`).

## 2. What the design became on the served path, stated before the sitting

- **B2's field on the H2D binary.** The submission now happens at the poll that lands the fill, one or more ticks
  after t0, while the request is parked and the owner thread serves other ticks. The field reads the owner thread's
  held time for the promote up to the line: the probe's segment (t0 to the hand-off), every Filling poll's, and the
  submission's (its start to the line), printed by part with the wall from t0: `owner segment X.XXms (probe A, fill
  polls B, submit C; wall W ms from t0)`. On the pre-H2D binary the two are the same (no parked interval). B2's
  clause (median at most 1.5 ms, max at most 3.0 ms, at least 20 steady promotes) is read on this field as it stands;
  B3 reads the wall.
- **Rule 3 is stronger than the pre-registration's words.** DAY31 section 2 item 1 has `take_h2d_spans` refuse
  `NotReady` before the landing; the engine here also refuses it after the landing until the owner stream's wait on
  every span event is installed, so no caller can name a destination before a read of it is ordered (the reader
  fence's own rule for the items). The settle installs the wait before the take, so the served order is the same.
- **The helper runs one job at a time by construction.** The probe settles a Demoting entry (with its `Hashing`
  phase) before a fill is handed; a demote settles a Filling entry (fill, submission, contract) before its own hash
  job. Fills have their own reply channel, so a reply of one kind is never read as the other.

## 3. The sitting (target card, BOX3, one RTX PRO 6000 Blackwell, 600 W; `pro-single-day32/box/`)

Lane B's pre-registered two-order chain held the box until `2026-09-23T13:40:42Z order=O2 done` (its `order.log`); at
13:45Z no compute app and no lock holder. Nothing of this lane ran inside B's orders, and no build ran before its end.
Then, all on BOX3 (`pro-single-day32/run.log`, which ends `LANE-A-PRO-DONE`):

- Build (`build.sh`, from one bundle, in `/root/wt-a`): the pre-H2D binary at `71d21057c`,
  `17ce33ad8c5518d81423eaa6c6033d008994572725d073b3e4eb37e6932ba5df` (`bins/memra-server-pre.sha256`), and the H2D
  binary at `5ae139734` (the code is `4d1335cd6`; the later commits move research files and `docs/TESTING.md` only),
  `3331dc0e064689ecb9ed980f91329378f14f8a94181640ebc72be574d39ca708` (`bins/memra-server.sha256`, `tree.sha`).
- One collector hold, `doublepark-pair`, 13:52:49Z to 14:31:32Z: the day-26 double-park cell byte for byte on the
  pre-H2D binary (13:52Z to 14:12:11Z, `box/pre/double-park/`), then on the H2D binary (14:12:11Z to 14:31:33Z,
  `box/double-park/`). Twenty boots each, N=5 per arm per order, both orders; `STALL REPLAY: PASS` 20 of 20 in each,
  `errors=0` in all 40 receipts; compute apps before and after `0/0` in each.
- Then `gates` (collector, 14:31:33Z to 14:41:26Z), the hit gate under its own `flock` (14:41:26Z to 14:42:57Z), and
  `unit-cell` (collector, 14:42:57Z to 14:46:24Z). Zero lock retries. Every `CELL.jsonl` closing row
  `executed-not-qualified False 0`.

Telemetry at 250 ms (`command.gpu.csv`, `box/counts.log`):

| cell | window (Z) | samples | temp | power | memory |
|---|---|---|---|---|---|
| doublepark-pair (pre, then H2D) | 13:52:49 to 14:31:32 | 9,265 | 33 to 51 C | 34.0 to 362.6 W | 0 to 17,109 MiB |
| gates | 14:31:33 to 14:41:26 | 2,369 | 37 to 60 C | 86.3 to 507.2 W | 0 to 21,939 MiB |
| hit gate (own flock) | 14:41:26 to 14:42:57 | (no collector) | | | |
| unit-cell | 14:42:57 to 14:46:24 | 826 | 32 to 41 C | 32.0 to 98.7 W | 0 to 627 MiB |

Every number in sections 4 to 6 comes from a command over the banked files, run from `research/spill-a-20260919`:
`python3 day32-reading.py --b2 pro-single-day32/box/pre/double-park/ev pro-single-day32/box/double-park/ev`
(`box/reading-day32-b2.log`); `python3 day32-reading.py --b4 pro-single-day32/box --spans 96 --kv 32,34 --exclude pre`
(`box/reading-day32-b4.log`); `bash pro-single-day32/counts.sh pro-single-day32/box` (`box/counts.log`); the day-30,
28, 25 and 26 readers unchanged over the H2D run (`box/reading-day30-doublepark.log`, `box/reading-day30-a3.log`,
`box/reading-day28.log`, `box/reading-day25.log`, `box/reading-day26.log`) and the day-30, 28 and 25 readers over
the pre-H2D run (`box/pre/reading-*.log`, the day-26 reader over `box/pre` too).

A reader scope defect, fixed before any B4 verdict was taken from it: the first B4 run over `box/` walked into
`box/pre/`, the pre-H2D binary's run, whose 100 receipts carry no span term by construction (`receipts=209 bad=100 by
(items, spans)=[((32, 0), 100), ((32, 96), 101), ((34, 96), 8)] ... -> FAIL`, banked as
`box/reading-day32-b4-first-run-with-pre.log`). The reader gained `--exclude NAME`; the clause is unchanged. The same
switch keeps the 5090 run out of `rtx5090-day32/gatecheck/`, whose synthetic logs carry 27B-shaped lines.

## 4. Verdicts against B1 to B5, verbatim

**B1, target card** (`box/counts.log` verdict section, each gate log):

- Identity x4 (default OFF, default ON, plain OFF, plain ON): `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)`, 12 ok
  each.
- Failure x2 (OFF, ON): `KV-HOST-SPILL FAILURE GATE: ALL GREEN`, 15 ok each.
- Contract fault, twelve cells: `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN`, 160 ok (`presubmit` 11, `postpublish` 11,
  `promote-presubmit` 11, `promote-postpublish` 11, `promote-reject` 14, `promote-readyview` 11, `d2d-capture` 12,
  `d2d-restore` 14, `hash-helper-gone` 14, `hash-never-lands` 14, `span-refusal` 19, `promote-span-refusal` 18). The
  new cell: `promote refusal handed back 96 span(s); the next H2D receipt landed 96`, `fill lines 2, refusal at 73,
  span submissions 1`, `staging fill line(s) [96], promote refusal N 96`, `byte-unequal request(s): none`, `ok:
  promote-span-refusal: r1..r4 byte-equal to the door-OFF boot`.
- Twin x2 (OFF, ON): `PREFIX-NEWEST-TURN-FITS: budget_bytes=1073741824 cohort_bytes=736755712 turns=8
  cold_turns_after_1=0 cached_ok=7/7 lines_ok=8/8 evictions=9 cohort_evictions=3 self_evictions=0 refused_or_skipped=0
  effective_free_ok=8/8 identity_ok=8/8 grid_ok=21/21 grid=32 off_grid_calls=0 V1=ok V2=ok V3=ok V4=ok V5=ok V6=ok ->
  PASS`.
- Hit OFF/ON: `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)`, 61 / 68 ok; the day-24 census: `DAY26 CLAUSE 5 hitgate-on
  counts against day 24: spec_on_census_equal=True spec_off_census_equal=True route_submissions=30 (day 24: 30)
  spec_boundary_captures_with_draft_plane=11 (day 24: 11) not_routed_lines=0 (must be 0: the gate has no
  promote-then-hit shape) -> PASS`.
- Unit cells (`box/unit/`): the door's GPU cells `test result: ok. 16 passed` (seven `option_b_`, nine `option_c_`,
  with `option_c_spans_ride_the_promote_ticket_and_land_bitwise`, `option_c_span_attach_fault_hands_every_span_back`
  and `option_c_span_postpublish_refusal_returns_the_staging_to_the_set`); the engine's `test result: ok. 7 passed`
  (four `d2d_`, the price reading, `d2h_span_batch_lands_with_its_ticket_on_the_copy_stream`,
  `h2d_span_batch_lands_with_its_ticket_on_the_copy_stream`) and its censuses `ok. 2 passed`; the CPU cells `ok. 17
  passed` (the day-28 to day-31 set plus the three `day32_` cells); the tier bindings `ok. 6 passed`.
- **B1 on the target card: ALL GREEN.**

**B1, local RTX 5090** (`rtx5090-day32/`, binary `7b268d3151bf8ddd617f29eab810ed693ac1a0c6aaaf1e9eafeb132a9255eada`
built at `4d1335cd6`, the gate tree `a86a39d22`, each gate under its own `flock` on `/tmp/memra-5090.lock`, 12:28:54Z
to 12:33:40Z, no wait logged, the 9B NVFP4 MTP artifact): identity default ON `KV-HOST-SPILL IDENTITY GATE: ALL GREEN
(teeth=0)` (12 ok); fault default and plain `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` (160 ok each; `promote refusal
handed back 48 span(s); the next H2D receipt landed 48`, `byte-unequal request(s): none`); hit OFF `SPEC-ON-CACHE-HIT
GATE: ALL GREEN (qwen)` (61 ok), hit ON the same (68 ok); `DAY30 A3 logs=43 copy_complete_lines=44 receipts_paired=44
... -> PASS`. Unit cells on this card before the sitting (`rtx5090-day32/unit/`): the door's 15 of 15 with the two
day-32 cells of `6461a9106` and then the three of `4d1335cd6`; the engine's two span cells 3 of 3 serial runs.
**B1 on the 5090: ALL GREEN.**

**B2** (`box/reading-day32-b2.log`):

- `DAY32 B2 before (pre-H2D binary) steady owner-segment N=90 median=4.58 min=4.46 max=5.11 boots_on=10 first-of-boot
  owner-segment N=10 median=4.57 min=4.54 max=4.97 (predicted 5 to 8 ms steady: did not hold)`
- `DAY32 B2 after (H2D binary) steady owner-segment N=90 median=0.70 min=0.65 max=1.22 boots_on=10 submissions=100
  with_spans=100 rule N>=20 median<=1.5 max<=3.0 every-submission-with-spans -> PASS`
- **B2 PASSES**: 4.58 ms to 0.70 ms steady (N=90 each, same hold), max 1.22. The baseline came in under its
  predicted 5 to 8 ms (the DAY31 subtraction estimate was "about 5 ms"); stated, not a clause.

**B3, a reading** (`box/reading-day32-b2.log`):

- `DAY32 B3 READING promote in-ms steady before in N=90 median=20.70 min=20.50 max=22.00; after in N=90 median=31.00
  min=30.90 max=32.30; after-minus-before median +10.30`
- `DAY32 B3 READING submission-to-completion steady before ms N=90 median=19.60 min=19.50 max=19.90; after ms N=90
  median=16.00 min=16.00 max=16.40`
- `DAY32 B3 READING helper fill steady ms N=90 median=6.40 min=6.20 max=6.50; fill landed at poll [1] (counts [90])`
- The promote end to end grows by 10.3 ms: the helper's fill (6.4 ms, under the predicted 8 to 16) plus the wait for
  the tick-top poll that lands it (every fill landed at its first poll). The copy's submission to completion falls
  from 19.6 to 16.0 ms: the old instant was taken before the 96 synchronous pageable uploads, the new copy carries
  the 156.9 MB from pinned staging on the copy stream.

**B4** (`box/reading-day32-b4.log`, over the H2D binary's cells): `DAY32 B4 receipts=109 bad=0 by (items,
spans)=[((32, 96), 101), ((34, 96), 8)] rule spans==96, items in [32, 34] and == 2 x planes (+2 with the draft) ->
PASS`. `counts.log`: `109 published retired acknowledged; 96 f32 spans landed under the ticket and taken back before
the retire`, `H2D receipts without a span term: 0`, `submissions without spans: 0`. On the 5090
(`rtx5090-day32/reading-day32-b4.log`): `DAY32 B4 receipts=13 bad=0 by (items, spans)=[((16, 48), 6), ((18, 48), 7)]
... -> PASS`. **B4 PASSES on both cards.**

**B5**: the identity gate's four arms `ALL GREEN (teeth=0)` (byte digests OFF against ON unchanged in kind); `verify
ok: promoted state digest matches demote digest` once in each identity arm's `host-on-server.log` (the ON arms through
the H2D spans, `counts.log` `verify ok lines: 4`; the four `VERIFY FAILED` lines are the failure gate's `digest` cell,
its expected catch); the native cell `option_c_spans_ride_the_promote_ticket_and_land_bitwise ... ok` on the target
card and on the 5090; `check-flags: every runtime MEMRA_* name resolves against 'docs/FLAGS.md' (no grandfather
list)`, no new `MEMRA_*` name. **B5 PASSES.**

**B1 to B5 hold on the target card, and B1, B4 and B5 hold on the 5090 (B2 and B3 are target-card clauses).**

## 5. The earlier readers over the same hold (their own clauses, not today's acceptance)

Pre-H2D run (`box/pre/`) against the H2D run (`box/`), same box, same hold:

| reading | pre-H2D binary | H2D binary |
|---|---|---|
| `DAY28 CLAUSE 1a` stall ON / OFF, o1 / o2 | 76.8 / 85.1, 76.9 / 85.3 `-> PASS` | 76.0 / 85.1, 76.1 / 85.2 `-> PASS` |
| `DAY28 CLAUSE 1b` e2e ON minus OFF, rule `<=+20.0` | `+12.2`, `+12.1 -> PASS` | `+22.5`, `+22.2 -> FAIL` |
| `DAY28 CLAUSE 1c` owner in-completion | `N=100 median=2.02 ... max=3.07 -> PASS` | `N=100 median=2.00 ... max=2.91 -> PASS` |
| `DAY28 VERDICT` | `clauses_failed=0 -> ALL PASS` | `clauses_failed=2 -> FAIL` |
| `DAY25` stall ON minus OFF | `-8.2`, `-8.4 -> isolated` | `-9.1`, `-9.2 -> isolated` |
| `DAY25` promote_in / promote_completion (ON) | 20.7 / 19.6 | 31.1 / 16.0 |
| `DAY30 A2` pre-submit steady | `N=80 median=0.63 min=0.58 max=1.08 -> PASS` | `N=80 median=0.59 min=0.54 max=1.04 -> PASS` |
| `DAY26 CLAUSE 2` e2e distance from its expected +15.8 | 3.6 / 3.6 `-> FAIL` | 6.8 / 6.5 `-> FAIL` (day 31: 3.9 / 4.0 `-> FAIL`) |

`DAY30 A3 logs=166 copy_complete_lines=247 receipts_paired=247 ... -> PASS`; the demote side is unchanged (helper
steady 79.80, copy 92.35, wall 179.20 ms; day 31, same command, 79.70, 92.40, 179.15; a cross-sitting reading, not a pair).

## 6. Findings

1. **The promote's owner segment is off the tick: 4.58 to 0.70 ms steady** (B2, N=90 each, one hold). The tenant's
   stall ON against OFF moved from -8.2 / -8.4 to -9.1 / -9.2 ms in the same hold: the 96 synchronous pageable uploads
   no longer sit on the owner thread.
2. **The hit request pays the fill, and that breaks DAY28's clause 1b.** The promoting request's end to end is +10.3
   ms (B3: the 6.4 ms fill plus the wait for its tick-top poll), so the double-park e2e ON minus OFF reads `+22.5` /
   `+22.2` against DAY28's pre-registered `<=+20.0` (`+12.2` / `+12.1` on the pre-H2D binary in the same hold). B3
   predicted the direction ("the hit request's TTFT pays it, the peers' ticks do not"); DAY31 section 2 set no
   bound on it and DAY28's bound was not re-registered. Stated, not tuned: no clause, threshold or cell moved. The
   trade is the design's: about 3.9 ms less owner-thread time per promote (every peer's tick) for about 10.3 ms
   more on the one parked request. The obvious lever, landing the fill inside the same tick instead of the next
   poll, is not built.
3. **The baseline was under its prediction**: 4.58 ms steady, not 5 to 8 ms (DAY31's subtraction estimate was about 5 ms).
4. **The staging set serves both directions without growth**: one `tier span staging:` fill per boot in every cell
   that demotes (`staging fill line(s) [96]`); the promote's fill reused it every time (114 fills, no second fill line
   in any boot; every fill landed at its first poll).
5. **The two native span cells share the primary context.** Run in parallel in one process on the 5090, the H2D cell
   failed once with `a batch with a running span has not landed` (`rtx5090-day32/unit/engine-span-cells-parallel.log`);
   serial runs were 3 of 3 green, and every sitting runs them with `--test-threads=1`. The cause is not isolated.
6. On the 5090 (gate traffic, not a cost cell): `owner segment` median 0.43 / 0.41 ms in the fault gate's default and
   plain arms (N=8 each), helper fill 3.9 ms for 52.7 MB in the identity arm.

## 7. What is owed, and integrability

- **Integrable as a complete H2D half against B1 to B5**: every pre-registered clause holds on the target card, and
  the 5090 half is ALL GREEN. Every cell is executed-not-qualified. The lead integrates from
  `origin/lane/spill-a-20260919`, with finding 2 as the decision it carries: under the door the promoting request is
  about 10 ms slower end to end than on day 31 (the DAY28 1b bound is breached by 2.2 to 2.5 ms), in exchange for the
  owner-thread segment going from 4.58 to 0.70 ms.
- Owed on Move 2 item 1: **the D2D half** (the capture and restore recurrent copies; the capture races the next
  decode, so its order needs its own rule) and **the strong-form receipt** (the helper's SHA-256 of the staged bytes
  required equal to the plane's share recorded at the demote, about +73 ms on the helper per promote on the target
  card). A same-tick fill landing (finding 2) is named, not pre-registered. Move 1 items 1, 3, 4 unchanged. The 1b
  limits of day 31 stand (charges release at the latch while a quarantined ticket or a detached helper may still hold
  buffers; a multi-model context's set grows per distinct plane length); a fill on a detached helper now holds
  staging too.
- Limits of this slice, stated: the tenant purge drops a Filling promote typed once its fill lands (it has no engine
  in hand to submit); a demote meeting a Filling promote waits for its fill (bounded by the 10 s deadline) before it
  submits both.

## 8. Checks, budget, cleanup

- Checks: `cargo fmt --all -- --check` clean; clippy `-D warnings` all targets on tier, engine and server clean; the
  `DOCS_RS=1 --target x86_64-unknown-linux-gnu` pass `docsrs_rc=0`; server lib `878 passed; 0 failed`, engine lib `546
  passed`, tier contracts `93 passed`; `check-flags: every runtime MEMRA_* name resolves against 'docs/FLAGS.md' (no
  grandfather list)`; `check-conflict-markers: OK`; `git diff --check` clean; `.gitattributes` (`*.log -whitespace`,
  `SUMMARY.txt -whitespace`) in `pro-single-day32/box/` and `rtx5090-day32/`; no em dashes in this lane's lines. No
  new `MEMRA_*` name.
- Budget: about 3.5 agent-hours of the one agent-day (11:29Z to about 15:00Z wall), of which about 1 h 10 min waited for
  lane B's chain.
- Cleanup: BOX3 `/root/a32.bundle` removed; `/root/wt-a` at `5ae139734` on `lane-a-day32-h2d`, clean;
  `/root/spill-receipts/a-day32/` mirrored to `pro-single-day32/box/` (binaries excluded, their sha256 kept); no process
  of mine on the box, no lock holder, no compute app at 14:50Z; nothing of other lanes touched. Local: `/tmp/wt-a-d32`
  removed at close; the 5090 carries no process of mine.
