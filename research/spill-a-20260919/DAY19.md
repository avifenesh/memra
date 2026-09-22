# WP-A day 19: rule 3 (the reader fence of an off-owner H2D), the settle-time owner wait, Move 2 pre-registered

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`, resumed from `614c53f71`. Every push today in the announced
`MEMRA_RELEASE_QUALIFICATION_MODE=development` mode (the #589 hook refuses an engine-touching range otherwise);
no qualification is claimed for any cell; every cell below is `executed-not-qualified`.

## The day-18 receipts, settled

The detached resume driver had finished (`local-driver-done` 03:13 UTC). Its receipts are committed as
`3a5c3cb86` (`data:`), and `DAY18.md`'s "Local RTX 5090" paragraph now carries the verdict table verbatim in place of
"not run": identity `ALL GREEN (teeth=0)` default ON, plain OFF and ON (default OFF NOT RUN: 30 lock-busy retries
across the two driver instances); failure `ALL GREEN` both arms; fault `ALL GREEN` (64 ok); hit `ALL GREEN (qwen)`
both arms; unit cells `8 passed`; the 9B twin refuses its cohort shape (both arms); the 27B twin `V3=FAIL` in BOTH
arms with the identical line (a constant `-410352980` B effective-free error on turns 2 to 7, `effective_free_ok=2/8`,
`evictions=1` where day 17's local run read `evictions=9` and `-> PASS`). Not a door delta (OFF fails the same way);
cause not established (no process listing of the card for the window); a repro with `nvidia-smi
--query-compute-apps` beside each boot is owed; flagged for lane B's gate and the lead.

## Merge

`origin/lane/spill-integ32-20260922` `1a351ebc6` merged `--no-ff` as `1467435f5` (clean; marker census OK). It carries
my day 18 with the lead's two revuto fixes on #627 (the cold memo names the REFUSED entry, captured before the
stale-generation `swap_remove`; a tenant purge clears the tenant's memo and releases the worker's one-tick insertion
pin, `release_promoted_pin_for_tenant`), and `origin/main` `3df055601` (#626). Pattern carried into today's
pre-registration: a state a purge can meet is released by the purge, memo and pin included, before the device purge.

## Task 1: rule 3, the reader fence (commit `b27f33a2d`)

`crates/memra-tier/src/conformance/reader_fence.rs`, unversioned beside the frozen schedules (the day-11
`recovery.rs` shape; every v1 to v1.3 schedule byte-identical, `WIRE_VERSION` 1). The rule: an H2D issued off the
destination's reader stream is published only behind a WAIT on its completion event installed on the reader's
stream; `consumer_fenced` means exactly that installed wait, never a flag and never the copy's landing.
(1) Readable: `ready_view` and `take_destination` succeed only with the producer observed complete AND the wait
installed (`consumer_fenced` with a `consumer_fence` present); a landed copy without the wait is `NotReady`.
(2) Retire: `record_consumer` before publication is `NotReady`; `retire(ticket, Some(fence))` is `Busy` until the
fence's event completes. (3) Off-owner reads: a read issued before the wait is unordered whatever the copy's state;
the engine keeps the destination bound and unpublished until the wait exists. The install may sit at submit (the
day-18 program) or at settle (the owed program); the schedule is the same, only WHEN `consumer_fenced` turns true
differs.

Schedules: `h2d_reader_fence(fixture, ReaderWaitInstall::{AtSubmit, AtSettle})` and
`h2d_reader_issued_before_its_wait_is_unordered(fixture)`, on `ReaderFenceFixture` (a concrete owner adapter with
the engine's answers and the stream hooks, the v1.3 `DeviceHandBackFixture` style). CPU bindings
(`tests/contracts/reader_fence_bindings.rs`, two streams as logs, `Completion::require(.., device=true)` as the
publication gate exactly as `DeviceOwner::ready_view` runs it):

| test | binds | outcome |
|---|---|---|
| `day19_h2d_reader_fence_at_submit_is_the_engines_day18_program` | the at-submit install (the engine's day-18 behaviour: the owner wait kept at submit for an H2D) | `ok` |
| `day19_h2d_reader_fence_at_settle_with_a_wait_on_the_reader_stream` | the at-settle install with an event wait on the reader stream, plus the forbidden early read | `ok` |
| `day19_red_arm_settle_time_flag_without_a_reader_wait_fails_the_schedule` | a settle that sets `consumer_fenced` and installs NO reader wait: the schedule must fail (`catch_unwind`), and the shape of the failure is shown (the flag publishes the destination, the next read is unordered) | `ok` (the schedule failed as required) |

`cargo test -p memra-tier --test contracts`: `71 passed; 0 failed` (68 before). Clippy `-D warnings` on
`memra-tier` clean. One assertion of my first draft was wrong against the engine and was removed before the commit:
`retire(ticket, None)` on an unpublished, landed ticket IS allowed (the abort path); the schedule notes it.

## Task 2: Move 2 pre-registration (commit `428b920eb`)

`OWNER-THREAD-OFFLOAD.md` "Move 2 pre-registration (day 19)": what a capture and a restore copy on the 27B (65
layers; the recurrent state is a fixed 157.3 MB per entry, the KV rows 29.6 KB per token from the twin gate's fit:
64 tokens 159.8 MB measured, 4096 tokens about 278 MB), where each runs today (owner stream, stream-ordered, inside
`prefill_tick` and admission), the one-program law under an event-ordered publication (a captured entry servable
only after its copy's event, the recurrent-state `clone_dtod` kept at the boundary on the owner stream; a restore's
destination readable only after its event and before the first prime chunk, by rule 3's reader fence), the
`Capturing` and `Restoring` states with what every path does on meeting them (hit, eviction, admission reclaim,
trim, demote, purge, shutdown, a second capture, a second hit, a latched tier), the failure paths, the D2D contract
in `memra-tier` terms (`TransferOp::D2d(ContiguousCopy)` on one device; ticket, producer fence, consumer fence,
receipt; the checksum term's decision rule between a device digest and an `Unwitnessed` arm) and its frozen
schedules `d2d_capture_publish` and `d2d_restore_reader_fence`, the decision cells (capture and restore stall arms in
the day-16 shape, the delayed copy stream fault, identity OFF versus ON on captured and restored entries, the
digest's price), and the slices priced (1.5, 1.5, 1 agent-days). **First slice, named: the capture on the copy
stream with an event-ordered publication.** Lane C's day-22 observation is folded in: a Move 2 capture asks the
budget before it copies.

## Task 1, the engine half: the wait moves to the settle (pre-registered before the code ran anywhere)

The schedule allows the at-settle install; the change: `CudaTransfers::submit_batch` installs the owner-stream wait
on the owner stream only (`if !off_owner`, a same-stream no-op, the day-16 program statement for statement) and
fences at submit the owner-stream program and every D2H; an off-owner H2D leaves submit unfenced. New
`CudaTransfers::install_consumer_wait(&ticket)`: for every unfenced H2D item, `owner.wait(item event)` then the
fence, idempotent, `Quarantined`/`NotReady` typed refusals, a CUDA error marks the entry unknown. The worker's
`host_kv_planes_settle_promote` calls it after the completion is observed (the `Pending` hand-back under `Poll`, the
host wait under `Block`) and before the receipt check and `ready_view`, then re-reads the completion; a refusal takes
the pre-publication abort arm (`tier H2D reader wait refused: ..`). The engine census
`copy_stream_issue_and_owner_wait_rules_are_as_stated` and the worker's frozen-restore-sequence census pin the new
order. `docs/FLAGS.md` door row gains the day-19 sentence. No new flag, no new numeric program, no new `unsafe`.

Gates that decide whether it stands, target card (BOX3, one RTX PRO 6000 Blackwell, 600 W, the 27B artifact,
`MEMRA_HOSTGATE_CACHE_MB=256`, through the collector): identity default and plain, OFF and ON; failure OFF and ON;
the fault gate; the twin gate OFF and ON; the hit gate OFF and ON (its own flock); the GPU unit cells `option_b_*`
and `option_c_*`. If the identity gate or the fault gate is red in either arm the change does not stand: it stays
`wip:` with the red receipt and the schedule stands alone. No stall cell today: the change is an ordering move
decided by the day-18 promote-arm reading; its measurement is the same-window A/B owed in the offload note. No 5090
cell today (the change is under the door on the target card class; the 5090 door gates on this tree are owed with
the lock, as on day 18).

### Target card, the result (BOX3, tree `c96d51862`, binary `bf353fc7…`; receipts `pro-single-day19/box/`)

Built on the box from the shipped bundle (`build rc=0`, `box/build.log`); the collector cell `gates` (`tools/tier-battery.py
--rig pro-single --external-lock`, `/tmp/memra-gpu.lock`, `LOCK.json` owner `collector`) ran every gate in one lock
hold; the hit gate under its own `flock` on the canonical lock; the unit cells in a second collector hold. No lock
retry was needed; nothing of mine is left running on the box. Every line verbatim, N=1 per cell, the card's regime in
the collector's `command.gpu.csv` beside each cell; `executed-not-qualified`.

| cell | arm | verdict line, verbatim | ok / fail |
|---|---|---|---|
| identity-default-off | door OFF | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` | 12 / 0 |
| identity-default-on | door ON | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` | 12 / 0 |
| identity-plain-off | `MEMRA_SERVE_SPEC=0` | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` | 12 / 0 |
| identity-plain-on | `MEMRA_SERVE_SPEC=0`, door ON | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` | 12 / 0 |
| failure-off | door OFF | `KV-HOST-SPILL FAILURE GATE: ALL GREEN` | 15 / 0 |
| failure-on | door ON | `KV-HOST-SPILL FAILURE GATE: ALL GREEN` | 15 / 0 |
| contract-fault | ON by construction | `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` | 65 / 0 (the lead's #626 floor item is the 65th) |
| twin-off | door OFF | `PREFIX-NEWEST-TURN-FITS: budget_bytes=1073741824 cohort_bytes=736755712 turns=8 cold_turns_after_1=0 cached_ok=7/7 lines_ok=8/8 evictions=9 cohort_evictions=3 self_evictions=0 refused_or_skipped=0 effective_free_ok=8/8 identity_ok=8/8 grid_ok=21/21 grid=32 off_grid_calls=0 V1=ok V2=ok V3=ok V4=ok V5=ok V6=ok -> PASS` | PASS |
| twin-on | door ON | the identical line, `-> PASS` | PASS |
| hitgate-off | door OFF | `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` | 61 / 0 |
| hitgate-on | door ON | `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` | 61 / 0 |
| unit-cell (`option_b_*`, `option_c_*`) | the copy-stream engine | `test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 794 filtered out; finished in 0.26s` | 8 / 0 |

The ON arms exercised the moved wait: the identity gates' server logs each carry one `[prefix-host] promote published
off the tick: ticket complete after 1 poll(s), 7.6ms from submission to completion (tick-top poll)` and one
`contracts door H2D receipt: .. require=ok .. published retired acknowledged`; the failure gate's ON boot two H2D
receipts through the `Block` route (the hook); the fault gate four off-tick publications and four receipts; zero
`reader wait refused` lines anywhere. The hit gate's ON arm publishes no promote (its entries stay on the device at
`MEMRA_HOSTGATE_CACHE_MB=256`; the same as day 18). The twin gate on this card reads `evictions=9` in both arms, the
day-17 shape, so the 5090's `V3=FAIL` of the day-18 record is a local-card reading, not a tree-wide one.

**Reading.** The identity and fault gates are green in both arms on the target card class, the pre-registered
condition: the settle-time reader wait STANDS under the door. Commit `c96d51862` keeps its `wip:` prefix (history is
not rewritten); this record and `OWNER-THREAD-OFFLOAD.md` are where it is promoted to "landed under the door". The
engine's `consumer_fenced` semantics for a copy-stream H2D now follow rule 3 exactly: unfenced at submit, fenced by
the installed owner-stream wait at the settle. Measurement is not claimed: the tenant-side effect (the kernels
between submit and settle no longer queue behind the copy's landing, up to about 6 ms per 160 MB on this card) is
the same-window A/B owed in the offload note; the day-18 promote arm read `at_off` already, so the expected reading
of that cell is a tail, not a median.

## Task 3: Move 2 slice 1

Not started: the budget closed at the records. Slice 1 is named and priced in `OWNER-THREAD-OFFLOAD.md`; its first
commit is the CPU schedule `d2d_capture_publish` and its bindings, before any engine code.

## Task 4: records and checks

`STATE.md` rewritten (day 19). `OWNER-THREAD-OFFLOAD.md`: the day-19 section closes Move 1's first owed item with the
verdict; the owed list is now the receipt hashes, the by-reference routes, the same-window decision cell.
`research/INDEX.md` row `spill-a-20260919/day19`. `docs/FLAGS.md` door row (day-19 sentence, in the code commit).
Checks on the final tree: `cargo fmt --check` (tier, engine, server) clean; clippy `-D warnings` on `memra-tier`
(`--all-targets`), `memra-engine` and `memra-server` (`--lib --tests`) clean; `memra-tier` contracts `71 passed`;
`memra-engine` lib `tier_transfer` `4 passed; 1 ignored`; `memra-server` lib `788 passed; 14 ignored`; `git diff
--check` clean on every source commit (the receipt commit `e3a6acd67` carries `box/unit/cargo-test.log:21: new blank
line at EOF`, a receipt's own bytes, kept); `tools/check-flags.sh` (no uncovered runtime names),
`tools/check-conflict-markers.sh` OK, `python3 tools/check-public-boundary.py check` `0 new`. The box worktree
`/root/wt-a` is at `c96d51862` on `lane-a-day19`; `/root/spill-receipts/a-day19/` mirrored to `pro-single-day19/box/`
(bins excluded); the bundle removed on both ends; `/tmp/spill-a-day19/` removed locally at close. #536 comment posted
with this receipt; the issue stays open.

## Budget

About 3.9 agent-hours against 4: the day-18 settle and the merge 0.4, rule 3's schedule and bindings 0.8, the Move 2
pre-registration 0.6, the engine and worker change with its censuses and the local battery 0.7, the target-card cells
0.8, records and the #536 comment 0.6. Blockers: none today (the box lock was free; the local 5090 was not touched).
Open: the 5090 door gates on this tree (owed with the lock), the 5090 twin27 `V3=FAIL` repro for lane B, the
same-window A/B of both Move 1 classes, the receipt hashes off the tick, the by-reference routes, Move 2 slice 1.
