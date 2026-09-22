# WP-A day 20: Move 2 slice 1, the prefix snapshot's KV plane copies on the copy stream with an event-ordered publication

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`, resumed from `494c5adc6`; `origin/main` `45c4c3cd2` merged
`--no-ff` as `a06a923b7` (clean; marker census OK). Every push today in the announced
`MEMRA_RELEASE_QUALIFICATION_MODE=development` mode (the #589 hook refuses an engine-touching range otherwise);
no qualification is claimed for any cell; every cell below is `executed-not-qualified`.

## Finding before any code: the pre-registered op shape does not fit a capture's source

The day-19 pre-registration named the contract op `TransferOp::D2d(ContiguousCopy)`. `ContiguousCopy` takes owned
`DeviceLease`s on BOTH sides and the engine's registry admits only moved buffers (`register_device`: "never an
unowned raw pointer"). A capture's source is the live session cache's plane (`KvLayer.k`, `.v`), which the decoding
session keeps and appends to past the boundary; it cannot be moved into the registry, and an aliasing lease would be
`unsafe` beyond the documented FFI. So the capture class is a same-device copy from a BORROWED source span into an
OWNED, registered destination lease. Its typed op lives in the engine (`CudaTransfers::submit_d2d_capture`, the
`D2dCapture` op) and not as a `TransferOp` variant; `ContiguousCopy` stays the owned-to-owned (peer) shape. Everything
else of the pre-registration holds: the copy stream behind the producer event, the recurrent-state `clone_dtod` at
the boundary on the owner stream, the `Capturing` state, publication only after every item's completion event, the
receipt term deferred to slice 3. Recorded here and in `OWNER-THREAD-OFFLOAD.md` as a correction made before any cell
ran; flagged for the lead.

## Task 1: the slice (commits `4e57efa00`, `1770c604c`, and the worker commit below)

- `crates/memra-tier/src/conformance/d2d_capture.rs`: the schedule `d2d_capture_publish` over `D2dCaptureFixture`
  (unversioned beside the frozen schedules; every v1 to v1.3 schedule byte-identical, `WIRE_VERSION` 1): a captured
  item is not `landed` until every completion event is observed complete, a publish before the event is `NotReady`
  and a schedule failure, `retire(None)` then `acknowledge` then every destination back; slice-1 receipt clause (no
  witnessed checksum, the host-contract gate `Completion::require` refuses `Corrupt`). CPU bindings
  `tests/contracts/d2d_capture_bindings.rs`: `day20_d2d_capture_publishes_only_after_every_items_event`,
  `day20_red_arm_publish_before_the_event_fails_the_schedule` (the red arm: a publish on issue; the schedule fails),
  `day20_d2d_item_without_a_witnessed_checksum_is_refused_by_the_host_contract_gate`. `memra-tier` contracts
  `74 passed` (71 before).
- Engine: `CopyDirection::DeviceToDevice` (additive; `CopyOp::validate` refuses it), `D2dCapture`,
  `CudaTransfers::submit_d2d_capture` (requires the copy stream, else `Unsupported`; all-or-nothing validation; the
  copy stream waits on the producer event, `memcpy_dtod`, a completion event on the copy stream, fenced at submit as a
  D2H is, NO owner-stream wait anywhere), `capture_landed` (the publication predicate), `progress` leaves a capture
  item's checksum `None` (so `require`, `ready_view`, `take_destination` refuse it by construction until slice 3),
  `retire_source` has nothing to retire for a borrowed source. Census `d2d_capture_rules_are_as_stated`; GPU unit cell
  `d2d_capture_lands_on_the_copy_stream_and_publishes_only_after_its_event` (ignored without a device). Engine
  `tier_transfer` lib tests `5 passed; 2 ignored`.
- Worker (`crates/memra-server/src/worker.rs`): `HostPrefixCache::capturing` (the one `Capturing` entry),
  `capture_off_tick_disabled` (the path's latch), `PendingCapture`, `PendingContractCapture`, `CapturePlane`,
  `CaptureSettle`, `CaptureSettled`, `HostCaptureFailure::Latched`, `CaptureRoute`, `HostCaptureOutcome`;
  `prefix_capture_off_tick` (the route under the door, called by `prefix_insert_from_session` after the budget
  preflight and before the tick program: the recurrent clones on the owner stream, fresh planes registered with
  retained twins, the producer fence, one batch; a pending capture settles first; TP, latent, SWA and off-boundary
  caches keep the tick program; refusals are typed and create no entry), `host_kv_planes_settle_capture`,
  `host_capture_settle_with` (the CPU-testable half with the fail-closed arm), `host_capture_settle_contract`,
  `host_capture_settle_pending`, `host_capture_publish` (through the ordinary `insert_demoting`),
  `host_capture_latch`, `host_capture_drain_at_shutdown`. Call sites: the tick top after the promote poll; both idle
  waits; `HostPrefixCache::purge_tenant` (settle, drop the revoked tenant's); the admission reclaim; the three device
  trims; the run loop's exit. `host_tier_context`: the ledger's in-flight dimension gains the capture term
  (`2 x (2 x max layers + 2)`). The fanout leader and the pause sweep keep the tick program. `docs/FLAGS.md` door row
  day-20 sentence (same commit). CPU tests: `a_pending_capture_poll_keeps_the_state_and_done_reaches_ready_once`,
  `a_pending_capture_missing_its_shell_or_ticket_fails_closed`,
  `a_latched_capture_settle_drops_the_entry_and_takes_the_tick_program`,
  `host_purge_drops_the_purged_tenants_capturing_entry_and_keeps_anothers`,
  `every_path_that_meets_a_capturing_entry_settles_or_ignores_it_as_stated` (source census); the day-17 census's
  idle-cap literal updated to the day-20 statement. Server lib suite: 798 passed (796 + 2 after the two fixes),
  14 ignored. No new flag, no new numeric program, no new `unsafe`.

## Task 2, pre-registration (this section is committed before the run)

Target card (BOX3, one RTX PRO 6000 Blackwell, 600 W, the 27B artifact, `MEMRA_HOSTGATE_CACHE_MB=256`), through the
collector, scripts `pro-single-day20/`: identity default and plain, OFF and ON; failure OFF and ON; the fault gate;
the twin gate OFF and ON; the hit gate OFF and ON (its own flock); the GPU unit cells `option_b_*`, `option_c_*` and
the engine's `d2d_capture_*`. If the identity gate or the fault gate is red in either arm the slice does not stand:
it stays `wip:` with the red receipt.

**The capture stall cell (memra#536 Move 2 cell (i) in the day-16 shape).** `stall-cell.sh off|on` boots the
day-16 server with the prefix cache ON at 1024 MB (a 4096-token entry is about 278 MB on the 27B) and the host tier at
8 GiB, `MEMRA_SERVE_SPEC=0`, door OFF or ON; the harness `stall_cell.py --mode capture` (a NEW arm; the three day-16
arms are byte-for-byte the day-16 program): the tenant streams 160 tokens, the intruder fires at the tenant's 24th
token with a fresh about-4096-token prompt and `max_tokens=1` (its prefill-done grid seed CAPTURES), and after the
tenant's stream ends the same prompt is re-posted untimed and its `cached_tokens` recorded. N=5 per arm per order, both
orders inside each boot (idle/arm, arm/idle); the two door arms run OFF, ON, ON, OFF in one collector lock hold
(`stall-both.sh`). Rules, fixed before the run (ms, the harness's `stall_median`):

- Admissibility: `errors=0` and `tenant_text_identical=True` on every receipt; every intruder's re-post reads
  `repost_cached_tokens >= repost_prompt_tokens - 64` (the grid-aligned seed published and hit). A receipt that fails
  admissibility decides nothing and is reported as such.
- C1, ON against OFF, same hold, per pass: `on_under_off` if `stall_median(ON) <= stall_median(OFF) - 6.0`; `flat` if
  within 6.0 either way; `on_over_off` above. The six-ms threshold is the day-18 promote arm's.
- C2, the copy's own duration: the ON arm's `server_capture_ms` (the `capture published off the tick` lines, ms from
  submission to completion) is reported as a list with its median; no threshold (a reading, not a decision).
- What the cell can and cannot say: the intruder's PRIME is compute on the tick by design (the day-16 prime arm read
  `stall_median=301.5`), so both arms carry it and only the capture's own on-tick share can differ between them. On
  this card a 278 MB D2D is well under a millisecond, so the expected reading is `flat`: the cell then records that the
  capture's share on the target card is under the cell's resolution, which is a finding, not a failure. Nothing here
  decides the door.

## Task 2, the target card: gates (BOX3, tree `c7a6d3a5b`, binary `263fe777…`; receipts `pro-single-day20/box/`)

Built on the box from the shipped bundles (`build rc=0`, `box/build.log`); the collector cell `gates`
(`tools/tier-battery.py --rig pro-single --external-lock`, `/tmp/memra-gpu.lock`, `LOCK.json` owner `collector`) ran
every gate in one lock hold; the hit gate under its own `flock` on the canonical lock; the unit cells in a second
collector hold; the stall cell in a third. No lock retry was needed. Every line verbatim, N=1 per gate cell, the
card's regime in the collector's `command.gpu.csv` beside each cell; `executed-not-qualified`.

| cell | arm | verdict line, verbatim |
|---|---|---|
| identity-default-off | door OFF | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` |
| identity-default-on | door ON | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` |
| identity-plain-off | `MEMRA_SERVE_SPEC=0` | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` |
| identity-plain-on | `MEMRA_SERVE_SPEC=0`, door ON | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` |
| failure-off | door OFF | `KV-HOST-SPILL FAILURE GATE: ALL GREEN` |
| failure-on | door ON | `KV-HOST-SPILL FAILURE GATE: ALL GREEN` |
| contract-fault | ON by construction | `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` |
| twin-off | door OFF | `PREFIX-NEWEST-TURN-FITS: budget_bytes=1073741824 cohort_bytes=736755712 turns=8 cold_turns_after_1=0 cached_ok=7/7 lines_ok=8/8 evictions=9 cohort_evictions=3 self_evictions=0 refused_or_skipped=0 effective_free_ok=8/8 identity_ok=8/8 grid_ok=21/21 grid=32 off_grid_calls=0 V1=ok V2=ok V3=ok V4=ok V5=ok V6=ok -> PASS` |
| twin-on | door ON | the identical line, `-> PASS` |
| hitgate-off | door OFF | `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` |
| hitgate-on | door ON | `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` |
| unit-cell (`option_b_*`, `option_c_*`) | the copy-stream engine | `test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 804 filtered out; finished in 0.27s` |
| unit-cell (engine `d2d_capture_*`) | the capture class on the card | `test tier_transfer::tests::d2d_capture_lands_on_the_copy_stream_and_publishes_only_after_its_event ... ok`, `test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 553 filtered out; finished in 0.33s` |

**Where the route engaged.** The plain ON arm's identity server log carries two captures, verbatim:
`[prefix-cache] capture submitted off the tick (seed): 64 tokens, 32 planes (158.8MB) on the contracts door's copy
stream; recurrent state cloned at the boundary on the owner stream` then `[prefix-cache] capture published off the
tick (seed): 64 tokens complete after 1 poll(s), 0.2ms from submission to completion, 0.2ms to publication (tick-top
poll)` (the second pair reads `75.5ms`: the figure is submission to the OBSERVING poll, the same semantics as Move 1's
promote line; the remainder of the prompt primed between the seed and the next tick top). The 158.8 MB is the entry's
byte count, of which the recurrent state (cloned on the owner stream) is about 157 MB and the 32 KV planes (16 KV
layers, K and V) about 1.9 MB: on this model the copy stream carries the rows and the owner stream keeps the state,
exactly the split the pre-registration required. The default (spec) ON arm and the failure ON arm show zero captures
through the route: their publishes are spec-boundary captures (`prefix_insert_from_spec_boundary`, the MTP draft
plane) and keep the tick program; the route covers `prefix_insert_from_session` (the seed and the LCP split) only.
Zero `capture refused`, `capture dropped`, `CAPTURE OFF-TICK DISABLED` or `TIER DISABLED` lines anywhere under
`gates/`.

**Reading.** The identity and fault gates are green in both arms on the target card class, the pre-registered
condition: the slice STANDS under the door. Commit `1c3cbb0dc` keeps its `wip:` prefix (history is not rewritten);
this record and `OWNER-THREAD-OFFLOAD.md` are where it is promoted to "landed under the door".

## Task 2, the target card: the capture stall cell (pre-registered above; run after the gates, one collector hold)

`stall-both.sh` under the collector (`box/stall-both/`, `stall-both.collector.log`): OFF pass 1, ON pass 1, ON pass 2,
OFF pass 2 (`box/stall-capture-{off,on}/ev.pass{1,2}/`), each boot's harness interleaving idle/arm in both orders,
N=5 per arm per order, `MEMRA_SERVE_SPEC=0`, cache 1024 MB, host tier 8 GiB. The intruder prompt is what the tokenizer
made of the day-16 prime arm's text: 5120 to 5123 tokens (the pre-registration wrote "about 4096"; the recorded
figure stands), so the seed publishes at the grid boundary 5088. Rule lines, verbatim (`receipt.json` `rule_line`):

`STALL rule cell=stall-capture-off arm=capture n_per_order=5 pooled=10 idle_runs=10 idle_p50=13.5 idle_p95=14.7 idle_p99=14.8 idle_max=15.0 arm_runs=10 arm_p50=13.4 arm_p95=14.9 arm_p99=295.9 arm_max=373.1 stall_median=355.4 stall_min=283.6 stall_max=359.7 server_demote_ms=[72.3, 75.8, 73.8, 72.0, 71.0, 70.2, 69.9, 69.7] server_promote_ms=[] intruder_prompt_tokens=[5123, 5122, 5123, 5121, 5123, 5122, 5123, 5120, 5122, 5122] tenant_text_identical=True errors=0` (pass 1)

`STALL rule cell=stall-capture-on arm=capture n_per_order=5 pooled=10 idle_runs=10 idle_p50=13.4 idle_p95=14.8 idle_p99=14.9 idle_max=15.0 arm_runs=10 arm_p50=13.5 arm_p95=14.9 arm_p99=297.2 arm_max=370.9 stall_median=355.4 stall_min=283.8 stall_max=357.4 server_demote_ms=[297.7, 300.3, 299.9, 299.5, 299.8, 300.7, 299.9, 299.5] server_promote_ms=[] intruder_prompt_tokens=[5123, 5122, 5123, 5121, 5123, 5122, 5123, 5120, 5122, 5122] tenant_text_identical=True errors=0` (pass 1)

Pass 2: OFF `stall_median=353.9` (`stall_min=283.7 stall_max=356.2`, `server_demote_ms=[68.1, 70.9, 70.8, 70.5, 70.2,
69.5, 69.8, 69.2]`), ON `stall_median=354.0` (`stall_min=283.9 stall_max=356.4`, `server_demote_ms=[296.7, 299.8,
298.8, 298.4, 298.2, 299.4, 299.4, 298.1]`); full lines in the receipts.

Admissibility: `errors=0` and `tenant_text_identical=True` on all four; every intruder's re-post read
`repost_cached_tokens=5088` against `repost_prompt_tokens` 5120 to 5123 (within 64): the seed published and hit in
both arms, every run. The ON arm's `server_capture_ms` (submission to the observing poll): 10 lines per boot,
`[14.5, 14.5, 228.1, 227.9, 228.0, 227.6, 227.8, 229.4, 227.9, 228.3]` (pass 1, median 227.9) and `[14.5, 14.5,
228.0, 227.9, 228.0, 227.7, 227.9, 229.2, 228.1, 227.8]` (pass 2, median 227.9): the two runs before the cache fills
observe the copy at the tick top 14.5 ms after submission; once the insert evicts, the observing poll comes 228 ms
later (the off-tick demote's submission still hashes 160 MB of pinned memory twice on the tick, the Move 1 owed item).

**Verdict, by the pre-registered rules.** C1 pass 1: `flat` (ON 355.4 against OFF 355.4, delta 0.0, bound 6.0).
C1 pass 2: `flat` (ON 354.0 against OFF 353.9, delta +0.1). C2: ON `server_capture_ms` median 227.9 both passes
(list above), a reading. As pre-registered: the intruder's prime is on the tick in both arms and dominates the stall;
the capture's own on-tick share on this card is under the cell's resolution. The cell decides nothing about the door.
Same-window, one lock hold, N=5 per arm per order, both orders; `executed-not-qualified`.

## Task 3: records and checks

`STATE.md` rewritten (day 20). `OWNER-THREAD-OFFLOAD.md`: the day-20 section (the ownership correction, what landed,
what Move 2 still owes: slice 2, slice 3, a capture-isolating cell). `research/INDEX.md` row `spill-a-20260919/day20`.
`docs/FLAGS.md` door row day-20 sentence (in the worker commit). Checks on the final tree: `cargo fmt --check` clean on
the three crates; clippy `-D warnings` clean on `memra-tier` (`--all-targets`), `memra-engine` and `memra-server`
(`--lib --tests`); `memra-tier` contracts `74 passed`; `memra-engine` lib `tier_transfer` `5 passed; 2 ignored`;
`memra-server` lib `798 passed; 14 ignored`; `git diff --check` clean on every source commit (the merge of
`origin/main` carries one upstream raw log's trailing whitespace, not mine); `tools/check-flags.sh` (no uncovered
runtime names); `tools/check-conflict-markers.sh` OK; `python3 tools/check-public-boundary.py check` `0 new`. The
box worktree `/root/wt-a` is at `c7a6d3a5b` on `lane-a-day20`; `/root/spill-receipts/a-day20/` mirrored to
`pro-single-day20/box/` (bins excluded); both bundles removed on both ends; local `/tmp` scratch removed. Not run
today: the local RTX 5090 door gates (the box cells and the records filled the budget; owed with the lock, as on days
18 and 19). #536 comment posted with this receipt; the issue stays open.

## Budget

About 4.0 agent-hours against 4: reading and the merge 0.5, the tier schedule and bindings 0.4, the engine's capture
class 0.5, the worker's `Capturing` entry with its tests 1.1, the harness arm, box scripts and the ship 0.4, the
target-card cells 0.6, records and the comment 0.5. Blockers: none (the box lock was free throughout; the 5090 was not
touched). Open: the 5090 door gates on this tree, the spec-boundary capture (`prefix_insert_from_spec_boundary`) still
on the tick, Move 2 slices 2 and 3, the Move 1 owed items (receipt hashes, by-reference routes, same-window A/B).
