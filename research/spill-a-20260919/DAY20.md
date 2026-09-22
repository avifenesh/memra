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
