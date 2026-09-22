# WP-A day 18: Move 1, the promote half: the door's H2D promote leaves the tick

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`. Start: tip `b967b8d30` (day 17; PR #622 open as the
lead's integ30). First action, three merges: `origin/main` `dc192cd95` (#621) as `e58a62d74`,
`origin/lane/spill-c-20260919` `304e8235c` (C day 21: the failure gate and the fault gate fixed on the gate
side) as `5ec390ca3`, then, on the lead's note, `origin/lane/spill-integ30-20260922` `340e8a474` (revuto on
#622: the day-17 settle arm for a pending demote missing its shell or ticket failed OPEN, dropping a submitted
ticket without retire or acknowledge; the lead's `913199b4b` makes it fail closed, `340e8a474` its clippy
follow-up) as `ccbf002ed`. Each merge conflicted on `research/INDEX.md` in the same inherited shape (a
recursive-merge marker block on one side, a row on the other; no parent carries a marker): the marker lines
dropped, every row kept, `tools/check-conflict-markers.sh` OK before each commit. Every push today in the
announced `MEMRA_RELEASE_QUALIFICATION_MODE=development` mode (the #589 hook prints `UNQUALIFIED DEVELOPMENT:
refs/heads/lane/spill-a-20260919 at <sha>; no GPU qualification claimed` and records the skip in
`.git/memra-gate-skips.log`); nothing here claims qualification, every cell below is `executed-not-qualified`.
Budget 4 agent-hours.

## Pre-registration (this section is committed before any code)

Scope: under the existing door only (`MEMRA_KV_HOST_CONTRACTS=1`, default OFF, decide-by 2026-10-05). No new
flag, no new `MEMRA_*` read, no new numeric program (no token is produced by a copy; the restore that reads
the promoted planes is the unchanged D2D on the owner stream, and the request's first step runs after it
exactly as today), `unsafe` only through the documented FFI already in `tier_transfer.rs`. The OFF arm is
untouched statement for statement. The day-17 demote half is unchanged except where a path now also meets a
`Promoting` entry (listed below).

### The contract of a promote that leaves the tick

**Streams.** Under the door the H2D of every plane of one promote is issued on the transfer engine's COPY
stream (`CudaTransfers::new_with_copy_stream`, the day-17 stream; no third stream). The worker thread still
issues everything (`check_thread` unchanged). Symmetric to the day-17 demote with one deliberate asymmetry:
an H2D destination's consumer is the OWNER stream (the D2D restore and every kernel after it read the fresh
planes), so the engine KEEPS the submit-time `owner.wait(item event)` install for an H2D and moves only the
issue stream. What that buys and costs, stated: the host no longer blocks on the copy (the request parks
instead), the DMA overlaps the kernels already queued on the owner stream, and only kernels submitted after
the promote's submit queue behind the copy's completion (bounded by the copy time, about 6 ms for a 160 MB
entry on this card by the day-15 pair). Moving that wait to the settle (an engine method that installs it
after `event_done`) is a named follow-up, decided by the stall cell below, not taken here.

**Ordering events (the H2D).**
1. Producer fence: `record_producer` records an event on the OWNER stream after the fresh destination
   planes were allocated (unchanged); `submit_batch` makes the COPY stream wait on it before the first
   `memcpy_htod`. Nothing implicit is relied on.
2. Completion: one event per item recorded on the COPY stream after its copy; the OWNER stream waits on it
   at submit (the consumer ordering, by construction: no prime, decode or restore issued after the submit
   can read a destination plane before its copy landed). `progress` computes the receipt checksum over the
   SOURCE host bytes only after `event_done`.
3. Source protection: the sources are TWINS of the entry's own pinned leases (`retain_host`, unchanged);
   the twins live in the engine until `retire_source` after every item's event was observed complete. A
   host LRU eviction, a tenant reclaim or a purge that drops the host entry meanwhile drops the entry's
   handle only: the twin keeps the allocation, and `PinnedBacking::Drop` synchronizes the tracking event the
   copy recorded on the issue stream, so no source byte is freed under a running DMA.
4. Destination protection: the fresh planes are registered leases with retained twins inside the engine
   until `take_plane` after `retire`; `take_plane` drains both streams before storage moves. The device
   entry that will hold them does not exist in the device LRU until publication.
5. Publication: the promoted entry enters the DEVICE prefix index (`PrefixCache::insert_pinned_demoting`)
   only in the completion step, on the owner thread, at a tick-top poll (or a synchronous settle where a
   path must not wait), after (1) `poll` shows every item complete, (2) `Completion::require` against each
   plane's D2H receipt passed, (3) `ready_view` per item, (4) the consumer fence was recorded on the owner
   stream and observed by a drain (at the tick top the owner stream is idle, so this drain is cheap; it is
   the day-16 sequence with the host wait on the copy replaced by the poll), (5) the source twins retired,
   the producer fence released, the ticket retired and acknowledged, (6) the fresh planes came back into
   the `PrefixEntry`, (7) the `MEMRA_KV_HOST_VERIFY` digest (when armed) and the identity lease's `require`
   passed, (8) `insert_pinned_demoting` with the insertion pin. The cache becomes usable at (8) and never
   before: a session cache is restored from it only by a request admitted after publication.

**States.** A host entry whose promote was submitted is `Promoting` from submission until publication or
failure: it stays in the HOST LRU (its handles never move), it is NOT in the device LRU, its fresh planes
are in the engine's registry, its identity lease and residency charge are held by the pending state.
Exactly one `Promoting` entry exists per worker, and never together with a `Demoting` one in flight: the
ledger's in-flight dimension is one batch, so every submit path settles whatever is pending first. The
state has two phases: `Copying` (the ticket in flight) and `Ready` (the contract settled synchronously by a
path that had no device cache in hand: the fresh planes are back in a complete `PrefixEntry`, the ledger
is free, publication waits for the next tick top). The request whose admission submitted the promote is
PARKED: it goes back on the tick's requeue exactly as a VRAM-deferred request does (`requeue.push_back`,
FIFO, never shed; the requeue becomes the next tick's queue head), keeps its admission reservation, and is
re-admitted after the tick-top poll published the entry, where it finds a DEVICE hit and takes the
unmodified restore path. The step-OOM park's shape, not its code: nothing was primed, so nothing is replayed.

**Where the decision is made.** The door's promote decision moves from the admission body (`admit`, which
consumes the request) to the admission loop immediately before the `admit(..)` call, after every defer gate
has passed (`host_promote_park_probe`). It uses the SAME predicates as the body: `reuse_on`, `prefix_on`
and the continuation-pool match are factored into shared helpers so the two sites cannot drift, and a source
census asserts both call them. The body's hook (`host_promote_prefix_hit`) keeps the day-16 synchronous
program as its fallback: under the door in production it is reached only when the probe declined, and it
then declines the same way (a memo, below); the GPU unit cells and the by-reference callers still exercise
the synchronous route.

**What every other path does when it meets a `Promoting` entry.**
- The parked request's re-admission while the copy is still pending (the copy took more than one tick):
  the probe finds the pending promote is for the SAME entry and parks it again; no second submission.
- A second request in the same tick whose best host candidate IS the pending entry: parks too (same rule).
- A request whose best host candidate is a DIFFERENT entry: settles the pending promote synchronously first
  (contract settle plus publication, since the probe has the device cache in hand), then submits its own.
- A prefix hit on the pending entry's prompt in the DEVICE cache: none exists (unpublished): the probe parks
  the request; it never primes cold while its promote is in flight.
- A demote of any route (eviction sink, admission reclaim `evict_all_demoting`, pause sweep, handoff drain):
  settles the pending promote's CONTRACT synchronously first (`Block`: a host wait on its events, then the
  planes back into the entry, the ticket retired), leaving the state `Ready`; it has no device cache in
  hand, so publication waits for the next tick top. Never two tickets in flight, never a `Capacity` refusal.
- Host LRU eviction and the tenant-share reclaim of the SOURCE entry: proceed; the twins keep the bytes
  (rule 3); the identity lease then fails its `require` at publication and the promote ends `Refused`
  with the typed line, the parked request serves cold. Stated, not special-cased.
- Tenant purge (`PurgeTenantHost`): settles the contract first (`Block`); if the pending entry belongs to
  the purged tenant's row it is DROPPED unpublished (the fresh planes back to the pool), so a revoked
  tenant's bytes never land in the device cache after the purge's receipt; another tenant's stays `Ready`.
- Admission reclaim and trim (`TrimPools`, `evict_all`): proceed; the fresh planes are allocated registry
  storage, not free pool blocks; the sources are pinned.
- Tier latched off (`disable`) before publication: the pending promote is settled and DROPPED, nothing
  published, the parked request serves cold.
- Shutdown (worker return): the unretired ticket's inputs are forgotten by `Entry::drop` (the frozen rule),
  nothing published; the process exits.
- A parked request whose client disconnected: dropped at re-admission by the existing queue sweep; the
  promote still publishes (a usable device entry for the next hit), its insertion pin released at the next
  tick top.
- Idle: while a promote is pending the idle wait is capped at 2 ms (the day-17 shape) so a box with no
  traffic still publishes.
- The insertion pin: held by the worker from publication until the NEXT tick top (one tick), so a
  `TrimPools` or an insert between publication and the parked request's re-admission cannot evict the entry
  before the request takes its own serving pin. Released unconditionally then.
- The fail-closed arm (the lead's ruling on #622, mirrored): a pending promote missing its shell or its
  ticket never publishes. With a submitted ticket and no shell the ticket is settled (`synchronize`) and its
  sources retired through the engine where reachable and the tier latches off (`Latched`); with a shell and
  no ticket it drops whole (`Failed`), the tier stays armed. No arm drops a submitted ticket. CPU test of
  the same shape as `a_pending_demote_missing_its_shell_or_ticket_fails_closed`.

**Failure paths.** A refusal at SUBMIT (an allocation, a registration, a twin, the producer fence, a partial
acceptance, the `contract-promote-presubmit` and `contract-promote-reject` faults): the day-16 unwind runs
synchronously inside the probe with the day-16 typed outcome and the SAME line text (`promote failed (..)`,
`promote refused (contracts door): ..; serving without the host entry`, `TIER DISABLED`), the request is NOT
parked, and a one-tick MEMO (`promote_cold`: pool key plus the host entry's tokens) makes the body's hook
serve it cold in this same admission instead of retrying synchronously. A failure at SETTLE (a quarantined
completion, a refused `require`, a refused `ready_view`, the `contract-promote-postpublish` and
`contract-promote-readyview` faults, a ticket that does not retire, a plane that does not come back, the
`VERIFY FAILED` digest, an identity lease that no longer holds): nothing is published, the typed outcome and
line are the day-16 ones (`ReceiptMismatch` and `VERIFY FAILED` drop the host entry; `Latched` latches the
tier off; `Refused` keeps the entry), the memo is set, and the parked request's re-admission serves cold.
The memo is cleared at the next tick top: a waiter admitted after that retries once more (bounded: each
attempt parks one tick and ends in a memo), never a loop inside one tick.

**Identity law.** The bytes a restore reads after publication are exactly the bytes a synchronous promote
would have produced: same `memcpy_htod` per plane from the same pinned source, same D2H receipt required
against the same completion checksums, same `MEMRA_KV_HOST_VERIFY` digest over the assembled entry, same
identity `require`. The parked request's restore is the unmodified device path on an entry published before
its admission. Proof: the identity gate (OFF and ON, default and plain), the fault gate and the hit gate; a
difference is a FAIL of the slice, never a tolerance.

**Lines.** `[prefix-host] promote submitted off the tick: N tokens, X MB, ticket seq=S, M items on the
contracts door's copy stream; request parked` at submission; at publication `[prefix-host] promote published
off the tick: ticket seq=S complete after P poll(s), X ms from submission to completion (..)` then the
unchanged `contracts door H2D receipt: ..` and `promote: N tokens, X MB in Y ms (model ..)` lines (`in Y ms`
now spans submission to publication). Neither new line carries `promote:` adjacent, `promote refused`, or the
`(contracts door): ` refusal marker the gates count.

### Gates (pass/fail, both cards, door OFF and ON where the gate has arms)

Local RTX 5090 first (the Qwen3.5-9B NVFP4 MTP artifact, `MEMRA_HOSTGATE_CACHE_MB=64`, `flock
/tmp/memra-5090.lock`), then the target card (BOX3, one RTX PRO 6000 Blackwell, the Qwen3.8-27B NVFP4-Q5K
MTP artifact, `MEMRA_HOSTGATE_CACHE_MB=256`, the collector's `/tmp/memra-gpu.lock`): C's fixed
`tools/kv-host-spill-identity-gate.sh` default and plain (`MEMRA_SERVE_SPEC=0`),
`tools/kv-host-spill-failure-gate.sh`, `tools/kv-host-contract-fault-gate.sh`, `tools/spec-on-cache-hit-gate.sh
qwen`, `tools/prefix-newest-turn-fits-gate.py`; each with the door ON and OFF (the fault gate is ON by
construction); the GPU unit cells `option_b_*` and `option_c_*` on the copy-stream engine. Verdict lines
verbatim. If the identity gate or the fault gate is red on either card, the slice does not stand: it stays
`wip:` with the red receipt. The twin gate on the 9B refuses its cohort shape (day 17, a gate shape fact); it
is run locally on the 27B artifact as on day 17.

### The stall cell, promote arm, pre-registered rule

Shape: `pro-single-day16/stall-cell.sh` `on` boot (`MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4 MEMRA_SERVE_SPEC=0
MEMRA_PREFIX_CACHE_MB=256 MEMRA_KV_HOST_MB=8192 MEMRA_KV_HOST_CONTRACTS=1`), `stall_cell.py --mode promote`
(the intruder alternates P_A and P_B: each hit promotes one entry and its insert demotes the other), N=5 per
arm per order, both orders, one collector lock hold, the harness unchanged (rule line fixed, `--replay`). The
task names the comparison against the day-16 receipt on this box: `stall-promote-on stall_median=162.8
arm_max=211.2`, `stall-promote-off stall_median=85.0 arm_max=133.6` (idle p50 13.4, p99 14.8). Same box and
card, a different sitting: a same-box cross-sitting reading, not a same-window A/B.

What is already known and must be said first: day 17 ran this arm too (not claimed then, recorded verbatim in
`pro-single-day17/box/stall-on/ev/promote.log`): `stall_median=86.4`, because the DEMOTE inside each promote
intruder left the tick on day 17 and that demote was most of the day-16 gap. So against day 16 the rule below
is expected to read `at_off` before this slice does anything, and the honest claim of THIS slice is the
promote's own share: the day-17 server lines put the synchronous promote at 12 ms steady (47 to 49 ms for the
first pair: copy wait, receipt hash over 160 MB of pinned source, `ready_view`, the owner drain), all inside
the intruder's restore tick, which is the tenant's stretched tick.

Claim: the tenant's stall median for a promote drops toward the OFF arm's, and below it, because the OFF
promote's `htod_u8_into` still sits on the owner stream inside the restore tick while the door's now does
not. Rule against day 16 (fixed before the run; gap = 162.8 - 85.0 = 77.8 ms):
- `at_off` if `stall_median <= 93.5` (OFF plus ten percent);
- `toward_off` if `93.5 < stall_median <= 143.4` (at least a quarter of the gap closed);
- `flat` if `143.4 < stall_median <= 172.8`;
- `worse` if `stall_median > 172.8`.
Second reading, the promote half's own contribution, against the day-17 recorded `86.4` and the OFF `85.0`
(fixed before the run): `promote_half_moved` if `stall_median <= 80.4` (six ms under day 17: half the steady
promote share left the tick); `promote_half_flat` if `80.4 < stall_median <= 92.4`; `promote_half_worse`
above. Admissibility: `STALL REPLAY: PASS`, `errors=0`, `tenant_text_identical=True`, one `promote submitted
off the tick` and one `promote published off the tick` line per promote intruder, one `contracts door H2D
receipt` line per promote (the copy still crosses the contract), every intruder's `cached_tokens` equal to
the day-16 receipt's shape (the promoted entry is hit, not primed cold), and no `demote failed`, `promote
failed`, `promote refused`, `TIER DISABLED` line in the boot's server log. The arm's `server_promote_ms` now
spans submission to publication and is read as such.

## Task 1: what landed (commit `f3e6be867`, `wip:` until the gates below are read)

**Engine, `crates/memra-engine/src/tier_transfer.rs`.** In `submit_batch` the issue stream is the copy stream
whenever one exists, for BOTH directions (day 17 moved the D2H; today the H2D); every copy under `new` keeps
the owner stream (the day-16 program, statement for statement). The owner-stream wait on the item's
completion event is installed for every H2D (both issue streams) and for an owner-stream D2H only, so a
copy-stream D2H still installs none (day 17) and a copy-stream H2D orders every later owner-stream kernel
behind its landing by construction. Producer fence (`issue.wait(producer event)`), the `memcpy_htod` /
`memcpy_dtoh` calls, the per-item event, `check_thread`, `validate`, the registry, the receipts: unchanged.
No new `unsafe`. Source census `copy_stream_issue_and_owner_wait_rules_are_as_stated` (CPU).

**Server, `crates/memra-server/src/worker.rs`.** The frozen promote route is split at step 6:
`host_kv_planes_submit_promote` (steps 1 to 5: the plane list, fresh planes from `alloc_u8`, registration
with retained twins, source twins, the producer fence, one batch; returns `PendingContractPromote { ticket,
producer, registered, planned, sources, sizes, kv_slots, fault, submitted }`) and
`host_kv_planes_settle_promote(tier, pending, wait)` (steps 6 to 11: `synchronize(&ticket)` under
`ContractWait::Block` only, `poll(&ticket)`, `Pending(..)` handed back under `Poll` while
`!completion.producer_done`, then `require` against the D2H receipts, `ready_view` per item, the consumer
fence, the owner drain, `retire_source`, `release_producer`, `retire`, `acknowledge`, the fresh planes into
`PrefixPlane`s, the unchanged `contracts door H2D receipt:` line). `host_kv_planes_from_contract` is the
synchronous wrapper (submit, then `Block`), so the hook, the GPU unit cells and the frozen-order census
(`option_c_contract_route_is_door_only_and_keeps_the_frozen_promote_order`) read the same needles in the
same order across the two halves. `device_entry_from_host_parts(engine, src, tier, route: ContractH2d)`
builds the device entry with the KV planes deferred under `OffTick` (returns the shell plus the pending
ticket) and whole under `OnTick`; `device_entry_from_host` is the `OnTick` wrapper.

The decision moved: `host_promote_park_probe(engine, px, hpx, reuse, plan, req, ctx_cap)` runs in the
admission loop immediately before `admit(..)`, after every defer gate, only when no retained admission plan
holds a pin. It uses the body's own predicates, factored into `request_has_images`, `request_reuse_on`,
`request_prefix_on` (with `ring_prefix_excluded`) and `continuation_reuse_index`, which `admit` now calls
too (a `debug_assert_eq!` ties `prefix_on` to the helper). Decision (`host_promote_probe_decision`, pure):
`NoCandidate`, `ParkAgain` (the pending promote IS this prompt's candidate), `Cold` (the memo names it), or
`Submit { settle_first }`. A ready pending promote publishes first (the probe has the device cache in hand);
a pending promote of another entry settles and publishes (`Block`, "a second promote"), then the pending
demote (`Block`, "a promote"), then the decision is taken again on the settled state. `host_promote_prepare`
(the day-16 checks: generation, class, program, identity lease, residency charge, the same lines) is shared
with the hook. A submit refusal reports through the shared `host_promote_report_failure` (the day-16 arms,
same lines and counters) and sets the memo; a whole entry without a contract route publishes at once
through the shared tail; a submitted ticket becomes `HostPrefixCache::promoting = Some(PendingPromote {
pool_key, host_toks, host_id, host_len, shell, contract, ready, tier_identity, tier_charge,
expected_digest, t0, polls, copy_ms, settled_by })` with one `promote submitted off the tick: N tokens,
X MB, ticket seq=S, M items on the contracts door's copy stream; request parked` line, and the caller does
`requeue.push_back(req); continue;` (the VRAM-defer shape: FIFO, never shed, the reservation kept).

The state machine `host_promote_settle_with(host, wait, why, settle)` takes the contract step as a closure
(CPU tests): a ready state is handed out as `Ready`; the fail-closed arm mirrors the lead's #622 ruling (a
submitted ticket without its shell is `synchronize`d and `retire_source`d through the engine where reachable
and the tier latches off; a shell without a ticket drops whole; a ready state without a shell drops whole);
`Pending` keeps the state and counts the poll; `Done` fills the shell's `kv` and `draft`, records `copy_ms`
and the settle mode, drops unpublished with the memo if the tier latched off meanwhile, else `Ready`; a
typed failure goes through `host_promote_report_failure` with the memo. `host_promote_settle_contract`
(no device cache in hand: a demote route, a tenant purge) puts a `Ready` state back;
`host_promote_settle_pending(engine, px, host, wait, why)` publishes it through `host_promote_publish`,
which runs the shared tail `host_promote_finish` (the `MEMRA_KV_HOST_VERIFY` digest with `VERIFY FAILED`
dropping the host entry by id, the identity lease's `require`, the residency charge, the recency touch by
id, `insert_pinned_demoting`, the counters, then `promote published off the tick: ticket complete after N
poll(s), X ms from submission to completion (..)` and the unchanged `promote:` line) and holds the insertion
pin in `HostPrefixCache::promoted_pin`. Settle-first sites: the hook (`host_promote_settle_pending`, `Block`,
"a promote", before `host_promote_candidate`; plus the memo check), `host_demote_prefix_ref`
(`host_promote_settle_contract`, `Block`, "a second demote"), `purge_tenant` (`Block`, "a tenant purge", then
the purged tenant's pending promote is dropped unpublished with a typed line). Run loop: after the demote
poll, the tick top releases `promoted_pin`, clears `promote_cold`, polls `host_promote_settle_pending(..,
Poll, "the tick top")`; the indefinite idle block requires `hpx.promoting.is_none()` and the timed idle wait
is capped at 2 ms while pending. Boot line: `contracts door: D2H demotes and H2D promotes ride the transfer
engine's copy stream and publish at the tick top (memra#536 Move 1)`.

**Tests.** CPU: `promoting_entry_parks_its_prompt_again_and_a_memo_serves_it_cold` (the probe decision on
every arm), `a_pending_promote_poll_keeps_the_state_and_done_reaches_ready_once` (three pending polls keep
the state, `Done` reaches `Ready` exactly once with the shell filled, a ready state is handed out without a
contract step, nothing pending answers `None`), `a_failing_promote_never_publishes_and_keeps_the_day16_typed_outcomes`
(`Refused` keeps the host entry with the memo, `ReceiptMismatch` drops it, `Latched` latches the tier off,
`Failed` counts `rejected_allocs`; the real contract step on the CPU has no engine and latches, typed),
`a_pending_promote_missing_its_shell_or_ticket_fails_closed` (the three arms), `a_tier_latched_off_under_a_promoting_entry_drops_it_unpublished`,
and the census `every_path_that_meets_a_promoting_entry_settles_it_first` (the settle-first sites, the tick-top
order pin then memo then poll, the probe before `admit(` with the requeue, the shared predicates at both sites,
`ContractH2d::OffTick` at the probe and the route selector only, publication only in `host_promote_publish` through
the shared tail after the insert). Three existing censuses moved to the new shape in the same commit (the settle-first
census's idle window, the promote-order census's caller arms now in `host_promote_report_failure`, the probe wiring
census's `host_promote_candidate` count 3). The memra-tier crate is untouched (its frozen conformance schedules
unchanged). `cargo test -p memra-server --lib`: 786 passed, 0 failed, 14 ignored; the engine census 1 passed;
`cargo clippy -p memra-engine -p memra-server --lib --tests -- -D warnings` clean (one `large_enum_variant` fixed by
boxing `Ready`); `cargo fmt --all -- --check`, `git diff --check`, `tools/check-flags.sh`, `tools/check-conflict-markers.sh`
green. `docs/FLAGS.md` door row carries the day-18 sentence (same commit). No new flag, no new `MEMRA_*` read, no
new numeric program, no new `unsafe`.

**What is NOT in this slice, stated.** The submit-time owner wait for an H2D stays (its cost to the tenant is
bounded by the copy time and is what the stall cell reads; a settle-time install is the named follow-up). The
two receipt hashes stay on the owner thread inside the tick (`progress` at the poll, `bind_tier_image` at a
demote's publication). The by-reference demote routes keep the blocking program. The hook keeps the day-16
synchronous program as its fallback.
