# WP-A day 17: Move 1, first slice: the door's D2H demote leaves the tick

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`. Start: tip `a71bd8db9` (day 16; PR #618 open, the
lead's integ26 carries it), merged `origin/main` `e2e9e294a` (#616) as `71cdb1383` (one INDEX.md hunk:
my day-16 row against an empty side, diff3 base markers dropped, `tools/check-conflict-markers.sh` OK,
ruling 27). Every push today in the announced `MEMRA_RELEASE_QUALIFICATION_MODE=development` mode (the
hook prints `UNQUALIFIED DEVELOPMENT: refs/heads/lane/spill-a-20260919 at <sha>; no GPU qualification
claimed` and records the skip in `.git/memra-gate-skips.log`); nothing here claims qualification, every
cell below is `executed-not-qualified`. Budget 4 agent-hours.

## Pre-registration (this section is committed before any code)

Scope: under the existing door only (`MEMRA_KV_HOST_CONTRACTS=1`, default OFF, decide-by 2026-10-05). No
new flag, no new `MEMRA_*` read, no new numeric program (no token is produced by a copy), `unsafe` only
through the documented FFI already in `tier_transfer.rs`. The promote (H2D, Option C) keeps the day-16
program today; its turn is stated at the end. The OFF arm is untouched statement for statement.

### The contract of a demote that leaves the tick

**Streams.** `CudaTransfers` gains a second CUDA stream, the COPY stream, created from the owner
context on the CUDA owner thread at construction (`CudaTransfers::new_with_copy_stream`); the worker
thread still issues everything (`check_thread` unchanged; no second host thread, the census's context
ownership rule is untouched). Under the door the D2H of every plane of one demote is issued on the copy
stream; the H2D promote stays on the owner stream (unchanged submit-time `owner.wait(event)` install).

**Ordering events (the D2H).**
1. Producer fence: `record_producer` records an event on the OWNER stream after the planes were last
   written (unchanged). `submit_batch` makes the COPY stream wait on that event (`copy.wait(producer)`)
   before the first `memcpy_dtoh`. cudarc's context is not in multi-stream mode here, so this explicit
   wait is the ONLY producer ordering; nothing implicit is relied on.
2. Completion: one event per item recorded on the COPY stream after its copy. The owner stream is NOT
   made to wait on it at submit (that wait, correct for an H2D whose consumer is the owner stream,
   would put the tick's kernels behind the copy and re-serialize exactly what this slice removes; a
   D2H destination's consumer is the host, whose wait is `event_done` in `progress`, which computes the
   receipt checksum only after the event completed).
3. Source protection: the source planes leave the entry (`Option::take`) and live in the engine's
   registry until `retire_source` and `take_plane`, which require `producer_done` (every item's event
   observed complete) and which synchronize the copy stream as well as the owner stream before a plane
   returns to the pool. So no source byte is freed, returned to the pool or rewritten before its copy's
   event completed; a failure that keeps the completion unknown keeps the planes in the engine
   (`SourceQuarantined`, the frozen rule), never in the pool.
4. Destination protection: the pinned destinations are the engine's `CudaPinnedLease`s inside the
   ticket until `take_destination`, which runs only after `poll` reported `producer_done` and `require`
   passed; `PinnedBacking::as_slice`/`Drop` synchronize the lease's own tracking event, so a pinned
   buffer is neither read for its checksum nor freed under a running DMA.
5. Publication: the host entry enters `HostPrefixCache::entries` (the prefix index the hit path reads)
   only in the completion step, on the owner thread, at a tick-top poll, after (1) `poll` shows every
   item complete, (2) the completion `require`s against the receipt, (3) the planes came back, (4) the
   ticket retired against an observed consumer fence and was acknowledged, (5) `bind_tier_image`
   checked the bundle checksum against each plane's receipt, exactly the day-16 sequence with the host
   wait replaced by the poll.

**States.** A device entry evicted by the capacity loop into the door's demote is `Demoting` from
submission until publication or failure: it is out of the device LRU (evicted), not in the host LRU
(unpublished), its KV planes are in the engine's registry, its f32 planes and metadata are already on
the host (copied synchronously at submission, as today). Exactly one `Demoting` entry exists per worker
(the ledger's in-flight dimension is one batch). `Promoting` is named but NOT introduced today (the
promote is synchronous, the day-16 program).

**What every other path does when it meets a `Demoting` entry.**
- Prefix hit on the same prompt: device miss (evicted), host miss (unpublished): a COLD PRIME, never a
  wait on the copy and never a partial restore.
- Host promote of ANY entry (`host_promote_prefix_hit`): before its own submission the hit path
  SETTLES the pending demote synchronously (a host wait on its events, then the same publication code):
  the ledger's one-batch in-flight dimension is never raced, and the promoted request pays at most the
  remaining copy time instead of a refusal.
- A second demote (eviction sink, admission reclaim `evict_all_demoting`, pause sweep): SETTLES the
  pending one synchronously first, then runs; never two in flight, never a demotable entry dropped to a
  `Capacity` refusal.
- Host LRU eviction and the tenant-share reclaim: cannot see it (not in `entries`); the pending
  entry's pinned bytes are charged on the ledger at `alloc_host`, its pageable residency charge is taken
  at submission, so the ledger is never over-admitted; the host budget and the tenant share are
  re-checked at publication by `insert` and `reclaim_tenant_share`, as today, and a refusal there is a
  typed drop (`skip demote`, `evaporates`), nothing published.
- Admission reclaim and trim (`TrimPools`, `evict_all`): proceed; the pending entry is untouched (its
  device planes are allocated registry storage, not free pool blocks; its destinations are pinned).
- Tenant purge (`PurgeTenantHost`): settles the pending demote first, then purges, so a revoked
  tenant's bytes cannot land after the purge's receipt.
- Handoff export: does not see the pending entry (stated, not changed).
- Tier latched off (`disable`) before publication: the pending demote is settled and DROPPED, nothing
  published, planes back to the pool.
- Shutdown (worker return): the unretired ticket's inputs are forgotten by `Entry::drop` (the frozen
  rule), nothing published; the process exits.
- Idle: while a demote is pending the idle wait is capped at 2 ms so the poll runs on a box with no
  traffic (the same shape as the handoff drip).

**Failure paths.** A copy that fails after `Demoting` was entered (a quarantined submission, an event
that reports an error at poll, a refused `take_destination` or `require`, a plane that does not come
back, a ticket that does not retire): the entry is NEVER published, the typed outcome is the day-15 one
(`Refused` with the entry whole, `SourceQuarantined` with the planes kept by the engine and the tier
latched off, `TicketLeaked` with the tier latched off), one typed `[prefix-host] demote failed (..)` line,
and the request that next asks for that prefix primes cold. The eviction sink's source entry was already
evicted, so "the source stays authoritative" means: no half copy exists anywhere, the device bytes are
dropped exactly as a failed synchronous demote drops them today.

**Identity law.** The bytes a reader sees after publication are exactly the bytes a synchronous copy
would have produced: same `memcpy_dtoh` per plane, same source bytes (producer fence), same receipt
(`Completion` checksum per item, checked by `bind_tier_image` against the bundle checksum), same
`MEMRA_KV_HOST_VERIFY` digest at promote. Proof: the identity gate (OFF and ON, default and plain) and
the fault gate; a difference is a FAIL of the slice, never a tolerance.

### Gates (pass/fail, both cards, door OFF and ON where the gate has arms)

Local RTX 5090 first (Qwen3.5-9B NVFP4 MTP artifact, `flock /tmp/memra-5090.lock`), then the target
card (BOX3, one RTX PRO 6000 Blackwell, Qwen3.8-27B NVFP4-Q5K MTP artifact, the collector's
`/tmp/memra-gpu.lock`): `tools/kv-host-spill-identity-gate.sh` default and plain (`MEMRA_SERVE_SPEC=0`),
`tools/kv-host-spill-failure-gate.sh`, `tools/kv-host-contract-fault-gate.sh`,
`tools/spec-on-cache-hit-gate.sh qwen`, `tools/prefix-newest-turn-fits-gate.py`; each with the door ON
and OFF. Verdict lines verbatim. If the identity gate or the fault gate is red on either card, the slice
does not stand: it stays `wip:` with the red receipt.

### The stall cell (demote arm only), pre-registered rule

Shape: `pro-single-day16/stall-cell.sh` `on` boot (`MEMRA_PREFIX_CACHE_MB=256 MEMRA_KV_HOST_MB=8192
MEMRA_KV_HOST_CONTRACTS=1 MEMRA_SERVE_SPEC=0`), `stall_cell.py --mode demote`, N=5 per arm per order,
both orders, one collector lock hold, the day-16 harness unchanged (rule line fixed, `--replay`). The
comparison is against the day-16 receipt on the same box (`stall-demote-on stall_median=193.5`,
`stall-demote-off stall_median=117.5`, idle p50 13.5, idle p99 14.9); the task names this comparison and
it is the same box and card, but it is a different sitting, so the verdict is stated as a same-box
cross-sitting reading, not a same-window A/B.

Claim: the tenant's stall median for a demote drops toward the OFF arm's. What can and cannot move: the
copy wait (the day-16 census measured 42 ms of D2H inside the OFF tick and 118 ms of door D2H inside the
ON tick, of which about 76 ms is the two receipt hashes over 160 MB of cacheable pinned memory, lane C's
day-18 reading) leaves the tick; the two hashes do NOT (the completion checksum is computed by `progress`
on the owner thread at the poll, and `bind_tier_image`'s bundle checksum at publication), so the best
this slice can do is about the copy time, and the hashes now land on the ticks where the items complete.

Rule (fixed before the run; gap = 193.5 - 117.5 = 76.0 ms):
- `at_off` if `stall_median <= 129.3` (OFF plus ten percent);
- `toward_off` if `129.3 < stall_median <= 174.5` (at least a quarter of the gap closed);
- `flat` if `174.5 < stall_median <= 203.5` (within ten ms above the day-16 ON median);
- `worse` if `stall_median > 203.5`.
Admissibility: `STALL REPLAY: PASS`, `errors=0`, `tenant_text_identical=True`, one `[prefix-host]
demote:` line per intruder run (the same count as day 16), and one `contracts door D2H receipt` line per
demote (the copy still crosses the contract). The server's `demote:` `in X ms` now spans submission to
publication across ticks and is read as such; the new `demote submitted off the tick` and `published off
the tick` lines give the tick count the copy took.

## Task 1: what landed (commit `ba10a1252`, `wip:` until the gates below are read)

**Engine, `crates/memra-engine/src/tier_transfer.rs`.** `CudaTransfers` gains `copy: Option<Arc<CudaStream>>`;
`new` keeps `None` (the day-16 program, statement for statement); `new_with_copy_stream(owner, governor)`
creates the second stream from the owner context on the owner thread (a creation failure is a construction
refusal, never a silent owner-stream fall back); `copy_stream()` reads it. In `submit_batch` each item picks
its issue stream: a D2H takes the copy stream when one exists, an H2D and every copy under `new` take the
owner stream. On the issue stream: `wait(producer event)`, the `memcpy_dtoh`/`memcpy_htod`, `record_event`.
The submit-time `owner.wait(item event)` is installed only when the issue stream IS the owner stream: for a
copy-stream D2H there is no owner wait (the destination's consumer is the host, whose wait is `event_done`
in `progress` before the checksum; an owner wait would queue the tick's kernels behind the copy).
`release_device` and `take_plane` drain the copy stream as well as the owner stream before storage returns
to the pool. `check_thread`, `validate` (source planes on the owner stream), the registry, the fences and the
receipt are untouched. No new `unsafe`.

**Server, `crates/memra-server/src/worker.rs`.** The frozen demote route is split at step 6:
`host_kv_planes_submit_contract` (steps 1 to 5: the plane list, pinned destinations, the admission probe,
the planes into the registry with retained twins, the producer fence, one batch; returns
`PendingContractDemote { ticket, producer, registered, planned, sizes, fault, submitted }`) and
`host_kv_planes_settle_contract(tier, dead, pending, wait)` (steps 6 to 9: `synchronize(&ticket)` under
`ContractWait::Block` only, `poll(&ticket)`, `Pending(..)` handed back under `Poll` while
`!completion.producer_done`, then `take_destination`, `require`, `record_consumer`, `retire_source`, the
planes back, `release_producer`, `retire`, `acknowledge`, the `HostPlane`s with their receipts and the
unchanged `contracts door D2H receipt:` line). `host_kv_planes_through_contract` is now the synchronous
wrapper (submit, then `Block`), so the by-reference callers keep the day-16 program and the frozen-order
census (`option_b_contract_route_is_door_only_and_keeps_the_frozen_demote_order`) reads the same needles in
the same order across the two halves. `host_entry_from_device` takes `route: ContractD2h` and returns
`HostImage::Whole(entry)` or `HostImage::Demoting(entry, pending)` (the f32 planes, tokens, logits and the
residency charge are built at submission as before; `kv` and `draft` are empty until the settle).
`host_demote_prefix_ref` takes the route, settles a pending demote first (`Block`, "a second demote"), and
on `Demoting` stores `HostPrefixCache::demoting = Some(PendingDemote { dead, image, contract, host_bytes,
t0, polls })` with one `demote submitted off the tick: .. on the contracts door's copy stream` line (its first wording carried `(contracts door): `, the door's refusal marker, and the fault gate's `no_extra_refusal` counted it: the 5090 first attempt below); the publication tail (bind, reclaim, insert,
the `demote:` line) is `host_demote_publish`, shared by both routes. `host_demote_settle_pending(host, wait,
why)` is the state machine (`host_demote_settle_with` takes the contract step as a closure for the CPU
tests): `Pending` keeps the state and counts the poll; `Done` applies the `flip-demote` fault, drops the
entry unpublished if the tier latched off meanwhile, prints `demote published off the tick: ticket seq=..
complete after N poll(s), X ms from submission to completion (..)`, then publishes; a typed failure
publishes nothing with the day-15 outcomes (`Refused` drops whole, `SourceQuarantined` and `TicketLeaked`
latch the tier off). The eviction sink `host_demote_prefix_entry` is the only `OffTick` caller and attaches
the evicted shell to the pending entry in the same step. Settle-first sites: the hook, `host_promote_prefix_hit`
(before `host_promote_candidate`), `HostPrefixCache::purge_tenant`. The run loop's first statement is the
`Poll`; the indefinite idle block requires `hpx.demoting.is_none()` and the timed idle wait is capped at
2 ms while pending. Boot: `new_with_copy_stream` plus one receipt line. `HostDemoteOutcome::Demoting` added
(the pause sweep reads it as unpublished; unreachable on its route).

**Tests.** CPU: `demoting_entry_is_a_miss_and_a_pending_poll_publishes_nothing` (a hit on the Demoting
prompt is `None`; three pending polls keep the state, publish nothing, count; nothing pending answers
`None`), `completion_reaches_publication_exactly_once_and_only_after_done` (`Done` reaches
`host_demote_publish`, whose CPU refusal is bind's own `surface is not qualified`; the state is consumed;
nothing resident; the tier stays armed), `a_failing_copy_never_publishes_and_keeps_the_day15_typed_outcomes`
(`Refused` drops whole with the tier on; `SourceQuarantined` and `TicketLeaked` latch it off; nothing
published in any arm), `a_tier_latched_off_under_a_demoting_entry_drops_it_unpublished`, and the source
census `every_path_that_meets_a_demoting_entry_settles_it_first` (the three settle-first sites and their
order, the poll as the loop's first statement, the idle gate, `OffTick` at the sink and the selector only,
publication only after `Done` in the driver). Census updates in the same commit: the hook census spans the
hook plus `host_demote_publish`, its `host_entry_from_device(.., route)` and `Ok(HostImage::Whole(mut e))`
needles; the GPU unit cells build the copy-stream engine and pass `ContractD2h::OnTick`. The memra-tier
crate is untouched (its frozen conformance schedules unchanged). `cargo test -p memra-server --lib`: 765
passed, 0 failed, 14 ignored; `cargo clippy -p memra-engine -p memra-server --lib --tests -- -D warnings`
clean; `cargo fmt --all -- --check`, `git diff --check`, `tools/check-flags.sh`, `tools/check-conflict-markers.sh`
green. `docs/FLAGS.md` door row carries the day-17 sentence (same commit).

**What is NOT in this slice, stated.** The promote stays synchronous on the owner stream (`Promoting` is
named in the contract, not introduced). The two receipt hashes (`progress`'s completion checksum, computed
at the poll on the items that completed since the last poll, and `bind_tier_image`'s bundle checksum at
publication) stay on the owner thread inside the tick. The by-reference routes (admission reclaim flush,
pause sweep, handoff) keep the blocking program. The handoff export does not see a `Demoting` entry.

## Task 2: gates on both cards, then the stall cell (every cell `executed-not-qualified`)

**Local RTX 5090 Laptop GPU** (`rtx5090-day17/`, the Qwen3.5-9B NVFP4 MTP artifact, `MEMRA_HOSTGATE_CACHE_MB=64`
because the 9B's 64-token entry is 53.8 MB, each gate taking `/tmp/memra-5090.lock` itself with `flock -n`, the
driver retrying a busy lock 15 x 120 s and never signalling the holder; other lanes held the card for part of the
sitting). **Attempt 1** (`attempt1-line-collision/`, binary `520b88a5…`, tree `ba10a1252`): the four identity
gates `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` (12 `ok` each), the failure gate `KV-HOST-SPILL FAILURE
GATE: 1 FAILURE(S)` in both arms (`FAIL: pool-full refusal is LOUD and named`, the pre-existing day-13 line, 13
`ok`), the hit gate `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` in both arms, and the contract fault gate
`KV-HOST-CONTRACT-FAULT GATE: 10 FAILURE(S)`: six cells `FAIL: <cell>: no host-tier refusal line beyond the
injected one`, whose `no_extra_refusal` counts every `[prefix-host] .*\(contracts door\): ` line (the door's
refusal shape) and my submission line read `demote submitted off the tick (contracts door): ..`. The line was
reworded (`fc46e230d`), not the gate. Attempt 1's server logs are the first record of the mechanism on this card:
`demote submitted off the tick (contracts door): 64 tokens, 54.8MB, ticket seq=1, 18 items on the copy stream`,
`contracts door D2H receipt: .. items=18 (8 KV planes, draft) complete=18 require=ok .. retired acknowledged`,
`demote published off the tick: ticket seq=1 complete after 1 poll(s), 121.2ms from submission to completion
(tick-top poll)`, `demote: 64 tokens, 54.8MB in 145.6ms`. The attempt was stopped by me while its twin gate waited
on a busy lock (my driver, my processes), so twin and unit cells have no attempt-1 receipt.

**Attempt 2** (binary `3a92efab…`, tree `fc46e230d`), verbatim per arm:

| gate | door OFF | door ON |
|---|---|---|
| `kv-host-spill-identity-gate.sh` default | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` (12 ok) | the same line (12 ok); server: `demote published off the tick: ticket seq=1 complete after 1 poll(s), 101.8ms from submission to completion (tick-top poll)` |
| `kv-host-spill-identity-gate.sh` plain (`MEMRA_SERVE_SPEC=0`) | `ALL GREEN (teeth=0)` (12 ok) | `ALL GREEN (teeth=0)` (12 ok) |
| `kv-host-spill-failure-gate.sh` | `KV-HOST-SPILL FAILURE GATE: 1 FAILURE(S)` (`FAIL: pool-full refusal is LOUD and named`; `ok: the promote caught it: VERIFY FAILED, loud and named`; 13 ok) | the same line, 13 ok |
| `kv-host-contract-fault-gate.sh` (ON by construction) | n/a | `KV-HOST-CONTRACT-FAULT GATE: 5 FAILURE(S)`, 57 ok: presubmit, postpublish, promote-presubmit, promote-postpublish, promote-readyview every clause `ok`; the five FAILs are the `promote-reject` cell alone, whose expected line is fixed as `tier H2D batch partially refused: 1 of 34 items` (the 27B's plane count) while the 9B's route prints `.. 1 of 18 items ..`, so its follow-on clauses (the next demote receipt, the next promote receipt, the publish) cannot anchor. A gate model-shape assumption on this card class, left as is (no gate changed after a result); the cell is decided on the target card below |
| `spec-on-cache-hit-gate.sh qwen` | `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` (61 ok) | `ALL GREEN (qwen)` (61 ok) |
| `prefix-newest-turn-fits-gate.py`, the 9B artifact | `REFUSED: cohort promotion did not happen for 2800 tokens: second send cached=2800 of 2800, published 2784` (exit 2, `twin-9b-shape-refused/`) | the identical refusal: the gate's default cohort shape is the 27B's (lane B's local runs use the 27B artifact, `run-day23-twin.sh`), not a door result |
| `prefix-newest-turn-fits-gate.py`, the 27B artifact (lane B's local shape) | LOCAL-TWIN27-OFF | LOCAL-TWIN27-ON |
| GPU unit cells `option_b_*` (2), `option_c_*` (6) on the copy-stream engine (`cargo test -p memra-server --lib -- --ignored --test-threads=1`, under `flock` on the canonical lock) | `test result: ok. 8 passed; 0 failed` (`gpu-unit-cells.log`) | |

Driver correction, stated: attempt 2's first twin cell logged 15 "lock busy" retries and `rc=2`; the gate's line was
`REFUSED: --out must be a new directory: .../twin-off` (the first try created the directory before its lock check
and my driver's retry keyed on the word REFUSED without removing it): a driver bug of mine, not a held lock
(`attempt2-twin-outdir-refusal/`). The driver now removes the cell's out directory before every try and keys on
the lock's own refusal text. The lock WAS free when the corrected tail ran.

**Target card, BOX3** (one RTX PRO 6000 Blackwell Server Edition at its 600 W limit; `pro-single-day17/box/`;
binary built on the box from `fc46e230d`, `ceaf238f…`; the Qwen3.8-27B NVFP4-Q5K MTP artifact;
`MEMRA_HOSTGATE_CACHE_MB=256`; one collector hold for the nine door gates (`gates/`, `--external-lock`, lock proof
`gates/LOCK.json`, `--validate` rc=0, `CELL.jsonl` `status: executed-not-qualified`, sampler 1737 rows: 31 to 63 C,
31.9 to 510.1 W under 600 W), the hit gate under its own `flock` on the canonical lock (no `--external-lock` arm),
verbatim per arm:

| gate | door OFF | door ON |
|---|---|---|
| identity default | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` (12 ok) | the same (12 ok); server: `demote submitted off the tick: 64 tokens, 159.9MB, ticket seq=1, 34 items on the contracts door's copy stream`, `demote published off the tick: ticket seq=1 complete after 1 poll(s), 186.5ms from submission to completion (tick-top poll)` |
| identity plain | `ALL GREEN (teeth=0)` (12 ok) | `ALL GREEN (teeth=0)` (12 ok) |
| failure gate | `KV-HOST-SPILL FAILURE GATE: 1 FAILURE(S)` (`FAIL: pool-full refusal is LOUD and named`, 13 ok) | the same line, 13 ok |
| contract fault gate | n/a | `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN`, 62 ok; one typed refusal each: `demote failed (tier D2H producer fence refused: injected failure (MEMRA_KV_HOST_FAULT=contract-presubmit)); nothing demoted`, `demote failed (tier D2H receipt refused: injected failure (MEMRA_KV_HOST_FAULT=contract-postpublish)); nothing demoted`, `promote refused (contracts door): tier H2D batch partially refused: 1 of 34 items (injected failure (MEMRA_KV_HOST_FAULT=contract-promote-reject))`, `.. tier H2D destination 0 not publishable: injected failure (MEMRA_KV_HOST_FAULT=contract-promote-readyview)`, `.. tier H2D producer fence refused: injected failure (MEMRA_KV_HOST_FAULT=contract-promote-presubmit)`, `.. tier H2D publication refused: injected failure (MEMRA_KV_HOST_FAULT=contract-promote-postpublish)` |
| hit gate | `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` (61 ok) | `ALL GREEN (qwen)` (61 ok) |
| GPU unit cells `option_b_*` (2), `option_c_*` (6) on the copy-stream engine (`unit-cell/`, one collector hold, tree `fc46e230d`) | `test result: ok. 8 passed; 0 failed; 0 ignored` (`unit/cargo-test.log`) | |
| twin gate | `PREFIX-NEWEST-TURN-FITS: budget_bytes=1073741824 cohort_bytes=736755712 turns=8 cold_turns_after_1=0 cached_ok=7/7 lines_ok=8/8 evictions=9 cohort_evictions=3 self_evictions=0 refused_or_skipped=0 effective_free_ok=8/8 identity_ok=8/8 grid_ok=21/21 grid=32 off_grid_calls=0 V1=ok V2=ok V3=ok V4=ok V5=ok V6=ok -> PASS` | the identical line, `-> PASS` |

Reading: the identity gate and the fault gate are green on the target card class with the door ON and the demote
on the copy stream, so the slice stands on that card; on the 5090 class the identity gate is green in every arm
and the fault gate's one red cell is the gate's 34-item hardcode, stated above.

### The stall cell, demote arm (the pre-registered rule applied as written)

Boot `on` of `pro-single-day17/stall-cell.sh` (the day-16 script with the receipts root moved: `MEMRA_CTX=8192
MEMRA_MAX_SESSIONS=4 MEMRA_SERVE_SPEC=0 MEMRA_PREFIX_CACHE_MB=256 MEMRA_KV_HOST_MB=8192 MEMRA_KV_HOST_CONTRACTS=1`),
one collector hold (`stall-on/lock.json`: `inherited-flock-same-open-description`, `--validate` rc=0,
`executed-not-qualified`; sampler 443 rows: 40 to 52 C, 53.0 to 331.3 W under 600 W), harness `stall_cell.py`
unchanged, N=5 per arm per order, both orders, `STALL REPLAY: PASS (replay agrees with the harness's rule line)`
for both receipts (`box/replays.log`). Verbatim:

`STALL rule cell=stall-demote-on arm=demote n_per_order=5 pooled=10 idle_runs=10 idle_p50=13.4 idle_p95=14.7
idle_p99=14.8 idle_max=14.9 arm_runs=10 arm_p50=13.4 arm_p95=14.8 arm_p99=130.9 arm_max=163.4 stall_median=149.6
stall_min=75.9 stall_max=150.0 server_demote_ms=[127.9, 134.0, 133.4, 133.1, 132.3, 132.4, 132.7, 132.9, 132.8]
server_promote_ms=[] intruder_prompt_tokens=[95, 99, 97, 98, 97, 99, 97, 97, 97, 97] tenant_text_identical=True
errors=0`

Rule: gap 76.0; `129.3 < 149.6 <= 174.5` -> **`toward_off`** (the day-16 receipts on this box: ON `193.5`, OFF
`117.5`; a same-box cross-sitting reading, stated as such). What moved: the stretched tick is `arm_max=163.4`
against day 16's ON `207.6` and OFF `132.0`: 44 ms less than day 16's ON arm, the copy time the census measured
(42 ms of D2H in the OFF tick), and the two receipt hashes are what remains above the OFF arm. The server's
lines per demote: `demote submitted off the tick: 64 tokens, 159.8MB, ticket seq=N, 32 items on the contracts
door's copy stream` (no draft plane under `MEMRA_SERVE_SPEC=0`), `contracts door D2H receipt: .. items=32 (16 KV
planes) complete=32 require=ok .. retired acknowledged`, `demote published off the tick: ticket seq=N complete
after 1 poll(s), 52.8 to 58.8ms from submission to completion (tick-top poll)` (one poll: the copy completes
inside the intruder's own prime tick, the next tick top publishes), then `demote: 64 tokens, 159.8MB in 127.9 to
134.0ms` (submission to publication). Admissibility clauses: `errors=0`, `tenant_text_identical=True`, replay
PASS; the server log of the boot carries 21 `demote submitted`, 21 `demote published`, 21 `demote:` and 21 `D2H
receipt` lines and no `demote failed`, `TIER DISABLED` or `no longer whole` line: every submitted demote published.
The harness attributes a `demote:` time to 9 of the 10 demote-arm runs: run 2, the first demote intruder,
shows `demote_ms=[] stall=75.9`, and its window carries no `evict` line: the harness's demote mode seeds no entry
before the timed runs (`setup: []` in the receipt), so the first intruder's seed insert finds nothing to evict and
its stall is the 64-token prime alone; from run 4 on every intruder evicts the previous run's entry. The same shape
as the day-16 receipt of this cell, read from the receipt rather than inferred. The promote arm ran too because
the day-16 script runs both boots' arms; it is not the day's claim and is recorded verbatim in `box/stall-on/ev/promote.log`
(`stall_median=86.4` against day 16's ON `162.8`; its intruder's insert demotes the other entry through the new
route, `server_demote_ms=[207.6, 209.0, 172.6, ..]` spanning submission to publication, and the promote is
`11.8 to 12.0ms` after the first pair: not pre-registered, not a verdict, a reading for the promote's turn).

## Task 3: records

`STATE.md` rewritten (day 17), `OWNER-THREAD-OFFLOAD.md` gains "Move 1, first slice: what landed on day 17"
and the owed list (the promote's turn first, then the receipt hashes, the by-reference routes, the decision cell
(i) with both classes), `research/INDEX.md` row `spill-a-20260919/day17`, `docs/FLAGS.md` door row (with the
code). The box worktree `/root/wt-a` is at `fc46e230d` on `lane-a-day17`; `/root/spill-receipts/a-day17/`
mirrored to `pro-single-day17/box/` (bins excluded).
