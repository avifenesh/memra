# WP-A day 21: Move 2 slice 2, the hit restore off the tick behind rule 3's reader fence

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`, resumed from `62aa92279`; `origin/main` `4bb2afb63` merged
`--no-ff` as `2c656a3d4` (clean; marker census OK). Every push today in the announced
`MEMRA_RELEASE_QUALIFICATION_MODE=development` mode (the #589 hook refuses an engine-touching range otherwise);
no qualification is claimed for any cell; every cell below is `executed-not-qualified`.

## Task 1, pre-registration (this section is committed before any slice-2 code)

### The op shape: a borrowed destination, a pinned borrowed source

Day 20 found that a capture's source is the live session cache's plane (borrowed) and named the engine's
`D2dCapture` (borrowed source, owned registered destination). The restore has the mirror problem and one more: its
DESTINATION is the admitted request's fresh session cache (`pp::new_cache_planned`), owned by the session for its
whole life and never moved into the engine's registry; its SOURCE is a published device entry's `PrefixPlane` (a
plain `CudaSlice<u8>` the device LRU owns), which today is NOT registered with the engine either (a captured plane
comes back through `take_plane` into a `PrefixPlane` before publication). Registering the source would mean moving
the entry's planes out of the entry for the copy's duration, and the entry must stay servable to other hits while a
restore reads it. So the restore class is a same-device copy from a BORROWED source span into a BORROWED destination
span, with nothing owned by the registry on either side, and its typed op is the engine's `D2dRestore` with
`CudaTransfers::submit_d2d_restore`. What makes the borrowed source safe is the device LRU's PIN: `PrefixCache::pin`
takes the entry out of the LRU index (`oldest_evictable` never names it; `evict_all`, `evict_all_demoting`, the trims
and the tenant purge's device half skip pinned entries by construction), so no path can free the source under the
running copy. The pin is taken at submit and becomes the request's serving pin at re-admission ("the pin count carries
it"). What makes the borrowed destination safe is the park: the request that owns the cache is parked on the requeue
and nothing reads or writes its cache until the owner stream's wait on the copy's completion event is installed.

### The contract in `memra-tier` terms

- Ticket: one per restore (every KV plane of the entry, K and V, one `Epochs`), on the copy stream behind a
  producer fence recorded on the owner stream at submit (after the recurrent-state copies, below).
- Producer-side guarantee: the source entry is published (its bytes complete: an on-tick capture is stream-ordered
  before this submit on the owner stream; a copy-stream capture published only after its events) and PINNED from
  submit to acknowledge. The pin is the contract's producer guarantee for a borrowed source.
- Reader fence (rule 3, `conformance/reader_fence.rs`, re-bound to a D2D): the restore's items are NOT fenced at
  submit. At the settle, after every completion event is observed complete, the OWNER stream's wait on each item's
  event is installed (`install_consumer_wait`, now over unfenced D2D items too) and only then is the destination
  `ready`; the parked request's first prime chunk is issued on the owner stream after re-admission, behind that wait.
  A prime issued before the wait is unordered whatever the copy's state (the contract orders through the fence, not
  through observation).
- Receipt: byte count, epochs, completion event; no checksum term (slice 3's decision, as for the capture), so the
  host-contract gate refuses a restore item by construction and no path publishes a restore through it.
- Retire: `retire(ticket, None)` after the wait is installed (an unpublished-in-the-engine's-sense, landed ticket:
  the destination leaves through the caller, not through `take_destination`), then `acknowledge`, then the pin
  passes to the serving session.

The frozen schedule: `d2d_restore_ready` (a landed copy without the installed wait is not `ready`; a prime issued
before the wait is unordered and fails the schedule; the source pin is held from submit through acknowledge) with
CPU bindings and a red arm (`prime_early`: a prime issued between the landing and the wait).

### The recurrent state in this slice

The KV rows are the byte-span class (`CudaSlice<u8>` planes, `bytes` from the head). The recurrent state
(`conv_state`, `ssm_state`, `CudaSlice<f32>`) is not in that class and this slice keeps it on the OWNER stream at
submit (`engine.copy_into`, exactly the OFF statements), stream-ordered before the producer fence and so before the
first prime chunk. On the 27B that is about 157 MB of a 4096-token entry's 278 MB: the slice moves the rows, not the
fixed term. Stated here so the stall cell's reading is bounded correctly; moving the f32 planes is a later slice once
the class admits typed f32 spans.

### The `Restoring` state, on the session

`Restoring` is a state of the parked REQUEST, not of the entry: the entry stays published, servable and pinned. The
worker holds at most one (`HostPrefixCache::restoring`): the request's id (`request_id`, the HTTP envelope's), its
pool key, the source pin, the entry's tokens and boundary logits, the destination cache, the ticket, and `ready`
(the wait installed, the ticket retired and acknowledged; the cache is complete and waits for its request).

| path | on a `Restoring` request |
|---|---|
| admission of the SAME request (its re-admission) | pending: parked again (`requeue.push_back`, FIFO, never shed); ready: admitted, and the hit site takes the ready cache and the restore's pin instead of allocating and copying; the hit accounting is the OFF hit's (`hits`, `hit_tokens`, the `hit:` line) |
| admission of another request | its own whole-entry device hit takes the tick program (the synchronous restore); one restore in flight per worker; a second hit on the same entry is served from the same published source, the pin count carries both |
| the tick top | polls the ticket after the capture poll; on every event complete installs the owner-stream wait, releases the producer fence, retires, acknowledges: `ready`; a ready restore whose request has not re-admitted within three tick tops is dropped typed (its cache freed, its pin released) |
| eviction, trim, reclaim, the eviction sink | the source is pinned: not a candidate anywhere (by construction, unchanged code); the reclaim and every trim settle the restore first (`Block`) so no copy is running when a trim quiesces the pool |
| tenant purge | settles first (`Block`); the purged tenant's ready restore is dropped (cache freed, pin released) so a revoked tenant's bytes are never primed on after the purge's receipt; another tenant's stays ready |
| shutdown | a host wait on the events, then drop; nothing is primed after a stop |
| the tier latched off, the capture or Move 1 paths | independent: a restore reads device memory only; a `Latched` restore settle latches the tier off (below) |
| a second restore while one is pending | never submitted: the probe answers "through" and the request takes the tick program |

### The retire seam (the lead's note on revuto #634, received mid-day, after the first worker commit)

Slice 1's finding: a capture's source is a live session's plane and the engine retains none of it, so the lead
now settles a pending `Capturing` entry `Block` before any session leaves `active` (integ36, `bd6584f9b`, merged
into this lane as `14b2d7b0f` before the refusal-path mirror and the census below). The restore's mirror is built into its ownership rather than added at
the seam: its DESTINATION cache is owned by the pending state (`HostPrefixCache::restoring`) and never by a
session while the copy is in flight; the request that will own it is parked on the requeue, not in `active`, so
no retire, park, rewind or prime can meet the cache before `host_restore_take_ready` hands it over, and that
hand-over requires `ready` (every completion event observed complete, the owner-stream wait installed). Its
SOURCE is pinned from submit through the hand-over or the typed drop (a drop with a pending contract settles
`Block` first), and a pinned entry is out of every eviction index by construction. The census
`every_path_that_meets_a_restoring_request_settles_or_ignores_it_as_stated` pins both: the cache leaves the
pending state exactly once, under the `ready` check, and the pin is taken before the submit and carried to the
serving session. The lead's second item, a refused submission whose producer event is still pending, is mirrored
verbatim: the restore's refusal path drains the owner stream, releases the fence, and latches the route off typed
if it still will not release (never `let _`); the same census pins drain, then release, then latch.

### Failure paths, typed, one line each

- A refusal before or at submit (the fresh cache's allocation, the OFF validation, the producer fence, the engine's
  `submit_d2d_restore`): the cache is dropped, the pin released, one `[prefix-cache] restore refused (contracts door):
  ..; tick program` line, a one-tick memo (`restore_cold`, by request id) so the same admission takes the OFF program
  (the synchronous restore, or the cold prime when the OFF path would) without a second attempt or a second line.
- A lost observation at the settle (`Latched`): the copy may still be writing the destination and reading the
  source, so NEITHER is freed: the destination cache is forgotten and the source pin is kept forever (a leak by
  design, the frozen `Entry::drop` rule mirrored), the tier latches off, and the restore path latches to the tick
  program for the boot: `[prefix-cache] RESTORE OFF-TICK DISABLED: ..`. The parked request re-admits and takes the OFF
  program on a fresh cache.
- A ready restore whose request does not consume it (the request failed a later admission check, was refused, or
  disconnected): dropped typed at the next probe or after three tick tops (cache freed, pin released).
- A ready restore whose entry no longer matches its request (the entry gone, or the prompt no longer a whole-entry
  hit on it): dropped typed; the OFF program serves.
- The fail-closed arm (#622 mirrored): a pending restore missing its cache or its ticket never re-admits its request
  onto it; with a submitted ticket and no cache the ticket is settled and retired through the engine where reachable
  and the tier and the restore path latch off; with a cache and no ticket the cache drops whole (nothing submitted).

Nothing half-restored is ever primed on: the only cache a request primes on is the OFF path's (validated and copied
synchronously before the prime) or a ready restore's (every event observed complete, the owner-stream wait
installed before the request was re-admitted).

### The one-program law

The restored rows are the bytes a synchronous D2D would have produced: the same `memcpy_dtod` per plane from the
same source plane at the same offsets and lengths, into the same fresh cache the OFF path allocates, the same
`copy_into` for the recurrent state, the same `len`, `len_d` and `pos`; the request then primes its suffix on the
same program. The hit gate's identity clause (`spec-on-cache-hit-gate.sh`, plain and spec arms: a hit's bytes
equal the cold bytes) and the identity gate are the proof; any digest difference is a FAIL of the slice, never a
tolerance. The delayed-copy-stream fault (cell (iii)) is owed with slice 3's fault cells and not run today.

## Task 1, what landed (commits `2e55f3a9f` tier, `d8e66af09` engine, `ce2089638`, `c2874bd42`, `1350f118b` worker)

- `crates/memra-tier/src/conformance/d2d_restore.rs`: the schedule `d2d_restore_ready` over `D2dRestoreFixture`
  (unversioned beside the frozen schedules; `WIRE_VERSION` 1): not landed is not ready; landed without the installed
  reader wait is still not ready and a prime issued then is unordered; the install fences every item and makes it
  ready; `retire(None)`, `acknowledge`; the source pin held from submit through acknowledge; no witnessed checksum
  (the host-contract gate refuses). Plus `d2d_restore_primed_before_its_wait_is_unordered`. CPU bindings
  `tests/contracts/d2d_restore_bindings.rs`: `day21_d2d_restore_is_ready_only_after_the_landing_and_the_installed_reader_wait`,
  `day21_red_arm_prime_before_the_reader_wait_fails_the_schedule` (the red arm: a prime on issue; the schedule
  fails), `day21_d2d_restore_item_without_a_witnessed_checksum_is_refused_by_the_host_contract_gate`. `memra-tier`
  contracts `77 passed` (74 before).
- Engine: `D2dRestore<'a>` (a borrowed `&CudaSlice<u8>` source, a borrowed `CudaViewMut<u8>` destination of exactly
  the item's bytes, a producer fence), `CudaTransfers::submit_d2d_restore` (requires the copy stream, else
  `Unsupported`; all-or-nothing validation; the copy stream waits on the producer event, `memcpy_dtod`, a completion
  event on the copy stream; the items UNFENCED at submit; nothing registered; no owner-stream wait anywhere at
  submit), `restore_landed`; `install_consumer_wait` now fences every unfenced item that is not a D2H (an H2D as on
  day 19, a D2D restore now; a D2D capture, fenced at submit, is left alone). Census `d2d_restore_rules_are_as_stated`
  (two `memcpy_dtod` statements in the body, the capture's and the restore's); GPU cell
  `d2d_restore_lands_on_the_copy_stream_and_is_ready_only_after_the_installed_wait` (ignored without a device).
  Engine `tier_transfer` lib tests `6 passed; 3 ignored`.
- Worker (`crates/memra-server/src/worker.rs`): `HostPrefixCache::restoring` (the one `Restoring` request),
  `restore_off_tick_disabled`, `restores_landed`; `PendingRestore`, `PendingContractRestore`, `RestoreSettle`,
  `HostRestoreFailure::Latched`, `RestoreSettled`, `HostRestoreOutcome`, `RestoreProbe`; `prefix_restore_validate`
  (the OFF checks split out of `prefix_restore_at`, statements unchanged; `prefix_restore_at` calls it first);
  `host_restore_park_probe` (the admission loop, right after the promote probe: the same predicates, a whole-entry
  plain hit, the fresh cache, the OFF validation, the pin, `host_restore_submit`, the park), `host_restore_submit`
  (recurrent state, `len`, `len_d` on the owner stream, then the producer fence and ONE restore batch; a refusal
  drains the owner stream and releases the fence or latches typed), `host_kv_planes_settle_restore` (landed, then
  `install_consumer_wait`, then `release_producer`, `retire(None)`, `acknowledge`), `host_restore_settle_with` (the
  CPU-testable half: fail-closed arms; `Latched` forgets the cache and keeps the pin), `host_restore_settle_pending`,
  `host_restore_drop`, `host_restore_expire_ready` (three tick tops), `host_restore_purge_tenant` (worker level, with
  the device cache in hand, before either index purges), `host_restore_drain_at_shutdown`,
  `host_restore_probe_decision`, `host_restore_take_ready` (admit's hit site: the ready cache and the pin, under
  `ready` only; the pin carries over as the serving pin). Call sites: the tick top after the capture poll plus the
  expiry; both idle waits; the admission loop's park (`requeue.push_back`, FIFO, never shed); the reclaim and the three
  trims settle first; the purge; the run loop's exit. `host_tier_context`: the ledger's in-flight dimension gains the
  restore term (`3 x (2 x max layers + 2)`). CPU tests: `restore_probe_decision_parks_its_own_request_and_names_an_orphan`,
  `a_pending_restore_missing_its_cache_or_ticket_fails_closed`,
  `every_path_that_meets_a_restoring_request_settles_or_ignores_it_as_stated` (the census); the day-20 census now
  reads the restore drain after the capture drain. Server lib suite `803 passed; 14 ignored`; clippy `-D warnings`
  clean on the three crates. `docs/FLAGS.md` door row day-21 sentence (in the worker commit). No new flag, no new
  numeric program, no new `unsafe`, no external dependency.
- What the CPU tests cannot reach: the pending-to-ready path needs a device cache (`Cache` holds `CudaSlice`s), so it
  is exercised on the card only, by the hit gate's ON arm (every hit takes the route; the `restore landed off the
  tick` line is the evidence) and the engine's GPU cell. Stated, not hidden.

### Cells (target card, BOX3, one RTX PRO 6000 Blackwell, 600 W; the collector; `executed-not-qualified`)

Gates as on day 20: identity default and plain, OFF and ON; failure OFF and ON; the fault gate; the twin gate OFF and
ON; the hit gate OFF and ON (its own flock); the GPU unit cells `option_b_*`, `option_c_*`, the engine's
`d2d_capture_*` and the new `d2d_restore_*`. If identity, fault or hit is red in either arm the slice does not stand:
it stays `wip:` with the red receipt.

**The restore stall cell (memra#536 Move 2 cell (ii)).** `stall-cell.sh off|on` boots the day-20 server (prefix cache
ON at 1024 MB, host tier 8 GiB, `MEMRA_SERVE_SPEC=0`, door OFF or ON); the harness `stall_cell.py --mode restore`
(a NEW arm; the four earlier arms are byte-for-byte unchanged): an untimed setup posts the intruder prompt once so
its grid seed captures and publishes; then in every timed run the intruder re-posts the SAME prompt at the tenant's
24th token with `max_tokens=1`: a whole-entry HIT whose restore copies the entry (about 5088 tokens on the 27B) and
primes the 32-to-35-token suffix on the tick; its `cached_tokens` is recorded per run. N=5 per arm per order, both
orders inside each boot; the two door arms run OFF, ON, ON, OFF in one collector lock hold. Rules, fixed before the
run (ms, the harness's `stall_median`):

- Admissibility: `errors=0`, `tenant_text_identical=True`, and every intruder's `cached_tokens >= prompt_tokens - 64`
  on every receipt (every run a hit). A receipt that fails admissibility decides nothing and is reported as such.
- C1, ON against OFF, same hold, per pass: `on_under_off` if `stall_median(ON) <= stall_median(OFF) - 6.0`; `flat`
  within 6.0 either way; `on_over_off` above (the day-18 promote arm's threshold).
- C2, the copy's own duration: the ON arm's `server_restore_ms` (the `restore landed off the tick` lines, ms from
  submission to the observing poll), a list with its median; a reading, no threshold.
- What the cell can and cannot say: the intruder's suffix prime is on the tick in both arms, the recurrent state
  (about 157 MB) stays on the owner stream in both arms, and only the KV rows (about 150 MB at 5088 tokens) move; on
  this card a 150 MB D2D is well under a millisecond, so the expected reading is `flat`, which records that the rows'
  on-tick share is under the cell's resolution. A stall difference larger than the bound in either direction is a
  finding to explain, not a verdict on the door. Nothing here decides the door.

## Task 2, the target card: gates (BOX3, tree `1350f118b`, binary `ddde2083…`; receipts `pro-single-day21/box/`)

Built on the box from the shipped bundles under `nohup` (`build rc=0`, `box/build.log`; a first build died with an
ssh drop and left nothing). The collector cell `gates` (`tools/tier-battery.py --rig pro-single --external-lock`,
`/tmp/memra-gpu.lock`, `LOCK.json` owner `collector`) ran every gate in one lock hold; the hit gate under its own
`flock` on the canonical lock; the unit cells in a second collector hold; the stall cell in a third. No lock retry
was needed; the card was idle at the start (`0 MiB`). Every line verbatim, N=1 per gate cell, the card's regime in
the collector's `command.gpu.csv` beside each cell; `executed-not-qualified`.

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
| unit-cell (`option_b_*`, `option_c_*`) | the copy-stream engine | `test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 809 filtered out; finished in 0.27s` |
| unit-cell (engine `d2d_capture_*`, `d2d_restore_*`) | the two D2D classes on the card | `test tier_transfer::tests::d2d_capture_lands_on_the_copy_stream_and_publishes_only_after_its_event ... ok`, `test tier_transfer::tests::d2d_restore_lands_on_the_copy_stream_and_is_ready_only_after_the_installed_wait ... ok`, `test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 554 filtered out; finished in 0.52s` |

**Where the route engaged, and where it did not.** The plain ON arm's identity server log carries two restores,
verbatim: `[prefix-cache] restore submitted off the tick: 64 tokens, 32 planes (158.8MB), ticket seq=6 on the
contracts door's copy stream; recurrent state copied on the owner stream; request parked` then `[prefix-cache] restore
landed off the tick: 64 tokens (158.8MB) complete after 1 poll(s), 75.4ms from submission to completion, 75.5ms to
re-admission (tick-top poll)` (and `seq=7`; the 75 ms is submission to the OBSERVING poll, the same semantics as the
capture's and the promote's lines: the rest of the tick ran between; the 158.8 MB is the entry's byte count, of which
the recurrent state, copied on the owner stream, is about 157 MB and the 32 KV planes about 1.9 MB). Zero `restore
refused`, `restore dropped` or `RESTORE OFF-TICK DISABLED` lines anywhere under `gates/` (the one `TIER DISABLED` line
under `failure-on` is the failure gate's own `alloc-fail` arm, verbatim `[prefix-host] TIER DISABLED: pinned host alloc
of 69632 B failed: injected failure (MEMRA_KV_HOST_FAULT=alloc-fail) ...`). The hit gate's ON arm shows ZERO restores
through the route: every entry it publishes is a spec-boundary capture (`insert (spec-boundary)`, the MTP draft plane,
14 lines) and its 14 hits (`hit: 64 of 106`, `64 of 119`, `96 of 119`) are on draft-bearing entries, which the route
refuses by name; both hit-gate arms ran the tick program for their restores. The default (spec) identity arm likewise.
So the one-program proof of THIS slice on the card is the plain identity arm: OFF against ON, `teeth=0`, with the ON
arm's hits restored through the route; the hit gate's identity clause covered the tick program today and will cover
the route when the draft-bearing restore lands (owed, `OWNER-THREAD-OFFLOAD.md` item 3).

**Reading.** The identity and fault gates are green in both arms on the target card class and the hit gate is green
in both arms, the pre-registered condition: the slice STANDS under the door, with the scope stated above (the plain
whole-entry hit; draft-bearing hits keep the tick program). Commits `ce2089638`, `c2874bd42`, `1350f118b` keep their
`wip:` prefix (history is not rewritten); this record and `OWNER-THREAD-OFFLOAD.md` are where the slice is promoted to
"landed under the door".

## Task 2, the target card: the restore stall cell (pre-registered above; run after the unit cells, one collector hold)

`stall-both.sh` under the collector (`box/stall-both/`, `stall-both.collector.log`): OFF pass 1, ON pass 1, ON pass 2,
OFF pass 2 (`box/stall-restore-{off,on}/ev.pass{1,2}/`), each boot's harness interleaving idle/arm in both orders,
N=5 per arm per order, `MEMRA_SERVE_SPEC=0`, cache 1024 MB, host tier 8 GiB. The one intruder prompt is 5121 tokens as
the tokenizer made it (the pre-registration wrote "about 5088"; the recorded figure stands), seeded untimed in setup
so its grid seed published at 5088; every timed intruder then hit it whole (`cached_tokens=5088`) and primed the
33-token suffix. Rule lines, verbatim (`receipt.json` `rule_line`):

`STALL rule cell=stall-restore-off arm=restore n_per_order=5 pooled=10 idle_runs=10 idle_p50=13.5 idle_p95=14.7 idle_p99=14.9 idle_max=16.3 arm_runs=10 arm_p50=13.4 arm_p95=14.8 arm_p99=15.5 arm_max=104.8 stall_median=88.1 stall_min=87.8 stall_max=91.3 server_demote_ms=[] server_promote_ms=[] server_restore_ms=[] intruder_cached_tokens=[5088, 5088, 5088, 5088, 5088, 5088, 5088, 5088, 5088, 5088] intruder_prompt_tokens=[5121, 5121, 5121, 5121, 5121, 5121, 5121, 5121, 5121, 5121] tenant_text_identical=True errors=0` (pass 1)

`STALL rule cell=stall-restore-on arm=restore n_per_order=5 pooled=10 idle_runs=10 idle_p50=13.5 idle_p95=14.8 idle_p99=14.9 idle_max=14.9 arm_runs=10 arm_p50=13.5 arm_p95=14.8 arm_p99=22.1 arm_max=93.7 stall_median=80.1 stall_min=80.1 stall_max=80.2 server_demote_ms=[] server_promote_ms=[] server_restore_ms=[14.5, 14.5, 14.5, 14.5, 14.5, 14.5, 14.5, 14.5, 14.4, 14.5] intruder_cached_tokens=[5088, 5088, 5088, 5088, 5088, 5088, 5088, 5088, 5088, 5088] intruder_prompt_tokens=[5121, 5121, 5121, 5121, 5121, 5121, 5121, 5121, 5121, 5121] tenant_text_identical=True errors=0` (pass 1)

Pass 2: OFF `stall_median=87.9` (`stall_min=87.8 stall_max=88.1 arm_p99=15.5 arm_max=101.5`), ON `stall_median=80.2`
(`stall_min=80.1 stall_max=80.3 arm_p99=22.1 arm_max=93.8`, `server_restore_ms=[14.5, 14.5, 14.5, 14.5, 14.5, 14.5,
14.5, 14.5, 14.4, 14.4]`); full lines in the receipts and in `box/stall-restore-*.pass2.log`.

Admissibility: `errors=0` and `tenant_text_identical=True` on all four; every intruder read `cached_tokens=5088`
against `prompt_tokens=5121` (within 64): a whole-entry hit in both arms, every run. The ON arm's `server_restore_ms`
(submission to the observing poll) is 14.5 ms on all 20 lines: the copy is observed at the next tick top, one tick
after submission (the tick is 13.4 to 14.5 ms here), so the figure is the tick period, not the copy.

**Verdict, by the pre-registered rules.** C1 pass 1: `on_under_off` (ON 80.1 against OFF 88.1, delta -8.0, bound
6.0). C1 pass 2: `on_under_off` (ON 80.2 against OFF 87.9, delta -7.7). C2: ON `server_restore_ms` median 14.5 both
passes (the tick period; the copy's own duration is below it). The pre-registered expectation was `flat`; the reading
is larger than the bound and is a finding to explain, not a verdict on the door: the ON arm's `arm_p99` rose from
15.5 to 22.1 in both passes while its `arm_max` fell by about 8 ms, which is the shape of a SPLIT, not a removal. OFF
does the intruder's whole admission in one tick (the fresh cache's allocation, the recurrent-state and row copies, the
suffix prime); ON does the allocation, the recurrent-state copies and the submit in tick T and parks, and the
re-admitted request primes in tick T+1: about 8 ms of the intruder's owner-thread work moved from the prime's tick to
the previous one (the second bump at 22.1 - 13.5 = 8.6 ms). The KV rows' D2D share (about 150 MB at 5088 tokens) is
under the cell's resolution on this card, as pre-registered, and the recurrent 157 MB stays on the owner stream in
both arms. Same-window, one lock hold, N=5 per arm per order, both orders; `executed-not-qualified`; the cell decides
nothing about the door.

## Task 3: records and checks

`STATE.md` rewritten (day 21). `OWNER-THREAD-OFFLOAD.md`: the day-21 section (the pre-registration pointer, what
landed, what Move 2 still owes). `research/INDEX.md` row `spill-a-20260919/day21`. `docs/FLAGS.md` door row day-21
sentence (in the worker commit `ce2089638`). Checks on the final tree: `cargo fmt --all -- --check` clean; clippy
`-D warnings` clean on `memra-tier` (`--all-targets`), `memra-engine` and `memra-server` (`--lib --tests`); `memra-tier`
contracts `77 passed`; `memra-engine` lib `tier_transfer` `6 passed; 3 ignored`; `memra-server` lib `803 passed; 14
ignored`; `git diff --check` clean on every source commit of the day; `tools/check-flags.sh` (no uncovered runtime
names); `tools/check-conflict-markers.sh` OK; `python3 tools/check-public-boundary.py check` `0 new`. The lead's
integ36 branch (`bd6584f9b`, the #634 fixes) merged as `14b2d7b0f`. The box worktree `/root/wt-a` is at `1350f118b` on
`lane-a-day21`; `/root/spill-receipts/a-day21/` mirrored to `pro-single-day21/box/` (bins excluded); both bundles
removed on both ends; local `/tmp` scratch removed. Not run today: the local RTX 5090 door gates (owed with the lock,
as on days 18 to 20). #536 comment posted with this receipt; the issue stays open.

## Budget

About 4.2 agent-hours against 4: reading and the merge 0.6, the pre-registration 0.4, the tier schedule and bindings
0.3, the engine's restore class 0.4, the worker's `Restoring` request with the lead's #634 mirror and the census 1.3,
the harness arm, box scripts, the ship and the box cells 0.6 (one build lost to an ssh drop), records and the comment
0.6. Blockers: none (the box lock was free throughout; the 5090 was not touched). Open: slice 3 and the `d2d-delay`
fault, the f32 state, the spec-boundary route and the draft-bearing restore, an isolating stall cell, the 5090 door
gates on this tree, the Move 1 owed items.
