# WP-B day 37: O1, the `--kv-allocator vmm` deciding cell (decide-by 2026-10-04), with the allocator built to be measured at its best

OWED.md O1. The door is gate-only today: `kv-tier-gate --kv-allocator vmm` constructs fixed-VA VMM planes, and no
server path does. The decision record (`docs/decisions/KV-PHYSICAL-RECLAIM.md`, "Decide-by 2026-10-04") reads:
"Promote to the naked default for the tiered materializer only with the 32k residual classified on the target card
and the serving-shape gates run; otherwise delete the door". The classification half is met on both card classes
(`ACTIVE-32K G1 PASS (classified one-time-driver-mapping-metadata, 5 cycles)`, DAY12, ruling 6). The serving half has
never run, because there is no serving arm. `KV-RESIDENCY-DESIGN.md` option (b) names what a serving arm is and what
it owes: planes that reserve their virtual range at `max_ctx` and map physical granules only as the session's rows
grow, with a G1 grow series, the byte cells on both allocators, a stall cell at the granule boundary and an
admission accounting cell.

This day builds that arm, tunes it, and runs its deciding cell on both cards. Every cell is
`executed-not-qualified`; no default moves; the door verdict is the owner's.

## 1. Pre-registration

Committed and pushed before any day-37 code and before any day-37 boot. Nothing in section 1 changes after a
number is seen. A clause that fails is recorded as it reads; a design change after a failure is a new
pre-registration (section 1 addenda, each dated and pushed before its code).

### 1.1 What the arm is

A serving door, `MEMRA_KV_ALLOCATOR`: unset or `pooled` is today's program; strict `vmm` arms it. It is the serving
arm of the same decision as `kv-tier-gate --kv-allocator vmm` and carries the same decide-by, 2026-10-04, in a new
`docs/FLAGS.md` row in the same commit as the env read.

Armed, the full-attention K/V planes of a covered session (1.3) are on-demand VMM planes:

- **Reserve.** At construction each plane reserves a virtual range for its full capacity (`max_ctx` rows plus the
  8-byte tail pad, the same byte count the pooled plane allocates) and maps an initial extent covering
  `initial_rows` rows. The address never changes for the plane's life, so every captured graph, pointer table and
  kernel argument that bakes it stays valid.
- **Grow.** `ensure_rows(r)` maps one new extent per plane covering whole granules up to `r` rows. An extent is one
  `cuMemCreate` of `n x granularity` bytes, one `cuMemMap`, one `cuMemSetAccess`, and a stream-ordered zero fill of
  the new range on the owner stream (pooled planes are zeroed at allocation, `alloc_zeros`; the rows past `len` read
  the same zero bytes on both allocators).
- **Release.** Whole extents only (a `cuMemMap` range is unmapped whole). A release is never executed while work
  that may touch the range is in flight: the release records an event on the owner stream and the extent is
  unmapped and released only after the event completes (the reap). Dropping a plane moves its mapping to a graveyard
  reaped the same way; nothing on the release path calls `synchronize` on the owner thread.
- **Park trim.** When a covered session parks (plain pool or spec pool), extents wholly beyond `pos + SLACK` rows are
  released through the same deferred path. The parked session keeps its virtual range and its captured graphs; a
  resume maps again through `ensure_rows`. A trim whose event has not completed when the session resumes is
  cancelled (the extent is still mapped and its bytes are intact).
- **Everything else is unchanged.** Recurrent and latent state, the MTP draft graph state, prefix-cache entries,
  prime workspace and every transient stay in the CUDA pool. Covered sessions' planes never go through
  `into_pooled` (only prefix-entry planes do, and entries stay pooled).

The numeric program is unchanged by construction: the same kernels read and write the same bytes at the same
addresses a session keeps for its whole life. A session's allocator is chosen once, at construction; there is no
crossing between the two allocators inside a request. Section 1.5 A1 proves it in serving-shape gates.

### 1.2 The row bound and the ensure points

`SLACK = SPEC_SHRINK_SLACK = 64` rows (the engine's own bound on a speculative round's overshoot past the request's
need, `worker.rs` `SPEC_SHRINK_SLACK`). The bound a covered session's planes must be mapped through before any engine
call of a tick:

```text
bound(s) = min(capacity_rows, max(pos(s), floor_rows(s)) + tick_rows(s) + SLACK)
floor_rows(s) = the rows the request's prompt occupies (set at admission and at every resume to that request's P)
tick_rows(s)  = spec route: burst_t + k + 2   (MEMRA_SPEC_BURST, default 32; k = s.spec_k; the burst's surplus row and the verify row)
                plain route: max(1, serve_async_chain_k())
```

Ensure points, each named by a census test that reads the production text:

1. **Construction.** The covered creation seams (1.3) construct under a scoped ambient allocator that carries
   `initial_rows = min(capacity_rows, P + tick_rows + SLACK)`, so a restore or a resume copy that runs inside
   admission writes into mapped rows.
2. **The tick top.** After the admission block and before the first engine call of the tick (the `// 3. The tick.`
   seam in `run`), every active covered session is ensured through `bound(s)`.
3. **Adoption.** A parked covered session taken from a pool for a resume is ensured through the new request's
   `bound(s)` before any copy or prime into it.

A runtime fail-closed census backs the text census: the ambient scope counts the on-demand planes it allocated; the
session's ensure visitor must reach the same count (trunk cache K/V and the MTP scratch planes). A mismatch refuses
the session before its first engine call with a named error (`vmm: on-demand plane not reachable by ensure`), never
a fault.

### 1.3 Covered and uncovered routes

Covered under the door (the routes whose engine calls the tick-top ensure bounds): the MTP spec route
(`SpecSession`: trunk `Cache` K/V and `MtpScratch` K/V) and the plain route (`s.cache`, batched and async-chain),
single device, no SWA ring, no TP or PP stage ownership, no latent planes. Every other route and shape (gemma spec,
DSpark, glm5, ppN, TP, SWA-ring planes, MLA latent planes, the boot calibration probe, CLI binaries) allocates
pooled planes exactly as today, and the boot line names the door and the covered routes:
`[kv-vmm] door=ON routes=spec,plain granularity=<g> ...` (OFF: `[kv-vmm] door=OFF`).

### 1.4 Admission, accounting and failure

- **Owed growth.** Under the door every headroom reading of the admission block is reduced by
  `vmm_owed = sum over active covered sessions of (reserved - mapped)` bytes, through the existing
  `AdmissionHeadroom::less_pending` form. Pooled planes are allocated whole at admission, so the pooled gate already
  sees them; with the reduction the armed gate sees the same future footprint. A pool resume books its own owed
  growth in its own admission check. The booked cost (`estimate`, `booked_kv_bytes`) is unchanged.
- **Receipts.** `[kv-vmm] grow id=<req> plane=<k|v><layer> rows=<r> extent_bytes=<n> mapped=<m> reserved=<c>
  owner_us=<t>` per grow event (sampled: every event, gated on the door), `[kv-vmm] trim id=... released=<n>`,
  `[kv-vmm] reap released=<n> pending=<n>`; the admit and defer lines carry `vmm_owed=` when armed. `/metrics`:
  `kv_vmm_mapped_bytes`, `kv_vmm_reserved_bytes`, `kv_vmm_owed_bytes`, `kv_vmm_grows_total`,
  `kv_vmm_grow_failures_total`, `kv_vmm_released_bytes_total`, `kv_vmm_graveyard_bytes`.
- **Grow failure.** A `cuMemCreate` or map failure at an ensure point: reap the graveyard, then the existing single
  reclaim ladder (one LRU prefix entry, one parked session), then one retry. If it still fails, the session takes
  the step-OOM contract: a session that has emitted nothing parks back to the queue under `step_oom_retries`; an
  emitting session ends with the typed error the step-OOM path sends. Peers are untouched. Fault injection:
  `MEMRA_KV_VMM_FAULT=grow:<n>` makes the n-th grow event fail once with a synthetic out-of-memory (a new FLAGS row,
  a fault door of the gate).

### 1.5 Stage 0 before the design is final: the driver-call cost probe (5090 first, then the target card)

A diagnostic bin, `vmm-call-cost` (memra-engine), no serving change. On one device, for extent sizes of 1, 4, 16 and
64 granules, N=20 repetitions each, it times `cuMemCreate`, `cuMemMap`, `cuMemSetAccess`, the zero-fill enqueue,
`cuMemUnmap`, `cuMemRelease` and `cuMemAddressFree`, in three regimes: (i) owner stream idle; (ii) at least 50 ms of
queued work on the same stream (a chain of large device memsets on an unrelated buffer); (iii) the same queued work,
with the call made from a second thread while the owner thread times a small launch and its event. It prints
`VMM-CALL-COST op=<op> extent=<n> regime=<r> N=20 median_us=.. p95_us=.. max_us=.. queued_ms_remaining=..` per cell.

Branch rules, fixed now:

- **Blocking.** An op *blocks behind queued work* when its regime (ii) median is at least half the queued time
  still remaining at the call and at least 10 x its regime (i) median. It *blocks the owner thread from another
  thread* when regime (iii)'s owner launch-plus-event median is at least 10 x the same launch measured with no
  helper call in flight.
- **Release placement.** If neither `cuMemUnmap` nor `cuMemRelease` blocks behind queued work, the reap runs at the
  tick top on the owner thread. If either does and does not block the owner from another thread, the reap runs on a
  helper thread. If either blocks both ways, the reap runs only at a tick top where the owner stream reads idle
  (event query), and never inside a tick.
- **Grow placement.** If the sum of `cuMemCreate + cuMemMap + cuMemSetAccess` at 1 granule has p95 <= 100 us in
  regime (ii), grows run inline at the ensure points. Otherwise a helper thread pre-maps: when a covered plane's
  mapped headroom past `bound(s)` falls below two granules, the helper maps the next extent, and the ensure point
  waits only when the helper is behind (counted in `kv_vmm_grows_total{waited}`).
- **Extent size.** Each grow maps exactly the granules needed to reach `bound(s)` (so the initial extent is one call
  per plane whatever the prompt), rounded up to whole granules. No other extent policy is tuned.

Stage 0 decides placement per card class. The design uses the target card's reading for the target card and the
5090's for the 5090; if the two differ, both placements are built and selected by the device, the per-hardware rule.

### 1.6 Acceptance clauses (the deciding cell, both cards)

Workloads: the day-26 mix (`run-day26-cell.sh`, arms (i) open, (ii) bounded, (iii) continuation, three lengths,
N=5 per arm per length) on the spec path and on the plain path (`MEMRA_SERVE_SPEC=0`), and the day-34 burst cell
(`day34-client.py`, door `MEMRA_ADMIT_BY_MEMORY` OFF and ON at 32768). Arms: `pooled` (door unset) and `vmm`, one
binary, one boot per arm, both orders (pooled then vmm, vmm then pooled). Models and cards as DAY36 1.2 (the 9B at
`MEMRA_CTX=65536` on the 5090; the 27B at the checkpoint's 262,144 on the target card).

- **A1 identity.** Completion digests `vmm == pooled` on every request of every mix cell, both orders, both paths.
  Each of these gates reads its ALL GREEN line with the same ok count on both allocators: `serve-smoke`,
  `KV-HOST-SPILL IDENTITY GATE`, `KV-HOST-CONTRACT-FAULT GATE` (default and plain), `SPEC-ON-CACHE-HIT GATE` (OFF and
  ON), the twin gate, `prefix-newest-turn-fits`, `ADMIT-MEM BURST GATE`, `SPEC-CTX-EDGE GATE`. The CLI binaries
  (`kernel-check`, `run-gen`, `run-spec`) do not reach the serving door (1.3), so their lines are the pooled program
  on both arms by construction; they run once on the lane binary as the battery requires and prove nothing about
  the arm, and the serving-shape gates above carry the identity proof. (This refines `KV-RESIDENCY-DESIGN.md` (b)'s
  byte-cell list, which named them before the arm's scope was known.)
- **A2 G1 grow series.** `kv-tier-gate --case grow --kv-allocator vmm-ondemand` (new case, one cache, the gate's
  tokenwise program, collector-locked): one session decoding across at least 5 granule boundaries of every K/V plane
  in one process. At every crossing, read after a stream sync with no other allocation between the two reads,
  `driver_free_before - driver_free_after == extent_bytes` exactly; after the cache drops and the graveyard reaps,
  `driver_free == driver_free_before_the_cache` exactly (drift 0); tokens and logits bit-identical to the same case
  on `--kv-allocator pooled`.
- **A3 stall.** (i) Per grow event, `owner_us` p99 <= 500 over at least 20 events per card (inline placement), or the
  ensure point's wait p99 <= 500 us (helper placement). (ii) Serving, same-window interleaved, both orders, N >= 5 per
  arm, the day-34 burst at open output 8192 and the day-26 mix: TTFT p50 (vmm / pooled) <= 1.05, ITL p99
  (vmm / pooled) <= 1.05, TPOT p50 (vmm / pooled) <= 1.02, output token throughput (vmm / pooled) >= 0.98; every
  median with its N and the 250 ms regime.
- **A4 accounting.** Per request, from the retire line: `booked` equal on both arms; `mapped_at_retire - used <=
  planes x (granularity + SLACK x tok_bytes)` rounded up to whole granules, where `used = (P + G) x tok_bytes` per
  plane. Reported, no bound (these are the benefit readings): retained device bytes at idle after each mix
  (`cuda_driver_free_bytes`, `cuda_pool_cached_bytes`, parked sessions' mapped bytes) per arm, and the admitted
  concurrency of the burst per arm.
- **A5 fault.** `MEMRA_KV_VMM_FAULT=grow:<n>` on the day-34 burst and on the mix: the reclaim-retry line is present;
  with the retry forced to fail too (`grow:<n>,<n+1>`), the session outcome is the registered one (park when it has
  emitted nothing, the typed error when it has); peers' digests equal the no-fault boot; zero
  `CUDA_ERROR_ILLEGAL_ADDRESS` lines, zero worker panics, zero process exits.
- **A6 door OFF.** An OFF boot of the lane binary against the main binary (`17dceb981`) on the mix: digests equal on
  every request, zero `[kv-vmm]` lines apart from the boot line reading `door=OFF`.
- **A7 booking.** On every armed burst boot: zero `CUDA_ERROR_OUT_OF_MEMORY` lines, zero 503s, zero un-injected grow
  failures, and at every `verdict=admit` line the reading after `vmm_owed` covers the request's `required`.

CPU (before any card): unit tests per arm of the mechanism (extent arithmetic from rows and tail pad; grow, trim,
cancel-on-resume and reap state machine against a fake driver; the owed-growth reduction is the identity when
unarmed; the ambient scope count; the census tests of 1.2's ensure points); `cargo test -p memra-kv`, `-p
memra-server --tests`, `-p memra-engine --lib`, clippy `-D warnings`, fmt, check-flags. The spill review patterns
(move-then-match, parked requests and the idle wait, gate literals, happy-path release, borrowed sources, budgeted
bookings, guard identity) are read against every new state machine before the target-card sitting, and the reading
is written into section 2.

### 1.7 The decision rule (stated, not chosen) and what each card decides

Per card class, the rule reads:

- **PROMOTE-ELIGIBLE** when A1, A2, A5, A6 and A7 PASS and A3 (i) and (ii) PASS. The benefit readings of A4 go with it.
- **DELETE** when every correctness clause (A1, A2, A5, A6, A7) passes and A3 fails: the arm is correct and costs
  more than the bound. The owner may weigh A4's retention against the stall instead; the rule does not.
- **FAIL (no reading)** when a correctness clause fails: the cause is quoted and the arm is revised under a new
  pre-registration before the decide-by, or recorded as failed at the decide-by.

The 5090 reading decides the 5090 class only; the target card's decides the target class (the per-hardware rule in
`CLAUDE.md`). Neither card's reading is compared with the other's, and no timing crosses cards or boxes. Lead ruling
7's 8k cycles control is revisited only if the owner promotes the door, as ruling 7 says; it is not a clause here.

### 1.8 Order of work

1. Stage 0 probe, 5090 (this rig, `/tmp/memra-5090.lock`).
2. The mechanism in `memra-kv` and the engine seam, with the CPU tests.
3. The server wiring, the owed-growth reduction, receipts, the fault door, the census tests.
4. The grow case in `kv-tier-gate`.
5. 5090 cells: A2, A1, A5, A6, A7, A3, A4, in that order, each under the lock.
6. The target-card sitting: stage 0 on the card, then A2, A1, A5, A6, A7, A3, A4. Prepared in full (scripts
   committed and dry-checked, the expected duration and the box needs) before the lane reports `NEED TARGET CARD`.

### 1.9 Failures

Causes are quoted from captured stderr, never inferred. An OOM is a captured line plus `nvidia-smi` compute-apps at
the time. A rerun happens only as a whole cell under a new name, with the reason in section 2.

### 1.10 Addendum A (2026-09-24, after stage 0 on the 5090, before any mapper code)

Stage 0 on the 5090 (2.1) read `GROW-PLACEMENT extent=1 busy_p95_sum_us=11330.8 blocks_behind_queue=no rule p95<=100
-> helper` and `RELEASE-PLACEMENT unmap_or_release_blocks_behind_queue=no blocks_owner_from_helper=no -> owner-tick`.
The rules of 1.5 therefore select, for the 5090 class, helper-thread grows and owner-tick reaps. Section 1.1 to 1.9 is
unchanged; this addendum fixes the helper's design, which 1.5 named but did not specify:

- **The mapper.** One mapper thread per CUDA context, owned by `memra-kv`, started at the first on-demand plane of that
  context. Each on-demand plane's state (its extents and a `want` target) sits behind one lock shared by the plane and
  the mapper.
- **Ensure points under the helper placement.** At an ensure point the owner sets `want = need + 2 granules` (capped at
  the reservation). If the backed prefix already covers `need`, the owner returns at once and hands the plane to the
  mapper when the prefix is short of `want`. If it does not (the mapper is behind or failed), the owner maps the
  missing extent itself, inline; that grow's `owner_us` is the ensure point's wait of A3 (i), and its receipt line
  carries `waited=1`.
- **Mapper grows.** The mapper takes the lock, maps one extent up to `want` (create, map, access), enqueues its zero
  fill on the owner stream with the raw driver call, publishes the new backed prefix, and releases the lock. The owner
  reads the backed prefix under the same lock before it launches work on those rows, so every later kernel is ordered
  after the fill in stream order.
- **Releases stay ordered after the mapper.** A park trim sets `want` to the kept bytes under the lock before it
  records the release event; a drop marks the plane dead under the lock before it records the graveyard event; the
  mapper skips dead planes and never maps past `want`. So every release event is recorded after the mapper's last
  enqueue on that plane.
- **Construction** maps its initial extent inline (it is the allocation; its cost is inside A3 (ii)'s TTFT).
- **Per device class.** Stage 0 on the target card selects that class's placement by the same rule. If it reads
  inline, the target class grows inline and the mapper does not start there. The boot line names the placement:
  `[kv-vmm] door=ON routes=spec,plain granularity=<g> grow=helper|inline reap=owner-tick`.
- **The fault door, refined before code.** `MEMRA_KV_VMM_FAULT` is a comma list of `build:<n>` (the n-th construction
  grow fails once), `ensure:<n>` (the n-th owner grow at an ensure point fails once) and `mapper:all` (every mapper
  grow fails, so the owner is always behind). A5 runs `mapper:all` alone (the owner-behind path must meet A1 on the
  same mix) and `mapper:all,ensure:<n>` (an ensure-point failure: the registered outcome), and `build:<n>` (an
  admission-time failure: the reclaim-retry line, then the admission refusal the pooled allocator gives). A5's
  `grow:<n>` form in 1.6 reads as `ensure:<n>`.

### 1.11 Addendum B (2026-09-24, before any A1 to A7 run, with the code at its first commit)

Section 1.1 to 1.10 is unchanged. This fixes the instruments and parameters 1.6 left open, and a file name:

- **A3 (ii)'s instrument.** The day-26 mix client and the day-31/33 clients are non-streaming, so they cannot read
  TTFT, ITL or TPOT. A3 (ii) is read from `day37-stream-client.py` (streaming, `stream_options.include_usage`): per
  boot 32 greedy requests over 5000-char slices of `docs/SERVING.md`, 8 in flight on the 5090 and 16 on the target
  card, `max_tokens` alternating 256 and 2600 (a 2600-token request crosses at least one granule boundary of every
  plane on both models), 250 ms telemetry. Same-window interleaved: per order 5 boot pairs (O1 pooled first, O2 vmm
  first), one binary; each boot prints `DAY37 STREAM ...`. The A3 (ii) ratios use the per-arm medians of the per-boot
  readings over the 10 boots of each arm (N = 10 per arm, 5 per order), and are also stated per order. The mix and
  burst boots contribute E2E and throughput as readings, outside the bound. The stream rows' text digests join A1:
  equal across arms, row for row.
- **The file name.** 1.6's `day34-client.py` is a slip: the day-34 cell's client is `day31-client.py` through
  `day31-order.sh` (DAY34.md 1.3). The A7 burst boots use `day33-client.py --chars 5000 --skip-long --burst 64` at
  open output 8192 (the day-35 `G2`) and 32768 (`L64`) with `MEMRA_ADMIT_BY_MEMORY=1`, and the same client with the
  admission door off (`--burst 64`), one boot per arm and shape; `day33-compare.py`'s G-NOOM and G-BOOK terms read
  them unchanged.
- **A5's fault lists.** `mapper:all` alone on the spec mix (the owner-behind path must meet A1 on the mix);
  `mapper:all,ensure:1,ensure:2` on `G2` (the first owner ensure grow and its retry both fail); `build:1` on `G2` (one
  construction grow fails: the retry line, then the request succeeds); `build:1` to `build:64` on `G2` (every early
  construction fails: the admission refusal). Reachability, stated before the run: the tick-top bound covers the
  prompt from construction, so an ensure grow first happens at a granule crossing during decode, after the session has
  emitted; the ensure-point outcome is therefore the typed error, and the park branch is reachable only through
  admission, where a failure takes the `cache alloc failed` refusal. A5 reads what each arm reaches.
- **A4's receipt.** Every retire of a session with on-demand planes prints `[kv-vmm] retire id=... planes=..
  mapped=.. reserved=.. used=.. slack_bytes=.. booked=..`, where `used` is the rows committed times each plane's row
  bytes and `slack_bytes` is 64 rows of each plane. A4's bound per request is `mapped - used <= planes x granularity +
  slack_bytes`, rounded up to whole granules.
- **A2's command.** `kv-tier-gate --case grow --kv-allocator vmm-ondemand --context 32768 --tiers host
  --same-program` with the 27B artifact on both cards (the 5090 ran it at 32k on days 11 and 12), through
  `tools/tier-battery.py`.
- **A6's baseline binary.** `memra-server` built from main `17dceb981` in a detached worktree.

### 1.12 Addendum C (2026-09-24, before any serving boot, after the first gate set)

Found by reading design v1's receipts, not by any clause: every on-demand plane backs at least one whole granule.
On the vmm arm of the first 5090 gate set (`gates-vmm/hit-off/`), a bounded gate request retired with
`planes=18 mapped=37748736 reserved=37748736 used=2301440`: 36 MiB backed for 2.3 MB of rows, where the pooled
allocator holds the request's exact capacity. A short bounded request therefore costs more device memory on design v1
than on the pooled allocator, and a burst of them costs it once per session. Measuring the allocator at its best
needs the rule below; no clause and no bound of 1.6 changes.

- **The rule.** Under the on-demand scope a K/V plane is an on-demand plane only when its pooled capacity leaves at
  least one whole granule unbacked after the initial extent: `capacity >= whole_granules(initial) + granularity`.
  Otherwise the plane is allocated pooled, exactly as today. Below that line the on-demand plane can never back fewer
  bytes than the pooled one (it maps whole granules up to a reservation rounded above the capacity), so the pooled
  plane is never worse there. A session may hold both kinds (for example a K plane on demand and its smaller V plane
  pooled); the ensure visitor touches only on-demand planes, and the construction census counts only them.
- **The granularity** the rule reads is the device's VMM granularity, queried once per device and cached.
- **What reruns.** Design v1's receipts stay banked as they read: stage 0, the A2 series `grow-32768` and the first
  gate set (`gates-pooled`, `gates-vmm`, including the 9B twin cells that refused before any verdict, 2.2). The
  deciding cell runs on the revised binary: A2 again as `grow-32768-r2`, the gate set of both arms again as
  `gates-r2-pooled` and `gates-r2-vmm` (with the twin gate on the 27B), then the serving boots of addendum B
  (superseded by addendum D's r3 before any r2 cell ran).

### 1.13 Addendum D (2026-09-24, from the spill review patterns, before any cell on the revised binary)

Reading the day-37 state machines against the review patterns (1.6's CPU paragraph) found two defects of the same
shape, "parked work defeats the idle wait", before any addendum-C cell ran (`chain-r2` was stopped while A2 r2 still
waited on the rig lock; `grow-32768-r2-lockwait/` holds its two refused lock attempts):

- **The idle block.** With nothing active and nothing queued, the run loop blocks in `rx.recv()` with no timeout. The
  reap runs at the tick top, so the releases scheduled by the last retires (their graves) and by the last parks (their
  trims) stay backed until the next request arrives: device memory held at idle, which is what A4 reads. Fix: the
  owner-tick reap reports what is still pending; while anything is, the loop never enters the indefinite block and
  polls at most every 2 ms (the host tier's `Demoting`/`Promoting` precedent), so the pending releases land at idle.
- **The two reclaim paths.** The step-OOM teardown (memra#145's reclaim before retry) and the admin trim drop the parked
  pools, then trim the device pools back to the driver. Under the door the dropped on-demand planes sit in the
  graveyard until a reap, so neither path hands their bytes back. Fix: both reap the graveyard (waiting on the
  graves' own fences, the grow-failure path's form) right after the drops, before the pool trim, with a
  `[kv-vmm] reap (<why>)` line.

No clause or bound of 1.6 changes. The deciding cell runs on the binary with both fixes (r3): A2 as `grow-32768-r3`,
the gate set as `gates-r3-pooled` and `gates-r3-vmm`, then addendum B's serving boots.

## 2. Results

Written after the runs. Section 1 is unchanged.

### 2.1 Stage 0 on the local RTX 5090 Laptop GPU (`rtx5090-day37/stage0/`)

Binary `3dc08994...8783100fb` (`vmm-call-cost`, built from `34e7f42f3` plus the uncommitted probe; committed with this
section), one run under `/tmp/memra-5090.lock` (acquired 11:57:46Z after lane A's hold; `compute-apps` header-only
before and after), 20 repetitions per cell, the busy queue 96 x 512 MiB memsets (one 0.626 ms, about 60 ms queued).
Regime (250 ms samples, 308 rows): 58 to 82 C, 58.5 to 175.3 W, SM 2445 to 2745 MHz. Granularity 2,097,152 B.

Readings (medians, microseconds; `probe.log` has every cell with p95 and max):

| op | idle, 1 granule | busy, 1 granule | idle, 64 granules | busy p95, 1 granule |
|---|---|---|---|---|
| create | 6.2 | 31.0 | 6.6 | 4058.0 |
| map | 0.3 | 2.3 | 0.3 | 4.0 |
| access | 14.6 | 61.0 | 227.8 | 7268.8 |
| zero enqueue | 1.7 | 1.7 | 1.7 | 2.3 |
| unmap | 9.7 | 44.8 | 16.4 | 2977.7 |
| release | 11.1 | 30.7 | 36.8 | 5321.6 |

- No op waits for the queued work: every busy median is 5 orders below the 59.7 to 59.9 ms still queued at the call
  (`behind_queue=no` on all 32 op and extent pairs).
- No op blocks the owner from a helper thread: the owner's small launch plus event reads 9 to 11 us while a helper
  call is in flight against 4.5 us with none (`owner_from_helper=no`, the rule's bound is 10 x).
- The busy regime has a long tail: `create` + `map` + `access` at one granule sums to a 11,330.8 us p95.

Branch lines, verbatim:

```
GROW-PLACEMENT extent=1 busy_p95_sum_us=11330.8 blocks_behind_queue=no rule p95<=100 -> helper
RELEASE-PLACEMENT unmap_or_release_blocks_behind_queue=no blocks_owner_from_helper=no -> owner-tick
```

For the 5090 class the rules select helper grows and owner-tick reaps (addendum A, 1.10).
