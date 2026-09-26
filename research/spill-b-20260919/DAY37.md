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

### 1.14 Addendum E (2026-09-24, from the spill review patterns, before any serving boot on r3)

Reading the r3 state machines against the review patterns again, before the target-card sitting as 1.6 asks, found
three defects and one receipt gap. The r3 gate set had run (both arms); no serving boot had. `chain-r3` holds its boots
through the lane's `STOP` file (never a signal), and the r3 receipts stay banked as they read.

- **E1, a release behind a live extent.** The pending releases of a trimmed plane are meant to be a tail, reaped tail
  first. A resume's ensure cancels the pending releases below its need, then raises `want` to the need plus the
  lookahead; when that lookahead reaches past the pending tail, the mapper maps a live extent behind it. The reap then
  stops at the live tail, so the pending extent is never released, its bytes read as pending for the plane's life,
  and the idle wait of addendum D polls every 2 ms for as long as the plane is parked. Reproduced against the fake
  driver before any fix (`a_lookahead_past_a_pending_tail_cancels_it_and_never_maps_behind_it`, red on r3: extents
  `(5, 1, pending), (6, 1, live)` in granules). Fix: an ensure cancels every pending release below the `want` it sets,
  not only below the need, and the mapper never maps behind a pending extent. The invariant (every pending extent
  lies at or past `want`, so pending extents are always a tail) is asserted by the test.
- **E2, the idle decision reads a stale flag.** Addendum D's `vmm_pending` is computed by the tick-top reap, before the
  tick's retires. The last retire of a burst parks (a trim) or drops (a grave) after that point, so the next loop
  iteration finds nothing active, reads the stale `false`, and blocks in `rx.recv()` with the release pending: the
  defect addendum D meant to close, in its most common case. Fix: the idle decision recomputes the pending bytes
  itself (the graveyard and both parked pools, no reap) right before it chooses between the indefinite block and the
  2 ms poll. The tick-top reap also reaps the active sessions' planes (a resumed plane's tail past its `want` is
  released while it runs; no consumer touches bytes past the live prefix).
- **E3, a failed reap stays pending forever.** A failed unmap or release leaves the extent, or the grave, pending, so
  the idle poll never ends. Fix: a failed unmap cancels every pending release of that plane (the extents stay mapped
  and live, the plane keeps the bytes until it drops); a failed release after a successful unmap pops the extent and
  leaks its handle; a grave whose reap fails is quarantined out of the graveyard (leaked, never recycled). Each prints
  one `[kv-vmm] quarantined ...` line and adds to a `kv_vmm_quarantined_bytes` metric. No failure path leaves bytes
  counted as pending.
- **E4, the ensure-walls receipt drops its last batch.** `[kv-vmm] ensure-walls` prints every 256 walls, so up to
  255 walls of a boot are never printed and A3 (i) reads a truncated population. Fix: the partial batch also prints
  when the worker goes idle.
- **E5, the lookahead is one granule, not two.** Addendum A set `want = need + 2 granules` without a derivation. The
  lookahead only has to hide one mapper grow: stage 0's busy p95 for create, map and access is 11.3 ms per granule,
  and 17 planes cross together (2.2), so about 0.2 s, while a plane consumes one granule over about 1,000 to 1,900
  rows. One granule ahead hides the grow with that margin; the second granule is retained device memory on every
  on-demand plane of every active session (on the 27B's 34 planes, 68 MiB per session). E5 makes it one granule. A3
  (i) measures whether the owner ever waits because of it; the fault arm `mapper:all` still forces the owner-behind
  path.
- **The reader** reads A4's granule from the boot line's `granularity=` instead of a 2 MiB literal.
- **A4's bound, stated.** A4's per-request bound (1.6, 1.11) was registered before addendum A's lookahead, so under the
  helper placement the backed bytes at retire include the lookahead granule, which the bound counts as excess. A4 is
  a benefit reading outside the decision rule (1.7); its bound does not change and is read as it reads.

No clause, bound or rule of 1.6 and 1.7 changes. The deciding cell runs on the binary with E1 to E5 (r4), from a fresh
receipt root `rtx5090-day37/r4/`: A2 as `grow-32768-r4`, the gate set as `gates-r4-pooled` and `gates-r4-vmm`, then
addendum B's serving boots. The target-card sitting runs the same binary source. The r4 commit is also the lane tip at
DAY38's and DAY39's first boots, so their green binaries are built from it (their red arms apply their own patches to
it); both days' pre-registrations name the lane tip at the first boot, so neither changes.

### 1.15 Addendum F (2026-09-25, after the target-card sitting, before any rerun or reader change)

Section 2.5 placed four no-reading lines of the target card on the lane's harness and reader. This addendum fixes
them, names the rerun, and changes no clause, bound or rule of 1.6 and 1.7.

- **The box preflight.** The target-card chains refuse to start unless `ss` or `lsof` is on `PATH` (the gates' own
  port check needs one), and the box needs name iproute2.
- **The `main` arm clears every lane-only name.** `boots.sh` runs the main binary with `MEMRA_KV_ALLOCATOR`,
  `MEMRA_KV_VMM_GROW` and `MEMRA_KV_VMM_FAULT` unset, whatever the sitting exported.
- **Reader corrections (`day37-read.py`), all in how a clause is read, none in a bound:**
  1. A3 (i) reads each vmm boot's placement from its boot line (`grow=inline` or `grow=helper`). Inline: per grow event
     `owner_us` over the stream vmm boots, N >= 20, p99 <= 500 (1.6's inline branch). Helper: the ensure walls and the
     owner grows that waited, as before.
  2. A5-MAPPER under the inline placement prints `N/A (inline: no mapper)` with the digest comparison: the fault
     injects nothing there and the owner-behind path it targets is every grow, which A1 reads. Helper: as before.
  3. A5-ENSURE compares only rows whose `prompt_sha256` is equal in both boots, prints how many rows it excluded for a
     changed prompt (a later turn of a conversation whose earlier turn took the injected error carries that turn's
     different answer, so it is a different request), and prints the outcomes whole.
  4. A5-MAPPER's and A6's pooled side is `off-lane` when that boot exists (the lane binary, door unset, the mix).
- **The rerun on a target card (the r4 source `c6f9282c2`, `MEMRA_KV_VMM_GROW=inline` pinned from the class's stage-0
  receipt; stage 0 runs again as a recorded reading):** the gate set of both arms (`gates-r4b-pooled`,
  `gates-r4b-vmm`), then the boots `off-lane`, `fault-mapper`, `off-main`, `burst-g2-vmm`, `fault-ensure`, under the
  corrected reader. The target card's rule reading of 1.7 is then the first sitting's A2, A1-MIX, A1-STREAM, A3, A4,
  A5-BUILD and A7 lines with the rerun's A1-GATE, A5-MAPPER, A5-ENSURE and A6 lines, all from the r4 source; no timing
  crosses the two boxes (A3 is the first sitting's alone).
- **The 5090's r4 root** is read with the corrected reader. Its first boot batch stopped at `burst-l64-vmm` on the
  idle wait (`rig not idle after 7200 s`, a foreign process on the card), so `burst-l64-vmm`, `burst-l64-pooled`,
  `burst-boff-pooled`, `burst-boff-vmm`, `fault-ensure`, `fault-build1` and `fault-build64` run after the chain on the
  same binaries.
- **The target class's placement default** moves to inline in code (`kv_vmm_placement_for_device`, the stage-0 rule's
  selection on this class, as addendum A said it would after the sitting). The deciding cells keep the pinned env.

### 1.16 Addendum G (2026-09-26, the repro of 2.7's A1 cause, before it runs)

2.7 reads A1-GATE `admit-mem-burst` FAIL on the 5090's pooled arm (7 parked prefill OOMs) and left the cause unplaced.
The receipts point at one: every `[admit-mem]` admit line of that gate reads `pending_prime=0` while
`pending_prime_v1` grows to 21.7 GB, the signature of DAY39's `v2` term, which read red on the target card the same way
(`peak_pending_prime=0`, 10 parked prefill OOMs; DAY39 2.x). The r4 source `c6f9282c2` carries DAY39's first revision
(`6262506fc`, the shared slab booked once) and not its addendum B (`be2177ead`, the checkpoint snapshots and the
call's returned rows). The repro, on the 5090 under `/tmp/memra-5090.lock`, `tools/admit-mem-burst-gate.sh` at its
defaults (open 8192, burst 64, ctx 65536), door unset for the allocator (pooled):

- `r4` (`target/day37/r4/memra-server`) twice, and `v3` (`target/b2/v3/memra-server`, `a803d3080`, with addendum B) once.
- Reading: the gate's verdict and `AMB no prefill OOM` line per run, and the peak `pending_prime` against
  `pending_prime_v1`. If `r4` reds and `v3` greens, the cause is placed on the missing addendum-B terms (the door's
  booking, not the allocator), and the 5090 class's O1 rule reading stays FAIL (no reading) on the r4 tree as it read;
  a rerun of O1's 5090 cell on a tree with addendum B is then owed. If `r4` greens on both runs, the cause is not
  placed and says so. No clause of 1.6 or 1.7 changes.

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

### 2.2 Design v1's banked receipts and the notes of the day (no rule, arm, value or reader changed)

- **A2 on design v1** (`rtx5090-day37/grow-32768/`, gate binary `00cbfe16...a53cfa6` from `7e4ac87e4`, collector
  12:56:07 to 13:20:57Z, 5,885 samples at 250 ms, 56 to 87 C, 28.1 to 174.6 W), verbatim:
  `GROW-G1 PASS (grows=28 unequal=0 planes=34 planes_crossed=34 min_crossings_per_plane=12 rule>=5 drift=0
  tokens_equal=true logits_equal=true prefix_state_equal=true final_state_equal=true) committed=32768 generated=128`.
  The on-demand cache backed 71,303,168 B at construction (34 planes, one granule each) against 1,105,199,104 B
  reserved, grew in 28 steps of 17 extents (35,651,584 B each; every K crossing at the same row on all 17 layers,
  every V crossing likewise), and driver free returned to 8,849,260,544 B after the reap, the byte it read before the
  cache was built.
- **The first gate set on design v1** (`gates-pooled/`, `gates-vmm/`, server `2ec0b24e...0139b747`): both arms ALL
  GREEN on serve-smoke, identity default ON, fault default and plain (160 ok each), hit OFF and ON (61 and 68 ok),
  the admit-mem burst and spec-ctx-edge; `door=ON` on every vmm cell's server logs and `door=OFF` on every pooled one.
  The twin gate on the 9B refused on both pooled cells before any verdict, verbatim `REFUSED: cohort promotion did not
  happen for 2800 tokens: second send cached=2800 of 2800, published 2784`: its pressure preconditions are sized for
  the 27B, which days 17 to 29 ran as `twin27`. The vmm arm then ran it on the 27B (`twin27-off`, `twin27-on`, both
  `-> PASS`). From the r3 set on, the twin gate runs on the 27B in both arms.
- **serve-smoke's server log** lives in `/tmp/serve-smoke.log` (its own rule), so the first pooled cell's door count
  read 0/0; the log was copied into the cell and recounted (`door_off=1`), and the script copies it from then on.
- **Addendum C** (1.12) came from the vmm gate set's retire lines, addendum D (1.13) from the review patterns. Neither
  changed a clause; each moved the deciding cell to a new binary before any serving boot ran. `chain-r2` was stopped
  by the lane during its first lock wait (lane A held the card), before any cell ran; its two refused attempts are
  `grow-32768-r2-lockwait/`.
- **A review fix before any run:** the first park-trim implementation recorded its release event before taking the
  plane lock, so a mapper fill enqueued in between could still be in flight at the reap. The fence is now recorded
  by each plane under its own lock, after `want` is pulled back (commit `7e4ac87e4` carries the fixed form; the fix
  is also the census test `vmm_owed_growth_joins_the_booked_reduction_and_both_parks_trim`).
- **The reader** first read the door counts per line and so read every A1-GATE line FAIL; the fixed reader reads
  CELL.txt's one-line form (`e76c9bdbd`). No verdict of the fixed reader has been seen before the r3 cells.

### 2.3 The r3 receipts, banked (addenda C and D binary; superseded by r4 before any serving boot)

Binaries: server `c4f3f928...` and gate `380052d0...` (`r3-binaries.sha256`, source `d5923ccae`). `chain-r3` ran
14:04:09 to 15:14:34Z under `/tmp/memra-5090.lock`: A2, then the gate set of both arms; its boots found the lane's
`STOP` file (addendum E, 1.14) and ran nothing (`run.log`). The reader's lines (`read-r3.log`), verbatim:

```
DAY37 A2 card=rtx5090 receipt=grow-32768-r3/receipt/GROW.txt GROW-G1 PASS (grows=28 unequal=0 planes=34 planes_crossed=34 min_crossings_per_plane=12 rule>=5 drift=0 tokens_equal=true logits_equal=true prefix_state_equal=true final_state_equal=true) committed=32768 generated=128
DAY37 A1-GATE card=rtx5090 cell=admit-mem-burst pooled=[rc=0 ALL GREEN] vmm=[rc=0 ALL GREEN] ok_lines=0/0 door_on/off pooled=0/1 vmm=1/0 -> PASS
DAY37 A1-GATE card=rtx5090 cell=fault-default pooled=[rc=0 ALL GREEN] vmm=[rc=0 ALL GREEN] ok_lines=160/160 door_on/off pooled=0/14 vmm=14/0 -> PASS
DAY37 A1-GATE card=rtx5090 cell=fault-plain pooled=[rc=0 ALL GREEN] vmm=[rc=0 ALL GREEN] ok_lines=160/160 door_on/off pooled=0/14 vmm=14/0 -> PASS
DAY37 A1-GATE card=rtx5090 cell=hit-off pooled=[rc=0 ALL GREEN (qwen)] vmm=[rc=0 ALL GREEN (qwen)] ok_lines=61/61 door_on/off pooled=0/2 vmm=2/0 -> PASS
DAY37 A1-GATE card=rtx5090 cell=hit-on pooled=[rc=0 ALL GREEN (qwen)] vmm=[rc=0 ALL GREEN (qwen)] ok_lines=68/68 door_on/off pooled=0/2 vmm=2/0 -> PASS
DAY37 A1-GATE card=rtx5090 cell=identity-default-on pooled=[rc=0 ALL GREEN (teeth=0)] vmm=[rc=0 ALL GREEN (teeth=0)] ok_lines=12/12 door_on/off pooled=0/2 vmm=2/0 -> PASS
DAY37 A1-GATE card=rtx5090 cell=serve-smoke pooled=[rc=0 serve-smoke: 0 failed] vmm=[rc=0 serve-smoke: 0 failed] ok_lines=31/31 door_on/off pooled=0/1 vmm=1/0 -> PASS
DAY37 A1-GATE card=rtx5090 cell=spec-ctx-edge pooled=[rc=0 ALL GREEN] vmm=[rc=0 ALL GREEN] ok_lines=0/0 door_on/off pooled=0/3 vmm=3/0 -> PASS
DAY37 A1-GATE card=rtx5090 cell=twin27-off pooled=[rc=0 PREFIX-NEWEST-TURN-FITS: budget_bytes=1073741824 cohort_bytes=736755712 turns=8 cold_turns] vmm=[rc=0 PREFIX-NEWEST-TURN-FITS: budget_bytes=1073741824 cohort_bytes=736755712 turns=8 cold_turns] ok_lines=0/0 door_on/off pooled=0/2 vmm=2/0 -> PASS
DAY37 A1-GATE card=rtx5090 cell=twin27-on pooled=[rc=0 PREFIX-NEWEST-TURN-FITS: budget_bytes=1073741824 cohort_bytes=736755712 turns=8 cold_turns] vmm=[rc=0 PREFIX-NEWEST-TURN-FITS: budget_bytes=1073741824 cohort_bytes=736755712 turns=8 cold_turns] ok_lines=0/0 door_on/off pooled=0/2 vmm=2/0 -> PASS
DAY37 A4 card=rtx5090 retires=0 within_bound=0 worst_over_bound_bytes=0 rule mapped-used<=planes*granule+slack -> FAIL
```

The A4 line reads FAIL because no serving boot ran (`retires=0`); it is no reading. The reader also listed design v1's
`grow-32768` line, omitted above (2.2 quotes it). The deciding cell is r4 (1.14).

### 2.4 The review reading (1.6's CPU paragraph), on the r4 tree, before the target-card sitting

The seven spill review patterns, read against every state machine this lane added for O1 (the on-demand plane, the
mapper, the graveyard, the worker's ensure, trim, reap and idle paths) and against the O2 and O5 changes:

- **Move-then-match.** No let-else moves a field on these paths: `KvPlane::drop` takes `on_demand` and buries it;
  the park trims and the ensure visitor borrow. None found.
- **Parked work defeats the idle wait.** Two found, both fixed in addendum E: the idle decision read a pending flag
  computed before the tick's retires (E2), and a release stuck behind a live extent kept the 2 ms poll alive for a
  parked plane's life (E1). The step-OOM park back to the queue after a failed ensure is bounded by the step-OOM retry
  budget, as the spec phase's is.
- **Gate literals that rot.** The reader's 2 MiB granule literal now reads the boot line (addendum E). The census
  tests match code text, not counts of a run, except the two that pin the number of ensure and refresh sites, which
  are the point of those tests.
- **Release only on the happy path.** Found and fixed in E3: a failed unmap, release or grave reap left bytes pending
  forever. `map_extent` undoes each driver step on failure; a construction census refusal drops the cache, whose
  planes go to the graveyard; `bury` quarantines when no fence can be recorded.
- **Borrowed sources settled at the seam that moves them.** A trim records its fence under the plane lock after
  `want` is pulled back (the `7e4ac87e4` fix, 2.2); `bury` marks the plane dead under the lock before the grave's
  fence; the mapper upgrades a weak reference and re-checks `dead` under the same lock.
- **A booking of a future insert into a budgeted cache.** Owed growth is `reserved - backed` per active session: the
  footprint the pooled gate already charges at admission, with no eviction inside a plane to cap it. O5's term books
  the prime slab's growth, and the slab is grow-only with no eviction, so there is no budget left to cap at; the
  seed term keeps its day-35 cap.
- **A guard that hands memory back checks the reply is its own.** No reply-carrying guard on these paths: the mapper
  is fire-and-forget on a weak reference, and every state change it makes is under the plane's lock.

O2's errored flag (DAY38 addendum A) is set before every session-error send the census test pairs, and O5's term is
computed only with the door armed (its census test pins both).

### 2.5 The target-card sitting (r4; one RTX PRO 6000 Blackwell Workstation Edition; 2026-09-24)

`pro-single-b-sitting.sh` ran the day-37 half 16:41:55 to 19:04:12Z (`pro-single-day37/box/chain.log`): lane built from
`c6f9282c2` (server `3dc05d17...`), main from `17dceb981` (`e0da013e...`), receipts mirrored by the lead and checked file
for file against the box manifest. Stage 0 on the card read, verbatim:

```
GROW-PLACEMENT extent=1 busy_p95_sum_us=55.3 blocks_behind_queue=no rule p95<=100 -> inline
RELEASE-PLACEMENT unmap_or_release_blocks_behind_queue=no blocks_owner_from_helper=no -> owner-tick
```

So the target class grows inline (addendum A: the mapper does not start there) and reaps at the owner tick; the sitting
exported `MEMRA_KV_VMM_GROW=inline`, and every vmm boot line reads `grow=inline (MEMRA_KV_VMM_GROW)`. The reader's
lines (`pro-single-day37/box/SUMMARY.txt`), verbatim, the eight A4-IDLE readings listed after:

```
DAY37 A2 card=pro6000 receipt=grow-32768-r4/receipt/GROW.txt GROW-G1 PASS (grows=28 unequal=0 planes=34 planes_crossed=34 min_crossings_per_plane=12 rule>=5 drift=0 tokens_equal=true logits_equal=true prefix_state_equal=true final_state_equal=true) committed=32768 generated=128
DAY37 A1-GATE card=pro6000 cell=admit-mem-burst pooled=[rc=2 no-verdict] vmm=[rc=2 no-verdict] ok_lines=0/0 door_on/off pooled=0/0 vmm=0/0 -> FAIL
DAY37 A1-GATE card=pro6000 cell=fault-default pooled=[rc=1 no-verdict] vmm=[rc=1 no-verdict] ok_lines=0/0 door_on/off pooled=0/0 vmm=0/0 -> FAIL
DAY37 A1-GATE card=pro6000 cell=fault-plain pooled=[rc=1 no-verdict] vmm=[rc=1 no-verdict] ok_lines=0/0 door_on/off pooled=0/0 vmm=0/0 -> FAIL
DAY37 A1-GATE card=pro6000 cell=hit-off pooled=[rc=1 no-verdict] vmm=[rc=1 no-verdict] ok_lines=0/0 door_on/off pooled=0/0 vmm=0/0 -> FAIL
DAY37 A1-GATE card=pro6000 cell=hit-on pooled=[rc=1 no-verdict] vmm=[rc=1 no-verdict] ok_lines=0/0 door_on/off pooled=0/0 vmm=0/0 -> FAIL
DAY37 A1-GATE card=pro6000 cell=identity-default-on pooled=[rc=1 no-verdict] vmm=[rc=1 no-verdict] ok_lines=0/0 door_on/off pooled=0/0 vmm=0/0 -> FAIL
DAY37 A1-GATE card=pro6000 cell=serve-smoke pooled=[rc=1 no-verdict] vmm=[rc=1 no-verdict] ok_lines=0/0 door_on/off pooled=?/? vmm=?/? -> FAIL
DAY37 A1-GATE card=pro6000 cell=spec-ctx-edge pooled=[rc=2 no-verdict] vmm=[rc=2 no-verdict] ok_lines=0/0 door_on/off pooled=0/0 vmm=0/0 -> FAIL
DAY37 A1-GATE card=pro6000 cell=twin27-off pooled=[rc=0 PREFIX-NEWEST-TURN-FITS: budget_bytes=1073741824 cohort_bytes=736755712 turns=8 cold_turns] vmm=[rc=0 PREFIX-NEWEST-TURN-FITS: budget_bytes=1073741824 cohort_bytes=736755712 turns=8 cold_turns] ok_lines=0/0 door_on/off pooled=0/2 vmm=2/0 -> PASS
DAY37 A1-GATE card=pro6000 cell=twin27-on pooled=[rc=0 PREFIX-NEWEST-TURN-FITS: budget_bytes=1073741824 cohort_bytes=736755712 turns=8 cold_turns] vmm=[rc=0 PREFIX-NEWEST-TURN-FITS: budget_bytes=1073741824 cohort_bytes=736755712 turns=8 cold_turns] ok_lines=0/0 door_on/off pooled=0/2 vmm=2/0 -> PASS
DAY37 A1-MIX card=pro6000 kind=spec order=O1 compared=45 equal=45 differ=0 excluded=0 differ_tags=[] faults_pooled=0 faults_vmm=0 door_on_vmm=1 -> PASS
DAY37 A1-MIX card=pro6000 kind=spec order=O2 compared=45 equal=45 differ=0 excluded=0 differ_tags=[] faults_pooled=0 faults_vmm=0 door_on_vmm=1 -> PASS
DAY37 A1-MIX card=pro6000 kind=plain order=O1 compared=45 equal=45 differ=0 excluded=0 differ_tags=[] faults_pooled=0 faults_vmm=0 door_on_vmm=1 -> PASS
DAY37 A1-MIX card=pro6000 kind=plain order=O2 compared=45 equal=45 differ=0 excluded=0 differ_tags=[] faults_pooled=0 faults_vmm=0 door_on_vmm=1 -> PASS
DAY37 A1-STREAM card=pro6000 pairs=10 differing_rows=0 -> PASS
DAY37 A3-ii card=pro6000 order=both ttft_p50 pooled=3603.56(N=10) vmm=3686.06(N=10) ratio=1.0229 rule<=1.05 ok; itl_p99 pooled=38.46(N=10) vmm=38.45(N=10) ratio=0.9999 rule<=1.05 ok; tpot_p50 pooled=35.47(N=10) vmm=35.03(N=10) ratio=0.9877 rule<=1.02 ok; out_tok_per_s pooled=364.94(N=10) vmm=364.88(N=10) ratio=0.9998 rule>=0.98 ok -> PASS
DAY37 A3-ii card=pro6000 order=O1 ttft_p50 pooled=3526.49(N=5) vmm=3686.45(N=5) ratio=1.0454 rule<=1.05 ok; itl_p99 pooled=38.47(N=5) vmm=38.50(N=5) ratio=1.0008 rule<=1.05 ok; tpot_p50 pooled=35.70(N=5) vmm=35.02(N=5) ratio=0.9810 rule<=1.02 ok; out_tok_per_s pooled=365.00(N=5) vmm=364.92(N=5) ratio=0.9998 rule>=0.98 ok
DAY37 A3-ii card=pro6000 order=O2 ttft_p50 pooled=3683.23(N=5) vmm=3685.67(N=5) ratio=1.0007 rule<=1.05 ok; itl_p99 pooled=38.45(N=5) vmm=38.45(N=5) ratio=1.0000 rule<=1.05 ok; tpot_p50 pooled=35.28(N=5) vmm=35.04(N=5) ratio=0.9932 rule<=1.02 ok; out_tok_per_s pooled=364.77(N=5) vmm=364.87(N=5) ratio=1.0003 rule>=0.98 ok
DAY37 A3-i card=pro6000 placement=helper tick_ensure_walls N=33811 p50_us=7.0 p99_us=26.0 max_us=2602 owner_grows_waited N=0 p99_us=nan rule N>=20 p99<=500 -> PASS
DAY37 A4 card=pro6000 retires=488 within_bound=488 worst_over_bound_bytes=0 boots_without_granularity=[] rule mapped-used<=planes*granule+slack -> PASS
DAY37 A5-MAPPER card=pro6000 compared=45 equal=45 differ=0 owner_grows_waited=0 faults=0 -> FAIL
DAY37 A5-ENSURE card=pro6000 reclaim_retry_lines=1 outcomes=['n'] errored_rows=['i-L0-r0'] peers_compared=79 equal=78 differ=1 faults=0 -> FAIL
DAY37 A5-BUILD1 card=pro6000 retry_lines=11 cache_alloc_failed_lines=0 non200_rows=0 faults=0 -> PASS
DAY37 A5-BUILD64 card=pro6000 retry_lines=9 cache_alloc_failed_lines=50 non200_rows=50 faults=0 -> PASS
DAY37 A6 card=pro6000 lane_boot=mix-spec-O1-pooled compared=0 equal=0 differ=0 lane_kv_vmm_lines=1 main_kv_vmm_lines=0 -> FAIL
DAY37 A7 card=pro6000 shape=g2 oom_lines=0 r503=0 grow_failures=0 admit_lines=80 est_over_booked_free=0 status={200: 80} faults=0 -> PASS
DAY37 A7 card=pro6000 shape=l64 oom_lines=0 r503=0 grow_failures=0 admit_lines=66 est_over_booked_free=0 status={200: 66, 429: 14} faults=0 -> PASS
DAY37 A7 card=pro6000 shape=boff oom_lines=0 r503=0 grow_failures=0 admit_lines=0 est_over_booked_free=0 status={200: 80} faults=0 -> PASS
```

A4-IDLE (benefit readings, no bound; driver free, pool cached, on-demand mapped and reserved bytes at idle after each
mix boot):

```
DAY37 A4-IDLE card=pro6000 boot=mix-plain-O1-pooled driver_free=45749698560 pool_cached=8767078360 pool_reserved=55633248256 vmm_mapped=0 vmm_reserved=0 spec_pool=0 continuation_pool=2
DAY37 A4-IDLE card=pro6000 boot=mix-plain-O1-vmm driver_free=69797740544 pool_cached=512688632 pool_reserved=30836523008 vmm_mapped=499122176 vmm_reserved=16684941312 spec_pool=0 continuation_pool=2
DAY37 A4-IDLE card=pro6000 boot=mix-plain-O2-pooled driver_free=45749698560 pool_cached=8767078360 pool_reserved=55633248256 vmm_mapped=0 vmm_reserved=0 spec_pool=0 continuation_pool=2
DAY37 A4-IDLE card=pro6000 boot=mix-plain-O2-vmm driver_free=69797740544 pool_cached=512688632 pool_reserved=30836523008 vmm_mapped=499122176 vmm_reserved=16684941312 spec_pool=0 continuation_pool=2
DAY37 A4-IDLE card=pro6000 boot=mix-spec-O1-pooled driver_free=43702878208 pool_cached=9867833272 pool_reserved=57646514176 vmm_mapped=0 vmm_reserved=0 spec_pool=2 continuation_pool=0
DAY37 A4-IDLE card=pro6000 boot=mix-spec-O1-vmm driver_free=67818029056 pool_cached=1546334680 pool_reserved=32782680064 vmm_mapped=499122176 vmm_reserved=16684941312 spec_pool=2 continuation_pool=0
DAY37 A4-IDLE card=pro6000 boot=mix-spec-O2-pooled driver_free=43702878208 pool_cached=9867833272 pool_reserved=57646514176 vmm_mapped=0 vmm_reserved=0 spec_pool=2 continuation_pool=0
DAY37 A4-IDLE card=pro6000 boot=mix-spec-O2-vmm driver_free=67818029056 pool_cached=1546334680 pool_reserved=32782680064 vmm_mapped=499122176 vmm_reserved=16684941312 spec_pool=2 continuation_pool=0
```

Causes, placed from the cell logs before any reading:

- **A1-GATE, eight cells on both arms: no verdict.** Every one refused before booting a server, verbatim (serve-smoke):
  `serve-smoke: FAIL`, then `cannot observe listening sockets (no ss, no lsof), so this run cannot prove port 8177 is
  free.` (the two fragments of one line; the separator between them is omitted here).
  The identity, fault, hit, admit-mem-burst and spec-ctx-edge gates print the same refusal for their own ports. The box
  had neither `ss` (iproute2) nor `lsof`; the lane's box needs did not list them. The two twin27 cells (no port check)
  ran and PASS on both arms. The gate set has no reading on the target card: it reruns on a box with iproute2.
- **A6: no reading.** The main binary refused to boot, verbatim:
  `Error: "[env-audit] REFUSED: the environment names doors this build does not have:\n  MEMRA_KV_VMM_GROW: unknown MEMRA_KV_* name`.
  The sitting exported the lane-only `MEMRA_KV_VMM_GROW` to every boot, and `boots.sh`'s `main` arm did not clear it.
  A harness defect of the lane, not a finding about either binary; the 5090 never exported the name, so its `off-main`
  booted. A6 reruns with the `main` arm clearing every lane-only name.
- **A5-MAPPER reads FAIL on `owner_grows_waited=0`.** Under the inline placement there is no mapper, so `mapper:all`
  injects nothing and no grow can wait on one; the reader's `waited > 0` term presumes the helper placement. The mix
  under the fault arm read 45 of 45 digests equal.
- **A5-ENSURE reads FAIL on one differing row.** The injected failure ended `i-L0-r0` with the typed error (`ensure
  failed: not parked ... generated 362, streamed 362`, the registered outcome for a session that has emitted). The one
  differing row is `iii-L0-r0`, the same conversation continued: its prompt carries `i-L0-r0`'s answer, so its
  `prompt_sha256` differs between the two boots (`05519b6e...` against `bed810eb...`). It is a different request, not a
  peer. The reader compared digests of unequal prompts. Every row with an equal prompt reads equal (78 of 78). The
  `outcomes=['n']` field is the reader printing the first character of `not parked`.
- **A3-i prints `placement=helper` and reads the ensure walls.** The reader did not read the placement; the inline
  clause is per grow event `owner_us`. From the same receipts (stream vmm boots, `[kv-vmm] grow ... owner_us=`):
  N=4,420, p50 24 us, p99 139 us, max 1,999 us, `waited=1` on none. The walls read N=33,811, p99 26 us.

What stands as read: A2 GROW-G1 PASS; A1-MIX PASS on both paths and both orders (45 of 45 each); A1-STREAM PASS (10
pairs, 0 differing rows); A3 (ii) PASS in both orders (TTFT p50 ratio 1.0229, ITL p99 0.9999, TPOT p50 0.9877,
throughput 0.9998; N=10 per arm, medians of the per-boot values, 250 ms samples in each boot, 62 to 70 C at the boot
boundaries); A4 PASS (488 retires within the bound; under inline there is no lookahead granule); A5-BUILD1 and
A5-BUILD64 PASS; A7 PASS on g2, l64 and boff (no OOM line, no 503, no grow failure, every admit within the booked
reading).

The rule of 1.7 has no reading on the target card yet: A1 (the gate set) and A6 have no verdict, and A5's two lines
need the corrected reader. Addendum F (1.15) pre-registers the rerun and the reader corrections.

### 2.6 Addendum F's rerun on the target card (a second RTX PRO 6000 Blackwell Workstation Edition, 2026-09-25)

`pro-single-b-sitting2.sh` ran `rerun-f.sh` 01:45:33 to 02:35:56Z (`pro-single-day37/box-f/chain.log`): lane built from
`c6f9282c2` (server `7e586b8c...`), main from `17dceb981` (`92c1bf3b...`), `MEMRA_KV_VMM_GROW=inline` pinned, the box
preflight found `ss`. The corrected reader (`pro-single-day37/box-f/SUMMARY.txt`), verbatim:

```
DAY37 A1-GATE card=pro6000 cell=admit-mem-burst pooled=[rc=0 ALL GREEN] vmm=[rc=0 ALL GREEN] ok_lines=0/0 door_on/off pooled=0/1 vmm=1/0 -> PASS
DAY37 A1-GATE card=pro6000 cell=fault-default pooled=[rc=0 ALL GREEN] vmm=[rc=0 ALL GREEN] ok_lines=160/160 door_on/off pooled=0/14 vmm=14/0 -> PASS
DAY37 A1-GATE card=pro6000 cell=fault-plain pooled=[rc=0 ALL GREEN] vmm=[rc=0 ALL GREEN] ok_lines=160/160 door_on/off pooled=0/14 vmm=14/0 -> PASS
DAY37 A1-GATE card=pro6000 cell=hit-off pooled=[rc=0 ALL GREEN (qwen)] vmm=[rc=0 ALL GREEN (qwen)] ok_lines=61/61 door_on/off pooled=0/2 vmm=2/0 -> PASS
DAY37 A1-GATE card=pro6000 cell=hit-on pooled=[rc=0 ALL GREEN (qwen)] vmm=[rc=0 ALL GREEN (qwen)] ok_lines=68/68 door_on/off pooled=0/2 vmm=2/0 -> PASS
DAY37 A1-GATE card=pro6000 cell=identity-default-on pooled=[rc=0 ALL GREEN (teeth=0)] vmm=[rc=0 ALL GREEN (teeth=0)] ok_lines=12/12 door_on/off pooled=0/2 vmm=2/0 -> PASS
DAY37 A1-GATE card=pro6000 cell=serve-smoke pooled=[rc=0 serve-smoke: 0 failed] vmm=[rc=0 serve-smoke: 0 failed] ok_lines=31/31 door_on/off pooled=0/1 vmm=1/0 -> PASS
DAY37 A1-GATE card=pro6000 cell=spec-ctx-edge pooled=[rc=0 ALL GREEN] vmm=[rc=0 ALL GREEN] ok_lines=0/0 door_on/off pooled=0/3 vmm=3/0 -> PASS
DAY37 A1-GATE card=pro6000 cell=twin27-off pooled=[rc=0 PREFIX-NEWEST-TURN-FITS: budget_bytes=1073741824 cohort_bytes=736755712 turns=8 cold_turns] vmm=[rc=0 PREFIX-NEWEST-TURN-FITS: budget_bytes=1073741824 cohort_bytes=736755712 turns=8 cold_turns] ok_lines=0/0 door_on/off pooled=0/2 vmm=2/0 -> PASS
DAY37 A1-GATE card=pro6000 cell=twin27-on pooled=[rc=0 PREFIX-NEWEST-TURN-FITS: budget_bytes=1073741824 cohort_bytes=736755712 turns=8 cold_turns] vmm=[rc=0 PREFIX-NEWEST-TURN-FITS: budget_bytes=1073741824 cohort_bytes=736755712 turns=8 cold_turns] ok_lines=0/0 door_on/off pooled=0/2 vmm=2/0 -> PASS
DAY37 A4 card=pro6000 retires=74 within_bound=74 worst_over_bound_bytes=0 boots_without_granularity=[] rule mapped-used<=planes*granule+slack -> PASS
DAY37 A5-MAPPER card=pro6000 against=off-lane compared=45 equal=45 differ=0 owner_grows_waited=0 faults=0 -> N/A (inline: no mapper)
DAY37 A5-ENSURE card=pro6000 reclaim_retry_lines=1 outcomes=['not parked'] errored_rows=['i-L0-r0'] dependent_rows=['iii-L0-r0'] peers_compared=78 equal=78 differ=0 differ_tags=[] faults=0 -> PASS
DAY37 A6 card=pro6000 lane_boot=off-lane compared=45 equal=45 differ=0 lane_kv_vmm_lines=1 main_kv_vmm_lines=0 -> PASS
DAY37 A7 card=pro6000 shape=g2 oom_lines=0 r503=0 grow_failures=0 admit_lines=80 est_over_booked_free=0 status={200: 80} faults=0 -> PASS
```

Stage 0 on this box read, verbatim, `GROW-PLACEMENT extent=1 busy_p95_sum_us=271.2 blocks_behind_queue=no rule p95<=100
-> helper`, against 55.3 us (`-> inline`) on the first box of the same class: the probe's busy regime puts the two
boxes on opposite sides of the rule's 100 us line. The serving grows read nearly the same on both under inline:
per grow event p99 141 us (N=1,714, this box's vmm boots) and 139 us (N=4,420, the first box's stream boots), both
under A3 (i)'s 500 us bound. The class default in code (`inline`, `0b75283fe`) rests on the first box's receipt; the
two stage-0 readings disagree, which the owner weighs with the door.

**The rule of 1.7 on the target class**, from the r4 source across the two sittings (A3 timing from the first
sitting alone): A1 PASS (every gate cell on both arms here; A1-MIX and A1-STREAM in 2.5), A2 PASS (2.5), A5 PASS
(ENSURE, BUILD1 and BUILD64; MAPPER does not apply under inline, addendum F), A6 PASS, A7 PASS, A3 (i) PASS (inline
branch, 139 us) and A3 (ii) PASS. The rule reads **PROMOTE-ELIGIBLE** for the RTX PRO 6000 Blackwell Workstation
class. A4's retention readings go with it (2.5). The 5090 class has no reading: its r4 half stopped when the card
went to `GPU requires reset` (Xid 119, then 154, from 01:25Z; `rtx5090-day37/r4/run.log`), with the stream pairs
from `stream-O1-2` on and the seven boots of addendum F still to run.

### 2.7 The local RTX 5090 Laptop GPU, r4 (`rtx5090-day37/r4/`, the 9B at `MEMRA_CTX=65536`)

The gate set ran on 2026-09-24 (`gates-r4-pooled`, `gates-r4-vmm`, tree `826ff8d78`), A2 as `grow-32768-r4`, and the
boots on 2026-09-25 to 26 (queue-e after the card's reset; the seven boots addendum F named ran after the stream pairs,
on the same binaries). The two `burst-boff` server logs are committed gzipped (107 and 112 MB; their raw sha256 in
`r4/gzipped-logs.sha256`). The corrected reader's lines, verbatim (`r4/read.log`, A4-IDLE lines omitted):

```
DAY37 A2 card=rtx5090 receipt=grow-32768-r4/receipt/GROW.txt GROW-G1 PASS (grows=28 unequal=0 planes=34 planes_crossed=34 min_crossings_per_plane=12 rule>=5 drift=0 tokens_equal=true logits_equal=true prefix_state_equal=true final_state_equal=true) committed=32768 generated=128
DAY37 A1-GATE card=rtx5090 cell=admit-mem-burst pooled=[rc=1 no-verdict] vmm=[rc=0 ALL GREEN] ok_lines=0/0 door_on/off pooled=0/1 vmm=1/0 -> FAIL
DAY37 A1-GATE card=rtx5090 cell=fault-default pooled=[rc=0 ALL GREEN] vmm=[rc=0 ALL GREEN] ok_lines=160/160 door_on/off pooled=0/14 vmm=14/0 -> PASS
DAY37 A1-GATE card=rtx5090 cell=fault-plain pooled=[rc=0 ALL GREEN] vmm=[rc=0 ALL GREEN] ok_lines=160/160 door_on/off pooled=0/14 vmm=14/0 -> PASS
DAY37 A1-GATE card=rtx5090 cell=hit-off pooled=[rc=0 ALL GREEN (qwen)] vmm=[rc=0 ALL GREEN (qwen)] ok_lines=61/61 door_on/off pooled=0/2 vmm=2/0 -> PASS
DAY37 A1-GATE card=rtx5090 cell=hit-on pooled=[rc=0 ALL GREEN (qwen)] vmm=[rc=0 ALL GREEN (qwen)] ok_lines=68/68 door_on/off pooled=0/2 vmm=2/0 -> PASS
DAY37 A1-GATE card=rtx5090 cell=identity-default-on pooled=[rc=0 ALL GREEN (teeth=0)] vmm=[rc=0 ALL GREEN (teeth=0)] ok_lines=12/12 door_on/off pooled=0/2 vmm=2/0 -> PASS
DAY37 A1-GATE card=rtx5090 cell=serve-smoke pooled=[rc=0 serve-smoke: 0 failed] vmm=[rc=0 serve-smoke: 0 failed] ok_lines=31/31 door_on/off pooled=0/1 vmm=1/0 -> PASS
DAY37 A1-GATE card=rtx5090 cell=spec-ctx-edge pooled=[rc=0 ALL GREEN] vmm=[rc=0 ALL GREEN] ok_lines=0/0 door_on/off pooled=0/3 vmm=3/0 -> PASS
DAY37 A1-GATE card=rtx5090 cell=twin27-off pooled=[rc=0 PREFIX-NEWEST-TURN-FITS: budget_bytes=1073741824 cohort_bytes=736755712 turns=8 cold_turns] vmm=[rc=0 PREFIX-NEWEST-TURN-FITS: budget_bytes=1073741824 cohort_bytes=736755712 turns=8 cold_turns] ok_lines=0/0 door_on/off pooled=0/2 vmm=2/0 -> PASS
DAY37 A1-GATE card=rtx5090 cell=twin27-on pooled=[rc=0 PREFIX-NEWEST-TURN-FITS: budget_bytes=1073741824 cohort_bytes=736755712 turns=8 cold_turns] vmm=[rc=0 PREFIX-NEWEST-TURN-FITS: budget_bytes=1073741824 cohort_bytes=736755712 turns=8 cold_turns] ok_lines=0/0 door_on/off pooled=0/2 vmm=2/0 -> PASS
DAY37 A1-MIX card=rtx5090 kind=spec order=O1 compared=44 equal=44 differ=0 excluded=1 differ_tags=[] faults_pooled=0 faults_vmm=0 door_on_vmm=1 -> PASS
DAY37 A1-MIX card=rtx5090 kind=spec order=O2 compared=44 equal=44 differ=0 excluded=1 differ_tags=[] faults_pooled=0 faults_vmm=0 door_on_vmm=1 -> PASS
DAY37 A1-MIX card=rtx5090 kind=plain order=O1 compared=44 equal=44 differ=0 excluded=1 differ_tags=[] faults_pooled=0 faults_vmm=0 door_on_vmm=1 -> PASS
DAY37 A1-MIX card=rtx5090 kind=plain order=O2 compared=44 equal=44 differ=0 excluded=1 differ_tags=[] faults_pooled=0 faults_vmm=0 door_on_vmm=1 -> PASS
DAY37 A1-STREAM card=rtx5090 pairs=10 differing_rows=0 -> PASS
DAY37 A3-ii card=rtx5090 order=both ttft_p50 pooled=956.17(N=10) vmm=965.90(N=10) ratio=1.0102 rule<=1.05 ok; itl_p99 pooled=28.00(N=10) vmm=27.98(N=10) ratio=0.9993 rule<=1.05 ok; tpot_p50 pooled=24.46(N=10) vmm=24.41(N=10) ratio=0.9980 rule<=1.02 ok; out_tok_per_s pooled=280.48(N=10) vmm=279.49(N=10) ratio=0.9965 rule>=0.98 ok -> PASS
DAY37 A3-ii card=rtx5090 order=O1 ttft_p50 pooled=955.60(N=5) vmm=964.30(N=5) ratio=1.0091 rule<=1.05 ok; itl_p99 pooled=27.62(N=5) vmm=27.95(N=5) ratio=1.0119 rule<=1.05 ok; tpot_p50 pooled=24.40(N=5) vmm=24.06(N=5) ratio=0.9861 rule<=1.02 ok; out_tok_per_s pooled=281.76(N=5) vmm=280.78(N=5) ratio=0.9965 rule>=0.98 ok
DAY37 A3-ii card=rtx5090 order=O2 ttft_p50 pooled=960.86(N=5) vmm=967.50(N=5) ratio=1.0069 rule<=1.05 ok; itl_p99 pooled=28.56(N=5) vmm=28.00(N=5) ratio=0.9804 rule<=1.05 ok; tpot_p50 pooled=24.50(N=5) vmm=24.54(N=5) ratio=1.0016 rule<=1.02 ok; out_tok_per_s pooled=280.19(N=5) vmm=279.01(N=5) ratio=0.9958 rule>=0.98 ok
DAY37 A3-i card=rtx5090 placement=helper tick_ensure_walls N=65014 p50_us=2.0 p99_us=6.0 max_us=686 owner_grows_waited N=0 p99_us=nan rule N>=20 p99<=500 -> PASS
DAY37 A4 card=rtx5090 retires=438 within_bound=36 worst_over_bound_bytes=121175040 boots_without_granularity=[] rule mapped-used<=planes*granule+slack -> FAIL
DAY37 A5-MAPPER card=rtx5090 against=mix-spec-O1-pooled compared=44 equal=44 differ=0 owner_grows_waited=1070 faults=0 -> PASS
DAY37 A5-ENSURE card=rtx5090 reclaim_retry_lines=1 outcomes=['not parked'] errored_rows=['burst-b11', 'burst-b12', 'burst-b14', 'burst-b15', 'burst-b16', 'burst-b21'] dependent_rows=['iii-L0-r0'] peers_compared=44 equal=44 differ=0 differ_tags=[] faults=0 -> PASS
DAY37 A5-BUILD1 card=rtx5090 retry_lines=11 cache_alloc_failed_lines=0 non200_rows=19 faults=0 -> PASS
DAY37 A5-BUILD64 card=rtx5090 retry_lines=9 cache_alloc_failed_lines=52 non200_rows=52 faults=0 -> PASS
DAY37 A6 card=rtx5090 lane_boot=mix-spec-O1-pooled compared=44 equal=44 differ=0 lane_kv_vmm_lines=1 main_kv_vmm_lines=0 -> PASS
DAY37 A7 card=rtx5090 shape=g2 oom_lines=0 r503=0 grow_failures=0 admit_lines=61 est_over_booked_free=0 status={200: 61, 429: 19} faults=0 -> PASS
DAY37 A7 card=rtx5090 shape=l64 oom_lines=0 r503=0 grow_failures=0 admit_lines=35 est_over_booked_free=0 status={200: 35, 429: 45} faults=0 -> PASS
DAY37 A7 card=rtx5090 shape=boff oom_lines=0 r503=0 grow_failures=0 admit_lines=0 est_over_booked_free=0 status={200: 80} faults=0 -> PASS
```

- **A1-GATE `admit-mem-burst` FAIL on the pooled arm** (the control, today's allocator): `AMB no prefill OOM:
  oom_lines=7 status={200: 45, 429: 19} -> FAIL`, then `ADMIT-MEM BURST GATE: RED`. The seven lines read `[admit-mem]
  prefill OOM parked session back to queue (model amb, retry 1/3): DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of
  memory")`, during a run of `lcp-split` prefix-cache inserts at the cache's 2,052 MB budget, and were followed by a
  step-OOM reclaim; every parked session was retried and served (45 x 200, 19 typed 429s, no crash). The vmm arm of
  the same gate reads ALL GREEN. No co-tenant (`compute-apps` empty before and after). The r3 pooled run of the same
  gate read ALL GREEN; r3 to r4 (`d5923ccae` to `c6f9282c2`) also changed `admit_memory.rs` (9 lines) and the
  worker, so whether the pooled program moved or the run placed differently is not placed here (repro needed).
- Every other correctness clause PASS: A1-GATE's other nine cells, A1-MIX both kinds both orders, A1-STREAM, A2, A5
  (MAPPER, ENSURE, BUILD1, BUILD64), A6, A7 on all three shapes. A3 (i) and (ii) PASS (TTFT p50 ratio 1.010, ITL p99
  0.999, TPOT p50 0.998, throughput 0.997, N=10). A4 reads FAIL on its stated bound (36 of 438 retires within it; the
  helper placement's lookahead granule counts as excess, 1.14); it is outside the rule.
- **The rule of 1.7 reads FAIL (no reading) for the 5090 class**, on A1: a correctness clause failed, on the pooled
  arm. The cause is the door-armed admission booking on the pooled allocator (the prefix cache's LCP-split inserts
  during the burst), not the vmm arm; the same gate on O5's corrected prime booking runs in the DAY39 5090 halves
  (queue-i), which is where this reading goes next. No clause changes.
- A4-IDLE (a reading): at idle the vmm arm leaves 2.7 to 3.7 GB more driver-free memory than pooled on the mix
  (13.62 against 10.93 and 9.92 GB plain; 11.47 against 8.41 GB spec).

### 2.8 Addendum G's repro (`rtx5090-day37/r4-repro/`, the 5090, 2026-09-26)

Verbatim (`run.log`), with each gate's verdict lines in `<run>.gate.log`:

```
2026-09-26T14:33:19Z r4-1 rc=0 ADMIT-MEM BURST GATE: RED
2026-09-26T15:07:29Z v3-1 rc=0 ADMIT-MEM BURST GATE: ALL GREEN
2026-09-26T15:20:43Z r4-2 rc=0 ADMIT-MEM BURST GATE: RED
r4-1: AMB no prefill OOM: oom_lines=8 status={429: 19, 200: 45} -> FAIL
v3-1: AMB no prefill OOM: oom_lines=0 status={200: 44, 429: 20} -> PASS
r4-2: AMB no prefill OOM: oom_lines=8 status={200: 45, 429: 19} -> FAIL
```

- **Placed:** the r4 binary reds twice (8 parked prefill OOMs each, as 2.7 read 7), and the addendum-B binary (`v3`,
  DAY39's `be2177ead`) greens on the same gate. 2.7's A1 red is the door's booking without DAY39 addendum B's terms,
  not the pooled allocator. The 5090 class's rule reading on the r4 tree stays FAIL (no reading) as it read; a rerun
  of O1's 5090 cell (the gate set and the A1 lines) on a tree with addendum B is owed.
