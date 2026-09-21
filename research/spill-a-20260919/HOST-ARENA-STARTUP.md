# Pinned host arena startup (memra#385): measurement plan and the BOX3 harness receipt

Lane `lane/spill-a-20260919`, day 12. Issue #385 measured the 288 GiB startup `malloc_host` of
the pinned host arena at 121.9 s, 124.1 s, 134.4 s, 146.6 s and 221.7 s across separate boots on
a 2x B200 box (fill 0 ms every time; the whole cost is the driver allocation). Those are
observed boots, not a controlled comparison, and the cause of the residual time is not
established. This file fixes the measurement law, names the candidate mechanisms with their
code sites, and records one scaled harness cell on the single RTX PRO 6000 Blackwell (BOX3)
that proves the harness runs, not the result. The decision cell is the 2x B200 pair and is out
of scope here; nothing measured on BOX3 transfers to it.

## What the arena does today (code sites, lane tip `405466cf7`)

| Site | What |
|---|---|
| `crates/memra-engine/src/pinned_host.rs:95-117` | `PinnedHostArena::reserve(context, bytes)`: one `cudarc::driver::result::malloc_host(bytes, CU_MEMHOSTALLOC_PORTABLE)` (`:102-105`, `cuMemHostAlloc`), timed into `ArenaInner::alloc_ms` (`:108`, `:114`). Backing is deliberately uninitialized (no fill; the doc comment at `:89-94`). |
| `crates/memra-engine/src/pinned_host.rs:9-64` | `Extents`: the offset allocator over ONE contiguous base pointer (`ArenaInner::ptr`, `:67`). Every lease is `ptr.add(offset)` (`try_reserve_planes`, `:119-140`). A chunked reserve changes this to a list of bases, or keeps one virtual range. |
| `crates/memra-engine/src/pinned_host.rs:79-84` | `Drop for ArenaInner`: one `free_host`. |
| `crates/memra-engine/src/pinned_host.rs:145-147` | `reserve_timings_ms() -> (alloc_ms, 0.0)`: the `alloc_ms=` / `fill_ms=` figures in the startup line. |
| `crates/memra-server/src/worker.rs:14779-14811` | The startup reserve under `MEMRA_GLM5_TP_KV_HOST=1`: `MEMRA_KV_HOST_MB` parsed afresh (`:14782-14790`), `host_memory::check_headroom(bytes)` (`:14793`), `PinnedHostArena::reserve(engine.ctx().clone(), bytes)` (`:14795`), then the `[prefix-host DEBUG] arena startup: reserve_ms=... alloc_ms=... fill_ms=... capacity=... leased=0 free=... request_pin_count=0` line (`:14801-14804`). Reserve happens before readiness, by design. |
| `crates/memra-server/src/worker/host_memory.rs:4`, `:20-24` | `MARGIN = 32 GiB`: the headroom the reserve must leave (`startup pinned arena headroom refused: budget=... margin=... available=...`). |
| `crates/memra-server/src/worker.rs:8035-8050` | `HostPrefixCache::log_arena`: every later arena line carries the same three timings, so a boot's `alloc_ms` is on every receipt. |

Invariants the issue keeps and any replacement must keep: reserve before readiness; all-or-none
image admission (`try_reserve_planes` is atomic over the plane list); the 32 GiB margin;
initialized-range protections (`PinnedHostBuf.written`, a lease is unreadable until fully
written or fenced); zero request-path pin or free (the arena never calls the driver after
`reserve`).

## Measurement law

1. **Same box, one window, interleaved.** Every arm of a comparison runs in one process on one
   box inside one thermal window, interleaved A/B/A/B with N >= 5 pairs per order and both
   orders (AB then BA). Cross-boot and cross-day figures are not comparable (the issue's own
   121.9 s versus 221.7 s at one size shows the spread); neither are figures from another box.
2. **Free between measurements.** Each allocation is released before the next so the box never
   holds two arenas and every measurement starts from the same resident set. Record
   `MemAvailable` before each allocation; the first pin after a long idle also evicts the page
   cache and is reported with the rest, not dropped.
3. **Measure the driver call, not the boot.** The arm is `cuMemHostAlloc` (what
   `malloc_host` is) with `CU_MEMHOSTALLOC_PORTABLE`, timed wall-clock around the call(s);
   READY time is recorded separately in the serving cell because model load sits in front of
   it. Report the median, min and max per arm and the derived GiB/s.
4. **Record the regime.** GPU temperature and power at start and end, host `loadavg`,
   `MemTotal`, `MemAvailable`, hugepage state (`Hugepagesize`, `HugePages_Total`,
   `AnonHugePages`, THP mode), cgroup memory limit of the process, NUMA node count and the
   process's node placement (`numactl -H`, `/proc/self/numa_maps` for the arena range on the
   decision box), driver version, and the collector's lock proof and power limits.
5. **Correctness beside speed.** A mechanism that changes the arena's shape (chunked bases,
   hugepage backing) reruns the arena admission tests (`pinned_host.rs` unit tests) and the
   host-tier target-card gates (`tools/kv-host-spill-identity-gate.sh`,
   `tools/kv-host-spill-failure-gate.sh`) at the same budget before its speed is quoted. The
   byte layout of a lease is unchanged by construction (a lease is still one contiguous
   range); what a chunked arena changes is which base a range lives on.
6. **Decision on the target.** The decision cell reserves the production budget
   (309237645312 bytes) on the 2x B200 pair under `/tmp/memra-gpu.lock`; a scaled cell on a
   smaller box is a harness check only.

## Candidate mechanisms

1. **Chunked parallel `cudaHostAlloc`.** N threads each `cuMemHostAlloc(bytes / N)` with the
   same flags, joined; `Extents` becomes a list of `(base, capacity)` chunks (or one chunk per
   thread with the plane allocator refusing a plane larger than a chunk). Code sites:
   `reserve` (`pinned_host.rs:95-117`), `ArenaInner.ptr` (`:67`), `try_reserve_planes`
   (`:119-140`), `Drop` (`:79-84`). Hypothesis: the driver pins and maps pages on the calling
   thread, so N callers use N cores for the page walk. Measured below on BOX3 as a harness
   check.
2. **Hugepage-backed reserve.** Allocate the backing with `mmap(MAP_HUGETLB)` or THP
   `madvise(MADV_HUGEPAGE)` and register it with `cuMemHostRegister(CU_MEMHOSTREGISTER_PORTABLE)`
   instead of `cuMemHostAlloc`, so the driver pins 2 MiB pages (512x fewer PTEs and pin
   operations). Code sites: `reserve` and `Drop` (`cuMemHostUnregister` then `munmap`), plus
   the boot check for hugepage availability (`HugePages_Total` or THP mode) with a typed refusal
   when the pool is absent. Not measurable on BOX3 today: `HugePages_Total: 0`, THP mode
   `madvise` (recorded in the arena cell's `command.log`); the decision box needs a reserved
   pool or THP `madvise` at least.
3. **Lazy commit.** Reserve the address range but pin in slices as leases are taken. This
   trades startup time for request-path pin calls, which the issue forbids ("zero request-path
   pin/free") and which the arena's admission law forbids (a lease must be backed before
   `try_reserve_planes` returns it). Listed to be rejected on the law, not measured.

## Harness

`tools/pinned-host-reserve-bench.py` (lane tip): ctypes on `libcuda.so.1`; `cuInit`,
`cuDevicePrimaryCtxRetain`, `cuCtxSetCurrent` per thread; arm `single` = one
`cuMemHostAlloc(bytes, PORTABLE)`, arm `chunked` = N threads each `cuMemHostAlloc(bytes/N,
PORTABLE)` joined; free after every measurement; AB x pairs then BA x pairs; one JSON row per
measurement and a summary with per-arm medians, min, max, per-order medians and GiB/s. Size
defaults to 75 percent of `MemFree` at start (the task's "largest arena the box's free host
RAM allows minus 25 percent", read literally: a pin sized on `MemAvailable` would evict the
page cache, and on BOX3 today that cache holds another lane's artifact; `--basis available`
is for a box that runs nothing else), rounded down to 2 MiB. It runs under the collector
because it takes the device's primary context. No engine binary is involved and no engine file
changes.

## BOX3 harness receipt (one RTX PRO 6000 Blackwell, 600 W, N=5 pairs per order, executed-not-qualified)

Cell `pro-single-day12/arena-r2/` (collector `--rig pro-single`, lock `/tmp/memra-gpu.lock`,
`status: executed-not-qualified`, exit 0; `receipt.json` is the harness's own record). The box:
30 CPUs, one NUMA node, `MemTotal` 94,879,371,264 B (88.4 GiB), `MemFree` 31,317,016,576 B at
start, `MemAvailable` ~90.7 GB (page cache holds another lane's artifact), `HugePages_Total: 0`,
THP `madvise`, driver 580.178.04. Size = 75 percent of `MemFree` = 23,486,005,248 B (21.87 GiB),
8 chunks of 2,933,915,648 B. 20 measurements: AB x 5 then BA x 5, every allocation freed before
the next, every row `ok`.

| arm | n | alloc median | AB median | BA median | min | max | free median | GiB/s |
|---|---|---|---|---|---|---|---|---|
| `single` (one `cuMemHostAlloc`, today's `reserve`) | 10 | 3598.8 ms | 3578.2 ms | 3609.7 ms | 3569.9 ms | 3626.5 ms | 1389.7 ms | 6.1 |
| `chunked` (8 threads, `cuMemHostAlloc(bytes/8)` each, joined) | 10 | 3583.8 ms | 3582.8 ms | 3584.8 ms | 3571.0 ms | 3595.1 ms | 1421.0 ms | 6.1 |

Regime: GPU idle throughout, 40 C / 95.96 W at start, 36 C / 89.19 W at end (600 W limit,
2355 MHz); host `loadavg` 0.33 at start, 0.88 at end; `MemFree` 31.3 GB before every row (the
pin never touched the page cache, by the sizing). No other GPU process during the cell.

What the harness shows on this box, and only this box: the two medians differ by 0.4 percent
(15 ms on 3.6 s), inside the single arm's own AB/BA spread (31 ms). The per-chunk completion
times inside one `chunked` measurement step by ~450 ms each (`chunk_alloc_ms` 451, 898, 1345,
1790, 2238, 2687, 3132, 3152 ms in one row): the eight concurrent `cuMemHostAlloc` calls
complete one after another, which reads as the driver serializing host-pinned allocation under
one lock, so eight callers do not use eight cores for the page walk. Candidate 1's hypothesis
did not hold here. Whether the 2x B200 box behaves the same at 288 GiB (its 1.3 to 2.4 GiB/s
against this box's 6.1 GiB/s at 22 GiB) is the decision cell's question, not answered here: a
different box, a different driver build, 13x the size, and two devices' primary contexts. The
harness runs unchanged there (`--basis available --chunks 8 --pairs-per-order 5` on an idle
box); the hugepage arm needs a reserved pool first.
