# A GB10 boot can fail at context creation while the page cache is full (memra#1316)

Date: 2026-11-14. Follows `research/unified-readers-20261114/` (memra#1307), whose cells lost two
boots to it, and `research/admit-headroom-unified-20261113/` (memra#1257), whose page-cache fill
the cells here reuse.

**Verdict: fixed.** On GB10 a `memra-server` boot could die 0.2 s after its start line with
`Engine::new failed: DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")` while `MemAvailable`
read about 120 GB. The call that fails is context creation (`cuDevicePrimaryCtxRetain`): the
driver needs one free 2 MiB block of system memory for the new channel's context buffer, asks the
kernel for it without waiting, and gets none when the page cache is full and busy. The same call
succeeds later, usually within a second, once after 8.5 s.

Context creation is now retried on a unified-memory device while it answers out of memory and the
host reports at least 256 MiB available (waits of 50 ms doubling to 800 ms, until 90 s have
passed since the first try began), and every boot step names its driver call when it fails. With real `memra-server` boots, arms
alternating boot by boot: the red arm (no retry) died in 7 of 385 boots, every one a boot that started with MemFree under 3,000 MB (7 of 156); the retry arm died in none of 384, its first creation failing in 10 of them and succeeding on a later try each time, as did one full serving boot that then served its requests. Of the 463 boots that started with 3,000 MB free or more, none of the red arm's died and two of the retry arm's failed their first try and came up on a later one.
Decision record: [`docs/decisions/GB10-CONTEXT-CREATE-RETRY.md`](../../docs/decisions/GB10-CONTEXT-CREATE-RETRY.md).

## Conditions

- Two DGX Spark units (GB10, sm_121a, driver 580.178.04, kernel 7.0.0-1019-nvidia, 4 KiB pages,
  128 GB unified memory, `vm.min_free_kbytes` 45166, no CMA): `unit-a` and `unit-b`, the names
  memra#1307's record uses. One lease per campaign; nothing else ran on a unit during a cell.
- No host setting was changed and the page cache was never dropped. The cells read files and
  `/proc`, create CUDA contexts and start `memra-server`.
- **The state.** `tools/fill_live.sh` fills the page cache with four readers (`cat` over the model
  files, memra#1257's `fill_cache2.sh`) until MemFree is under 3,000 MB, then keeps N readers
  going. `MemAvailable` stays near 119 to 122 GB throughout.
- **Driver-level boots** (`tools/ctx_probe.py`, Python and `libcuda` only): the calls `Engine::new`
  makes up to its first stream, in order (`cuInit`, `cuDeviceGet`, `cuDevicePrimaryCtxRetain`,
  `cuCtxSetCurrent`, two `cuStreamCreate`, a 64 MiB allocation), one process per boot. A call that
  answers out of memory is repeated in place every 50 ms.
- **Server boots** (`tools/server_boot.py`): the real `memra-server` with the serving both
  occurrences booted (`tools/serve_env.sh`: Qwen3.8-27B GGUF with the DFlash2 drafter, 32,768
  context, 4 sessions), started and read until the worker logs `Engine ready` (past
  `Engine::new`; the process is killed there, before it reads weights) or until it exits on a
  FATAL. A boot takes 0.3 to 0.6 s, so hundreds fit in a lease.
- **Arms**, both `memra-server` for sm_121a built on the unit from the same tree
  (`tools/build_unit.sh`), differing in one constant of `crates/memra-runtime/src/ctx_boot.rs`:
  - T = the retry;
  - B = the red arm: the retry budget set to zero (`tools/arm_b.py`), so the context is created
    once, as before the change. B still names the failing call and prints the host's memory, so
    every boot of this record, in either arm, shows whether its first creation failed.
  - Arms alternate boot by boot inside a cell, so both meet the same state.
- **Four builds of the branch**, because the cells and two reviews changed the code. Which try
  fails first does not depend on the build (B's path is the same in all four, and no boot of T
  reached its bound); what differs is what the lines say and how long T would have waited. A
  cell's `run.log` has its binaries' hashes and the hashes of the arm's `memra-runtime` sources.

| build | what it is | T | B | cells |
|---|---|---|---|---|
| 1 | the first version: 10 s of waits at most | `3383967009ca...` | `5cb624e733bd...` | `g*`, `h1`, `t0` to `t5`, `g-smoke` on unit-a; `t1` on unit-b (the two units built byte-identical binaries) |
| 2 | after review: `cuInit` and `cuDeviceGet` fail under their own names, the retry line states no cause, the block count is labelled by order, host memory lines on unified devices only, a wait is at least 1 ms | `a71ff5b76228...` | `1f5b10e5242e...` | `t7-live8`, `g-smoke-final` |
| 3 | 30 s of waits at most, after a boot of build 2 needed 8 s. Also the first build on the branch rebased from main 63c351a2f4 onto 0db086f87a: `dsv41_gpu/experts.rs`, `dsv4_gpu.rs`, `lib.rs`, `spec.rs`, `unified_memory.rs` of `memra-engine` and `memra-server`'s `worker.rs` differ from builds 1 and 2 (memra#1310, memra#1317), none of them in the context's path | `f7a90773d436...` | `92c00f1a24bd...` | `t8-live8`, `g-smoke-final3` |
| 4 | the head (on main 0db086f87a): the bound is 90 s of elapsed time on an injected clock, the retry and recovered lines name the driver's code by number and log elapsed time, the wiring of `create_context` is a testable function. Built from the head's `ctx_boot.rs` before a last edit inside its test module (two tests' fake clocks); everything outside `mod tests` is identical | `ae0101548b74...` | `ea7a1a4dec6e...` | `u1-live8`, `g-smoke-fold` |

## 1. The failing call

`Engine::new` is `CudaContext::new` (cuInit, cuDeviceGet, cuDevicePrimaryCtxRetain, one device
attribute, cuCtxSetCurrent), a stream, the cuBLASLt handle, eight `cuModuleLoadData` and a second
stream. The error carried no site, so the change names each step, and the first reproduced server
boot (arm B, `gb10/unit-a/g4-live4`, boot 6; build 1's wording, the head labels the last field
`free blocks of order 9 and up (2 MiB with 4 KiB pages)`) says:

```
[ctx-retry] device 0: context creation still out of memory after 1 tries and 0 ms of waits (host MemFree 1861 MiB, MemAvailable 118868 MiB, Cached 117817 MiB, free blocks of 2 MiB and larger: 0); giving up
[server] FATAL: worker init failed: Engine::new failed: cuDevicePrimaryCtxRetain (context creation) on device 0 failed: DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory") (host MemFree 1860 MiB, MemAvailable 118868 MiB, Cached 117817 MiB, free blocks of 2 MiB and larger: 0)
```

- Every failure in this record is that call: 7 arm-B server boots, the first tries of
  10 arm-T server boots and of one full serving boot, 5 creations of the
  cells' holder process and one driver-level creation (`gb10/unit-a/d2-live4-churn`). No stream,
  cuBLASLt handle or module load failed in 769 server boots.
- `cuMemGetInfo` cannot be read before the call (it needs a context). Right after a context
  exists it reads 160 to 250 MB under MemFree (`gb10/unit-b/a0-asis-loop`, 200 boots), so the
  host's MemFree at a failed try is close to the driver's reading: 0.8 to 6.7 GB, never near
  zero.
- The kernel log has the same lines for every failure, the two original ones included
  (`gb10/unit-a/kernel-original-12-T.log`, `gb10/unit-b/kernel-original-18-B.log`, each cell's
  `nvrm.log`): `NV_ERR_NO_MEMORY` from `_memdescAllocInternal` (`mem_desc.c:1359`) returned
  through `kernel_graphics_context.c:1178`, `kgrctxAllocMainCtxBuffer`
  (`kernel_graphics_context.c:1387`) and `kgrctxAllocCtxBuffers`
  (`kernel_graphics_object.c:215`), one set per failed try. In the original failures and in most
  of these a `NV01_MEMORY_SYSTEM` allocation of the driver's memory manager
  (`nv_gpu_ops.c:4969`) fails 50 to 120 ms earlier in the same boot; that one alone is
  survivable (it appears in boots that came up).

## 2. Why it fails

- **The buffer is one contiguous block.** In the driver's open source (the 580 branch; the line
  numbers above match it) `kgrctxAllocMainCtxBuffer` creates the main context buffer of the new
  channel's compute engine with `kgraphicsShouldForceMainCtxContiguity`, which returns true. On a
  unified-memory device that is system memory: `nv_alloc_contig_pages` (`nv-vm.c`, shipped with
  the driver under `/usr/src`) asks the kernel's page allocator for one block of
  `get_order(size)` with `GFP_KERNEL | __GFP_NOWARN | __GFP_RETRY_MAYFAIL`
  (`nv_compute_gfp_mask`). With that flag the kernel reclaims and compacts for a bounded number
  of rounds and then fails instead of waiting or invoking the OOM killer.
- **The block is 2 MiB.** A context created on a quiet host takes, by `/proc/buddyinfo` before
  and after (`gb10/unit-a/d0-asis-churn`, 151 creations, the median of each order): 2 blocks of
  order 9 (2 MiB), 1 of order 7, 4 of order 5, 2,734 of order 4 (64 KiB), about 13,400 single
  pages and nothing above order 9: 229 MiB in all. At the driver-level failure the kernel had 2
  free blocks of order 8, 108 of order 7 and none of order 9, so the buffer needs order 9.
- **Only this allocation has no fallback.** The driver's other large-page allocations log
  `Allocation failed with big page size, retrying with default page size`
  (`_kgmmuClientShadowFaultBufferPagesAllocate`) or fall back inside the memory manager, in boots
  that then come up: 8 and 5 such events in 400 boots of `gb10/unit-b/a1-live2-loop`, none of
  them fatal.
- **The state.** With the page cache full the free lists hold no block of 2 MiB (the count the
  server prints was 0 at every failed try) and the kernel has to make one by reclaiming or moving
  page cache. The failed call returned after 22 ms (driver-level). MemFree at the failed tries
  was 0.8 to 2.8 GB, and in two boots 4.3 to 6.7 GB: free memory in pieces smaller than 2 MiB
  does not help. `MemAvailable` was between 118.3 and 118.9 GB.
- **It passes.** The booting process holds nothing, so there is nothing to release. The same
  call succeeded on the first repeat 50 ms later (driver-level) and in every retried server
  boot: one on try 2 after 50 ms of waits, three on try 3 after 150 ms of waits, two on try 5 after 750 ms of waits, one on try 6 after 1,550 ms of waits, one on try 14 after 7,950 ms of waits (builds 1 to 3 logged the sum of their waits);
  on the head, which logs the clock: one on try 2 after 738 ms, one on try 5 after 2,862 ms. The full serving boot of section 5 came up on try 5 after 750 ms of
  waits. In the longest one (`gb10/unit-a/t7-live8`, eight readers; 8.5 s on the clock) MemFree
  stayed flat between 1,036 and 1,157 MiB and every try, about 800 ms apart, found no block.
- **A try costs time, and the state can outlast any of these recoveries.** A failed creation
  returned in 22 to 266 ms and one that succeeded took 3.6 s (`gb10/unit-a/t8-live8`, round 1:
  750 ms of waits, 4.4 s on the clock). The 50 ms memory records hold stretches with no free
  2 MiB block of 30.0, 31.6 and 32.2 s (`gb10/unit-b/t1-live4`, `gb10/unit-a/g-smoke`,
  `gb10/unit-b/c1-cycles`); no boot happened to start inside one.
- **The same failure is older than these cells.** The units' kernel logs of the three weeks
  before hold 13 more context-buffer failures from other workloads
  (`gb10/unit-a/kernel-history.log`, `gb10/unit-b/kernel-history.log`), among them runs of three
  within 2.1 s and five within 4.1 s, about one second apart. Their host state is not known.

## 3. How often

`tools/readme_tables.py` prints the tables and totals below from `gb10/`.

**Driver-level** (no memra; each creation is `cuDevicePrimaryCtxRetain` in a fresh process, or
`cuCtxCreate` over and over in one process in the `churn` cells):

| cell | creations | started under 3,000 MB | out of memory |
|---|---|---|---|
| `unit-a/d0-asis-churn` | 151 | 0 | 0 |
| `unit-a/d1-live2-churn` | 927 | 927 | 0 |
| `unit-a/d2-live4-churn` | 149 | 149 | 1 |
| `unit-a/e1-live2-gap4` | 225 | 7 | 0 |
| `unit-b/a0-asis-loop` | 200 | 0 | 0 |
| `unit-b/a1-live2-loop` | 400 | 279 | 0 |
| `unit-b/a2-live2-exit` | 187 | 31 | 0 |
| `unit-b/b1-live2-dip` | 5 | 2 | 0 |

- `a0`, `d0`: the host as found, no reader. `a1`, `d1`, `e1`: two readers (boots 0.5 s apart, back
  to back in one process, 4 s apart). `a2`: two readers, a 30 GiB holder exits and a boot follows
  every 1.4 s. `d2`: four readers.
- The holder of the transition cells below is this probe too, and its own context creation
  answered out of memory 5 times in 58 rounds (`t1-live4`, `t5-live4`, `t8-live8`: `holder failed` in their `run.log`).

**Server boots**, arms alternating boot by boot (each entry: boots / boots that started with
MemFree under 3,000 MB / deaths for B, retried first tries for T):

| cell | MemFree at the boot, median MB | B: boots / under 3,000 MB / died | T: boots / under 3,000 MB / first try failed |
|---|---|---|---|
| `unit-a/g-main-live4` | 7,931 | 80 / 0 / 0 | 79 / 0 / 0 |
| `unit-a/g0-asis` | 9,616 | 20 / 0 / 0 | 20 / 0 / 0 |
| `unit-a/g2-live2` | 1,917 | 40 / 39 / 0 | 40 / 38 / 0 |
| `unit-a/g4-live4` | 5,108 | 40 / 5 / 1 | 40 / 5 / 1 |
| `unit-a/g8-live8` | 9,377 | 40 / 5 / 0 | 40 / 5 / 0 |
| `unit-a/h1-01` | 4,495 | 6 / 0 / 0 | 6 / 0 / 0 |
| `unit-a/t0-live4` | 1,966 | 16 / 12 / 0 | 16 / 12 / 0 |
| `unit-a/t1-live4` | 2,646 | 51 / 28 / 1 | 51 / 26 / 1 |
| `unit-a/t2-plain4` | 2,376 | 4 / 3 / 0 | 4 / 3 / 0 |
| `unit-a/t3-plain4` | 1,204 | 6 / 6 / 0 | 6 / 6 / 0 |
| `unit-a/t5-live4` | 2,461 | 13 / 13 / 1 | 13 / 11 / 2 |
| `unit-a/t7-live8` | 1,677 | 18 / 15 / 2 | 18 / 13 / 1 |
| `unit-a/t8-live8` | 2,478 | 21 / 12 / 1 | 21 / 12 / 3 |
| `unit-a/u1-live8` | 2,481 | 24 / 16 / 1 | 24 / 17 / 2 |
| `unit-b/t1-live4` | 4,764 | 6 / 2 / 0 | 6 / 2 / 0 |

- arm B: 385 boots, 7 died; under 3,000 MB: 7 of 156; at 3,000 MB and over: 0 of 229; first boot of a round after a holder's exit: 3 of 29; first boot of a round with no holder: 0 of 3; median boot without a retry 363 ms over 378
- arm T: 384 boots, 10 first try failed, all came up; under 3,000 MB: 8 of 150; at 3,000 MB and over: 2 of 234; first boot of a round after a holder's exit: 5 of 28; first boot of a round with no holder: 0 of 2; median boot without a retry 361 ms over 374
- recovered boots (cell, try that succeeded, ms of waits): [('`unit-a/t8-live8`', 2, 50), ('`unit-a/g4-live4`', 3, 150), ('`unit-a/t1-live4`', 3, 150), ('`unit-a/t5-live4`', 3, 150), ('`unit-a/t5-live4`', 5, 750), ('`unit-a/t8-live8`', 5, 750), ('`unit-a/t8-live8`', 6, 1550), ('`unit-a/t7-live8`', 14, 7950)]
- recovered boots of the build that logs elapsed time (cell, try that succeeded, ms since the first try began): [('`unit-a/u1-live8`', 2, 738), ('`unit-a/u1-live8`', 5, 2862)]
- the holder's own context creation failed (cell, times, rounds): [('`unit-a/t1-live4`', 2, 26), ('`unit-a/t5-live4`', 1, 18), ('`unit-a/t8-live8`', 2, 14)]
- longest stretches with no free block of order 9 and up (seconds, cell): [(32.2, '`unit-b/c1-cycles`'), (31.6, '`unit-a/g-smoke`'), (30.0, '`unit-b/t1-live4`'), (18.2, '`unit-b/c1-cycles`'), (15.7, '`unit-a/t5-live4`')]

- `g0`: the host as found. `g2`, `g4`, `g8`, `g-main`: the page cache full with 2, 4, 8 and 4
  readers, boots 1 to 2 s apart. `t0`, `t1`, `t5` (four readers) and `t7`, `t8` (eight): a
  24 GiB holder exits before each round of boots, and a round starts when MemFree is back under
  2,500 MB (`tools/transition_cell.sh`); when it does not come back, the readers pause 60 s and
  start again. `h1`, `t2`, `t3`: rounds without a holder (the readers start, then the boots).
  `u1`: the `t7` design on the head.

What the cells say:

1. **No reader, no failure.** 351 creations and 40 server boots on a host with 9 to 49 GB free.
2. **Boots in a row protect themselves.** A boot at the floor takes about 2,700 blocks of 64 KiB
   and the kernel answers by keeping more memory free for as long as the readers run. In `g4`
   MemFree was 1.1 to 1.7 GB for the first ten boots and 3.3 to 5.4 GB for the other seventy; in
   `g8` 0.9 to 1.9 GB, then 3.4 to 12.0 GB. Of the boots that started with 3,000 MB free or
   more, none of the red arm's 229 died and two of the retry arm's 234 failed their first try (6.8 and 7.3 GB free, no free 2 MiB block; both came up). A back-to-back loop therefore measures a state a restart does not meet,
   which is why the first campaigns (`a1`, `d1`, `e1`, about 1,550 creations) saw nothing.
3. **Two readers were not enough here.** `g2` held MemFree near 1.9 GB for 80 boots with no
   failure, like the driver-level cells with two readers. The two original failures did happen
   with two readers, right after a server holding about 27 GB had exited.
4. **The failures sit where a restart is: at the first context after the readers have used up
   what an exiting process freed.** Of the first boots of a round that followed a holder's exit,
   8 of 57 (red arm 3 of 29, retry arm 5 of 28)
   met an out-of-memory answer: B's died, T's failed their first try and recovered. Of the
   later boots of such rounds it was 7 of 241, and of the first boots of rounds with no
   holder (`t2`, `t3`) 0 of 5. The original cells
   lost 2 of the 14 boots that followed a DFlash2 cell with live readers; memra#1307's cells
   started the next server 15 s after the last one's SIGTERM (`waitfor --gone` polls every 15 s),
   about the time two readers need to use up what that server freed.
5. **Under 3,000 MB the rate was a few in 100 boots**, each arm: B died in 7 of 156 (4.5 %), T's first try failed in 8 of 150 (5.3 %).

## 4. The retry

- **B: 385 boots, 7 died.** Each death is the line of section 1, at MemFree 1.0 to 2.5 GB,
  `MemAvailable` 118.4 to 118.9 GB and no free 2 MiB block.
- **T: 384 boots, none died.** Its first creation failed in 10 boots, each a boot B's code
  loses, and every one came up: one on try 2 after 50 ms of waits, three on try 3 after 150 ms of waits, two on try 5 after 750 ms of waits, one on try 6 after 1,550 ms of waits, one on try 14 after 7,950 ms of waits; on the head, which logs the clock: one on try 2 after 738 ms, one on try 5 after 2,862 ms.
- A retried boot logs each try (`gb10/unit-a/t5-live4`, round 16; build 1's wording):

```
[ctx-retry] device 0: context creation answered out of memory on unified memory (try 1, 0 ms waited so far; host MemFree 1557 MiB, MemAvailable 118388 MiB, Cached 118058 MiB, free blocks of 2 MiB and larger: 0); the kernel has not freed a block for the context buffers yet, trying again in 50 ms
[ctx-retry] device 0: context creation answered out of memory on unified memory (try 4, 350 ms waited so far; host MemFree 2614 MiB, MemAvailable 118842 MiB, Cached 117016 MiB, free blocks of 2 MiB and larger: 0); the kernel has not freed a block for the context buffers yet, trying again in 400 ms
[ctx-retry] device 0: context created on try 5, after 750 ms of waits
```

- The kernel log agrees: one `kgrctxAllocCtxBuffers` line per failed try and none for the try
  that succeeded (`nvrm.log`; `gb10/unit-a/t5-live4` has 8: 1 death of B, 2 + 4 failed tries of
  T and 1 of the holder).
- **The bound moved twice.** Builds 1 and 2 bounded the waits at 10 s (9,550 ms, 16 tries); the
  fourteenth try of `t7-live8` came after 7,950 ms, 83 % of that. Build 3 bounded them at 30 s.
  A review then pointed at the two things in this record that 30 s of waits ignores: the tries
  take time themselves, and the state lasted over 30 s three times. The head bounds the clock:
  no try starts once its wait would end more than 90 s after the first try began, about three
  times the longest stretch and ten times the longest recovery.
- **The head's lines** (build 4) name the driver's code by number, so that a boot the retry
  saved does not trip the serving gates' scan of the server log for `out of memory` and
  `CUDA_ERROR` (`tools/serve-stress-gate.sh`, `tools/drafter-admission-gate.py`,
  `tools/same-pass-admission-gate.py`), and give elapsed time:

```
[ctx-retry] device 0: context creation returned driver code 2 (no memory) on unified memory (try 1, 92 ms since the first try began; host MemFree 1738 MiB, MemAvailable 118436 MiB, Cached 117637 MiB, free blocks of order 9 and up (2 MiB with 4 KiB pages): 0); trying again in 50 ms
[ctx-retry] device 0: context creation returned driver code 2 (no memory) on unified memory (try 4, 683 ms since the first try began; host MemFree 1120 MiB, MemAvailable 118261 MiB, Cached 118253 MiB, free blocks of order 9 and up (2 MiB with 4 KiB pages): 0); trying again in 400 ms
[ctx-retry] device 0: context created on try 5, after 2862 ms
```

## 5. A normal boot

- **No retry line when the first call succeeds.** 374 of the 384 arm-T boots logged
  no `[ctx-retry]` line, among them all 20 of `g0` (the host as found) and all 79 of `g-main`
  (4 to 10 GB free): the first call succeeded and nothing else ran.
- **Boot time is unchanged.** Boots that needed no retry took a median of 363 ms in arm B (378 boots) and 361 ms in arm T (374 boots) from the start to
  `Engine ready`.
- **Full serving cycles of arm T** (`tools/serve_cycle.sh`: boot, `/readyz`, two readers, four
  requests of 384 tokens, SIGTERM), two per build:
  - `gb10/unit-a/g-smoke` (build 1): both cycles served their four requests (HTTP 200) and
    stopped cleanly. The first started at MemFree 970 MB, right after the last transition cell:
    its context creation answered out of memory four times and came up on try 5 after 750 ms,
    1.4 s after the process started, and the server went on to load the weights and serve.
    Before the change that boot is the one of the issue.
  - `gb10/unit-a/g-smoke-final` (build 2): both cycles, no retry line, eight requests served.
  - `gb10/unit-a/g-smoke-final3` (build 3): 2 cycles, no retry line, 8 of 8 requests served, ready after 36.44 and 32.4 s.
  - `gb10/unit-a/g-smoke-fold` (build 4, the head): 2 cycles, no retry line, 8 of 8 requests served, ready after 37.45 and 40.52 s.
- **A discrete card.** On an NVIDIA GeForce RTX 5090 Laptop GPU (`rtx5090/discrete-b`, `tools/discrete_cell.sh`, the head's tree built for sm_120a): 10 of 10 boot-only starts came up with 0 `[ctx-retry]` lines, and a start on device 7, which the host does not have, ended with `cuDeviceGet on device 7 failed: DriverError(CUDA_ERROR_INVALID_DEVICE, "invalid device ordinal")` and no retry.
- **Earlier build, full cycles** (`gb10/unit-b/c1-cycles`, the tree before the first rebase, with
  a 2 GiB floor): 13 serving cycles, 7 of T and 6 of B, MemFree 5.4 to 25.9 GB at the boot: all
  came up and served.

## 6. CPU checks

Dev desk, CPU only (`tools/dd_checks.sh` on the head, logs in `cpu/`, `cpu/summary.txt`):

- `memra-runtime`: its 13 tests pass, 12 of them in `ctx_boot`. The policy is
  `ctx_boot::retry_reclaim_race`, a pure function over injected closures (the call, the
  out-of-memory test, the host reading, the wait, the clock), so the tests run it with a fake
  allocator on a fake clock that moves with every call and every wait:
  - `unified_boot_survives_an_out_of_memory_answer_the_page_cache_contradicts`: an allocator that
    answers out of memory twice and then succeeds, with 119 GiB available: the third try's
    context is returned, after waits of 50 and 100 ms, 750 ms on the clock with tries of 200 ms;
  - `the_budget_bounds_elapsed_time_not_the_sleeps`: with tries the kernel stalls in for 3 s
    each, the boot stops after 25 tries and 91.75 s, having slept 16.75 s (a bound on the sleeps
    would have let all 116 tries of the ladder run);
  - `discrete_card_keeps_its_first_answer`: one call, no wait, no log line;
  - `a_host_that_is_really_short_is_not_retried`: `MemAvailable` under the floor, or no host
    reading, returns the first answer; exactly at the floor is retried;
  - `the_wiring_retries_only_a_unified_device_only_out_of_memory_only_above_the_floor`:
    `create_with`, the function `create_context` runs, over a scripted driver: the device's
    attribute decides "unified" (0 and an unreadable attribute are discrete), only the driver's
    out-of-memory code is retried, the serving policy's floor and its 90 s are the ones applied;
  - `the_lines_of_a_boot_that_recovers_carry_no_phrase_the_log_gates_scan_for`: the retry and
    recovered lines hold none of the serving gates' phrases; the give-up and not-retried lines
    keep `out of memory`;
  - `the_waits_double_to_the_cap_and_stop_at_the_budget` (a wait that ends exactly on the budget
    is made, the serving policy with instant tries is 116 tries and 89,550 ms, a zero backoff
    cannot spin), `the_host_reading_is_taken_again_before_every_retry`,
    `another_error_ends_the_retry_at_once`, `only_a_driver_out_of_memory_answer_is_one`,
    `host_memory_is_parsed_from_its_own_lines`, `a_failed_step_names_its_driver_call`.
- **Red arms.** `tools/mutants.py` holds 12 mutations of `ctx_boot.rs`; `dd_checks.sh` applies
  each, requires cargo to rebuild and requires the tests to fail. All 12 fail
  (`cpu/summary.txt`, `cpu/red-arms.log`):
  - `budget-zero`, the file the on-device arm B is built from: 7 tests fail;
  - the policy: `budget-ge` (a wait that ends on the budget is dropped), `bound-sleeps-only`
    (the first version's bound), `retry-discrete`, `any-error`, `no-floor`, `no-reread`;
  - the wiring of `create_context`, which a review found untested (all four survived then):
    `wiring-unified-always`, `wiring-any-error`, `wiring-no-floor`, `attr-error-is-unified`;
  - `line-names-the-error`: the retry line with the phrase the gates scan for.
  `dd_checks.sh` touches the source before each arm and requires cargo's `Compiling
  memra-runtime` line, since a first pass reused a stale build and reported a red result as the
  green one.
- `cargo fmt --all -- --check` and `cargo clippy -p memra-runtime -p memra-engine -p memra-server
  --all-targets -- -D warnings` are clean; the engine library suite passes 784 tests
  and the server's 1,272 (`cpu/engine.log`, `cpu/server.log`; `cpu/clippy.log` is
  kept without cargo's `Compiling` and `Checking` progress lines).
- Two reviewers with clean contexts read the change: one after build 1 (its findings are
  build 2), one after build 3 (its findings are build 4 and the corrections of this record:
  the first-boot stratum, the builds' bases, the test count).

## Excluded cells and limits

- **Not a rate for every host.** The cells hold one kind of load: the page cache full and `cat`
  readers filling it. How often a restart fails under another load (a second model loading, a
  copy, a backup) was not measured; the mechanism is the same.
- **Few recoveries, one long.** 10 arm-T boots and one full serving boot needed the
  retry; one of them needed 8.5 s. How long the state can last is not known (the memory records
  show over 30 s), and a boot that outlasts the 90 s fails as before, with every try in its
  log. The cell that would measure the distribution of recovery times, and a fixed 50 ms
  cadence against the ladder, is memra#1330; it was not run.
- **Excluded from the counts:** `gb10/unit-b/b1-live2-dip` (boots armed to start when MemFree
  fell under 800 MB: it never did in 26 s after an exit, so its boots ran at the timeout and came
  up); cells stopped before their first boot (not copied: two transition cells whose rounds
  never saw MemFree come back under the threshold, and cells restarted with other settings).
  `gb10/unit-a/h1-01`, `t2-plain4` and `t3-plain4` are in the tables but are side designs: boots
  right after the readers start, which met no failure.
- **The four builds are pooled.** B's code path is the same in all four and no boot of T reached
  its bound, so the builds differ in what they print and in how long T would have waited, not
  in which boots fail first. Builds 3 and 4 are on a newer main than builds 1 and 2 (the build
  table names the files).
- **The respawn path was not run.** A respawned worker creates its context the same way, with
  the systemd watchdog's ping withheld; the decision record says why the 90 s is not sized to
  fit `WatchdogSec`, and memra#1331 is the older problem found on the way (the MTP serving's
  reload alone can miss it).
- **unit-b has few server boots with these arms** (12 in `gb10/unit-b/t1-live4`): the unit was
  handed to another lane's record. Its build-1 binaries are byte-identical to unit-a's.
- **One discrete card, one cell.** The retry cannot engage there by construction (`create_with` hands the policy the device's attribute; the tests above), and `rtx5090/discrete-b` is ten boots and one absent ordinal on an RTX 5090 Laptop GPU, not a battery.
- **No push evidence was run.** The change reaches every binary through `memra-runtime` and
  `memra-engine/src/lib.rs`, so `tools/cell_evidence.py need` lists `sm_120a` and `sm_121a`,
  numeric and perf. This record is the lane's own cells.

## Reproduce

Build both arms on a GB10 host under a lease, then run a campaign; `tools/on_unit.sh` does the
copy, the detached start and the bounded wait from the workstation:

```bash
# on the workstation, inside `sparklock run --unit U --ttl 45m --for "..." --`
tools/on_unit.sh HOST t1 43m 'bash research/gb10-boot-oom-20261114/tools/build_unit.sh T 9m \
  && bash research/gb10-boot-oom-20261114/tools/build_unit.sh B 6m; \
  bash research/gb10-boot-oom-20261114/tools/campaign_t.sh t1 t1-live4 4 6 1500 24 2500 $(( $(date +%s) + 1980 ))'
```

`tools/assemble.py` copies the fetched cells into `gb10/` and writes `gb10/summary.txt`
(`tools/tab_probe.py` for the driver-level cells, `tools/tab_boots.py` for the server cells).
