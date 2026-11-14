# On unified memory, context creation retries an out-of-memory answer the host contradicts (2026-11-14)

**Status:** adopted, memra#1316. Receipts:
[`research/gb10-boot-oom-20261114/`](../../research/gb10-boot-oom-20261114/).

## Question

On GB10 a `memra-server` boot died 0.2 s after its start line with
`Engine::new failed: DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")` while `MemAvailable`
read 121.9 GB. It happened twice in memra#1307's cells, once on each unit, both times with the
page cache full and readers still filling it; the next boot, seconds later, came up. The error
named no call. Which call fails, why, and what should a boot do about it?

## What the cells showed

- **The call is context creation.** `cuDevicePrimaryCtxRetain` (cudarc's `CudaContext::new`, the
  first thing `Engine::new` does) returns `CUDA_ERROR_OUT_OF_MEMORY`. A reproduced server boot
  without the retry now says so: `Engine::new failed: cuDevicePrimaryCtxRetain (context
  creation) on device 0 failed: DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory") (host
  MemFree 2505 MiB, MemAvailable 118829 MiB, Cached 117093 MiB, free blocks of order 9 and up
  (2 MiB with 4 KiB pages): 0)`.
- **The kernel names the allocation.** Every failure, the two original ones included, has the same
  lines in the kernel log: `NV_ERR_NO_MEMORY` from `_memdescAllocInternal` (`mem_desc.c:1359`)
  under `kgrctxAllocMainCtxBuffer` (`kernel_graphics_context.c:1178`, `:1387`) and
  `kgrctxAllocCtxBuffers` (`kernel_graphics_object.c:215`): the main context buffer of the new
  channel's compute engine. In the driver's open source that buffer is forced physically
  contiguous (`kgraphicsShouldForceMainCtxContiguity` returns true), so on a unified-memory device
  it is one multi-page block from the kernel's page allocator, asked for with
  `__GFP_RETRY_MAYFAIL` (`nv_compute_gfp_mask`, `nv-vm.c`): the kernel reclaims and compacts for
  a bounded time and then fails the allocation instead of waiting.
- **It is a 2 MiB block.** A context created on a quiet host takes two free blocks of order 9
  (2 MiB), one of order 7, about 2,730 of order 4 (64 KiB) and single pages, 229 MiB in all, and
  nothing above order 9 (`/proc/buddyinfo` before and after, 151 creations). The
  driver's other large-page allocations fall back to small pages when no block is free (the
  kernel logs `Allocation failed with big page size, retrying with default page size`); the
  contiguous context buffer has no fallback.
- **The failing state.** At each failed try the host had no free block of 2 MiB or larger,
  MemFree between 0.8 and 6.7 GB, and `MemAvailable` near 119 GB. The allocation failed within
  22 ms of the call (driver-level). Nothing the booting process holds is involved: it holds nothing yet.
- **It passes, usually within a second.** In the same process the same call succeeded on the
  first repeat 50 ms later (driver-level probe) and in every retried server boot:
  one on try 2 after 50 ms of waits, three on try 3 after 150 ms of waits, two on try 5 after 750 ms of waits, one on try 6 after 1,550 ms of waits, one on try 14 after 7,950 ms of waits
  (the builds before the elapsed bound logged the sum of their waits);
  on the head, which logs the clock: one on try 2 after 738 ms, one on try 5 after 2,862 ms.
  A full serving boot came up on try 5 after 750 ms of waits and went on to serve.
- **Once it took 8.5 s.** With eight readers one boot's context came on try 14, after 7,950 ms
  of waits and 8.5 s on the clock: MemFree stayed flat near 1.1 GB with no free 2 MiB block,
  and every try about 800 ms apart failed the same way. The first version of this policy
  bounded the waits at 10 s (9,550 ms); that boot used 83 % of it.
- **The tries cost time too.** A failed creation returned in 22 to 266 ms; one that succeeded
  took 3.6 s (the kernel stalled in it). A bound on the sleeps alone is not a bound on the boot.
- **The state can last half a minute.** The cells' 50 ms memory records hold stretches with no
  free block of 2 MiB lasting 30.0, 31.6 and 32.2 s. No boot happened to start inside one.
- **Where it happens.** Not in boots made one after another: after about ten boots at the floor
  the kernel keeps 3 to 12 GB free while the readers run. Of the boots that started with
  3,000 MB free or more,
  none of the red arm's 229 died and two of the retry arm's 234 failed their first try (6.8 and 7.3 GB free, no free 2 MiB block; both came up).
  It happens at the first context created after the readers have used up what an exiting
  process freed, which is where a restart is:
  8 of 57 (red arm 3 of 29, retry arm 5 of 28)
  such boots met an out-of-memory answer, and the two original failures were 2 of 14.
- **Before and after.** Arms alternated boot by boot. The red arm (the retry budget at zero):
  385 boots, 7 died. The retry: 384 boots, none died, with 10 first tries failed and every one
  recovered.

## Decision

`memra_runtime::ctx_boot::create_context` creates the context for every engine. On a
unified-memory device (`CU_DEVICE_ATTRIBUTE_INTEGRATED`, read before any context exists) it
repeats `CudaContext::new` while the answer is `CUDA_ERROR_OUT_OF_MEMORY` and the host reports at
least 256 MiB of `MemAvailable`, until 90 s have elapsed since the first try began. Each retry
logs one `[ctx-retry]` line with the host's MemFree, MemAvailable, Cached and free 2 MiB blocks;
a recovered boot logs the try that succeeded and the time it took. Those two lines name the
driver's code by number (`driver code 2`), not by name or text: the serving gates fail a cell on
`out of memory` or `CUDA_ERROR` anywhere in the server log, and a boot the retry saved is not a
failure. The give-up line, the not-retried lines and the boot's own error keep the words.

| number | value | from |
|---|---|---|
| first wait | 50 ms | the first repeat the driver-level probe made, which succeeded; the failed call itself returns in about 20 ms |
| later waits | doubling, 800 ms at most | most recoveries came within 750 ms of waits; past the 400 ms step a boot still tries about once a second, the spacing of the older failures in the kernel logs |
| total time | 90 s on the clock, from the start of the first try, tries included | about three times the longest stretch with no free 2 MiB block in the receipts (32.2 s) and ten times the longest recovery (8.5 s). A first boot runs under `TimeoutStartSec=600` and loads in 20 to 140 s on GB10, so 90 s more fits. A try whose wait would end past the bound is not made; the last try can pass it by its own length |
| `MemAvailable` floor | 256 MiB | a context takes 229 MiB; under that, or with `/proc/meminfo` unreadable, the answer stands and the boot fails at once |

**The respawn path.** A respawned worker creates its context the same way, and `WATCHDOG=1` is
withheld until it is live again (`WatchdogSec=180`, a ping every 90 s). Measured on GB10 from the
start line: 20 to 54 s to `listening` for Qwen3.8-27B with the DFlash2 drafter at 32,768 context
(21 boots), 115 to 140 s to `/readyz` with the MTP draft at 262,144 context and 32 sessions
(memra#1307's ten cells). So 90 s plus the load fits the watchdog for the first serving (144 s)
and not for the second (230 s). The budget is not sized to fit it: on that path a watchdog
restart and a give-up end the same way (the unit restarts, and the fresh boot runs under
`TimeoutStartSec=600` with the full 90 s), so a smaller budget would buy nothing there and
would only make first boots give up earlier. That the MTP serving's respawn load alone can miss
the watchdog is older than this change and is memra#1331.

Every boot step now names its driver call when it fails (`cuInit`, `cuDeviceGet`, the context,
`cuStreamCreate`, `cublasLtCreate`, each `cuModuleLoadData`), and on a unified-memory device an
out-of-memory answer carries the host's memory lines.

Why this is the fix and not a workaround: the allocation that fails is the kernel's, for a block
the kernel can make by reclaiming page cache it reports as available; the caller cannot ask for a
smaller or non-contiguous buffer, cannot make the driver wait, and holds nothing it could release.
Repeating the one call is what the kernel flag leaves to the caller.

## Rejected

- **Bounding the sleeps** (the first version: 10 s, then 30 s of waits). The tries are not free,
  and a kernel that stalls in each one would have stretched 41 tries far past the number in the
  record. The clock is injected, so the CPU tests run the bound with tries of any length.
- **30 s.** The receipts' own memory records show the state lasting longer than that three
  times.
- **Leaving the boot to the supervisor.** A restart costs the whole boot and reports a failure on
  a host with 119 GB available. The retry usually costs under a second.
- **Retrying the whole of `Engine::new`, or on every device.** Only context creation failed, in
  every failure of the record; the later steps allocate through paths with a small-page fallback. On a
  discrete card an out-of-memory answer at context creation means the card is full, and waiting
  for it would only delay the true error: `retry_reclaim_race` returns a discrete
  card's first answer unchanged.
- **Waiting for a MemFree level before creating the context.** MemFree was 0.8 to 6.7 GB at the
  failed tries and under 3 GB at over two hundred boots that came up; what was missing was a free 2 MiB block, which
  no reading taken before the call predicts (boots with none came up too, because the kernel made
  one in time).
- **Freeing memory first** (dropping the page cache, compacting through `/proc/sys/vm`, a
  transparent-huge-page allocation of our own to make the kernel compact). The first two are host
  settings a serving process must not touch; the third depends on the host's THP policy and does
  not reserve the block it frees.

## Discrete cards

Unchanged. `retry_reclaim_race(.., unified = false, ..)` makes one call and returns its result;
the test `discrete_card_keeps_its_first_answer` pins that, and
`the_wiring_retries_only_a_unified_device_only_out_of_memory_only_above_the_floor` pins that
`create_context` hands the policy the device's own attribute (an attribute the driver will not
read is a discrete card). On an NVIDIA GeForce RTX 5090 Laptop GPU (`rtx5090/discrete-b`, `tools/discrete_cell.sh`, the head's tree built for sm_120a): 10 of 10 boot-only starts came up with 0 `[ctx-retry]` lines, and a start on device 7, which the host does not have, ended with `cuDeviceGet on device 7 failed: DriverError(CUDA_ERROR_INVALID_DEVICE, "invalid device ordinal")` and no retry. The only new driver calls before the
context are `cuInit` and `cuDeviceGet`, which `CudaContext::new` makes anyway, and one
`cuDeviceGetAttribute`. The error text of a failed boot step changed on every device: it names the call.

## Limits

- The rate was measured in one state per cell: the page cache full, readers kept running. A host
  under a different load may fail more or less often.
- The retry covers context creation only. No other boot step failed here; each now names itself
  if it does.
- The recoveries measured are few (10 boots) and one of them took 8.5 s, so the tail is
  not known. A boot that outlasts the 90 s fails as before, with every try in its log. The cell
  that would measure the distribution, and a fixed 50 ms cadence against the ladder, is
  memra#1330.
- An active remedy was not tried: allocating more anonymous memory than is free, so that the
  kernel reclaims whole page-cache folios, might end the wait sooner, but it puts memory
  pressure on a shared host to help one boot.
- The floor reads `MemAvailable`, the kernel's estimate. A host whose reclaimable memory cannot be
  compacted into a 2 MiB block within 90 s still fails the boot, with the retries in its log.
