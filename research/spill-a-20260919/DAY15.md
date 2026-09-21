# WP-A day 15: the pinned host arena startup cells on the target card class (memra#385), and the door's cached-destination pair

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`. Start: tip `9a4a3acef` (day 14; reached `main` through
integ21), merged `origin/main` `a51e29abb` (#605 to #610) as `261241786`, pushed in the announced
`MEMRA_RELEASE_QUALIFICATION_MODE=development` mode (`day15/push-attempt1.log`: `UNQUALIFIED DEVELOPMENT:
refs/heads/lane/spill-a-20260919 at 2612417864219d4eaa5f5d37ba31a1438ace68e1; no GPU qualification claimed`,
`pre-push: skip recorded in .git/memra-gate-skips.log`); nothing here claims qualification, every cell below
is `executed-not-qualified`. Rulings read: 22 (the per-device pinned kind), 23 (it stands as landed; the
door's review reads its cost again on the target card with cached destinations), 24 (union resolves get a
set-difference check). Lane C's day 17 arena pair (`research/spill-c-20260919/DAY17.md`) is context, not
repeated: under the arena the pageable tier's first-touch step (about 35 ms on the first three demotes of a
boot) is gone because the arena pinned its 8 GiB once at boot in 1.3 s; that boot reserve is the call this
day measures.

## Pre-registration of the arena startup cells (written and committed before any GPU run)

**The call.** `PinnedHostArena::reserve` (`crates/memra-engine/src/pinned_host.rs:95-117`) is one
`cudarc::driver::result::malloc_host(bytes, CU_MEMHOSTALLOC_PORTABLE)`, that is one `cuMemHostAlloc` with
flag bits 1, timed into `alloc_ms`; no fill. The pinned kind the engine resolves for this card
(`PinnedKind::for_device` on the RTX PRO 6000 Blackwell class, ruling 22) is `Cached`, whose flag bits are 0,
so the arena's flags on this card are `PORTABLE | Cached` = 1, exactly what `reserve` passes today; the
write-combined bit (4) is absent from every arm below by construction and its absence is read back from the
driver in every correctness pass. The harness is `tools/pinned-host-reserve-bench.py` (ctypes on
`libcuda.so.1`, no engine binary, no engine change), extended today with the second candidate, the
correctness pass, the admissible sizing and the rule line; `research/spill-a-20260919/arena-ab.py` replays
the rule from `receipt.json` and must agree with the harness.

**Rig and regime.** BOX3, one RTX PRO 6000 Blackwell Server Edition at its 600 W limit, driver 580.178.04,
kernel `6.8.0-139-generic`. Host: 30 CPUs, one NUMA node, `MemTotal` 88.4 GiB, no swap, `HugePages_Total: 0`
(no hugetlb pool), THP mode `madvise` (so the hugepage arm is THP through `madvise(MADV_HUGEPAGE)` on an
anonymous mapping, not `MAP_HUGETLB`). At the snapshot before this section: `MemFree` 28.2 GiB, `MemAvailable`
85.1 GiB (57 GiB of page cache holding other lanes' artifacts), loadavg 0.00, GPU idle at 32 C / 31 W,
`/tmp/memra-gpu.lock` free, no `tmux` session, no `memra-server`. Every cell runs through
`tools/tier-battery.py --rig pro-single` (one `/tmp/memra-gpu.lock` hold per cell, 250 ms telemetry, the
lock proof), the regime recorded per the plan's rule 4 (`HOST-ARENA-STARTUP.md`): GPU temperature and power
at start and end, `loadavg`, `MemTotal`, `MemFree`, `MemAvailable` before every allocation, hugepage state
and THP mode, cgroup, NUMA, driver. Lane B or C may take the card: the runner retries a busy lock 15 x 120 s
and never signals a holder.

**Size, and why.** The reserve is sized by the engine's own admission rule, not by the B200 pair's 288 GiB
(`MemTotal` here is 88.4 GiB): `host_memory::check_headroom` admits a budget when `budget + MARGIN <=
MemAvailable` with `MARGIN = 32 GiB` (`crates/memra-server/src/worker/host_memory.rs`), so the largest arena
this box admits is `MemAvailable - 32 GiB` read at the harness's start, rounded down to 2 MiB (`--basis
admissible`), about 53 GiB today. That is 1.9x `MemFree`, so the first pin of the sitting evicts about 25 GiB
of page cache (other lanes' artifacts; a boot-time cost for their next server, not a correctness matter; no
lane is running now); the plan's rule 2 says that first pin is reported with the rest, never dropped, and
`MemFree`/`MemAvailable` are recorded before every allocation so the resident set of every measurement is
on the record. Pre-registered fallback: if the driver refuses the admissible size (any non-zero `CUresult`,
quoted), the cell reruns once at 75 percent of `MemFree` (the day-12 size) and this file says so; no other
parameter moves.

**Arms.** A = `single`: one `cuMemHostAlloc(bytes, PORTABLE)`, today's `reserve`. B1 = `chunked`: 8 threads,
each `cuMemHostAlloc(bytes / 8 rounded down to 2 MiB, PORTABLE)` on its own thread bound to the primary
context, joined; the reserve time is start of the first thread to the join. Pre-registered values: 8 threads,
chunk = bytes / 8 (about 6.6 GiB), the day-12 shape, chosen so the day-12 harness receipt (8 concurrent calls
completing one after another about 450 ms apart, read as the driver serializing pinned allocation) is
repeated as a decision-shape cell with correctness beside it; if the driver serializes, more threads cannot
help, so no thread sweep. B2 = `thp`: `mmap(NULL, bytes + 2 MiB, PROT_READ|PROT_WRITE,
MAP_PRIVATE|MAP_ANONYMOUS)`, the start aligned up to 2 MiB, `madvise(ptr, bytes, MADV_HUGEPAGE)`,
`cuMemHostRegister(ptr, bytes, CU_MEMHOSTREGISTER_PORTABLE)` (flag bits 1); the reserve time is `mmap` +
`madvise` + register wall clock (the register call alone recorded beside it, since the page faults that back
the mapping happen inside it); free is `cuMemHostUnregister` + `munmap`. The arm's backing is proven, not
assumed: after the register call the mapping's `AnonHugePages` is read from `/proc/self/smaps`; the arm's
verdict counts only if it is at least 90 percent of `bytes`, otherwise the arm is `hugepage-requested-not-
granted` and its cell is void.

**Design.** Two A/B cells, each one collector lock hold: cell `arena-chunked` (A against B1) and cell
`arena-thp` (A against B2). In each cell, first the correctness pass per arm (A then B), then order 1: A B x 5,
then order 2: B A x 5; every allocation freed before the next; N=5 per arm per order, N=10 pooled per cell;
the medians, min and max per arm and per order, and GiB/s, from the receipt. The correctness pass runs first
so the first pin's page-cache eviction is inside it and every timed measurement starts from the post-
eviction resident set (rule 2); its own reserve time is recorded and reported, not pooled.

**Correctness beside speed (per arm, per cell, full size).** Host fill: 2 MiB block `i` set to byte
`(7 i + 13) mod 256`; H2D into a device buffer of the same size (`cuMemAlloc`, `cuMemcpyHtoD`); the host
region wiped to zero; D2H back (`cuMemcpyDtoH`); every block compared with `memcmp` against its expected
block: `mismatched_blocks` must be 0. Driver flags read back with `cuMemHostGetFlags` on the region (per
chunk for B1; for B2 the call's `CUresult` and value are recorded as the driver returns them): the write-
combined bit (4) must be 0 in every read-back; for the two `cuMemHostAlloc` arms the PORTABLE bit (1) must
be set (the DEVICEMAP bit, 2, is expected on every UVA platform, as days 13 and 14 read). Device memory is
freed after the pass.

**Rule (fixed now, applied by the harness and by the replay, per cell).**
1. Integrity: every allocation `ok` and freed; `mismatched_blocks = 0` in both arms' roundtrips; the write-
   combined bit absent in every read-back; for `thp`, `AnonHugePages >= 0.9 x bytes` on the registered
   mapping. A failure makes the cell `void`, with the failing clause named.
2. Speed: the candidate's reserve time strictly below `single`'s at every one of the 10 pairs (compared at
   the timer's 1 us resolution, no rounding) and in both orders' medians.
3. Materiality: the pooled medians' ratio `single / candidate >= 1.10`.
Verdict: `wins-on-this-card` if 1, 2 and 3 hold; `inconclusive` if 1 holds and 2 or 3 fails; `void` if 1
fails. The harness prints one line, `ARENA-RESERVE rule cell=<name> candidate=<arm> ... candidate_arm=<verdict>`;
`arena-ab.py` recomputes it from `receipt.json` and must agree. No clause or threshold moves after a run.

**What the verdict is and is not.** A receipt for the target card class on this box's host (30 CPUs, one
NUMA node, 88.4 GiB, one driver build): no number here is divided into a number from any other box, and the
2x B200 pair's 288 GiB question is not answered here (3.3x this box's whole RAM, two devices' contexts,
another driver build). Landing rule, stated before the run: an arm lands as the engine's reserve path only
if it `wins-on-this-card` AND the change is bounded (one function, no new numeric program, no new flag, a
per-device default in the shape of ruling 22). By the plan's census `chunked` is outside that class whatever
it measures (it changes `Extents`, `ArenaInner.ptr`, `try_reserve_planes` and `Drop`: a list of bases, not
one), so a `chunked` win stops at the verdict with the landing named. `thp` touches `reserve` and `Drop` and
needs a boot check of THP availability with a typed refusal or a fall-back to today's call, so it is two
sites plus a policy, also outside "one function"; a `thp` win stops at the verdict with the landing and its
gates named. Either way the receipt is commented on #385 and the issue stays open.

## Pre-registration of the door's cached-destination pair (task 2, lane C's harness, only if the card is free)

The door's decide-by review (2026-10-05, `research/spill-c-20260919/HOSTPREFIX-DOOR.md`) still owes the
door's cost on the target card with cached destinations (ruling 23: C's day-16 WC pair is superseded on this
card class). The cell is C's `pro-single-day16/wc-cell.sh` statement for statement, with only the receipts
root (`/root/spill-receipts/a-day15`), the worktree (`/root/wt-a`) and the default port (18131, so it never
boots over a lane C server) changed, on a `memra-server` built from this lane's tree after the merge of
`origin/main` `a51e29abb` (engine source identical to `main`'s; `PinnedKind::for_device` resolves `cached` on
this card, so the door's two demote-side hashes and one promote-side hash run over cacheable memory).
Four boots OFF, ON, ON, OFF, seven requests each over C's two prompts, `MEMRA_PREFIX_CACHE_MB=256`,
`MEMRA_KV_HOST_MB=8192`, `MEMRA_KV_HOST_VERIFY` unset, default spec environment, one collector lock hold.
Pass/fail is C's `wc-pair.py` at the C lane tip on the mirrored cell: `WC PAIR REPLAY: PASS` (6 demotes and 5
promotes per boot, receipts present in the ON boots only, every promote with an inline demote). The timing
shape is C's, unchanged: demotes r2..r6 and promotes r3..r7 per boot (N=5 per arm per order, N=10 pooled),
the promote minus its inline demote, the first-touch step, per order and pooled medians with min and max,
the regime from the collector's sampler. No verdict rule: the reading is the review's; the numbers are
banked under this day and stated as its input. Expectation, stated and not a rule: the ON demote median
sits below day 16's 169 ms.
