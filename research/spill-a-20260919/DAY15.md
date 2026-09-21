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

## The arena startup cells: one sitting, one RTX PRO 6000 Blackwell Server Edition at 600 W (driver 580.178.04)

Both cells ran in the pre-registered shape through `tools/tier-battery.py --rig pro-single` (one
`/tmp/memra-gpu.lock` hold each, `lock.json`, 250 ms telemetry, `CELL.jsonl`, `status:
executed-not-qualified`, exit 0, `--validate` rc=0: `day15/collector-validate-*.log`), harness at tree
`f1aa058da` (`tools/pinned-host-reserve-bench.py`, ctypes on `libcuda.so.1`, no engine binary), zero lock
retries (the card was free; no `tmux` session, no `memra-server`, zero compute processes before the sitting),
receipts mirrored to `pro-single-day15/` without any binary. Size: `--basis admissible`, `MemAvailable -
32 GiB` at each cell's start, rounded down to 2 MiB: **56,916,705,280 B (53.01 GiB)** in `arena-chunked`
(`MemAvailable` 85.0 GiB, `MemFree` 28.0 GiB at start) and **56,969,134,080 B (53.06 GiB)** in `arena-thp`
(`MemAvailable` 85.1 GiB, `MemFree` 58.2 GiB at start, after the first cell's eviction). The driver
accepted the admissible size in both cells, so the pre-registered fallback never ran. Host: 30 CPUs, one
NUMA node, `MemTotal` 88.4 GiB, no swap, `HugePages_Total: 0`, THP `always [madvise] never`, defrag
`madvise`, `AnonHugePages` 0 at start, kernel `6.8.0-139-generic`, cgroup `user.slice` with no
`memory.max` file at the process's cgroup path (`arena-*/command.log` head). Replays `day15/arena-ab-*.log`
(`research/spill-a-20260919/arena-ab.py`): `ARENA AB REPLAY: PASS` on both cells, each `replay agrees with
the harness's verdict` and the collector's mirrored `command.log` carries the harness's rule line.

### `arena-chunked` (`single` against 8-thread `chunked`, N=5 per arm per order, N=10 pooled)

Regime: 1122 samples, GPU 32 to 36 C, 32.0 to 106.2 W under the 600 W limit, SM 180 to 2362 MHz (the
card idles; the harness only holds its primary context and copies once per correctness pass); host loadavg
0.09 before, 1.09 after. Correctness pass (full size, before the timed pairs; its reserve time not pooled):
`single` byte exact, 27,140 blocks, 0 mismatched, `cuMemHostGetFlags` = 3 (PORTABLE|DEVICEMAP, the
write-combined bit absent), reserve 9437.1 ms, H2D 53.8 GiB/s, D2H 52.6 GiB/s; `chunked` byte exact, 27,136
blocks across 8 chunks of 7,113,539,584 B (bytes / 8 rounded down to 2 MiB, so the chunked region is 8 MiB
smaller than the single one; the pooled ratio compares the reserve times as pre-registered), 0 mismatched,
flags 3 on all eight chunks, reserve 9525.1 ms. That first pin evicted the page cache the size rule said it
would: `MemFree` 28.0 GiB at start, 56.3 to 58.3 GiB before every timed allocation.

| Order | Pair | `single` reserve (ms) | `chunked` reserve (ms) | chunk completions (ms, one row) |
|---|---|---|---|---|
| AB | 1 | 9261.329 | 9144.351 | 1223, 2370, 3489, 4603, 5710, 6805, 7904, 9143 |
| AB | 2 | 8877.996 | 8831.017 | 1205, 2306, 3389, 4483, 5565, 6649, 7740, 8830 |
| AB | 3 | 8786.797 | 8826.381 | 1206, 2311, 3393, 4472, 5551, 6637, 7734, 8826 |
| AB | 4 | 8847.079 | 9025.955 | 1218, 2367, 3493, 4610, 5718, 6826, 7925, 9025 |
| AB | 5 | 8828.417 | 8831.741 | 1207, 2312, 3398, 4483, 5571, 6659, 7748, 8831 |
| BA | 1 | 8862.753 | 8839.646 | 1205, 2310, 3393, 4482, 5576, 6663, 7752, 8839 |
| BA | 2 | 8888.179 | 8835.071 | 1202, 2306, 3380, 4468, 5556, 6645, 7751, 8834 |
| BA | 3 | 8859.376 | 8824.031 | 1209, 2309, 3392, 4474, 5553, 6636, 7727, 8823 |
| BA | 4 | 8878.074 | 8830.772 | 1213, 2322, 3412, 4494, 5577, 6661, 7743, 8829 |
| BA | 5 | 8899.597 | 8804.635 | 1209, 2314, 3395, 4481, 5561, 6645, 7722, 8804 |

| arm | n | pooled median | AB median (N=5) | BA median (N=5) | min | max | GiB/s | free median |
|---|---|---|---|---|---|---|---|---|
| `single` (today's `reserve`) | 10 | **8870.374 ms** | 8847.079 | 8878.074 | 8786.797 | 9261.329 | 5.98 | 3561.4 ms |
| `chunked` (8 threads) | 10 | **8831.379 ms** | 8831.741 | 8830.772 | 8804.635 | 9144.351 | 6.00 | 3571.0 ms |

Verdict, verbatim: `ARENA-RESERVE rule cell=arena-chunked candidate=chunked bytes=56916705280 n_per_order=5
pooled=10 alloc_ok=true roundtrip_exact_single=true roundtrip_exact_candidate=true wc_bit_absent=true
flags_read_ok=true portable_bit_set=true hugepage_fraction=na hugepage_ok=na cand_below_single_pairs=7/10
medians_both_orders=true pooled_single_ms=8870.374 pooled_candidate_ms=8831.379 ratio=1.0044 floor=1.10
materiality=false candidate_arm=inconclusive`. Replay: `FAIL rule 2a: chunked below single at every pair:
7/10`, `FAIL rule 3: pooled ratio single/chunked = 1.0044 >= floor 1.10`, `candidate chunked on this card:
INCONCLUSIVE`, `ARENA AB REPLAY: PASS`. The eight concurrent `cuMemHostAlloc` calls complete about 1.1 s
apart in every one of the 10 rows (6.6 GiB per step, 6 GiB/s), the day-12 pattern at 2.4x the size: this
driver serializes host-pinned allocation across threads, so eight callers do not use eight cores for the
page walk and the two arms are the same program at 0.4 percent, inside the single arm's own spread. The
day-12 harness result is repeated as a decision-shape cell; candidate 1 does not move the startup on this
box.

### `arena-thp` (`single` against `thp`, N=5 per arm per order, N=10 pooled)

Regime: 783 samples, GPU 36 C throughout, 86.8 to 106.6 W under 600 W, SM 2347 to 2355 MHz; loadavg 1.09
before, 1.21 after. Correctness pass: `single` byte exact, 27,165 blocks, flags 3, reserve 8943.0 ms;
`thp` byte exact, 27,165 blocks, 0 mismatched, `cuMemHostGetFlags` on the registered mapping returns
CUresult 0 with flags 3 (the driver reports PORTABLE|DEVICEMAP for registered memory as for allocated;
the write-combined bit absent), `mmap` 0.03 ms, `madvise` 0.006 ms (rc 0), **`cuMemHostRegister` 18,335.2
ms with `AnonHugePages` 36,240,883,712 B of 56,969,134,080, fraction 0.6361**: the kernel backed 64
percent of the mapping with huge pages and, under defrag `madvise`, reclaimed about 22 GiB of page cache
to do it (`MemFree` 58.2 GiB at the cell's start, 79.7 to 83.8 GiB before every timed allocation). H2D
53.8 GiB/s, D2H 52.6 GiB/s through both arms' regions.

| Order | Pair | `single` reserve (ms) | `thp` reserve (ms) | `thp` register (ms) | `thp` huge fraction | `MemFree` before single / thp (GiB) |
|---|---|---|---|---|---|---|
| AB | 1 | 8875.271 | 2875.962 | 2875.932 | 1.0000 | 80.0 / 80.1 |
| AB | 2 | 9012.341 | 2877.137 | 2877.102 | 1.0000 | 79.9 / 80.1 |
| AB | 3 | 8909.944 | 2900.193 | 2900.163 | 1.0000 | 79.9 / 79.9 |
| AB | 4 | 8878.534 | 2905.274 | 2905.246 | 1.0000 | 79.7 / 80.1 |
| AB | 5 | 8924.322 | 2897.929 | 2897.902 | 1.0000 | 79.8 / 80.1 |
| BA | 1 | 8887.340 | 2894.138 | 2894.107 | 1.0000 | 79.9 / 79.8 |
| BA | 2 | 8824.353 | 2869.960 | 2869.931 | 1.0000 | 79.9 / 80.1 |
| BA | 3 | 8866.670 | 2874.502 | 2874.472 | 1.0000 | 79.7 / 80.0 |
| BA | 4 | 8811.930 | **6629.635** | 6629.607 | **0.8435** | 83.5 / 79.9 |
| BA | 5 | 8772.534 | 2869.277 | 2869.249 | 1.0000 | 83.5 / 83.8 |

| arm | n | pooled median | AB median (N=5) | BA median (N=5) | min | max | GiB/s | free median |
|---|---|---|---|---|---|---|---|---|
| `single` (today's `reserve`) | 10 | **8876.902 ms** | 8909.944 | 8824.353 | 8772.534 | 9012.341 | 5.98 | 3558.5 ms |
| `thp` (mmap, madvise, register) | 10 | **2885.637 ms** | 2897.929 | 2874.502 | 2869.277 | 6629.635 | 18.39 | 562.5 ms |

Verdict, verbatim: `ARENA-RESERVE rule cell=arena-thp candidate=thp bytes=56969134080 n_per_order=5 pooled=10
alloc_ok=true roundtrip_exact_single=true roundtrip_exact_candidate=true wc_bit_absent=true
flags_read_ok=true portable_bit_set=true hugepage_fraction=0.6361 hugepage_ok=false
cand_below_single_pairs=10/10 medians_both_orders=true pooled_single_ms=8876.902 pooled_candidate_ms=2885.637
ratio=3.0762 floor=1.10 materiality=true candidate_arm=void (hugepage-requested-not-granted)`. Replay:
`FAIL rule 1e: thp backed by huge pages at >= 0.9 (min seen 0.6361494570219032)`, every other clause `ok`,
`candidate thp on this card: VOID (HUGEPAGE-REQUESTED-NOT-GRANTED)`, `ARENA AB REPLAY: PASS`.

**Reading, as pre-registered.** The cell is void under rule 1e and the rule is applied as written: the
minimum huge-page fraction over every `thp` reserve of the cell is 0.6361 (the correctness pass), and one
timed row (BA pair 4) fell to 0.8435, so the clause fails even without that pass. Nothing is relaxed after
the result. What the receipt shows, stated as observations: (1) when the kernel grants the whole mapping
as huge pages (9 of 10 timed rows, `MemFree` about 80 GiB), `cuMemHostRegister` pins 53 GiB in
2,870 to 2,905 ms, 18.4 GiB/s, 3.1x today's call, and the free is 0.56 s against
3.56 s; (2) when it grants less, the arm is slower than today's call at the low end: 18.3 s at 64 percent
(the first THP pin on a box whose page cache held 57 GiB of other lanes' artifacts) and 6.6 s at 84
percent (BA pair 4, with about 80 GiB free), because the register call then pays page-cache reclaim and
compaction inside its window; (3) the production boot reserves the arena after the model load, which is
exactly the state in which the first pin here ran (a page cache full of a model artifact), so on a box
shaped like this one the cold regime is the production regime. The mechanism is real and the win is large
where the huge pages exist; whether they exist at reserve time is a property of the box's memory state,
not of the call, and the arm as measured today does not control it. Rule 3's 10 percent floor was never
the question; rule 1e was.

### What landed, and what a landing would need

No engine change landed today. `chunked` measured flat under the rule (inconclusive at 1.0044) and was
outside the bounded class before the run (a list of bases). `thp` is void under its integrity clause, and
the receipt says what a landing needs, in order: (a) a reserve whose huge-page backing does not depend on
the page cache at boot, which on a box like this means a reserved hugetlb pool (`HugePages_Total`,
`MAP_HUGETLB`: a pool is carved at boot and never competes with the page cache; this box has none) or a
pre-registered page-cache regime for the THP form (the cell's definition would include the state before
the reserve, and the fall-back when the fraction read from `smaps` is below the floor); (b) the engine
change itself: `PinnedHostArena::reserve` (mmap, madvise, `cuMemHostRegister`, the fraction read back and
printed on the arena startup line, a typed refusal or a fall-back to today's `cuMemHostAlloc` when the
backing is not granted) and `Drop` (`cuMemHostUnregister` plus `munmap`, or `free_host`, per backing kind),
two sites plus a policy, so not "one function"; the fall-back is a second reserve program for one arena
and would carry its own gate; (c) the decision cell on the 2x B200 pair at the production budget with the
same harness (`--basis admissible --arms single,thp --roundtrip --cell ...`, plus a `MAP_HUGETLB` arm if
that box exposes a pool), under the same rule. Until then today's one `cuMemHostAlloc(PORTABLE)` stays the
reserve path; `PinnedKind::for_device` is unaffected (the arena's flags on this card are already the cached
kind plus PORTABLE, and every read-back today was 3).

## The door's cached-destination pair (task 2, lane C's harness on this tree)

The card was free after the arena cells, so the pair ran inside the same sitting: `pro-single-day15/wc-pair`,
C's `pro-single-day16/wc-cell.sh` with the three path changes named above (`pro-single-day15/wc-cell.sh`,
`diff` against C's file: the header comment and three lines), one collector lock hold of 108 s
(`marks.tsv` 16:58:02.916Z to 16:59:50.427Z), four boots `o1-off`, `o1-on`, `o2-on`, `o2-off`, seven
requests each, `memra-server` `5ea68afa08cf3330…` built on the box from `ad11f2a70` (`pro-single-day15/build/`,
`dirty.txt` empty; `git diff ad11f2a70 f1aa058da -- crates/` is empty, so the binary is this lane's tree
after the merge of `origin/main` `a51e29abb`, engine source identical to `main`'s), the model page-cached
(boots under 10 s). Regime: 430 samples at 250 ms, 36 to 51 C, power draw at most 491 W under 600 W, SM
2160 to 2422 MHz. `--validate` rc=0. Replay with C's `wc-pair.py` at the C lane tip `a6b62c998`
(`day15/wc-pair-replay.log`): **`WC PAIR REPLAY: PASS (12 checks)`**, every boot 6 demotes and 5 promotes,
receipts present in the ON boots only (D2H 6, H2D 5 per boot), every promote with an inline demote in its
window. Identity: r1..r7 response texts byte-identical across the four boots (r3..r7 `cached_tokens` 64,
r1 and r2 0, identical), demote byte counts `64 tok/159.9 MB` in every boot; no `TIER DISABLED`, `refused`,
`WARNING`, `VERIFY FAILED` or `leaked` in any log; each ON promote's H2D receipt carries the D2H digest of
the same entry's demote (`5b58bfb9…` on `seq=1` and `seq=2` of `o1-on`). The door's boot line names the
program identities and `host tier armed; KV plane D2H through the transfer engine`; the binary's
destinations are cached on this card by `PinnedKind::for_device` (the day-14 target-card sitting read
driver flags 2 for every default lease on this card; the server prints no per-lease flag line).

| Boot | demotes r2..r7 (ms) | promotes r3..r7 (ms) | promote minus inline demote (ms) |
|---|---|---|---|
| o1-off | 37.2, 41.7, 41.1, 6.2, 6.8, 6.2 | 46.1, 45.5, 10.6, 11.3, 10.6 | 4.4, 4.4, 4.4, 4.5, 4.4 |
| o1-on | 114.2, 118.4, 118.4, 83.0, 82.1, 81.9 | 124.2, 124.1, 88.8, 88.0, 87.7 | 5.8, 5.7, 5.8, 5.9, 5.8 |
| o2-on | 113.0, 117.4, 117.4, 82.1, 82.7, 81.8 | 123.2, 123.1, 87.9, 88.5, 87.5 | 5.8, 5.7, 5.8, 5.8, 5.7 |
| o2-off | 38.4, 43.1, 42.1, 6.1, 6.9, 6.1 | 47.6, 46.6, 10.4, 11.4, 10.4 | 4.5, 4.5, 4.3, 4.5, 4.3 |

| Line (C's shape) | pooled OFF (N=10) | pooled ON (N=10) |
|---|---|---|
| `demote:` r2..r6 | median **37.8** ms (6.1 to 43.1) | median **113.6** ms (82.1 to 118.4) |
| steady-state demote r5..r7 (N=6) | 6.1 to 6.9 | 81.8 to 83.0 |
| `promote:` r3..r7 (window contains the inline demote) | **11.4** (10.4 to 47.6) | **88.7** (87.5 to 124.2) |
| promote minus its inline demote | **4.4** (4.3 to 4.5) | **5.8** (5.7 to 5.9) |

The two orders agree within 1.2 ms at every position. The first-touch step of about 35 ms on r2..r4 is in
both arms (C's day-16 and day-17 finding; the fresh pinned region of the pageable tier). Read on the
steady-state rows, the door's cost at demote on this card with cached destinations is about 76 ms per
160 MB entry in this window (82 against 6.1 ms), and its cost on the promote's own share is about 1.4 ms
(5.8 against 4.4). The prior record on this card class, C's day-16 pair with write-combined destinations
(another sitting on the same box, so a record beside this one and not a same-window comparison), read
demote ON 169.2 ms pooled with a steady state of 136 to 140 ms, and promote minus inline demote 33.2 ms.
Which part of the remaining 76 ms is the two hashes over cacheable memory and which the ticket lifecycle
is the hash-speed micro-cell C's review owes; not measured here, no split inferred. Banked as the review's
input, not a verdict.

## Side effect of the sitting, stated

The two admissible-size pins and the THP reclaim took the box's page cache from 57 GiB to 17 GiB
(`free -g` before and after the sitting; `MemFree` 28 GiB before, 69 GiB after). Other lanes' next server
boot re-reads its artifact from disk once; no process or lock was touched. The lane's `tmux` session was
closed after the driver's `done`; `/root/wt-a` is clean at `f1aa058da` on `lane-a-day15`;
`/root/spill-receipts/a-day15/bins/` holds the server binary (not mirrored).

## CPU gates on the tree at `f1aa058da` (`day15/gates/`, under `systemd-run --user --scope -p CPUQuota=1200% -p MemoryMax=28G`)

| gate | exit | verbatim tail |
|---|---|---|
| `cargo fmt --all -- --check` | 0 | (no output; no Rust changed today) |
| `bash tools/check-flags.sh` | 0 | `check-flags: every runtime MEMRA_* name resolves against 'docs/FLAGS.md' (no grandfather list)` (no new `MEMRA_*` read: the harness and the replay read none) |
| `python3 tools/check-public-boundary.py check` | 0 | `public-boundary: 582 matches (582 grandfathered, 0 new).` |
| `bash tools/docs-registry-census.sh` | 0 | `docs-registry-census: flags-table-census: docs/FLAGS.md tables=58 rows=905, every row matches its header` |
| `git diff --check` | 0 | (no output) |
| `python3 -m py_compile` on the harness and the replay; `bash -n` on the four cell scripts | 0 | (no output) |
| em-dash scan over the day's prose, harness, replay and scripts | | none |

Local harness smokes on the RTX 5090 Laptop GPU before the box (`rtx5090-day15/`, through the collector
with `/tmp/memra-5090.lock`, 1 GiB, N=1 per order, not evidence for any claim): `smoke-chunked` and
`smoke-thp` are the harness's development crash (`TypeError: unsupported operand type(s) for +: 'c_void_p'
and 'int'`, the roundtrip adding an offset to a ctypes pointer object; kept as the record), `smoke2-*` the
fixed harness end to end (both byte exact, flags 3, the rule line printed, `arena-ab.py` `PASS` on both).

## Scope and effort

Done: the two pre-registered #385 cells on the target card class at the engine-admissible size (53 GiB)
with correctness beside speed and the rule applied as written (`chunked` inconclusive at 1.0044 with the
serialized completions; `thp` void on the huge-page clause, 3.08x where granted, slower than today's call
where not, the regime dependence named); the landing rule kept (no engine change; what a landing needs
stated); the door's cached-destination pair on C's harness (`WC PAIR REPLAY: PASS`, demote ON 113.6 ms
pooled, steady 82 against 6.1, promote's own share 5.8 against 4.4); receipts mirrored and validated;
records. Not done, stated: the 2x B200 decision cell (another box); a hugetlb-pool arm (this box has no
pool); the hash-versus-ticket split under the door (C's micro-cell). About 1.5 agent-hours against the
4-hour budget (the card was free for the whole sitting; the two arena cells took 4.8 and 3.5 minutes,
the pair 108 s).
