# Session C day seventeen: memra#586 repaired with its red/green fixture; the arena cell pre-registered; the WC item closed by pointer

Scope (lead brief, day 17): (1) memra#586, the CPU mirrored prefetch double-decrement, test first, then the
minimal repair, proven on the local rig and as a pass/fail cell on the target rig's host; (2) one
pre-registered arena-path cell on the target card; (3) the WC decision item closed by pointer at lane A's
`docs/decisions/PINNED-DESTINATIONS.md` (ruling 23); (4) the day's records. Tree: lane merge of main
`fa5185d90` (`035773c1b`, #605 to #609), the #586 repair `0b55f7b39`. Every push of this lane today is in
the announced `MEMRA_RELEASE_QUALIFICATION_MODE=development` mode (the #589 hook prints `UNQUALIFIED
DEVELOPMENT ... no GPU qualification claimed` and appends a `log_skip` row): no qualification is claimed
by anything here; every cell is `executed-not-qualified`. Nothing here is a support state.

## Push mode, stated

`git push origin lane/spill-c-20260919` at `035773c1b` and at `0b55f7b39` ran with
`MEMRA_RELEASE_QUALIFICATION_MODE=development`, as the brief directs. Verbatim from the hook:
`UNQUALIFIED DEVELOPMENT: refs/heads/lane/spill-c-20260919 at 035773c1b1722b7f61f671cd2fa7bed55bbf2aeb;
no GPU qualification claimed` then `pre-push: skip recorded in .../.git/memra-gate-skips.log`; the same
shape at `0b55f7b39b26ae58b5830ebe695656f5b3962b95`, at `0222d24c4b6ec70b68b90fb35c2c6812e4fccac8` (the arena
pre-registration, pushed before the run) and at the day-docs tip after this section. The day-16 tip that
the two earlier refusals left unpushed (`bea6bd2ff`) went out inside the first push.

## memra#586: the CPU mirrored prefetch double-decrement (`0b55f7b39`)

**Confirmed from source** (`tools/memra_cpu_experts.cpp` at `035773c1b`). `memra_cpu_expert_prefetch_v2`
builds, for a projection whose `alternate_read_fd >= 0 && length >= 8192` (a mirrored direct read), TWO
`IoJob`s (the primary half on `read_fd`, the alternate half on the mirror's fd, `jobs.push_back` twice),
stores `projection_pending[slot] = 2`, and charges `prefetch_inflight().fetch_add(1)` ONCE per projection.
`IoPool::worker_loop` ran `prefetch_inflight().fetch_sub(1)` per JOB, outside the
`projection_pending[index].fetch_sub(1) == 1` final-half branch, so a mirrored projection released two
charges for one, the signed counter went negative, and the admission check
`prefetch_inflight().load() >= max_inflight` compared the cap against a number below the true in-flight
count. `memra_cpu_expert_prefetch_stats_v2` clamps the counter with `std::max(0, ..)` into its `uint64_t`
out-parameter, so `inflight == 0` in the CI bank could never establish balance. The issue's frozen SHAs are
not in this repository (its "scoped" path is not in this tree); the raw mirrored path here carries the same
defect, at one site.

**The fixture first** (`tools/test_cpu_expert_prefetch.sh`, `tools/memra_cpu_expert_prefetch_test.cpp`).
The test includes the production translation unit (the shm test's pattern), so it reads the SIGNED counter
`prefetch_inflight()` directly (never the clamp) and adds no symbol to the companion ABI. The unit's one
`pread` call (`pread_exact`) is renamed at preprocessing time to a forwarding function that can HOLD a read
at its entry and FAIL one on demand; a worker that enters a held read has finished every job it took before
it, so "every alternate half held, queue empty" is the deterministic point at which the projections' charge
is read (no sleep, no race). Fixture: a 260 KB source on the root filesystem and a byte-identical mirror on
a second device (`MirrorFiles::resolve` refuses a mirror on the source's `st_dev`), both read `O_DIRECT`
(`MEMRA_CPU_EXPERT_IO=direct`), the v2 mirror map written by the test from `file_key()`. Cells:

| Cell | Shape | Unfixed source (`prefetch-test-red.log`) | Repaired (`prefetch-test-green.log`) |
|---|---|---|---|
| `barrier` | three mirrored 64 KiB projections, `MEMRA_CPU_EXPERT_IO_THREADS=3`, `MEMRA_CPU_EXPERT_PREFETCH_MAX_INFLIGHT=3`; every alternate half held once its primary half landed; then a fourth projection offered under the cap; then release, take every buffer out of the annex and compare its bytes with the file, drain, retry the fourth | `barrier: primary halves completed=3 alternate halves held=3` then `FAIL: REGRESSION memra#586 at barrier: primaries complete, alternates held: inflight_signed=0 expected 3` | `inflight_signed=3 expected=3`, `extra projection admitted=0 expected=0`, `drained after the three landed: inflight_signed=0`, retry admitted and landed, `drained after the retry landed: inflight_signed=0`, `BARRIER_OK` |
| `failure` | one mirrored projection on ONE worker, its alternate half held then failed with `EIO`; a direct single-job sentinel queued behind it and held at ITS entry (the worker reaches it only after the failed projection's `abort_read` and release) | `FAIL: REGRESSION memra#586 at failure: primary half complete, alternate half held: inflight_signed=0 expected 1` | `inflight_signed=1 expected=1` at the hold, `2` with the sentinel queued, `alternate half failed with EIO, sentinel read entered: inflight_signed=1 expected=1`, the failed key absent from the annex, sentinel landed, `drained: 0`, the failed projection re-admitted on retry and landed, `drained after the retry landed: inflight_signed=0`, `FAILURE_PATH_OK` |
| `parity` | three buffered non-mirrored projections (one job each), held at entry under three workers | `inflight_signed=3 expected=3`, `drained: inflight_signed=0 expected=0`, `PARITY_OK` | identical |
| summary | | `cpu expert prefetch accounting tests: 9 FAILURE(S)` | `cpu expert prefetch accounting tests: ALL GREEN` |

The fixture runs every cell after a failure and its summary carries the count. Local rig, CPU only, under
`systemd-run --user --scope -p CPUQuota=1200% -p MemoryMax=28G`; logs `day17-local/`.

**The repair** (one hunk in `worker_loop`): `prefetch_inflight().fetch_sub(1)` moves INSIDE the
`projection_pending[index].fetch_sub(1) == 1` branch, after the annex `complete_read` or `abort_read`. One
accounting rule for every path: one charge per projection at submit, one release per projection with its
final half, success or failure; a non-mirrored projection (one job) is unchanged. The clamp in the stats
function stays (the unsigned ABI field cannot carry a negative; the comment now says it is not the
balancing oracle). Not changed, stated: the submit-side exception path (an exception between an earlier
projection's `begin_read`/`fetch_add` and `pool.submit` leaves those charges and annex claims behind) is
pre-existing and out of this repair; the issue's acceptance names the worker's failure and unwind paths,
which the `failure` cell covers.

**Proof.** Local: the table above; `tools/build_cpu_expert_companion.sh` with the production flags
(`-O3 -march=native -Wall -Wextra -Wpedantic -Werror`) builds clean (`companion-build.log`, sha256
`72b4a3d9...`); the CI CPU bank `cpu_native_check` against it: `detached prefetch (annex promote +
bit-identity): PASS` and `memra native CPU quant check: ALL GREEN` (`cpu-native-check.log`). The fixture is
wired into `ci.yml`'s engine-tests job right after `cpu_native_check` (it refuses to run where a second
`O_DIRECT`-capable filesystem is absent; it does not skip) and has a `docs/TESTING.md` Fixtures paragraph.
Target rig's host: the `cpubank` cell below (no GPU cell exists for the companion in `tier-battery.py`;
the collector run is for the lock proof and the record, pass/fail, not timed). Issue comment: the receipt;
the issue stays open for the lead's integ.

## The arena cell: pre-registration (written before any GPU run; the cell follows)

**Open item** (`STATE.md` NEXT, `HOSTPREFIX-DOOR.md` "what remains"): the startup pinned arena
(`MEMRA_GLM5_TP_KV_HOST=1`, one portable cacheable `cuMemHostAlloc` of the whole `MEMRA_KV_HOST_MB` budget at
boot, whole-image leases carved from it, no per-demote pin, no zero-fill) is REFUSED with the door at boot
(`host_tier_arena_refusal`): its fixed backing is not governor-charged and its slices are not leases. Before
the decide-by the review must know whether routing the arena under the door (a lease handoff, engine work)
is worth pricing, which turns on whether the arena is the cheaper host shape on the target card at all.

**Question.** On the target card, door OFF in both arms, does the startup arena change the demote and
promote wall times of the same ~160 MB entry against the pageable tier (`PinnedHostBuf::new` per plane at
demote), and in particular does it remove the first-touch step (about 35 ms on the first three demotes of
every boot, seen in BOTH arms of the day-16 WC pair and attributed to page-locking and zero-filling fresh
pinned regions)?

**Rule, fixed now.** Observations exactly as the WC pair: demotes r2..r6 (N=5) and promotes r3..r7 (N=5)
per arm per order from the `[prefix-host] demote:`/`promote:` lines; the promote's inline demote is the
`demote:` line inside its window. Derived per boot: first-touch step = median(r2..r4 demotes) minus
median(r5..r7 demotes). Verdict clauses, applied by `arena-pair.py` from the mirrored logs:
1. `arena_first_touch_absent` if BOTH arena boots show a step below 10 ms while BOTH pageable boots show a
   step of at least 20 ms (the day-16 shape); `first-touch step present in arena arm` if the arena boots
   show 10 ms or more; `void (pageable step absent this sitting)` if the pageable arm shows no step (the
   comparison has no baseline then).
2. `arena_not_slower` if the arena arm's pooled (N=10) steady-state demote median (r5..r7) AND its pooled
   promote-minus-inline-demote median are each within +10 % of the pageable arm's or below; otherwise
   `arena_slower`.
3. Integrity gates that VOID the verdict: 6 demotes and 5 promotes per boot; the arena boot line
   (`[prefix-host DEBUG] arena startup:`) present in arena boots only; no `TIER DISABLED`, `refused`,
   `WARNING`, `arena reserve failed`, `VERIFY FAILED`, `leaked` in any log; r3..r7 response texts and
   `cached_tokens` identical across the four boots; demote byte counts equal across arms.
No clause is read from the result and no threshold moves after the run.

**Shape.** `arena-cell.sh`, the day-16 `wc-cell.sh` statement for statement with the arm variable changed:
four boots `o1-page`, `o1-arena`, `o2-arena`, `o2-page` (both orders), the same binary (my tree's
release `memra-server`), the same two prompts and seven requests per boot, `MEMRA_PREFIX_CACHE_MB=256`,
`MEMRA_KV_HOST_MB=8192` (the arena reserves 8 GiB at boot after `check_headroom`; the host has 85 GB
available against the 32 GiB margin), default spec environment, `MEMRA_KV_HOST_VERIFY` unset,
`MEMRA_KV_HOST_CONTRACTS` never set. One collector lock hold (`tools/tier-battery.py --rig pro-single
--external-lock`, `/tmp/memra-gpu.lock`), the collector's 250 ms telemetry, regime recorded. N=5 per arm
per order, N=10 pooled, one card, one window, `executed-not-qualified`. Budget: about 2 agent-hours
including the replay; if the cell cannot run in that budget, this pre-registration and the harness stand
and say so.

## The arena cell: results (`pro-single-day17/arena-pair/`, replay `arena-pair.py`: `ARENA PAIR REPLAY: PASS (18 checks)`)

One collector lock hold (50 s; the card was free, no wait), four boots `o1-page`, `o1-arena`, `o2-arena`,
`o2-page`, seven requests each, binary `6489c4f8...` built from `0b55f7b39` (the pre-registration commit
`0222d24c4` changed no code), the model page-cached (8 s boots), `MEMRA_KV_HOST_VERIFY` unset, the door
never set. Regime over the window (201 samples at 250 ms, `command.gpu.csv`): temperature 33 to 51 C,
power draw at most 493 W under the 600 W limit, SM clock 180 (idle between boots) to 2422 MHz. The arena
boots reserved 8 GiB in `reserve_ms=1317.569` and `1320.227` (`fill_ms=0.000`), after the headroom lines
(`MemAvailable` about 89.7 GB against the 32 GiB margin). Collector `--validate` rc=0
(`arena-pair-validate.log`). `executed-not-qualified`.

| Arm | demotes r2..r7 (ms) | promotes r3..r7 (ms) | first-touch step |
|---|---|---|---|
| o1-page | 38.2, 42.6, 41.5, 6.1, 6.9, 6.0 | 47.1, 46.0, 10.8, 11.4, 10.5 | 35.4 |
| o1-arena | 3.6, 6.2, 6.2, 6.2, 6.2, 6.1 | 6.6, 6.6, 6.6, 6.6, 6.6 | 0.0 |
| o2-arena | 3.6, 6.2, 6.2, 6.2, 6.2, 6.2 | 6.7, 6.7, 6.7, 6.6, 6.7 | 0.0 |
| o2-page | 37.9, 42.5, 41.9, 6.0, 6.7, 6.1 | 47.0, 46.3, 10.5, 11.2, 10.5 | 35.8 |

| Line | pooled PAGE (N=10) | pooled ARENA (N=10) |
|---|---|---|
| `demote:` r2..r6 | median 38.0 (6.0 to 42.6) | median 6.2 (3.6 to 6.2) |
| steady-state demote r5..r7 (N=6) | 6.1 (6.0 to 6.9) | 6.2 (6.1 to 6.2) |
| `promote:` r3..r7 (the window contains the inline demote) | 11.3 (10.5 to 47.1) | 6.6 (6.6 to 6.7) |
| promote minus its inline demote | 4.5 (4.4 to 4.7) | 0.4 (0.4 to 0.5) |

The two orders agree to within 0.4 ms on every position. Identity: r3..r7 response texts and
`cached_tokens` identical across the four boots; demote byte counts equal (`64 tok/159.9 MB`; the entries
are 64 tokens today, day 16's were 89 and 86, main's #602 moved both capture sites onto the 32-token grid);
no `TIER DISABLED`, `refused`, `WARNING`, `VERIFY FAILED` or `leaked` in any log; the arena boot line present
in the arena boots only.

Verdict, verbatim from the replay of the pre-registered rule:

`ARENA-PAIR rule first_touch_page_o1=35.4 first_touch_page_o2=35.8 first_touch_arena_o1=0.0 first_touch_arena_o2=0.0 steady_demote_page=6.1 steady_demote_arena=6.2 demote_page=38.0 demote_arena=6.2 promote_page=11.3 promote_arena=6.6 promote_excl_page=4.5 promote_excl_arena=0.4 N=5/arm/order pooled=10 orders=2 window=50s temp_c=33..51 power_max_w=493 power_limit_w=600.00 W identity=ok integrity=ok -> arena_first_touch_absent; arena_not_slower`

Observations, stated as observations:

1. **The first-touch step is the fresh pinned region.** Under the arena both boots show `0.0` (every
   demote from r3 on is 6.2 ms and r2 is 3.6); under the pageable tier both boots show the day-16 step
   (35.4 and 35.8 ms on r2..r4). That confirms day 16's attribution: `PinnedHostBuf::new` page-locks and
   zero-fills a fresh 160 MB region per demote until the tier recycles one (from r5), while the arena
   pinned its 8 GiB once at boot (1.3 s, `fill_ms=0`). The cost moved, it did not vanish: 35 ms times
   three demotes against 1.3 s of boot for this budget.
2. **Steady-state demote is equal.** 6.1 against 6.2 ms (r5..r7, N=6 each): the D2H into a recycled
   pinned buffer and into an arena slice cost the same on this card; the pageable tier reaches the arena's
   speed once it recycles.
3. **The promote's own share is 0.4 ms under the arena against 4.5 ms pageable.** The two arms run
   different promote programs (the arena's whole-image lease path against `plane_up`'s per-plane
   `alloc_u8` plus `htod_u8_into`), and the OFF pageable 4.5 ms was read on day 16 as allocations plus an
   asynchronous launch whose DMA completes in the next demote's window. Which side of that the arena
   saves is a census item for the review, not a claim here.
4. **r2 under the arena is 3.6 ms, r3..r7 are 6.2.** The first demote of each arena boot is faster than the
   rest by 2.6 ms in both orders; unexplained, N=1 per boot, recorded.

For the decide-by review: the arena is the cheaper host shape on the target card by the first-touch step
only (about 35 ms on each of the first three demotes per boot, a per-fresh-region cost the pageable tier
stops paying once it recycles), and by 4 ms on the promote's own share whose mechanism is not established
here; its steady-state demote is equal. Whether that prices a lease handoff to route the arena under the
door is the lead's call; the door's own measured cost with write-combined destinations (130 ms per entry
at demote, `WC-DESTINATIONS.md`) is superseded on this card class by lane A's cached default and is owed
again below.

## CPU bank on the target rig's host (`pro-single-day17/cpubank/`, pass/fail, not timed)

Through the collector for the lock proof and the record (`--rig pro-single --external-lock`, validate
rc=0): `cpu expert prefetch accounting tests: ALL GREEN` (barrier, failure, parity; source fixture on the
root filesystem, mirror on `/dev/shm`, kernel `6.8.0-139-generic`, tree `0b55f7b39`) and `memra native CPU
quant check: ALL GREEN` (`cpu_native_check` `cd12c468...` against the companion `9221ab32...` built there
with the production flags, `detached prefetch (annex promote + bit-identity): PASS`). No GPU cell exists for
the companion in `tier-battery.py`; this is the CPU bank on the target rig's host, nothing more.

## The WC decision item, closed by pointer (task 3)

Lane A day 14 decided the allocation flag this lane's `WC-DESTINATIONS.md` asked for:
`docs/decisions/PINNED-DESTINATIONS.md` (lead ruling 23), `PinnedKind::for_device` in `tier_transfer.rs`,
`Cached` on the RTX PRO 6000 Blackwell class (day-13 receipt, bind hash 77.6 against 1711 ms at 160 MiB,
D2H not above at every pair), `WriteCombined` on the RTX 5090 class (inconclusive under the rule as
pre-registered) and every unrecognized name; resolved once in `CudaTransfers::new`; no environment
variable; `alloc_host_kind` is the gate's seam. `HOSTPREFIX-DOOR.md` Status and `WC-DESTINATIONS.md` now
point there and mark the day-16 pair as the write-combined cost on that card class, kept as the record.
What the door's decide-by review still owes, named in both: the same pair cell (demote and promote lines
OFF against ON, N=5 per arm per order, both orders, one lock hold) on a binary carrying `for_device`, so
the door's two demote-side hashes and one promote-side hash run over cacheable memory, and the hash-speed
micro-cell that splits the remaining delta between the hashes and the ticket lifecycle. No new engine flag,
no engine change for this item.

## Local battery on the final tree (`day17-local/final-*.log`) and effort

No Rust changed today; `cargo fmt --all -- --check` clean; `tools/check-flags.sh` 865 runtime names, none
uncovered (no new `MEMRA_*` read: the fixture's knobs are existing companion names, set, not read; the
test-only directories are positional arguments, not variables); `python3 tools/check-public-boundary.py
check` 582 grandfathered, 0 new; `tools/docs-registry-census.sh` clean (58 tables, 905 rows); `git diff
--check` clean; the em-dash scan over the day's prose and scripts finds none (the three hits in
`docs/TESTING.md` are pre-existing lines 429, 432 and 637, not today's paragraph); `shellcheck -x` clean on
`tools/test_cpu_expert_prefetch.sh`; `ci.yml` parses with no duplicate mapping keys after the new step.
Collector `--validate` rc=0 on both day-17 cells. Local runs of the fixture and the companion build ran
under `systemd-run --user --scope -p CPUQuota=1200% -p MemoryMax=28G`; no local GPU run today.

Effort: about 1.5 agent-hours for the day against the 4-hour budget (the arena cell, pre-registration to
replay, about 35 minutes of it: the card was free and the model page-cached).
