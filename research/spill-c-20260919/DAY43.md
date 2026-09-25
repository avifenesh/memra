# WP-C day 43 (2026-09-24): the MoE slot cache door, improvement I6: the host tier at the machine's scale

`OWED.md` C1 step (b), first improvement in the order day 40's rule set (`DAY40.md` section 5: `demand` is the top
stage, 16.36 of 20.47 ms per token on the RTX 5090's window, and at the 16-record tier every GPU miss is a host miss
and a physical read). Tree at start: `9085bdf50`. Every cell is `executed-not-qualified`. Nothing moves a default.

## 1. Pre-registration (committed before any I6 code)

**What is wrong at scale, from source (tree `9085bdf50`).**

1. `host_bank_slots` (`banked_residency.rs`) refuses above 256 MiB ("qualification ceiling") and clamps to 16
   records; the installer builds one SLRU class of `max_bytes` (860,160 B) slots, so a 450,560 B record takes an
   860,160 B slot; the governor's pageable capacity is a fixed 512 MiB.
2. `SlruPolicy` (`crates/memra-tier/src/bank/slru.rs`) is "a readable CPU oracle, not a tuned O(1) replacement":
   `hit` and `remove` scan `VecDeque`s (`position`, `retain`), and its maps are `BTreeMap<BankId, _>`.
3. `BankService::collect_evicted` (`residency.rs`), called by every `finish`, compares every owned lease with every
   cached one (`owned.values().filter(|l| !cache.values().any(..))`): quadratic in the tier's size. `cache_bytes()`
   sums the cache on every call.

At 16 records none of this shows. At the budget day 40's profile asks for (a window LRU hit fraction of 0.439 from
8 GiB up, 0.000 at 2 GiB or less) it does.

**The design.**

- (a) **The plan.** `--expert-bank-host-bytes=N` is planned into one SLRU class per exact record size in the catalog
  (three here: 450,560, 557,056 and 860,160 B), each class's slot count proportional to its record count under N
  (the Hamilton remainder pass of `moe_cache.rs::size_class_plan`, each class capped at its record count), so the
  tier holds records, not max-size slots. Refused (the typed `ExpertBankRefusal`, exit 2) when N cannot hold one slot
  of the largest record size, and when N exceeds three quarters of the host's `MemAvailable` read at install (a
  machine fact, `/proc/meminfo`; no environment variable). The 16-record clamp and the 256 MiB ceiling are removed;
  the default stays 256 MiB. The governor's pageable capacity is the planned bytes plus the metadata and staging it
  already charges. The installer prints `[experts-via-tier] host_bank_plan requested=N planned=P classes=[(bytes,
  slots), ...] records_held=R ceiling=C`.
- (b) **O(1) host SLRU.** `SlruPolicy` keeps its interface and its decisions; its segments become intrusive
  doubly-linked lists over slot indices (the `moe_cache.rs` `SlruList` shape) and its maps `HashMap`s. The current
  `VecDeque` implementation moves to the tier's tests as the oracle; every existing SLRU test runs on the new policy,
  and a randomized trace (at least 200,000 operations over three classes with `keep` sets, reserve, publish, hit,
  remove and abort) must produce the oracle's decision (`slot`, `evicted`) at every step and its `orders()` at every
  1,000th.
- (c) **Bookkeeping in O(evicted).** Leases the SLRU evicts from `cache` go on an explicit list; `collect_evicted`
  tries to release only those (a `Busy` lease stays listed); `cache_bytes` is maintained on insert and remove. A CPU
  test drives a bank of 4,096 records through 20,000 demands with evictions and asserts, after every `finish`, that
  every owned lease is either cached or held by an open ticket.
- (d) Unchanged: the per-record verify, the lease identity, the trace line's form (its `slot=` field now ranges over
  the planned slots), `physical_reads`, one lease at a time, the drains. I6 changes where bytes can stay, not how a
  miss is served.

**The cell `resid` (RTX 5090 first; the target card in the ladder sitting).** Day 40's shape unchanged
(`MEMRA_MOE_RESIDENT=0 MEMRA_NGEN=32 MEMRA_MOE_SLOTS=9986`, prompt `55 88 13`, the approved artifact). Four arms,
every door arm with `--expert-bank-stages` (its cost was within bound on day 40): OFF (no door); BASE (the day-40
binary `b73bb4d3...`, default budget, today's program); I6D (the I6 binary, default 256 MiB); I6G (the I6 binary,
`--expert-bank-host-bytes=8589934592`). Order 1 (OFF, BASE, I6D, I6G) x 5, order 2 (I6G, I6D, BASE, OFF) x 5; N=5 per
arm per order, 40 runs in one collector hold, `/tmp/memra-5090.lock`, 250 ms telemetry, the runner waiting for no
compute app and at least 46 GiB `MemAvailable` (the 8 GiB tier beside the loader's 15 GB of pinned slabs).

**Integrity (a failure voids the reading, the clause named).** All 40 runs exit 0 with `MATCH`; one `tokens:` tape
and one STEADY-STATE line across the 40; `slots=9986` everywhere; door arms print `installed`, `physical_reads=` and
the four stage lines; I6 arms print the plan line and BASE does not; in every door run `physical_reads` equals the
count of `hit=false` trace lines and the trace line count equals the host demands.

**Readings and the decision rule.** Per arm: the window's door cost per token (median over 10 of the arm's window
minus OFF's, over 32), per-order medians, and day 40's R2 stage lines (host hits per token, `verify`, `step`,
`stage`, `demand`). Let `noise` be the larger IQR of the two arms compared (window seconds, N=10 each).

- **I6 at the default budget does not regress:** `median(I6D window) - median(BASE window) <= noise` pooled and in
  both orders. If this fails, I6 is fixed or reverted before anything builds on it.
- **The budget's effect (a reading with a direction registered):** host hits per window token on I6G above 0 and the
  window door cost on I6G below BASE's by more than `noise` in both orders, `resid_budget_helps`; otherwise
  `resid_budget_flat` or `resid_budget_hurts`, stated as it reads. Day 40's LRU profile predicts roughly 40 host hits
  per token (0.44 of 92.3) and a `demand` drop of the same fraction; SLRU is not LRU, so the number is a direction,
  not a bound.
- I6 stays in the tree if the no-regression clause holds: it is the tier the fill (the next improvement) fills.

**What each card can decide.** The RTX 5090 decides I6's regression clause on this card and reads the budget's
effect here. The target card reads both in the ladder sitting; its budget arm is registered there with that host's
`MemAvailable`.

## 1a. Addendum before the cell: two costs that grow with the tier, found by a dry check

A dry check of the I6 binary at 8 GiB (`day43-cpu/dry-check-i6g.log`, one run under the lock outside the collector,
not a receipt) printed the plan (`classes=[(450560, 11271), (557056, 5635), (860160, 433)] records_held=17339`), the
tape unchanged and 40.5 host hits per window token (day 40's LRU profile predicted about 40), with `verify` 9.38 to
5.29 and `pread` 3.46 to 1.92 ms per token, but the window slower (1.744 s against day 40's 0.915): `finish` 23.48 ms
per token (`retire` 20.97 of it) and the read step 8.96 (173 us per read against day 40's 55).

- (e) **The governor's release is O(outstanding charges).** `Governor::release` calls `prune_fairness`, which
  rebuilds a set of every charged tenant from `charged_tenants`, one entry per outstanding charge: 17,339 cached
  leases make each release O(17k), and a miss releases several charges. Part of I6 (the tier at scale): a per-tenant
  charge count kept on reserve and release, so the active set is O(queued requests + distinct tenants); decisions
  unchanged, checked by an in-module test that recomputes the old prune's active set from `charged_tenants` after
  every operation of a randomized trace and asserts it equal to the new one, and that `last_served` and
  `evicted_floor` evolve identically. Registered here before its code and before the cell.
- **Not I6's: the read step's allocations.** Each host miss allocates and zero-fills two fresh `Vec`s; with 17k live
  records in the heap the allocator hands out untouched pages, so every read pays first-touch faults. That is I2's
  (a preallocated, pre-faulted slot pool, the read straight into the slot), registered already. The cell runs as
  registered; this cost is expected in the I6G arm and is a reading, not a reason to move a clause.

## 3. Results, cell `resid` (RTX 5090 Laptop GPU, `rtx5090-day43/resid/`)

One collector hold, 21:32:40Z to 22:11:10Z, 40 runs, tree `f9f5f3953`, binaries `run-gen` `b73bb4d3...` (BASE) and
`run-gen-i6` `771ba66b...` (I6, with I6 (e)), the approved artifact. Three lock-busy attempts before it
(`lock-retries.log`: another lane held the card); no holder touched. Regime over the hold (`regime.log`, the
collector's 250 ms CSV, N=9179): SM 172 to 2790 MHz, power 9.2 to 154.8 W, 53 to 73 C. Collector `--validate` rc=0.

Verbatim (`resid/reading.log`):

`DAY43 RESID CHECKS rig=rtx5090 runs=40 integrity=ok`

`DAY43 ARM base window_door_ms_per_token pooled=19.27 o1=19.03 o2=19.37 gen_door_ms_per_token=41.27 window_s median=0.948 iqr=0.013 | per token: demand=16.733 verify=9.369 step=5.470 pread=3.878 stage=1.294 alloc=0.768 drain=2.072 enqueue=1.024 validate=0.650 finish=0.261 collect=0.135 publish=0.233 miss_total=20.713 gpu_misses=92.3 host_hits=0.0 host_misses=92.3 reads=92.3`

`DAY43 ARM i6d window_door_ms_per_token pooled=21.59 o1=21.69 o2=21.28 gen_door_ms_per_token=46.33 window_s median=1.022 iqr=0.019 | per token: demand=18.951 verify=9.355 step=5.927 pread=3.893 stage=2.676 alloc=2.089 drain=2.069 enqueue=1.038 validate=0.685 finish=0.339 collect=0.220 publish=0.348 miss_total=22.936 gpu_misses=92.3 host_hits=0.0 host_misses=92.3 reads=92.3`

`DAY43 ARM i6g window_door_ms_per_token pooled=24.17 o1=24.62 o2=22.47 gen_door_ms_per_token=60.27 window_s median=1.105 iqr=0.154 | per token: demand=18.953 verify=5.289 step=10.985 pread=2.267 stage=2.072 alloc=1.418 drain=2.070 enqueue=2.504 validate=0.691 finish=0.132 collect=0.025 publish=0.223 miss_total=25.333 gpu_misses=92.3 host_hits=40.5 host_misses=51.8 reads=51.8`

`DAY43 CLAUSE no_regression i6d_minus_base pooled=+0.075 o1=+0.085 o2=+0.061 noise=0.019 rule <=noise pooled and both orders -> FAIL`

`DAY43 READING budget i6g_minus_base pooled=+0.157 o1=+0.179 o2=+0.099 noise=0.154 host_hits_per_token=40.5 -> resid_budget_flat`

`DAY43 RESID rig=rtx5090 integrity=ok no_regression=FAIL budget=resid_budget_flat`

**Read, not tuned.** I6's no-regression clause FAILS at the default budget: +2.3 ms per window token over BASE. The
plan line says what changed (`host_bank_plan requested=268435456 planned=268271616 classes=[(450560, 353), (557056,
176), (860160, 13)] records_held=542`, against BASE's 16 slots of the largest record): 542 records held, still zero
host hits in the window, and the per-token lines put the difference in exactly two stages, `alloc` 0.768 to 2.089
and `stage` 1.294 to 2.676 ms (per miss about 8 to 23 us and 14 to 29 us), while `verify`, `pread` and `drain` are
unchanged. That is the mechanism section 1a registered for the budget arm before the cell ("each host miss allocates
and zero-fills two fresh `Vec`s ... the allocator hands out untouched pages"; "That is I2's"), here at the default
budget too: a tier that holds 34 times more records than BASE makes every read land in cold, freshly faulted memory.
I6G at 8 GiB gets its 40.5 hits per token (the day-40 profile's prediction) and `verify` 9.37 to 5.29, but pays the
same allocation cost and twice the `step`, so the window is flat. Section 1's rule: "If this fails, I6 is fixed or
reverted before anything builds on it." The fix is I2 (day 47), already in the tree: one pre-faulted pinned pool, the
read straight into its buffer, no per-read `Vec` and no assembly copy. Section 4 registers the check that the fixed
tree passes I6's clause, before the deciding cell.

## 4. The fix check, registered before it runs (cell `residfix`)

**Question.** On the tree after every rung (I10's binary, which carries I2's pool), does I6's clause hold at the
default budget?

**Cell `residfix`** (`day43-fix-cell.sh`, reader `day43-fix.py`, RTX 5090 first; the target card reads it in the
DAY52 sitting's DAY51 phase). Day 43's shape (`MEMRA_MOE_RESIDENT=0 MEMRA_NGEN=32 MEMRA_MOE_SLOTS=9986`, prompt
`55 88 13`, the approved artifact); arms OFF (`run-gen-i10`, no door), BASE (`run-gen`, the day-40 binary, default
budget), I10D (`run-gen-i10`, default budget); both door arms with `--expert-bank-stages`; order 1 (OFF, BASE, I10D) x
5, order 2 reversed x 5, one collector hold.

**Integrity** as day 43's (plan line on I10D, not on BASE), plus the I10 fill-complete line with `fill_refused=0` on
every I10D run.

**Clause** (I6's, unchanged): `median(I10D window) - median(BASE window) <= noise` pooled and in both orders, noise
the larger IQR. PASS: I6's regression is fixed in the tree the deciding cell runs. FAIL: the default budget still
regresses on the final tree; the cause is found from the per-token stage lines and fixed, with its own cell, before
`DAY51.md` section 2 names the final tree. Readings beside it: the per-token stage lines of both arms.

## 5. Results, the fix check `residfix` (RTX 5090 Laptop GPU, `rtx5090-day43/residfix/`)

One collector hold, 00:28:29Z to 00:44:06Z, 30 runs, tree `2f86e3078`, binaries `run-gen` `b73bb4d3...` (BASE) and
`run-gen-i10` `90c0496d...` (the tree after every rung through I10), the approved artifact, the runner under the
1200% cap. Regime (`regime-residfix.log`, 250 ms, N=3727): SM 172 to 2775 MHz, power 9.3 to 148.9 W, 54 to 73 C.
Collector `--validate` rc=0.

Verbatim (`residfix/reading.log`):

`DAY43 RESIDFIX CHECKS rig=rtx5090 runs=30 integrity=ok`

`DAY43 FIX ARM base window_door_ms_per_token pooled=19.64 o1=19.97 o2=19.41 gen_door_ms_per_token=42.03 window_s median=0.960 iqr=0.027 | per token: demand=16.956 verify=9.376 step=5.637 pread=4.043 stage=1.291 alloc=0.756 drain=2.073 enqueue=1.139 validate=0.652 finish=0.266 collect=0.135 publish=0.235 miss_total=21.055 gpu_misses=92.3 host_hits=0.0 host_misses=92.3 reads=92.3`

`DAY43 FIX ARM i10d window_door_ms_per_token pooled=15.50 o1=15.94 o2=15.16 gen_door_ms_per_token=33.47 window_s median=0.827 iqr=0.029 | per token: demand=16.207 verify=9.356 step=5.698 pread=5.686 stage=0.539 alloc=0.008 drain=0.000 enqueue=0.191 validate=0.079 finish=0.003 collect=0.159 publish=0.273 miss_total=16.704 gpu_misses=92.3 host_hits=0.0 host_misses=92.3 reads=92.3`

`DAY43 CLAUSE no_regression i10d_minus_base pooled=-0.133 o1=-0.129 o2=-0.136 noise=0.029 rule <=noise pooled and both orders -> PASS`

`DAY43 RESIDFIX rig=rtx5090 integrity=ok no_regression=PASS`

I6's clause holds on the tree the deciding cell runs: at the default budget the tuned door is 4.1 ms per window token
faster than BASE, the allocation (0.756 to 0.008) and the stage (1.29 to 0.54) gone with I2's pool, the drain gone
with I1. At 256 MiB the window still gets no host hits (every miss a read and a verify, 16.2 ms of the 16.7): the
default budget holds 542 records against the window's reach of thousands, the day-40 profile's reading; the door's
16 GiB arms are where the host tier serves.

## 6. The fix check on the target card (BOX8, DAY52 section 9; receipts `pro-single-day52/residfix/`)

One collector hold, 01:41:15Z to 01:56:22Z, 30 runs, the box's `run-gen` `550baa4b...` (day 40's tree) and
`run-gen-i10` `4870d3ef...` (`70d6633f5`), the runner pinned to 12 cores. Regime (`residfix/regime.log`, 250 ms,
N=3607): SM 172 to 2872 MHz, power 15.7 to 203.8 W, 34 to 45 C. Collector `--validate` rc=0.

Verbatim (`residfix/reading.log`):

`DAY43 RESIDFIX CHECKS rig=pro-single runs=30 integrity=ok`

`DAY43 CLAUSE no_regression i10d_minus_base pooled=-0.128 o1=-0.126 o2=-0.128 noise=0.004 rule <=noise pooled and both orders -> PASS`

`DAY43 RESIDFIX rig=pro-single integrity=ok no_regression=PASS`

At the default budget the tuned door is 4.0 ms per window token faster than BASE on this card too (17.44 to 13.45),
`alloc` 0.712 to 0.008 and `stage` 1.27 to 0.56.
