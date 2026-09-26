# WP-C day 63 (2026-09-25): the owner demand on the prefetch path, split and tuned (lead owed item 2, next step), registered before code

Lead, after the days 59 to 61 sitting: "Continue with DAY63 (the owner demand on the prefetch path: the governor's
per-ticket reservation, the lease clones, publish plus finish), same rules as before: pre-register before code, CPU
ladder then card cells, no bound moved after a result." Tree at start: `e6217375e` (being integrated as integ61).

## 0. Where the demand stands (receipts)

- Target card, BOX12 (`DAY60.md` section 2): the door's prefetch path costs 0.607 ms per window token more than REF's,
  the owner demand 0.464 of it (`pf_demand_ns`, 5.2 us per issued prefetch) on the `c60` door; I11 then took the door
  from 0.240 to 0.234 s per 32-token window and from 0.276 to 0.267 s gen-only, and it still `loses` to REF by 0.25 ms
  per window token and 0.34 per generated token (`DAY61.md` section 3).
- Local CPU, the day-61 profile at I11's last change (`cpu-day61/ladder-5.log`): 3620 ns per host-hit prefetch cycle;
  the bank's `stage` 1164.5, `publish` 296.0, the retire side 334.9 (`finish_host_use` + `retire` + `acknowledge`),
  `collect` 13.0; the governor's reservation and release 317 ns alone; the rest of `SlruExpertDispatch::demand`, the
  proxy and `TracedDispatch` about 1.2 us between them.
- From source, per host-hit ticket: `stage` reserves the ticket's queue charge from the governor (the request's
  validation, the fit test, two `TierBudget` temporaries, the issuer's `Arc<Mutex>` record and its map entries, the
  tenant maps) and `acknowledge` releases it; a `BankLease` clone copies its `BankId` and its whole `RecordLayout`
  (the segment and requirement vectors and the tensor names), and a host-hit cycle clones a lease at least three
  times (the ticket's record map, `publish`'s output vector, the proxy's `finish` rebuilding the `ExpertDemand`);
  `finish` looks the ticket up in the pending map four times (`finish_host_use`, `retire`, twice in `acknowledge`) and
  once more to remove it.

## 1. Pre-registration: the split (an instrument; decides nothing)

**The instrument.** The bank's stage clock (`BankStageTimes`, installed only by `with_stage_clock`, the
`--expert-bank-stages` diagnostic) gains fields appended to its line, every existing field unchanged in meaning and
order: inside `stage`, `stage_lookup_ns` (the per-id catalog loop), `stage_cache_ns` (the unique set and the host cache
pass), `stage_charge_ns` (the governor reservations); inside `publish`, `publish_output_ns` (the output leases) and
`publish_policy_ns` (the hotness and the SLRU); on the retire side, `host_use_ns`, `retire_only_ns` and `ack_ns`, one per
call, their sum what `retire_ns` has always counted, and `ack_release_ns` (the governor release inside
`acknowledge`). A census pins that the clock's new writes are counter additions only. Without the stage clock nothing
runs differently.

**The profile additions** (the day-61 `#[ignore]` profile, same harness, same catalog and routed cycles): P3 prints the
new fields per cycle; P7 times, over the same routed ids, one `BankLease` clone and its drop, taken from a cached lease
(`BankedResidency::resident`).

**Reading.** Per cycle ns of every new field, and each named part's share of P3's `demand` plus `finish`. The parts
decide what section 2 registers, before any change's code. The same contracts bind as `DAY61.md` section 2: a lease
pins its host bytes until `finish`; a record is released only after every ticket holding it retired; the governor
charges and releases exactly what it did (the same dimensions, the same refusals, in the same order); the SLRU and
the hotness see the same demands in the same order; the same bytes reach the same slot. These are the local CPU's
numbers, a profile for the design, not a card result.

## 1a. The split's result (`cpu-day63/split.log`, `split-run2.log`; commit `ad73d242c`)

Two runs, one thread under the cap, the median of 5 repeats of 200000 host-hit cycles. Another tenant's CPU work
shared the rig in both (load average 3.8 at the second run's start), so P2's absolute cycle reads 3949.8 and 3800.5 ns
against ladder-5's 3619.9 on the same program; the split's shares are the reading, not the absolute numbers.

| part (ns per cycle, P3, the bank alone with its clock) | run 1 | run 2 |
|---|---:|---:|
| `stage` whole | 1288.8 | 1263.1 |
| `stage_lookup` (the per-id catalog loop) | 158.3 | 140.4 |
| `stage_cache` (the unique set and the host cache pass) | 703.9 | 696.2 |
| `stage_charge` (the governor's queue reservation) | 271.0 | 274.6 |
| `publish` whole / `publish_output` / `publish_policy` | 372.2 / 68.8 / 241.7 | 370.1 / 68.3 / 241.1 |
| retire side whole / `host_use` / `retire_only` / `ack` (`ack_release` inside it) | 352.0 / 32.3 / 31.9 / 287.8 (152.5) | 349.9 / 31.7 / 31.5 / 286.7 (152.3) |
| P7, one lease clone and its drop | 68.7 | 74.5 |

Read against the lead's list: the governor's per-ticket reservation and release is 425 ns a cycle (271 + 152); the
retire side's pending-map work beside that release is about 135 ns; `publish` is mostly the SLRU (the demand ticket's
`hit`, 241 ns), its output lease 68; a lease clone is about 70 ns, three a cycle. The largest single part is not on
the list: `stage_cache`, about 700 ns, a one-record set and one lookup in the host cache's `BTreeMap<BankId, _>` of
30720 records (each comparison walks two 32-byte digests and a tensor name). The per-id lookups keyed by the full
`BankId` recur through the cycle: the catalog (twice), the host cache, the SLRU (three times).

## 2. Pre-registration: I13, the per-ticket and per-lease work tuned (before any of its code)

Every change keeps the contracts of section 1 exactly: the same charges in the same dimensions, the same refusals in
the same order, the same state transitions, the same bytes. One commit per change, in this order:

1. **The governor without temporaries.** `reserve` tests the fit without collecting the three value vectors,
   issues the lease, then adds the request into `used` in place (checked before any component changes); `release`
   subtracts in place the same way; `prune_fairness` returns before building its sets when `last_served` holds no
   more tenants than `queue_limit` (then no idle tenant can exceed the limit, so it would remove nothing).
2. **One body per lease.** `BankLease`'s fields (id, layout, class, charge, backing) move into one `Rc`; a clone is a
   count increment, every accessor returns what it returned.
3. **The retire side in one lookup.** `BankService::finish_ticket` runs `finish_host_use`, `retire` and `acknowledge`
   on one pending entry with the same checks, flags, charge releases and removals, and the same errors at the same
   points (a ticket not yet retired refuses `NotReady` after marking host use done, as the three calls did);
   `SlruExpertDispatch::finish` calls it. The three public calls stay for every other caller. The stage clock keeps
   `host_use_ns`, `retire_only_ns`, `ack_ns` and `ack_release_ns` as brackets inside it and `retire_ns` as their sum.
4. **The SLRU's maps hashed deterministically and cheaply.** `SlruPolicy`'s `table` and `reserved` take an in-crate
   Fx-style hasher (the rotate, xor, multiply step rustc uses) instead of the randomly keyed SipHash; neither map is
   iterated anywhere, so no order is observable, and the keys are catalog records, not untrusted input.

**CPU gates, per change:** the memra-tier suite and the engine lib suite green, clippy `-D warnings`, fmt; and one pin
per change: (1) a differential test of the in-place add and subtract against `TierBudget::checked_add` and
`checked_sub` over randomized budgets (the same values, the same errors, nothing changed on an error), and a governor
test that pruning still evicts idle tenants past the limit; (2) a clone shares its body (`Rc::ptr_eq`) and every
accessor reads the same; (3) `finish_ticket` against the three calls on twin banks over the refusal cases (unknown
ticket, not published, not retired) and the success path: the same results and the same governor totals; (4) the SLRU
oracle tests (`slru_oracle`, day 43's randomized oracle) unchanged and green.

**The CPU ladder:** the day-61 profile at each commit (`cpu-day63/ladder-<n>.log`), P2 and P3's split per row, read
beside `split.log`. The rig's other load is recorded with each row (`uptime`); a row read under a different load is
marked, not compared as if equal.

**The card cell** is registered in section 4 before its script is written, with I13 against I12 and against REF, and
any further improvement that section 3 registers from this ladder.

## 2a. I13 on the CPU: the ladder

Every change's CPU gates passed at its commit (the memra-tier suite 299, 300, 301, 301; engine lib 567 each time;
clippy `-D warnings`; fmt): the differential budget test (20000 randomized budgets), the governor's fairness test
past the limit (unchanged), `a_lease_clone_shares_its_body`, `finish_ticket_is_the_three_calls` on twin banks, the SLRU
oracle tests (`day4::slru_matches_recorded_native_semantics_synthetic_trace`,
`day43::o1_host_slru_matches_the_vecdeque_oracle_on_a_randomized_trace` and the rest) unchanged and green. The rig's
load moved between rows (`uptime` in each `ladder-<n>.log`, load average 1.85 to 3.53), so each row as run is
recorded and, beside it, all five binaries were re-read in one window in both orders (`window.log`, row 0 to 4 then
4 to 0, one profile run each; the P2 figure is the mean of the two):

| row | commit | change | P2 as run (ns per cycle) | P2 same window | P3 demand / finish same window |
|---|---|---|---:|---:|---|
| 0 | `ad73d242c` | none (I12 plus the split) | 3949.8, 3800.5 (`split*.log`) | 3569.2 | 2300.1 / 448.8 |
| 1 | `9bbab60ed` | 1, the governor without temporaries | 3434.5 | 3432.4 | 2261.6 / 386.1 |
| 2 | `e8195cbf3` | 2, one body per lease | 3077.7 | 3080.8 | 2002.2 / 364.6 |
| 3 | `bc0e31d2e` | 3, the retire side in one lookup | 3032.1 | 3033.1 | 2020.8 / 306.1 |
| 4 | `c9379c051` | 4, the SLRU on the Fx hasher | 2975.1 | 2939.9 | 1965.0 / 301.6 |

I13 takes the host-hit prefetch cycle from 3569 to 2940 ns in one window (17.6% less CPU per block; day 61 began at
5370). By part at row 4 (`ladder-4.log`): the governor's charge and release 240 plus 93 ns (from 271 plus 152), a
lease clone 11 ns (from 70), `publish_output` 21 (from 68), the retire side 220 (from 350), an SLRU lookup 71 ns (from
101). What stays largest is per record and keyed by the full `BankId`: `stage_cache` 478 ns (the host cache's
`BTreeMap` walk), the residency check's 510 ns, the SLRU's `hit` inside `publish_policy` 207, the two catalog
lookups (the validation's and `stage_lookup`'s 123).

## 3. Pre-registration: the card cell `i13` (both cards; before its script)

**Binaries.** Labels `c60=da649107c` (`run-gen-c60`, REF), `i12=117302725` (`run-gen-i12`, the door before I13) and
`i13=c9379c051` (`run-gen-i13`, I13's last change), built on each card's host as day 61's were.

**The cell** (`day63-cell.sh`, reader `day63-read.py`): day 18's pressure shape (`MEMRA_MOE_RESIDENT=0 MEMRA_NGEN=32
MEMRA_MOE_SLOTS=9986`, prompt `55 88 13`), one collector hold, 250 ms telemetry, the 1200% cap (12 pinned cores on a
box without systemd). Arms: REF (`MEMRA_MOE_PREFETCH=1`, `run-gen-c60`); I12 (the door at the full-bank host budget,
`run-gen-i12`); I13 (the same door, `run-gen-i13`); I13S (I13 with `--expert-bank-stages`: read for its split only,
never for a timing verdict). Order 1 (REF, I12, I13, I13S) x 5, order 2 reversed x 5, 40 runs.

- **Integrity** (any failure voids the card's reading): every run exit 0 and `MATCH`; one `tokens:` tape across all
  four arms; every door run's fill complete before decode and `physical_reads=0`; one host demand sequence (the
  `[expert-host-slru]` lines without their slot numbers) across every I12, I13 and I13S run; I13S prints the stage
  lines with the day-63 fields.
- **Readings** (`noise` the larger IQR of the two arms compared; gen-only decode the primary reading, the steady window
  beside it): I13 against I12, `improves`, `regresses` or `flat` exactly as `DAY61.md` section 2 defines them.
- **The door against REF:** the door arm is I13, or I12 if I13 `regresses`: `beats`, `matches` or `loses` as `DAY61.md`
  section 2 defines them, recorded plainly whatever it reads.
- **The split on the card** (I13S, the median over its 10 runs of each field's window-minus-warm per window token):
  printed beside the verdict for the next improvement's design; it decides nothing.
- **What follows.** I13 `regresses` on either card: reverted by this lane with its receipt. `flat` or `improves`: kept.
  If the door still `loses` on the target card, the next improvement (the per-record `BankId` lookups, or the three
  blocks of one expert under one ticket) is registered from this cell's split, before its code.

## 3a. The sitting, prepared before any cell

`day63-cell.sh` and `day63-read.py` written after section 3; the reader dry-checked for mechanics only on relabelled
target logs (the day-61 `i11` cell and the day-58 stage-clock tip runs): one host demand sequence (`4bdc2610...`)
across the door arms, and its integrity check refuses stage lines without the day-63 fields, as the old logs lack them
(its verdict there is `void`, as it must be). The driver `day63-box.sh` (builds by `day63-box-build.sh`, `run-gen`
only), dry-checked for control flow (`day63-cpu/dry-check-driver.log`). Run as
`D63_BUILDS="c60=da649107c i12=117302725 i13=c9379c051" bash /root/wt-c/research/spill-c-20260919/day63-box.sh`.
Expected: three builds about 15 minutes, the cell about 8 (40 runs at about 11 s). Box needs as `DAY61.md` section
2b: one RTX PRO 6000 Blackwell Workstation Edition, the approved 35B artifact at `/root/artifacts/`, the lane tip in
`/root/wt-c` and a detached build worktree at `/root/wt-c-build`, CUDA 13 and Rust, at least 48 GB host
`MemAvailable`, about 20 GB free under `/root`. The RTX 5090's cell is queued (queue v7, behind v6) with
`run-gen-i13` built locally (`/tmp/c61-build/build-i13.log`, sha256 `596b620e...`).

## 4. The target card (BOX13, run by the lead as registered; `pro-single-day63/`)

The lead ran `day63-box.sh` exactly as section 3a names it on the tree `cfbda42f8` (BOX13, one RTX PRO 6000 Blackwell
Workstation Edition at 600 W, the same host class as BOX12), 05:54Z to `box done 2026-09-25T06:07:45Z`; the binaries
built on the box (`run-gen-c60` `13a5cf85...`, `run-gen-i12` `d7e07114...`, `run-gen-i13` `8e29bcf0...`, by hash in
`box-binaries.sha256`); the runner pinned to 12 cores with `taskset`. The receipts were mirrored by the lead into this
worktree and read here: 187 of 187 files `OK` against `box-mirror-manifest.sha256` (`MIRROR-CHECK.txt`). Regime
(`regime.txt`): 30 to 45 C, SM median 2610 MHz, N=1773 samples at 250 ms. Verbatim (`i13/reading.log`):

- `DAY63 host demand sequence sha256 4bdc2610c3534e42 lines=[22077]`
- `DAY63 I13 CHECKS rig=pro-single runs=40 integrity=ok`
- `DAY63 gen-only decode medians (N=10 each): ref=0.255 i12=0.267 i13=0.264 i13s=0.266`
- `DAY63 STEP i13_vs_i12 gen-only decode: pooled=-0.0030 o1=-0.0030 o2=-0.0030 noise=0.0010 -> improves`
- `DAY63 steady window medians (N=10 each): ref=0.226 i12=0.234 i13=0.232 i13s=0.234`
- `DAY63 STEP i13_vs_i12 steady window: pooled=-0.0020 o1=-0.0020 o2=-0.0020 noise=0.0010 -> improves`
- `DAY63 DOOR i13_vs_ref gen-only decode: pooled=+0.0090 o1=+0.0090 o2=+0.0090 noise=0.0010 -> loses`
- `DAY63 DOOR i13_vs_ref steady window: pooled=+0.0060 o1=+0.0060 o2=+0.0060 noise=0.0010 -> loses`
- `DAY63 SPLIT owner per window token (N=10 runs): host_hits=92.3 host_misses=0.0 inner_demand_ns=0.2160ms pread_ns=0.0000ms reads=0.0 trace_ns=0.0060ms`
- `DAY63 SPLIT bank per window token (N=10 runs): ack_ns=0.0312ms ack_release_ns=0.0120ms alloc_ns=0.0000ms collect_ns=0.0016ms host_use_ns=0.0037ms publish_ns=0.0215ms publish_output_ns=0.0027ms publish_policy_ns=0.0131ms retire_ns=0.0359ms retire_only_ns=0.0010ms stage_cache_ns=0.0807ms stage_charge_ns=0.0307ms stage_lookup_ns=0.0154ms stage_ns=0.1489ms stages=92.3 step_ns=0.0000ms steps=0.0 verified=0.0 verify_ns=0.0000ms`
- (the cache side of the same runs: `retire_ns=0.0751ms`, `validate_ns=0.0144ms`, `prefetches=88.6`; the full line in the reading)
- `DAY63 VERDICT rig=pro-single integrity=ok i13=improves door=i13 vs_ref=loses (window: i13=improves vs_ref=loses)`

**Read as registered.** I13 improves the door on the target card (3 ms gen-only over 32 tokens, 2 ms of the window) and
stays. The door still loses to REF: 0.264 s against 0.255 gen-only (+0.28 ms per token) and 0.232 against 0.226 in
the window (+0.19 ms per token), from +0.69 and +0.44 before day 61. Recorded plainly for the owner's 2026-10-04
reading. The owner side of the door now spends 0.216 ms per window token on 92.3 host-hit demands (2.3 us each, from
0.423 ms at the day-58 tip); inside the bank, `stage_cache` is the largest part (0.081 ms, 0.87 us per demand), then
the retire side (0.036), the governor's charge (0.031), `publish` (0.022) and the catalog loop (0.015); on the cache
side, retiring leases costs 0.075 ms per window token (0.8 us per lease, each a proxy `finish`). The RTX 5090's cell
waits on its reset (queue v7). The next improvement is registered from this split in `DAY64.md`, before its code.

## 5. The RTX 5090 (queue v9, 2026-09-26 01:24Z to 01:32Z; `rtx5090-day63/i13/`)

Queue v9 ran it after the rig's reboot, with binaries rebuilt from their named commits (`c-local-build.sh`, CUDA 13.1; hashes differ from the pre-reboot builds, trees the same), behind `/tmp/memra-5090.lock` with the card idle before the hold. `run-gen-c60` `809132ce...`, `run-gen-i12` `285adc69...`, `run-gen-i13` `c719ebf2...` (tree `c9379c051`).
Regime: 56 to 63 C, SM median 1590 MHz, N=1942. Verbatim:

- `DAY63 I13 CHECKS rig=rtx5090 runs=40 integrity=ok`
- `DAY63 gen-only decode medians (N=10 each): ref=0.363 i12=0.369 i13=0.367 i13s=0.370`
- `DAY63 STEP i13_vs_i12 gen-only decode: pooled=-0.0025 o1=-0.0010 o2=-0.0030 noise=0.0035 -> flat`
- `DAY63 STEP i13_vs_i12 steady window: pooled=+0.0000 o1=+0.0000 o2=+0.0010 noise=0.0053 -> flat`
- `DAY63 DOOR i13_vs_ref gen-only decode: pooled=+0.0040 o1=+0.0050 o2=+0.0040 noise=0.0035 -> loses`
- `DAY63 VERDICT rig=rtx5090 integrity=ok i13=flat door=i13 vs_ref=loses (window: i13=flat vs_ref=matches)`

**Read as registered: I13 `flat` on the RTX 5090; it stays.** The door (I13) loses to REF gen-only by 4 ms over 32
tokens and matches it on the window. The split (I13S) reads the owner's demand at 0.260 ms per window token and the
bank's `stage` at 0.175 on 92.3 stages.
