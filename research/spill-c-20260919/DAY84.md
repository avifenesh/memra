# WP-C day 84 (2026-09-26): OWED C11, I21, the lease path's residency read by record position, before any code

`DAY83.md` section 2 named `outer` (171.0 us per generated token, 26 percent of the door-only leaves in situ at I20)
as I21's target, and placed about 70 percent of it on one read that `pf_resident` (the second leaf, 125.0) makes too:
per record, the adapter's `ids` map and the SLRU policy's `table`. Tree at start: the DAY83 reading's commit.

## 0. The read, from the source

Both lookups run over every record the door holds (30720 here) with wide keys:
- `ids`, in the traced adapter (`TracedDispatch`, `banked_residency/native.rs`) and in the dispatch adapter
  (`SlruExpertDispatch`, `memra-tier` `bank/expert_dispatch.rs`), is a `BTreeMap` from the record's local id
  `(layer, projection, expert)` to its `BankId`; a lookup walks four or five tree levels and returns a `BankId`
  (about 130 bytes: two 32-byte digests and a heap string) stored inline in the leaf;
- the SLRU policy's `table` (`bank/slru.rs`) is an Fx-hashed map from that `BankId` to the host slot: the lookup hashes
  the whole key and compares it against the stored one (its string on the heap).
In a decode, between two leases the forward runs; each lookup touches a dozen or more cache lines, and the day-61
harness puts one residency read at 445 to 455 ns over the same record count. `outer` makes it once per demanded record
(the trace's pre-demand slot), `pf_resident` once per prefetched record (the host residency check), and the trace's
slot map (another `BTreeMap`, slot to local id) takes one insert per record inside `outer`.

## 1. Pre-registration: I21

**The change.** A record's catalog position replaces the hashed id on these reads; the maps stay, authoritative, for
everything else.
- `Catalog::position(&BankId)` (the index it already keeps; a hashed read, used at install and on the SLRU's
  admission path only) and `Catalog::len()`.
- `SlruPolicy` keeps, beside `table`, the host slot of each catalog position and the position of each slot (two
  `u32` vectors, installed by the bank when it takes the policy, empty policy only), updated at every `table` change:
  `publish_at(id, position)` in place of `publish(id)` where the view is installed (a plain `publish` on an indexed
  policy refuses `Unsupported`, so the view cannot go stale), the eviction in `reserve` and `remove` by slot. New read
  `resident_at(position)`: the same answer as `resident(id)` for the record at that position, by construction and by
  test. `BankService` calls `publish_at` with `catalog.position(id)` at its two SLRU publications (the fill admission
  and a missing record's publication).
- The SLRU's metadata charge (`slru_metadata_bytes`) counts the view: 4 bytes per catalog record and 4 per slot, added
  to today's formula, so the installer reserves what the view holds (about 240 KB here, against a 16 GiB host tier).
- `SlruExpertDispatch` resolves each local id to its position once, in `new`, into a dense table indexed by
  `(layer, projection, expert)` arithmetic; `host_resident(local)` reads the position and `resident_at`; a new
  `resident_slot(local)` returns the slot the same way, with the errors the traced adapter's read gave (`NotFound` for
  an unknown local id, then `Incomplete` without a policy).
- `TracedDispatch`'s pre-demand reads (single and grouped demand) and the trace's read for a missing record's
  published slot use `resident_slot`; the trace's slot map is a vector indexed by host slot, growing on demand, with the
  same replace-and-compare that yields each line's `victim`.

Not in I21 (named for later, each its own registration): the bank's own hashed reads on the same path (the catalog
entry in `stage`, the host cache in `stage_cache`, the SLRU `hit` in `publish_policy`, the dispatch adapter's
`validated`), and `outer`'s other parts (the proxy's registry entry, identity checks and pending insert).

**The same program.** No decision reads anything new: every answer is the one the hashed read gave, so the host demand
sequence (the trace, including each `victim`) and the tokens must equal I20's.

**CPU gates before any card** (`day84-cpu/`):
- the SLRU view against `table` on the day-4 synthetic fixture (every row) and the oracle's randomized trace: after
  every operation, `resident_at(position(id)) == resident(id)` for every id; a plain `publish` on an indexed policy
  refuses;
- `Catalog::position`: distinct, in the catalog's order, `NotFound` for an unknown id;
- the dispatch adapter: `host_resident` and `resident_slot` equal the map-based reads for every local id, before and
  after a fill admission and an eviction; an unknown local id `NotFound`;
- the day-61 profile at I20 and I21 in one window (P1 and P9, both orders), and P10's allocation census;
- the tier suites, the engine library, clippy (`-D warnings`, all targets) and fmt;
- a local RTX 5090 check: I20 and I21, both orders, `MATCH`, the same tape and host demand sequence; then the in-situ
  split of I21 beside I20 (DAY83's cell with arms `i20s` and `i21s`, 10 runs each in both orders, pinned to the
  P-cores), read by `day83-read.py`, deciding nothing: whether `outer` and `pf_resident` moved as section 0 says.

## 2. Pre-registration: the card cell `i21`

DAY82's cell with I21 for I20's step: binaries `i15=2243b1fe2`, `i20=8efea3a54` and `i21` (named in section 2a); arms
REF (`run-gen-i20` with `MEMRA_MOE_PREFETCH=1`), I15, I20, I21, I21C (I21 with `--moe-dispatch-clock`); 50 timed runs in
both orders and the profiled pair (REF, I21); integrity with I15's, I20's and I21's host demand sequences equal;
admissibility; I21 against I20 (`improves`, `regresses`, `flat`, gen-only primary, the window beside it), I21 against
I15 beside it deciding nothing, the door against REF. `regresses` on either class reverts I21 with its receipt. On the
285K class, then a 9950X.

## 2a. I21 on the CPU, before any card

I21 landed as `b555b4141` (memra-tier `bank/slru.rs`, `bank/types.rs`, `bank/residency.rs`, `bank/expert_dispatch.rs`;
memra-engine `banked_residency/native.rs`). One thing the source gave that section 1 did not name: with its reads by
position, the traced adapter no longer reads its own copy of the id map, so that copy (a second `BTreeMap` of every
record's `BankId`) is gone; the dispatch adapter's map stays for `validated`.

**CPU gates** (`day84-cpu/gates.log`): the tier suites (the bank suite 92 passed, with the five `day84` tests: the view
against the table after every one of the day-43 trace's 250,000 operations and the day-4 fixture's 2,013 rows, the
refusals of an indexed policy, the catalog's positions and the metadata charge, the dispatch adapter's reads through two
fills and 400 demands with misses and evictions), the engine library (574 passed), clippy (`-D warnings`, all targets)
and fmt clean, `git diff --check` clean.

**The CPU profile** (`day84-cpu/profile.log`, the engine library's test binaries at I20 and I21, one pinned P-core,
order I20, I21, I21, I20, I20, I21, the host shared with other lanes' work, load average 4.6 to 5.9): P1's
`host_resident` 574 to 607 ns per call at I20, 47 to 51 at I21; P9's grouped cycle 2325 to 2422 ns per block at I20,
1782 to 1907 at I21 (about 23 percent less); P1's single `demand` 2001 to 2081 at I20 and 1961 to 2055 at I21, not
resolved. P10: 16 allocations per grouped cycle at both. So the residency check lost nearly all its cost, and the
demand did not lose the pre-demand read's: that read had also brought the record's SLRU `table` entry into the cache
for the bank's own `hit` in `publish` (the same entry, still hashed), which now pays for it. The in-situ split below
reads where the demand's part went; the bank's own hashed reads are the next registration's (section 1, "Not in I21").

**The local check and the in-situ split** are queued together (queue v18, `rtx5090-queue-v18-20260926.sh`, dry-checked
under stubs in `day84-cpu/dry-check-queue.log`): the check's four runs, then 20 runs of `i20s` and `i21s`, one lock hold,
pinned to the P-cores; `day83-read.py` gained a `--change a,b` line for it (additive, deciding nothing).

**The card sitting, prepared before any cell** (`day84-cell.sh`, `day84-read.py`, `day84-box.sh`: DAY82's with I21 for
I20's step and I20 for I18's, REF on `run-gen-i20`): the reader on a synthetic cell from DAY82's BOX39 receipts
relabelled (`day84-cpu/make-synthetic.py`; meaningless) reads 50 runs, integrity ok, every line printed
(`dry-check-reader.log`); the cell under stubs exits 0 with 10 calls of `run-gen-i15`, 22 of `run-gen-i20` (REF's 10 and
2 profiled, I20's 10) and 22 of `run-gen-i21` (`dry-check-cell.log`); the driver under stubs names the three builds, the
cell, `--validate` and the reader, in order (`dry-check-driver.log`).

**The local check and the in-situ split** (queue v18, 2026-09-26 20:08Z to 20:15Z, `rtx5090-day84/`; run-gen-i20
`3f58bca9...`, run-gen-i21 built by `c-local-build.sh` from `b555b4141`; pinned to P-cores 0-7; the host shared with
other lanes' work, load average 3.4 to 7.0; the card 63 to 68 C):
- `check/reading.log`: `DAY84 GPU CHECK PASS`: every run exits 0 with `MATCH`; I21's tape and host demand sequence
  (`4bdc2610c3534e42`, 22077 lines, every `victim` field included) are I20's in both orders.
- `split21/reading.log`: `DAY83 SPLIT CHECKS rig=rtx5090 runs=20 integrity=ok`, then, verbatim:
  `DAY84 CHANGE rig=rtx5090 generate (i21s minus i20s, us per token, deciding nothing): outer=-129.5
  dispatch_inner=+26.1 own_trace=-0.5 bank_stage_lookup=+0.2 bank_stage_cache=+13.1 bank_stage_charge=-0.2
  stage_rest=-0.0 bank_publish_output=+0.5 bank_publish_policy=+53.6 publish_rest=+0.1 pf_resident=-113.1
  bank_host_use=+0.1 bank_retire_only=-0.0 bank_ack=-1.1 bank_collect=-0.1 retire_outer=+0.3 pf_demand=-26.1
  pf_retire=+0.2 pf_stage=+4.7`.

**Read, deciding nothing.** The two targeted leaves lost 243 us per generated token (`outer` -129.5, `pf_resident`
-113.1), as section 0 said. About 93 of it reappeared in the next reads of the same records, each still hashed and now
the first to touch its entry: the bank's SLRU `hit` in `publish_policy` (+53.6), the dispatch adapter's `validated`
(its id tree and the catalog's layout, `dispatch_inner` +26.1) and the host cache lookup (`stage_cache` +13.1). Net,
the door-only leaves fell by about 150 us per token here (673 to about 523, 22 percent), and `pf_demand` by 26. So
most of the per-record cost is the first cold touch of a record's entries, wherever it happens; the next cut takes
those three reads by position too (the bank's `hit`, `validated`, the host cache), which is `DAY85.md`'s registration.
The card cell measures I21 as it stands.

## 3. The target card, 285K class (BOX41, a Core Ultra 9 285K; run by the lead as registered; `pro-single-day84/`)

BOX41: one RTX PRO 6000 Blackwell Workstation Edition, 249 GB, driver 580.173.02, a fresh box.
`D84_BUILDS="i15=2243b1fe2 i20=8efea3a54 i21=b555b4141" bash .../day84-box.sh` on the tree `012e76a1d`, `box start
2026-09-26T20:29:01Z` to `box done 2026-09-26T20:44:18Z`. The builds name the three trees (`build-*.log`: `tree=2243b1fe2...`,
`tree=8efea3a54...`, `tree=b555b4141...`, each `rc=0`) and the cell's `binary.sha256` equals the box's `BINARIES.sha256`.
Receipts: 261 `OK` against `MANIFEST.sha256` (re-checked here), ELFs and the profiler's files by hash. Regime: 40 to 49 C,
SM median 2820 MHz (2610 to 2850), P0 and P1, N=582 busy samples of 2309; the power brake not active. Verbatim
(`i21/reading.log`):

- `DAY84 host demand sequence i15 sha256 4bdc2610c3534e42 lines=[22077]`, and the same for `i20`, `i21` and `i21c`
- `DAY84 I21 CHECKS rig=pro-single runs=50 integrity=ok`
- `DAY84 ADMISSIBILITY rig=pro-single ceiling=0.005 max_iqr_gen=0.0022 max_iqr_window=0.0013 failing=[] -> admissible`
- `DAY84 gen-only decode medians (N=10 each): ref=0.312 i15=0.316 i20=0.315 i21=0.314 i21c=0.315`
- `DAY84 STEP i21_vs_i20 gen-only decode: pooled=-0.0010 o1=-0.0010 o2=-0.0020 noise=0.0022 -> flat`
- `DAY84 STEP i21_vs_i20 steady window: pooled=-0.0005 o1=+0.0000 o2=-0.0010 noise=0.0010 -> flat`
- `DAY84 BESIDE i21_vs_i15 gen-only decode: pooled=-0.0020 o1=-0.0020 o2=-0.0020 noise=0.0015 -> improves (deciding
  nothing)`
- `DAY84 DOOR i21_vs_ref gen-only decode: pooled=+0.0020 o1=+0.0020 o2=+0.0020 noise=0.0010 -> loses`
- `DAY84 DOOR i21_vs_ref steady window: pooled=+0.0000 o1=+0.0000 o2=+0.0000 noise=0.0002 -> matches`
- `DAY84 I21C brackets per window token (ms, medians): dispatch_ns=0.0781 prefetch_ns=0.4415 pf_demand_ns=0.1313
  pf_resident_ns=0.0028 pf_retire_ns=0.0251 pf_stage_ns=0.2261`
- `DAY84 B i21_minus_ref per window token (ms, medians of two; deciding nothing): gpu_busy=+0.0040 gpu_idle=+0.1054
  h2d_exposed=+0.0519 kernel_sum=+0.0042`
- `DAY84 VERDICT rig=pro-single integrity=ok i21=flat door=i21 vs_ref=loses (window: i21=flat vs_ref=matches)`

**Read as registered: I21 `flat` on the 285K class (-1 ms over 32 tokens gen-only, inside a 2.2 ms noise term), so it
stays; the door `loses` to REF by 2 ms gen-only and `matches` it on the window.**

**Beside it, deciding nothing.** I21 against I15 reads -2 ms in both orders (`improves` by the rule, above its 1.5 ms
noise term), where DAY82's I20 against I15 read -1 ms. The path's own clock moved as the split said: `pf_resident` 0.0028
ms per window token against I20C's 0.0443 on BOX39, `pf_demand` 0.1313 against 0.1499, `prefetch_ns` 0.4415 against
0.5125; under the profiler the door's extra GPU idle is 0.105 ms per window token against 0.228 on BOX39. The door is 2
ms behind REF gen-only here, from 3 on BOX39 and 4 on BOX34 (other hosts of this class).

## 3a. The target card, 9950X (BOX31, a Ryzen 9 9950X; run by the lead as registered; `pro-single-day84-9950x/`)

BOX31: one RTX PRO 6000 Blackwell Workstation Edition, 123 GB; the box had carried lane A's sittings for 19 h before
(the lead's `MIRROR-CHECK.txt`; after the cell: load 0.28, no other process). The same command on the tree `012e76a1d`,
`box start 2026-09-26T20:27:16Z` to `box done 2026-09-26T20:43:48Z`. The chain's own checkout of `FETCH_HEAD` in the
build worktree failed (`fetch rc=128`, `c9-chain.out`); the builds are unaffected: `day63-box-build.sh` checked out each
arm's sha itself, and `build-*.log` name `tree=2243b1fe2...`, `tree=8efea3a54...` and `tree=b555b4141...`, each `rc=0`,
with the cell's `binary.sha256` equal to `BINARIES.sha256`. Receipts: 261 `OK` against `MANIFEST.sha256`. Regime: 29 to
42 C, SM median 2857 MHz, P0 and P1, N=511 busy samples of 2623; the power brake not active. Verbatim:

- `DAY84 I21 CHECKS rig=pro-single runs=50 integrity=ok` (one host demand sequence, `4bdc2610c3534e42`, in every door arm)
- `DAY84 ADMISSIBILITY rig=pro-single ceiling=0.005 max_iqr_gen=0.0570 max_iqr_window=0.0468 failing=['i15:gen_s=0.0570',
  'i15:window_s=0.0468', 'i20:gen_s=0.0535', 'i20:window_s=0.0448', 'i21:gen_s=0.0500', 'i21:window_s=0.0450'] ->
  inadmissible`
- `DAY84 VERDICT rig=pro-single integrity=ok -> void (inadmissible) [as read: i21=flat door=i21 vs_ref=matches (window:
  i21=flat vs_ref=matches)]`

**Read as registered: void, inadmissible; it decides nothing, and the 9950X half of I21's step is still owed.**

**What made it inadmissible, deciding nothing: C12's slow state, back on a long-running host.** Every door arm is
bimodal by boot and REF is not (gen-only seconds per run, order 1 then order 2, `ev/*.log`):

| arm | runs | fast boots (about 0.246 s) | slow boots (0.291 to 0.307 s) |
|---|---|---|---|
| REF | 0.243 to 0.244 | 10 | 0 |
| I15 | | 7 | 3 |
| I20 | | 6 | 4 |
| I21 | | 3 | 7 |
| I21C | | 1 | 9 |

The slow boots are about 50 ms behind over 32 tokens, the size DAY64 read on BOX15 and DAY80 placed on compaction failing
to migrate the door's pinned pool pages. Under the profiler both door runs were slow (`gpu_idle` +2.94 ms per window
token). The counts rise with the arm in both orders (I21C is last in order 1 and first in order 2), so position in the
round does not explain them; ten runs per arm cannot separate the arm from the host's drift either, and this cell does
not sample compaction. DAY82's 9950X half on BOX40 (a fresh host) had no slow boot in any arm. Two things follow, each
registered before it runs: the admissibility clause's rerun of this cell on a fresh 9950X host (as `i15b` was; the
lead runs it on BOX42, a fresh 9950X with one RTX PRO 6000 WS, 184 GB, driver 580.65.06, into
`pro-single-day84-9950x-r2/`), and a cell that asks the arm question directly on a long-running 9950X host with
compaction sampled per run and the registered pool as the control (`DAY86.md`; BOX31 is released, so it waits for such
a host). The registered pool as the door's default stays the owner's question
(`DAY80.md` section 4a); this reading is more evidence for it.

## 3b. The 9950X rerun (BOX42, a fresh Ryzen 9 9950X; run by the lead as registered; `pro-single-day84-9950x-r2/`)

BOX42: one RTX PRO 6000 Blackwell Workstation Edition, 184 GB, driver 580.65.06, a fresh host (not DAY82's 9950X
machine; the same class). The chain fetched the lane tip at launch, `2bebd4900`; the cell, reader, driver and runner
scripts are byte-for-byte those of `012e76a1d` (`git diff` empty), and the arms are the registered builds (`build-*.log`
name `tree=2243b1fe2...`, `tree=8efea3a54...`, `tree=b555b4141...`, each `rc=0`; the cell's `binary.sha256` equals
`BINARIES.sha256`). `box start 2026-09-26T20:56:27Z` to `box done 2026-09-26T21:13:09Z`. Receipts: 261 `OK` against
`MANIFEST.sha256`, ELFs and profiles by hash; the reader re-run here on the mirror prints `reading.log` byte for byte.
Regime: 31 to 47 C, SM median 2850 MHz, P0 and P1, N=578 busy samples of 2612; the power brake not active. No slow boot
in any arm (every door run 0.320 to 0.338 s gen-only). Verbatim (`i21/reading.log`):

- `DAY84 I21 CHECKS rig=pro-single runs=50 integrity=ok` (one host demand sequence, `4bdc2610c3534e42`, in every door arm)
- `DAY84 ADMISSIBILITY rig=pro-single ceiling=0.005 max_iqr_gen=0.0012 max_iqr_window=0.0010 failing=[] -> admissible`
- `DAY84 gen-only decode medians (N=10 each): ref=0.319 i15=0.323 i20=0.323 i21=0.321 i21c=0.322`
- `DAY84 STEP i21_vs_i20 gen-only decode: pooled=-0.0020 o1=-0.0020 o2=-0.0020 noise=0.0010 -> improves`
- `DAY84 STEP i21_vs_i20 steady window: pooled=+0.0000 o1=+0.0000 o2=+0.0000 noise=0.0003 -> flat`
- `DAY84 BESIDE i21_vs_i15 gen-only decode: pooled=-0.0020 o1=-0.0020 o2=-0.0030 noise=0.0010 -> improves (deciding
  nothing)`
- `DAY84 DOOR i21_vs_ref gen-only decode: pooled=+0.0020 o1=+0.0030 o2=+0.0020 noise=0.0010 -> loses`
- `DAY84 DOOR i21_vs_ref steady window: pooled=+0.0010 o1=+0.0010 o2=+0.0000 noise=0.0010 -> matches`
- `DAY84 I21C brackets per window token (ms, medians): dispatch_ns=0.0721 prefetch_ns=0.4386 pf_demand_ns=0.1246
  pf_resident_ns=0.0020 pf_retire_ns=0.0203 pf_stage_ns=0.2292`
- `DAY84 B i21_minus_ref per window token (ms, medians of two; deciding nothing): gpu_busy=+0.0076 gpu_idle=+0.1330
  h2d_exposed=+0.0021 kernel_sum=+0.0076`
- `DAY84 VERDICT rig=pro-single integrity=ok i21=improves door=i21 vs_ref=loses (window: i21=flat vs_ref=matches)`

**Read as registered, the pair: I21 `improves` on the 9950X class (-2 ms over 32 tokens gen-only in both orders, a 1
ms noise term; the window flat) and is `flat` on the 285K class (-1 ms inside a 2.2 ms noise term). Neither class
regresses, so I21 stays. I21 is the first cut of the door's prefetch path since I15 that one class resolves on its own
step.** The door still `loses` to REF by 2 ms gen-only on both classes and `matches` it on the window on both. On this
host `pf_resident` reads 0.0020 ms per window token (0.0443 at I20 on BOX39's 285K) and `pf_demand` 0.1246. The DAY86
question (whether the cuts make C12's slow state more frequent on a long-running 9950X) stays open: this fresh host
had no slow boot, as BOX40 had none.
