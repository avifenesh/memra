# WP-C day 85 (2026-09-26): OWED C11, I22, the lease path's remaining per-record hashed reads by position, before any code

Lead: "Register DAY85 meanwhile" (DAY84's cell `i21` is running on both classes). Also: "my chain logs before today's
c8/c9 print rc= after a $(date) expansion, so their rc is date's 0, never the step's. Your pro-single-day13/driver.sh
has the same pattern. Check whether any DAY record quoted a lead-chain or driver rc line as evidence, and fix the
pattern in live scripts." Tree at start: `012e76a1d`.

## 0. The rc check

In bash, `$?` read after a command substitution in the same statement is the substitution's status: `false; echo
"$(date) rc=$?"` prints `rc=0` (checked here; `PIPESTATUS` survives a substitution and is not affected).

- **Lane scripts.** `rc-scan.py` (new, this directory) reads every `.sh` under the lane's records for a `$?` after a
  `$(` in one statement (`day85-cpu/rc-scan.log`). One hit in all 134 top-level scripts, 47 CPU-gate scripts and every
  mirrored receipt directory: `pro-single-day13/driver.sh:9`, `echo "$(date -u +%FT%TZ) $name rc=$?"`. It is a receipt
  (the script that ran on the day-13 box), so it stays as it ran. No live script has the pattern (`--live`: 0). A second,
  wider pass (`day85-cpu/rc-scan-wide.py`, `rc-scan-wide.log`) read every `rc=$?` with the statement before it (an
  `echo`, `cat`, `tee` or pipeline without `pipefail` in front of the read, or a closing `fi`); of its five flags, one
  is the day-13 line and four are correct reads on inspection (a `{ ...; }` group whose last command is the step, an
  `if` whose branches end in the step).
- **Where it mattered.** `pro-single-day13/driver.log` holds two runs: attempt 1 by `driver.sh` (twelve `rc=0` lines,
  meaningless) and the second run by `driver2.sh`, which reads the status first (`local rc=$?`); its twelve lines agree
  with the cells' `.exit` files (six gates red, as `DAY13.md` reads them by their gate lines). `DAY13.md` never quoted
  a driver `rc=` line; a note under it now says so.
- **Records against lead-chain lines.** No DAY record cites a lead chain log's `rc=` line. The `rc=` values quoted
  across DAY17 to DAY84 come from this lane's own build logs, `--validate` runs and cell drivers, whose reads the scan
  covers (for example `c-dayNN-driver.out`'s `i15b rc=0 <date>` reads `$?` before the date).

## 1. What I21's in-situ split leaves (`DAY84.md` section 2a)

At I21, per generated token on the local host: `stage_cache` 102.6 us, `dispatch_inner` 94.0, `publish_policy` 88.3,
`retire_outer` 55.7, `outer` 46.6. The first three grew under I21 by 13.1, 26.1 and 53.6 because they are now the
first touch of a record's entries, and each is a hashed read keyed by the record's `BankId`:
- `stage_cache`: the bank's host cache (`CacheIndex`, an Fx-hashed map from `BankId` to its lease) read per unique id;
- `dispatch_inner`: the dispatch adapter's `validated` (its id tree, then the catalog's hashed index for the layout);
- `publish_policy`: the SLRU `hit` (the `table`, hashed) and, for a prefetch, `resident`;
plus the catalog's hashed index in `stage`'s lookup loop (`stage_lookup`, 8.8).

## 2. Pre-registration: I22

**The change.** The catalog position I21 resolves once per record travels with the lease to the bank, and these reads
use it; every map stays, authoritative, for every other caller.
- `Catalog::entry_at(position)` and `Catalog::id_at(position)` (the catalog's own vector; `MaskedId` and `NotFound` as
  `entry` gives them).
- `CacheIndex` keeps, when the SLRU is installed, a lease per catalog position beside its map, kept equal at each of
  its changes (the fill admission, a missing record's publication, the eviction paths and `release`, each resolving
  the position through the catalog's hashed index: miss and eviction paths only); `get_at(position)`. The SLRU
  metadata charge counts it (8 bytes per catalog id, beside I21's 4 per id and 4 per slot).
- `SlruPolicy::hit_at(position)`: `hit` for the record at that position, reading `resident_at`.
- `BankService::stage_at(batch, positions)`, crate-private (only the dispatch adapter, whose positions are the
  catalog's own, can call it): `stage`'s program with the catalog entry and the host cache read by position; the
  pending ticket keeps the positions, and its publication reads `hit_at` / `resident_at` for them. `stage(batch)`
  keeps its hashed reads unchanged for every other caller. A debug build asserts `id_at(position) == id` for every
  record of a positioned batch.
- `SlruExpertDispatch::validated` reads the position (I21's dense table), the layout through `entry_at` and the id
  through `id_at` (the same `BankId`: the adapter's map was built from the catalog's ids), with the errors it gave
  (`NotFound` for an unknown local id, then the layout's); `demand` and `demand_many` stage through `stage_at`.

Not in I22 (named, each its own registration after its reading): the batch's `BankId` clones (one heap string each),
the proxy's registry entry, identity checks and pending insert (`outer`), the retire side (`retire_outer`,
`bank_ack`).

**The same program.** No decision reads anything new; the host demand sequence and the tokens must equal I21's.

**If DAY84's cell `i21` reads `regresses` on either class,** I21 is reverted with its receipt and this registration is
rewritten on I20 before any code (I22 needs I21's positions).

## 3. CPU gates before any card (`day85-cpu/`)

- the positioned bank against the hashed one: twin banks driven by one randomized trace of grouped and single demands
  (host hits, misses, evictions, fill admissions, cancels) through `stage_at` and `stage`, with the same tickets,
  leases (ids and charges), SLRU orders and host cache after every operation; the cache view against its map after
  every operation; `hit_at` against `hit` on the day-43 trace; `validated` by position against the map for every
  local id;
- the day-61 profile at I21 and I22 in one window (P1 and P9, both orders) and P10's census;
- the tier suites, the engine library, clippy (`-D warnings`, all targets) and fmt; `rc-scan.py --live`;
- the local RTX 5090 check (I21 and I22, both orders, `MATCH`, the same tape and host demand sequence), then the
  in-situ split of I22 beside I21 (queue v18's shape, arms `i21s` and `i22s`), deciding nothing.

## 4. Pre-registration: the card cell `i22`

DAY84's cell with I22 for I21's step: binaries `i15=2243b1fe2`, `i21=b555b4141` and `i22` (named in section 3a); arms
REF (`run-gen-i21` with `MEMRA_MOE_PREFETCH=1`), I15, I21, I22, I22C; 50 timed runs in both orders and the profiled pair;
integrity with I15's, I21's and I22's host demand sequences equal; admissibility; I22 against I21 (gen-only primary, the
window beside it), I22 against I15 beside it deciding nothing, the door against REF. `regresses` on either class
reverts I22 with its receipt. On the 285K class, then a 9950X.

## 3a. I22 on the CPU, before any card

I22 landed as `4b378a064` (memra-tier `bank/types.rs`, `bank/slru.rs`, `bank/residency.rs`, `bank/expert_dispatch.rs`;
no engine source changed). As registered, with two details the source gave: the positioned path keeps `stage`'s
refusals in their order (a record outside the domain, then the id's validation, then the entry), and `stage_at`
refuses `Unsupported` without the cache's view or with a position list of another length, before any change.

**CPU gates** (`day85-cpu/gates.log`): the tier suites (the bank suite 96 passed, with four `day85` tests: the adapter,
which now stages by position, equals a twin bank driven through the hashed `stage` with the pre-I22 adapter's batches
over 600 randomized grouped demands, single demands and fill admissions, for demand and for prefetch priority, the same
outcomes, leases and bytes, reads, SLRU orders, residency and host cache after every step; `hit_at` against `hit` over
the day-43-style trace; `validate` by position against the id's layout and the catalog's `id_at`), and `day84`'s charge
test updated to the formula with the cache view; the engine library (574 passed); clippy (`-D warnings`, all targets)
and fmt clean; `git diff --check` clean; `rc-scan.py --live` 0.

**The CPU profile** (`day85-cpu/profile.log`, the engine library's test binaries at I21 and I22, one pinned P-core,
order I21, I22, I22, I21, I21, I22, the host shared with other lanes' work, load average up to 10.7 by the end): P1's
single `demand` 2337 to 2529 ns at I21, 1281 to 1367 at I22 (about 45 percent less); P9's grouped cycle 2142 to 2282 ns
per block at I21, 830 to 958 at I22 (about 60 percent less). P10: 17 allocations per grouped cycle at I22 against 16 at
I21 (the positions travel in their own vector; 718 bytes against 830).

**The local check and the in-situ split** are queued together (queue v19, `rtx5090-queue-v19-20260926.sh`, queue
v18's shape with I21 and I22, dry-checked under stubs in `day85-cpu/dry-check-queue.log`).

**The card sitting, prepared before any cell** (`day85-cell.sh`, `day85-read.py`, `day85-box.sh`: DAY84's with I22 for
I21's step and I21 for I20's, REF on `run-gen-i21`): the reader on a synthetic cell from DAY82's BOX39 receipts
relabelled (`day85-cpu/make-synthetic.py`; meaningless) reads 50 runs, integrity ok (`dry-check-reader.log`); the cell
under stubs exits 0 with 10 calls of `run-gen-i15`, 22 of `run-gen-i21` and 22 of `run-gen-i22` (`dry-check-cell.log`);
the driver under stubs names the three builds, the cell, `--validate` and the reader, in order (`dry-check-driver.log`).

**Before the card: the local check's place.** Queue v19 waits for the local RTX 5090 (other lanes' work holds it). As
in `DAY82.md` section 2a, recorded before any result: if the card cell `i22` runs first, its own integrity reads the
same two properties on the target card (`MATCH` in every run, one host demand sequence across I15, I21 and I22), the
local check is read when it lands, and the in-situ split beside it decides nothing either way.

## 5. The target card, both classes (run by the lead as registered, chain tree `8d1863980`)

Both halves: `D85_BUILDS="i15=2243b1fe2 i21=b555b4141 i22=4b378a064" bash .../day85-box.sh`; the cell, reader and driver
scripts are those committed in `168bcc0f5` (`git diff` empty); the builds name the three trees, each `rc=0`, and each
cell's `binary.sha256` equals its box's `BINARIES.sha256`; 261 receipts `OK` against each box manifest, ELFs and
profiles by hash; the reader re-run here on each mirror prints its `reading.log` byte for byte. No slow boot in any arm
on either host.

**285K class** (BOX41, a Core Ultra 9 285K, one RTX PRO 6000 WS, 249 GB, driver 580.173.02, after integ69's GPU run 4
on the same card; `pro-single-day85/`; 21:51Z to 22:06Z; 39 to 48 C, SM median 2820 MHz, N=549 busy samples of 2315):
- `DAY85 I22 CHECKS rig=pro-single runs=50 integrity=ok` (one host demand sequence, `4bdc2610c3534e42`, in every door arm)
- `DAY85 ADMISSIBILITY rig=pro-single ceiling=0.005 max_iqr_gen=0.0020 max_iqr_window=0.0020 failing=[] -> admissible`
- `DAY85 STEP i22_vs_i21 gen-only decode: pooled=-0.0010 o1=+0.0010 o2=-0.0020 noise=0.0020 -> flat`
- `DAY85 STEP i22_vs_i21 steady window: pooled=-0.0005 o1=+0.0000 o2=-0.0010 noise=0.0020 -> flat`
- `DAY85 BESIDE i22_vs_i15 gen-only decode: pooled=-0.0020 o1=-0.0010 o2=-0.0030 noise=0.0020 -> flat (deciding
  nothing)`
- `DAY85 DOOR i22_vs_ref gen-only decode: pooled=+0.0020 o1=+0.0030 o2=+0.0010 noise=0.0020 -> matches`
- `DAY85 DOOR i22_vs_ref steady window: pooled=+0.0005 o1=+0.0010 o2=+0.0000 noise=0.0010 -> matches`
- `DAY85 I22C brackets per window token (ms, medians): dispatch_ns=0.0727 prefetch_ns=0.3489 pf_demand_ns=0.0669
  pf_resident_ns=0.0019 pf_retire_ns=0.0259 pf_stage_ns=0.2245`
- `DAY85 VERDICT rig=pro-single integrity=ok i22=flat door=i22 vs_ref=matches (window: i22=flat vs_ref=matches)`

**9950X class** (BOX43, a Ryzen 9 9950X, one RTX PRO 6000 WS, 184 GB, driver 580.65.06, its container's first sitting,
host uptime 130029 s at the chain's start; `pro-single-day85-9950x/`; 21:43Z to 22:00Z; 32 to 47 C, SM median 2850 MHz,
N=538 busy samples of 2613):
- `DAY85 I22 CHECKS rig=pro-single runs=50 integrity=ok` (the same one host demand sequence)
- `DAY85 ADMISSIBILITY rig=pro-single ceiling=0.005 max_iqr_gen=0.0010 max_iqr_window=0.0020 failing=[] -> admissible`
- `DAY85 STEP i22_vs_i21 gen-only decode: pooled=-0.0020 o1=-0.0020 o2=-0.0020 noise=0.0010 -> improves`
- `DAY85 STEP i22_vs_i21 steady window: pooled=-0.0010 o1=-0.0020 o2=+0.0000 noise=0.0020 -> flat`
- `DAY85 BESIDE i22_vs_i15 gen-only decode: pooled=-0.0040 o1=-0.0030 o2=-0.0040 noise=0.0010 -> improves (deciding
  nothing)`
- `DAY85 DOOR i22_vs_ref gen-only decode: pooled=+0.0010 o1=+0.0010 o2=+0.0010 noise=0.0010 -> matches`
- `DAY85 DOOR i22_vs_ref steady window: pooled=+0.0000 o1=-0.0010 o2=+0.0010 noise=0.0020 -> matches`
- `DAY85 I22C brackets per window token (ms, medians): dispatch_ns=0.0669 prefetch_ns=0.3436 pf_demand_ns=0.0590
  pf_resident_ns=0.0016 pf_retire_ns=0.0210 pf_stage_ns=0.2243`
- `DAY85 VERDICT rig=pro-single integrity=ok i22=improves door=i22 vs_ref=matches (window: i22=flat vs_ref=matches)`

**Read as registered: I22 `improves` on the 9950X class (-2 ms over 32 tokens gen-only in both orders) and is `flat` on
the 285K class; neither regresses, so I22 stays. The door `matches` REF on both classes, gen-only and window, for the
first time since C11 opened.** Stated plainly, as the rule reads it: on the 285K the door is still 2 ms behind REF
gen-only by the medians (1 and 3 by order), and it reads `matches` because this sitting's noise term is 2 ms (DAY84's
BOX41 sitting read the same +2 ms against a 1 ms term as `loses`); on the 9950X it is 1 ms behind against a 1 ms term.
So the gen-only gap is now one to two ticks of the printed resolution on both classes, from 9 to 10 ms at I15. The
door's own clock: `pf_demand` 0.059 to 0.067 ms per window token (0.125 to 0.131 at I21, 0.150 at I20), `prefetch_ns`
0.344 to 0.349 (0.513 at I20).
