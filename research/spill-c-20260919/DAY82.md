# WP-C day 82 (2026-09-26): OWED C11, where the grouped host-hit cycle still spends, then I20, before any code

`DAY81.md` section 2a withdrew I19 (its order was I18's) and left C11 here: the door's gap is its prefetch's CPU cost
between two launches in a launch-bound decode, and only cutting that cost closes it. I17 and I18 each cut about 150 ns
per block and read `flat`. Tree at start: `5f8824d6b`.

## 1. Pre-registration: an allocation census of the grouped cycle (an instrument; decides nothing)

The day-61 profile times parts; it does not say how many heap allocations a cycle makes, and in the grouped cycle
(P9, the door's path: `host_resident_many`, `demand_many`, `with_bytes_each`, `finish_group`) several structures are
built per call: the dispatch's `BankBatch` (a `Vec` and a cloned `BankId`, whose `TensorId` holds a `String`, per
record), clones of the `BudgetRequest` (its budgets hold vectors) in the dispatch and twice in `stage`, the ticket's
records, the pending ticket's map node, the proxy's lease entry and token. The census: a counting wrapper around the
system allocator in the engine library's test build only (`#[cfg(test)]`, counts and bytes, no behavior change),
read around P9's loop, printed as `DAY82 P10 allocations per grouped cycle=<n> bytes=<b>` and per call
(`host_resident_many`, `demand_many`, `with_bytes_each`, `finish_group`, each read in its own loop over the same
records). It runs in the same invocation as P9, on the local CPU, one pinned core.

## 2. Pre-registration: I20, chosen by the census (registered now, its content fixed by the census's largest term)

**The rule, stated before the census reads.** I20 removes the allocations of the census's largest call, by the
smallest change that keeps every answer, order and refusal: borrowed or reused structures in place of per-call clones
(for example a request held by reference where it is only read, ids by position where they are only compared), never
a skipped protocol step. If the largest term is the `BudgetRequest` clones, I20 keeps the dispatch's request by
reference into `stage` and stores what the pending ticket needs of it; if it is the per-record `BankId` clones, I20
builds the batch from references the catalog already holds; if it is the proxy's own entry, I20 reuses its buffers.
The choice and its before and after counts are written in section 2a before any card cell.

**CPU gates before any card** (`day82-cpu/`): the census before and after (the allocations per grouped cycle must
fall, and each call's count is printed); the day-61 profile's P9 at I18 and I20 in one window (both orders); the tier
suites and the engine library; clippy and fmt; a local RTX 5090 check that I20 reads `MATCH`, I18's tape and I18's
host demand sequence (the program is the same, so the sequence must be too).

## 3. Pre-registration: the card cell `i20`

DAY79's cell with I20 in I19's place: binaries `i18=c7294b912` and `i20` (named in section 2a); arms REF, I18, I20,
I20C; 40 timed runs in both orders and the profiled pair; integrity as DAY79's with I20's host demand sequence equal
to I18's; admissibility; I20 against I18 and the door against REF; `regresses` reverts I20 with its receipt. On the
285K class, then a 9950X.

## 2a. The census, I20, and one addition to section 3 before any cell

**The census at I18** (`day82-cpu/census-before.log`, the local CPU, one pinned core): `DAY82 P10 allocations per
grouped cycle=22.00 bytes=878.2 | host_resident_many=0.00/0.0 demand_many=21.00/854.2 with_bytes_each=0.00/0.0
finish_group=1.00/24.0`. The largest call is `demand_many`, and by section 2's rule its largest removable term is the
`BudgetRequest` clones: each clone copies the three budget vectors of `TierBudget` (device, peer, replicas), and
`stage` made two per ticket (the queue's request, and the pending ticket's copy), six allocations of the 21.

**I20** (`8efea3a54`, memra-tier `residency.rs`): `stage` reserves the queue charge on the batch's own request, its
three queue dimensions (`pageable`, `staging`, `inflight`) set for the reservation and restored after it; a missing
record's request is built from the batch's priority, deadline and tenant with a zero budget (as the clone had after its
budget was replaced); the pending ticket takes the batch's request by move. The same values are reserved and kept. The
dispatch's own clone into the batch stays (the ticket owns its request); the census after (`census-after.log`): 16
allocations per grouped cycle, `demand_many` 15. CPU gates (`day82-cpu/gates.log`): the tier suites and the engine
library green, clippy and fmt clean.

**The CPU profile** (`day82-cpu/profile.log`, the engine library's test binaries at I18 and I20, one pinned core, order
I18, I20, I20, I18, I18, I20, the rig shared with other lanes' agents): P9 reads 2440.7 to 2547.7 ns per block at I18
and 2494.1 to 2598.4 at I20: no difference this machine resolves under that load. Six small allocations are a few tens
of nanoseconds per block; the grouped cycle's time is not mostly allocation.

**Added to section 3 before any cell: I15 as a fifth arm.** Three improvements in a row (I17, I18, I20) have each been
below the target card's resolution by the per-step rule. The cell `i20` therefore also runs I15 (`2243b1fe2`) and
prints I20 against I15 beside the registered step (the same rule, deciding nothing), so the card reads what the
cumulative path cut since I15 buys. The arms are REF, I15, I18, I20, I20C, 50 timed runs in both orders; integrity also
requires I15's and I18's host demand sequences to equal I20's (the same program, as DAY77 and DAY79 read).

**The cell's scripts, dry-checked before any card** (`day82-cpu/`): `day82-cell.sh` (DAY79's with the five arms),
`day82-read.py` (DAY79's with I15 and the beside line), `day82-box.sh` (DAY79's with the three builds). The reader on a
synthetic cell from DAY79's BOX34 receipts relabelled (`make-synthetic.py`; the reading means nothing):
`dry-check-reader.log`, 50 runs, integrity ok, the beside line printed. The cell under stubs (`dry-check-cell.sh`): rc
0, 10 calls of `run-gen-i15`, 22 of `run-gen-i18` (REF's 10 and 2 profiled, I18's 10) and 22 of `run-gen-i20`. The
driver under stubs (`dry-check-driver.sh`): the three builds named, the cell, `--validate`, the reader, in order.

**The local RTX 5090 check, queued** (`day82-cpu/gpu-check.sh`, its reader `gpu-check-read.py`, queue v16): the card
has been held by another lane's timing queue since before the check was ready, so the check waits (queue v16, up to
48 h, behind `/tmp/memra-5090.lock` and an idle card) and writes `day82-cpu/gpu-check.log` when it runs. Its reader,
dry-checked on DAY79's local receipts, flags a run without the door's demand lines. Recorded before any result: if the
target-card sitting runs before the check lands, the cell's own integrity (MATCH in every run, one host demand sequence
across I15, I18 and I20) reads the same two properties on the target card, and the local check is read when it lands,
deciding nothing further for the cell.

**The local RTX 5090 check landed** (queue v16, `day82-cpu/gpu-check.log`, raw logs in `day82-cpu/gpu-check/`):
`DAY82 GPU CHECK PASS`. Every run exits 0 with `MATCH`; I20's tape and host demand sequence (`4bdc2610c3534e42`, 22077
lines) are I18's in both orders.

## 4. The target card (BOX39, a Core Ultra 9 285K; run by the lead as registered; `pro-single-day82/`)

BOX39: one RTX PRO 6000 Blackwell Workstation Edition, 249 GB, driver 580.173.02, after integ69's GPU battery on the
same card. `D82_BUILDS="i15=2243b1fe2 i18=c7294b912 i20=8efea3a54" bash .../day82-box.sh` on the tree `ab5e91667`,
`box start 2026-09-26T18:12:23Z` to `box done 2026-09-26T18:27:39Z`. Receipts: 260 `OK` against the box manifest, ELFs
by hash; the profiler's reports and exports (8 files, all `OK` against `profiles.sha256`) moved out of the tree. The
reader re-run here on the mirror with the exports restored prints `i20/reading.log` byte for byte. Regime: 39 to 49 C,
SM median 2820 MHz (2610 to 2850), P0 and P1, N=581 busy samples of 2308; the power brake not active. Verbatim
(`i20/reading.log`):

- `DAY82 host demand sequence i15 sha256 4bdc2610c3534e42 lines=[22077]`, and the same for `i18`, `i20` and `i20c`
- `DAY82 I20 CHECKS rig=pro-single runs=50 integrity=ok`
- `DAY82 ADMISSIBILITY rig=pro-single ceiling=0.005 max_iqr_gen=0.0025 max_iqr_window=0.0010 failing=[] -> admissible`
- `DAY82 gen-only decode medians (N=10 each): ref=0.312 i15=0.316 i18=0.315 i20=0.315 i20c=0.315`
- `DAY82 STEP i20_vs_i18 gen-only decode: pooled=+0.0000 o1=+0.0000 o2=+0.0000 noise=0.0012 -> flat`
- `DAY82 steady window medians (N=10 each): ref=0.286 i15=0.286 i18=0.286 i20=0.286 i20c=0.286`
- `DAY82 STEP i20_vs_i18 steady window: pooled=-0.0005 o1=-0.0010 o2=+0.0000 noise=0.0010 -> flat`
- `DAY82 BESIDE i20_vs_i15 gen-only decode: pooled=-0.0010 o1=-0.0010 o2=-0.0010 noise=0.0025 -> flat (deciding nothing)`
- `DAY82 BESIDE i20_vs_i15 steady window: pooled=+0.0000 o1=+0.0000 o2=-0.0010 noise=0.0010 -> flat (deciding nothing)`
- `DAY82 DOOR i20_vs_ref gen-only decode: pooled=+0.0030 o1=+0.0040 o2=+0.0030 noise=0.0010 -> loses`
- `DAY82 DOOR i20_vs_ref steady window: pooled=+0.0000 o1=+0.0000 o2=+0.0000 noise=0.0010 -> matches`
- `DAY82 I20C brackets per window token (ms, medians): dispatch_ns=0.0827 prefetch_ns=0.5125 pf_demand_ns=0.1499
  pf_resident_ns=0.0443 pf_retire_ns=0.0249 pf_stage_ns=0.2292`
- `DAY82 B i20_minus_ref per window token (ms, medians of two; deciding nothing): gpu_busy=+0.0064 gpu_idle=+0.2280
  h2d_exposed=+0.0633 kernel_sum=+0.0065`
- `DAY82 VERDICT rig=pro-single integrity=ok i20=flat door=i20 vs_ref=loses (window: i20=flat vs_ref=matches)`

**Read as registered: I20 `flat`, so it stays; the door `loses` to REF gen-only (+3 ms over 32 tokens, both orders)
and `matches` it on the window.** The host demand sequence is one across I15, I18 and I20, as the program requires.

**Beside it, deciding nothing.** I20 against I15 reads -1 ms gen-only in both orders and 0 on the window, inside its
noise term (0.0025, set by I15's own spread): the whole cut since I15 (I17, I18, I20, about 300 ns per block on the
local CPU) does not resolve on this card either. The clocks agree: I20C's `pf_demand_ns` is 0.1499 ms per window token
against I18C's 0.1495 on BOX34 and `pf_stage_ns` 0.2292 against 0.2274, so the six allocations I20 removed are not
visible in the path's own clock. Part B holds its shape (the door's GPU idles 0.23 ms per window token more than REF's
with the same kernels, `h2d_exposed` +0.06, as BOX34's +0.20 and +0.06). What this leaves for C11, stated plainly: on
one host, the three cuts since I15 (I17, I18, I20) together read -1 ms over 32 tokens, inside the noise. The door's
gen-only gap to REF on this class runs +10 ms (BOX32, `DAY77.md`), +4 (BOX34) and +3 (BOX39); that spread is the host,
as `DAY79.md` section 3 read it, not the cuts. The window matches REF on BOX34 and BOX39.

**The 9950X half stands.** Section 3 registered the 285K class, then a 9950X, and `regresses` on either class reverts
I20; a `flat` on one class does not answer the other. It runs the same command on a 9950X host.
