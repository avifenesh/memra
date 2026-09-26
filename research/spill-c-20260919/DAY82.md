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
