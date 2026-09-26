# WP-C day 74 (2026-09-26): OWED C12, compaction induced on purpose, registered before any cell or script

`DAY73.md` section 2a ends: "That points to compaction running while the process runs as the trigger, and to the door's
process being the one it lands on, but no registered cell has read it ... The next registration tests it directly."
Tree at start: `8d0d01d01`.

## 0. Why induce it

Across the three 9950X-family sittings with the sampler, the slow state appeared only in door runs whose span had
compaction (machine `b` 19 of 20, BOX30 1 of 10, BOX31 none of either), and REF saw compaction only in its first two
runs on `b`, where it read 7.4 to 8.4 cycles per step later in those runs. Whether compaction happens is set by the host
(its memory and fragmentation), so waiting for it gives a cell that reads `not_reproduced` on a clean host (BOX31) and
cannot place cause on a host where it always happens (`b`). Inducing it in half the runs, for both programs, on the
same host, separates the two questions: does compaction running during a run slow the core of whatever process it
lands on, and does it land on the door's process and not on REF's.

## 1. Pre-registration: the cell `induce` (any Ryzen 9 9950X machine, then the 285K class; before its scripts)

A measurement cell; no code, no default, no host setting changes. The binary `p71=6bad38150`, DAY73's run shape (ONE
pin, `--cpu-probe`, `--cpu-probe-phases`, `--cpu-probe-counters` when its check passes, the day-18 pressure shape),
DAY73's sampler, one collector hold.

**The inducer (`day74-compactor.py`, one Python process started and stopped by the cell, pinned to the sampler's CPU,
which lies outside the ONE pin and its SMT siblings).** Before an induced run it fragments host memory:
it maps `F` GiB of anonymous memory with `MADV_NOHUGEPAGE`, touches every page, and returns every other 4 KiB page with
`MADV_DONTNEED`, leaving the free memory in order-0 holes. During the run it asks for huge pages: in a loop it maps
256 MiB with `MADV_HUGEPAGE`, touches it and unmaps it, so the kernel compacts (directly, in the inducer's context, as
`defrag=madvise` on these hosts does). It stops when the run ends and releases everything. `F` is the host's
`MemAvailable` at the cell's start minus 56 GiB, capped at 64 GiB (at least 8 GiB, or the induced arms do not run and
the cell reads `not_run`). The inducer writes its own row per second (`ev/compactor.tsv`: pages held, huge-page maps
done, its CPU time) so its activity is on record.

**Arms and order (24 runs):** REF, REF+I (REF with the inducer running), door, door+I; order 1 (REF, REF+I, door,
door+I) x 3, order 2 reversed x 3. Each induced run's fragmentation is set up before the run starts and held until it
ends; an uninduced run follows a released inducer.

**Readings** (each run over its gate-to-window span; the state from the gate probe and from the `window` phase probe):
- R1 per run: `cycles_per_step` at `gate` and at `window` (APERF over 2^20 steps), and whether the span had compaction
  (`compact_isolated` moved).
- R2 per arm: the median `cycles_per_step` at `window`, and the median `compact_isolated` and `pgmigrate_fail` per
  second over the span.
- R3 per arm: gen-only and window medians (deciding nothing here; the timing cells decide timing).

**The verdict** (`DAY74 INDUCE VERDICT rig=<rig>`):
- `void` if integrity fails (24 runs, exit 0, `MATCH`, 32 generated and 32 window tokens, one tape, the door's fill
  and zero physical reads, the sampler bracketing every span), `not_run` if the inducer could not run (`F` below 8 GiB
  or its setup failed), and `not_induced` if fewer than 5 of the 6 induced door runs, or fewer than 5 of the 6 induced
  REF runs, had compaction in their span (the inducer did not reach its goal on this host).
- Otherwise two fields, each by the strict rule over the runs' `window` probes: `door_slows` when every door+I run's
  `cycles_per_step` is above every door run's; `ref_slows` when every REF+I run's is above every REF run's. The verdict
  names the fields that hold, or `neither`.

**What it decides, stated before the cell.** `door_slows` and `ref_slows`: compaction running on the host slows any
process's core, and the door is exposed because its runs see more of it; the remedy question becomes the host's
memory state and the door's share of it. `door_slows` alone: compaction lands on the door's process specifically
(what it maps or pins), and the remedy is the door's memory. `neither`: induced compaction does not produce the state
and the association in `DAY73.md` section 2a was not the cause. A remedy is its own registration after this reads.

**If the local dry check cannot induce compaction** (section 1a, on the development host, at a small `F`), the inducer's
design is amended here, before any cell, and the amendment is stated.

## 1a. Amendment after the local dry check, before any cell; the sitting prepared

**What the local dry check showed** (`day74-cpu/dry-check-compactor.log`, the development host, one core under the cap).
At `F` = 6 GiB and then 2 GiB the inducer fragments and loops as designed, but every 256 MiB burst came back fully as
huge pages (`AnonHugePages` 262144 kB) and `compact_stall` did not move during the loop: on a host with free high-order
blocks it does not force compaction at a small `F`. A larger `F` could not be tried here (the rig is shared). So, as
section 1 provides, the inducer's design is amended before any cell:
- **Its size comes from free memory, not available memory.** `F` = min(64, 2 x (`MemFree` - 48 GiB)) at the cell's
  start: the fragmented half it holds comes out of free pages, and at least 48 GiB of free memory stays in order-0 holes
  for the run, so the inducer does not push the artifact's page cache out (a door run needs the fill from the page
  cache, `physical_reads=0`). Below `F` = 8 GiB the induced arms do not run and the cell reads `not_run`.
- **Its huge-page loop starts at the run's `gate` line**, not before the run: the cell watches the run's log and signals
  the inducer (SIGUSR1) when the gate probe prints, so the load and the door's fill are never under induced
  compaction, and the loop covers the span the readings measure.
- **It also triggers compaction directly when it can**: at the loop's start it opens `/proc/sys/vm/compact_memory`; when
  the file is writable (root with a writable `/proc/sys`), every loop pass writes `1` to it. That is a one-shot
  kernel action per write, not a setting; whether it was writable is on record (`loop trigger=writable|none`).
The fragmentation before the run and the verdict's `not_induced` guard are unchanged; `not_induced` is the reading if
the host still gives huge pages without compacting.

**Scripts and dry checks** (`day74-cpu/`): `day74-compactor.py` as amended; `day74-cell.sh` (DAY73's run shape and
sampler, four arms, the inducer's start, ready wait, gate signal and stop by its own pid; one knob, `D74_DRY_F`, used
only by the dry check to run a 1 GiB inducer, never set by the driver); `day74-read.py`; `day74-box.sh`. The cell's
control flow with stub binaries (`dry-check-cell.log`: 24 runs, each of the 12 induced runs shows its inducer
`ready`, `loop` at the gate, `stopped`; nothing outlives the cell); the reader on machine `b`'s runs relabelled into the
four arms (`dry-check-reader.log`, its reading meaningless) and with `induce=0` (`not_run`); the driver's control flow
(`dry-check-driver.log`).

Run as `D74_BUILDS="p71=6bad38150" [D74_RIG=<name>] bash /root/wt-c/research/spill-c-20260919/day74-box.sh` on a Ryzen 9
9950X machine with one RTX PRO 6000 Blackwell Workstation Edition and at least 56 GiB `MemFree` at the start (else
`not_run`), root in the container; the 285K class after it. Receipts land in `/root/spill-receipts/c-day74-<rig>/`.
Expected: the build about 5 minutes when not shared, the cell about 12 minutes (24 runs, each induced run adding its
setup of up to about 20 s).

## 2. BOX31 (a Ryzen 9 9950X, run by the lead as registered; `pro-single-day74-box31/`)

**Attempt 1 (03:56Z; `attempt1-notrun-memfree26/`):** `DAY74 INDUCER rig=box31 memfree_gib=26 F_gib=-44 induce=0`,
`DAY74 INDUCE VERDICT rig=box31 -> not_run`: the page cache held 77 GB (staged artifacts and build trees) and the
container refuses `drop_caches`. Recorded as it reads.

**Between the attempts (the lead's action, outside the cell):** the clean page cache of every file the cell does not
use was evicted with unprivileged `posix_fadvise(POSIX_FADV_DONTNEED)`, the 35B artifact kept cached: `MemFree` 26 to
57 GiB (`Cached` 77 to 46 GB). No host setting was changed.

**Attempt 2 (04:13Z to `box done 2026-09-26T04:31:00Z`):** 128 receipts in the two directories `OK` against their
manifests (re-checked), `run-gen-p71` by hash. Regime: 27 to 40 C, SM median 2610 MHz, N=1105. Verbatim:

- `DAY74 INDUCER rig=box31 memfree_gib=56 F_gib=16 induce=1`
- `DAY74 INDUCE CHECKS rig=box31 runs=24 integrity=ok`
- `DAY74 R2 rig=box31 ref: window_cycles median=6.010 (N=6) compacted=0 of 6 ... | R3 gen median=0.243 window median=0.217`
- `DAY74 R2 rig=box31 refi: window_cycles median=6.009 (N=6) compacted=0 of 6 ... | R3 gen median=0.252 window median=0.221`
- `DAY74 R2 rig=box31 i15: window_cycles median=6.011 (N=6) compacted=0 of 6 ... | R3 gen median=0.247 window median=0.218`
- `DAY74 R2 rig=box31 i15i: window_cycles median=6.010 (N=6) compacted=0 of 6 ... | R3 gen median=0.257 window median=0.224`
- `DAY74 INDUCE VERDICT rig=box31 integrity=ok -> not_induced (compaction in 0 of 6 REF+I and 0 of 6 door+I spans)`

**Read as registered: `not_induced`.** The inducer ran in all 12 induced runs (each `ready`, `loop` at the gate,
`stopped`), and every 256 MiB huge-page burst it asked for came back whole (`AnonHugePages` 262144 kB in all 24
per-second rows): with 16 GiB fragmented and about 48 GiB of free memory left untouched, the kernel had free
high-order blocks and never compacted. No span moved `compact_isolated`, and no run's core left 6.0 cycles per step.
Beside it, deciding nothing: the induced arms ran slower in wall time with the core unchanged (REF 0.243 to 0.252 s
gen-only, the door 0.247 to 0.257), the inducer's own load on the host (one core, memory traffic); one REF run
(`o2-ref-r2`) read 0.303 gen-only with a normal window and core.

## 3. BOX29 (the 285K class, run by the lead as registered; `pro-single-day74-box29-285k/`)

After DAY72's cell on the same card, 05:28Z to `box done 2026-09-26T05:35:57Z`; 128 receipts `OK`. Regime: 35 to 45 C,
SM median 2610 MHz, N=2080. Verbatim:

- `DAY74 INDUCER rig=box29-285k memfree_gib=105 F_gib=64 induce=1`
- `DAY74 INDUCE CHECKS rig=box29-285k runs=24 integrity=FAIL failed=the counters check did not pass (the cycle readings need it)`
- `DAY74 INDUCE VERDICT rig=box29-285k integrity=FAIL -> void`

**Read as registered: `void`.** The counters check reads `[cpu-probe] counters-check cpu=1 counters=unavailable`: the
probe's counters come from AMD's `RDPRU`, which an Intel CPU does not have, and section 1 reads the state only in
APERF cycles, with no fallback. That is a gap in the registration, not a fault of the host. What the 285K half needs
is registered in section 4 below, before any rerun. Beside it, deciding nothing: the inducer ran at 64 GiB and every
burst again came back as whole huge pages; no compaction counter moved anywhere in the cell; the phase probe's wall
time per chain step reads 1.112 to 1.118 ns in every run of every arm, and the induced arms ran slower in wall time
(REF 0.255 to 0.258 s gen-only, the door 0.264 to 0.270).

## 4. Registered after sections 2 and 3, before any rerun: the cell `induce-b`

Two gaps, each found by a reading, each closed before the next cell:
- **The inducer did not induce** on either host with free high-order blocks. `induce-b` fragments all but a small
  reserve of free memory: at the cell's start it maps `X` = `MemFree` - 2 GiB of anonymous memory with
  `MADV_NOHUGEPAGE`, touches every page, and returns every other 4 KiB page; free memory is then about 2 GiB of
  intact blocks plus `X`/2 in order-0 holes, so a huge-page request must compact once the 2 GiB are taken. The door's
  pool and allocations are 4 KiB-page allocations and fit in the holes. The induced arms run only when `X`/2 is at
  least 48 GiB (`MemFree` at least 98 GiB), else `not_run`. The fragmentation is set up once per induced run, as
  before, and the huge-page loop and the `compact_memory` trigger start at the gate, as section 1a has them.
- **The state reading needs `RDPRU`.** On a CPU where the counters check reads `counters=unavailable`, `induce-b`
  reads the state as the phase probe's wall nanoseconds per chain step at `gate` and at `window` (`compute_ns`), with
  the same strict rule (`door_slows`: every door+I run's `window` value above every door run's; `ref_slows` likewise),
  and the counters are not an integrity condition there. Where the check passes, cycles per step as section 1.
- **The page cache the fill reads may be reclaimed** once free memory is this tight (a huge-page fault may reclaim before
  it compacts). So before every run, of every arm, the cell reads the artifact once through the page cache (`cat` to
  `/dev/null`, timed into `ev/rewarm.tsv`), and the door's `physical_reads=0` stays an integrity condition.
Everything else is section 1's: the arms, the order, N, the `not_induced` guard (fewer than 5 of 6 induced spans of
either program with compaction), the verdict names. Receipts under `c-day74b-<rig>/`. It runs on a 9950X machine (the
registered class) and on the 285K class, each with at least 98 GiB `MemFree` at the start.

### 4a. `induce-b`'s sitting, prepared before any cell

`day74b-cell.sh` (section 1's cell with section 4's changes: `X` = `MemFree` - 2 GiB, the induced arms only when
`X`/2 is at least 48 GiB, the artifact reread before every run into `ev/rewarm.tsv`), `day74-read.py --induce-b` (the
wall-time state where the counters are unavailable), `day74b-box.sh` (receipts under `c-day74b-<rig>/`). The inducer
script is unchanged; its size argument is `X`. Dry checks (`day74-cpu/`): the cell's control flow
(`dry-check-cell-b.log`: 24 runs, 24 rewarm rows, each induced run's inducer ready, looping from the gate, stopped);
the reader's mechanics over BOX29's `induce` receipts (`dry-check-reader-b.log`: the wall-time state reads, and the
same receipts without `--induce-b` stay `void`; that output decides nothing, section 3 stands); the driver
(`dry-check-driver-b.log`).

Run as `D74_BUILDS="p71=6bad38150" [D74_RIG=<name>] bash /root/wt-c/research/spill-c-20260919/day74b-box.sh` on a Ryzen
9 9950X machine and on the 285K class, each with one RTX PRO 6000 Blackwell Workstation Edition, at least 98 GiB
`MemFree` at the start (else `not_run`; the lead's page-cache eviction of unused files, as on BOX31, is the way to get
there), root in the container. Expected: about 25 minutes (24 runs, 12 inducer setups of up to about 30 s at 100 GiB,
24 rereads of the artifact).
