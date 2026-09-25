# WP-C day 71 (2026-09-25): OWED C12, what takes the owner's core in a slow boot, registered before any cell

Lead, after DAY70's cell: "Record DAY70 and register what the receipts point to next before any cell (same rules; no
bound moves). Given how many single-hypothesis cells C12 has taken on this one host, consider whether the next
registration should cover several remaining candidates in one cell, and whether the question also needs a second 9950X
machine to say whether it is this host or the class; that choice is yours under the rules." Tree at start: `527776aee`
plus `DAY70.md` section 2.

## 0. What `none_tracks` leaves (`DAY70.md` section 2)

On BOX15's Ryzen 9 9950X machine, in a slow door boot, a register-only chain, an L1 chase and an L2 chase all run about
1.4 times slower, and a DRAM chase about 1.65 times slower (the two DAY67 boots whose slow state lasted through it). The
owner thread is on its CPU for the whole span (0.998 to 1.002), waits on no run queue, and nothing else the scheduler
accounts runs on its CPU or on its SMT sibling. The reported clock (`/proc/cpuinfo`, from the APERF/MPERF ratio on this
CPU) stays at about 5.72 GHz and the temperature is normal. REF never enters the state; about half the door boots do,
from somewhere between the engine's start and the gate, until about a second after the window.

What fits all of that, and what the receipts so far cannot see:
- **K-irq, hard interrupts on the owner's CPU.** This kernel accounts no hard-interrupt time (the `irq` column is 0 on
  every CPU), so an interrupt load on the owner's CPU sits inside its own run time and its CPU's busy ticks. The
  interrupt counts in `/proc/interrupts` see it; the handlers' cache and TLB damage would slow the DRAM chase more than
  the others.
- **K-stop, a stopped or gated clock.** If the core's clock is stopped part of the time (a throttle that halts APERF and
  MPERF together), their ratio, and so the reported clock, stays full while wall time is lost. MPERF against the TSC
  over the chain sees it (below 1 while the thread runs).
- **K-smm, system management mode.** Time in firmware is invisible to the OS; depending on whether the counters freeze
  in it, it reads like K-stop or like lost cycles, with no interrupt count behind it.
- **K-ipc, more cycles per instruction at a full clock.** APERF per chain step (6 in the fast state: three
  shift-and-xor pairs) sees it; with no interrupt count and MPERF against the TSC at 1, what is left is the core itself
  (the sibling's idle state keeping the core in two-thread mode, or a power or current limit acting on dispatch).
- **K-limit, the package's limits.** The package's energy counter and sensors over the span say whether the socket is
  near a power, current or temperature limit when the state is present.
- **K-dma, the card reading host memory.** The card's PCIe receive and transmit rates over the span say whether a DMA
  stream shares the memory the DRAM chase measures.

And the class question: every C12 cell ran on one physical machine. A second Ryzen 9 9950X machine with the same card
says whether the slow state belongs to this host or to the class.

## 1. Pre-registration: the cell `core` (on two machines; before its script and before the probe change)

A measurement cell; no default, no host setting changes. One engine change, log only: the probe reads the core's own
counters.

**The probe change (binary `p71`, the lane tip after it).** A new `run-gen` flag `--cpu-probe-counters` (log only, with
`--cpu-probe-phases`): each phase probe reads the calling thread's TSC (`RDTSC`) and its CPU's MPERF and APERF
(`RDPRU` with ECX 0 and 1) and the monotonic clock before and after its 2^20-step chain, and appends to its phase line
`cpu_after=<cpu> wall_ns=<d> tsc=<d> mperf=<d> aperf=<d>` (deltas). On a CPU that does not report `RDPRU` (CPUID
Fn8000_0008 EBX bit 4) or reports fewer than two readable registers (EDX bits 23:16 below 1), it appends
`counters=unavailable` and runs nothing else. A second flag `--cpu-probe-counters-check` prints one such reading as
`[cpu-probe] counters-check ...` and exits before any engine work, so the cell can check that the instruction runs on
the host before it passes the flag to a run. Both flags carry a decide-by of 2026-10-09 in `MOE-SLOT-CACHE-DOOR.md`
(the diagnostic flags' row), and go with the probe when C12 closes.

**The runs.** The binary `p71`; REF and the door (I15), every run under DAY65's ONE pin with `--cpu-probe`,
`--cpu-probe-phases` and (when the check passes) `--cpu-probe-counters`; order 1 (REF, I15) x 10, order 2 (I15, REF)
x 10, 40 runs, the day-18 pressure shape, one collector hold. The check runs once before the first run, under the same
pin; its output is kept (`ev/counters-check.txt`).

**One sampler** (`day71-sampler.py`, DAY70's sampler extended, one Python process started by the cell and stopped by its
own pid, pinned as DAY70's to a CPU outside the pin and its siblings). Every 250 ms: every CPU's `/proc/stat` line;
each `run-gen` main thread's `schedstat`, last CPU, minor and major faults, user and system ticks; the per-CPU
deltas of every `/proc/interrupts` row and every `/proc/softirqs` row that moved (only the moved rows and CPUs are
written); every CPU's `cpuidle` state times and usage deltas (only the moved ones); every `/proc/vmstat` counter that
moved (its delta); every powercap zone's `energy_uj` and every hwmon temperature and power input present. Every 1 s:
DAY70's moving threads. The cell also runs `nvidia-smi dmon -s t -d 1 -o T` beside it (the card's PCIe receive and
transmit MB/s each second), pinned to the sampler's CPU. These reads are of kernel counters, except that the kernel
reads the package's energy register on a CPU of the package, which can be an interrupt to that CPU every 250 ms
(counted in the interrupt rows like any other).

**Readings** (`day71-read.py`), each door run:
- R1: slow or fast (gen-only slower than REF's median by more than 0.030 s), as before.
- R2, the core's counters at `gate` (the phase where the slow state was in place in 9 of 9 slow boots in DAY68), from
  the gate probe's deltas, read only when `cpu_after` equals `cpu`: `delivered_ghz` = APERF / wall ns,
  `ref_share` = MPERF / TSC, `counted_ghz` = APERF / MPERF x (TSC / wall ns), `cycles_per_step` = APERF / 2^20.
  REF's 20 gate readings are printed beside them as the fast state's reference.
- R3, over the span from the run's `gate` line to its `window` line: `irq_rate` (all hard-interrupt rows' counts on
  the owner's CPU per second), `softirq_rate` (the same for softirqs), `sibling_poll` (the owner's SMT sibling's time in
  its `POLL` idle state as a share of the span), `pkg_w` (the package zone's energy over the span per second, when a
  zone exists), `pcie_rx` (the median of the card's receive MB/s over the span's seconds), `owner_sys` (the owner's
  system ticks as a share of its ticks over the span).
- R4, printed and deciding nothing: the three interrupt rows with the most counts on the owner's CPU over the span,
  the five `vmstat` counters that moved most, the hottest sensor, the card's transmit rate, and DAY70's top movers
  restricted to passes stamped inside the span.

**The verdict per machine** (`DAY71 CORE VERDICT rig=<rig>`, the strict rule of `DAY67.md`, a count per field
beside it):
- `void` if integrity fails (40 runs, exit 0, argmax MATCH, 32 generated and 32 window tokens, the phase lines, the
  door's fill and zero physical reads, one tape, one host demand sequence, the sampler's rows bracketing every door
  run's span);
- `not_reproduced` if fewer than two slow or fewer than two fast door runs;
- otherwise the fields that track, where a field tracks when every slow run's value is beyond every fast run's in its
  registered direction: lower for `delivered_ghz`, `ref_share`, `counted_ghz`; higher for `cycles_per_step`,
  `irq_rate`, `softirq_rate`, `sibling_poll`, `pkg_w`, `pcie_rx`, `owner_sys`; or `none_tracks`. A field not read in
  every door run (the counters unavailable, a migrated gate probe, no powercap zone) prints `not read` and decides
  nothing.

**What the fields decide, stated before the cell.** `ref_share` tracking: the core's clock is stopped part of the time
(K-stop, or K-smm with frozen counters). `irq_rate` or `softirq_rate` tracking: interrupts (K-irq), and R4 names the
source. `cycles_per_step` tracking with neither: cycles lost inside the core at a full clock (K-ipc, or K-smm with
counting counters), and `sibling_poll` or `pkg_w` tracking says which. `delivered_ghz` and `counted_ghz` tracking
together: a lower clock after all, and DAY68's reported clock was stale. `pcie_rx` tracking: the card's DMA shares the
window. It changes no code; a remedy is its own registration after this reads.

**The class verdict** (`DAY71 CLASS VERDICT`, from the two machines' readings, BOX15's machine as `a`, the second as
`b`): `class` if `b` has two or more slow door runs; `this_host` if `b` has none and `a` reproduces (two or more slow
and two or more fast); `undecided` otherwise (one slow run on `b`, or `a` does not reproduce). `b` is any Ryzen 9 9950X
machine other than BOX15's with one RTX PRO 6000 Blackwell Workstation Edition, the approved artifact, the same tree
and the same driver script. Each machine's reading stands on its own; the class line needs both.

## 1a. The sitting, prepared before any cell

The probe change landed as `6bad38150` (binary `p71`), after section 1: `cpu_probe.rs` keeps DAY67's compute chain as
it ran and adds a copy of the chain for the counted path, bracketed by the monotonic clock, `RDTSC` and `RDPRU` (ECX 0
then 1) on each side, with the chain's seed and result passed through `black_box` so it runs between the two readings
(the release disassembly shows clock, `rdtsc`, `rdpru`, `rdpru`, the loop, clock, `rdtsc`, `rdpru`, `rdpru` in that
order). On the local host (an Intel CPU, no `RDPRU`) the check prints `[cpu-probe] counters-check cpu=5
counters=unavailable`, and the unit test covers that path and the field format; the `RDPRU` path first runs on the
target machine, behind the cell's check.

`day71-sampler.py`, `day71-cell.sh`, `day71-read.py` and `day71-box.sh` were written after section 1. Dry checks, all
in `day71-cpu/`:
- the sampler on the local host for 3.4 s against a busy stand-in `run-gen` process: every row kind written (`C`, `O`,
  `I`/`IH`, `S`/`SH`, `D`/`DH`, `V`, `EH`, `H`, `T`); the local powercap zones read `unreadable` (not root), which the
  reader turns into `pkg_w` not read;
- the cell's control flow (`dry-check-cell.sh`, `dry-check-cell.log`): a stub `run-gen-p71` and a stub `nvidia-smi` in
  a sandbox; the check's `rdpru=ok` passes `--cpu-probe-counters` to all 40 runs, the sampler and `dmon` start pinned
  and are stopped by their own pids, nothing outlives the cell;
- the reader's mechanics (`make-synthetic.py`): DAY70's receipts with invented counter, interrupt, idle, vmstat, energy,
  sensor and dmon rows give every line, the counts and a verdict (meaningless); DAY70's own receipts, which lack the
  counters files, read `void`; the class mode reads two cells and prints its line;
- the driver's control flow (`dry-check-driver.sh`, `dry-check-driver.log`): one build, the cell, the validation and
  the reader per machine label, each machine into its own receipts root, a rerun skipping the done cell.

Run as `D71_BUILDS="p71=6bad38150" bash /root/wt-c/research/spill-c-20260919/day71-box.sh` on BOX15's machine, and as
`D71_BUILDS="p71=6bad38150" D71_RIG=pro-single-b bash /root/wt-c/research/spill-c-20260919/day71-box.sh` on the second
Ryzen 9 9950X machine; each: one RTX PRO 6000 Blackwell Workstation Edition, the approved 35B artifact, `/root/wt-c` at
the lane tip and a detached `/root/wt-c-build`, CUDA 13, Rust and Python 3, at least 48 GB host `MemAvailable`, root in
the container (the powercap energy files are root-only). Receipts land in `/root/spill-receipts/c-day71-<rig>/`.
Expected per machine: one build about 5 minutes, the cell about 10 (40 runs). The class line comes from the two mirrored
cells: `python3 day71-read.py --class pro-single-day71/core pro-single-day71-b/core`. If only BOX15's machine can be
had, its reading stands on its own and the class line stays owed.

## 2. Machine `b` (BOX24, a second Ryzen 9 9950X machine, run by the lead as registered; `pro-single-day71-b/`)

BOX15's machine was not on the market, so the lead ran the `b` half first: BOX24 (one RTX PRO 6000 Blackwell
Workstation Edition at 600 W, a Ryzen 9 9950X with two L3 domains, 60 GB RAM with 58 GB available, driver 595.58.03,
kernel 5.15, `acpi-cpufreq` with the `ondemand` governor; idle 14.8 W at 180 MHz, P8, FMA spin 2850 to 2852 MHz, no
brake), `D71_BUILDS="p71=6bad38150" D71_RIG=pro-single-b bash .../day71-box.sh` on the tree `d819faea7`, 13:08Z to
`box done 2026-09-25T13:20:30Z`. The lead mirrored the receipts; read here: 194 of 194 files `OK` against
`box-mirror-manifest.sha256` (`MIRROR-CHECK.txt`). Regime (`regime.txt`): 25 to 39 C, SM median 2610 MHz, N=2258; the
card's link reads PCIe gen 5 x8 in 2253 of 2258 samples (BOX15's machine read gen 5 x16 in DAY70's cell). Verbatim
(`core/reading.log`, the box's reading):

- `DAY71 PINS rig=pro-single-b home_l3=0-7,16-23 one=0,1,2,3,4,5,6,7,16,17,18,19 sampler_cpu=31`
- `DAY71 COUNTERS rig=pro-single-b on check: [cpu-probe] counters-check cpu=19 rdpru=ok cpu_after=19 wall_ns=1097927 tsc=4712156 mperf=4712155 aperf=6291632 | rc=0`
- `DAY71 CORE CHECKS rig=pro-single-b runs=40 integrity=ok`
- `DAY71 REF gate delivered_ghz: min=5.612 median=5.699 max=5.727 (N=19 of 20)`
- `DAY71 REF gate ref_share: min=1.000 median=1.000 max=1.000 (N=19 of 20)`
- `DAY71 REF gate counted_ghz: min=5.612 median=5.699 max=5.727 (N=19 of 20)`
- `DAY71 REF gate cycles_per_step: min=6.000 median=6.000 max=6.746 (N=19 of 20)`
- the fast door run and one slow one:
  `DAY71 R2R3 o1-i15-r1: gen=0.327 fast gate_ns=1.047 span_ms=1250 owner_cpu=1 sibling=[17] delivered_ghz=5.732 ref_share=1.000 counted_ghz=5.732 cycles_per_step=6.000 irq_rate=0.000 softirq_rate=80.000 sibling_poll=0.000 pkg_w=not_read pcie_rx=1172.000 owner_sys=0.008`,
  `DAY71 R2R3 o1-i15-r2: gen=0.467 slow gate_ns=2.421 span_ms=1251 owner_cpu=0 sibling=[16] delivered_ghz=5.722 ref_share=1.000 counted_ghz=5.722 cycles_per_step=13.852 irq_rate=0.000 softirq_rate=91.926 sibling_poll=0.000 pkg_w=not_read pcie_rx=2856.500 owner_sys=0.008`
- `DAY71 R1 rig=pro-single-b ref_median=0.323 slow=19 fast=1 of 20 door runs`
- `DAY71 CORE VERDICT rig=pro-single-b integrity=ok -> not_reproduced`

**Read as registered: `not_reproduced` on `b`** (one fast door run, fewer than two). No field decides on this
machine. The class line needs BOX15's machine too and stays owed. For when it reads: under section 1's rule, `b`'s 19
slow door runs give `class`.

**A reader defect, found reading these receipts.** In this container `/proc/interrupts` reads empty (no `IH` or `I`
row in `ev/sched.tsv`, and the `cat /proc/interrupts` of `host-before.txt` printed nothing). The reader summed the
absent rows as zero and printed `irq_rate=0.000`, where section 1 says a field not read prints `not read`. The verdict
is not affected (`not_reproduced` decides before any field), but the `irq_rate` values in this reading are not
measurements. Section 3 fixes the reader before BOX15's half runs, and the corrected reading of these receipts sits
beside the box's (`core/reading-rev2.log`).

**REF's own median, read before any conclusion** (the lead asked). REF's gen-only is 0.322 to 0.323 s in 18 of its 20
runs here, against 0.245 on BOX15's machine. REF's core reads normal: 6.0 to 6.1 cycles per step at every phase from its
third run on (one gate probe migrated between CPUs and is not read), at 5.61 to 5.73 GHz at the gate. Its cache behaves
the same as on BOX15's machine (`o1-ref-r3` prints the same cache lines on both: hit rate 79.1 percent cumulative, 90.4
percent and 43.5 MB per token in the steady window). What differs is the machine: the card's link is gen 5 x8 here
against x16, and the host has 60 GB against 197. The decode that streams experts over the link is slower, and the core
is not. So `b`'s REF-relative slow mark and BOX15's are on different baselines; the door's gate probe is the
within-machine measure of the state, and at the gate it agrees with the mark in all 20 door runs (the fast run's state
starts later, at `warm`).

**What the receipts show beside the verdict, deciding nothing** (read from `ev/` after the verdict):
- **The core state.** Every slow door run reads 12.4 to 14.2 APERF cycles per chain step at `gate` (6.0 in the fast
  run and in REF). The delivered clock is 5.67 to 5.74 GHz and MPERF over TSC is 1.000 in every door and REF gate
  reading. The chain is 2.2 times slower in wall time than the fast state (2.17 to 2.48 ns per step against 1.047) with
  the core in C0 at its full clock the whole time. So on this machine the lost time is cycles the core spends
  while its clock runs: not a stopped clock (K-stop), not a lower clock. It is interrupt work on that CPU, which APERF
  counts (K-irq), system management mode with counting counters, or a loss inside the core (K-ipc). The sibling's
  POLL share is 0.000 in every run, the owner's system share at most 0.027, softirqs 38 to 110 a second. The one fast
  door run (`o1-i15-r1`) reads 6.0 at `gate` and `generate` and 13.7 from `warm` on: its state began inside its
  decode, as `o2-i15-r1` did in DAY68.
- **Memory compaction runs against the door's process, and only the door's.** Over each door run's gate-to-window span,
  `/proc/vmstat` moves `compact_isolated` by 32,134 to 93,553 and `pgmigrate_fail` by nearly the same (every isolated
  page fails to migrate), with `pgmigrate_success` at most 292. Over the REF spans from REF's third run on, both are 0.
  Placed by phase in four door runs (`o1-i15-r1`, `r2`, `r5`, `o2-i15-r5`), the failures run at 4,700 to 6,200 a
  second from the run's start to its fill's end, 29,000 to 54,000 from the fill's end to the gate, 35,000 to 81,000
  over the span, and 50,000 to 60,000 from the window to the run's end mark; successful migrations appear only in that
  last stretch (42,000 to 168,000 a second), as the process frees its memory. REF's first two runs are the exception
  that fits: over their spans compaction migrated 381,440 and 324,044 pages successfully, and their later phases read
  7.4 to 8.4 cycles per step. `compact_stall` never moves (no direct compaction); the kcompactd counters move little
  (`compact_daemon_migrate_scanned` 52,270 over the cell against `compact_migrate_scanned` 46,512,361). Moving a page
  that a process maps means unmapping it there, and each unmap flushes that mapping from the TLB of every CPU the
  process runs on, the owner's among them, by an interrupt. That is a candidate mechanism for K-irq, not a reading:
  this container hides the interrupt counts.
- **The mover rows see the container only.** No kernel thread appears in any `T` row across the cell (the movers are
  `run-gen-p71`, `python3`, `nvidia-smi`, `bash`, `ssh`), so the sampler's `/proc` is the container's PID namespace.
  DAY70's R4 read the container's threads, not the host's: its "the host's movers" (`DAY70.md` section 2) means the
  container's, and kcompactd, kswapd or another tenant would not have appeared there.
- The package energy files are absent in this container (`pkg_w` not read). This kernel accounts no hard-interrupt time
  either (the `irq` column of `/proc/stat` is 0 in every row).

## 3. Registered after machine `b`, before BOX15's half runs: the reader defect fixed, two fields added

**The fix.** `irq_rate` reads `not read` when the cell has no `/proc/interrupts` rows at all, and `softirq_rate` when
it has no `/proc/softirqs` rows, as section 1 already says for a source that is not there.

**Two fields for BOX15's half, registered now, on `b`'s receipts only as a post-hoc reading.**
- `intr_rate`: the host-wide interrupt total over the span, per second, from the first number of `/proc/stat`'s
  `intr` line. On x86 it includes the CPUs' interrupt-controller interrupts (timer, rescheduling, function-call and TLB
  shootdown), so it still reads when a container hides `/proc/interrupts`. It is the whole host's, not the owner CPU's.
  The sampler adds one `N` row per pass (`intr` total and `ctxt`); `b`'s receipts have none, so there it is `not read`.
- `migrate_fail`: the span's `pgmigrate_fail` delta per second, from the sampler's `V` rows.
Both are higher-is-tracking under the strict rule of section 1, joining the ten registered fields. They decide on
BOX15's cell; nothing else in section 1 moves. BOX15's cell runs the same binary `p71=6bad38150`; the sampler's added
row and the reader change come with the tree the sitting checks out.
