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
