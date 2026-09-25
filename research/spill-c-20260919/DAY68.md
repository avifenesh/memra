# WP-C day 68 (2026-09-25): OWED C12, the effective core clock and the CPU temperature across the run, registered before code

Lead, after DAY67's cell: "note what compute2_ns says about the slow state ending during the probe (it bears on DAY66's
clock sampler, which read the reported frequency)." Tree at start: `2e046ef8b` (main `5228ff0cd`, #726, merged in)
plus `DAY67.md` section 2.

## 0. What the receipts point to (`DAY67.md` section 2)

In 19 of 20 door boots on BOX15's Ryzen 9 9950X machine, the probe after the timing read the core about 1.4 times
slower in the slow boots on a register chain, an L1 and an L2 chase (all counted in core cycles), with DRAM latency
unchanged: a lower effective core clock, near 4.1 GHz against the 5.72 GHz the cpufreq sysfs reported. The slow state
was already present during the fill before decode (the fill split exactly by mode), lasted through decode and the
window, and ended within about 0.4 s of the probe's start in 9 of 10 slow boots; once it appeared after a fast decode.
REF's 20 runs never showed it. Two things are not read yet: the clock the core actually ran at, sampled across the run,
and what drives it. A CPU of this class lowers its boost clock with temperature, current and power limits, on the
time scale of seconds, and the door, unlike REF, runs a six-thread, memory-bound fill of 15 GB just before its decode.

## 1. Pre-registration (before any code)

**The instrument, `run-gen --cpu-probe-phases`** (log only; decide-by 2026-10-04 in `MOE-SLOT-CACHE-DOOR.md`): a
2^20-step compute chain of `cpu_probe` (about 1 ms at 5.7 GHz) run on the main thread at the start of `main` (phase
`start`) and at each of the stage-line points (`gate`, before the timed generation; `generate`, after it; `warm`,
before the steady window; `window`, after it), each printed as `[cpu-probe] phase=<p> cpu=<cpu> compute_ns=<per step>`.
Every point is outside a timed span. Without the flag nothing runs. The day-67 `--cpu-probe` after the window stays.

**The cell `eclock`** (BOX15's machine preferred, else a Ryzen 9 9950X-class host; `day68-cell.sh`, reader
`day68-read.py`, both written after the instrument's code and before any cell). One binary, label `p68` (the commit
that adds the phase probe; REF and the door programs as in `p67`), REF and the door (I15), every run under DAY65's ONE
pin with `--cpu-probe` and `--cpu-probe-phases`, order 1 (REF, I15) x 10, order 2 (I15, REF) x 10, 40 runs, the day-18
pressure shape, one collector hold. Samplers, log only, stopped by their own pids:
- every 250 ms (`ev/eclock.tsv`): the owner process's CPU and that CPU's `cpu MHz` line of `/proc/cpuinfo` (on x86 Linux of this
  era computed from APERF and MPERF, the clock the core ran at, where `scaling_cur_freq` under `amd-pstate-epp` is the
  requested one), with `scaling_cur_freq` beside it; and every readable CPU temperature of the `k10temp` hwmon (`Tctl`, `Tccd*`), or
  `absent`;
- every 1 s (`ev/sched.tsv`): the day-67 `/proc/<pid>/sched` counts.

**Readings** (`day68-read.py`):
- R1: every run's gen-only seconds and its slow or fast mark (a door run slower than REF's median by more than 0.030 s).
- R2: every run's phase probe values (`start`, `gate`, `generate`, `warm`, `window`) and its day-67 probe line.
- R3: each door run's effective clock (the median `cpu MHz` of its owner CPU over its samples) and its hottest `Tctl`.

**The verdict** (`DAY68 ECLOCK VERDICT`):
- `not_reproduced` if fewer than two slow or fewer than two fast door runs;
- otherwise, with the strict rule of `DAY67.md` (a field tracks iff every slow door run lies beyond every fast one):
  `generate_tracks` (the phase probe at `generate` larger in every slow run), `gate_tracks` (the same at `gate`, the
  decode's start), `eclock_tracks` (the effective clock lower in every slow run), `tctl_tracks` (the hottest `Tctl`
  higher in every slow run); the tracking ones listed, or `none_tracks`.
Beside the strict verdict, a count per field (how many slow runs lie beyond the fast runs' extreme), deciding nothing.

**What it decides.** Whether the decode itself runs in the slow state, whether the effective clock reads it, and
whether the CPU's temperature follows it. It changes no default and no program; a remedy (the fill's thread count or
its placement against the decode, for example) is its own registration after this reads.

## 1a. The instrument as built, and the sitting, before any cell

`run-gen --cpu-probe-phases` landed in `48c098374` (`cpu_probe::phase_line`, a 2^20-step chain; the `start` point runs
right after the CUDA engine is created, before the model loads, the other four inside the stage-line closure; engine
lib 572, clippy `-D warnings`, fmt). The label `p68=48c098374` carries main `5228ff0cd` (#726, lane A's contracts
code, which neither program under test reaches) beside the door of I15 and REF's legacy program. `day68-cell.sh` and
`day68-read.py` were written after section 1. The samplers' reads were checked on the local host (`/proc/cpuinfo`
`cpu MHz` read 1849.7 against `scaling_cur_freq` 3046554 kHz on the same CPU in one sample, the two clocks the cell
tells apart; the local host's sensors are `coretemp`, so the cell reads `coretemp` where `k10temp` is absent). The
reader was dry-checked for mechanics on DAY67's receipts with synthetic phase lines and clock samples (its verdict
there means nothing). The driver `day68-box.sh` (one build, `p68`) is dry-checked for control flow
(`day68-cpu/dry-check-driver.log`). Run as
`D68_BUILDS="p68=48c098374" bash /root/wt-c/research/spill-c-20260919/day68-box.sh` on BOX15's Ryzen 9 9950X machine
where it can be had (else another host of that class), one RTX PRO 6000 Blackwell Workstation Edition, the approved
35B artifact, `/root/wt-c` at the lane tip and a detached `/root/wt-c-build`, CUDA 13 and Rust, at least 48 GB host
`MemAvailable`. Expected: one build about 5 minutes, the cell about 10 (40 runs).
