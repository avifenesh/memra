# WP-C day 66 (2026-09-25): OWED C12, the door's slow boots on the 9950X class, the next reading registered before any cell

Lead, after DAY65's cell: "So the bimodal door on this host is not the L3 domain; register what the receipts point to
next before any cell (same rules as before)." Tree at start: `5c6e7c0cf` plus `DAY65.md` section 2.

## 0. What the receipts point to

On the Ryzen 9 9950X machine of BOX15 and BOX17, across 60 door boots of six door arms and two pins
(`DAY64.md` section 4, `DAY65.md` section 2):

- The door is slow in about 60 to 80% of boots and fast in the rest, per boot, at any position of the interleave and
  on either pin; REF (the legacy slot cache with `MEMRA_MOE_PREFETCH=1`) is 0.245 or 0.246 s gen-only in all 30 runs.
- In a slow boot every CPU-side part of the door is about 1.85 times larger (BOX15's stage clock: the owner's demand,
  the cache's lease retire, the bank's `stage`), the GPU time per copy is unchanged and the card's SM clock reads the
  same; the host fill before decode, a multi-threaded memory-bound pass that ends before decode, is 15 to 20% slower in
  the same boots (BOX17: 1278 to 1342 ms slow, 1114 to 1120 ms fast).
- The owner thread stays on CPU 0's L3 domain in slow and fast boots alike (DAY65 R3).

A cause that slows every CPU-side part of one boot by the same factor, and a memory-bound pass by less, while a
GPU-bound program on the same host is untouched, fits two readings the receipts cannot yet separate:

- **H1, the core clock.** The owner thread's core runs at a lower frequency in a slow boot (this host scales its
  frequency: `lscpu` reads `CPU min MHz: 600.0000`, `CPU max MHz: 5752.0000`, `CPU(s) scaling MHz: 27%` at idle), and
  the door's bursty CPU work, waiting on the card between layers, sits near a governor's ramp point.
- **H2, the page backing.** The process's anonymous memory (the catalog, the host cache index, the SLRU, the leases) is
  backed by huge pages in some boots and by 4 KiB pages in others, set by the host's memory state at each boot; the
  door's pointer-chasing lookups pay TLB walks in the 4 KiB boots, and REF's lighter CPU work does not.

## 1. Pre-registration: the cell `freq` (a 9950X-class host; before its script)

A measurement cell; it changes no code, no default and no host setting.

**Host and binaries** as `DAY65.md` section 1 (the same machine class, preferably the same machine; `c60=da649107c`
REF, `i15=2243b1fe2` the door). Every run starts under `taskset -c` of DAY65's ONE pin (the first 12 CPUs of CPU 0's
L3 domain), so placement is held fixed.

**Host settings, read before the arms** (`ev/topology.txt`): `lscpu`; per CPU the L3 domain; the cpufreq driver,
governor, energy-performance preference and boost state where sysfs exposes them; the transparent huge page
`enabled` and `defrag` modes. Each is recorded as read, or as absent.

**Arms.** REF and I15, order 1 (REF, I15) x 10, order 2 (I15, REF) x 10, 40 runs, N=20 per arm (more door boots, so both
modes appear more than once).

**Samplers, log only, stopped by their own pids when the arms end:**
- every 250 ms: each `run-gen` process's pid, the CPU it last ran on, and that CPU's current clock
  (`cpufreq/scaling_cur_freq`, or the CPU's `cpu MHz` line of `/proc/cpuinfo` where cpufreq is absent)
  (`ev/clock.tsv`);
- every 1 s: each `run-gen` process's `AnonHugePages` (`/proc/<pid>/smaps_rollup`) and its voluntary and involuntary
  context switches (`/proc/<pid>/status`) (`ev/memory.tsv`).

**Readings** (`day66-read.py`):
- R1: every run's gen-only seconds; a door run is slow iff it exceeds REF's median by more than 0.030 s (DAY65's rule).
- R2: each door run's owner-core clock: the median of its `clock.tsv` samples over the whole run, and over its last
  3.0 s before its end mark (the decode and the window with the teardown).
- R3: each door run's largest `AnonHugePages` and its last sampled context-switch counts.

**The verdict** (`DAY66 FREQ VERDICT`):
- `not_reproduced` if fewer than two slow or fewer than two fast door runs (the modes cannot be separated);
- otherwise two findings side by side: `clock_tracks_mode` iff every slow run's whole-run owner-clock median is below
  every fast run's, else `clock_does_not_track`; and `thp_tracks_mode` iff the largest `AnonHugePages` of every slow run
  lies on one side of every fast run's (all below or all above), else `thp_does_not_track`.

**What it decides.** Which of H1 and H2, or neither, the slow boots follow. A remedy (for example, a fixed-clock arm
where the host permits it, or huge-page backing for the door's metadata) would be its own registration after this
reads, with its own cells.

## 1a. The sitting, prepared before any cell

`day66-cell.sh` (cell `freq`; the day-40 runner body unchanged; the host settings and DAY65's ONE pin read from sysfs;
the two samplers, each stopped by its own pid) and `day66-read.py` were written after section 1. The samplers' reads
were checked on the local host against a stand-in `run-gen` process (`scaling_cur_freq` in kHz, `AnonHugePages`, the
context-switch counts). The reader was dry-checked for mechanics on DAY65's receipts relabelled with synthetic clock
and memory files and printed its lines end to end; its verdict there means nothing. The driver `day66-box.sh` (builds
`c60` and `i15`; no runner pin; each run pinned by the cell) is dry-checked for control flow
(`day66-cpu/dry-check-driver.log`). Run as
`D66_BUILDS="c60=da649107c i15=2243b1fe2" bash /root/wt-c/research/spill-c-20260919/day66-box.sh` on the Ryzen 9 9950X
machine of BOX15 and BOX17 where it can be had (else another host of that class), one RTX PRO 6000 Blackwell
Workstation Edition, the approved 35B artifact at `/root/artifacts/`, `/root/wt-c` at the lane tip and a detached
`/root/wt-c-build`, CUDA 13 and Rust, at least 48 GB host `MemAvailable`. Expected: two builds about 10 minutes, the
cell about 9 (40 runs).

## 2. The cell, as it ran (BOX18, BOX15's machine, run by the lead as registered; `pro-single-day66/`)

The lead ran `day66-box.sh` as section 1a names it on the tree `52544aed9`, on BOX18 (BOX15's machine again; idle
26.6 W, FMA spin 2658 MHz, no brake), 09:58Z to `box done 2026-09-25T10:09:14Z`; binaries built on the box. The lead
mirrored the receipts; read here: 191 of 191 files `OK` against `box-mirror-manifest.sha256` (`MIRROR-CHECK.txt`).
Regime (`regime.txt`): 26 to 42 C, SM median 2610 MHz. Verbatim (`freq/reading.log`):

- `DAY66 HOST rig=pro-single cpu0 cpufreq/scaling_driver=amd-pstate-epp | cpu0 cpufreq/scaling_governor=powersave | cpu0 cpufreq/energy_performance_preference=performance | ...` (the full line in the reading; `thp/enabled=always [madvise] never`)
- `DAY66 FREQ CHECKS rig=pro-single runs=40 integrity=ok`
- `DAY66 R1 ref_median=0.245 slow=14 fast=6 of 20 door runs`
- the owner core's whole-run clock median 5715 to 5723 MHz in every door run, slow and fast; `anon_huge_kb=0` in every
  run; voluntary context switches 25808 to 28282 per door run, involuntary 41 to 168, in both modes (the per-run lines
  in the reading)
- `DAY66 FREQ VERDICT rig=pro-single integrity=ok -> clock_does_not_track thp_does_not_track`

**Read as registered.** Neither H1 (the core clock) nor H2 (huge-page backing: this host's THP mode is `madvise`, so the
door's anonymous memory never had huge pages in any boot) follows the slow boots. The fill before decode still splits
by mode (1337 to 1490 ms in most slow boots, 1178 to 1217 in the fast ones, with two slow boots at 1178.6 and 1181.2),
and the owner thread's most-sampled CPU was 1 or 4 in slow and fast boots alike (a post-hoc table, deciding nothing).
Two of REF's 20 runs read slow gen-only (0.314 and 0.319 s) with a normal window (0.218): the first REF runs slower than
0.246 in 50 on this machine. This cell also sampled `/proc/<pid>/smaps_rollup` every second, which walks the process's
page tables; that sampler is a possible cause of those two REF readings, so the next cell does not carry it. The next
registration is `DAY67.md`.
