# WP-C day 66 (2026-09-25): OWED C12, the door's slow boots on the 9950X class, the next reading registered before any cell

Lead, after DAY65's cell: "So the bimodal door on this host is not the L3 domain; register what the receipts point to
next before any cell (same rules as before)." Tree at start: `5c6e7c0cf` plus `DAY65.md` section 2.

## 0. What the receipts point to

On the Ryzen 9 9950X machine of BOX15 and BOX17, across 50 door boots of five door arms and two pins
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
