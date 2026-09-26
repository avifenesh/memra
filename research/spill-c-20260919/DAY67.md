# WP-C day 67 (2026-09-25): OWED C12, a per-boot CPU and memory probe inside the process, registered before code

Lead, after DAY66's cell: "So neither H1 nor H2 explains the bimodal door on this host. Register what the receipts point
to next before any cell, same rules as before." Tree at start: `52544aed9` plus `DAY66.md` section 2.

## 0. What is left, from the receipts (`DAY64.md` section 4, `DAY65.md` section 2, `DAY66.md` section 2)

On BOX15's Ryzen 9 9950X machine, the door's slow boots slow its pure CPU work (the owner's demand, the bank's
`stage`, the cache's lease retire: hash lookups and bookkeeping, no CUDA calls in the first two) by about 1.85 times,
at the same core (CPU 1 or 4 of one L3 domain), the same reported clock (about 5720 MHz), no huge pages in any boot,
the same GPU copy time; the multi-threaded fill before decode is 15 to 20% slower in the same boots; the mode holds
through the whole process (gen-only and the later window agree). What the receipts do not have is a direct reading of
how fast that thread's core and memory are, per boot. A same-clock, same-core slowdown of pure CPU work can come from
the core itself doing less per cycle (a busy SMT sibling, an effective clock below the reported one) or from memory
(the process's physical pages and their cache behaviour), and the two predict different things for a fixed probe:
- the core: a register-only compute loop and an L1-resident chase are both slower in slow boots;
- memory: they are not, and a DRAM-sized chase (or an L2-sized one) is.

## 1. Pre-registration: `run-gen --cpu-probe` (log only; before its code)

**The instrument.** `run-gen --cpu-probe` runs, on the main thread after every timed phase (after the steady window's
line and its stats, before the text output), a fixed probe and prints one line:
`[cpu-probe] cpu=<sched_getcpu before> cpu_after=<after> compute_ns=<per iteration> l1_ns=<per load> l2_ns=<per load>
dram_ns=<per load> compute2_ns=<the compute loop again, last>`:
- compute: 2^26 dependent xorshift64 steps;
- l1: 2^24 dependent loads walking a 4 KiB cyclic permutation (L1-resident);
- l2: 2^22 dependent loads over a 512 KiB cyclic permutation;
- dram: 2^21 dependent loads over a 256 MiB random cyclic permutation;
each timed with `Instant`, its result kept (`black_box`). The probe allocates its arrays itself and frees them before
returning; without the flag nothing runs. Its code is a small module with a CPU unit test at reduced sizes. Decide-by
2026-10-04, recorded in `MOE-SLOT-CACHE-DOOR.md` (a CLI diagnostic's decide-by lives in its design doc).

**The cell `probe`** (a 9950X-class host, BOX15's machine preferred; `day67-cell.sh`, reader `day67-read.py`, both
written after the instrument's code and before any cell). Binaries: REF and the door from one new label, `p67`, the
commit that adds the probe (so REF and the door carry the same probe code; REF is `MEMRA_MOE_PREFETCH=1` on it, the
door `--experts-via-tier --expert-bank-host-bytes=17179869184` on it; their programs are those of `c60`'s REF and of
I15, the probe added after the timing). Every run under `taskset -c` of DAY65's ONE pin, `--cpu-probe` on every run.
Order 1 (REF, I15) x 10, order 2 (I15, REF) x 10, 40 runs. The 250 ms placement and clock sampler of `DAY66.md`; every
1 s the main thread's `/proc/<pid>/sched` fields `se.nr_migrations`, `nr_switches`, `nr_voluntary_switches` (no
`smaps_rollup`). The day-18 pressure shape and one collector hold, as before.

**Readings** (`day67-read.py`):
- R1: every run's gen-only seconds; a door run is slow iff it exceeds REF's median by more than 0.030 s.
- R2: every run's probe line (both arms), with the run's slow or fast mark.
- R3: each door run's largest `se.nr_migrations`.

**The verdict** (`DAY67 PROBE VERDICT`):
- `not_reproduced` if fewer than two slow or fewer than two fast door runs;
- otherwise, for each of `compute_ns`, `l1_ns`, `l2_ns`, `dram_ns` and the largest `se.nr_migrations`, `<name>_tracks`
  iff every slow door run's value is larger than every fast door run's, and the verdict lists the ones that track, or
  `none_tracks`.
Beside it, REF's per-run probe spread (does a REF boot's core or memory read slow while REF's timing does not?).

**What it decides.** Whether the slow boots are the core or memory, or neither. It changes no default and no program;
the probe runs after all timing.

## 1a. The instrument as built, and the sitting, before any cell

`run-gen --cpu-probe` landed in `ab3d8f18e` (`cpu_probe.rs`, the `run_gen.rs` call after the steady window's lines;
engine lib 569, clippy `-D warnings`, fmt): at the registered sizes it reads, on the local CPU (by hand,
`the_registered_probe_on_this_host`): `compute_ns=1.265 l1_ns=0.771 l2_ns=2.630 dram_ns=117.609 compute2_ns=1.157`,
1.1 s in all. A first build walked the 256 MiB permutation once before the timed loads (9.2 s); that warm pass was
replaced by nothing (the permutation's own construction writes every slot, so its pages are mapped), before any
cell; the registered sizes stand. The label `p67=ab3d8f18e`: the door's engine is I15's (`2243b1fe2`) and REF's
program is the legacy slot cache's with its prefetch, unchanged since `c60`; only the probe is added. `day67-cell.sh`
and `day67-read.py` were written after section 1; the reader was dry-checked for mechanics on DAY66's receipts with
synthetic probe lines appended (its verdict there means nothing); the driver `day67-box.sh` (one build, `p67`) is
dry-checked for control flow (`day67-cpu/dry-check-driver.log`). Run as
`D67_BUILDS="p67=ab3d8f18e" bash /root/wt-c/research/spill-c-20260919/day67-box.sh` on BOX15's Ryzen 9 9950X machine
where it can be had (else another host of that class), one RTX PRO 6000 Blackwell Workstation Edition, the approved
35B artifact, `/root/wt-c` at the lane tip and a detached `/root/wt-c-build`, CUDA 13 and Rust, at least 48 GB host
`MemAvailable`. Expected: one build about 5 minutes, the cell about 10 (40 runs, each with a 1 s probe after its timing).

## 2. The cell, as it ran (BOX19, BOX15's machine, run by the lead as registered; `pro-single-day67/`)

The lead ran `day67-box.sh` as section 1a names it on the tree `3e9bf6bda`, on BOX19 (BOX15's machine; it read 239.5 W
and 2872 MHz with no app while settling after the previous tenant, then 27.6 W at 180 MHz, FMA spin 2630 to 2656 MHz,
no brake), to `box done 2026-09-25T10:59:36Z`. The lead mirrored the receipts; read here: 190 of 190 files `OK`
against `box-mirror-manifest.sha256` (`MIRROR-CHECK.txt`). Regime (`regime.txt`): 27 to 42 C, SM median 2610 MHz.
Verbatim (`probe/reading.log`):

- `DAY67 PROBE CHECKS rig=pro-single runs=40 integrity=ok`
- `DAY67 R1 ref_median=0.245 slow=10 fast=10 of 20 door runs`
- `DAY67 REF compute_ns: min=1.049 median=1.050 max=1.050 (N=20)`, `l1_ns: min=0.874 median=0.875 max=0.876`,
  `l2_ns: min=2.470 median=2.782 max=2.938`, `dram_ns: min=87.817 median=89.453 max=92.576`
- the door's per-run probe lines (all 20 in the reading), e.g. `o1-i15-r1: gen=0.314 slow compute_ns=1.449 l1_ns=1.207 l2_ns=3.910 dram_ns=150.435 compute2_ns=1.209`,
  `o1-i15-r2: gen=0.248 fast compute_ns=1.050 l1_ns=0.874 l2_ns=2.780 dram_ns=93.059 compute2_ns=1.049`,
  `o2-i15-r1: gen=0.248 fast compute_ns=1.461 l1_ns=1.204 l2_ns=4.004 dram_ns=150.266 compute2_ns=1.464`
- `DAY67 PROBE VERDICT rig=pro-single integrity=ok -> none_tracks`

**The reader checked against the rule, on these receipts.** The registered rule is strict: `<name>_tracks` iff every
slow door run's value is larger than every fast door run's. The reader implements exactly that (`min` over the slow
runs above `max` over the fast ones). One fast boot, `o2-i15-r1` (gen-only 0.248 s, a fast decode, fill 1113.5 ms),
carries slow probe values (`compute_ns=1.461`, `l1_ns=1.204`, `l2_ns=4.004`), above the smallest slow run's
(`compute_ns=1.449`, `l1_ns=1.202`, `l2_ns=3.910`), so no field separates by the rule. This is not a reader defect; the
verdict stands as `none_tracks`, and no bound moves.

**What the lines show beside the verdict, deciding nothing.**
- In 19 of the 20 door boots the probe's first three fields split by mode: the 10 slow boots read `compute_ns` 1.449 to
  1.487, `l1_ns` 1.202 to 1.273, `l2_ns` 3.910 to 4.092; nine fast boots read 1.049 to 1.050, 0.874 to 0.875, 2.472 to
  2.785, as REF's 20 runs do. `dram_ns` does not split (88.9 to 150.4 across both modes). A same-factor slowdown of a
  register chain, an L1 chase and an L2 chase (each counted in core cycles) with DRAM latency (counted in
  nanoseconds) unchanged is what a lower effective core clock gives: about 1.4 times, near 4.1 GHz for a core the
  cpufreq sysfs reported at about 5.72 GHz in DAY66's cell.
  **Corrected in `DAY70.md` section 2:** the DRAM chase runs about a second after the compute chain, when the slow
  state is over in most slow boots; in the two boots whose slow state lasted through it (`o1-i15-r1`, `o2-i15-r1`)
  `dram_ns` reads 150.4 and 150.3 against REF's 87.8 to 92.6, so DRAM latency is not unchanged in the slow state and
  this bullet's "lower effective core clock" reading does not follow. The next bullet's "about 0.4 s later" is also
  wrong: the probe line prints 1.56 s after the window line in `o1-i15-r1`, so `compute2_ns` runs about 1.5 s after
  the first chain.
- `compute2_ns`, the same chain about 0.4 s later (after the DRAM chase), reads 1.049 or 1.050 in 9 of the 10 slow
  boots and 1.209 in the tenth (`o1-i15-r1`); in the fast boot `o2-i15-r1` it stays 1.464. The slow state ends during the probe in most slow
  boots: it is a transient of the core, not a property fixed for the process. It also appeared after a fast decode
  once (`o2-i15-r1`).
- The host fill before decode splits exactly by the decode's mode here: 1113.5 to 1128.0 ms in the 10 fast boots, 1298.6
  to 1344.9 ms in the 10 slow ones. The slow state is already present during the fill in the slow boots.
- What this says about DAY66's clock sampler: it read `cpufreq/scaling_cur_freq`, which under `amd-pstate-epp` is the
  frequency the driver requested, not the one the core ran at. DAY66's `clock_does_not_track` is a finding about the
  requested clock; it does not exclude an effective clock below it, which this probe points to.

The next registration (`DAY68.md`) reads the effective clock and the CPU's temperature across the run's phases.
