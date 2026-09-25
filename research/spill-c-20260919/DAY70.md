# WP-C day 70 (2026-09-25): OWED C12, who shares the owner's core in a slow boot, registered before any cell

Lead, after DAY68's cell: "Record DAY68, then register what `gate_tracks` points to next before any cell (same rules;
no bound moves)." Tree at start: `1a06db505` plus `DAY68.md` section 2.

## 0. What `gate_tracks` points to (`DAY68.md` section 2)

On BOX15's Ryzen 9 9950X machine the door's slow state is in place at `gate`, before the timed decode, and absent at
`start`; it ends by itself about a second after the window; a register-only chain runs about 1.45 times slower in
wall time while `/proc/cpuinfo` and `scaling_cur_freq` both read the full clock and the temperature is normal; REF
never enters it. Two mechanisms fit that and the receipts cannot separate them:

- **H1, time on the owner's CPU.** Another thread is runnable on the owner thread's CPU (a thread of the process, or a
  kernel thread the door's install sets working: page pinning, compaction, the driver), so the owner waits on the
  run queue part of the time; the chain's wall time includes the wait.
- **H2, the core's other hardware thread.** Something runs on the owner's SMT sibling, sharing the core's pipeline.
  With the ONE pin (`0-7,16-19`) the owner's sibling may lie inside or outside the pin (CPU 1's sibling 17 is inside,
  CPU 4's sibling 20 is not), and slow runs had the owner on CPU 4 in 5 of 9, so a sibling co-runner would be outside
  the process's pin: another process or a kernel thread.

## 1. Pre-registration: the cell `sched` (BOX15's machine preferred; before its script)

A measurement cell; no code, no default, no host setting changes. The binary is `p68=48c098374` again (its phase probes
give each run's `gate`, `generate`, `warm` and `window` instants in the stamped log). REF and the door (I15), every run
under DAY65's ONE pin with `--cpu-probe` and `--cpu-probe-phases`, order 1 (REF, I15) x 10, order 2 (I15, REF) x 10, 40
runs, the day-18 pressure shape, one collector hold.

**One sampler** (`day70-sampler.py`, one Python process started by the cell and stopped by its own pid), pinned to one
CPU that is neither in the ONE pin nor a hardware-thread sibling of a CPU in it (read from each CPU's
`topology/thread_siblings_list`; recorded in `ev/pins.txt`), so it is not itself a co-runner. It writes
`ev/sched.tsv`:
- every 250 ms: every CPU's `/proc/stat` counters (`cpu` lines); and for each `run-gen` process, its main thread's
  `/proc/<pid>/task/<pid>/schedstat` (time on the CPU, time waiting on a run queue, time slices) and its last CPU;
- every 1 s: every thread on the host (`/proc/*/task/*/stat`) whose user or system time moved since the last pass, with
  its pid, tid, name, last CPU and the move (only the movers are written, to bound the file).

**Readings** (`day70-read.py`), each over the span from a door run's `gate` line to its `window` line (the decode and the
steady window), and for fast runs the same span:
- R1: the slow or fast mark (a door run slower than REF's median by more than 0.030 s gen-only).
- R2: the owner thread's run-queue wait as a share of the span (from its `schedstat` deltas).
- R3: the owner CPU's busy time not spent by the owner thread, as a share of the span (the CPU's `/proc/stat` busy
  delta minus the owner's own on-CPU delta), and its SMT sibling's busy share.
- R4: the three threads, anywhere on the host, that moved most in the 1-second passes that overlap the span, with their
  names and CPUs (the owner's own thread excluded).

**The verdict** (`DAY70 SCHED VERDICT`, the strict rule of `DAY67.md`, with a count per field beside it):
- `not_reproduced` if fewer than two slow or fewer than two fast door runs;
- otherwise `rqwait_tracks` (R2 larger in every slow run), `owncpu_tracks` (R3's own-CPU other share larger in every
  slow run), `sibling_tracks` (R3's sibling share larger in every slow run); the ones that track, or `none_tracks`.
R4's list is printed for each run beside it, deciding nothing.

**What it decides.** Whether the owner waits for its CPU (H1), shares its core with a busy sibling (H2), or neither, and
who the co-runner is. It changes no code; a remedy is its own registration after this reads.

## 1a. The sitting, prepared before any cell

`day70-sampler.py`, `day70-cell.sh` and `day70-read.py` were written after section 1. The sampler was checked on the
local host against a busy stand-in `run-gen` process for 3.2 s (312 CPU rows, 13 owner rows with its `schedstat`
moving, 62 moving-thread rows; the stand-in's own thread moving 99 to 100 ticks a second). The cell's choice of the
sampler's CPU on the local host (24 CPUs, no SMT siblings) is CPU 23; on a 9950X with the ONE pin `0-7,16-19` it is
CPU 31, whose sibling 15 lies outside the pin too. The reader, one detail stated here: a door run counts only if the
owner's samples bracket its gate-to-window span inside the run's marks, else the cell is void on integrity. It was
dry-checked for mechanics on DAY68's receipts with a synthetic `sched.tsv` (its verdict there means nothing). The
driver `day70-box.sh` (one build, `p68`, the binary of DAY68) is dry-checked for control flow
(`day70-cpu/dry-check-driver.log`). Run as
`D70_BUILDS="p68=48c098374" bash /root/wt-c/research/spill-c-20260919/day70-box.sh` on BOX15's Ryzen 9 9950X machine
where it can be had (else another host of that class), one RTX PRO 6000 Blackwell Workstation Edition, the approved
35B artifact, `/root/wt-c` at the lane tip and a detached `/root/wt-c-build`, CUDA 13, Rust and Python 3, at least 48 GB
host `MemAvailable`. Expected: one build about 5 minutes, the cell about 10 (40 runs).

## 2. The cell, as it ran (BOX23, BOX15's machine, run by the lead as registered; `pro-single-day70/`)

The lead ran `day70-box.sh` as section 1a names it on the tree `527776aee`, on BOX23 (BOX15's machine again), 11:58Z to
`box done 2026-09-25T12:07:52Z`, and mirrored the receipts; read here: 189 of 189 files `OK` against
`box-mirror-manifest.sha256` (`MIRROR-CHECK.txt`). Regime (`regime.txt`, the card over the hold): 25 to 41 C, SM
median 2610 MHz, N=1563. Verbatim (`sched/reading.log`):

- `DAY70 PINS rig=pro-single home_l3=0-7,16-23 one=0,1,2,3,4,5,6,7,16,17,18,19 sampler_cpu=31`
- `DAY70 SCHED CHECKS rig=pro-single runs=40 integrity=ok`
- `DAY70 R1 ref_median=0.245 slow=11 fast=9 of 20 door runs`
- `DAY70 COUNT rqwait_tracks: 0 of 11 slow runs beyond the fast runs' extreme (deciding nothing)`
- `DAY70 COUNT owncpu_tracks: 0 of 11 slow runs beyond the fast runs' extreme (deciding nothing)`
- `DAY70 COUNT sibling_tracks: 0 of 11 slow runs beyond the fast runs' extreme (deciding nothing)`
- `DAY70 SCHED VERDICT rig=pro-single integrity=ok -> none_tracks`
- two of the 20 per-run lines, a slow one and a fast one:
  `DAY70 R2R3R4 o1-i15-r3: gen=0.311 slow gate_ns=1.519 span_ms=1000 owner_cpu=1 sibling=[17] rqwait=0.000 owncpu_other=0.000 sibling_busy=0.0 | top: python3[5911/5911] cpu=3 ticks=5, nvidia-smi[4169/4169] cpu=1 ticks=1, python3[4365/4365] cpu=31 ticks=1`,
  `DAY70 R2R3R4 o1-i15-r4: gen=0.248 fast gate_ns=1.050 span_ms=1000 owner_cpu=1 sibling=[17] rqwait=0.000 owncpu_other=0.000 sibling_busy=0.0 | top: python3[6671/6671] cpu=4 ticks=5, python3[4365/4365] cpu=31 ticks=1`

**Read as registered: `none_tracks`.** Over every door run's gate-to-window span the owner thread waited on no run
queue (`rqwait=0.000` in all 20), its CPU spent at most 0.029 of the span on anything else (0.000 in 16 of 20), and its
SMT sibling was idle (0 ticks in all 20), slow or fast. The slow state is not a co-runner the scheduler sees, on the
owner's CPU or on its core's other hardware thread. H1 and H2 of section 0, as the scheduler accounts them, do not hold.

**What the receipts show beside the verdict, deciding nothing** (read from `ev/sched.tsv` and the run logs after the
verdict; nothing here moves a bound):
- The owner thread's own time on its CPU over the span (its `schedstat` run delta over the span) is 0.998 to 1.002 of
  the span in all 20 door runs, in 3 to 14 time slices, slow or fast.
- R4's large movers are the host fill's workers, not co-runners: in 6 runs (4 slow, 2 fast) three `run-gen-p68`
  threads other than the owner move 99 to 100 ticks in one 1-second pass, and every such pass is stamped 0.26 to 0.53 s
  before the run's gate and before its fill's end line: the reader's pass window (the span widened by 1 s each side)
  takes in the end of the fill. Inside the spans the host's movers are the collector's Python and `nvidia-smi`, at
  most 5 ticks a pass.
- What this sampler cannot see. This kernel accounts no hard-interrupt time: the `irq` column of `/proc/stat` is 0 on
  all 32 CPUs after 41 days up (`softirq` is counted). Hard-interrupt time is then charged to the task it interrupts:
  it sits inside the owner's `schedstat` run time and its CPU's busy ticks, where R2 and R3 cannot separate it. Time
  in system management mode and a stopped core clock are outside the scheduler's accounting too.
- A correction to `DAY67.md` section 2's reading of `dram_ns`. That probe builds its 256 MiB permutation after the L2
  chase and before the timed DRAM chase, so the DRAM chase runs about a second after the compute chain (DAY67's probe
  line prints 1.56 s after the window line in its `o1-i15-r1`). In 18 of DAY67's 20 door runs `compute2_ns` reads 1.049
  or 1.050 (a fast state, or a slow state already over), and their `dram_ns` (88.9 to 110.8) mostly read the fast
  state. In the
  two runs whose slow state lasted through the DRAM chase (`o1-i15-r1`, `compute2_ns=1.209`, and `o2-i15-r1`,
  `compute2_ns=1.464`) `dram_ns` reads 150.435 and 150.266, against 87.8 to 92.6 in REF's 20 runs: about 1.65 times.
  So DAY67's "DRAM latency unchanged, which is what a lower effective core clock gives" does not hold: in the slow state
  a DRAM-latency chain slows at least as much as the core-bound ones. With `DAY68.md`'s reported clock unchanged at
  about 5.72 GHz (a measured-looking value that moves by a few MHz between samples; on x86 with the `aperfmperf` flag
  the kernel derives it from the APERF/MPERF ratio), the receipts now point at time taken from the owner's core that
  neither the scheduler nor that ratio shows: hard interrupts, system management mode, or a stopped clock that stops
  APERF and MPERF together, rather than a lower clock. None of these is read yet; they are candidates, not findings.

What the receipts point to next, and why one cell: C12 has taken six single-hypothesis cells on this one host (L3
domain, requested clock and THP, memory against core, reported clock and temperature at phases, scheduler waits and
co-runners). The remaining candidates can all be read at once from counters that cost the owner nothing: the core's own
TSC, APERF and MPERF around the probe's chain (read on the owner thread with `RDPRU`, which this CPU has), every CPU's
interrupt and softirq counts, the idle states of the owner's sibling, the package's energy counter and sensors, and the
card's PCIe traffic. `DAY71.md` registers that one cell, on BOX15's machine and on a second Ryzen 9 9950X machine, so
the same reading says whether the slow state belongs to this host or to the class.
