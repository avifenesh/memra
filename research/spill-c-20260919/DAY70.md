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
