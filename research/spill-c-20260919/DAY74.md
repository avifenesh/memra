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
