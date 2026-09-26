# WP-C day 86 (2026-09-26): OWED C12 and C11, whether the door's recent cuts make C12's slow state more frequent on a long-running 9950X host, before any cell

`DAY84.md` section 3a: on BOX31, a 9950X that had carried other sittings for 19 h, every door arm of cell `i21` was
bimodal by boot and REF was not; the slow boots rose with the arm in both orders (I15 3, I20 4, I21 7, I21C 9 of 10).
Ten runs per arm do not separate the arm from the host's drift, and the cell did not sample compaction. This day
registers the cell that asks it. Tree at start: the DAY84 reading's commit (`2bebd4900`).

## 0. What is known

- C12's slow state is compaction isolating the door's pinned pool pages and failing to migrate them (`DAY76.md`,
  `DAY78.md`: the pool is shared `/dev/zero` memory; `DAY80.md`: a private anonymous pool pinned with
  `cuMemHostRegister`, `--expert-bank-pool-registered`, cleared it on the 285K class with the door's timing flat on both
  classes). Making the registered pool the door's default is the owner's question (`DAY80.md` section 4a).
- A fresh 9950X host shows no slow boot (BOX40 in DAY82, BOX38 in DAY80); a long-running one does (BOX15 in DAY64,
  BOX31 now). The state needs the host's memory to be fragmented enough that the kernel compacts during a run.
- I21 and I22 add a few vectors at install (I21: two `u32` views and a dense position table, about 0.4 to 0.6 MB,
  without or with the MTP layer's row index; I22: a lease-handle view, about 0.25 MB), and I21 turns the trace's slot
  map from a tree into a vector that grows by reallocation. Nothing else in the door's memory changes.

## 1. Pre-registration: the cell `slow86`

**Host.** A Ryzen 9 9950X with one RTX PRO 6000 Blackwell Workstation Edition that has carried at least 12 hours of
other sittings since boot (the lead names its uptime and prior work in the mirror note). A fresh host reads
`not_reproduced` below and decides nothing.

**Binaries** (`day63-box-build.sh`): `i20=8efea3a54`, `i21=b555b4141`, `i22` (DAY85's, named in `DAY85.md` once it
lands; if I22 is not landed when the cell runs, its two arms are dropped and the rule reads the rest).

**Arms.** The DAY82 cell's argv: `ref` (`run-gen-i21`, `MEMRA_MOE_PREFETCH=1`, no door), `i20`, `i21`, `i22` (the door),
`i22r` (`run-gen-i22` with `--expert-bank-pool-registered`). Order 1 (ref, i20, i21, i22, i22r) x 5, order 2 reversed
x 5: 50 runs, one collector hold. Every run's boundary snapshot adds `/proc/vmstat`'s `compact_*`, `pgmigrate_*` and
`thp_fault_*` counters (before and after), and `/proc/buddyinfo`, to the card and memory lines the cells keep.

**Reader** (`day86-read.py`, written after this section). Integrity: every run exits 0 with `MATCH`, one tape, one host
demand sequence across the door arms. Then per run: gen-only seconds, the compaction deltas (`compact_isolated`,
`compact_migrate_scanned`, `compact_fail`, `pgmigrate_fail`), and the slow mark (`DAY80.md` section 3's: gen-only above
REF's median by more than 0.030 s). Per arm: slow boots of 10, in each order.

**The rule, fixed now.**
- `not_reproduced` when the default-pool door arms (i20, i21, i22) have fewer than 3 slow boots together: the host is
  not in the state, nothing is read.
- Otherwise the arm question, by a one-sided Fisher exact test on slow boots of 10 (the reader computes it):
  `i21_more_slow` when I21 has more slow boots than I20 with p < 0.05; `i21_not_shown` otherwise. The same for I22
  against I21, printed as `i22_more_slow` or `i22_not_shown`.
- The control: `registered_clears` when `i22r` has no slow boot, `registered_does_not` when it has 2 or more, else
  `undecided`.
- Beside it, deciding nothing: slow boots against each run's compaction deltas (whether every slow boot has
  `pgmigrate_fail` above the fast boots' largest), and the arms' slow counts by position in the round.

**What each verdict does.** `i21_more_slow` reverts I21 with its receipt (a higher slow-state rate on this class is a
regression of the default program), and I22 is re-registered on I20; `i22_more_slow` reverts I22 the same way. Either
`not_shown` keeps the change. The control's verdict adds to the owner's question in `DAY80.md` section 4a and changes
no default.

## 1a. The sitting, prepared before any cell

`day86-cell.sh`, `day86-read.py` and `day86-box.sh` were written after section 1. The reader on a synthetic cell from
DAY84's BOX31 receipts relabelled, with invented vmstat sections (`day86-cpu/make-synthetic.py`; meaningless): 50 runs,
integrity ok, every line printed, the Fisher tests checked against known values (9 of 10 against 1 of 10: p 0.0005; 7
against 3: 0.089) (`dry-check-reader.log`). The cell under stubs, with and without `run-gen-i22`: 50 and 30 runs in the
registered order, the registered flags per arm, every snapshot with its vmstat and buddyinfo sections
(`dry-check-cell.log`). The driver under stubs: the builds, the cell, `--validate`, the reader; a rerun skips the cell
(`dry-check-driver.log`).

Run as `D86_BUILDS="i20=8efea3a54 i21=b555b4141 i22=4b378a064" bash
/root/wt-c/research/spill-c-20260919/day86-box.sh` on a Ryzen 9 9950X host with one RTX PRO 6000 Blackwell
Workstation Edition that has carried at least 12 hours of other sittings since boot (at least 48 GB MemAvailable, the
artifact and the two worktrees staged as before; about 25 minutes: three builds and 50 runs).

## 1b. The host condition, stated exactly before any cell (the lead asked)

"At least 12 hours of other sittings since boot" means: measured from the host's boot (the host's uptime, not the
container's), at least 12 hours during which other sittings ran on that host, meaning any lane's cells or batteries
that load the model and pin host memory. That is BOX31's history in `DAY84.md` section 3a (19 hours of lane A sittings),
the condition the slow state has needed. Host uptime alone does not count (an idle host does not fragment its memory
the same way), and no minimum of timed GPU hours is required. The lead's plan meets it: BOX43's host has been up 130029
s at its container's start (21:36Z), and the lanes' 9950X-class sittings run on it from then until at least
2026-09-27T09:40Z, before `slow86`. The mirror note names the host uptime and the sittings it carried. The rule is
unchanged: a host that reads `not_reproduced` decides nothing.
