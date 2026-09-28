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

## 1c. Whether BOX43 qualifies, and the load that fills it (the lead asked; before any cell)

The lead's record for BOX43 since its container started (21:36Z, host uptime 130029 s then): eight sittings that load
a model and pin host memory, 21:43Z to 05:52Z (this lane's DAY85 half, integ69's GPU run 5, integ70's and integ71's
GPU batteries, lane B's DAY46, DAY50 stage 0, DAY46C and DAY48), about 6.5 h of sittings inside 8.3 h, idle since
05:52Z.

- **The count.** Section 1b's 12 hours are hours in which sittings ran (their running time, as BOX31's 19 hours of
  lane A sittings were), not the span of the container or the host's uptime; the host's history before the container
  is unknown and not counted. BOX43 has about 6.5 h.
- **Idle is not neutral.** While the host idles, the kernel's proactive compaction rebuilds free high-order pages,
  which is the opposite of the state the cell needs. So `slow86` runs right after the twelfth hour of sittings, with no
  idle gap longer than one hour before it. BOX43 idle from 05:52Z to 09:40Z does not qualify.
- **What counts.** Any lane's cell or battery that loads a model and pins host memory (every sitting on the lead's list
  counts). When nothing else is queued on the host, the fill is the door's own load: `day86-load.sh`, door runs of
  `run-gen-i22` (the 35B and the door's 16 GiB pinned pool, the cells' argv) back to back for `D86_LOAD_HOURS`, the
  GPU lock taken per run so another lane's sitting can take the card between runs, each run's gen-only seconds and
  compaction counters logged (`runs.tsv`), the full log kept for a failed run and every 50th. It decides nothing; its
  table also shows when, if ever, the host's door runs turn slow. Dry-checked under stubs (`day86-cpu/dry-check-load.log`).
- **For BOX43 now:** the queued sittings first, then `D86_LOAD_HOURS` set to what the 12 hours still lack (about 5.5
  if nothing else runs), then `slow86` right after the load ends.

**Correction to the count (the lead's, before any cell).** The eight sittings sum to 306 minutes (17, 37, 36, 47, 6, 52,
97 and 14), 5.1 h, not the 6.5 h first written above. The load fills the rest: `D86_LOAD_HOURS=6.9 bash day86-load.sh`
started 2026-09-27T06:29Z on the tree `ce026510b` (host uptime 161850 s), 37 minutes after the last sitting ended (inside
the one-hour gap), and the chain starts `slow86` right after the load ends (about 13:25Z) with no gap. A sitting that
runs between the load's runs (lane B's DAY44 rerun, if it comes) counts toward the 12 hours too.

## 2. The cell `slow86` on BOX43 (run by the lead as registered; `pro-single-day86-9950x/`, `pro-single-day86-load/`)

BOX43: a Ryzen 9 9950X with one RTX PRO 6000 WS, 184 GB, driver 580.65.06. The host condition as section 1c set it:
306 minutes of sittings before the load; the load (`D86_LOAD_HOURS=6.9`) 06:29Z to 13:23Z, with integ71's GPU fix run
(07:06Z to 08:19Z), integ72's GPU run 1 (09:43Z to 10:23Z) and lane B's DAY48 rerun (12:41Z to 12:56Z) between its
runs; `slow86` from 13:23:29Z (host uptime 186697 s), no gap, to 13:38:50Z (`lead-extra/c13-chain.out`). The chain tree
`ce026510b`; the builds name `tree=8efea3a54...`, `tree=b555b4141...` and `tree=4b378a064...`, each `rc=0`. Receipts:
229 and 1190 files `OK` against the lead's box manifests (`LEAD-MANIFEST.sha256`, re-checked here), ELFs by hash.
Regime: 38 to 48 C, SM median 2850 MHz, N=627 busy samples of 3140. Verbatim (`slow86/reading.log`):

- `DAY86 SLOW CHECKS rig=pro-single runs=50 arms=ref,i20,i21,i22,i22r integrity=ok`
- `DAY86 SLOW rig=pro-single ref_median=0.319 mark=0.349 ref=1/10 (o1 0, o2 1) i20=1/10 (o1 0, o2 1) i21=1/10 (o1 0,
  o2 1) i22=0/10 (o1 0, o2 0) i22r=0/10 (o1 0, o2 0)`
- `DAY86 VERDICT rig=pro-single integrity=ok -> not_reproduced (default-pool door arms slow=2)`

**Read as registered: `not_reproduced`; it decides nothing.** Whether I21 or I22 makes C12's slow state more frequent
stays open.

**What the three marked runs are, deciding nothing.** They are not C12's state. They are the last round of order 2
(`o2-i21-r5` 2.598 s, `o2-i20-r5` 2.581 s, `o2-ref-r5` 3.740 s, 13:35Z to 13:38Z), ten times the host's 0.319 s rather
than 50 ms behind it, and REF is among them, which C12's state never touched. The load's table shows the same event
once an hour: 15 of its 1184 door runs (0 failed, median 0.321 s; the driver's count of 1185 includes one lock retry) ran 0.376 to 4.888 s, in bursts at 06:34, 08:31 to
08:33, 09:34, 10:31 to 10:32, 11:31 to 11:33 and 12:34 to 12:36, some with about two million page migrations failing
inside the run and some with none. So BOX43 carries an hourly host-wide stall at about half past the hour, and in about
12 hours of sittings and 1234 door runs it never showed C12's signature (the door arms bimodal by about 50 ms over 32
tokens, REF steady, as DAY84 read on BOX31).

**Registered now, before any further cell: where the question goes next.** The next 9950X-class sitting of any lane
whose cell reads C12's signature (door arms bimodal by about 50 ms over 32 tokens with REF steady, `DAY84.md` section
3a's pattern) runs `slow86` on that same host before the host is released, with no load needed (the host is in the
state). Until such a host appears, the question stays open and nothing is reverted; the registered pool, which cleared
the state on the 285K class, stays the owner's question (`DAY80.md` section 4a).
