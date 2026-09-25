# WP-A day 44: OWED item 3's owed half, design T on a 9950X-class host

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`. Ruling 54 ("T is the door's fill program"; the lead's order:
"the 9950X-class fill reading for T"). Item 3 closed for the slower-CPU host class on BOX7 (DAY39 section 7); the
9950X-class reading is owed. Every cell `executed-not-qualified`.

## 1. Pre-registration (committed before any cell runs)

**The host class.** A host whose `lscpu` model name names a Ryzen 9 9950X-class part (Zen 5 desktop, 16 cores, 32
threads) with one RTX PRO 6000 Blackwell. `pro-single-t9950/build.sh` checks it first; any other host writes
`HOST-NOT-9950X-CLASS` with its model name, builds and runs nothing, and the reading stays owed. On such a host
`fill_threads_for_host(32)` is 12 (`min(12, cpus / 2)`), the same T as BOX7's 92-CPU quota gave.

**The cells, DAY39's, unchanged** (sections 1, 3, 5, 5a and 5b), on the arms' tree `b4816eda8` (the lane before design
S2's code, its crates equal to main `5d653e851`, which carries T): `ft` (as built, design T), `f1` (plus
`rtx5090-day39/f1-g4.patch`, design F at one fill thread), `hk` (plus `rtx5090-day39/hk-revert-tip.patch`, the day-32
helper fill with design K), all from one clone with the tree checked back at the tip after each (both patches apply on
`b4816eda8`, checked). Sitting `pro-single-t9950/` (receipts under `/root/spill-receipts/a-t9950`):

1. `fill-survey.sh`: the probe `day39-fill-survey --gpu` (the 27B's and the 9B's staging shapes at T = 1, 2, 4, 8, 12,
   bitwise, the span copies on this card). **Section 1's rule, applied to this host's reading and nothing else**: T
   alone if some T at or below the host's physical cores fills the 27B's 156.9 MB in at most 13.1 ms minus the measured
   27B span-copy time minus 1.0 ms; O with T if none does (then O is pre-registered next, and the cells below are read
   but decide nothing about O).
2. `item3.sh`: DAY39 section 5's target cell, 40 boots (hk ft f1 off, o1 and o2, `--n 5`), read by
   `day39-reading.py <root> target`: (a) ft's steady polls `[1]` on at least 80 of its 90; (b) in both orders
   `median(HK) - median(FT)` above the order's pair noise for E2E and for PIN; readings: F1 against FT, every ON arm's
   `on_minus_off`.
3. (c): the gate set on ft's binary (identity default and plain, OFF and ON; failure OFF and ON; the fault gate default
   and plain; the twin OFF and ON; the hit gate OFF and ON) with the arms' tree's `tools/` (the tip's fault gate carries
   design S2's cells, which a pre-S2 binary does not know), and ft's unit cells (the engine's native cells in one process
   and serially, the door's GPU cells, the CPU suites).

**Decision** (DAY39 section 5's, per host class): (a), (b) and (c) pass: T is kept for the 9950X class and item 3
closes. (a) fails: T is refuted on this host class as it reads; O with T is pre-registered next. (a) passes and (b)
fails: recorded as it reads; T kept (the 5090's (e) passed: T is not worse anywhere); F against the helper fill on this
class goes to the lead as an owner item.

**Predictions.** The survey reads T=1 near or inside the budget (a desktop Zen 5 core copies faster than BOX7's EPYC
core, 14.28 GB/s there); T=12 well inside it; (a) holds; (b) holds with a smaller margin than BOX7's +12.6 ms if T=1
already fits (the day-35 5090 shape).

**Budget.** 0.2 agent-day to prepare (this commit); about 2.5 hours of card time.

## 2. The 9950X-class host, as it ran (`pro-single-t9950/box/`, the S2 sitting's box)

- Host (`host-shape.txt`): `Model name: AMD Ryzen 9 9950X3D2 16-Core Processor` (the X3D2 part of the class; recorded as
  it reads), 16 cores, 2 threads per core, 32 CPUs (`available_parallelism=30` in the probe), 124 GB; one RTX PRO 6000
  Blackwell Workstation Edition. The build's host-class check passed.
- Build attempt 1 (in `run-all.sh`) refused the hk patch (`rc=2 (hk patch)`: `git apply -3` merges through the index,
  which the arms' crate diff had not been applied to; the build script fixed in `9b3c4e82c`, attempt 1's logs in
  `build-attempt1/`); attempt 2 (in `run-all-2.sh`) `rc=0`: ft `167743d0c3e7ecc4..`, f1 `0521e322ee66a1cc..`, hk
  `6bd4267508077dcd..`. Mirrored 738 of 738 files against the box's manifest, mismatched 0.
- **The survey and section 1's rule** (`fill/survey.log`, every run bitwise): 27B `threads=1 .. median=6.429`, `threads=2
  .. 5.310`, `threads=4 .. 4.757`, `threads=8 .. 4.793`, `threads=12 .. 4.591 .. gbps=34.18`; `SPANS shape=27B .. median=3.045`;
  9B `threads=1 .. 1.069`, `threads=12 .. 0.679`. The budget is 13.1 - 3.045 - 1.0 = 9.06 ms: every T fits, T=1
  included (BOX7's T=1 read 10.985). **T alone is picked for this host class.**
- **(a) and (b), verbatim** (`item3/reading-day39-target.log`, 40 boots): `DAY39 T CLAUSE (a) ft steady promotes N=90
  polls==1 90 rule N=90 and >=80 -> PASS`; `DAY39 T CLAUSE (b) order=o1 metric=e2e hk=114.39 ft=102.16 hk-minus-ft=+12.23
  pair-noise=0.64 .. -> CLEARS`, `metric=pin hk=25.70 ft=13.50 hk-minus-ft=+12.20 pair-noise=0.10 .. -> CLEARS`, `order=o2
  metric=e2e .. hk-minus-ft=+12.18 pair-noise=0.23 .. -> CLEARS`, `metric=pin .. +12.20 .. -> CLEARS`; **`DAY39 T TARGET
  (a) and (b) -> PASS`**. Readings: F1 against FT `f1-minus-ft=-0.00` and `-0.11` ms e2e, `+0.00` PIN (T costs nothing
  and gains nothing where one thread already fits); `on_minus_off` -3.0 to -3.2 ms for ft and f1, +9.1 to +9.2 for hk.
- **(c)**: the gates on ft's binary with the arms' tree's tools, each `.exit` 0 (identity x4, failure x2, the fault gate
  default and plain, twin x2, the hit gate x2). The unit cells' attempt 1 read `server-cpu=101`: one failure,
  `build_identity_tests::build_id_is_rederivable_from_the_source_tree` (`baked fingerprint disagrees with a re-derivation
  over 679 files`), because ft's test binary was built on the arms' crates and run in the tip's working tree, whose
  crates are S3's; the test re-derives the id from the working tree by design. Attempt 2 ran the same cell with the arms'
  crates checked out (`unit-attempt2-tree.txt`: `8 files changed`, then the tree back, `0` dirty): `unit-cells
  engine-parallel=0 engine-serial=0 door=0 server-cpu=0 engine-cpu=0` (`911 passed`). Attempt 1 kept as
  `unit-attempt1-tip-tree/`.

**Decision, by DAY39 section 5's rule for this host class: (a), (b) and (c) pass; T is kept for the 9950X class and
item 3 closes.**
