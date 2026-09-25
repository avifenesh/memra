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
