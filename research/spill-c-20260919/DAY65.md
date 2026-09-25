# WP-C day 65 (2026-09-25): OWED C12, the door's host placement on a two-complex CPU, registered before any cell

Lead, after BOX16: "Until then, work what does not need a card: pre-register C12 (the one-complex against two-complex
pin on a 9950X-class host, beside lane A's owed 9950X fill reading) and anything else in your ledger that is CPU-side,
same rules as before." Tree at start: `7a95f4924` plus `DAY64.md` section 5a.

## 0. The finding this cell tests (`DAY64.md` section 4)

On BOX15 (a Ryzen 9 9950X host, the runner pinned with `taskset -c 0-11`) every door arm was bimodal by boot: fast
boots within 3 to 6 ms of REF over 32 generated tokens, slow boots 57 to 80 ms behind, with every CPU-side stage part
about 1.85 times larger and the GPU copies unchanged; REF was unimodal. The hypothesis stated there: the 12 pinned CPUs
span both of the 9950X's core complexes (two L3 domains), and the door's decode, bound by its owner thread's CPU
work, runs slow when that thread (or the memory it touches) sits on the other complex. No receipt placed the thread.

## 1. Pre-registration: the cell `pin` (a 9950X-class host; before its script)

**Host.** One RTX PRO 6000 Blackwell Workstation Edition on a host whose CPU has at least two L3 domains (the Ryzen 9
9950X class), the approved 35B artifact, the lane tip in `/root/wt-c` with a detached build worktree. It may share a
sitting with lane A's owed 9950X-class fill reading; neither cell reads the other.

**Binaries.** `c60=da649107c` (REF, `run-gen-c60`) and `i15=2243b1fe2` (the door with I13, I14, I15, `run-gen-i15`),
built on the box by `day63-box-build.sh`.

**The two pins, read from the host before the arms** (`ev/topology.txt`: `lscpu`, and every CPU's
`cache/index3/shared_cpu_list`):
- TWO: `0-11`, the pin of BOX15's cell. The cell requires it to span at least two L3 domains; if it does not, the
  cell runs no arm and reads `not_applicable`.
- ONE: the first 12 CPUs, in number order, of the L3 domain that holds CPU 0 (on a 9950X, `0-7,16-19`: CCD0's eight
  cores and four of their SMT siblings), the same CPU count as TWO.

**The cell** (`day65-cell.sh`, reader `day65-read.py`): day 18's pressure shape (`MEMRA_MOE_RESIDENT=0 MEMRA_NGEN=32
MEMRA_MOE_SLOTS=9986`, prompt `55 88 13`), one collector hold, 250 ms telemetry. Each run is started under
`taskset -c <its arm's pin>` (the pin is the run's CPU budget; the runner itself is not pinned). Arms: REF2 and I15TWO
(REF and the door on TWO), REF1 and I15ONE (on ONE). Order 1 (REF2, I15TWO, REF1, I15ONE) x 5, order 2 reversed x 5,
40 runs, N=10 per arm. The placement sampler of `DAY64.md` section 5 (`ev/placement.tsv`: every 250 ms, each `run-gen`
process's last CPU).

**Readings.**
- R1: every arm's gen-only and steady-window median and IQR.
- R2: the slow boots: a door run is slow iff its gen-only seconds exceed the median of REF's runs on the same pin by
  more than 0.030 s (BOX15's slow boots read 0.062 to 0.080 over REF, its fast ones 0.003 to 0.006).
- R3: the placement of each door run: the share of its samples on CPUs of CPU 0's L3 domain, beside its slow or fast
  mark.
- R4: the door against REF on ONE, under `DAY64.md` section 5's admissibility clause (REF1's and I15ONE's IQRs at most
  0.005 s): `beats`, `matches` or `loses` as `DAY61.md` section 2 defines them, or `void (inadmissible)`.

**The verdict** (`DAY65 PIN VERDICT`):
- `not_applicable`: TWO does not span two L3 domains on this host.
- `not_reproduced`: no I15TWO run is slow (the bimodality did not appear; nothing about the pin can be read).
- `pin_removes`: at least one I15TWO run is slow and no I15ONE run is.
- `pin_does_not`: at least two I15ONE runs are slow.
- `inconclusive`: otherwise.
Beside it, `placement_tracks_mode` iff every slow I15TWO run has more than half its samples outside CPU 0's L3 domain
and every fast I15TWO run more than half inside, else `placement_does_not_track`.

**What it decides.** Where the bimodality comes from, and the door's admissible reading on a two-complex host when the
run is kept on one complex. It changes no code and no default: a placement fix (for example, the owner thread's own
affinity) would be its own registration with its own cells, after this one reads.

## 1a. The sitting, prepared before any cell

`day65-cell.sh` (cell `pin`; the day-40 runner body of days 60 to 64 unchanged, the pins read from sysfs, a
`not_applicable` exit when `0-11` spans one L3 domain, the placement sampler stopped by its own pid) and
`day65-read.py` were written after section 1. The reader was dry-checked for mechanics on BOX15's receipts
relabelled (REF as REF2 and REF1, I15 as I15TWO, I14 as I15ONE, a synthetic placement file) and printed its lines
end to end; its verdict there means nothing. The pin derivation run on the local host (one L3 domain, CPUs 0 to 23)
reads `two_l3_domains=1`, the `not_applicable` case. The driver `day65-box.sh` (builds `c60` and `i15` by
`day63-box-build.sh`; no runner pin, each run pinned by the cell) is dry-checked for control flow
(`day65-cpu/dry-check-driver.log`). Run as
`D65_BUILDS="c60=da649107c i15=2243b1fe2" bash /root/wt-c/research/spill-c-20260919/day65-box.sh` on a Ryzen 9 9950X
class host with one RTX PRO 6000 Blackwell Workstation Edition, the approved 35B artifact at `/root/artifacts/`,
`/root/wt-c` at the lane tip and a detached `/root/wt-c-build`, CUDA 13 and Rust, at least 48 GB host `MemAvailable`.
Expected: two builds about 10 minutes, the cell about 8 (40 runs).

## 2. The cell, as it ran (BOX17, run by the lead as registered; `pro-single-day65/`)

The lead ran `day65-box.sh` as section 1a names it on the tree `5c6e7c0cf`, on BOX17, BOX15's own machine (a Ryzen 9
9950X host with two L3 domains, one RTX PRO 6000 Blackwell Workstation Edition, driver 595.84; the BOX15 regime:
idle 27.2 W, FMA spin 2650 to 2673 MHz, no brake), 09:17Z to `box done 2026-09-25T09:28:25Z`; binaries built on the
box (`box-binaries.sha256`). The lead mirrored the receipts; read here: 190 of 190 files `OK` against
`box-mirror-manifest.sha256` (`MIRROR-CHECK.txt`). Regime (`regime.txt`): 25 to 40 C, SM median 2610 MHz. Verbatim
(`pin/reading.log`):

- `DAY65 PINS rig=pro-single two=0-11 one=0,1,2,3,4,5,6,7,16,17,18,19 home_l3=0-7,16-23 two_l3_domains=2`
- `DAY65 PIN CHECKS rig=pro-single runs=40 integrity=ok`
- `DAY65 R1 ref2: gen median=0.245 iqr=0.0000 | window median=0.218 iqr=0.0010 (N=10)`
- `DAY65 R1 i15two: gen median=0.311 iqr=0.0200 | window median=0.273 iqr=0.0175 (N=10)`
- `DAY65 R1 ref1: gen median=0.245 iqr=0.0000 | window median=0.218 iqr=0.0000 (N=10)`
- `DAY65 R1 i15one: gen median=0.312 iqr=0.0192 | window median=0.273 iqr=0.0162 (N=10)`
- `DAY65 R2 i15two: slow=8 of 10 (gen-only over ref2 median + 0.03) [...]`
- `DAY65 R2 i15one: slow=8 of 10 (gen-only over ref1 median + 0.03) [...]`
- R3, every door run: `home_share` 0.93 to 1.00 on TWO (the two fast runs 1.00 and 0.98), 1.00 on ONE, slow runs
  among them in both (`o1-i15one-r1: samples=54 home_share=1.00 slow`, and the rest in the reading).
- `DAY65 R4 i15one_vs_ref1 gen-only decode: pooled=+0.0670 o1=+0.0680 o2=+0.0660 noise=0.0192 -> void (inadmissible)`
- `DAY65 PIN VERDICT rig=pro-single integrity=ok -> pin_does_not placement_does_not_track; one-domain door vs REF: gen void (inadmissible), window void (inadmissible)`

**Read as registered.** The pin does not remove the bimodality: on one L3 domain the door is slow in 8 boots of 10,
the same count as on two, and its owner thread sat on CPU 0's domain in every sample of every one-domain run, slow or
fast. The hypothesis of `DAY64.md` section 4 (the other complex) is refuted. What stays true in every door boot on
this machine, slow or fast: the slow boots' host fill before decode takes 1278 to 1342 ms and the fast boots' 1114 to
1120 (read from the logs, in run order in `pin/ev/marks.tsv`); REF is 0.245 or 0.246 s in all 20 runs. The next
registration (`DAY66.md`) reads what the receipts cannot yet separate: the owner core's clock and the process's page
backing, per boot.
