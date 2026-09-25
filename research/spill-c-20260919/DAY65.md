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
