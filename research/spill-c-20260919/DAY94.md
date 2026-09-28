# WP-C day 94 (2026-09-27): OWED C2, option 1: where the 285K class's gen-only gap sits, the cell `where285`, registered before any card

`DAY91.md` section 3: across I20 to I24 the local door-only work fell 57 percent. The 285K class's gen-only gap to
REF stayed at +3.0 ms, while the 9950X class's closed from +3.0 to -1.0. So on the 285K the remaining gap is
something the door-only CPU cut does not reach, and its steady window matches in every sitting. The 285K box has no
systemd scope, so every cell ran under `taskset -c 0-11`, 12 of its 24 cores, on a hybrid P- and E-core part whose
core layout no receipt records. This cell locates the gap on the 285K itself, before any further cut is sized against
it. Tree at start: `9818dd464`.

## 1. The cell `where285` (one sitting on the 285K class)

**Binaries.** `run-gen-i25` (`ee41ede8f`: the naked door, untraced, the lane's current program) and `run-gen-i24`
(`1fd4b24c0`: the program the I24 sitting ran, traced), built on the box by `day88-box-build.sh` from their commits.

**The shape.** `DAY88.md` section 5's spill shape: `MEMRA_MOE_RESIDENT=0 MEMRA_NGEN=32 MEMRA_MOE_SLOTS=9986`, prompt
`55 88 13`.

**The core lists** (`day94-box.sh`, recorded in `cores.log` and the provenance).
- **The P-cores:** the kernel's hybrid list `/sys/devices/cpu_core/cpus`; where absent, the CPUs with the highest
  max MHz in `lscpu -e`. The E-core list (`cpu_atom`) and the full `lscpu -e` table are recorded beside it.
- **The wide list:** the box cap's cores, `0-11` under the taskset cap, the affinity every earlier 285K cell ran with.
- **Fail closed:** the driver stops before the cell if it finds no P-core list distinct from the wide one.

**Arms.** Each run gets its own affinity (`taskset -c`), recorded per run in `<label>.cpus`.

| arm | binary and program | cores | clock |
|---|---|---|---|
| `wn` | `i25` naked | wide | none |
| `wl` | `i25` legacy (`MEMRA_EXPERTS_VIA_TIER=0`) | wide | none |
| `pn` | `i25` naked | P-cores | none |
| `pl` | `i25` legacy | P-cores | none |
| `wnc`, `wlc`, `pnc`, `plc` | as `wn`, `wl`, `pn`, `pl` | as those | `--moe-dispatch-clock` |
| `wo` | `i24` naked | wide | none (beside: I25 against I24 on this host) |

Order 1 runs the arms in the table's order x 5 and order 2 reverses it x 5: 90 timed runs. Then two untimed traced
twins of `pn` (`--expert-bank-trace`) read the host demand sequence. One collector hold under `/tmp/memra-gpu.lock`.

**Integrity** (`day94-read.py`):
- every run exits 0 with `MATCH` and 32 tokens, and one tape runs across all 92 runs;
- each arm prints its program's lines (the door arms qualified, installed, the registered pool and the prefetch on;
  the legacy arms `off: MEMRA_EXPERTS_VIA_TIER=0` and the prefetch on);
- the door arms show the fill complete and `physical_reads=0`;
- `i25`'s arms write no host demand line, while `wo` and both twins give the door's sequence `4bdc2610c3534e42` in
  22077 lines;
- every run's CPU list is its arm's.

**Admissibility.** DAY64's clause over the nine timed arms: gen-only and window IQR at most 0.005 s.

**Readings** (DAY61 section 2's rule through `day88-read.py`'s `compare`, gen-only primary, the window beside):
- `GAP wide`: `wn` against `wl`, the gap as every earlier cell measured it. The prediction is `loses`.
- `GAP pinned`: `pn` against `pl`, the gap with both programs on the P-cores alone.
- Beside, deciding nothing:
  - `PIN door` (`pn` against `wn`) and `PIN legacy` (`pl` against `wl`): what pinning does to each program;
  - `I25_vs_I24` (`wn` against `wo`);
  - the clocked twins' dispatch brackets per phase (`dispatch`, `prefetch`, `pf_demand`, `pf_resident`, `pf_retire`,
    `pf_stage`, in us per generated token), generate against window, wide against pinned.

**The verdict**, in this order:
1. `void` on failed integrity or admissibility.
2. `no_non_p_cores_in_wide` if the wide list holds only P-cores: the hypothesis cannot apply, and the gaps are
   reported as read.
3. `gap_not_reproduced` if `GAP wide` does not read `loses`.
4. `pinning_closes` if `GAP pinned` reads `matches` or `beats`.
5. `pinning_does_not` otherwise.

**What each verdict leads to** (registered now, no reading argued after the card):
- **`pinning_closes`:** the 285K gap is where the door's CPU work lands on a non-P core. The fix registers on its own
  before its code: the door's latency-bound CUDA-owner thread placed on a P-core by detection (the per-hardware rule:
  detection over flags, keyed on the device class), with its own gates and a `promo` sitting. It is not a cell
  setting.
- **`pinning_does_not`:** the gap is not core placement. The clocked twins' generate-phase brackets, against the
  window's, name the next registration.
- **`no_non_p_cores_in_wide`:** the hypothesis is refuted on this host as its lists read. The clocked twins lead
  likewise.

**Where.** The 285K class (the BOX44 and BOX46 machine), one RTX PRO 6000 Blackwell Workstation Edition, at least 48 GB
MemAvailable, about 60 minutes (two builds and 92 runs). Receipts go to `c-day94`, mirrored as `pro-single-day94/`.

## 2. Checked before the card (`day94-cpu/`)

- **The cell under stubs** (`dry-check-cell.log`): 92 calls in the registered order. Each stub read its own affinity:
  the wide arms on the wide list, the pinned arms on the P-list. Each arm's environment is its program's. All exits
  0, and the per-run CPU files match.
- **The reader over a synthetic cell** (`make-synthetic.py`, `dry-check-reader.log`), built from the I24 sitting's
  `promo` logs mapped onto the arms: integrity `ok`, admissible, and `pinning_does_not` (the synthetic pinned arms are
  copies of the wide ones). The readings mean nothing; the plumbing reads.
- **Shellcheck** is clean on `day94-box.sh` and `day94-cell.sh`.

NEED TARGET CARD: `D94_BUILDS="i24=1fd4b24c0 i25=ee41ede8f" bash /root/wt-c/research/spill-c-20260919/day94-box.sh` on
the 285K class.

## 3. The lead's scheduling and two rulings, registered before the card (2026-09-27)

- **Where.** BOX46, the same 285K machine every 285K half ran on (chain `c21-box46.sh`, after lane A's F2, about
  01:20Z). Tree `836b3a96c`, receipts `c-day94`, mirror `pro-single-day94/`.
- **The program.** The lead: "Keep I25 as registered. The cell locates the gap (placement against program), so an
  I26 addendum would only move the question." The cell runs `i25` and `i24` as section 1 says; I26's record is its
  fixture and queue v23.
- **The E-core reading stays a hypothesis** until this cell reads.
- **If the verdict is `pinning_closes`,** every earlier 285K verdict ran under `taskset -c 0-11`. Each record gets a
  banner naming that placement and pointing here; its readings are not rewritten. The records are `DAY64.md` to
  `DAY91.md`, the 285K halves.
