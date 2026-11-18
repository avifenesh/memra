# memra#1315's RTX PRO 6000 gates on the landing head, 2e42b5549830

Scoped extras for the landing of memra#1315, memra#1327, memra#1308 and memra#1326. They are not a cell record; the gate does not read them.

## Why

memra#1315 (a call flag names its rows, memra#1292) ran its PRO 6000 gates on its own head. This landing's head also holds memra#1327, which edits two of the same files (`crates/memra-engine/src/dsv41_gpu/tp.rs` and `crates/memra-engine/src/bin/dsv41_gate.rs`); git merged them without a conflict, and no lane had run that combination. These runs are memra#1315's landing test list on the combined head.

## What ran

On a four-card RTX PRO 6000 Blackwell Max-Q workstation on 2026-11-18, untimed, right after the device record `2e42b5549830-rtxpro6000-battery-serving`. `dsv41-gate` was built from the landing head on the same box with CUDA 13.4, in its own worktree and target directory (`scripts/build1315.sh`; not LTO). The serve smoke used the record's own `memra-server` (its sha256 in `one-card/summary.txt` equals the record's `build.binaries` digest). Every gate ran with `MEMRA_FULL_PREC=1` on DeepSeek-V4.1-Flash; the commands are in `scripts/gates1315_inner.sh` and `scripts/gates1315_pair_inner.sh`.

### One card (`one-card/`, 02:06:54 to 02:11:08)

| item | command | result |
|---|---|---|
| 5 | `dsv41-gate kernels` on the random fixtures | 1,027 checks, 0 failed |
| 6a | `rowfault --layers 2 --devices 0` | 9 checks, 0 failed |
| 6b | `rowfault --layers 2 --experts-host --devices 0` | 9 checks, 0 failed |
| 6e | `rowfault --layers 40 --experts-host` (the one-card residency) | 9 checks, 0 failed |
| 7 | the PR's 40-layer serve smoke, `smoke_wrx2.sh` | SMOKE PASS, 16 PASS lines, 0 FAIL |

The smoke script is the PR's (`research/modelplan-onboarding-dsv41-20261109/row-faults/receipts/pro6000-6371a9975b/scripts/smoke_wrx2.sh`) with one line changed, the directory it takes `memra-server` from (`one-card/smoke27.diff`; the script as run is `one-card/smoke27.sh`).

### Red arm (`one-card/red-*`)

`dsv41-gate` rebuilt with the PR's `red.patch` (`Group::moe` back to main's check): 6a still passes (9 checks, 0 failed) and 6b fails as it must, with `route: 0 of 174 pairs placed` (2 checks, 1 failed, exit 1). So 6b reads the change.

### A pair (`pair/`, 02:17:23 to 02:18:22, cards 0 and 1)

| item | command | result |
|---|---|---|
| 6f | `rowfault --layers 2 --devices 0,1` | 9 checks, 0 failed |
| 6g | `rowfault --layers 2 --experts-host --devices 0,1` | 9 checks, 0 failed |
| 6h | `rowfault --layers 40 --experts-host --devices 0,1` | 9 checks, 0 failed |

## Conditions

Each run held its cards' locks for its whole length. The cards' thermal slowdown counters did not move, and the heat guard (`scripts/heat_guard27.sh`: stop only at 92 C or more with the fan at 95% or more, three samples in a row) never fired; the peak was 64 C on the one card and 68 C on the pair (`throttle-1hz.csv` and `heat.log` in each directory). The run's processes were pinned to 16 of the box's 64 host threads. Nothing here is timed.

## Not run here

Item 8 of the list (the support record's `verify checkpoint` and `verify serve`) is for a re-pin; this landing does not re-pin. The GB10 items of the list are in `../2e42b5549830-gb10-dsv41-gates/`.
