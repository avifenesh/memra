# memra#1315's and memra#1327's GB10 gates on the landing head, 2e42b5549830

Scoped extras for the landing of memra#1315, memra#1327, memra#1308 and memra#1326. They are not a cell record; the gate does not read them.

## Why

The landing head holds memra#1315 (a call flag names its rows, memra#1292) and memra#1327 (the split MoE keeps the bf16 gather while a rank's stores fit a push, memra#1311). Both edit `crates/memra-engine/src/dsv41_gpu/tp.rs` and `crates/memra-engine/src/bin/dsv41_gate.rs`; git merged them without a conflict, and no lane had run that combination on GB10. The device record `2e42b5549830-gb10-battery-perf-serving` holds no DeepSeek-V4.1 model, so these runs are the GB10 items of both PRs' landing lists on the combined head.

## What ran

On one DGX Spark (NVIDIA GB10, sm_121a) on 2026-11-18, 09:33:05 to 10:16:50 local time, untimed, in the same unit lease as the device record and right after it. `dsv41-gate` was built from the landing head in the record's own tree with the record's build command and environment plus that one binary (`build.log`); the six record binaries were byte-equal to the record's after the build (`summary.txt`). The serve smoke used the record's own `memra-server` (its sha256 in `summary.txt` equals the record's `build.binaries` digest). Every gate ran with `MEMRA_FULL_PREC=1` on DeepSeek-V4.1-Flash; the checkpoint's cached pages were dropped before each load, as the PRs' GB10 scripts do. The commands are in `scripts/gb10_extras27.sh`. The gate binary is left out; its digest is in `BINARIES.sha256`.

### memra#1315

| item | command | result |
|---|---|---|
| kernels | `dsv41-gate kernels` on the random fixtures | 1,027 checks, 0 failed |
| rowfault, resident, one rank | `rowfault --layers 2 --devices 0` | 9 checks, 0 failed |
| rowfault, experts on host, one rank | `rowfault --layers 2 --experts-host --devices 0` | 9 checks, 0 failed |
| rowfault, resident, two ranks | `rowfault --layers 2 --devices 0,0` | 9 checks, 0 failed |
| rowfault, experts on host, two ranks | `rowfault --layers 2 --experts-host --devices 0,0` | 9 checks, 0 failed |
| smoke | the PR's serve smoke, `smoke5.sh` | SMOKE PASS, 18 PASS lines, 0 FAIL |

The smoke script is the PR's (`research/modelplan-onboarding-dsv41-20261109/row-faults/receipts/gb10-50413a3a4d/scripts/smoke5.sh`) with its two path lines changed, the directory it takes `memra-server` from and its output directory (`smoke27.diff`; the script as run is `smoke27.sh`).

### memra#1327

The `tp` gate on the needle capture, layers 1, 2, 14, 20, 21 and 24, two decode steps, under each transport. Two ranks on the card (`--devices 0,0`), then four (`--devices 0,0,0,0`), the form the PR's own GB10 receipts ran. Its kernel gate is the one above.

| ranks | default | peer-pull | host-canonical |
|---|---|---|---|
| 2 (`0,0`) | 560 checks, 0 failed | 558 checks, 0 failed | 558 checks, 0 failed |
| 4 (`0,0,0,0`) | 1,056 checks, 0 failed | 1,054 checks, 0 failed | 1,054 checks, 0 failed |

`stress/push_gather`: 20,000 gathers, 0 words differ, 0 ranks flagged, on two and on four ranks. Its control arm (`push_gather_control_word_first`) catches the word before the data on both, as it must.

## Conditions

One GB10, held by this landing's unit lease for the whole run; the gate was the only GPU process on it. Host load (1-minute) 0.67 at the start and 2.36 at the end. Nothing here is timed.
