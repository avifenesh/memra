# F8F4 tile v3 probes, unit 2 (2026-11-07)

Receipts for `../GEMM-V3-DESIGN.md` and ledger rows 33-37. One DGX Spark (GB10, sm_121a, 48 SMs),
CUDA 13.0, SM clock 2.10-2.19 GHz under load. MAC/clk/SM uses 48 SMs at 2,200 MHz.

## Logs and the source state that produced them

| log | what ran | arms (current `src/tma_probe.cu` ids) |
|---|---|---|
| `stage1.log` | `src/stage1.sh`: the shipped tile in PROBE_MODE 0-3 (`src/driver_ncu.cu` over the round-3 `src/mmq_probe.cu`), `harness/ceil_probe3.cu`, the round-3 `src/probe_mma_f8f4.cu`, `src/nreg_rt.cu`, then ncu | |
| `stage2-round1.log` | the first TMA probe build: bounded spins (trap after 2^26 polls), and issue-end issuing stage kt + S without stage S - 1 in its prologue, so stage S - 1's full barrier was never armed; the bounded wait trapped and the context died, which is every `FAIL` and `cuTensorMapEncodeTiled failed 719` line of the issue-end pass. The issue-top and producer-warpgroup passes and the co-issue probe (`src/coissue.cu`, last 8 lines) stand | issue-top = 0 (with the spin-bounded wait), producer-wg = 7 |
| `stage2.log` | `src/stage2.sh`, plain waits: 5 shapes at T 4,096, then gate+up at T 1,024 and 130, one process per arm | 0, 1, 2, 3, 4 |
| `stage3.log` | `src/stage3.sh`, plain waits, order 1 5 6 6 5 1 | 1, 5, 6 |
| `stage4.log` | `src/stage4.sh`, suspended waits, order 8 9 9 8 | 8, 9 |
| `stage5.log` | `src/stage5.sh`, suspended waits, order 8 10 11 12 13 13 12 11 10 8 | 8, 10-13 |
| `stage6.log` | `src/stage6.sh`, suspended waits, order 8 14 14 8 8 14 | 8, 14 |

The logs print arm names without the `plain` / `hint` suffix the current source adds; within each log
the wait form is the one in the table. Every arm that ran to completion matched the shipped tile bit
for bit (`bitdiff=0`) on 5 shapes at T 4,096; only the `stage2.log` arms also ran gate+up at T 1,024
and 130.

`stage3.sh` to `stage6.sh` filtered the `v2 shipped (again)` lines before `tee`, so those logs lack
the same-process A/A check. `stage2.log` keeps it: the shipped tile timed twice in one process differs
by -3.7..+2.2% on gate+up and -4.7..+0.3% on down (n = 5 per shape, `pairs-nohint.txt`), so down's
+1.9% is inside the noise and gate+up's +7.6% (all 7 pairs +5.0..+10.0%) is not.

## Aggregates

`src/pairs.py` divides each arm's rate by the shipped tile's rate timed just before it in the same
process, grouped by arm, token count and shape: `pairs-hint.txt` (stages 4-6), `pairs-nohint.txt`
(stages 2-3), `pairs-round1.txt`.

## ncu

`ncu/*.raw.csv`: `--set full` raw pages, one launch each at the gate shape (and down for the shipped
tile). `ncu/summary-gate.txt` (`src/ncu_summary.py`) puts them side by side;
`ncu/sass-stalls-*.txt` (`src/sass_stalls.py`) attribute warp-stall samples to SASS opcodes.

## Reproduce

On a GB10 under its lease, from `src/` (the dir holds `mmq_probe.cu` and `mmq_common.cuh`):

    nvcc -gencode arch=compute_121a,code=sm_121a -O3 -std=c++17 --expt-relaxed-constexpr tma_probe.cu -lcuda -o tma_probe
    ./tma_probe 4096 11 -1 8,14 | tee raw.log      # T, launches, shape (-1 all), arm ids

ncu needs root there: `sudo -n -E env PATH=$PATH ncu --set full ...` (see `src/stage1.sh`).
