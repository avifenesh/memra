# TP/EP B-row steps and their graphs (memra #710, 2026-09-25/26)

Owner ruling 2026-09-25, after the flip's concurrency cost: "Flip now, B-row next". The TP/EP
default (#727) served one lane. At c2 PP-2's two pipelined lanes aggregated 120.9 tok/s against
TP/EP's 76.9 on the Workstation pair.

Model: `tiyuvta/DeepSeek-V4-Flash-0731-NVFP4@bafd09f8cab4f4f4f25e1cdafbcdefc05b90ee38`.

## What changed

- **TP/EP B-row step.** One TP/EP decode step over several requests.
  - `tp_ep_trunk_walk_rows` walks row groups: each group is one request's caches and
    checkpoints on both ranks. The one-request walk is one group.
  - Both ranks read every weight once for all rows. Each row's attention reads only its own
    caches.
  - Each request commits both planes as its one-row step would.
  - A request armed for full-token replay may ride the step and resume its replay afterwards.
- **Graphs for it.** `decode_rows_draw` captures a forward and a commit graph per rank for a
  batch.
  - The batch is named by request serials in serial order.
  - The graphs run on the full-token replay's device-position kernels, one row per request.
  - Each row's token, position and uniform live in device words, so the same graphs step the
    same batch at every position below the limit.
  - Rows draw the device argmax or the device sampler inside the commit graph.
  - A new batch recaptures. A state's serial is process-unique, so a new request at a freed
    request's address never reuses its graphs.
- **Multi-row dense-fast kernels.** `dsv4_dense_fast_fp8_kernel` and `dsv4_dense_fast_dots_kernel`
  take M = 2..8 token rows.
  - Each row keeps its own accumulator in the same leaf order and reduces through the same tree.
  - Two barriers per row, where the m-row kernels used seven.
  - They replace `dsv4_gemv_fp8_m_kernel<M>` and `dsv4_dots_f32acc_mrow_kernel<M>` at those
    widths. B-row steps and verify rounds both take them.
- **Hoisted compressor projections.** A multi-request attention step computes each
  compressor's kv and score projections once over every row, and each request copies its rows
  in.
- **Server.** The plain TP/EP route defaults to four lanes. Each lane's plain steps coalesce
  into B-row steps (`MEMRA_DSV4_ROWS` defaults to the lane count on TP/EP). A lane alone steps
  on its own full-token replay.

## Correctness

`dsv4_rows_gate` on TP/EP (`DSV4_ROWS_GATE_TOPOLOGY=tp_ep`, Workstation pair, `raw/ws-gates/`,
final at 7150cb5f1). Four sessions (prompts of 256, 197, 311 and 150 tokens) decode 24 greedy
steps alone for the solo trace. Every arm compares each step's full logits bits against it:

```
PASS: every row bit-identical to its solo step across join, leave and row moves
PASS: replayed, then 8 B-row steps beside a peer while armed, then replayed again: bit-identical to solo steps
PASS: B-row graph steps bit-identical to solo steps across join, leave and row moves (5 captures, 28 graph steps)
PASS: sampled B-row graph draws equal the eager B-row draws (24 steps x 4 rows)
```

The first graph run captured on every step (28 captures, 28 graph steps), because the gate
rotated row order and the batch key was the ordered tuple. From fbcdfbe64 the key is the batch
in serial order, so reordered rows replay the same graphs.

`tests/dsv4_dense_fast_rows_gpu.rs`:
- Every output bit of the M-row FP8 GEMV and dots equals the old m-row kernels' and each row's
  one-row launch.
- It covers the served shapes, contiguous and strided activations, and BF16 and f32 weight
  planes.
- A red arm moves one token row alone.

Regressions:
- The walk refactor keeps the replay program: `dsv4_tp_replay_long_gate` plain 304 steps
  PASS, tokens sha `112c2fc6...` unchanged.
- It keeps the DSpark verify program: `dsv4-gpu-dspark-gate` on TP/EP and PP-2 PASS.
  - The accept shas are unchanged on both pods.
  - The proposal digests match within each pod. The same binary gives `62b368f1/d404da5f` on
    the Server Edition pair and `cc6082dd/373e6557` on the Workstation pair: the control arm
    `raw/ws-gates/tp-rows-ctl/` is the pre-kernel binary on the WS pod. That makes it a pod
    difference in draft confidence bits, not a program change.
- The PP-2 rows gate passes too, at the final head: the hoisted compressor projections also
  run in PP B-row attention. `raw/ws-gates/rows-pp-7150cb5f1/` has every row identical and the
  pipelined groups identical.
- The DSpark gate on PP-2 at the final head passes with the pod's digests.

Served: every request's text sha is equal across every arm of every cell below.

## Gate timing (one TP/EP B-row step, 64 steps, WS pair, `raw/ws-gates/rows-hoist-7150cb5f1/`)

| rows | eager B-row | graph B-row | tok/s, graph |
|---|---|---|---|
| 1 | 15.94 ms (the one-row eager step: 16.00) | (one row replays) | 62.8 eager |
| 2 | 21.35 ms | 17.80 ms | 112.4 |
| 4 | 29.18 ms | 25.42 ms | 157.4 |

## Anatomy

**Eager B-row, WS pair** (`raw/ws-anat/`, `raw/ws-gates/tp-rows-fast/anat-*`):
- The multi-row dense-fast GEMV took the dense cost at B=2 from 4.09 to 3.36 ms per card, and
  at B=4 from 5.89 to 4.34.
- The wall time did not move, because the eager step is launch-bound: B=1 had about 6 ms of
  host gap per step.

**Graph step, SE pair** (`raw/se-graph-anat/`, B=2), dev1:
- 19.9 ms per step, 18.5 busy.
- MoE multi-row stream: 3.93 ms, the union of both rows' experts.
- Dense FP8: 3.57 ms.
- Row gathers: 1.48 ms. Expert all-reduce: 1.03 ms.
- Compressor and head dots: 1.04 + 0.91 ms.
- Sink attention: 1.0 ms.
- Compressor snapshot copies: 0.57 ms (49 MB of device-to-device copies per step for the
  refusal rollback).
- Each added row costs about 5 ms of device time, 2 ms of it expert bytes.

## Served

One boot per row, median of the rows. `agg` is the cell's aggregate completion tok/s. `dec` is
the per-request decode p50.

### Final head 7150cb5f1, Workstation pair (`raw/ws-final/`, N=3, order Gd Pp Mn Mn Pp Gd Gd Pp Mn)

| cell | TP/EP default (four lanes, graphs) | main (the flip, one lane) | PP-2 (`pp`) |
|---|---|---|---|
| greedy c1 dec | 80.60 | 80.69 | 68.48 |
| greedy c2 agg | 103.17 | 76.85 | 120.79 |
| greedy c4 agg | 135.76 | 76.69 | 120.63 |
| sampled c2 agg | 102.79 | 76.41 | 107.39 |
| greedy c2 TTFT p50 | 242 ms | 3488 ms | 309 ms |
| greedy c4 TTFT p50 | 430 ms | 10155 ms | 4462 ms |
| 2k-context c2 agg | 20.26 | 18.96 | 21.57 |

Against main: c2 +34%, c4 +77%, c1 -0.1%. Against PP-2: c4 +12.5% with a tenth of the TTFT,
c2 -15%.

### Server Edition pair, graph lane before the hoist, 47df550b2 (`raw/se-graph/`, N=3)

| cell | TP/EP default (four lanes) | two lanes | main | PP-2 |
|---|---|---|---|---|
| greedy c1 dec | 71.20 | 71.43 | 71.43 | 62.36 |
| greedy c2 agg | 93.36 | 93.40 | 69.75 | 113.15 |
| greedy c4 agg | 121.94 | 93.39 | 69.71 | 112.95 |
| greedy c4 TTFT p50 | 452 ms | 5697 ms | 11168 ms | 4783 ms |

### Workstation pair, graph lane 88907ed28 (`raw/ws-graph/`, N=3; Pp N=2, see below)

| cell | two lanes, graphs | four lanes, graphs | flip (main, one lane) | PP-2 (`pp`) |
|---|---|---|---|---|
| greedy c1 dec | 80.60 | 80.55 | 80.67 | 68.49 |
| greedy c2 agg | 102.04 | 102.03 | 76.86 | 120.85 |
| greedy c4 agg | 102.18 | 132.80 | 76.74 | 120.59 |
| sampled c2 agg | 101.68 | 101.47 | 76.42 | 107.21 |
| greedy c2 TTFT p50 | 238 ms | 243 ms | 3490 ms | 301 ms |
| greedy c4 TTFT p50 | 5208 ms | 419 ms | 10140 ms | 4468 ms |
| 2k-context c2 agg | 20.19 | 20.20 | 18.97 | 21.51 |

The default is four lanes: at c4 that is 132.8 against 102.0 aggregate, TTFT 0.42 s against
5.2 s, and c1 and c2 are unchanged.

### Server Edition pair, eager B-row plus the dense-fast rows kernel, e7a3f32a1 (`raw/se2/`, N=3; R4 N=2)

| cell | two lanes, eager B-row | four lanes | flip (main) | PP-2 |
|---|---|---|---|---|
| greedy c2 agg | 80.69 | 80.60 | 69.69 | 113.15 |
| greedy c4 agg | 81.21 | 109.66 | 69.77 | 112.87 |
| greedy c2 TTFT p50 | 255 ms | 259 ms | 3842 ms | 326 ms |

The dense-fast rows kernel also serves DSpark verify rounds. DSpark greedy c1 went from 72.80
(main) to 77.91 (+7.0%), and sampled from 63.30 to 66.56 (+5.2%). N=3 each, same text.

### What remains against PP-2

- At c2 the graph step (17.8 ms for two rows) is still slower than PP-2's two pipelined one-row
  steps: 102 against 121 tok/s on WS.
- At c4, TP/EP with four lanes passes PP-2: 132.8 against 120.6, TTFT 0.42 s against 4.5 s.
- The per-row cost is mostly expert bytes (the union of each row's top-6), the dense GEMV at
  half shapes, and the per-row attention.

## PP-2 hang, #722

- Row r10-Pp of the WS graph A/B stalled in its c2 cell with two requests in flight, and the
  45-minute row timeout ended it.
- On SIGTERM the server drained in 0.1 s, so the wait was host-side, not a device that never
  finished.
- That is the fourth PP-2 stall on this pod, recorded on #722. No TP/EP row has stalled.
