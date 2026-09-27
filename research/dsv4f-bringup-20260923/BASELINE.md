# DSv4-Flash served baseline, 2026-09-23

> **SUPERSEDED, rates invalid (2026-09-23).** Every rate and timing on this page was measured
> under a latched HW Power Brake Slowdown: effective SM clock 722 MHz behind a reported 2865 MHz.
> The correctness findings stand. Finding and proof: `power-brake/POWER-BRAKE.md`; healthy-pair
> numbers: `REBASELINE.md`.

Scope: one model, one hardware shape. `tiyuvta/DeepSeek-V4-Flash-0731-NVFP4@bafd09f8cab4f4f4f25e1cdafbcdefc05b90ee38`
on 2x RTX PRO 6000 Blackwell Workstation Edition (600 W limit, 3090 MHz max SM clock, driver
595.71.05, GPUs on one NUMA node, `NODE` PCIe path). Rented box; memra-server built from main
`5f1b0eda4` (binary sha256 in each `raw/*/binary.sha256`). One scored campaign at a time under
`/tmp/memra-gpu.lock`, 250 ms GPU telemetry per cell.

This is the floor the bring-up tunes from (memra #4). The DSv4 serving bring-up pause of
2026-09-12 is lifted for this lane by the owner's 2026-09-23 goal.

## Served cells

memra-server naked except where the arm says so. Streaming chat completions, 256 max tokens,
8 fixed prompts (`raw/bench.py`), usage-authoritative token counts. Loaded program: PP-2
(`split_at=22` plain, `23` with the DSpark tail reserve), matrix expert program, EP off,
host sampler.

| arm | cell | ok | decode tok/s (1/TPOT p50) | TPOT p50/p95/p99 ms | TTFT p50 ms | agg tok/s |
|---|---|---|---|---|---|---|
| plain | greedy c1 | 8/8 | 15.86 | 63.06 / 63.16 / 63.19 | 549 | 15.40 |
| plain | sampled c1 | 8/8 | 15.28 | 65.44 / 65.54 / 65.56 | 552 | 14.85 |
| plain | sampled c4 | 16/16 | 15.28 | 65.43 / 65.56 / 65.57 | 52,242 | 14.85 |
| plain | sampled c16 | 9/32 | 15.28 | 65.43 / 65.53 / 65.55 | 52,209 | 14.82 |
| DSpark | greedy c1 | 8/8 | 22.00 | 45.44 / 50.03 / 50.63 | 707 | 21.26 |
| DSpark | sampled c1 | 8/8 | 18.23 | 54.85 / 63.39 / 64.33 | 720 | 17.13 |
| DSpark | greedy c1 repeat | 4/4 | 21.92 | 45.61 / 50.41 / 50.79 | 710 | 20.49 |

Boot to `/readyz`: 135 s plain, 145 s DSpark. Peak power during decode: 157 W per card of 600 W.

The DSv4 route is serial (`route-contract ... capacity=serial`): c4 and c16 aggregate equal
c1 and the queue shows up as TTFT. At c16, 23 of 32 requests got 429 from the admission book
(#501). Concurrency on this route is a missing mechanism, not a tuning knob.

Receipts: `raw/base-plain-r1/`, `raw/spec-dspark-r1/` (cells.jsonl, per-request rows with text,
serve.log, /metrics snapshots between cells, telemetry).

## Correctness: served DSpark greedy != plain greedy

Greedy text sha256, same prompts, same binary:

| idx | plain | DSpark | DSpark repeat | first differing char |
|---|---|---|---|---|
| 0 | 6bd935354b38b161 | f5c0ccb131487b76 | f5c0ccb131487b76 | 168 |
| 1 | 6bd092ba5334120c | 8f30afa30dee412f | 8f30afa30dee412f | 260 |
| 2 | 65399f0a59f8d1e5 | 36fc18d286595138 | 36fc18d286595138 | 143 |
| 3 | 9d5751569fc92bb8 | 26ac772422d48b5b | 26ac772422d48b5b | 11 |
| 4 | 437790e6b2d71a7e | a0f8b5a9a6c771a6 | | 434 |
| 5 | 052f1203e9050edb | bec5e251ef56ee74 | | 530 |
| 6 | 846790995da33c6a | 4fd72390cf1cd748 | | 478 |
| 7 | a0efb32a48373fcf | cac851b27ff0c94f | | 114 |

DSpark is self-deterministic and differs from plain on every prompt. The DSpark numbers above
are therefore not a servable spec speedup yet. Tracked and fixed in memra #660: the HC24 dot
split S16 ran only on one-row calls, so a T=k+1 verify row took a different numeric class from
the same row decoded alone. The gate that should have caught it
(`dsv4-gpu-dspark-gate`) primed through the monolithic prefill the matrix program refuses
(`raw/dspark-gate-r1/gate.log`) and pinned the split off. Fix, bisect and the fixed-binary served
cells: `spec-identity-660/RESULTS.md`.

## Where the plain step goes

`dsv4_decode_profile`, naked plain decode at absolute positions 8224..8256, nsys over 32 steps
(`raw/prof/nsys-naked/ana.txt`):

- 77.41 ms/step wall. dev0 busy 37.29 ms, dev1 busy 33.63 ms; PP-2 runs them in series, so the
  step is GPU-bound with about 6.5 ms of gaps.
- `moe_kq_sktail_kernel` (routed experts, M=1): 13.97 ms dev0 + 12.26 ms dev1, 38% and 37% of
  each card.
- `dsv4_dense_fast_fp8_kernel`: 14.7% / 14.1%. `dsv4_hc_sinkhorn_m_kernel`: 7.3% / 7.1%
  (58 us per call). `dsv4_topk_idx_numeric_kernel`: 5.5%. `dsv4_sink_out_mq_f32acc_kernel`: 5.1%.
- 3,356 kernel launches per step, 381 D2D copies (13.1 MB).
- The EP-composed arm (`raw/prof/nsys-ep-composed/`) runs 62.4 ms/step
  (1.997542 s / 32 steps), 16% under the naked 74.3 ms/step (2.377706 s / 32) of the same
  profiler without nsys.

The two `.nsys-rep` captures stay machine-local (too large for the repo); their sha256 are in
`raw/prof/nsys-rep.sha256`, and `ana.txt` plus the `stats_*.csv` exports carry every number above.

The naked profiler run ends with `assertion failed: frozen sampled stream` (`raw/prof/naked/run.log`):
its frozen sha predates the matrix program. The timing line prints before the assertion.

## Roofline and the public anchor

Per-token weight bytes from the safetensors headers (`raw/roofline.py`, `raw/roofline.json`):
7.79 GB non-routed + 6/256 of 155.83 GB routed = **11.44 GB/token** (DSpark tensors excluded).
At the card's 1.792 TB/s nominal:

| topology | bound ms/token | bound tok/s c1 |
|---|---|---|
| PP-2 (cards in series) | 6.38 | 157 |
| TP-2 / EP-2 (cards in parallel) | 3.19 | 313 |

Measured plain c1 is 63 ms/token, 10.1% of the PP-2 bound.

Public floor to beat, same hardware shape (vLLM TP=2 on 2x RTX PRO 6000 Workstation, published
by Infatoshi): c1 decode 17.8 eager, 77.1 piecewise graphs, 94.8 tuned, 109.3
FULL_AND_PIECEWISE, 193.2 with the DSpark-style k=3 drafter, 202.7 with Marlin W4A16. 109.3 tok/s
is 9.15 ms/token, 35% of the TP-2 bound.

## Lever ranking from this profile

1. Routed MoE M=1 kernel: 38% of each card and far off bandwidth. A bit-identical kernel with
   more memory-level parallelism is the largest single lever.
2. Concurrency: the route is serial, so every c>1 number equals c1. Batched decode across
   requests is the largest aggregate lever.
3. DSpark served by default, after #660 restores spec == plain.
4. TP/EP served topology (#454): the EP-composed profile already runs 16% less time per step.
5. Fused Sinkhorn, dense FP8 and the small-kernel tail (launch count 3,356 per step).
6. Graph capture and the device sampler.
