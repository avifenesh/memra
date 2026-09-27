# DSpark under TP/EP: expert-id halves of the drafter's routed bank (memra #718)

Scope: one model, one hardware shape. `tiyuvta/DeepSeek-V4-Flash-0731-NVFP4@bafd09f8cab4f4f4f25e1cdafbcdefc05b90ee38`
on 2x RTX PRO 6000 Blackwell Server Edition (gates, `raw/`), 2026-09-24. Lane
`lane/dsv4-dspark-ep-20260924`.

## Why

Under TP/EP (exact attention TP2 plus the trunk's expert-id EP pair), the DSpark drafter loaded
whole onto rank 1, the head rank, and the served TP/EP DSpark boots ran out of memory at load
(`../tpep-serve/RESULTS.md`). A census of the checkpoint's `mtp.*` tensors (`raw/mtp_census.py`)
finds the three MTP blocks at 10.86 GB, and **10.27 GB of that is routed experts** (3 blocks x 256
experts). Attention is 0.32 GB, the Markov head 0.13, the shared expert 0.08, and the rest 0.07.

## What changed

- **Loader.** Under TP/EP each drafter block loads on both ranks through the trunk's partitioning
  loader. Rank 0 holds expert ids 0..128 and rank 1 holds 128..256, each with a local-only
  `EpLayer` (`dspark_local_bank`), so `LayerDev::expert_ptrs` refuses any peer id. Rank 1 keeps
  everything else. The fused drafter MoE (`MEMRA_DSV4_DSPARK_FUSED_MOE`) indexes the bank by global
  id, so it refuses the boot under TP/EP.
- **Forward.** `dspark_moe_tp_ep` replaces `moe_forward` for the drafter's blocks under TP/EP:
  - Rank 1 routes on the host exactly as before and quantizes x once.
  - Rank 0 receives the same FP8 codes and scales by peer copy.
  - Each rank runs the native per-expert program, now shared through `moe_native_expert_loop`, for
    the experts it owns, into a zeroed slot plane `[rows x topk, hidden]`.
  - The trunk's one-shot all-reduce joins the planes. Each slot has one owner, so each joined slot
    is `c + 0`.
  - Rank 1 sums every row in ascending expert id (`combine_rows_m`), which is the order
    `moe_forward` scatters in, then adds the shared expert.
- **Transactions.** Under TP/EP the drafter forward holds the TP/EP walk lock, taking it before the
  reduction mutex as the trunk does, and reads both refusal words before a proposal leaves.
- **Also.** The drafter's `main_x` GEMV and its tap copy feed only the capture outputs, so a plain
  round now skips them. PP-2 and TP/EP both benefit, and no output bits move.

## Correctness

`dsv4-gpu-dspark-gate`, with a new per-round proposal digest: the draft ids and fp32 confidence
bits of every round, for the sequential arm and both batched arms. It ran on two trees:
- control: main plus only the digest (`d7e79f5a3`), with the whole drafter on the last stage;
- lane: `acd3986b6`.

Each tree ran on PP-2 and on TP/EP with exact attention TP (`raw/q-v3r.summary`,
`raw/dsparkep/*/gate.log`):

| tree, topology | proposal sha (DS / DB0 / DB1) | rank 0 used | rank 1 used |
|---|---|---|---|
| control, PP-2 | 62b368f1 / d404da5f / d404da5f | 84.42 GiB | 84.51 GiB |
| lane, PP-2 | 62b368f1 / d404da5f / d404da5f | 84.42 GiB | 84.51 GiB |
| control, TP/EP (whole drafter on rank 1) | 62b368f1 / d404da5f / d404da5f | 81.73 GiB | 92.39 GiB |
| lane, TP/EP (split drafter) | 62b368f1 / d404da5f / d404da5f | 87.26 GiB | 87.61 GiB |

- The drafter's proposals and confidences are bit-identical across all four runs:
  - lane against control on PP-2, which covers the shared expert loop and the `main_x` skip;
  - the split drafter against the whole drafter on TP/EP;
  - both against PP-2.
- The split moves 4.78 GiB off rank 1, the card that binds.
- Every run keeps its other verdicts:
  - batched equals sequential (77 logit rows, 3,206 cache-class comparisons, all bit-identical);
  - ring writes: 0 mismatches;
  - determinism: identical tokens, accepts and rings.

The two TP/EP runs each reported one finding, a gate expectation rather than a program fault. The
gate expected fused one-token MoE dispatches whenever the fused pair was on. TP/EP layers run the
expert-id EP path, which never reaches it. The gate now expects zero fused dispatches on TP/EP
(`09ca39c92`).

## Served (WS pair, measurement branch)

The served A/B needs a load-time TP/EP selector, which main does not carry (it was decided out
before merge, `docs/FLAGS.md`). It ran on `lane/dsv4-tpep-serve-measure-20260924` `d1be33841`:
this lane plus that selector, never merged. One boot per row, serial route, order
Pp Tp Pd Td Td Pd Tp Pp Pp Tp Pd Td (N=3 each), cells `cells-spec.txt`
(`raw/served-ws/`). The same binary first passed `dsv4-gpu-dspark-gate --tpep` with the fixed
expectation: `GPU DSPARK GATE [PASS]`.

| arm | greedy c1 | sampled c1 | TTFT p50 |
|---|---|---|---|
| PP-2 DSpark (`Pd`) | 70.60 | 56.85 | 235 ms |
| **TP/EP DSpark, split drafter (`Td`)** | **82.73 (+17.2%)** | **65.42 (+15.1%)** | 189 ms (-19.7%) |
| PP-2 plain (`Pp`) | 68.39 | 59.89 | 182 ms |
| TP/EP plain, eager (`Tp`) | 62.73 (-8.3%) | 55.71 (-7.0%) | 141 ms (-22.4%) |

- **Text:** every request's hash is identical across all twelve rows, TP/EP and PP-2 alike,
  greedy and sampled. DSpark text equals plain text on both topologies.
- **Fit:** the drafter fits at the served context. The TP/EP DSpark boot calibrates at
  `max_seq` 1,048,576 with a ceiling of 10.03 and 9.70 GB and a fixed spec cost of 4.53 and
  4.56 GB. That leaves rank 1 about 5.1 GB for one session's cache, about 370k tokens at
  13,806 B/token. PP-2 DSpark on the same pair has 11.47 GB, 4.12 GB fixed, and 7,223 B/token,
  about 1.0M tokens. The difference is TP/EP's replicated caches, not the drafter.
- **Plain:** TP/EP plain is slower than PP-2 on the eager step. The full-token replay graphs close
  that gap: 80.8 tok/s against 68.4 on this pair in the replay gate (`../tp-replay-stream/`).
