# DSv4-Flash on 2x RTX PRO 6000: bounds, achieved, and the gap (2026-09-24, updated 2026-09-25)

Scope: one model (`tiyuvta/DeepSeek-V4-Flash-0731-NVFP4@bafd09f8cab4f4f4f25e1cdafbcdefc05b90ee38`), one
hardware shape (2x RTX PRO 6000 Blackwell). Every number cites its receipt under `raw/` or a sibling
lane directory.

## Bounds

- **Bytes per token.** 7.79 GB non-routed plus 6/256 of 155.83 GB routed makes **11.44 GB**
  (`../raw/roofline.json`). Since #706 the engine streams the checkpoint's own BF16 compressor
  weights, so it reads exactly that.
- **Practical bandwidth.** A 128-bit streaming read gets **1.54 TB/s** per card at 1 to 4 GiB and
  1.43 TB/s at 64 MiB, against the spec sheet's 1.79 TB/s. This is a single run on one Server
  Edition card (`raw/bw/`).
- **c1 bounds at that bandwidth:**
  - PP-2 runs the two stages in series for one request, so a token costs all 11.44 GB on one card's
    bandwidth: 7.43 ms, **135 tok/s**.
  - TP-2 streams both halves in parallel: 3.71 ms, **270 tok/s**, before its all-reduces.
  - Two or more pipelined PP-2 lanes: each card streams its half per token on a different request,
    **270 tok/s** aggregate.

These are weight-streaming bounds. They leave out attention over the cache and assume perfect
overlap, so they are ceilings, not targets.

## Achieved: PP-2 on main `2ad82c2e9` (WS pair, served, one boot per row, N=3)

`raw/now-ws/`:

| route | greedy c1 | sampled c1 | of the PP-2 c1 bound |
|---|---|---|---|
| plain | 68.39 tok/s | 59.93 | 51% |
| DSpark, full-depth ladder | 70.54 | 56.80 | n/a |
| DSpark, per-slot window (`MEMRA_DSV4_VT=slot`, #709) | 75.59 | 63.20 | n/a |

The day's merges moved plain from 53.5 tok/s (#706's base) to 68.4:
- #704, fused one-token MoE;
- #715, commit cleanup;
- the HC and sink fusions.

DSpark on the full ladder no longer beats plain on sampled traffic. The per-slot window restores its
lead, at +7.2% greedy and +11% sampled over the ladder.

**TP/EP is the served default since #710 (2026-09-25).** Served, one boot per row, N=3, same text
on every request of every arm (`../tpep-default/RESULTS.md`, `raw/tp-replay-served-ws/`):

| route | WS pair greedy / sampled c1 | SE pair greedy / sampled c1 | of the TP-2 c1 bound |
|---|---|---|---|
| plain, TP/EP on the full-token replay | 80.73 / 80.17 | 71.20 / 72.65 | 30% (WS) |
| plain, PP-2 | 68.51 / 59.96 | 62.38 / 58.40 | 51% of the PP-2 bound (WS) |
| DSpark, TP/EP | 82.73 / n/a | 72.79 / 63.57 | n/a |

At c2 and above, the plain route loses on TP/EP:
- PP-2 pipelines two lanes: 120.9 tok/s aggregate on WS.
- TP/EP serves one lane: 76.9.

The TP/EP B-row step closes that (lane `lane/dsv4-tp-rows-20260925`).

## Where a token goes

**PP-2 plain, WS pair** (`raw/now-ws/now/prof-plain/ana.txt`, nsys, 510 steps, 15.52 ms per step
under capture against 14.62 ms TPOT served):

| part | dev0 | dev1 | note |
|---|---|---|---|
| FP8 dense GEMV (`dense_fast_fp8`) | 2.18 ms | 2.09 ms | attention projections and the shared expert |
| fused MoE, gate/up and down (#704) | 1.67 ms | 1.59 ms | 3.65 GB, about 1.12 TB/s, 73% of practical |
| dots (compressor, head on dev1) | 0.37 ms | 1.04 ms | the 129k-vocab head sits on dev1 |
| small chains (HC finish, sink, route, indexer, rope, packs) | about 1.7 ms | about 1.7 ms | |
| kernel sum | 5.95 ms | 6.45 ms | 12.4 ms for 11.44 GB, 60% of practical on average |
| no kernel on either card | about 3 ms | | launches (1,878 per step) and the step-end readbacks |

**TP/EP replay on the served stream MoE with the checks on, Server Edition pair**
(`raw/tp-anatomy-se-stream/tpanat2/replay/ana.txt`, 304 replayed steps, nsys graph-node tracing).
The profile mode is `DSV4_REPLAY_GATE_PROFILE`. A step takes 14.07 ms:
- dev0 is busy 12.42 ms;
- dev1 is busy 13.10 ms.

| part | dev0 | dev1 | note |
|---|---|---|---|
| FP8 dense GEMV (`dense_fast_fp8`, 365 launches) | 3.13 ms | 3.12 ms | the half-width attention and shared-expert shapes |
| one-token MoE stream visitor (`moe_kq_m1_stream`, 129 launches) | 1.93 ms | 1.97 ms | each card's 128 experts |
| dots (compressor; the 129k-vocab head on dev1) | 0.88 ms | 1.51 ms | the head costs dev1 about 0.6 ms that dev0 idles |
| attention-TP row gathers (`tp_ar_gather_rows`) | 1.05 ms | 1.03 ms | two per layer |
| expert all-reduce (`tp_ar_1stage`) | 0.82 ms | 0.80 ms | one per layer |
| HC finish, FP8 gather, sink, indexer, route, norms, rope | about 2.9 ms | about 2.9 ms | small chains, latency-bound |

The earlier split-K replay (`raw/tp-anatomy-se/`, 17.62 ms per step) spent 5.8 ms per card in
`moe_kq_sktail_gu` plus `moe_kq_sktail`. The stream visitor does that work in 1.9 ms.

## Levers taken, 2026-09-26

`../levers-20260926/`, SE pair, greedy c1 aggregate 67.53 to 77.19 tok/s (decode p50 80.95), every
gate bit-identical:
- programmatic dependent launch: +4.99%;
- vocab-parallel head: +2.44% (lever 3 below);
- one kernel for the compressor snapshots: +0.94%;
- push joins: +7.75% (lever 4 below, the both-idle barrier half, larger than the instrument's
  ceiling predicted because the ranks no longer march in lockstep).

## The gap, by lever, largest first

1. **Concurrency on TP/EP.** The B-row step: several requests' rows in one TP step, each weight
   read once for all of them. Lane `lane/dsv4-tp-rows-20260925`.
2. **Dense FP8 GEMV at the half shapes.** 3.1 ms of the 13.1 ms dev1 step. Those bytes stream at
   well under the MoE visitor's rate.
3. **The head.** A vocab-parallel head, with rows split exactly across the ranks, would move
   about 0.6 ms off dev1 and balance the two cards.
4. **The collectives.** About 1.85 ms per card of row gathers and the expert all-reduce: 43 layers
   times three one-shot collectives. Fewer or fused joins per layer would cut it.
5. **Small chains.** About 2.9 ms per card, a latency floor per layer.
6. **Context under TP/EP.** Not a speed lever. Closed for plain by the position-split C4 store
   (`../kv-split/`): 1M plain, 500k DSpark, at 0.4% to 2.0% decode. The decode cost is the
   remote row reads; an owner-push of the rows both ranks know are selected would trade them for
   posted writes plus one join per C4 layer.
7. **Prefill.** About 360 tok/s at 8k prompts on the exact CUDA-core tiles (#713). The owner ruled
   PP-2 is not the target, so prefill work follows the TP program.

## What a realistic ceiling looks like

Take 85% of practical bandwidth on every weight byte, plus the measured small-kernel floor:
- **PP-2 plain c1:** 11.44 GB / (0.85 x 1.54 TB/s) is 8.7 ms, plus about 1 ms of latency-bound
  chains, about 9.7 ms: **about 100 tok/s**. Achieved 68.4, 68% of it.
- **TP-2 plain c1:** about 6 GB per card is about 4.6 ms, plus about 1.7 ms of all-reduce and gathers,
  plus about 1 ms of chains, about 7.3 ms: **about 135 tok/s**. vLLM's published 109.3 tok/s on this
  shape is TP=2 with full graphs, inside that ceiling. memra serves TP/EP at 80.7 tok/s on the WS
  pair (60% of it) and 71.2 on the SE pair.
