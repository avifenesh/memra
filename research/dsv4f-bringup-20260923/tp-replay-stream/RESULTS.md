# TP/EP full-token replay on the served one-token stream MoE (memra #710)

Scope: one model, two pair variants. `tiyuvta/DeepSeek-V4-Flash-0731-NVFP4@bafd09f8cab4f4f4f25e1cdafbcdefc05b90ee38`
on 2x RTX PRO 6000 Blackwell Workstation Edition (`raw/ws/`) and 2x RTX PRO 6000 Blackwell Server
Edition (`raw/se/`), 2026-09-24. Program: the pinned TP2 attention, expert-id EP full-token
program with the device sampler at temperature 1. Lane `lane/dsv4-tp-replay-stream-20260924`
`8dee5e824`, gate `dsv4_tp_replay_long_gate`.

## What changed

Full-token replay admitted exactly one expert program, the gate-only graph split-K set: five
`*_for_gate` toggles (GU fuse, M1 tensor-core, GU M1 tensor-core, GU half2, down M1 half2), none of
which a serving process sets. The served TP/EP step runs the #664 one-token stream visitor instead.
Replay now admits either program whole: the stream visitor with every split-K arm off, or the
split-K set with all five on. A mix is refused. `DSV4_REPLAY_GATE_MOE=stream|sktail` picks which
one the gate runs; the default is `stream`.

## Correctness

On both pairs, both programs:

```
REPLAY_VARIANTS rank0=[228, 304, 74, 2] rank1=[228, 304, 74, 2] (ordinary, commit, c4, c4+c128) captures=[3, 1]
PASS: 304 replayed steps from position 400 to 704 bit-identical to eager (token, logits, cache and hidden digests); tokens_sha256=112c2fc66fe1f760...
```

That is the same tokens sha as the split-K run of `../tp-replay-long/`. Stream and split-K give the
same tokens on both pairs, which is #664's kernel-boundary identity carried through the captured
graphs.

## Timing (same continuation, alternating order, wall time per token, N=3)

| pair | MoE | eager tok/s | replay tok/s | replay ms/token |
|---|---|---|---|---|
| WS | stream | 62.03 / 62.03 / 62.03 | **80.87 / 80.86 / 80.74** | 12.37 |
| WS | split-K | 52.31 / 52.30 / 52.29 | 64.22 / 64.16 / 64.10 | 15.59 |
| SE | stream | 56.74 / 57.78 / 57.79 | **71.55 / 71.68 / 72.00** | 13.94 |
| SE | split-K | 47.09 / 47.63 / 47.86 | 55.70 / 56.43 / 56.62 | 17.72 |

The stream visitor makes replay 1.26x faster on WS and 1.27x on SE, with the same tokens. It beats
replayed split-K by 16.6 and 15.4 tok/s.

Replayed TP/EP on WS serves 80.8 tok/s against 68.4 for current-main PP-2 plain on the same pair
(`../ceiling/raw/now-ws/`), 1.18x. That is the first c1 measurement where the TP-2 program is clearly
ahead of PP-2. It is still gate-only: replay admits capacities of 512..=1024, vendor-default
sampling and no drafter (#710).
