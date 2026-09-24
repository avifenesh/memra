# TP/EP full-token replay past position 512 (memra #710)

Scope: one model, one hardware shape. `tiyuvta/DeepSeek-V4-Flash-0731-NVFP4@bafd09f8cab4f4f4f25e1cdafbcdefc05b90ee38`
on 2x RTX PRO 6000 Blackwell Server Edition, 2026-09-24. Program: the pinned plain TP2 attention,
expert-id EP full-token program (`validate_full_token_program`), device sampler, temperature 1.
Gate: `dsv4_tp_replay_long_gate`, lane `lane/dsv4-tp-graph-20260924` `1c23caccf`. Raw logs are in
`raw/tpreplay/` and `raw/tpreplay.pass1/`, and the script is `raw/q-v3p.sh`.

## What changed

The full-token replay (per-rank CUDA graphs of the whole TP/EP token, with three emission variants
and a commit graph) was armed with a 512-position cap in three places:

- `ws.replay_limit = 512`;
- arming refused `pos >= 512`;
- the replay step refused `pos >= 512`.

The replay indexer already scores up to 4,096 compressed blocks and reads the live count from the
device position, so the cap was a constant, not a kernel bound. Replay now arms for
`min(capacity, 16384)`, and arming refuses a score buffer too small for that. Arming needs
`pos < capacity` and the step refuses `pos >= replay_limit`. Arming still admits capacities of
512 to 1024 only, and the prefix restore still needs a prefix below 512. Both are lifted
separately.

## Correctness

One eager prefix to position 400 is restored into two states. The eager state steps and samples
through the unarmed program; the replay state decodes through the armed graphs. Each state
compares every step:

```
REPLAY_VARIANTS rank0=[228, 304, 74, 2] rank1=[228, 304, 74, 2] (ordinary, commit, c4, c4+c128) captures=[3, 1]
PASS: 304 replayed steps from position 400 to 704 bit-identical to eager (token, logits, cache and hidden digests); tokens_sha256=112c2fc66fe1f76029c4ff1a8ce3446af91c1570df6594bce9311533650f02ce
```

Every one of the 304 steps was a replay on both ranks: 228 ordinary, 74 C4 emissions and 2 C4
plus C128 emissions. The run crosses 512 and the C128 emission at position 639.

Two earlier runs are kept as receipts of the bounds they hit:
- `q-v3p.summary.void1`: the eager arm used the combined replay API, which refuses unarmed
  states.
- `q-v3p.summary.void2`: the replay step refused position 512 on the second hard-coded bound.
  Every step from 400 to 511 had matched eager before that.

## Timing (same continuation, alternating order, wall time per token)

| rep | eager ms/token | replay ms/token | eager tok/s | replay tok/s | speedup |
|---|---|---|---|---|---|
| 0 | 21.218 | 17.662 | 47.13 | 56.62 | 1.201 |
| 1 | 20.822 | 17.662 | 48.03 | 56.62 | 1.179 |
| 2 | 20.821 | 17.356 | 48.03 | 57.62 | 1.200 |

Replay serves about 57 tok/s, against 48 for the eager TP/EP program on the same pair. PP-2 plain
serves about 51.8 tok/s on this pair (`../graph-step/stage0/`).

## Where the replayed step's time goes

The kernels inside the graphs are the eager step's kernels. The eager `tp_ep_attn` anatomy
(`../tpep-serve/raw/tpep3/prof/tpa/ana.txt`, 510 steps) puts each card at 13.40 to 15.36 ms of
device busy per token:
- FP8 dense GEMV: 3.13 ms, 496 launches.
- MoE stream: 1.80 ms, 130 launches.
- All-reduce: 3.18 ms on dev0 against 0.69 ms on dev1. dev0 waits on dev1, the head rank.
- Dots: 1.10 and 1.75 ms.
- About 4 ms of small kernels: HC, sink attention, gathers, packs, norms, rope.

Replay at 17.4 ms is close to that device time, so the TP step is no longer launch-bound. The rest
of the gap to the TP-2 bandwidth bound (about 3.7 ms per token at 1.54 TB/s per card) is kernel
work: the half-width dense projections, the small-kernel chains, and the head-rank imbalance behind
dev0's all-reduce wait.
