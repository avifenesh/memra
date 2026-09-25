# TP/EP as the served two-card default (memra #710, 2026-09-25)

Owner ruling 2026-09-25: "flip and continue to head split". Then, given the concurrency cost
below: "Flip now, B-row next".

Model: `tiyuvta/DeepSeek-V4-Flash-0731-NVFP4@bafd09f8cab4f4f4f25e1cdafbcdefc05b90ee38`.

## What changed

`memra-server` now loads DSv4 on two cards as TP/EP by default:
- every layer sits on both cards;
- experts are split by id (0..128 on card 0, 128..256 on card 1);
- attention is split by head (exact attention TP2);
- the drafter's experts are split the same way (#720).

`MEMRA_DSV4_TOPOLOGY=pp` is the rollback to PP-2.

On TP/EP:
- The plain sampler defaults to the device sampler (`MEMRA_DSV4_SAMPLER` still overrides).
- Greedy and vendor-default sampled plain requests replay their steps on the full-token graphs
  (#719, #721), up to position `min(capacity, 16384)`.
- Past that position the request drops its graphs and continues on the eager step, the same
  numeric program.

Lane: `lane/dsv4-tpep-default-20260925`. Served binary `9d19b5014` (server sha256 prefix
`0e7d5e6f7f69fa7b`). Gate binary `275f83a75`.

## Served A/B

Protocol:
- One boot per row.
- Serial queue under `/tmp/memra-gpu.lock`.
- 250 ms GPU telemetry in every row directory.
- Decode tok/s is the per-request p50, and the table shows the median over the rows.
- `agg` is the cell's aggregate completion tok/s.

### Server Edition pair (`raw/se/`, rows r1..r11, order Df Pp Pp Df Df Pp, then Hc, DsDf DsPp DsPp DsDf)

| cell | TP/EP default (N=3) | PP-2 `pp` (N=3) | delta |
|---|---|---|---|
| greedy c1 decode | 71.20 (70.77..71.49) | 62.38 (62.36..62.38) | +14.1% |
| sampled c1 decode | 72.65 (71.49..72.70) | 58.40 (58.38..58.49) | +24.4% |
| greedy c1 ignore-eos decode | 72.97 | 63.61 | +14.7% |
| TTFT p50, c1 greedy | 172 ms | 210 ms | -18% |
| greedy c2 aggregate | 69.78 | 113.02 | -38.3% |
| sampled c2 aggregate | 69.51 | 105.80 | -34.3% |
| greedy c2 TTFT p50 | 3839 ms | 330 ms | queued behind the running request |
| greedy c1, 2k context: decode / TTFT | 68.64 / 6061 ms | 61.34 / 6012 ms | +11.9% / flat |

DSpark route (`MEMRA_DSV4_DRAFTER=dspark`), N=2 per arm:

| cell | TP/EP | PP-2 | delta |
|---|---|---|---|
| greedy c1 decode | 72.79 | 61.72 | +17.9% |
| sampled c1 decode | 63.57 | 53.90 | +17.9% |
| TTFT p50 greedy c1 | 207 ms | 267 ms | -22% |

Parked host cache on (`MEMRA_DSV4_KV_HOST_MB=16384`, row r7, N=1): the ignore-eos cell repeats
the greedy cell's first four prompts. It serves at the default's rates: 71.48 greedy, 72.95
ignore-eos. No replay arm was refused on any row (`TP/EP replay not armed` count 0).

### Workstation pair, concurrency (`raw/ws-conc/`, measurement branch 758b05089, N=3 per arm)

| cell | TP/EP (one lane) | PP-2 (two pipelined lanes) |
|---|---|---|
| greedy c1 decode | 80.73 | 68.51 |
| greedy c2 aggregate | 76.94 | 120.92 |
| greedy c4 aggregate | 76.64 | 120.50 |
| sampled c2 aggregate | 76.44 | 107.19 |
| greedy c2 TTFT p50 | 3484 ms | 301 ms |

The earlier c1 A/B on the same pod (`../ceiling/raw/tp-replay-served-ws/`, N=3) put:
- TP/EP at 80.69 greedy and 80.17 sampled;
- PP-2 at 68.40 greedy and 59.96 sampled.

### Text identity

Every request's text sha is equal across every arm of every cell, sampled cells included. That
covers TP/EP, PP-2, TP/EP with the host cache, and DSpark on both placements. TP/EP draws with
the device sampler, PP-2 with the host sampler, and they gave the same tokens on this prompt set.

## The cost, stated

- **Concurrency on the plain route.** TP/EP runs one serving lane, because its steps do not
  pipeline across requests. PP-2 runs two, one per card, since #667. So plain c2+ aggregate is
  34-38% lower on TP/EP and TTFT under load is about ten times worse. The DSpark route is serial
  on both placements (a spec round holds the launch turn), so TP/EP wins it at every concurrency.
  Fix: a TP/EP B-row step that decodes several requests in one TP step (owner: next).
- **Context.** TP/EP keeps every layer's KV cache on both cards. A session holds about 370k
  tokens with DSpark and 790k plain, against PP-2's 1M (`../dspark-ep/RESULTS.md`). Fix: the
  head-split KV lane.

## Replay-to-eager handoff (`raw/se/tpep-default/gate-*`)

`dsv4_tp_replay_long_gate` on the Server Edition pair, gate binary `275f83a75`. Each run restores
one eager prefix to position 400 and compares, on every step:
- the replayed state against the eager state;
- the token, the logits bits, and the TP/EP cache and hidden digests.

| row | steps | replayed | handoff | verdict |
|---|---|---|---|---|
| `plain-304` (capacity 1024) | 304 | 304 | none | PASS |
| `handoff-greedy` (capacity 2048, `DSV4_REPLAY_GATE_LIMIT=640`) | 500 | 240 | at 640 | PASS, then eager to 900 |
| `handoff-default` (vendor sampling, same limit) | 500 | 240 | at 640 | PASS, then eager to 900 |

The first `9d19b5014` runs of the two handoff rows (`gate-9d19b5014.log`) died after their step
loop: the gate read the replay capture count after the handoff had dropped the graphs. That was
a gate bug, and `275f83a75` reads it before the drop.

The served limit itself (capacity 20000, 16100 steps, handoff at 16384) first ran with the
digests on every step. At that capacity the digests read tens of megabytes of cache per step, so
the run was stopped after 20 minutes without a verdict (`gate-275f83a75-stopped.log`).
`6695e7c0b` adds `DSV4_REPLAY_GATE_DIGEST_EVERY`: the token and logits bits are checked every
step, and the digests every 256 steps and on the 64 steps either side of the handoff. That run
is `raw/ws-limit/`.

That run (Workstation pair, gate `6695e7c0b`, `raw/ws-limit/`) PASSES. It ran 16100 steps from
position 400 to 16500, bit-identical to eager. 15984 steps were replayed, then the run continued
eager from the handoff at 16384. Replay variants per rank: 11988 ordinary, 3871 C4, 125 C4+C128.
`tokens_sha256=0c17e11b42db18583a2f07309c099719d13c796340d1f00a30d37db7f60c8be8`.
