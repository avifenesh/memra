# DSv4-Flash on 2x RTX PRO 6000: bounds, achieved, and the gap (2026-09-24, updated 2026-09-28)

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
- push joins: +7.75% (lever 4 below). That is more than the 2026-09-10 AR instrument's ceiling
  for removing both barriers (1.9% of that program's token); what the extra comes from is not
  measured yet.

## Levers, 2026-09-27

Source: `../levers-20260927/`.
- **Adopted: the fused MoE pair on the TP/EP partition.** Measured on the second SE pair, greedy
  c1 77.48 to 85.16 tok/s (decode p50 89.32), +9.91%, bit-identical.
- **Refuted: weight loads ahead of the PDL wait** (-1.8%).
- **Refuted: a deeper MoE stream ring** (+1.6% and +2.3% ms per token).
- **Adopted: the shared expert on the rank with fewer routed slots.** The other rank skips it and
  the expert join carries its rows (second SE pair). Rebased on main, greedy c1 goes from 91.18
  to 94.66 tok/s (decode 99.1), +3.7%, and the long gate from 10.65 .. 10.77 to 10.17 .. 10.27
  ms/token. Bit-identical.
- **Adopted, merged earlier the same day:**
  - sibling projections in one launch;
  - register-resident row kernels;
  - the HC finish's Sinkhorn projection on a fifth warp, c1 +2.0% (`../hc-finish/`);
  - the prefill tile's register diet, TTFT -18.9% at 8k (`../prefill-tile/`).
- **Refuted:**
  - the h mirror published once per slot (-0.7%);
  - register-held mirror groups (+1.8% ms per token; the gate/up kernel goes from 47 to 57
    registers);
  - an L2 weight prefetch before the PDL wait (flat).

The step census after the Sinkhorn warp (`../levers-20260927/raw/se-census-v6q/`, replay with
PDL off, first SE pair) is 11.21 ms captured, 10.42 ms of kernels per rank and 2,118 launches per step:
- dense FP8 GEMV, single and pair: 2.96 ms;
- the fused MoE pair: 1.98 ms;
- dots: 1.05 ms;
- joins: 1.15 ms;
- HC finish: 0.40 ms;
- sink attention: 0.56 ms;
- 1,235 launches under 3 us: 1.44 ms together.

With PDL on the same replay is 10.70 ms.

At four B-row requests (`../levers-20260927/raw/se-census-rows-v6r/`, the captured B-row step,
PDL off) a step is 23.9 ms against 13.3 at one row, over 5,405 launches:
- the fused MoE pair: 5.5 ms, from the union of the rows' experts;
- dense GEMV: 4.4 ms;
- per-row attention launches that one row does not pay: sink, indexer, compressor pool, and the
  replay row copies.

Per added row that is about 1,150 launches, most of them per-row-group kernels. One launch over
all the row groups would remove most of them.

The plain step after the fused pair, with PDL off: 11.3 ms of kernels per step. That breaks
down as:
- dense FP8 GEMV 3.13 ms;
- the fused MoE pair 2.01 ms, at about 0.9 TB/s;
- dots 1.17 ms;
- joins 1.16 ms, most of the expert reduce being one rank waiting for the other;
- about 3 ms of small latency-bound kernels. HC finish alone is 0.63 ms: two single-block calls
  per layer at 7 us each. After it come sink attention, the router, the q norm pack, HC split
  dots and the indexer.

**Expert placement is free in this numeric class.** The rank-order expert sum adds a slot's value
on its owning rank to the other rank's cleared +0.0. A fused or chain contribution starts its
accumulator at +0.0 and cannot become -0.0, so `x + 0.0 = x` exactly. Which rank computes a slot
therefore does not change a bit.

Splitting every expert's rows across both ranks (TP inside the experts) would also keep each
output element's dot on one rank. That removes the per-token imbalance the reduce waits on. The
expected max of a Binomial(6, 0.5) split is 3.94 experts against 3, 31% more MoE time on the
critical path. The cost is one more small join per layer, for the intermediate.

## Close, 2026-09-27

Source: `raw/close-se-v6y/`. First SE pair, served, one boot per row, 8 requests per cell unless
noted. B is main `359e850d0` from the morning, F is main `80f734c77` from the evening. In between
landed the Sinkhorn warp, the prefill tile, the shared expert's owner, 16 lanes, the joined MoE
tail, the attention fusions and the router's x mirror. Every text both arms served is identical:
40 of 40 on the c1-c4 cells and 76 of 76 on the c4-c24 cells.

| cell | B | F | change |
|---|---|---|---|
| greedy c1 | 89.20 / 89.10 (TPOT 10.6 ms) | 97.10 / 97.26 (TPOT 9.8 ms) | +9.0% |
| sampled c1 | 89.84 / 89.87 | 97.96 / 97.97 | +9.0% |
| greedy c2 | 121.64 / 121.56 | 130.18 / 130.28 | +7.1% |
| greedy c4 | 155.75 / 155.89 | 165.98 / 165.27 | +6.3% |
| greedy c2, 1500-word context | 21.76 / 21.76 (TTFT 9.19 s) | 25.95 / 25.94 (TTFT 7.45 s) | +19.2% |
| greedy c8, 16 requests | 153.79 | 206.15 | +34% |
| greedy c16 | 154.17 (TPOT 24.3 ms) | 202.72 (TPOT 73.6 ms) | +31% |
| greedy c24 | 128.06, 20 of 24 served | 201.69, 24 of 24 | |
| sampled c8 | 152.45 | 195.32 | +28% |
| TTFT, 8k prompt | 17.70 s | 14.31 s | -19.1% |
| TTFT, 32k prompt | 73.93 s | 61.01 s | -17.5% |

Long gate, order B F F B: 9.84 .. 9.97 ms per token against 10.70 .. 10.81, -7.9%, the same
`PROGRAM_SHA256` in every run.

Two things set the c8 to c24 rows. B served one lane at a time. F runs 16 lanes in one B-row step,
but captured B-row steps stopped at 8 rows, so a wider step walked eagerly: 73.6 ms at 16 rows,
bound by the host issuing about 20,000 launches. The B-row diet lane (`../levers-20260927/`)
takes a captured step to 16 rows and turns each per-row attention and compressor stage into one
launch for every row.

## The step's floors (2026-09-27)

Source: `raw/floors-se-v7a/`. First SE pair, the replayed one-row step of main `80f734c77`'s
program (the build also carried a greedy argmax change the long gate found flat and that was
dropped).

**Launch floor.** The long gate's floor mode (`DSV4_REPLAY_GATE_FLOOR=1`, `docs/TESTING.md`)
launches the captured forward and commit graphs of both ranks 200 times back to back with no host
step. It then launches clones with every kernel node replaced by an empty kernel on the node's
own grid and block, with edges, copies and memsets unchanged, and clones that keep only the 129
cross-rank joins. Three rotating reps, ms per step:

| clone | PDL on | PDL off |
|---|---|---|
| captured graphs | 9.47 .. 9.59 | 9.96 .. 10.02 |
| every kernel empty (1,709 kernel nodes per rank in the forward graph) | 1.42 .. 1.46 | 1.69 .. 1.71 |
| empty except the joins | 1.64 | 2.23 |

A node costs about 0.85 us of launch and dependency time with PDL, and the joins' handshakes add
about 0.19 ms per step.

**Byte floor.** nsys GPU metrics (`gb20x` set, 20 kHz) over 64 replayed steps. The graphs are busy
98% of the 641.8 ms span:
- DRAM read averages 41.6% of the metric's peak on rank 0 and 39.6% on rank 1;
- SMs are active 67% and 66% of the time;
- SM issue slots are 16% used;
- the tensor pipes are active under 2%.

The metric's 100% is not the 1.79 TB/s of the spec sheet. The streaming probe (`bw_read.cu`)
reads 4 GiB at 1.537 TB/s and shows 95.9%, so 100% is about 1.60 TB/s, the memory clock's 12,481
MHz on a 512-bit bus. At that scale a step reads 6.68 GB on rank 0 and 6.37 GB on rank 1. The
tensor census gives about 6.0 to 6.2 GB of weights per rank, and the KV rows, activations and
scratch make up the rest. The average read rate over the step is 0.67 TB/s, 43% of the practical
1.54 TB/s.

**Census after the B-row lanes** (`raw/census-se-v7d/`, main with #906 and #909, PDL off,
nsys graph-node tracing). The one-row replayed step is 10.24 ms captured, 9.53 ms of kernels on
rank 0, and 1,754 launches:
- dense-fast FP8 GEMVs, single, pair and gated: 2.74 ms;
- the fused MoE pair: 1.82 ms;
- dots: 1.05 ms;
- joins: 0.88 ms;
- HC finish: 0.40 ms;
- sink attention: 0.54 ms;
- the rest is the small chains.

At 16 rows (`raw/census-se-v7d/rows-b16/`) a captured step is 37.6 ms over 1,696 launches, as
many as 4 rows take. The fused MoE pair over the union of the rows' experts is 14.4 ms, dense-fast
GEMVs and dots 12.0 ms, and the expert join 3.4 ms.

**The ceiling of this program.** Streaming 6.68 GB at 1.54 TB/s takes 4.34 ms. With the 1.64 ms
launch-and-join floor on top, a step that ran every kernel at practical bandwidth would take
about 6.0 ms, 167 tok/s of device time. Overlapping launches with streaming could shave that
somewhat, but not past the 4.34 ms of bytes. The captured step takes 9.5 ms of device time, 97
tok/s served with the host step. The 3.5 ms between the two sits in three places:
- streaming kernels that run well under practical bandwidth: short GEMVs pay their ramp, and the
  fused MoE pair is bound by its MMA numeric class;
- the work of about 1,200 kernels under 3 us, beyond their launch cost;
- the reduce's wait for the slower rank.

vLLM with b12x kernels reaches 130.8 tok/s plain on the same card shape, 7.6 ms per token, on a
different numeric program (`../PRIOR-ART.md`). That is between this program's step and its floor.
Moving the floor itself takes fewer graph nodes, at 0.85 us each, or a persistent program that
keeps a layer's kernels resident. The second is a large lane and an owner call.

## The gap, by lever, largest first (rewritten 2026-09-28)

The 2026-09-25 list's first four levers have landed:
- the B-row step, then the B-row diet (#906) and dense-fast to 16 rows (#909);
- the vocab-parallel head (#783);
- the push joins (#782), with the owned-row expert join measuring.

What is left, measured on the one-row replayed step unless noted:

1. **Concurrency past 16 rows and the 16-row step itself.** After #906 and #909 a 16-row
   captured step takes 37.4 ms, 427 tok/s, against 68.3 ms on main `80f734c77`. Served c16 goes
   from about 200 to 338 tok/s. The rest of that step is the union of the rows' experts in the
   fused MoE pair, about 14 ms, and the expert join. The join carries all seven slot rows of
   every token over the fabric, 3.9 ms at 16 rows; the owned-row join
   (`lane/dsv4-owned-join-20260928`) pushes only each rank's own rows.
2. **Streaming efficiency.** A step reads 6.7 GB per rank at an average 0.67 TB/s, 43% of the
   practical 1.54. The short GEMVs pay their ramp, and the fused MoE pair is bound by its MMA
   numeric class. Changing that class is an owner call.
3. **Graph nodes.** 1,709 kernel nodes per rank cost 1.45 ms of launch and dependency time
   before any work (0.85 us each). The fusions of 2026-09-26/27 took about 400 of them out;
   about 1,200 kernels under 3 us remain.
4. **The expert reduce's wait.** The reduce waits for the slower rank. The shared expert already
   runs on the rank with fewer routed slots. Splitting every expert's rows across both ranks
   would end the imbalance, at the cost of one more small join per layer.
5. **Context under TP/EP.** Not a speed lever. Closed for plain by the position-split C4 store
   (`../kv-split/`): 1M plain, 500k DSpark, at 0.4% to 2.0% decode. The C4 gather's remote row
   reads are about 9 us per C4 layer at c1; an owner-push of the selected rows would trade them
   for posted writes.
6. **Prefill.** The exact CUDA-core tile's register diet took TTFT -19% at 8k (#852). The
   tensor-core prefill (#472) is a numeric-class change and an owner call.

## What a realistic ceiling looks like

The 2026-09-25 estimate (below, kept as written) took 85% of practical bandwidth plus guessed
latency floors. It put TP-2 plain c1 at about 135 tok/s. The floors are now measured
(`## The step's floors`):
- the byte floor is 4.34 ms, 6.68 GB at 1.54 TB/s;
- the launch floor is 1.45 ms and the join handshakes add 0.19 ms;
- together that is about 6.0 ms, about 167 tok/s of device time, for this program at its current
  node count;
- at the 2026-09-25 rule's 85% of practical bandwidth it is about 6.8 ms, about 147 tok/s.

The captured step takes 9.5 ms, and memra serves 97 tok/s greedy c1 on the SE pair: 1.56x the 62.4
tok/s PP-2 served on the same pair on 2026-09-25. vLLM with b12x kernels serves 130.8 tok/s on the
same card shape with a different numeric program. Closing the rest within this program means
streaming nearer the practical bandwidth and fewer nodes; moving the floor itself means a
persistent program. Both are listed above.

As written on 2026-09-25:

Take 85% of practical bandwidth on every weight byte, plus the measured small-kernel floor:
- **PP-2 plain c1:** 11.44 GB / (0.85 x 1.54 TB/s) is 8.7 ms, plus about 1 ms of latency-bound
  chains, about 9.7 ms: **about 100 tok/s**. Achieved 68.4, 68% of it.
- **TP-2 plain c1:** about 6 GB per card is about 4.6 ms, plus about 1.7 ms of all-reduce and gathers,
  plus about 1 ms of chains, about 7.3 ms: **about 135 tok/s**. vLLM's published 109.3 tok/s on this
  shape is TP=2 with full graphs, inside that ceiling. memra serves TP/EP at 80.7 tok/s on the WS
  pair (60% of it) and 71.2 on the SE pair.
