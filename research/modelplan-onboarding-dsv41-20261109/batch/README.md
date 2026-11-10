# DeepSeek-V4.1-Flash onboarding: one decode call over several sequences (2026-11-10)

Verdict: a decode call can carry one token of each of several sequences
(`Runner::forward_batch`), every row run as that sequence's own one-token call, and it gives that
call's bytes: 3,830 of 3,830 rows on logits, on the draft-head taps and on the device argmax, over
mixed chat and long-context sequences, both sides of the 128-token window, one 18k-token sequence
beside short ones, and a batch that changes width as sequences finish. The same holds on two ranks
against one rank (both device transports) and on four ranks on four cards over the full 40-layer
trunk. The served route steps its running requests in such calls and answers byte for byte as it
does one request at a time. On the 9-layer all-resident trunk, eight requests at once went from
163 to 465 tokens a second (2.86x) on one card at 250 W, with no cost to a lone request. On four
cards and the full trunk eight requests share a 23.6 ms step (339 tokens a second) against 93.4 ms
one after another. On a pair of cards the 9-layer call takes 8.5 ms for eight sequences (16.7 on
one card) and the served route reaches 644 to 777 tokens a second for eight requests. The baselines are ahead on both shapes (vLLM: 1,007 tokens a second on the
9-layer shape, a 14.8 ms eight-row step on four cards), so this is a step, not the finish: the
per-kernel splits below say where the rest is. It is in the expert tiles (49% of the one-card call
at 28.8 distinct experts a layer), the head on 8 rows, the dense fp8 GEMMs and the launch count
(2,076 launches a rank for a one-row step on four cards), not in per-sequence work, which is
0.25 ms of a 16 ms call on 9 layers and about 0.9 ms of a 23.6 ms call on 40.

Hardware: one RTX PRO 6000 Blackwell Max-Q at a time (sm_120a, 96 GB), 250 W cap, CUDA 13.4, rustc
1.97.1, 16 host cores. Every time is from short leases on a box that ran other work on its other
cards; the card ran at 87 to 89 C and its clock moved with the load (1,560 to 1,846 MHz), so a time
is comparable only to the other arm of its own run. The trunk is the checkpoint's first 9 layers
with every weight on the card (`--layers 9 --resident`, `MEMRA_DSV41_TRUNK_LAYERS=9`): the shape
the lane's baselines were measured on. It is not the model; no quality claim comes from it.

Commits on `dsv41/batch-20261110` (on top of the drafting branch's `df3751407c`, with the speed
lane's heads merged in as they were gated: `13d16037b4`, `3c83d8c1a6`, `74e1e68de1`, `9500e9960a`,
`1a2bdbef7d`, `5d74b1db2a`, `76fc85cbae`, `72b2cf30c3`):

| commit | what |
|---|---|
| `d6031c5e48` | `Runner::forward_batch`: a decode call over several sequences, `dsv41-spec batch-rows` and `batch-time` |
| `4203c2fb2f` | the served route steps its running requests in batched calls (`MEMRA_DSV41_BATCH_ROWS`, default 8) |
| `c3836e18b3` | the draft's picks chain on the device on one rank (no read-back between picks) |
| `fad66e75f1` | a lone step is its own call; an all-greedy batch takes the device argmax; `draft-tape` checks the untraced draft |
| `b46edb2477` | a row's keys read in place (`memra_v41_sparse_attn_keys`), a batch's writes one launch a kind (`memra_v41_put_rows`), the launch tally |
| `3d06dafc74` | `dsv41-spec`: a draft's time and launches, wide batched calls (`--batch-max`) |
| `7137c8efb2` | the served stream keeps its bytes: a token no longer decodes the whole answer |
| `fe305d1b47` | a refused device pick falls back to the chunk's logits; `argmax_rows` reads the route flag |
| `487ee70847` | `dsv41-spec` on the rank group: per-rank timed tally, `--stop-on-fail`; the gather flag in `argmax_rows` |
| `e0cee9a9df` | the route's memory book stops booking the compressed rows a decode call used to copy |
| `3eeedf89fa` | the draft's pick chain reduces over several blocks through one 64-bit key |

## What a batched call is

`forward_batch(seqs, rows)` takes `(sequence, token)` pairs, the sequences distinct and past their
prefill. Row i is the one-token decode program of its sequence at its own position, against its
own state. The kernels of this route act per row, so the rows share every launch that does not
read a sequence's own buffers:

- once for all rows: embedding, Engram, every projection and norm, RoPE at each row's position
  (a device array of positions), the quantize-dequantize steps, the index query and head weights,
  the routed MoE (one plan over the rows, an expert read once however many rows picked it), the
  mHC mixes, the head, and since `b46edb2477` every row's writes into its own sequence's buffers
  (ring slot, a ratio-2 compressor's open pair, the appended compressed row and index key: one
  `memra_v41_put_rows` each through an address table) and sparse attention (each row reads its
  ring and its group's compressed rows where they live, through a key table);
- once per sequence, on the kv and index source layers only: the index score over that sequence's
  keys, its top-k, the candidate pool on the candidate source, and the pooling of a ratio-2 pair
  that completes on this step.

A sequence's state is the same objects its own calls use (`Seq`: per layer per rank the ring, the
open pair, and per group the compressed KV and index keys), so a sequence can move between batched
calls, its own one-token calls and several-token verification calls freely. A batch of one is the
plain call's bytes but about 2% slower, so the server sends a lone step through the plain call.

## The byte gate

`dsv41-spec batch-rows`: arm A runs every sequence alone (prompt, then its forced tokens one call
at a time); arm B prefills the sequences and steps them together, eight rows at most a call, a
sequence leaving when its tokens end. Every row's logits and taps must hash equal to arm A's row,
and `Runner::argmax_rows` must name the host argmax's token.

| case | sequences | what it covers | rows equal |
|---|---|---|---|
| chat8 | 8 | eight short chats, widths 8 down to 1 | 503 / 503 |
| window | 4 | prompts of 27 tokens forced across position 128, and past it | 1,384 / 1,384 |
| mixed | 5 | a 3.7k and an 18k context beside short chats | 701 / 701 |
| nine | 9 | nine sequences: calls of 8 + 1 | 1,242 / 1,242 |

The same 3,830 rows passed on every merged head (`3c83d8c1a6`, `74e1e68de1`, `9500e9960a`,
`5d74b1db2a` as `487ee70847`), and the several-token verification gate (`verify-rows`, 692 rows on
this trunk) with them. Calls wider than 8 rows (`--batch-max`, the GEMMs' general bodies) give the
same bytes too: 16 sequences in calls of 16 rows 480 of 480, 20 sequences in calls of 20 rows 600
of 600.

The keyed switch (`b46edb2477`) is gated against the path it replaces, which stays in the tree as
`dsv41-spec --copy-keys` and as what a traced call runs: plan `keyed9` (verify-rows at 42, 27,
3.7k and 41k tokens of context, the four batch-rows cases) in three arms, copy, keyed, keyed with
`put_rows`. 12 of 12 jobs pass in each arm and the per-step logits and taps hashes of the 8 gate
jobs (4,638 lines) are identical between the arms. The final binary (`487ee70847`) repeats it: copy
arm and keyed arm 12 of 12, the 8 hash files identical.

## Times of the call

`dsv41-spec batch-time`, milliseconds for one step of 1 / 2 / 4 / 8 sequences, forward plus the
rows' logits unless noted; "alone" is the same step as one call a sequence.

| head, case | batched 1 / 2 / 4 / 8 | alone 1 / 2 / 4 / 8 | distinct experts a layer at 8 |
|---|---|---|---|
| `74e1e68de1`, same forced tokens (chat8) | 5.83 / 6.84 / 8.00 / 10.54 | 5.8 / 11.7 / 23.4 / 46.9 | 12.8 |
| `74e1e68de1`, distinct tokens (distinct8) | 5.85 / 7.93 / 11.22 / 17.46 | same | 28.8 |
| `b46edb2477`, distinct8, device argmax | 5.30 / 7.28 / 10.44 / 16.18 | 5.27 / 10.70 / 21.51 / 43.25 | 28.8 |
| `b46edb2477`, distinct8, no head | 4.24 / 6.02 / 9.00 / 14.12 | | 28.8 |
| `487ee70847`, distinct8, device argmax (another card) | 5.47 / 7.29 / 10.50 / 16.72 | 5.44 / 11.05 / 22.17 / 44.31 | 28.8 |
| `487ee70847`, distinct8, no head | 4.34 / 5.90 / 8.97 / 14.47 | | 28.8 |

Sequences fed the same tokens route alike and flatter the batch: the distinct case is the honest
one. The lane's baselines on this shape: vLLM 4.02 / 4.59 / 6.17 / 7.14 ms; SGLang 7.41 / 7.54 /
8.41 / 10.0 ms and 10.47 ms for eight chats at 28.6 distinct experts.

The `487ee70847` rows ran on a different card than the rows above them, so the two heads do not
compare as whole calls. Kernel by kernel the merged mHC rows schedule shows: the mixes' own time at
8 sequences went from 0.99 to 0.49 ms a step, while the expert tiles, the fp32 FMA GEMM and the
dense fp8 GEMM read 3 to 6% higher on that card. Sixteen sequences on one rank: one call of 16
rows 33.0 ms, two calls of 8 31.9 ms.

Two builds on one card, A B B A, one short lease a run (`scripts/ab_spec.sh`): `3d63df643d`
(the speed lane's `76fc85cbae` merged: the mHC mixes under their sublayers) against `395e64da77`
(`72b2cf30c3` merged: the small-M expert tiles staged through shared memory, the fp32 head with
16-byte loads; and the draft's multi-block pick). Medians of two runs each, which agree within 3%:

| build | device argmax 1 / 2 / 4 / 8 | forward only 1 / 2 / 4 / 8 | draft |
|---|---|---|---|
| `3d63df643d` | 4.87 / 6.50 / 9.54 / 15.28 | 3.91 / 5.33 / 8.08 / 13.29 | 3.17 ms |
| `395e64da77` | 4.65 / 5.64 / 7.89 / 12.82 | 3.72 / 4.59 / 6.52 / 11.06 | 2.58 ms |

At 8 distinct sequences the expert tiles' own time went from 7.75 to 6.12 ms a step (29.9 to
23.6 us per distinct expert per layer) and the fp32 FMA GEMM from 2.33 to 1.93. On this card the
same call was 16.72 ms on `487ee70847` earlier the same morning.

The keyed switch, copy to keyed to keyed with `put_rows`, one run each in one lease chain:
one-token call 6.05 / 5.95 / 6.05 ms at 3.7k tokens of context and 6.79 / 6.57 / 6.63 at 41k;
8-row verification call 15.44 / 15.08 / 15.27 at 41k; 8-sequence call 18.37 / 18.35 / 18.15. The
copies were worth 1.5 to 3%, most at long context. Taking 244 per-sequence copies out of the
8-sequence call bought 0.2 ms: the call waits on the card, not on its small launches.

## Where an 8-sequence call goes

`batch-time` counts the launches of two consecutive steps by layer and kernel (`dsv41_gpu::tally`,
through `ck` and `copy_d2d`), then runs the same steps with a card sync after every launch, so
each kernel shows its own cost.

Launches of one call, 9 layers and the head: 437 for one sequence (44 on a window or plain ratio-2
layer, 48 on the Engram layer, 60 on a kv and index source layer, 5 for the head), 481 for eight.
The eight-sequence call differs only on the two source layers (82 each): index score 8, top-k 8,
pair pooling 4, and four launches that appear when any row completes a latent.

Each kernel's own time, milliseconds a step, one sequence to eight distinct sequences (the sum is
7.3 to 17.3; the call itself takes 5.30 to 16.18):

| kernel | launches a step | 1 sequence | 8 sequences |
|---|---|---|---|
| expert tiles (`gemm_f8_tiles`) | 27 | 1.19 | 7.94 |
| fp32 FMA GEMM (the head is one of 14) | 14 | 1.29 | 2.45 |
| dense fp8 GEMM | 66 | 1.51 | 2.27 |
| mHC mixes | 18 | 0.30 | 0.99 |
| activation quantize | 93 | 0.60 | 0.63 |
| bf16 GEMM | 13 | 0.51 | 0.60 |
| index score, top-k, pair pooling (per sequence) | 5 to 40 | 0.04 | 0.29 |
| everything else | about 200 | 1.9 | 2.1 |

The expert tiles cost 30.6 us per distinct expert per layer at 28.8 experts a layer (25.6 us at
12.8). The head block (collapse, norm, head GEMM, argmax) is 1.05 ms for one row and 2.1 ms for
eight. A one-sequence call is 437 launches in 5.3 ms: about 12 us a launch, so there the launch
count is the cost.

## Served

`memra-server`, the deepseek-v4.1 route: the engine thread forms a batch per step from the
running requests' one-token steps, up to `MEMRA_DSV41_BATCH_ROWS` (default 8, the small-M GEMM
bodies' limit) a call, more requests as more calls; a lone step is a plain call; a batch whose
requests are all greedy takes the device argmax and reads back ids, any other batch reads the
rows' logits and samples per request. Prefill stays one call per request. `MEMRA_MAX_SESSIONS`
keeps its meaning: it caps the running requests, the batch forms from whoever runs.

The served chat gate on the 9-layer trunk with batching on (`serve_gate9.sh`): rendering 7 of 7
prompts against the tape ids, solo answers equal to the native runner on 7 cases, five concurrent
requests byte for byte as solo, streaming, a sampled request twice alone and once beside four
others, stops, penalties, reasoning budget: all pass, on `4203c2fb2f`, on `a7a5add916` and on
`487ee70847`.

Served A/B through the lane's shared client (`decode_c1`, `conc8`; arms interleaved A B B A A B,
three samples a run; A is `MEMRA_DSV41_BATCH_ROWS=1`, every step its own call):

| head | conc8 aggregate tok/s, alone | batched | ratio | decode_c1 tok/s, alone | batched |
|---|---|---|---|---|---|
| `4203c2fb2f` | 156.2 / 154.2 / 154.8 | 405.4 / 402.1 / 407.5 | 2.62x | 171.5 / 167.0 / 164.4 | 163.4 / 163.8 / 163.3 |
| `a7a5add916` | 151.3 / 153.2 / 155.0 | 441.5 / 436.2 / 435.3 | 2.85x | 170.5 / 162.7 / 165.6 | 164.5 / 164.5 / 163.5 |
| `487ee70847` (another card) | 154.1 / 163.1 / 162.6 | 461.6 / 464.8 / 464.8 | 2.86x | 171.8 / 174.5 / 173.6 | 173.2 / 175.1 / 173.8 |

On `4203c2fb2f` a lone request paid about 2% for going through the batched call; `a7a5add916`
sends it through the plain call and the single-request medians are within the alone arm's spread.
The batched arm ran the card at 1,560 to 1,586 MHz under the 250 W cap, the alone arm at 1,804 to
1,846. vLLM's eight-request plain step on this shape is 7.14 ms, 1,007 tok/s aggregate.

Where the served run's time goes on `a7a5add916` (route log of the three batched runs, 2,016
eight-row steps each): a step's call and pick take 15.15 to 15.33 ms on average. A run is 8
prompts of 1,024 tokens and 8 x 512 generated tokens in about 9.4 s: the prompts' prefill, one
prompt a step at about 7,400 tokens a second, is 1.1 to 1.4 s of it (the last request's first
token arrives at 1.44 s), the 512 steps 7.8 s, and what the engine thread does between calls about
0.4 ms a step. So the aggregate (436) sits 17% under the pure decode rate of the step (8 / 15.2 ms
= 526): prefill throughput counts in this workload too.

On `487ee70847` the served eight-row step averages 14.45 to 14.56 ms and a request decodes at
63.8 tokens a second.

`7137c8efb2` takes one piece of that between-calls time out: the stream decoded every generated id
again at every token (8.8 us a token averaged over a 512-token answer, growing with it); it now
appends the token's bytes (0.43 us), and every prefix reads the same text.

## More than one rank

`forward_batch` is written over the rank group like the one-token call: the keys side of
attention runs on every rank from its own copy of each sequence's state (so a row's key table
names buffers of the rank's own card), each rank runs its heads and output groups and the columns
are gathered, and the MoE is the group's own entry (`Group::moe`), so a change of the expert split
across ranks lands underneath the batched call without touching it.

The gate against one rank is the `keyed9` plan with two ranks on one card (`--devices 0,0`): a
batched or verification row must equal its sequence's own call on two ranks, and the hashes of
that call must equal the one-rank arm's; then the served chat gate with
`MEMRA_DSV41_DEVICES=0,0` against the one-rank native references. On `487ee70847`:

| transport | `keyed9`, 12 jobs | hashes against the one-rank arm | calls of 20 rows against one rank | served chat gate |
|---|---|---|---|---|
| peer-pull | 12 / 12 | 8 / 8 files, 4,638 lines | 600 / 600 | 9 / 9 |
| device-push | 12 / 12 | 8 / 8 files, 4,638 lines | 600 / 600 | 9 / 9 |

Two ranks on one card share the card, so their times say what the group's overhead is, not what a
second card buys: an 8-sequence step is 16.74 ms on one rank, 17.74 on two with peer-pull, 17.94
with device-push; a one-token call 5.5 against 6.2 ms.

### Two ranks on two cards

The same plan and compares on a pair of cards, a rank a card, on the merged head `3d63df643d`
(the speed lane's `76fc85cbae` in), against the one-rank arm of the earlier build:

| transport | `keyed9`, 12 jobs | hashes against the one-rank arm | calls of 20 rows against one rank | served chat gate |
|---|---|---|---|---|
| peer-pull | 12 / 12 | 8 / 8 files, 4,638 lines | 600 / 600 | 9 / 9 |
| device-push | 12 / 12 | 8 / 8 files, 4,638 lines | 600 / 600 | 9 / 9 |

`batch-time` distinct8 on the pair, milliseconds a step (9-layer trunk):

| sequences | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| one rank, one card (`487ee70847`) | 5.50 | 7.30 | 10.60 | 16.74 |
| two cards, peer-pull, device argmax | 3.92 | 4.55 | 5.89 | 8.51 |
| two cards, peer-pull, forward only | 3.39 | 4.05 | 5.25 | 7.65 |
| two cards, device-push, device argmax | 3.68 | 4.36 | 6.10 | 9.42 |
| two cards, device-push, forward only | 3.16 | 3.77 | 5.28 | 8.26 |
| vLLM, one card (the lane's baseline) | 4.02 | 4.59 | 6.17 | 7.14 |

The second card buys 1.4x at one sequence and 1.97x at eight. A one-token call is 4.13 ms and an
8-row verification call 7.78 ms on peer-pull (5.5 and 14.0 on one rank), the same at 3.7k, 18k and
41k tokens of context.

Served on the pair (`scripts/pair_serve.sh`: the 9-layer trunk, batching on, the shared client,
three samples a run, runs interleaved one / peer / device / device / peer / one), tokens a second:

| arm | decode_c1 | conc8 aggregate | dconc8 aggregate | served step, 1 row / 8 rows (ms) |
|---|---|---|---|---|
| one rank | 186.2, 173.7 | 499.9, 459.7 | 484.7, 451.5 | 5.03 / 13.84, 5.98 / 14.89 |
| two cards, device-push | 226.9, 238.6 | 643.7, 690.6 | 686.0, 710.4 | 4.34 / 10.28, 4.13 / 9.54 |
| two cards, peer-pull | 230.6, 236.1, 234.9 | 725.3, 776.8, 755.4 | 737.9, 771.2, 756.4 | 4.14 / 9.39, 4.03 / 8.83, 4.02 / 8.99 |

The first peer-pull run is left out: its prompts took 7 to 8 s to first token (0.3 s elsewhere)
and its rounds climbed from 153 to 239 (conc8) and 502 to 624 (dconc8) tokens a second while the
first card drew 137 W; the next peer-pull server was fast from its first request. The cause was
not found. An idle card of this box reports its PCIe link at generation 1 (the loaded ones at 5),
and two later peer-pull runs logged the links: generation 1 before the server starts, 5 on both
cards once the weights are loaded and before the first request, 5 to the end. So the link is up
before a request arrives and does not explain that run.

Peer-pull leads at eight requests in every clean run (755 against 667 tokens a second, medians)
and the two transports are even for a lone request (235 against 233), so the one-row edge
device-push shows in the gate binary does not show served. Against one rank the pair serves eight
requests 1.39x (device-push) to 1.57x (peer-pull) faster and a lone request 1.3x. The first six
runs ran while another lane's build held the other half of the lane's host cores (the server and
the client here had eight logical cores); the last two peer-pull runs did not.

With the experts on the card and more than one rank, the MoE under the call is the split one
(`173d61cf6f`, merged: every rank holds its rows of every expert), and its gathers and the
attention's can run inside kernels (`MEMRA_TP_TRANSPORT=device-push`, `940e7e6289`) when each
rank's region of a gather is at most 256 KiB. What a call of R rows (a batch or a verification
block) gathers, bytes a rank:

| gather | bytes a rank | 8 rows, 2 ranks | 8 rows, 4 ranks | first R past 256 KiB at 2 / 4 ranks |
|---|---|---|---|---|
| expert hidden columns `[7R][2304/N]` | R x 32,256 / N | 129,024 | 64,512 | 17 / 33 |
| attention `wo_a` columns `[R][8/N x 1024]` | R x 16,384 / N | 65,536 | 32,768 | 33 / 65 |
| attention output columns `[R][5120/N]` | R x 10,240 / N | 40,960 | 20,480 | 52 / 103 |
| MoE output columns `[R][5120/N]` | R x 10,240 / N | 40,960 | 20,480 | 52 / 103 |

So the served route's calls (8 rows at most) stay inside the bound on 2 and on 4 ranks. Past it
`Group::allgather` decides per gather: the one whose region is too large takes the copy-engine
gather (events and peer copies) and the others of the same layer stay in kernels. The `wide20`
case above (one call of 20 rows on two ranks: the hidden gather is 322,560 bytes a rank) is that
mix, and it gives the one-rank bytes.

### First run on four ranks, the full trunk

`487ee70847`, 40 layers resident, one rank a card on four cards (70.3 GiB a rank, loaded in
11.8 s), `scripts/n4.sh`, 322 s of cards in one lease. The byte reference here is each sequence's
own one-token call on the same four ranks (one rank cannot hold 40 layers; the speed lane's gate
ties the four-rank one-token call to one card).

| transport | case | rows equal (logits, taps, device argmax) |
|---|---|---|
| device-push | chat8 | 503 / 503 |
| device-push | nine | 1,242 / 1,242 |
| device-push | window | 1,384 / 1,384 |
| device-push | mixed (3.7k and 18k contexts) | 701 / 701 |
| peer-pull | chat8 | 503 / 503 |
| peer-pull | wide16 (29 calls of 16 rows) | 480 / 480 |

It is also the first run of the batched call past layer 8: the ratio-1 layers, the candidate pool
of layer 20 and the six index sources after it.

`batch-time` on device-push, milliseconds a step:

| sequences | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| one call, device argmax (distinct8) | 12.48 | 14.13 | 17.29 | 23.58 | |
| forward only | 12.19 | 13.86 | 16.96 | 23.07 | |
| each sequence alone | 12.35 | 23.93 | 47.09 | 93.43 | |
| one call, device argmax (distinct16 set) | 12.47 | 13.98 | 17.35 | 23.12 | 72.57 |
| calls of 8 rows | | | | | 46.94 |
| vLLM, same layout (the lane's baseline) | 7.82 | 9.21 | 11.56 | 14.77 | |

Eight requests share a step at 23.58 ms (339 tokens a second) against 93.43 ms alone, 29.4
distinct experts a layer. One call of 16 rows is slower than two calls of 8: past 8 rows the
GEMMs leave the small-M bodies (the expert tiles take three times the launches). The served
route keeps calls of 8.

A step at one sequence is 2,076 launches a rank (activation quantize 410, dense fp8 GEMM 290, the
gathers' push and pull 320), 6 us a launch: the one-row step is bound by its launch count. Each
rank's launches serialized (a card sync after every launch on the rank's thread, which inflates
the small kernels; the differences are the reading), milliseconds a step a rank, one sequence to
eight: fused w1 + w3 + SwiGLU tiles 1.57 to 5.53, w2 tiles 0.71 to 2.34, dense fp8 3.89 to 5.08,
push 1.73 to 2.69, pull 1.22 to 1.88, the per-sequence index path (eight index sources on the full
trunk) 0.14 to 1.07, fp32 FMA 1.46 to 2.06, mHC mixes 1.38 to 1.93, bf16 GEMM 1.78 to 2.29.

Two review findings on the greedy pick are fixed with it. `Runner::argmax_rows` answers for a call
as a whole, so one row with a non-finite logit used to fail every running request of the step; the
route now takes the chunk's logits when the device pick refuses and each request chooses from its
own row, as on the logits path (`pick_rows`, with a unit test that fails on the old behaviour).
And `argmax_rows` reads each rank's route flag as `Group::head_logits` does.

## Drafting on this branch

The draft's five picks chain on the device on one rank: the Markov embedding row is gathered by
the id the pick before it wrote, the bias is the same fp32 GEMM, one kernel adds it and takes the
argmax, and the ids come back once. `draft-tape` runs every step's draft a second time untraced
and requires the same tokens and confidences: 339 of 339 steps on the 12 tapes, with the vendor
bands unchanged (1,620 of 1,695 draft tokens equal the vendor's).

A draft took 3.37 ms on `487ee70847` (median of the 12 tapes' last steps, 150 launches). Its
launches' own time, 4.08 ms serialized: the three stages 2.08 (43 launches each; dense fp8 0.54,
expert tiles 0.53), the head block on the five rows 1.43 (the fp32 head GEMM about 1.35 of it),
the pick chain 0.55 (the add-and-argmax over 129k logits in one block was 0.28 of it, 57 us a
pick).

`3eeedf89fa` replaces that one-block pick in the device chain with `memra_v41_add_argmax_key`:
blocks of 4,096 logits reduce their slices and meet in one 64-bit key through `atomicMin` (the
high word falls as the value rises, the low word is the index, so the next pick's gather reads
the token from the key). Its gate is the same `draft-tape` check, the untraced draft against the
traced one, which keeps the old kernel: 339 of 339 steps equal. In the A B B A above a draft went
from 3.17 to 2.58 ms: the pick chain 0.55 to 0.30 ms of own time (this kernel), the head block
1.38 to 1.17 (the speed lane's head), the three stages 2.05 to 1.89. The lane's vLLM draft is
2.1 ms.

## Receipts

`receipts/` holds the job outputs (JSON lines, per-job results, hashes), the served gate and A/B
summaries (without the per-step server logs), the logs and one row per GPU lease, with the run
box's paths and card ids removed; `plans/` and `scripts/` are what produced them (`chain.sh` is
the one-card chain of the final head, `n4.sh` the four-card step, `cmp_arms.py` the hash compare
and the tables; `pair_serve.sh` the served cell on a pair; `ab_spec.sh` two builds on one card).
Final-head runs are the `k15-*` directories, the pair's the `k16-*` and `ps-k16` ones, the two
builds' A B B A `abt-k16-k18`.

## Not done

- The served default transport on more than one rank (the build default is host-canonical; on
  this pair peer-pull served best at both widths).
- Fast bodies for calls of 9 to 32 rows: until then more than 8 running requests are more calls
  of 8, and a 32-request step on four cards is about four 23 ms calls.
- The per-sequence index path (score, top-k, candidates from a key table): 0.25 ms of the call on
  9 layers, about 0.9 ms on the full trunk at 8 sequences.
- The served route on four cards (this block's four-card run is the gate binary's).
- Drafting through the server (the door stays off): its served gates, acceptance and A/B.
