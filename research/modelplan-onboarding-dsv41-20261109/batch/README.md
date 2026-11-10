# DeepSeek-V4.1-Flash onboarding: one decode call over several sequences (2026-11-10)

Verdict so far: a decode call can carry one token of each of several sequences
(`Runner::forward_batch`), every row run as that sequence's own one-token call, and it gives that
call's bytes: 3,830 of 3,830 rows on logits, on the draft-head taps and on the device argmax, over
mixed chat and long-context sequences, both sides of the 128-token window, one 18k-token sequence
beside short ones, and a batch that changes width as sequences finish. The served route steps its
running requests in such calls and answers byte for byte as it does one request at a time. On the
9-layer all-resident trunk, eight requests at once went from 153 to 436 tokens a second (2.85x) on
one card at 250 W. The same shape on vLLM runs at 1,007 tokens a second, so this is a step, not the
finish: a per-kernel split of the 8-sequence call (below) says where the rest is. It is in the
expert tiles (49% of the call at 28.8 distinct experts a layer), the head on 8 rows, the dense fp8
GEMMs and the launch count, not in per-sequence work, which is 0.25 ms of a 16 ms call.

Hardware: one RTX PRO 6000 Blackwell Max-Q at a time (sm_120a, 96 GB), 250 W cap, CUDA 13.4, rustc
1.97.1, 16 host cores. Every time is from short leases on a box that ran other work on its other
cards; the card ran at 87 to 89 C and its clock moved with the load (1,560 to 1,846 MHz), so a time
is comparable only to the other arm of its own run. The trunk is the checkpoint's first 9 layers
with every weight on the card (`--layers 9 --resident`, `MEMRA_DSV41_TRUNK_LAYERS=9`): the shape
the lane's baselines were measured on. It is not the model; no quality claim comes from it.

Commits on `dsv41/batch-20261110` (on top of the drafting branch's `df3751407c`, with the speed
lane's heads merged in as they were gated: `13d16037b4`, `3c83d8c1a6`, `74e1e68de1`, `9500e9960a`,
`1a2bdbef7d`):

| commit | what |
|---|---|
| `d6031c5e48` | `Runner::forward_batch`: a decode call over several sequences, `dsv41-spec batch-rows` and `batch-time` |
| `4203c2fb2f` | the served route steps its running requests in batched calls (`MEMRA_DSV41_BATCH_ROWS`, default 8) |
| `c3836e18b3` | the draft's picks chain on the device on one rank (no read-back between picks) |
| `fad66e75f1` | a lone step is its own call; an all-greedy batch takes the device argmax; `draft-tape` checks the untraced draft |
| `b46edb2477` | a row's keys read in place (`memra_v41_sparse_attn_keys`), a batch's writes one launch a kind (`memra_v41_put_rows`), the launch tally |
| `3d06dafc74` | `dsv41-spec`: a draft's time and launches, wide batched calls (`--batch-max`) |
| `7137c8efb2` | the served stream keeps its bytes: a token no longer decodes the whole answer |

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

The same 3,830 rows passed on every merged head (`3c83d8c1a6`, `74e1e68de1`, `9500e9960a`), and
the several-token verification gate (`verify-rows`, 692 rows on this trunk) with them.

The keyed switch (`b46edb2477`) is gated against the path it replaces, which stays in the tree as
`dsv41-spec --copy-keys` and as what a traced call runs: plan `keyed9` (verify-rows at 42, 27,
3.7k and 41k tokens of context, the four batch-rows cases) in three arms, copy, keyed, keyed with
`put_rows`. 12 of 12 jobs pass in each arm and the per-step logits and taps hashes of the 8 gate
jobs (4,638 lines) are identical between the arms.

## Times of the call

`dsv41-spec batch-time`, milliseconds for one step of 1 / 2 / 4 / 8 sequences, forward plus the
rows' logits unless noted; "alone" is the same step as one call a sequence.

| head, case | batched 1 / 2 / 4 / 8 | alone 1 / 2 / 4 / 8 | distinct experts a layer at 8 |
|---|---|---|---|
| `74e1e68de1`, same forced tokens (chat8) | 5.83 / 6.84 / 8.00 / 10.54 | 5.8 / 11.7 / 23.4 / 46.9 | 12.8 |
| `74e1e68de1`, distinct tokens (distinct8) | 5.85 / 7.93 / 11.22 / 17.46 | same | 28.8 |
| `b46edb2477`, distinct8, device argmax | 5.30 / 7.28 / 10.44 / 16.18 | 5.27 / 10.70 / 21.51 / 43.25 | 28.8 |
| `b46edb2477`, distinct8, no head | 4.24 / 6.02 / 9.00 / 14.12 | | 28.8 |

Sequences fed the same tokens route alike and flatter the batch: the distinct case is the honest
one. The lane's baselines on this shape: vLLM 4.02 / 4.59 / 6.17 / 7.14 ms; SGLang 7.41 / 7.54 /
8.41 / 10.0 ms and 10.47 ms for eight chats at 28.6 distinct experts.

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
others, stops, penalties, reasoning budget: all pass, on `4203c2fb2f` and again on `a7a5add916`.

Served A/B through the lane's shared client (`decode_c1`, `conc8`; arms interleaved A B B A A B,
three samples a run; A is `MEMRA_DSV41_BATCH_ROWS=1`, every step its own call):

| head | conc8 aggregate tok/s, alone | batched | ratio | decode_c1 tok/s, alone | batched |
|---|---|---|---|---|---|
| `4203c2fb2f` | 156.2 / 154.2 / 154.8 | 405.4 / 402.1 / 407.5 | 2.62x | 171.5 / 167.0 / 164.4 | 163.4 / 163.8 / 163.3 |
| `a7a5add916` | 151.3 / 153.2 / 155.0 | 441.5 / 436.2 / 435.3 | 2.85x | 170.5 / 162.7 / 165.6 | 164.5 / 164.5 / 163.5 |

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

`7137c8efb2` takes one piece of that between-calls time out: the stream decoded every generated id
again at every token (8.8 us a token averaged over a 512-token answer, growing with it); it now
appends the token's bytes (0.43 us), and every prefix reads the same text.

## Drafting on this branch

The draft's five picks chain on the device on one rank: the Markov embedding row is gathered by
the id the pick before it wrote, the bias is the same fp32 GEMM, one kernel adds it and takes the
argmax, and the ids come back once. `draft-tape` runs every step's draft a second time untraced
and requires the same tokens and confidences: 339 of 339 steps on the 12 tapes, with the vendor
bands unchanged (1,620 of 1,695 draft tokens equal the vendor's).

## Receipts

The job outputs (JSON lines, per-job results, hashes), the served gate and A/B summaries, the logs
and one row per GPU lease are exported with the run box's paths and card ids removed after the
final gate chain below; until then they are on the run box under the lane's `spec/out`.

## Not done

- The final-binary gate chain on the head with `1a2bdbef7d` merged (`7137c8efb2`), waiting for a
  card: the keyed gate's three arms, the standard gate plan, `batch-time` with the per-kernel
  table, the draft's time and launches, the served chat gate and the conc8 A/B.
- More than 8 running requests: the server makes more calls of 8; one wider call through the
  GEMMs' general bodies is coded in the gate binary (`--batch-max`, plan `wide9`) and not run.
- The per-sequence index path (score, top-k, candidates from a key table) is 0.25 ms of the call
  and waits behind the tiles, the head and the launch count.
- The batched call on more than one rank through the pipeline's stages (another lane builds it
  from this branch).
- Drafting through the server (the door stays off): its served gates, acceptance and A/B.
