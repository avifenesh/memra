# Hy4-flash GPU route: decode tuning on one GB10 (2026-11-20)

The route of step 8 (`hy4f-gpu`, numeric class `hy-v4-flash-bf16x2-activations`), tuned for
decode. Every change keeps the class: the same products and the same summation order for every
output, so each step's tapes equal step 8's bit for bit (`receipts/*/check3.log`: `chat` forced,
`long` and `long-split` free, every logit, router and indexer array), and the checkpoint gate
passes with the same numbers.

## Roofline

Bytes one c=1 decode token reads (`tools/roofline_bytes.py` over the artifact's headers):

| what | GB |
|---|---:|
| attention projections (FP8 128x128 blocks and BF16) | 6.174 |
| shared experts | 1.161 |
| latent projections (`routed_expert_down/up_proj`) | 0.774 |
| dense first layer MLP | 0.494 |
| Engram projections | 0.191 |
| router (F32 islands read BF16) | 0.129 |
| indexer projections | 0.063 |
| norms, HC functions, sinks | 0.060 |
| head (BF16, F32 GEMM) | 0.742 |
| routed experts, 16 of 512 a layer, NVFP4 codes + group scales | 2.612 |
| **a token** | **12.40** |

The KV cache adds about 3.5 MB at 36 positions and 95 MB at 2.1k (F32 latent rows, 512-row rings
on the sliding layers, the indexer's recall of at most 2,048 keys); Engram rows are 64 x 1 KB.

GB10 streams 229 to 242 GB/s (`receipts/base/bw-roofline.txt`: `bw-roofline` contiguous 128-bit
loads; 656 scattered 4 MiB slabs, a token's expert pattern, 239 to 242). At 240 GB/s a token
costs 51.7 ms: **the c=1 decode roofline is 19.4 tokens/s** (19.2 at 2.1k context).

## Method

One GB10 (`sm_121a`), the same build flags for every arm, the GPU otherwise idle (`memwait.sh`:
no other compute process, 108 GiB free). `hy4f-gpu --bench 3`: a warm-up, then three timed runs of
the request's prefill and 63 decode calls; medians with the range of the three. The reading
decodes as the checkpoint's `generation_config.json` says: sampled, temperature 0.9, no top-k
or top-p cut, every run from seed 20261121, so two builds of the class draw the same ids
(`ids_fnv1a`). Up to step 17 the reading ran on the argmax (`--greedy` now); those rows are
marked greedy below and are not comparable with the sampled ones (a sampled step is about 0.4 to
0.9 ms longer). The correctness checks (`check3`, the checkpoint gate) are unchanged: forced
and greedy tapes. Profiles:
`nsys` over `--gen 24` (`chat`), the decode calls after the first two summarized by
`tools/trace_steps.py` (per kernel ms a step), `tools/gemm_grid.py` (per GEMM shape) and
`tools/gaps.py` (GPU idle between kernels). `kbench/` builds the route's `.cu` alone and checks
every small-row kernel bit for bit against the tiled kernel it replaces, then times it with a cold
L2.

## Steps

### Sampled readings (the checkpoint's generation settings)

| build | chat decode tok/s | step p50 ms | long (2.1k) decode tok/s | step p50 ms | chat prefill tok/s | long prefill tok/s |
|---|---:|---:|---:|---:|---:|---:|
| step 9 (87b68fd76c, this session's start) | 14.50 [14.50-14.52] | 68.8 | 13.67 [13.67-13.69] | 72.9 | 106.0 | 317.4 |
| step 17 (abb019d9d2) | 17.33 [17.30-17.37] | 57.6 | 16.19 [16.13-16.20] | 61.5 | 105.8 | 389.4 |
| step 18: prefill hyper-connection reductions over one read | 17.35 [17.29-17.35] | 57.5 | 16.20 [16.17-16.20] | 61.5 | 105.8 | 403.8 |
| step 21: prefill GEMM live slices, pair dequant, 8 warps (`receipts/s21-gemm-warps/`) | 17.29 [17.28-17.30] | 57.8 | 16.17 [16.14-16.20] | 61.6 | 116.3 | 434.8 |
| step 22: prefill attention, queries resident, pieces read ahead (`receipts/s22-attn-prefetch/`) | 17.23 [17.23-17.30] | 57.8 | 16.15 [16.11-16.18] | 61.7 | 116.2 | 472.9 |

`receipts/sampled-step9-step17/`: both builds run the same runner and draw the same ids. On `chat`
the first sampled token is the end token (as the argmax was), so its 63 calls decode past the
end; on `long` the end token comes at call 33. The reading times the calls either way.

### Greedy readings, steps 0 to 17 (argmax decode; kept as measured)

| step | chat decode tok/s | step p50 ms | long (2.1k) decode tok/s | step p50 ms | chat prefill tok/s | long prefill tok/s |
|---|---:|---:|---:|---:|---:|---:|
| step 8 (base) | 6.04 [6.03-6.04] | 165.7 | 5.40 [5.25-5.40] | 185.3 | 84.2 | 225.0 |
| 1: small-row GEMMs, F32 islands GEMV, split HC, one table upload a call | 11.60 [11.57-11.61] | 86.2 | 9.57 [9.55-9.58] | 104.2 | 93.9 | 222.0 |
| 2: programmatic dependent launch | 12.37 [12.36-12.37] | 80.9 | 10.16 [10.15-10.18] | 98.3 | 93.1 | 221.3 |
| 3: Engram reads on a resident thread pool, beside the first layers | 12.56 [12.56-12.57] | 79.6 | 10.23 [10.22-10.24] | 97.5 | 92.8 | 220.9 |
| 4: attention key tiles, indexer logits and recall in parallel | 12.90 [12.76-13.01] | 76.9 | 11.98 [11.98-12.00] | 83.2 | 93.7 | 280.8 |
| 5: F32 islands split at their 16-input seams (router, head) | 13.36 [13.36-13.36] | 74.9 | 12.29 [12.28-12.30] | 81.1 | 94.8 | 280.1 |
| 6: one launch for the GEMMs that share an input | 13.52 [13.52-13.53] | 73.9 | 12.40 [12.40-12.42] | 80.4 | 100.2 | 283.4 |
| 7: GEMM weights resident in 4 KiB tiles | 14.18 [14.18-14.18] | 70.5 | 13.00 [13.00-13.04] | 76.7 | 106.0 | 294.1 |
| 8: hyper-connection reductions as partials, tree in the mix | 14.46 [14.44-14.46] | 69.2 | 13.23 [13.22-13.25] | 75.3 | 107.1 | 298.2 |
| 9: attention tiles prefetched, heads a block by SM count | 14.69 [14.69-14.71] | 67.9 | 13.81 [13.79-13.83] | 72.2 | 106.6 | 318.0 |
| 10: a small-row stage's weight tile as one bulk copy | 16.53 [16.52-16.54] | 60.4 | 15.43 [15.40-15.46] | 64.6 | 107.1 | 319.5 |
| 11: the per-row kernels' strided loads issued together | 16.79 [16.76-16.82] | 59.5 | 15.71 [15.66-15.72] | 63.5 | 107.4 | 318.1 |
| 12: the small kernels let their dependent launch before they wait | 16.94 [16.93-16.99] | 58.9 | 15.87 [15.84-15.88] | 62.9 | 102.7 | 319.2 |
| 13: the hyper-connection mix writes the row's RMS norm; indexer chains side by side | 16.98 [16.96-16.99] | 58.7 | 15.92 [15.91-15.94] | 62.6 | 102.9 | 320.9 |
| 14: a small-row stage loads its operands first; a BF16 stage is two tiles | 17.30 [17.30-17.31] | 57.7 | 16.17 [16.16-16.19] | 61.7 | 102.3 | 320.0 |
| 15: the router's top-k as k rounds of the block's best | 17.37 [17.36-17.37] | 57.5 | 16.22 [16.20-16.23] | 61.5 | 102.6 | 320.0 |
| 16: the router's scores, picks, weights and one-row plan in one block | 17.51 [17.50-17.54] | 57.1 | 16.33 [16.33-16.35] | 61.0 | 106.1 | 319.6 |
| 17: prefill attention: a block a (query, chunk) scores every head, four heads a warp | 17.47 [17.45-17.49] | 57.1 | 16.32 [16.29-16.35] | 61.0 | 105.3 | 388.3 |

Greedy ids equal in every run of every arm (`ids_fnv1a`). Each step's section below quotes its
greedy reading.

## Gate

`memra model verify checkpoint --against hy_v4_flash` over the f64 oracle tapes with the tuned
route as the native runner, then the five red arms (`tools/verify-chain.sh`). On step 13's binary
(`receipts/gate-step13/verify-chain.log`), step 17's and step 18's (`receipts/gate-step17/`,
`receipts/gate-step18/`):
`verified=Checkpoint`; Engram off fails at 0.0182 past the 0.0058 band, BF16 dequantization at
0.0060, reversed ties are refused, the cache off by one fails at 0.0097, the early pool at 0.894
past 0.0224: the points step 8 fails at.

### On main with step 21 (`receipts/rebase-main/`)

The lane replayed on main (the serving route and the DFly2 drafter landed), one build of
f37cbd6cec (step 21; the head adds a whitespace-only edit and receipts): check3 3 of 3
`BITWISE_EQUAL`; the 12 drafter gate logits hashes equal the drafter branch's pins
(`gate2-hashes.txt`); `memra model verify` Checkpoint and Serve passed; the serving chat gate
PASS with drafts on and off; sampled bench and real turns, drafts on against off: draft, bench
and turns identity PASS.

### On the DFly2 drafter's branch (`receipts/rebase-dfly2/`)

The 23 steps replayed on the drafter's branch (the serving route and the DFly2 drafts); the
drafter reads the same 4 KiB weight tiles. One build of source 2815fbd9ab (`bins.sha256`;
the branch head adds only docs and receipts on top), one GB10, `prb-chain.log` in order:

- check3: 3 of 3 `BITWISE_EQUAL` to step 8's reference tapes (`prb-check3.log`).
- sampled reading: chat 57.70 ms a step (17.29 tok/s), long 61.71 ms (16.11), long prefill
  404.0 tok/s (`bench/rb-*.jsonl`); the same ids as before the replay.
- drafter: both gens run; the 12 drafter gate logits hashes (chat and long, the gate, its four
  red arms and `gate-route`) equal the drafter branch's own pins (`prb-gate-*.log`).
- `memra model verify`: Checkpoint and Serve passed (`prb-verify-rb.log`).
- serving gate with drafts on and off (`serve/*/chat-gate.log`: CHAT-GATE PASS both), sampled
  bench and real turns drafts on, off, on again: draft, bench and turns identity PASS
  (`serve/*identity*.json`); the greedy fault cells (draft and backup faults at round 3) give the
  plain answers, 6 of 6 refusals logged.

### 1. Small-row GEMMs (`receipts/s1-gemv/`)

Base profile (`receipts/base/profile-chat-*.txt`): a step was 171.5 ms of wall, 162.3 ms of GPU
time over 2,079 operations. The weight GEMMs took 98.2 ms at 117 to 180 GB/s, because a decode
call ran the 16-row tile of the prefill GEMM (four warps, a block barrier every 32 inputs, one
weight load in flight); the F32 islands' GEMM (router 26.4 ms at 5 GB/s, head 10.6 ms) a 64x64
SIMT tile; the hyper-connection pre mix (16.0 ms) one block a row looping over 17 reductions;
the expert planner (1.6 ms) one thread; and every row-index upload from pageable memory waited for
the stream.

- `r_gemv_kernel`: up to 16 rows. A warp owns an n8 column tile over the whole K; weights, scales
  and activations stream through a 4-stage cp.async ring; with up to 4 rows the activations are
  split into their hi/lo BF16 pairs once a stage. Each output gets the tiled kernel's mma
  sequence (hi then lo every 16 inputs, the FP8 block scale folded every 128, `weight_scale_2` at
  the end), so the bits are the same (`kbench`).
- `r_gemv_f32_kernel`: router and head for up to 4 rows, one lane a weight row, each output the
  same chain of separately rounded products and 16-input partial sums as `hy4f_gemm_kernel`.
- `r_hc_dots_kernel` + `r_hc_mix_kernel`: the 17 reductions of a row as 17 blocks, each the same
  strided sums and fixed tree.
- `r_moe_plan_kernel`: one block's scans instead of one thread's walk; the same tile table.
- the call's index tables (token ids, positions, cache rows, ring slots) in one upload at the
  call's start; the Engram codes through page-locked staging.

Dense FP8 GEMVs reach 210 to 249 GB/s alone, the routed experts 169, the router 59, the head 190
(`receipts/s1-gemv/kbench.txt`). Decode step 165.7 -> 86.2 ms on `chat`, 185.3 -> 104.2 ms on
`long`. What the profile shows next (`profile-chat-gaps.txt`): one 5.3 ms GPU idle a step while
the host reads the first Engram layer's rows, and about 2,000 inter-kernel gaps of 1 to 5 us.

### 2. Programmatic dependent launch (`receipts/s2-pdl/`)

About 2,000 kernels a step, each a few microseconds behind the previous one and each with its own
ramp. Every route kernel now launches with programmatic stream serialization (`r_launch`), lets
its dependent launch as soon as its own blocks run, and waits (`griddepcontrol.wait`) before it
reads a predecessor's output or writes memory. The small-row GEMMs stream their first weight
stages (weights are resident and read-only) before the wait, so the next GEMM's weights load
while the previous one finishes. No value changes: 3 of 3 tapes bit-equal. Decode step 86.2 ->
80.9 ms on `chat`, 104.2 -> 98.3 ms on `long`.

Tried with it and dropped: reading the Engram rows for all four layers on a thread spawned each
call (32 reader threads) beside the first layers: 0.9 ms a step slower, the spawn cost more than
the reads (128 page-cached `pread`s of about 2 us each spread over 8 ms).

### 3. Engram reads on resident threads (`receipts/s3-engram-pool/`)

The Engram rows of a call depend only on its token ids. The call now hashes all four Engram
layers at its start and hands the reads (four rows a task) to 16 threads started with the route;
the first Engram layer (layer 2) collects them, by which time the GPU has two layers queued. The
first layer's GPU idle (5.3 ms a step under the profiler) is gone; decode step 80.9 -> 79.6 ms on
`chat`, 98.3 -> 97.5 ms on `long`. The bench rows now carry every step's time (`steps_ms`).

### 4. Attention key tiles, indexer logits, recall (`receipts/s4-attn-indexer/`)

A profile with programmatic launch off (`receipts/s3-engram-pool/profile-long-decode-pdl-off.txt`,
a diagnostic build: kernels then run back to back, so each kernel's time is its own) put `long`'s
decode step at 63.0 ms of GEMMs, 14.0 ms of attention (333 us a layer), 6.5 ms of indexer group
logits and 1.2 ms of recall.

- Attention: each of a key tile's 18,432 values was a load whose address came from another load
  (the key's row in the recall list), one after the other per thread. The tile's 32 rows are now
  read once into shared memory, then its values as independent 16-byte loads. The values, and
  every score, sum and output, are the same.
- Indexer group logits: one block a query looped over its 526 groups. Now one thread a (query,
  group), 64 groups a block with their keys in shared memory; each logit is the same chain.
- Recall: one thread a query walked up to 512 groups. Now a thread an entry: own's place among
  the picks, each distinct group's keys at its prefix offset; the same list.
- Pooling: the 2 x 4 dot chains of a key run side by side, each in input order.

Decode step 79.6 -> 76.9 ms on `chat`, 97.5 -> 83.2 ms on `long`; `long`'s prefill 221 -> 281
tokens/s (the same attention kernel). 3 of 3 tapes bit-equal.

### 5. Router and head split at their seams (`receipts/s5-f32-split/`)

The F32 islands' chain for an output is a sum of 16-input partial sums, each from zero. A thread
now computes one partial (one row, output and step) and one thread an output adds them in step
order: the same bits with 192 times the parallelism. The router (512 outputs) went 53 -> 19 us a
layer and the head at one row 3.90 -> 3.05 ms (243 GB/s; four rows keep the lane kernel).
Decode step 76.9 -> 74.9 ms on `chat`, 83.2 -> 81.1 ms on `long`.

### 6. One launch for the GEMMs that share an input (`receipts/s6-multi/`)

`memra_hy4r_gemm_multi` runs up to six small-row GEMMs of one format and K as one grid, each
output's blocks as its own launch had them: `q_a`, `kv_a`, the indexer's key and head weights
(BF16, one launch; the FP8 output gate its own), the latent down projection with the shared
expert's gate and up, the routed experts' gate and up, the latent up projection with the shared
expert's down projection, and an Engram layer's value and four key projections. About 240 fewer
launches a step and small outputs running beside large ones: 74.9 -> 73.9 ms on `chat`, 81.1 ->
80.4 ms on `long`.

### 7. GEMM weights resident in 4 KiB tiles (`receipts/s7-tiles/`)

In the model the decode GEMMs ran at 160 to 185 GB/s, against 220 to 248 alone in `kbench`. Back
to back over 600 copies of `o_proj` spread through 30 GB (`kbench spread`, the model's pattern) the
same kernel held 178 GB/s: each stage of a block read 32 rows x 128 bytes, 32 places 16 KB apart.
Every `memra_hy4r_gemm` weight is now resident in a tile layout (`memra_hy4r_tile` at load, through
one reused staging buffer): 32 rows x 128 bytes contiguous, a 32-row block's tiles in K order, so a
stage is one 4 KiB read; the prefill kernel addresses the same layout. Same bytes, same values:
spread 178 -> 200.5 GB/s, decode step 73.9 -> 70.5 ms on `chat`, 80.4 -> 76.7 ms on `long`, prefill
100 -> 106 and 283 -> 294 tokens/s. Resident 87.0 GiB (the head and router stay row-major for the
F32 kernels; `kv_b`'s nope rows are kept only transposed).

### 8. Hyper-connection reductions (`receipts/s8-hc/`)

Each of a row's 17 reductions (12,288 values) ran as one 256-thread block, 85 launches a step at
24 us. The 256 strided per-thread partials now run as four 64-thread blocks a reduction, and the
mix kernel's 256 threads take partial t each and run the same warp and block tree: 70.5 -> 69.2
ms on `chat`, 76.7 -> 75.3 ms on `long`.

### 9. Attention: prefetched tiles, heads a block by SM count (`receipts/s9-attn-tiles/`)

A decode call has few (head, chunk) units: 16 blocks of 8 heads on a sliding layer, one block an
SM (a key tile is 74 KB of shared memory), each loading then scoring its tiles in turn. Now the
call takes the most heads a block that keeps its blocks within one wave of the SMs (4 on a
sliding layer: 32 blocks), all 8 warps load, the next tile's values are loaded into registers
while one is scored, and the score reads the query as 16-byte broadcasts. Each score, sum and
output is the same chain. `kbench attn`: 164 -> 79 us (sliding, 512 keys), 170 -> 119 us
(indexed, 2,048 keys). Decode step 69.2 -> 67.9 ms on `chat`, 75.3 -> 72.2 ms on `long`;
`long`'s prefill 298 -> 318 tokens/s.

### 10. A stage's weight tile as one bulk copy (`receipts/s10-bulk-tiles/`)

A small-row stage loaded its 4 KiB weight tile as 256 16-byte `cp.async` copies, two a thread,
each with its own address and its own place in shared memory (rows 144 bytes apart to keep a
warp's 8 rows on distinct banks). The tile is now one `cp.async.bulk` copy issued by one thread,
completing on the stage's mbarrier; the tile lands as stored (rows 128 bytes apart), and to keep
the banks distinct `memra_hy4r_tile` stores row r's 16-byte pieces XOR-swizzled by r % 8 (`r_toff`;
the prefill kernel reads the same layout). The bytes a product reads are the same, so no value
moves: 3 of 3 tapes bit-equal. Decode step 67.9 -> 60.4 ms on `chat` (16.53 tokens/s), 72.2 ->
64.6 ms on `long` (15.43); prefill unchanged.

### 11. Per-row kernels: loads issued together (`receipts/s11-loads/`)

With the GEMMs at the bus (the profile of step 10 with programmatic launch off: 50.3 ms a step
for 11.5 GB, 229 GB/s), about 6 ms of a step was 1,443 small launches. In the per-row kernels a
thread walked its strided elements one load after the other, each waiting for the last:
`r_rms_kernel` 8.5 us for 4,096 values, the hyper-connection reductions and mix, the expert
combine (16 slots x 16 outputs a thread), the router weights. Each thread now issues up to 16 of
its loads together and then runs the same chain in the same order; RoPE takes a thread a pair.
Decode step 60.4 -> 59.5 ms on `chat`, 64.6 -> 63.5 ms on `long`; 3 of 3 tapes bit-equal.

### 12. Small kernels: launch the dependent first (`receipts/s12-trigger-first/`)

A small kernel waited for its predecessor and only then let its own dependent launch, so each
kernel of a chain of small ones was launched while the one before it already ran. They now
signal first and wait second (as the small-row GEMMs did): a whole chain and the GEMM after it
are launched while the GEMM before it streams, and the GEMM's first stages load during the
chain. Decode step 59.5 -> 58.9 ms on `chat`, 63.5 -> 62.9 ms on `long`; 3 of 3 tapes bit-equal.
`chat`'s 36-token prefill reads 107 -> 103 tokens/s; `long`'s is unchanged.

With steps 11 and 12 the profile with programmatic launch on
(`profile-chat-pdl-on-step10-vs-step12.txt`, `tools/excl.py`: a kernel's time is its end minus
the later of its start and its predecessor's end) has the RMS norms at 0.92 ms a step (2.56),
the expert combine 0.23 (0.68), the router weights 0.10 (0.51). A top-k by rank instead of the
radix select was tried with them and dropped: the same 0.60 ms a step.

### 13. Hyper-connection mix with its RMS norm (`receipts/s13-hc-rms/`)

Each hyper-connection mix was followed by the RMS norm of the row it wrote (82 launches a step).
`r_hc_mix_kernel` now runs that norm in the same block: `r_rms_kernel`'s chain over the values it
just wrote (a thread's elements in order, the block tree). The indexer's group logits run a
group's 16 head dots side by side (each in input order) instead of one after the other, and
the pooled keys' two weight rows load 16 inputs at a time. Decode step 58.9 -> 58.7 ms on `chat`,
62.9 -> 62.6 ms on `long`; 3 of 3 tapes bit-equal.

### 14. Small-row stage: operands first, BF16 two tiles (`receipts/s14-gemv-stage/`)

`ncu` on the small-row kernels (`kbench`): the routed experts ran 1.27 instructions a cycle an SM
with 0.39 eligible warps a scheduler, stalled on shared-memory loads and fixed waits, the tensor
pipe not throttled. A q-step loaded its code bytes, then the table entries they index, then
multiplied, one step after the other. A stage now loads the operands of up to 8 q-steps together
(weights, scales, table entries, split activations) and then runs their mma chain in q order;
the step count is a constant (a row is a whole number of tiles) and the FP8 block scale folds at
the stage's end, so the loop has no data-dependent branch. A BF16 stage takes two tiles (128
inputs, one 8 KiB copy), half the stage turns of a long BF16 row. Each output's mma sequence is
unchanged (`kbench`: every case equal to the tiled kernel).

`kbench.txt`: experts 108 -> 104 us, `qa` 33.5 -> 29.3 us and `kva` 30.6 -> 24.6 us with the
two-tile stage, the BF16 latent projections 5 to 10% faster. Decode step 58.7 -> 57.7 ms on
`chat`, 62.6 -> 61.7 ms on `long`; 3 of 3 tapes bit-equal.

### 15. Top-k of a small k (`receipts/s15-topk/`)

The router picks 16 of 512 with a radix select: four histogram passes, each with a serial scan
of 256 bins, 13.7 us a launch. For up to 32 picks of up to 1,024 values `r_topk_best_kernel`
takes the best live element k times (a thread's own elements, a warp reduction, the 8 warp
results), under the same order (larger value; equal values the lower index, or the higher with
reversed ties), and writes the picks in ascending index order: 11.6 us. Decode step 57.7 -> 57.5
ms on `chat`, 61.7 -> 61.5 ms on `long`; 3 of 3 tapes bit-equal (the router's picks are in the
tapes).

### 16. The router in one block (`receipts/s16-route/`)

After the router's logits a decode call launched seven small kernels: scores, top-k, weights, a
memset, the expert counts, the tile plan and the row placement, each about 2 us of launch on
top of its work. `r_route_kernel` runs the scores, the top-k rounds and the weights of a row in
one block, and for a single row also writes the experts' tile table and rows directly: every
pick is one expert with one row, so tile i is (pick i, row i, 1 row) and slot i's row is i, the
table the count, plan and place kernels build for those picks. The input gather then runs alone
(`memra_hy4r_moe_gather`). Forced picks (the gate's forcing tapes) keep the separate kernels.
Decode step 57.5 -> 57.1 ms on `chat`, 61.5 -> 61.0 ms on `long`; `chat`'s prefill 103 -> 106
tokens/s. 3 of 3 tapes bit-equal (`long` and `long-split` run the new kernel and record its
picks; `chat` forced runs the old path).

### 17. Prefill attention: every head on one copy of a key tile (`receipts/s17-attn-prefill/`)

`long`'s prefill spent 43% of its time in attention. A block scored 8 heads of a query against a
32-key tile it staged for itself, a warp a head: each lane loaded a key value once for one head.
`ncu` on the kernel (`ncu-attn.txt`): shared-memory load throughput bounds it (the memory pipes
83% busy, 0.93 instructions a cycle an SM, the top stall the memory instruction queue), five
shared loads for four multiply-adds in the score dot and 17 for 16 in the output sum.

`r_attn_all_kernel` (a prefill call: at least two waves of (query, chunk) units): one block a
(query, chunk) scores every head, 32 a pass, four heads a warp, so a lane loads a key value
once for four heads' chains. To leave shared memory for 32 queries the tile is staged a piece
at a time (four pieces of 128 latent values a key and the 64 rope values): the score dots piece
by piece in value order, the softmax step, then each latent piece's outputs. A head's sequence
for a tile (its score chain in value order, the tile's max and sums, each output's sum in key
order) is unchanged; a decode call keeps `r_attn_kernel`.

`kbench attn` (a slab of 256 queries, 64 heads, 512-key windows; `kbench.txt`): 9.2 ms with 8
heads a block -> 8.4 ms with all the heads a block and a warp a head -> 6.8 ms with two heads a
warp -> 5.4 ms with four; the 33.5 MB of outputs equal byte for byte in each. `long`'s prefill
319.6 -> 388.3 tokens/s; decode unchanged. 3 of 3 tapes bit-equal.

### 18. Prefill hyper-connection reductions over one read (`receipts/s18-hc-prefill/`, sampled)

A hyper-connection's nine reductions of a row ran as nine sets of blocks, each reading the row's
16,384 values again: fine for a decode call's one row, 1.2 GB of reads a launch for a 2,104-row
prefill. With 16 rows and up `r_hc_dots_all_kernel` runs every reduction of a row in the block
that reads its values once; a (row, reduction, thread) partial is the same chain. `long`'s
prefill 389.4 -> 403.8 tokens/s (sampled reading); decode unchanged. 3 of 3 tapes bit-equal.

Tried and dropped (kernels kept bit-equal, so only their speed decided): a persistent small-row
GEMM that walks its items as one stream of stages, with the A operands laid out for direct
loads and one-warp blocks for narrow outputs (+2.9 ms a step on `chat`); the routed experts'
NVFP4 codes decoded with byte permutes instead of a table (no gain, `kbench` 125 -> 131 us).

### 19. Prefill GEMM: only the live 16-row slices (`receipts/s19-gemm-live/`, sampled)

`nsys` of `long`'s prefill on step 18 (5.46 s of GPU time): the GEMMs are half of it (NVFP4
grouped experts 1.18 s, FP8 1.03 s, BF16 0.54 s), attention 1.67 s, its merge 0.29 s. Useful
rate (4 m n k: the hi and the lo mma of each product) 33 to 49 TF/s for FP8 and the experts, 36
to 44 for the BF16 projections; `ncu` (`kbench`, `KB_PRE`: the model's shapes at 2104 rows) has
the tensor pipe 33% active on the FP8 shapes, stalled on the global loads of the next K step
(2 blocks an SM at 172 registers) and on the block barriers.

`r_gemm_kernel` now skips the fragment loads and mma of a warp's 16-row slices past the tile's
rows (they held zero rows and wrote nothing): a grouped expert tile is often a few rows of 64.
`kbench` output hashes equal; `long`'s prefill 404.2 -> 406.7 tokens/s (interleaved, two runs
each), decode unchanged. 3 of 3 tapes bit-equal.

The o_proj gap seen in `kbench` (m1 232 GB/s, m4 280) is the first case of a process, not the
row count: with the rows in the order 4, 1, 1, 4 the first (m4) reads at 223 GB/s and the rest at
272 to 284; warm or with a 4 GiB eviction read the same. 280 GB/s is above the bus (273), so
`kbench`'s cold reads are partly cached; `ncu` (caches flushed, base clock) times m1 and m4 at
250 us alike. The model's o_proj at 232 GB/s reads at the bus's streaming rate.

### 20. Prefill GEMM: weights to BF16 by pairs (`receipts/s20-gemm-dequant/`, sampled)

`r_gemm_kernel`'s staging converts the weights as the small-row kernel does: FP8 code pairs by
the hardware pair conversion (E4M3 to F16 to BF16, exact), NVFP4 code bytes through a 256-entry
table of BF16 pairs times the group scale in BF16 (the product is exact, so it equals the F32
product rounded); the gate's red arm keeps the scalar path. `kbench` (`KB_PRE`) output hashes
equal; FP8 shapes 2 to 4% faster, the experts 3 to 4%. `long`'s prefill 406.3 -> 412.2
tokens/s (interleaved, two runs each), decode unchanged. 3 of 3 tapes bit-equal.

### 21. Prefill GEMM: ldmatrix fragments, 8 warps of 16 x 32 (`receipts/s21-gemm-warps/`, sampled)

`ncu` on the FP8 shapes: 172 registers a thread (the 32 x 32 warp tile's sums and its FP8 block
partials), 2 blocks an SM, the tensor pipe 33% active. Each block of 64 x 64 outputs now runs 8
warps of 16 x 32 (half the accumulators a thread, twice the warps an SM), and the fragments come
from ldmatrix (the values the per-lane loads read). Per-element sequence unchanged; `kbench`
output hashes equal at every arm. `kbench` (`kbench-*.log`): ldmatrix alone experts -3.5%, BF16
-3%; 8 warps of 32 x 16: FP8 o_proj (3072 x 16384) -14%, the per-head BF16 products (K 192, 512)
-34%, experts +2.5%; of 16 x 32: experts -4.4% and o_proj -12% more, the rest level.

Tried and dropped: FP8 at 3 or 4 blocks an SM by launch bounds (level; spills at 4, 2x slower);
a cp.async ring of each thread's raw operands, converted by the thread that copied them
(`kbench-pipelined-S22.log`: the global-load stalls go, but the 40-50 KB ring halves the blocks
an SM: BF16 and experts 1.3 to 1.8x slower, FP8 level); the attention merge's chunk weights once
a block instead of once an output (level, the bytes equal).

`long`'s prefill 412.1 -> 434.8 tokens/s, `chat`'s 108.0 -> 116.3 (interleaved vs step 20);
decode unchanged. 3 of 3 tapes bit-equal.

### 22. Prefill attention: a pass's queries resident, the next piece read ahead (`receipts/s22-attn-prefetch/`, sampled)

`ncu` on `r_attn_all_kernel` (step 21 with the output weights read four keys at a time,
`ncu-opcodes-S30.txt`): the shared stores stalled 32% of the samples at 2.7% of the
instructions, waiting on the global reads before them. Every piece of a key tile was read
from global memory and stored in the same step, and the 32 queries of a pass (74 KB) were read
again for every tile.

Now a pass (32 heads) walks the chunk's tiles with its queries resident (read once, 16-byte
reads), a tile is nine piece steps (the dots of pieces 0 to 4, the outputs of 3 to 0), and each
step's next piece is read into registers while the current one is used. The output sums read a
head's weights of four keys in one broadcast read. A pass keeps only its own heads' sums (the
first try kept both passes' sums, 254 registers, and spilled: 30% slower,
`kbench-attn-S30-S31.log`). Each head's sequence is unchanged; the slab's 33.5 MB equal byte
for byte. `kbench attn` slab 5.43 -> 5.32 ms (weights by four) -> 3.99 ms. `long`'s prefill
435.5 -> 472.9 tokens/s (interleaved vs step 21), `chat` (36 tokens, the decode kernel) and
decode unchanged. 3 of 3 tapes bit-equal.
