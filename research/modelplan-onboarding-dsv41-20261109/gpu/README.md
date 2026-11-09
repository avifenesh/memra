# DeepSeek-V4.1-Flash onboarding, step 2 (GPU): single-card kernels and layers (2026-11-09)

Verdict: every V4.1 operation memra lacked now has an sm_120a kernel in `cu/dsv41_gpu.cu`, and
`crates/memra-engine/src/dsv41_gpu.rs` runs real checkpoint layers from them on one card, at
prefill and at decode. On random inputs, 73 of 73 kernel checks pass against the vendor program.
On the checkpoint, six trunk layers (Engram, the encoder's ratio-2 sources, the decoder's ratio-1
source and candidate pool, a Reuse layer and a Reindex layer) land inside the vendor's own
CPU-vs-GPU band on every layer-level metric: 161 of 161 checks pass. All 40 trunk layers chained
on one card, each layer's weights loaded in turn, give the vendor's next token with KL 2.9e-8
(16-token prompt) and 7.6e-7 (3,761-token prompt). The support state stays none: this is a
kernel-and-layer surface, not a qualified model path.

Hardware: one RTX PRO 6000 Blackwell Max-Q (sm_120a, 96 GB, 250 W cap), CUDA 13.4, rustc 1.97.1.
Single runs, not timed for publication.

## What the kernels compute

The table and the numeric program are in `docs/KERNELS.md` ("DeepSeek-V4.1 single-card kernels").
In short: FP8 32x32 dense and MXFP4 expert GEMMs on e4m3 `m16n8k32` MMA, one MMA per 32-wide K
block from zero, scaled into a separate fp32 accumulator (the reference `fp8_gemm` / `fp4_gemm`
schedule); per-32 power-of-two activation quantization; fp4 g32/ue8m0 and g16/e4m3
quantize-dequantize with ties to the even mantissa; RMSNorm and RoPE in fp32 with bf16 storage;
sink sparse attention with the tilelang schedule (key blocks of 64, online max, bf16 P into PV,
fp32 denominator); the CSA2 compressor at ratio 1 and 2; the indexer score with bf16 rounding where
the reference rounds; exact radix top-k with lowest-position ties; the candidate pool; the
384-expert sqrtsoftplus router with selection bias; clamped SwiGLU; single-pass mHC generic in d;
Engram dequant and gate. Engram hashing, the pinned-staging gather and the H2D copy run on the host
(`dsv41_gpu/engram.rs`).

Two choices the measurements made:

- fp32-output linears (the ratio-2 compressor's `wkv` and `wgate`, the router) run as an fp32 FMA
  chain on CUDA cores, not on tensor cores. Recomputed in float64 from the capture's input, the
  layer-2 latent of the tensor-core path was 2.66e-4 rel L2 from the truth, against 4.69e-5 for the
  reference's own fp32 GEMM (layer 14: 2.58e-4 vs 4.57e-5; `analyze_compressor.py`). The FMA path
  brought the random-input latent from 4.0e-4 to 8.5e-5 and the router weights from 1.1e-6 to 2.0e-7.
- The minted NVFP4 experts (paired e4m3 per-16 scales times one global scale, the `dsv4_mint`
  cast) run through the same GEMM as the MXFP4 experts as stored. On layer 20 with all 384 real
  experts the minted path writes byte-identical stream, pre-mix, MoE output and routing
  (`layer-needle-v2-mint20`). The default loads MXFP4 as stored; the mint costs nothing in accuracy
  and buys nothing on this kernel, so it matters only to a kernel that wants NVFP4 hardware scales.

## Kernel gate (random inputs)

`dsv41-gate kernels` on fixtures written by `make_fixtures.py`: random inputs at the real sizes run
through the vendor modules (`model.py` with the oracle's torch stand-ins for its tilelang kernels).
Receipt: `receipts/kernels-rand.json` / `.log`. 73 checks, 0 failed.

| op | result | rule and why |
|---|---|---|
| act_quant, codes / scales / in-place | bit-exact | deterministic arithmetic |
| fp8 32x32 GEMM, 6 shapes (M 1..200, N up to 25,600, K up to 8,192) | rel L2 0 to 3.0e-5, 99.986-100% exact | rel <= 2e-4, max abs <= 0.03 x rms: fp32 summation order is the only difference |
| fp4 g32 ue8m0 / g16 e4m3, ties included | bit-exact | |
| RMSNorm d 5120/1280/512/128 | rel <= 4.1e-5, <= 1 ULP | order only |
| RoPE tables (host) vs vendor | rel <= 7.1e-9 | f32 cos/sin from an f64 evaluation |
| RoPE forward / inverse | bit-exact | |
| sparse attention vs the tilelang schedule | rel 1.25e-4 / 7.8e-5 | rel <= 5e-4: order only |
| sparse attention vs a global-max stand-in | rel 1.8e-3 / 8.4e-4 | rel <= 3e-3: P rounded to bf16 against another max; the stand-in itself sits 2.3e-3 from fp32-P attention |
| mHC mixes, pre, post, comb | rel <= 8.2e-7 | fp32 |
| mHC collapse / expand | bit-exact | |
| compressor latent, ratio 1 / 2 | rel 1.1e-4 / 8.5e-5 | rel <= 1e-3 |
| compressed KV after fp4 g16 | 1 of 18,944 / 0 of 9,216 elements off the reference grid point | <= 1%: one bf16 ULP in can move one fp4 step |
| indexer q (fp8 linear, RoPE, fp4), score | bit-exact on the fixture | |
| candidate pool, top-k (prefill and two decode steps, two layers) | equal up to ties | tie-aware: same set above the k-th score, same count at it |
| router ids / weights | identical / rel 2.0e-7 | |
| MXFP4 expert with clamps engaged / shared FP8 expert | rel 1.7e-6 / 7.8e-5 | |
| minted NVFP4 expert vs MXFP4 | bit-exact | |
| full MoE (16 experts, top-6) | rel 3.8e-5 | |
| Engram dequant / wkv / gate | bit-exact / rel 3.9e-5 / bit-exact | |

## Layer gate (checkpoint)

`capture_layers.py` runs the vendor model on one card with hooks and records, per layer: the
stream in and out, the carried pre-mix, the CSA2 state the layer reads (compressed KV, index keys,
selection, candidate pool) and the per-op intermediates; it reproduces the reference tapes bit for
bit. `layer_floor.py` reruns chosen layers on the CPU (torch CPU GEMMs, the same stand-ins) from the
capture's inputs: the vendor program on another backend with identical inputs. A layer-level check
passes when the GPU rewrite sits at most 2x as far from the GPU vendor output as the CPU vendor does
(plus 1e-4); routing passes when its expert-set agreement is within 1% of the floor's.

Prompt: the 3,761-token needle case, tilelang attention schedule in the oracle (`kernel.py` sha256
`fb33c6ea91c4e1ab...`). Receipt: `receipts/layer-needle-v2.json` / `.log`; per-token distributions
in `receipts/compare-needle-v2.json`. 161 checks, 0 failed.

Prefill, rel L2 against the GPU vendor (rewrite / CPU floor), route agreement (rewrite / floor):

| layer | stream out | pre-mix out | attention out | MoE out | routes | attention index Jaccard |
|---|---|---|---|---|---|---|
| 1 (Engram, window) | 4.12e-03 / 4.64e-03 | 5.50e-04 / 6.04e-04 | 3.81e-03 / 4.41e-03 | 1.34e-02 / 1.55e-02 | 98.8% / 98.2% | 1.00000 |
| 2 (ratio-2 source) | 7.41e-03 / 9.87e-03 | 2.65e-04 / 6.28e-04 | 8.75e-03 / 1.91e-02 | 3.89e-02 / 5.40e-02 | 96.9% / 93.8% | 0.99994 |
| 14 (Engram, ratio-2 source) | 7.71e-03 / 1.13e-02 | 6.67e-04 / 1.44e-03 | 9.66e-03 / 2.17e-02 | 2.60e-02 / 3.75e-02 | 98.2% / 95.7% | 0.99989 |
| 20 (ratio-1 source, candidate pool) | 5.02e-04 / 1.02e-03 | 3.39e-04 / 1.46e-03 | 5.86e-03 / 2.63e-02 | 8.81e-04 / 1.78e-03 | 98.8% / 94.0% | 0.99995 |
| 21 (Reuse) | 3.35e-04 / 4.07e-04 | 8.08e-04 / 7.55e-04 | 5.04e-03 / 4.86e-03 | 6.55e-04 / 7.59e-04 | 99.2% / 99.1% | 1.00000 |
| 24 (Reindex) | 1.00e-04 / 2.11e-04 | 3.80e-04 / 1.47e-03 | 5.82e-03 / 2.59e-02 | 9.01e-04 / 1.91e-03 | 99.0% / 94.5% | 0.99993 |

The Engram hash ids match the vendor's `NgramHashState` exactly (90,264 at each Engram layer's
prefill, 24 per decode step). Most of the attention-output difference is the fp8 quantize-dequantize
of the attention output before `wo_a`: a 1e-3 difference in the output flips about 2% of e4m3
roundings, about one 9% step each. Most of the MoE difference is routing: a token whose top-6 set
moves changes its output by 0.3-0.5. Both are the vendor program's own sensitivity, and the CPU
floor shows them larger than the rewrite.

Decode, two steps after the prefill (positions 3,761 and 3,762), from each layer's own window
ring, compressor pair and caches. Decode is one token, so the band is the floor's prefill per-token
p99 for the same tensor (rel L2 rewrite / band):

| layer | step | stream out | MoE out | attention index Jaccard |
|---|---|---|---|---|
| 1 | 1 | 7.56e-05 / 3.99e-02 | 1.70e-04 / 1.53e-01 | 1.00000 |
| 1 | 2 | 3.10e-08 / 3.99e-02 | 0.00e+00 / 1.53e-01 | 1.00000 |
| 2 | 1 | 7.72e-03 / 9.74e-02 | 2.31e-02 / 2.92e-01 | 1.00000 |
| 2 | 2 | 1.20e-02 / 9.74e-02 | 2.88e-02 / 2.92e-01 | 1.00000 |
| 14 | 1 | 1.41e-02 / 1.33e-01 | 3.15e-02 / 2.49e-01 | 1.00000 |
| 14 | 2 | 1.26e-02 / 1.33e-01 | 2.52e-02 / 2.49e-01 | 1.00000 |
| 20 | 1 | 9.39e-03 / 7.33e-02 | 3.45e-02 / 2.68e-01 | 1.00000 |
| 20 | 2 | 9.55e-03 / 7.33e-02 | 2.93e-02 / 2.68e-01 | 1.00000 |
| 21 | 1 | 4.27e-03 / 1.76e-02 | 1.36e-02 / 5.07e-02 | 1.00000 |
| 21 | 2 | 5.65e-03 / 1.76e-02 | 2.31e-02 / 5.07e-02 | 1.00000 |
| 24 | 1 | 0.00e+00 / 6.80e-02 | 0.00e+00 / 2.90e-01 | 1.00000 |
| 24 | 2 | 7.16e-03 / 6.80e-02 | 1.59e-02 / 2.90e-01 | 1.00000 |

Every decode step routes to the vendor's expert set. Step 1 closes a ratio-2 pair opened by the
prefill's odd tail, and step 2 opens the next.

## The whole trunk on one card

`--chain 40 --head` runs layers 0..39 back to back on the rewrite's own outputs, one layer's
weights on the card at a time, then the collapse, the final norm and the fp32 head on the last
token:

| prompt | next token | top-5 overlap | KL(vendor, rewrite) | stream at layer 39, rel L2 / cos |
|---|---|---|---|---|
| capital (16 tokens) | equal (51119) | 5/5 | 2.86e-8 | 7.5e-3 / 0.999986 |
| needle (3,761 tokens) | equal (28224) | 4/5 | 7.59e-7 | 0.52 / 0.871 |

Receipts: `receipts/chain-capital-v2-full.*`, `receipts/chain-needle-v2-full.*`. The long prompt's
stream drifts by 1-7% through layer 35, then jumps at layer 36 (rel 0.29, one token's error 429x
its own norm). Layer 36 alone, from the capture's input, sits at rel 4.0e-3
(`receipts/layer-needle-v2-35-37.*`), so the jump comes from what reaches it, not from layer 36's
kernels. The vendor program does the same to itself: the reference tape made with a global-max
attention stand-in against the capture made with the tilelang schedule, both the vendor model on the
same card, drifts 1-8% to layer 35, jumps to 0.22 (one token 409x) at layer 36 and ends at 0.44 /
0.900 (`chain_floor.py`, `receipts/chain-floor-needle-schedules.json`). A forward is judged at the
head, where both agree.

## Commands

```bash
# fixtures (CPU): random inputs through the vendor modules
python3 make_fixtures.py ORACLE_DIR REF_DIR fixtures/rand
# per-layer capture of the vendor program (one GPU) and the CPU floor from it
python3 capture_layers.py ORACLE_DIR CKPT REF TAPES captures --ids capital,needle \
  --layers 0,1,...,39 --ops 1,2,3,14,20,21,24 --decode 2
python3 layer_floor.py ORACLE_DIR CKPT REF captures/needle captures/needle-floor --layers 1,2,14,20,21,24
# gates (memra-engine, sm_120a release build)
dsv41-gate kernels fixtures/rand --out kernels-rand.json
dsv41-gate layer CKPT captures/needle --layers 1,2,14,20,21,24 --decode 2 \
  --engram-consts ORACLE_DIR --floor captures/needle-floor --save mine --out layer-needle-v2.json
dsv41-gate layer CKPT captures/needle --layers 20 --mint --floor captures/needle-floor --save mint
dsv41-gate layer CKPT captures/capital --layers 0 --chain 40 --head --engram-consts ORACLE_DIR
python3 compare_dumps.py captures/needle captures/needle-floor mine compare-needle-v2.json
```

`ORACLE_DIR` holds the vendor-code oracle: its torch stand-ins for the tilelang kernels and its
loader (`run_oracle.py`, whose loader `capture_layers.py` and `layer_floor.py` execute unchanged),
plus the Engram constants the vendor code derives (`engram_constants.json` and its token map). The
fixtures (0.8 GB) and captures (about 23 GB) are not in the repo.

## Files

- `fixio.py`: the fixture format (raw tensors + `index.json`).
- `make_fixtures.py`, `add_sched_ref.py`: random-input fixtures; the tilelang-schedule attention reference.
- `capture_layers.py`, `layer_floor.py`: per-layer vendor capture and its CPU floor.
- `compare_dumps.py`, `chain_floor.py`, `analyze_compressor.py`: the analyses quoted above.
- `dev_kernels.py`: the ctypes development loop over a standalone build of `cu/dsv41_gpu.cu`.
- `receipts/`: gate receipts (JSON) and their logs.

## Not done here

- Multi-card: the full model needs 4 cards (about 313 GB on-card, 77 GB each at TP4/EP4). Nothing
  here is partitioned; `LayerW` loads one whole layer.
- Serving: no `Engine` integration, no scheduler, no CUDA graphs; the decode path is per-layer and
  syncs per routed expert.
- Speed: the GEMM tiles are plain (no cp.async pipeline, no split-K for M=1), the attention runs
  one block per (query, 16 heads), the MoE launches per expert. No performance claim is made.
- DSpark (`mtp.*`) blocks and the vision tower: not run.
- Plan surfaces: the ten `deepseek_v41` plan operations still list no surface; registering these
  kernels as a rewrite comes with the multi-card route and its rewrite receipts.
