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

Step 4:

```bash
# the native checkpoint runner, and CheckpointParity through the CLI (on the pack's capital probe)
MEMRA_FULL_PREC=1 dsv41-run CKPT 0 128803 ... --token-map TOKEN_MAP --last-logits native.f32
MEMRA_DSV41_TOKEN_MAP=TOKEN_MAP memra model verify checkpoint CKPT --against deepseek_v41 \
  --oracle vendor-cuda-capital.tsv --native-runner dsv41-run --out verify-capital
# long context: light captures, CPU floors with the row-sliced indexer, layer gates, whole trunk
python3 capture_layers.py ORACLE_DIR CKPT REF TAPES captures --ids needle1200 --layers 20,21,24,25,28 \
  --ops 20,21,24,25,28 --light --index-qchunk 512
python3 layer_floor.py ORACLE_DIR CKPT REF captures/needle1200 captures/needle1200-floor \
  --layers 20,21,24,25,28 --index-qchunk 512 --max-seq 20000
dsv41-gate layer CKPT captures/needle1200 --layers 20,21,24,25,28 --token-map TOKEN_MAP \
  --floor captures/needle1200-floor
MEMRA_FULL_PREC=1 dsv41-run CKPT --ids needle16800.ids.i32 --token-map TOKEN_MAP --last-logits n16800.f32
MEMRA_FULL_PREC=1 dsv41-run CKPT --ids tf128k_en_moby.ids.i32 --token-map TOKEN_MAP --tf tf128k/
python3 trunk_vs_tape.py TAPES/tf128k_en_moby.pt tf128k/
# per-position KL and paired top-1 against the vendor tape, for the native run and each floor
python3 tf_kl.py TAPES/tf128k_en_moby.pt tf128k/ --json kl-native-tf128k_en_moby.json
python3 tf_kl.py TAPES/tf128k_en_moby.pt GLOBALMAX/tf128k_en_moby.pt --json kl-schedule-tf128k_en_moby.json
python3 tables_kl.py RECEIPTS
python3 paired_top1.py TAPES/tf128k_en_moby.pt tf128k/ GLOBALMAX/tf128k_en_moby.pt
# decode steps: prompt plus generated tokens as one teacher-forced prefill, native and vendor
python3 decode_vs_tf.py prep dec/ TAPES/decode32k_he_summary.pt TAPES/decode64k_en_continue.pt
python3 make_tfids_oracle.py ORACLE_DIR ORACLE_TFIDS
MEMRA_FULL_PREC=1 dsv41-run CKPT --ids dec/decode32k_he_summary.ids.i32 --token-map TOKEN_MAP --tf dec32k/
python3 decode_vs_tf.py cmp TAPES/decode32k_he_summary.pt dec32k/
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
- `logit_band.py`: logit-level spread of two runs of one prompt (the checkpoint bands).
- `trunk_vs_tape.py`: a whole-trunk run, or another vendor tape, against a vendor tape (next-token or
  teacher-forced metrics).
- `make_globalmax_oracle.py`: the oracle copy with global-max attention (the schedule floor).
- `make_tfids_oracle.py`: the oracle copy whose teacher-forced cases take token ids (decode floors).
- `tf_kl.py`, `paired_top1.py`: per-position KL from both sides' top-32, per entropy subset; paired
  top-1 sign test of the native run against a floor.
- `decode_vs_tf.py`: a decode tape's steps against a teacher-forced prefill of the same tokens.
- `tables_step4.py`, `tables_kl.py`: the step-4 tables from the receipts.
- `receipts/checkpoint/`, `receipts/long/`: step-4 receipts.
- `receipts/`: gate receipts (JSON) and their logs.

## CheckpointParity on one card (step 4)

Verdict: `memra model verify checkpoint --against deepseek_v41` passes with the one-card runner as
the native side, on a pack-declared real chat probe, inside a band set from the vendor's own
CPU-vs-GPU spread. The candidate pool's long-context layers pass against the CPU floor at 18k and
73k tokens, the whole trunk gives the vendor's next token at 18k, 73k, 128k and 257k tokens, and
five teacher-forced texts (2k to 262k tokens) score inside the vendor program's own floors, in mean
NLL and in per-position KL on the positions where the vendor is unsure.

`dsv41-run` (crates/memra-engine/src/bin/dsv41_run.rs) is the pack's native checkpoint runner: the
embedding, the 40 trunk layers each loaded from the safetensors mmap, run and freed (about 7.5 GB a
layer with its 384 MXFP4 experts, so the 510 GB checkpoint runs on one 96 GB card), the collapse
with the carried pre-mix, the final norm and the fp32 head. It runs only with `MEMRA_FULL_PREC=1`
(the route has one numeric program, the checkpoint's), derives the Engram primes and multipliers
with `memra_gguf::dsv41_forward` (the bit-exact numpy port) and takes the compressed token map as an
input (`--token-map` or `MEMRA_DSV41_TOKEN_MAP`). It writes the `memra-checkpoint-oracle-v1` tape
as engine `memra-native-full-precision`, numeric class `deepseek-v41-source-program`; the vendor
side is `model.py` on CUDA with the oracle's tilelang-schedule stand-ins
(`oracle_to_checkpoint_tape.py --numeric-class deepseek-v41-source-program`). Long prompts run in
query slices of 8,192 rows (`Ctx::chunk`): every sliced step acts per row, and the 3,761-token
layers rerun with 1,024-row slices write byte-identical tensors (43 of 43,
`receipts/long/chunk-needle-v2.cmp`).

The probe. On the default tokens 1,2,3,4 (`<eos> <pad> ! "`) the vendor's own CPU and GPU runs of the
checkpoint program differ by up to 13.32 logits (KL 0.49, top-5 overlap 3/5, hidden states 0.45
apart by layer 38, same argmax); the native runner sits at 11.21 from the GPU vendor (KL 0.31). No
band wide enough for the vendor's two backends can tell programs apart there, so the pack declares
the `capital` chat turn (16 tokens) as its checkpoint probe (`checkpoint_probe`, docs/ONBOARDING.md);
the probe's sha256 is in the lock and the receipt.

| capital probe (16 tokens) | max abs logit diff | KL(vendor GPU, other) | top-5 | argmax |
|---|---|---|---|---|
| vendor CPU vs vendor GPU (the band's source) | 3.18 | 4.7e-7 | 4/5 | equal (51119) |
| native GPU vs vendor GPU | 2.14 | 2.9e-8 | 5/5 | equal (51119) |

The class band is atol 6.4 at rtol 0 plus argmax: twice the vendor's own CPU-vs-GPU spread, the
factor the layer gates use. `memra model verify checkpoint DeepSeek-V4.1-Flash-1017 --against
deepseek_v41 --oracle vendor-cuda-capital.tsv --native-runner dsv41-run` passes
(`receipts/checkpoint/verify-capital-rel/`, max_abs 2.14, Config, TokenizerTemplate, TensorCensus and
CheckpointParity passed).

## Long context (step 4)

The candidate pool prunes once a row sees more than 2,048 blocks of 8 positions, i.e. past 16,384
tokens. Captures at 18,343 (`needle1200`) and 73,520 (`needle4800`) tokens record layers 20 (ratio-1
source, candidate source), 21 (Reuse), 24 and 28 (Reindex inside the pool) and 25 (Reuse after a
Reindex); the floor is the same layers rerun on the CPU from the captured inputs (`layer_floor.py
--index-qchunk 512`, the oracle's row-sliced indexer).

Rel L2 against the GPU vendor capture (native / CPU floor), and the share of tokens routed to the
same expert set (native / CPU floor); every check passes the 2x-floor rule
(`receipts/long/layer-needle1200.*`, `layer-needle4800*.*`, 42 checks):

| prompt | layer | stream out | attention out | MoE out | routes (same expert set) |
|---|---|---|---|---|---|
| 18,343 | 20 | 1.25e-03 / 2.62e-03 | 8.06e-03 / 3.31e-02 | 2.31e-03 / 4.79e-03 | 98.1% / 90.7% |
| 18,343 | 21 | 5.91e-04 / 5.26e-04 | 4.27e-03 / 3.99e-03 | 1.23e-03 / 1.08e-03 | 99.0% / 99.2% |
| 18,343 | 24 | 1.99e-04 / 4.85e-04 | 5.40e-03 / 3.40e-02 | 1.96e-03 / 4.87e-03 | 98.9% / 92.0% |
| 18,343 | 25 | 1.56e-04 / 1.40e-04 | 4.68e-03 / 4.95e-03 | 1.71e-03 / 1.47e-03 | 98.9% / 99.2% |
| 18,343 | 28 | 1.16e-04 / 4.29e-04 | 3.84e-03 / 3.17e-02 | 9.09e-04 / 2.47e-03 | 98.8% / 89.7% |
| 73,520 | 20 | 2.25e-03 / 5.92e-03 | 7.32e-03 / 3.53e-02 | 4.38e-03 / 1.09e-02 | 98.3% / 89.0% |
| 73,520 | 21 | 1.06e-03 / 9.38e-04 | 4.02e-03 / 4.34e-03 | 2.46e-03 / 2.18e-03 | 99.0% / 99.2% |
| 73,520 | 24 | 3.53e-04 / 9.09e-04 | 5.25e-03 / 3.58e-02 | 3.67e-03 / 9.35e-03 | 99.0% / 92.2% |
| 73,520 | 25 | 2.67e-04 / 2.26e-04 | 4.99e-03 / 4.88e-03 | 3.63e-03 / 3.04e-03 | 98.7% / 99.1% |
| 73,520 | 28 | 2.38e-04 / 7.02e-04 | 4.35e-03 / 3.30e-02 | 2.06e-03 / 5.26e-03 | 98.6% / 88.8% |

At both lengths the native layers route closer to the GPU vendor than the CPU vendor does, the
Reindex layers 24 and 28 included, where the pool now removes blocks. The first 73k run gated
layers 25 and 28 against a fixed band (their floor was still running) and their routing missed a
fixed 99% bar at 98.7% and 98.6%; against their own CPU floor (99.1% and 88.8%) they pass.

Whole trunk, last prompt token against the vendor tape (`trunk_vs_tape.py`):

Every argmax matches. The global-max floor is the same vendor program with P rounded against one
global max per row (`make_globalmax_oracle.py`), run on the same card: at 128k the vendor disagrees
with itself by KL 1.3e-4, the native by 2.2e-4; at 257k by 1.4e-4 against the native's 5.3e-4. One
position is one sample of a wide distribution: between the same two vendor realizations the
per-position KL of the teacher-forced texts below has a mean of 1.3e-2 to 1.8e-2 and a p99 of 0.15
to 0.36, so the 257k verdict rests on the teacher-forced 262k text further down. The runner held the
257k prompt on one card (10.5 GB per stream copy, the candidate pool at 8.4 GB).

| prompt | tokens | next token | top-5 | KL(vendor, native) | KL floor (global-max vendor) | max abs diff | whole trunk |
|---|---|---|---|---|---|---|---|
| needle1200 | 18,343 | equal (17227) | 4/5 | 3.5e-06 | - | 2.53 | 42 s |
| needle4800 | 73,520 | equal (21391) | 4/5 | 6.0e-07 | - | 2.21 | 80 s |
| needle8400 | 128,745 | equal (20982) | 4/5 | 2.2e-04 | 1.3e-04 | 2.99 | 121 s |
| needle16800 | 257,185 | equal (10186) | 3/5 | 5.3e-04 | 1.4e-04 | 2.28 | 238 s |

Teacher-forced texts, every position scored (`--tf`): mean NLL, top-1 agreement and the share of
positions whose NLL moves more than 0.1 nats, against two floors of the vendor program itself: the
same text as a shorter prefix (the length effect: different GEMM shapes, same program) and the
global-max attention realization at the same length (another valid schedule, `make_globalmax_oracle.py`).
A text passes when |delta mean NLL| is at most twice the largest floor's, top-1 agreement at most
1 point under the lowest floor's, and the moved share at most 2 points over the highest floor's.

| text | tokens | mean NLL vendor / native | delta | top-1 agree | moved > 0.1 | floor delta (length, schedule) | floor top-1 | floor moved | pass |
|---|---|---|---|---|---|---|---|---|---|
| tf_en_lighthouse | 2,048 | 0.4041 / 0.4058 | +0.0017 | 97.4% | 12.3% | -0.0063 | 97.9% | 11.6% | PASS |
| tflong_he_haifa | 31,859 | 1.4920 / 1.4835 | -0.0086 | 94.8% | 30.1% | -0.0039, -0.0062 | 94.9% | 30.2% | PASS |
| tf128k_en_moby | 131,072 | 0.0743 / 0.0646 | -0.0097 | 99.1% | 4.3% | -0.0022, -0.0081 | 99.1% | 4.3% | PASS |
| tf64k_he_cities | 131,072 | 1.2295 / 1.2313 | +0.0018 | 95.5% | 26.9% | +0.0016, -0.0001 | 95.5% | 26.8% | PASS |
| tf256k_en_moby | 262,144 | 0.0826 / 0.0741 | -0.0085 | 99.0% | 4.7% | +0.0022, -0.0097 | 98.9% | 4.7% | PASS |

The 262k text is the 131k one (Moby Dick from chapter 1) scored twice as far; its length floor is the
131k tape, on the first 131,072 positions. The native run sits at the vendor's own floors on every text. The vendor tape at full length is
itself the outlier on mean NLL at 32k and 131k English: its global-max realization scores the same
text 0.0062 and 0.0081 nats lower, and the native 0.0086 and 0.0097 lower. The 2048-vs-2422 length
effect measured earlier (delta 0.0006, top-1 98.2%, 11% moved) is the length floor of one text at
2k under the earlier attention stand-in; at 2k under the tilelang schedule the global-max floor is
delta 0.0063, top-1 97.9%, 11.6% moved, and at 32k to 131k the Hebrew texts move 27-30% of
positions by more than 0.1 nats between two valid runs of the vendor program.

Per-position KL (`tf_kl.py`). `dsv41-run --tf` also keeps every position's 32 largest logits, as the
tape does. At each position both distributions are known exactly on the ids in both top-32 sets, and
everything else is one rest bucket; the KL of the two coarsened distributions is a lower bound of the
full KL(vendor || other), computed the same way for the native run and for each vendor floor. The
English text is memorized (mean NLL 0.07 to 0.08), so most of its positions are near one-hot and
agree under any realization; the signal is where the vendor is unsure, so each text is also cut to
the positions whose vendor entropy (top-32 plus the rest bucket, a lower bound) exceeds 1 nat and to
its top entropy decile. Mean KL against the vendor tape, native and floor on the same positions (the
schedule floor on every position, the length floor on the shorter tape's prefix with the native cut
to it), and top-1 agreement (`tables_kl.py`, `receipts/long/kl-*.json`, `kl-tables.md`). A subset
passes on the larger floor's band, as the NLL rule: mean KL at most twice that floor's, top-1 at most
1 point under it, or two binomial standard errors when the subset is small (the 2k text's subsets
have about 160 positions).

| text | positions | native KL | schedule floor KL | ratio | native KL, prefix | length floor KL | ratio | top-1: native, schedule; native, length | pass |
|---|---|---|---|---|---|---|---|---|---|
| tf_en_lighthouse (2k) | all: 2,048 | 2.23e-02 | 1.76e-02 | 1.26 | - | - | - | 97.4%, 97.9% | PASS |
| | entropy > 1 nat: 159 | 1.06e-01 | 7.35e-02 | 1.44 | - | - | - | 81.1%, 83.0% | PASS |
| | top decile: 205 | 1.14e-01 | 7.95e-02 | 1.44 | - | - | - | 81.5%, 84.4% | PASS |
| tflong_he_haifa (32k) | all: 31,859 | 1.66e-02 | 1.68e-02 | 0.99 | 1.64e-02 | 1.66e-02 | 0.98 | 94.8%, 95.0%; 94.9%, 94.9% | PASS |
| | entropy > 1 nat: 11,387 | 3.34e-02 | 3.38e-02 | 0.99 | 3.27e-02 | 3.28e-02 | 1.00 | 87.4%, 87.6%; 87.7%, 87.8% | PASS |
| | top decile: 3,186 | 2.99e-02 | 3.10e-02 | 0.96 | 2.95e-02 | 2.97e-02 | 0.99 | 81.6%, 81.9%; 81.5%, 82.4% | PASS |
| tf128k_en_moby (131k) | all: 131,072 | 1.34e-02 | 1.32e-02 | 1.02 | 1.33e-02 | 9.01e-03 | 1.47 | 99.1%, 99.1%; 99.1%, 99.2% | PASS |
| | entropy > 1 nat: 1,322 | 3.08e-01 | 2.94e-01 | 1.05 | 3.15e-01 | 1.99e-01 | 1.58 | 71.5%, 71.5%; 71.4%, 73.3% | PASS |
| | top decile: 13,108 | 1.27e-01 | 1.24e-01 | 1.02 | 1.26e-01 | 8.54e-02 | 1.47 | 90.6%, 90.7%; 90.8%, 91.6% | PASS |
| tf64k_he_cities (131k) | all: 131,072 | 1.49e-02 | 1.44e-02 | 1.04 | 1.50e-02 | 1.40e-02 | 1.07 | 95.5%, 95.5%; 95.4%, 95.5% | PASS |
| | entropy > 1 nat: 43,069 | 3.23e-02 | 3.17e-02 | 1.02 | 3.23e-02 | 3.07e-02 | 1.05 | 88.0%, 88.1%; 88.0%, 88.2% | PASS |
| | top decile: 13,108 | 3.23e-02 | 3.17e-02 | 1.02 | 3.21e-02 | 3.00e-02 | 1.07 | 82.1%, 82.5%; 82.0%, 82.5% | PASS |
| tf256k_en_moby (262k) | all: 262,144 | 1.31e-02 | 1.41e-02 | 0.93 | 1.11e-02 | 8.79e-03 | 1.27 | 99.0%, 98.9%; 99.1%, 99.1% | PASS |
| | entropy > 1 nat: 3,057 | 2.65e-01 | 2.83e-01 | 0.94 | 2.71e-01 | 1.88e-01 | 1.44 | 70.6%, 69.2%; 69.4%, 72.0% | PASS |
| | top decile: 26,215 | 1.23e-01 | 1.32e-01 | 0.93 | 1.06e-01 | 8.19e-02 | 1.29 | 89.8%, 89.5%; 91.0%, 91.4% | PASS |

The 257k verdict rests on the 262k text's high-entropy positions: against the vendor's global-max
realization at the same length the native run is 0.94x its KL on the 3,057 positions above 1 nat and
0.93x on the top decile (26,215), with top-1 70.6% against 69.2%. On the non-memorized Hebrew texts
both floors coincide and the native sits at 0.96x to 1.07x of either on every subset. On Moby the
length floor is the tighter one at both lengths (the native is 1.27x to 1.58x of it, inside 2x); on
its 1,229 prefix positions above 1 nat the native's top-1 is 2.6 points under the length floor's
(69.4% against 72.0%), 0.1 point past that floor's own band, so that subset passes on the schedule
floor alone. Paired, position by position (`paired_top1.py`, `receipts/long/paired-top1.txt`): of
the positions where exactly one of the two agrees with the vendor, the native wins 138 and the length
floor 170 (sign test p=0.077); against the schedule floor at 262k the native wins 352 to 309
(p=0.10). Against every floor of the Hebrew texts p is 0.19 or more.

Decode steps at 41k and 64k (`decode_vs_tf.py`). Two vendor decode tapes (a 41,098-token Hebrew
summary and a 64,443-token English continuation, 96 and 92 greedy tokens, full logits on the first 8
steps) put each decode step at a position of one teacher-forced prefill of the prompt plus the
generated tokens. The native prefill and the vendor's own (`make_tfids_oracle.py`, the oracle taking
token ids for a teacher-forced case) are both compared with the vendor's decode steps:

| decode tape | prefill | top-1 = vendor decode | first disagreement | KL on the 8 full-logit steps, mean / max |
|---|---|---|---|---|
| decode32k_he_summary | native | 90 / 96 | step 13 | 2.5e-04 / 1.1e-03 |
| decode32k_he_summary | vendor | 93 / 96 | step 13 | 1.0e-03 / 7.2e-03 |
| decode64k_en_continue | native | 86 / 92 | step 4 | 2.4e-02 / 1.3e-01 |
| decode64k_en_continue | vendor | 85 / 92 | step 4 | 1.2e-02 / 4.1e-02 |

The native prefill reproduces the vendor's decode as closely as the vendor's own prefill does: both
first part from it at the same step, and the step counts and KLs straddle each other. Eight KL steps
per tape set no band; these are context for the decode route, not a gate.

## Not done here

- Multi-card: the full model at serving speed needs 4 cards (about 313 GB on-card, 77 GB each at
  TP4/EP4). The one-card runner streams every layer's weights from the page cache instead. (Step 5,
  `../tp/README.md`: the N-rank partition, byte-identical to this route at N=2 on two cards.)
- Serving: no `Engine` integration, no scheduler, no CUDA graphs, no decode in `dsv41-run` (decode
  is gated per layer in `dsv41-gate layer --decode`).
- Speed: the GEMM tiles are plain (no cp.async pipeline, no split-K for M=1), the MoE launches per
  expert. No performance claim is made; the whole-trunk times above are single runs.
- DSpark (`mtp.*`) blocks and the vision tower: not run.
- Plan surfaces: the ten `deepseek_v41` plan operations still list no surface; registering these
  kernels as a rewrite comes with the multi-card route and its rewrite receipts.
- The Engram compressed token map is still an input (the vendor builds it from tokenizer
  normalizers at load).
