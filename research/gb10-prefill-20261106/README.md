# Qwen3.8-27B long-context prefill on GB10 (2026-11-06)

Device: one DGX Spark (GB10, compute capability 12.1, 48 SMs, unified LPDDR5x), SM clock 2,177 MHz
under load (clock cap active), memra built `121a` on the device (aarch64, CUDA 13.0) from
`5d70118b99` and the lane commits named below. Model: the Qwen3.8-27B NVFP4+Q5_K GGUF with its masked
MTP draft (the recommended GB10 arm of `research/gb10-tuning-20261102/tools/serve-arm.sh`), served
context 262,144, no tuning flags. Every trunk linear in this GGUF is NVFP4 (type 40), so every
trunk GEMM runs the F8F4 MMQ tile (GB10 default); the embedding and LM head are Q5_K.

Architecture (config + plan): 64 layers, 48 Gated DeltaNet layers (16 key / 48 value heads x 128,
conv 4) and 16 full-attention layers (every 4th; 24 Q / 4 KV heads x 256, output gate, partial
RoPE 0.25), MLP 5120 -> 17408, one MTP block. Trunk linear work is 48.66 GFLOP per token; causal
attention is 196,608 x T^2 FLOP. KV cache is q8_0 K / q5_1 V (31,552 B per token over 16 + 1
attention layers); the prime walks 4,096-token chunks and dequantizes the KV prefix into a BF16
workspace once per layer-chunk before the attention kernel.

## Method

- Two DGX Spark units, same model files and prompts. Unit 1: baseline, profiles, ceilings, the
  attention A/B, the GEMM bench, the final A/B. Unit 2: the dead-conversion / slab-copy A/B with
  its own base control, a second exactness battery and a 256K replication. Arms are compared only
  within a unit; the answer fingerprints also match across units.
- Served prefill (`harness/pf_long_memra.py`, the memra twin of the SGLang lane's `pf_long.py`):
  one user message = a nonce line (per boot target + rep, so no prefix-cache hit and identical
  prompts across arms) + the synthetic record document sized with `/v1/tokenize` + needles at
  10% and 85% depth + the question. Streaming chat completion, T=0, thinking off,
  presence_penalty 0, 40 max tokens; TTFT = first streamed delta. Fresh boot per cell
  (`harness/serve.sh`, `harness/run_arm.sh`, `harness/campaign.sh`), MemAvailable at 1 Hz, GPU
  telemetry at 250 ms, 3 GiB MemAvailable guard.
- Served token fingerprint: sha256 of the 40-token answer for each (target, rep) must be
  identical across arms (`harness/summarize.py`).
- Per-part profile: Nsight Systems 2025.3.2 over one cold 131,051-token served prefill
  (`harness/profile_arm.sh`), kernels classified by `harness/parts.py`, idle gaps and device
  copies by `harness/gaps.py` / `harness/memcpy.py`. nsys TTFT 240.3 s against 239.1 s
  unprofiled (0.5% overhead). The h2 build compiled on the host's CPU during this run; the
  profile is used for attribution only.
- Ceilings measured on this device: `harness/ceil_probe.cu` (tensor-core peaks with 8 independent
  accumulators, the F8F4 program's per-k32 form, DRAM read/copy over 2 GiB); ncu on the F8F4
  tile at the 4.1k-token monolithic prefill (`harness/ncu_gemm.sh`, `ceilings/`). The first
  probe run's `f8resc` rows are invalid (identical operands let the compiler merge MMAs;
  `ceilings/ceil_probe.txt`); `ceilings/ceil_probe2.txt` is the fixed run.

## Baseline served prefill (`5d70118b99`, `baseline-served/`)

| prompt tokens | TTFT s | runs |
|---:|---|---|
| 32,784 | 44.199 / 44.302 / 44.289 (median 44.289) | 3 |
| 131,051 | 239.130 / 239.077 | 2 |
| 259,951 | 636.901 | 1 (single run) |

256K completes; needles found at every length. MemAvailable never fell below 57.39 GiB.

## Per-part profile, one cold 131,051-token prefill (`profile-128k/`)

Window 240.6 s, kernels busy 227.3 s (94.5%); device-to-device copies another 12.0 s.

| part | measured s | work | achieved | measured ceiling on this device | time at ceiling |
|---|---:|---|---|---|---:|
| full attention `fa_prefill_qw_db` (16 layers) | 84.33 | 3.375 PFLOP | 40.0 TFLOPS | bf16 mma.sync 103 TFLOPS; FA2 BF16 at serving shapes 82.3 TFLOPS (SGLang lane, same box) | 32.8 / 41.0 |
| trunk GEMMs, F8F4 MMQ (400 per chunk) | 81.74 | 6.38 PFLOP | 78 TFLOPS | f8f6f4 block-scale MMA 204 TFLOPS; the F8F4 per-k32 rescale form 175 TFLOPS | 31.3 / 36.5 |
| e4m3 activation quantize (one per GEMM) | 8.99 | | 0.68 ms per call | DRAM read 232.5, copy 215.3 GB/s | |
| device-to-device copies (GEMM outputs into prime slabs) | 11.98 | 1.36 TB | 114 GB/s | removable (same values) | 0 |
| glue: silu_mul 7.44, add 4.34, f16 cvt 4.11, rms_norm 3.74, mul 0.65 | 20.31 | | each at 219-271 GB/s | DRAM floor per pass | |
| GDN: chunk state/solve/output/attn 13.97, conv/l2/gates 6.20 | 20.17 | ~1.77 TB minimal traffic | | 221 GB/s copy | 8.0 |
| alpha/beta dp4a matvecs + q8_1 quantize | 6.09 | | | | |
| KV dequant workspace, append, rope/norm/gate | 2.83 | | | | |

ncu, F8F4 tile at the gate shape (4.1k tokens x 17,408 x 5,120, 9.13 ms, 80 TFLOPS): 1 CTA of
8 warps per SM (178 registers, 94.7 KB smem), tensor pipe 36.7%, issue slots 41%, no eligible warp
on 61% of cycles; stalls per issued instruction: math-pipe throttle 1.00, wait 0.94, MIO throttle
0.57, barrier 0.27; 17.3 warp instructions per MMA; dequant shared stores 6-way bank conflicted.

The f16 conversion (4.11 s) is dead work on this GGUF: it builds an f16 operand for a weight
that has no f16 mirror, so the f16 GEMM refuses and the f32 MMQ runs.

## Levers

One lever at a time, each exact by construction and checked by the served fingerprint;
LEVER-LEDGER.md has every row with its commit.

### Attention: two query heads per CTA (adopted, GB10 default)

`attention-ab/`, one GB10, cells B T3 H2 H2 T3 B (fresh boot each), 32,784 + 131,051 tokens:

| arm | 32k TTFT | 131k TTFT |
|---|---|---|
| B: db twin (main) | 44.198, 44.232 s | 238.836, 239.106 s |
| T3: `MEMRA_PRIME_KV_T3=1` (existing door) | 42.382, 42.343 s (-4.19%) | 210.073, 209.889 s (-12.13%) |
| H2: `MEMRA_PRIME_KV_H2=1` (this lane) | 42.149, 42.077 s (-4.75%) | 205.908, 205.495 s (-13.92%) |

Answer text identical in every boot. kernel-check ARC B cells at t >= 128 (added in af5f356c97)
are bitdiff=0 for the h2, t3 and db twins (`gates-stack/`). h2 becomes the default on a
unified-memory 120a device (942e26ad8d); `MEMRA_PRIME_KV_H2=0` restores db. Decision record:
`docs/decisions/GB10-PRIME-ATTENTION-H2.md`.

### Trunk GEMM: three exact edits, none adopted

`gemm-ab/`, `gemm-ab-2/`: run-gen prefill-only, 3,855-token record prompt, 5 timed reps per
process, cells interleaved in both orders; last-row logits sha256 `f81e9690...` identical in all
24 processes (every edit is bit-exact).

| edit | median s | vs base | why |
|---|---|---|---|
| base | 4.2286, 4.2058, 4.2228 | | 187-200 registers, no spill |
| B fragments for all four k32 steps by two ldmatrix.x4 + 16-byte scale loads + 16-byte dequant stores | 5.8566, 5.8609 | +38.5% | 254 registers, 72-112 B spill per thread |
| the same + one ratio LUT per scale pair | 5.8082, 5.8052 | +37.4% | still spilling; the LUT trim is -0.8% on top |
| one ratio LUT per scale pair, alone | 4.2076, 4.2106 | -0.12% (flat) | the dequant ALU is not the limiter |
| one ldmatrix.x2 per k32 step + 16-byte dequant stores | 5.3381, 5.3364 | +26.6% | 255 registers, 136-152 B spill |

The tile sits at one 8-warp CTA per SM with 187-200 registers and 94.7 KB smem; any edit that
adds live state spills. Its measured limiter is latency at 1.9 warps per scheduler plus 9 FP32
instructions per MMA for the per-k32 rescale (inherent to this numeric program). The exact path to
the 175 TFLOPS program ceiling is structural: a dequantize-once weight pass with an L2-ordered
raster and a deeper cp.async pipeline, or warp-specialized dequant, not loader micro-edits.

### Dead conversion and slab copies (adopted)

`copies-ab-unit2/`, unit 2 with its own base control, cells B F N N F B:

| arm | 32k TTFT | 131k TTFT |
|---|---|---|
| B: base | 44.128, 44.183 s | 238.892, 238.702 s |
| F: f16 operand only when a mirror takes it (b5eee93d96) | 43.140, 43.605 s (-1.77%) | 234.545, 234.510 s (-1.79%) |
| N: F + GEMM outputs fed forward without slab copies (022f93ce30) | 40.184, 40.131 s (-9.05%) | 222.624, 222.787 s (-6.74%) |

Answer text identical across arms and identical to unit 1's answers for the same prompts.
In the 131k profile these remove the 2,112 `memra_f16_cvt_kernel` launches (4.11 s) and 11.5 s
of device-to-device copies (11.98 s -> 0.48 s); GPU busy rises from 94.5% to 99.1%.

## Per-part, before and after, against the tuned SGLang profile on the same Spark

One cold ~131k-token prime each. memra: Nsight Systems on unit 1, 131,051 tokens, 4,096-token
chunks; base `5d70118b99` (`profile-128k/`) and the final stack `5fd85d3513`
(`profile-stack2-128k/`). SGLang: the lane receipt `q27-prefill-budget-per-part.json` (torch
profiler, 131,017 tokens, 8,192-token chunks, A4 config), on the same Spark. The two engines run
different numerical programs: SGLang serves NVIDIA's NVFP4 safetensors with FP4 x FP4 MLP GEMMs
and FP8 projections; memra serves the owner's NVFP4+Q5_K GGUF with e4m3 activations on every
NVFP4 weight (F8F4) and q8_0/q5_1 KV.

| part (seconds of GPU time) | memra base | memra final | SGLang tuned |
|---|---:|---:|---:|
| full attention | 84.33 | 50.88 | 42.26 |
| trunk GEMMs (+ activation quantize) | 81.74 + 8.99 | 81.36 + 8.91 | 15.77 FP4 MLP + 10.71 FP8 projections |
| GDN kernels | 20.17 | 20.05 | 8.70 |
| glue (norms, adds, activation, conversions) | 20.31 | 15.93 | 13.82 |
| device-to-device copies | 11.98 | 0.48 | |
| alpha/beta matvecs, other | 8.91 | 8.22 | |
| KV dequant, append, rope/norm/gate | 2.83 | 2.77 | |
| GPU busy / window | 227.3 + 12.0 / 240.6 | 188.1 / 189.9 | 91.2 busy |

## Final: base against the adopted stack (`final-ab/`)

Unit 1, cells B S S B (fresh boot each, all three lengths per boot), base `5d70118b99` against the
stack `5fd85d3513` (h2 prime attention by default on GB10, no dead f16 operand pass, no slab
copies):

| prompt tokens | base TTFT | stack TTFT | change |
|---:|---|---|---:|
| 32,784 | 44.269, 44.257 s (median 44.263) | 38.091, 38.128 s (median 38.109) | -13.90% |
| 131,051 | 238.933, 239.086 s (median 239.010) | 189.871, 189.586 s (median 189.728) | -20.62% |
| 259,951 | 637.825, 637.222 s (median 637.524) | 476.034, 475.744 s (median 475.889) | -25.35% |

Answer text identical in all four boots at every length (and identical to unit 2's answers);
needles found everywhere; no request error. MemAvailable never fell below 66.82 GiB in any cell,
the same floor as base: the levers add no memory. Unit 2 replication at 259,951 tokens
(`final-256k-unit2/`): stack 474.848 s, base 637.564 s (-25.52%), same answer.

Exactness battery on the timed binary (`gates-stack2-unit1/`) and on unit 2's build of the same
commit (`gates-stack2-unit2/`): kernel-check ALL GREEN (112 cells, 5 skipped) with h2 (default),
db (`MEMRA_PRIME_KV_H2=0`) and t3 (`MEMRA_PRIME_KV_T3=1`), the t >= 128 workspace cells bitdiff=0
in all three; `run-gen` argmax-margin gate flips=0 bad=0 PASS; `run-spec` K=1..8 self-consistency
PASS. The first stack (`gates-stack/`, `c8fa962e5a`, with the rejected tile edits) passed the same
battery. `tests/prime_kv_h2_gpu.rs` is added but was not run in this lane (its build compiles
every engine binary); the kernel-check cells cover the same contract.

Against the tuned SGLang profile on the same Spark (14.9 / 92.3 / 269.2 s at 32k / 131k / 260k,
a different checkpoint and numerical program): memra goes from 2.97x / 2.59x / 2.37x of SGLang's
TTFT to 2.56x / 2.06x / 1.77x. At 131k the gap is now the trunk GEMMs (90.3 s with the activation
quantize, against SGLang's 26.5 s of FP4 and FP8 GEMMs), the GDN kernels (20.1 s against 8.7 s)
and attention (50.9 s against 42.3 s; 66 TFLOPS against FA2's 82 on BF16 KV).

## Next exact levers (not done here)

- F8F4 GEMM, structural (largest): the tile is at one 8-warp CTA per SM with 187-200 registers and
  any added live state spills. A dequantize-once weight pass with an L2-ordered raster and a deeper
  cp.async pipeline, or warp-specialized dequant, is the exact route toward the 175 TFLOPS ceiling
  of the per-k32 program (now 78 TFLOPS, 81 s of 190 s at 131k).
- Fuse silu*up into the down projection's e4m3 activation quantize (one [t, n_ff] f32 write and
  read less per layer, about 5 s at 131k) and quantize each shared activation once per GEMM group
  (gate/up, GDN qkv/z, attention q/k/v: 144 of 400 quantize passes per chunk, about 3 s).
- GDN: `l2_norm_pp_v2_f32` runs at ~88 GB/s (one row per warp, one 16-byte load per lane); several
  rows per warp with loads issued first keeps the per-row reduction and is worth about 2 s at 131k.
  The GDN total is 20.1 s against an ~8 s DRAM floor.
- `MEMRA_PRIME_CHUNK=8192`: documented bit-identical for grid-aligned boundaries; untested here.

