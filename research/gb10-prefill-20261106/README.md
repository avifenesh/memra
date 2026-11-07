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

## Round 2 (same day): four more exact levers on top of the stack

All on unit 1 against the round-1 stack `ef37ca487b` (P), fresh boot per cell, both orders,
same answers in every boot unless noted. Ledger rows 10-14.

### Prime chunk 8192 rows (measured, then rejected on peer latency; not landed)

`round2-chunk-ab/`, cells S C8 C16 C16 C8 S on the PR binary with `MEMRA_PRIME_CHUNK`:

| chunk | 32k TTFT | 131k TTFT | admission workspace (131k) | MemAvailable floor |
|---|---|---|---|---|
| 4,096 | 38.118, 38.143 s | 189.959, 190.031 s | 4,563 MB | 81.4 GiB |
| 8,192 | 38.002, 37.994 s (-0.35%) | 188.544, 189.032 s (-0.64%) | 6,442 MB | 81.3 GiB |
| 16,384 | 38.066, 37.996 s (-0.26%) | 187.766, 187.639 s (-1.21%) | 10,200 MB | 74.5 GiB |

8,192 was armed at load for this trunk on a unified-memory 120a device in `0cb9642b98` and removed
in `09490b0aac` after the fairness cell below; 16,384's
extra 0.57% at 131k costs 3.8 GB more booking and 7 GiB of MemAvailable per concurrent 131k
prime. At 259,951 tokens with 8,192 rows the answer equals base and MemAvailable stayed above
71.8 GiB (`round2-mem-256k-chunk8192/`; a build ran alongside, so its 473.2 s is not a timing).
Decision record: `docs/decisions/GB10-PRIME-CHUNK-8192.md`.

**Rejected before landing (review item, fairness cell `fairness-chunk/`).** The prime walker serves
its peers between chunks, so the chunk is the peer quantum. Beside a cold 131,042-token prime with
three cold 2,003-token peers at +10/+15/+20 s, cells C4 C8 C8 C4: the long prime 196.42 / 197.39 s
at 4,096 and 195.21 / 195.18 s at 8,192 (-0.6%), the peers 5.09-8.72 s at 4,096 and 6.83-13.23 s at
8,192 (worst 8.7 -> 13.2 s), `tick_max_ms` 7,362 / 7,350 against 14,255 / 14,243. The landed lane
keeps 4,096 rows, so its 131k cold TTFT is about 0.6% above the round-2 stack figures that include
this lever (ledger rows 14 and 15). Harness: `harness/fairness_cell.py`, `harness/gb10-fairness.sh`.
The binary, `b645e05e0b`, is the pre-landing stack: `5d70118b99` with memra#1073 at `3783a79f8a`,
memra#1074 at `e42c425894` and memra#1075 at `aa309313d4` rebased on it; the ref
`bank/gb10-batch-b645e05e0b` keeps it fetchable. The 4,096 arm forces the chunk through
`MEMRA_PRIME_CHUNK`, so both arms ran the same bytes.

### Fused activation quantize, quantize-once groups, four-row L2 norm

`round2-levers-ab/`, stacked: A = silu(gate)*up fused into the down projection's e4m3 quantize
(`68a8a05730`), B = A + each shared activation quantized once per GEMM group (`873634c010`),
C = B + `l2_norm_pp` four rows per warp (measured as `253a52c36a` on the A+B base), cells
P A B C C B A P at 32k and 131k:

| arm | 32k TTFT | 131k TTFT | step |
|---|---|---|---|
| P: round-1 stack `ef37ca487b` | 38.107, 38.158 s | 190.063, 190.023 s | |
| A: fused activation quantize | 36.824, 36.832 s | 184.885, 184.873 s | -3.42% / -2.72% |
| B: A + quantize-once groups | 36.235, 36.277 s | 182.804, 182.505 s | -1.55% / -1.20% |
| C: B + four-row L2 norm | 36.250, 36.254 s | 182.616, 182.559 s | -0.01% / -0.04% |

Answer text identical across all eight boots at both lengths.

C is flat (+0.05% at 32k, -0.12% at 131k) and is rejected; it stays on
`research/gb10-prefill-round2-levers` with its kernel-check byte cell. Kernel-level on unit 2
(`round2-unit2-profiles/`, one 131k prime per binary): `l2_norm_pp_v2r4_f32` takes 3.050 s
(0.963 ms per call) against the one-row kernel's 2.960 s (0.934 ms) in the B binary, so the
one-row kernel's ~90 GB/s was not limited by bytes in flight per warp. The same profiles confirm
A and B engage: the fused `quantize_mmq_e4m3_d4_silu_kernel` runs 2,112 times (6.06 s),
`silu_mul_f32` falls to 66 calls (the MTP head), and the plain quantizer runs 6,336 times
(3.22 s) instead of 13,200 (8.99 s): 400 - 64 fused - 144 shared per chunk.

### Final: round-1 stack against round 2 (`round2-final-ab/`)

Unit 1, cells P S S P, all three lengths per boot, the round-1 stack `ef37ca487b` against
round 2 `0cb9642b98` (A + B + the 8,192-row chunk default; C left out):

| prompt tokens | round-1 stack | round-2 stack | change | against main `5d70118b99` |
|---:|---|---|---:|---:|
| 32,784 | 38.141, 38.162 s (median 38.151) | 36.162, 36.165 s (median 36.163) | -5.21% | 44.263 -> 36.163 s, -18.3% |
| 131,051 | 189.944, 189.935 s (median 189.940) | 180.949, 181.310 s (median 181.130) | -4.64% | 239.010 -> 181.130 s, -24.2% |
| 259,951 | 475.384, 475.914 s (median 475.649) | 457.342, 457.011 s (median 457.177) | -3.88% | 637.524 -> 457.177 s, -28.3% |

Answer text identical in all four boots at every length. MemAvailable floor 66.8 GiB (round 1)
and 65.2 GiB (round 2; the 8,192-row chunk's larger workspace). Unit 2 replication at 259,951
tokens (`round2-final-256k-unit2/`): 456.213 s against 474.442 s, -3.84%, same answer. The
exactness battery passes on the timed binary (`round2-gates-unit1/`) and on unit 2's build of the
same commit (`round2-gates-unit2/`): kernel-check ALL GREEN (113 cells) in h2, db and t3 modes
with `MMQ-F8F4-SILU-FUSED` 0 mismatches at T = 16, 130, 1,024; argmax-margin flips=0 bad=0;
run-spec K=1..8 PASS. Against the tuned SGLang profile (14.9 / 92.3 / 269.2 s) memra is now at
2.43x / 1.96x / 1.70x of its TTFT.

Per part at 131,051 tokens, Nsight Systems on unit 1 (`round2-profile-128k/`, nsys TTFT 181.9 s
against 181.1 s unprofiled), next to round 1 (`profile-stack2-128k/`):

| part (GPU seconds) | round 1 `5fd85d3513` | round 2 `0cb9642b98` |
|---|---:|---:|
| trunk GEMMs, F8F4 MMQ | 81.36 (13,200 launches, 33 chunks) | 80.34 (6,800 launches, 17 chunks) |
| e4m3 activation quantize | 8.91 | 9.32 (fused silu quantize 6.07, plain 3.25) |
| full attention | 50.88 | 50.79 |
| GDN kernels | 20.05 | 20.07 |
| glue (norms, adds, activation) | 15.93 | 8.68 (`silu_mul_f32` gone) |
| KV dequant workspace | 1.37 | 0.73 (half the chunks) |
| device-to-device copies | 0.48 | 0.50 |
| GPU busy / window | 188.1 / 189.9 | 179.6 / 181.4 |

The fused quantize kernel moves the activation's read into the quantizer (its 6.07 s reads gate and
up, 570 MB per layer-chunk), so the quantize line grows while the silu pass and its write and
re-read disappear from the glue line.

Raw Nsight Systems / ncu reports are not in the repo (the pre-push hook refuses nsys blobs): the
round-1 base and stack reports and the ncu F8F4 report are in the lab's evidence backup as
snapshot `f53a15ef`, the round-2 stack report as `0d1b56a0`. Unit 2's two round-2 reports were
not retained; their `parts.txt` / `parts.json` summaries are here.

### F8F4 GEMM rework

`GEMM-REWORK-DESIGN.md` (design, built below): a faithful probe of the tile's epilogue puts the
ceiling at 113 TFLOPS at today's 8 warps per SM and 132 at 16; today's tile runs 79. A
dequantize-once image pass with a pipelined mainloop and an L2-ordered raster is worth about
11 s at 131k at 8 warps per SM and about 20 s if it reaches 16 warps per SM (181.1 -> ~161 s).

## F8F4 GEMM v2 (built, same day; ledger rows 16-26)

`b6d10d41c7`: a pre-pass folds each NVFP4 weight once per call, with the v1 loader's expressions,
into a pooled image (e4m3 row-major, per-32 scales k-major, 1.125 B per weight); the F8F4
quantizers write the same activation bytes in a v2 layout; one 16-warp CTA per SM computes a
256 x 128 tile from a 3-stage cp.async ring (K 64, XOR-swizzled rows) with ldmatrix for A and B,
the same block-scale MMA and the same per-k32 epilogue; 4 row tiles per raster group. GB10 default
from 2,048 rows (`MEMRA_MMQ_F8F4_V2=0` rolls back). Decision record:
`docs/decisions/GB10-F8F4-GEMM-V2.md`. The timed binary is `b6d10d41c7` on the round-2
head `1ed40895ec` (kept on `research/gb10-f8f4-gemm-v2-measured`); the PR replays it on main as
`2ce09cd39b`, with two aarch64 clippy fixes and the review fixes of `c23cd3a158` (ledger row 26).

How it got there (`gemm-v2-probe/`, `harness/f8f4v2_probe.cu`, gate shape at 8,192 rows):

| step | best config | GEMM TFLOPS (v1 73.5) | what ncu showed next |
|---|---|---:|---|
| image + ring + ldmatrix, v1 activation layout, padded rows | one 16-warp 256 x 128 CTA | 92.3 | L2 77.7%, 45% excess sectors from the 144 B activation blocks |
| + v2 activation layout | two 8-warp 128 x 128 CTAs | 107.3 | L2 87%: cp.async writes 15-way conflicted in padded rows, 6 pointers spilled |
| + swizzled rows, 32-bit offsets | one 16-warp 256 x 128 CTA, K 64 x 3 | 111.9 | L2 50%, no excess; QMMA math throttle with the tensor pipe 52% busy: the 8 FP32 epilogue instructions per MMA |

The epilogue is the program's (`fma(dA * dB, C, sum)` per element per k32 block); packed f32x2
does not exist on GB10 (ptxas emits scalar FMUL/FFMA, `ceil_probe3.txt`). Per trunk shape, unit 1
(unit 2 within 2%):

| in x out | v1 TFLOPS | image pass ms | v2 (one 16-warp CTA) GEMM / with image | two 8-warp CTAs, G 8 |
|---|---:|---:|---:|---:|
| 17,408 x 5,120 | 76.6 | 0.688 | 116.3 / 110.2 | 106.4 / 101.4 |
| 5,120 x 1,024 | 73.1 | 0.034 | 95.5 / 92.0 | 98.5 / 94.9 |
| 5,120 x 10,240 | 73.6 | 0.405 | 107.8 / 102.6 | 106.5 / 101.4 |
| 5,120 x 12,288 | 73.5 | 0.482 | 111.7 / 106.2 | 108.5 / 103.3 |
| 5,120 x 17,408 | 73.1 | 0.678 | 112.1 / 106.5 | 107.4 / 102.3 |
| 5,120 x 6,144 | 74.0 | 0.245 | 106.8 / 101.7 | 107.7 / 102.4 |
| 6,144 x 5,120 | 75.2 | 0.244 | 110.4 / 104.9 | 109.8 / 104.4 |

Exactness: kernel-check `MMQ-F8F4-V2` (`gemm-v2-kc-unit1/`, `gemm-v2-gates-unit1/`) compares v2
with v1 on every f32 bit at all 8 NVFP4 GEMM shapes of this model and 4 of the 9B, T = 16 / 130 /
1,024 / 4,096, GGUF and split-plane weight layouts, plus the fused silu quantizer: 0 mismatching
outputs. `run-gen` pp3855 (`gemm-v2-gemm-ab-unit2/`, B V V B): 4.167 -> 3.673 s (-11.8%), the
same logits sha256 `f81e9690...` as rounds 1 and 2.

### Served A/B

Method of round 2 (`harness/served_ab.sh` over `campaign.sh`): cells P V V P, one fresh boot per
cell, 32k / 131k / 260k targets in each boot, base = the round-2 stack `1ed40895ec` (binary of
`0cb9642b98`'s code), V = `b6d10d41c7`.

Unit 1, 8,192-row prime chunks (the round-2 GB10 default, since dropped; `gemm-v2-served-unit1/`):

| prompt tokens | base (round 2, `1ed40895ec`) | F8F4 v2 (`b6d10d41c7`) | change |
|---:|---|---|---:|
| 32,784 | 36.199, 36.238 s (median 36.218) | 31.376, 31.360 s (median 31.368) | -13.39% (-4.9 s) |
| 131,051 | 181.306, 181.239 s (median 181.273) | 161.785, 161.862 s (median 161.823) | -10.73% (-19.4 s) |
| 259,951 | 457.527, 458.321 s (median 457.924) | 418.932, 419.036 s (median 418.984) | -8.50% (-38.9 s) |

Answer text identical in all four boots at every length, needles found, no request error.
MemAvailable floor 65.17 / 65.32 / 65.32 / 65.26 GiB (P V V P): the per-call image is pooled
scratch and moves no floor. The base reproduces round 2 on this unit (36.163 / 181.130 /
457.177 s then).

On the final commits, after #1073-#1076 landed on main (defaults, 4,096-row chunks), unit 2
(`gemm-v2-served-final-unit2/`, P V V P):

| prompt tokens | main `c2e04719b6` | main + v2 `c23cd3a158` | change |
|---:|---|---|---:|
| 32,784 | 36.137, 36.245 s (median 36.191) | 32.525, 32.437 s (median 32.481) | -10.25% (-3.7 s) |
| 131,051 | 181.768, 182.177 s (median 181.972) | 167.487, 167.682 s (median 167.584) | -7.91% (-14.4 s) |
| 259,951 | 460.503, 459.085 s (median 459.794) | 431.770, 431.174 s (median 431.472) | -6.16% (-28.3 s) |

Answer text identical in all four boots and to every earlier boot; MemAvailable floor 65.8-66.7
GiB. The exactness battery on that PR-head binary, unit 1 (`gemm-v2-gates-main-unit1/`):
FINALGATES_PLACEHOLDER

Per part at 131,051 tokens, Nsight Systems on unit 1 (`gemm-v2-profile-128k-unit1/`, nsys TTFT
162.5 s against 161.8 s unprofiled; the report and the three ncu reports of the tile steps are in
the lab's evidence backup as snapshot `435c2f82`), next to round 2
(`round2-profile-128k/`):

| part (GPU seconds) | round 2 `0cb9642b98` | v2 `b6d10d41c7` |
|---|---:|---:|
| trunk GEMMs, F8F4 | 80.34 (6,800 v1 launches) | 60.76: v2 tile 57.56 (6,400 launches, 8.99 ms each) + image pass 3.11 + v1 0.09 (the short chunk, under 2,048 rows) |
| e4m3 activation quantize | 9.32 | 9.40 (v2 store) |
| full attention | 50.79 | 50.76 |
| GDN kernels | 20.07 | 20.18 |
| glue (norms, adds, activation) | 8.68 | 8.74 |
| GPU busy / window | 179.6 / 181.4 | 160.2 / 162.0 |

The v2 tile runs the 6.38 PFLOP at about 111 TFLOPS (105 with its image pass), 84% of the
132 TFLOPS epilogue ceiling at 16 warps per SM; attention (50.8 s) is now the largest part.

4,096-row prime chunks, both arms with `MEMRA_PRIME_CHUNK=4096` (the GB10 default once #1073
drops the 8,192 default for peer fairness, `fairness-chunk/`), unit 1 (`gemm-v2-served-4k-unit1/`):

| prompt tokens | base `1ed40895ec` | F8F4 v2 `b6d10d41c7` | change |
|---:|---|---|---:|
| 32,784 | 36.286, 36.214 s (median 36.250) | 32.516, 32.562 s (median 32.539) | -10.24% (-3.7 s) |
| 131,051 | 182.609, 182.540 s (median 182.575) | 168.396, 167.852 s (median 168.124) | -7.91% (-14.5 s) |
| 259,951 | 461.140, 461.507 s (median 461.323) | 433.077, 433.143 s (median 433.110) | -6.12% (-28.2 s) |

Unit 2 (`gemm-v2-served-4k-unit2/`):

| prompt tokens | base `1ed40895ec` | F8F4 v2 `b6d10d41c7` | change |
|---:|---|---|---:|
| 32,784 | 36.250, 36.340 s (median 36.295) | 32.593, 32.595 s (median 32.594) | -10.20% (-3.7 s) |
| 131,051 | 182.231, 182.374 s (median 182.303) | 167.747, 167.351 s (median 167.549) | -8.09% (-14.8 s) |
| 259,951 | 459.923, 461.138 s (median 460.530) | 432.283, 432.582 s (median 432.433) | -6.10% (-28.1 s) |

Answer text identical in all eight boots and identical to the 8,192-row boots, needles found, no
request error. MemAvailable floor 66.8 GiB on unit 1 and 66.2-66.3 GiB on unit 2 in every cell.
The two units agree within 0.5 s at every length. The saving is smaller at 4,096 rows (14.6 s
against 19.4 s at 131k) because v2 builds each weight image once per chunk and there are twice as
many chunks (3.1 s of image passes at 8,192 rows, about 6 s at 4,096).

Exactness battery on the timed v2 binary, unit 1 (`gemm-v2-gates-unit1/`): kernel-check ALL GREEN
(115 cells, 5 skipped) in default (h2), `MEMRA_PRIME_KV_H2=0` and `MEMRA_PRIME_KV_T3=1` modes, with
`MMQ-F8F4-V2` 0 mismatching outputs over 8 shapes of this model and 4 of the 9B; argmax-margin gate
flips=0 bad=0 PASS; run-spec K=1..8 self-consistency PASS (acceptance 17/98 and 17/112, as round 2);
`MEMRA_MMQ_F8F4_V2=0` rollback: pp3855 logits sha256 `f81e9690...` in both arms (4.173 s off,
3.685 s on). The default argmax prompt (2,048 tokens) stays under the v2 threshold, so the gate
also ran on the 16k prompt (`p4-16k`, 23,476 tokens), which primes through v2: it reports
flips=3 bad=1 FAIL on v2 **and on the base binary**, with byte-identical probe tables
(`4800c57de4349345`). All three flips are margin-explained near-ties (0.012-0.031 against a
config spread of 0.071-0.138); the FAIL is the flip budget (1 per 12-position window, calibrated on
the 2,048-token prompt), so the 16k prompt is not a calibrated gate on this model, and v2 changes
none of its values.


## Next exact levers

- F8F4 v2 follow-ups: run the next GEMM's image pre-pass on a side stream under the current GEMM
  (the tile reads L2 at 50%, the pre-pass is DRAM-bound; 3.1 s at 131k with 8,192-row chunks,
  about twice that at 4,096); the tile sits at 84-88% of the 132 TFLOPS epilogue ceiling on the
  large shapes.
- A wider prime chunk only while no peer is admitted: v2 builds each weight image once per chunk,
  so 8,192 rows are worth about 3.5% at 131k with v2 (167.5 s at 4,096 rows on unit 2 against
  161.8 s at 8,192 on unit 1) where they were worth 0.5% without it; the 8,192 default was dropped
  for the peer first-token wait beside a 131k prime (worst peer 8.7 s at 4,096 against 13.2 s at
  8,192), so it needs its own fairness cell.
- GDN: 20.1 s at 131k against an ~8 s DRAM floor; the four-row L2 norm did not move it, so the
  next step is a per-kernel ncu of the chunk kernels (`gdn_chunk_state_mma`, `gdn_chunk_solve32`,
  `gdn_chunk_output_mma`, `ssm_conv1d_gdn_state`) before any rewrite.
- The alpha/beta dp4a matvecs (3.2 s) quantize their shared input twice; one dual kernel reading it
  once keeps both dot products.
