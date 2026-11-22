# Inkling 2 Mini: the nvfp4-rig numeric class

Status: adopted as the family's default class on the RTX 5090 Laptop (sm_120a, 24 GB), with its
measured price stated below. The checkpoint as stored stays the named class `shipped`.

## What was chosen

`nvfp4-rig` = the checkpoint's declared activation-quant experts, unit-scale e4m3 K/V on all 48
cache slots, the MXFP8 expert banks of layers 20-23 minted to NVFP4 (running the NVFP4 W4A4 program
of layers 0-19), an NVFP4 output projection, and NVFP4 attention projections (q, k, v, o, output
gate, relative-bias projection) on all 24 layers. The minted tensors are weight-only (f32 inputs).
The NVFP4 conversion is our own quantizer: e4m3 scales per 16 inputs, MSE choice among 16 scale
candidates, and a per-tensor (per-expert) scale `amax / 2688`. The overlay is immutable and bound by
hash to the pinned checkpoint.

History: coordinator decision 8 (2026-11-21 ~13:45) first set the attention conversion to the
layers outside the loop core (0-5, 18-23), from per-layer KL. The loop core (6-17) is read three
times per token and was the most sensitive (per-layer KL 5e-4 to 1.4e-3 against 1-3e-4 elsewhere).
The owner then ruled (2026-11-21 ~14:25): "convert the loop attention too, measure it". The first set
stays measurable as `nvfp4-rig-outside-loop` (same overlay, the loop-core planes not read).

## Measurements (2026-11-21, RTX 5090 Laptop, full clocks: clampcheck OK before every timed take, 250 ms telemetry with no P2 or 11,001 MHz sample)

Reference: lane F's most precise program (shipped weights dequantized exactly, f32 activations, f32
KV). Corpora: lane F's rawdoc and chatwrap, 15 sequences of 1,024 tokens each, not built from public
benchmarks. Metric: lane F's `stats()`. KL is mean token KL(ref || arm) on real-text positions;
dNLL is arm minus ref NLL on the real next token. 95% intervals come from a sequence bootstrap
(B = 4000).

memra GPU route (`ink2-gpu --corpus`, our kernels, the minted bytes):

| set | class | KL real [95% CI] | p99 | top-1 | dNLL real [95% CI] |
| --- | --- | --- | --- | --- | --- |
| rawdoc | shipped | 0.0127 [0.0108, 0.0147] | 0.132 | 0.961 | +0.0010 [-0.0015, +0.0044] |
| rawdoc | nvfp4-rig-outside-loop | 0.0212 [0.0176, 0.0249] | 0.199 | 0.951 | +0.0097 [+0.0043, +0.0158] |
| rawdoc | nvfp4-rig | 0.0279 [0.0240, 0.0318] | 0.281 | 0.945 | +0.0195 [+0.0144, +0.0252] |
| chatwrap | shipped | 0.0160 [0.0128, 0.0193] | 0.174 | 0.958 | +0.0047 [-0.0000, +0.0108] |
| chatwrap | nvfp4-rig-outside-loop | 0.0239 [0.0195, 0.0283] | 0.235 | 0.948 | +0.0062 [+0.0003, +0.0115] |
| chatwrap | nvfp4-rig | 0.0303 [0.0253, 0.0353] | 0.299 | 0.938 | +0.0115 [+0.0059, +0.0170] |

Lane F's harness (transformers PR program, its own quantization of the same set), cumulative:

| set | arm | KL real [95% CI] | top-1 | dNLL real [95% CI] |
| --- | --- | --- | --- | --- |
| rawdoc | actq only (lane F) | 0.0121 [0.0102, 0.0141] | 0.962 | +0.0040 [+0.0011, +0.0071] |
| rawdoc | outside loop | 0.0196 [0.0161, 0.0232] | 0.954 | +0.0102 [+0.0051, +0.0157] |
| rawdoc | all layers | 0.0267 [0.0220, 0.0316] | 0.945 | +0.0150 [+0.0084, +0.0225] |
| chatwrap | actq only (lane F) | 0.0133 [0.0111, 0.0155] | 0.957 | +0.0018 [-0.0006, +0.0041] |
| chatwrap | outside loop | 0.0211 [0.0172, 0.0255] | 0.950 | +0.0081 [+0.0035, +0.0126] |
| chatwrap | all layers | 0.0281 [0.0235, 0.0326] | 0.938 | +0.0143 [+0.0072, +0.0215] |

The route reproduces the harness. Its increment over its own shipped class is +0.0152 (all layers)
and +0.0085 (outside loop) of KL on rawdoc, against the harness's +0.0146 and +0.0075 over actq
alone. Converting the loop core's attention roughly doubles the class's KL and dNLL over the
declared program. Per layer (lane F's harness, one layer's six projections in NVFP4 and nothing else, unpinned
routing; KL real and dNLL real with their 95% intervals). Six chatwrap coda arms were still queued
when this was written; they land in `gpu-run/perlayer-stats.txt`. The pinned, in-distribution
per-layer figures are in lane F's screen table: loop core 5e-4 to 1.4e-3, elsewhere 1e-4 to 3e-4.

| L | rawdoc KL real [95% CI] | rawdoc dNLL [95% CI] | chatwrap KL real [95% CI] | chatwrap dNLL [95% CI] |
| --- | --- | --- | --- | --- |
| 0 | 0.00628 [+0.0049, +0.0077] | +0.00086 [-0.0009, +0.0026] | 0.00660 [+0.0054, +0.0078] | -0.00108 [-0.0025, +0.0004] |
| 1 | 0.00562 [+0.0046, +0.0067] | +0.00065 [-0.0026, +0.0038] | 0.00612 [+0.0050, +0.0073] | +0.00052 [-0.0024, +0.0029] |
| 2 | 0.00526 [+0.0044, +0.0062] | -0.00022 [-0.0026, +0.0023] | 0.00587 [+0.0048, +0.0069] | +0.00000 [-0.0023, +0.0022] |
| 3 | 0.00514 [+0.0043, +0.0060] | +0.00027 [-0.0014, +0.0018] | 0.00532 [+0.0042, +0.0068] | +0.00037 [-0.0017, +0.0025] |
| 4 | 0.00510 [+0.0042, +0.0060] | -0.00049 [-0.0025, +0.0015] | 0.00523 [+0.0044, +0.0060] | +0.00361 [+0.0015, +0.0058] |
| 5 | 0.00521 [+0.0043, +0.0061] | -0.00034 [-0.0027, +0.0019] | 0.00586 [+0.0049, +0.0068] | -0.00096 [-0.0037, +0.0014] |
| 6 | 0.00627 [+0.0053, +0.0073] | +0.00132 [-0.0016, +0.0043] | 0.00670 [+0.0057, +0.0077] | -0.00071 [-0.0032, +0.0016] |
| 7 | 0.00537 [+0.0046, +0.0062] | -0.00093 [-0.0030, +0.0013] | 0.00594 [+0.0051, +0.0068] | +0.00145 [-0.0010, +0.0038] |
| 8 | 0.00466 [+0.0039, +0.0055] | -0.00013 [-0.0018, +0.0015] | 0.00509 [+0.0043, +0.0059] | +0.00121 [-0.0003, +0.0026] |
| 9 | 0.00419 [+0.0036, +0.0048] | -0.00040 [-0.0025, +0.0017] | 0.00483 [+0.0041, +0.0056] | -0.00026 [-0.0023, +0.0016] |
| 10 | 0.00426 [+0.0036, +0.0049] | -0.00042 [-0.0023, +0.0013] | 0.00466 [+0.0040, +0.0054] | +0.00146 [-0.0005, +0.0034] |
| 11 | 0.00461 [+0.0038, +0.0055] | +0.00100 [-0.0005, +0.0025] | 0.00485 [+0.0040, +0.0057] | +0.00070 [-0.0008, +0.0023] |
| 12 | 0.00516 [+0.0043, +0.0061] | -0.00008 [-0.0024, +0.0022] | 0.00562 [+0.0047, +0.0066] | -0.00053 [-0.0019, +0.0011] |
| 13 | 0.00349 [+0.0029, +0.0041] | -0.00082 [-0.0019, +0.0002] | 0.00352 [+0.0029, +0.0042] | +0.00148 [+0.0004, +0.0026] |
| 14 | 0.00321 [+0.0027, +0.0037] | -0.00033 [-0.0018, +0.0010] | 0.00335 [+0.0028, +0.0039] | -0.00054 [-0.0020, +0.0010] |
| 15 | 0.00272 [+0.0023, +0.0031] | +0.00026 [-0.0007, +0.0014] | 0.00312 [+0.0025, +0.0037] | +0.00123 [+0.0001, +0.0024] |
| 16 | 0.00298 [+0.0024, +0.0035] | -0.00028 [-0.0019, +0.0014] | 0.00323 [+0.0027, +0.0038] | +0.00091 [-0.0003, +0.0022] |
| 17 | 0.00283 [+0.0023, +0.0034] | +0.00030 [-0.0010, +0.0015] | 0.00306 [+0.0025, +0.0037] | -0.00077 [-0.0019, +0.0003] |
| 18 | 0.00066 [+0.0005, +0.0008] | +0.00082 [+0.0002, +0.0015] | pending | pending |
| 19 | 0.00026 [+0.0002, +0.0003] | +0.00017 [-0.0002, +0.0006] | pending | pending |
| 20 | 0.00024 [+0.0002, +0.0003] | +0.00044 [-0.0000, +0.0010] | pending | pending |
| 21 | 0.00023 [+0.0002, +0.0003] | +0.00046 [-0.0000, +0.0009] | pending | pending |
| 22 | 0.00020 [+0.0002, +0.0003] | +0.00043 [+0.0001, +0.0008] | pending | pending |
| 23 | 0.00043 [+0.0003, +0.0005] | +0.00040 [-0.0000, +0.0008] | pending | pending |

Unpinned, a single prelude or core layer moves the routing of 16-34% of the rows, which is most of
its KL. The coda layers (18-23) move under 2.2%. No single layer's dNLL interval excludes zero by
more than 0.004. The cost shows only in the cumulative sets above.

Memory (resident weights; the maximum context the card admits at chunk 512):

| class | weights GiB | max context |
| --- | --- | --- |
| shipped | 17.59 | 73,546 |
| nvfp4-rig-outside-loop | 14.76 | 509,224 |
| nvfp4-rig | 14.33 | 537,896 |

Decode, greedy, 128 tokens on the p1 tape. Nine takes in a balanced order; each class's greedy ids
are identical across its takes:

| class | tok/s (3 takes) |
| --- | --- |
| shipped | 5.09, 4.99, 5.05 |
| nvfp4-rig-outside-loop | 4.38, 4.34, 4.56 |
| nvfp4-rig | 3.29, 3.27, 3.26 |

On today's route the class is slower: 1,637 launches a token, and the decode GEMV is one thread per
output row. Its NVFP4 branch decodes a nibble and applies a scale per 16 inputs, and that costs more
than the BF16 path saves in bytes. The byte budget the class removes per decode token is about 1.85 GB
of attention (48 slot visits), 0.66 GB of unembed and 0.12 GB of layer 20-23 experts. K/V adds
3 bytes per element: about 0.3 GB at 1k context. That is about 2.6 GB off roughly 6 GB, the gap
between a ~150 tok/s and a ~250 tok/s DRAM floor at 865 GB/s. The speed lever exists only with an
NVFP4 GEMV that streams at bandwidth (the perf lane's kernels).

## Rejected / not chosen

- Keeping the loop core in BF16 (decision 8's first set): about half the KL and dNLL cost of the
  all-layer set, 0.43 GiB more weights. Overruled by the owner; it stays as
  `nvfp4-rig-outside-loop` for measurement.
- Requantizing with an external tool: not chosen. The mint is our own, deterministic, and byte-equal
  to the study's quantizer on the CPU. On CUDA, torch computes the scalar division `amax / 2688` as a
  reciprocal multiply, which moves `t` by one ulp on 38 of 145 dense tensors. The mint follows the
  quantizer's stated division.

## Evidence

`research/modelplan-onboarding-inkling2mini-20261121/nvfp4-rig/` (README, mint and verify logs,
goldens, `gpu-run/`: kernel cells, chunking bit-identity, KL npz and json, harness arms, speed takes,
telemetry).
