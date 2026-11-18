# Prefill against decode at single rows of `board-2048` (memra#1203)

2026-11-18. One GB10 (DGX Spark, sm_121a, 48 SMs). Main 96390ef0e4 plus a diagnostic patch
(`tools/row-bisect.diff`, the five commits of `bank/1203-row-bisect`): the `row-bisect` binary and
last-row taps in `forward_last` and `decode_step_h`. Disarmed, the taps change no number: the
table rows of every binary used here are byte-equal to the GB10 calibration cells of
`research/argmax-gate-rule-20261110/` at the same rows (Ornith row 41: prefill 15 at 0.8015,
decode 15 at 2.8994; Qwen3.8-27B row 1198: prefill 47 at 7.0772, decode 47 at 8.6467), and
across the four binaries the decode rows agree to every printed digit.

## Outcome

No defect. These rows are near-tie amplifiers: a 0.1% difference between two programs at
one layer grows into a several-logit difference at the head. Which lever moves a row
depends on the row.

1. **Qwen3.8-27B row 1198: the quantized KV cache.** An all-f32 prefill (f32 activations,
   f32-dequantized weights, f32 K/V, naive f32 attention, sequential GDN scan) gives 47 at
   4.970. The same f32 prefill with only its K/V passed through q8_0 / q5_1 on the host gives
   47 at 8.802, next to decode's 8.647; its logits are 1.6% (max 0.47) from the f32 decode's.
   The sm_120a prefill program (4.916; on GB10 with F8F4 off, 1.3% and max 0.30 from the f32
   prefill's logits) is right at this row. Decode is right for the cache it reads.
2. **Decode's kernels are not the cause.** Against an all-f32 prefill, the f32 decode
   (`MEMRA_FAST=0`) is equal to 1.3e-7 through the first three layers (GDN). The first
   difference is at layer 3, the first attention layer, where the only difference left is the
   KV format: 2.2e-3 at the mixer, 8.5e-4 on the residual. It is 0.5% at layer 23, then grows
   about 1.5x per layer: 2.8% at layer 27, 11% at 31, 28% at 39, about 100% from 51. With the
   K/V round-trip the residual stays within 2.7% to the last layer.
3. **Ornith-1.5-35B-A3B rows 41 and 1809: a chaotic trajectory, amplified by MoE routing
   near-ties.** Every pair of programs selects a different top-8 at 11 to 29 of the 40 MoE
   layers of these rows, the two f32-class programs included. The median gap between the
   8th and 9th router logit is 0.025 to 0.076. Per layer, the largest router-logit move between
   two programs has a median of 0.14 to 0.65. The hidden states part by 1% at layer 0 (the activation-format
   class) and by 10% to 46% at layer 39 on every row of a 16-row sweep. No program is a
   reference within about 2 logits there, so no side is the defect.
4. **GB10's prefill is a different program from sm_120a's because of `MEMRA_MMQ_F8F4`.**
   On GB10 the NVFP4 prefill GEMMs take e4m3 activations by default. With `MEMRA_MMQ_F8F4=0`
   the GB10 prefill reproduces the sm_120a prefill to every printed digit: Ornith 41 is 18 at
   1.4724 (sm_120a 1.472389), 306 is 22 at 0.9269 (0.926936), 1809 is 22 at 4.0792
   (4.079245); Qwen 1198 is 47 at 4.9164 (4.916439). At Qwen 1198 the e4m3 activations put
   the layer-0 projections 0.87% from f32, against 0.16% for decode's int8 q8_1. The
   `MEMRA_GDN_MMA` bf16 lane is not part of it: forcing it on GB10 changes no bit, and its
   sm_120a default leaves the sm_120a prefill equal to GB10 with F8F4 off.

## Programs

- **prefill**: `forward_last` over `prompt[..=r]`, f32 K/V (the gate's reference).
- **decode**: `decode_step` from position 0 over the quantized cache (q8_0 K, q5_1 V).
- **prime**: `prime_cache` over `prompt[..=r]`, writing and reading the quantized cache.
- **f32 prefill** (`pf32`): `RB_PREFILL_ENV=MEMRA_NO_GEMM=1,MEMRA_MMQ_W4A8=0,MEMRA_PP_Q8MMQ=0,MEMRA_FAST=0,MEMRA_NOFA=1,MEMRA_GDN_CHUNKED=0`.
  The variables are set for the prefill forwards only. On Qwen3.8-27B this is f32 everywhere.
  On Ornith the MoE expert path of the prefill is not fully f32: its layer-0 FFN differs from
  the f32 decode by 0.13% with an identical input.
- **KV round-trip** (`RB_PREFILL_KVQ=1`): the prefill's full-attention layers pass their
  post-RoPE K through q8_0 and V through q5_1 (fp16 block scales) on the host before
  attention. The round-trip is not bit-equal to the engine's quantizer: at layer 3 it leaves
  3.7e-4 against the f32 decode, where f32 K/V leaves 2.2e-3.

## Qwen3.8-27B row 1198 (next token "P")

| program | K/V | top-1, margin | runner-up |
|---|---|---|---|
| f32 prefill | f32 | P 4.970 | F |
| GB10 prefill, F8F4 off (= sm_120a prefill, 4.916) | f32 | P 4.916 | F |
| GB10 prefill, default (e4m3 activations) | f32 | P 7.077 | F |
| GB10 prefill, default with naive attention, sequential GDN or FAST=0 | f32 | P 6.70 to 7.16 | F |
| f32 prefill, K/V round-tripped | q8_0 / q5_1 | P 8.802 | PS |
| GB10 prefill F8F4 off, K/V round-tripped | q8_0 / q5_1 | P 8.847 | PS |
| eager decode (GB10 = RTX 5090 Laptop) | quantized cache | P 8.647 | Ps |
| eager decode, `MEMRA_FAST=0` | quantized cache | P 8.606 | PS |
| eager decode, `MEMRA_FA_SPLIT=8` / `64` | quantized cache | P 9.30 / 9.56 | Ps |
| prime, default / F8F4 off | quantized cache | P 5.99 / 4.21 | F |
| prime, F8F4 off, `MEMRA_PRIME_F32CHUNK0=1` | f32 | P 5.02 | F |

`MEMRA_FA_SMEM_TKV=0`, `MEMRA_FA_EMIT=0`, `MEMRA_FA_F16PV=0` and `MEMRA_GDN_MMA=1` change no
bit at this row on GB10.

Default prefill (F8F4) against decode: at the hidden state's massive dimension (3994, about 16
at layer 0 and 50 from layer 11) the layer-0 GDN output is 14.684 in the prefill and 14.633 in
decode (f32 decode 14.637, f32 prefill 14.637). The GDN mixer's gain (relative difference out
over relative difference in) is 7.2, 6.2 and 5.2 at layers 24 to 26 of row 1198, against 2.7 to
3.6 at rows 1197 and 1199 (`analysis/q38-gain-1197-1199.txt`).

Neighbours, default programs, prefill against decode:

| row | 600 | 1190 | 1197 | **1198** | 1199 | 1200 | 1500 |
|---|---|---|---|---|---|---|---|
| logits rel | 0.032 | 0.057 | 0.045 | **0.379** | 0.119 | 0.064 | 0.065 |
| logits max abs | 0.41 | 1.06 | 0.83 | **10.96** | 2.73 | 2.19 | 1.32 |
| last residual rel | 0.037 | 0.086 | 0.079 | **0.801** | 0.208 | 0.087 | 0.114 |

The prime is not settled. Over the quantized cache it lands on prefill's side (4.21 with F8F4
off, 5.99 by default), not on decode's. With the W4A8 and Q8 MMQ tiles latched off (the
f32-prefill run, so the prime takes the int8 tiling GEMM) it lands on decode's (8.533), and with
`MEMRA_FAST=0` added it is back at 4.934. At a row this sensitive the prime's side follows the
sum of its errors. The decode's does not: its activation format, its split and its smem twin
all leave it at 8.6 to 9.6.

## Ornith-1.5-35B-A3B rows 41, 306, 727 and 1809 (eager)

Margin of the top-1 (runner-up in brackets); "f32-class" programs are bold.

| program | 41 ("0") | 306 ("7") | 727 ("9") | 1809 ("---") |
|---|---|---|---|---|
| **f32 prefill** | 0 2.36 (3) | 7 1.63 (8) | 1 5.07 (9) | 7 3.97 (**) |
| **f32 prefill, K/V round-tripped** | 0 3.83 (3) | 7 3.60 (5) | 1 2.57 (9) | 7 0.07 (**) |
| **eager decode, FAST=0** | 0 2.97 (3) | 7 1.61 (9) | 1 4.06 (9) | ** 0.13 (7) |
| eager decode | 0 2.90 (**) | 7 4.49 (8) | 1 2.84 (9) | ** 0.07 (7) |
| GB10 prefill F8F4 off (= sm_120a prefill) | 3 1.47 (0) | 7 0.93 (9) | 1 4.30 (9) | 7 4.08 (**) |
| GB10 prefill default | 0 0.80 (3) | 7 4.71 (5) | 1 6.42 (9) | 7 0.74 (---) |
| prime, F8F4 off, quantized cache | 0 1.97 (3) | 7 3.77 (5) | 1 3.02 (9) | 7 3.80 (**) |
| prime, F8F4 off, f32 chunk-0 K/V | 3 0.44 (0) | 7 0.94 (9) | 1 3.10 (9) | 7 3.89 (**) |

- Row 41: every f32-class program has the true token "0" at 2.4 to 3.8. The sm_120a prefill's
  "3" is the odd one. Which of its parts moves it (int8 W4A8 GEMMs, bf16 attention, chunked
  GDN, the MoE prefill path) was not separated.
- Row 1809: a three-way near tie of "7", "**" and "---" (the true token). The KV format is the
  largest lever: the f32 prefill's 3.97 for "7" becomes 0.07 with the round-trip.
- Row 306: the decode's own activation format moves it 4.49 to 1.61.

Router (`analysis/router-summary.txt`). The host top-k from each program's own post-attention
residual equals the engine's selection at all 40 layers in both programs, with weights within
3.7e-5 (`analysis/orn-router-validation.txt`, `MEMRA_MOE_WEIGHT_TRACE`). Prefill (default)
against decode first selects differently at layer 1 on rows 41, 306 and 1809 and at layer 8
on 727. At layer 1 of row 41 the 8th and 9th experts are 0.0013 apart in prefill (51 at
-4.1001, 134 at -4.1013), and decode moves expert 51 by 0.041. The selected sets then differ
at 29, 16, 28 and 17 of the 40 layers. The two f32-class programs (f32 prefill against f32
decode) differ at 13, 12, 22 and 15 layers, from layers 12, 5, 9 and 14.

Over the 16-row sweep (`analysis/orn-sweep-ffngain.txt`) a layer whose selection differs
multiplies the residual's relative difference by a median of 1.97 (p90 4.93) through its FFN,
against 1.62 (p90 2.46) where it is the same.

## Questions of the issue

1. Which program holds the true next token at Ornith rows 41 and 1809: at 41 the f32-class
   programs and decode ("0"); the sm_120a prefill's "3" is the outlier. At 1809 the true token
   "---" is top-1 in none; the f32 prefill says "7" at 3.97, decode "**" at 0.07.
2. The router decides differently between the programs at most layers of these rows (above).
   At none of them is it a single flip.
3. Qwen3.8-27B row 1198: the f32-K/V side (the sm_120a prefill) is closer to the reference.
   The quantized K/V moves it, not the decode kernels.
4. GB10: the same rows are not wide there because GB10's prefill is another program (F8F4).
   With F8F4 off it is the sm_120a prefill to every printed digit.

## Limits

One device. The PRO 6000 decode (another split ladder) was not run here. The Ornith f32
prefill's MoE path is not fully f32. The host KV round-trip is close to the engine's format,
not equal to it. Each cell is a single deterministic run (repeat runs of the same arm were
byte-equal).

## Files

- `runs/<arm>/{log,result.txt}`: one `row-bisect` run per arm (binary sha256, prompt sha256,
  device, wall time). The arm lists and batch logs are in `batches/`. Arm names: `pf32` = f32
  prefill, `kvq` = K/V round-trip, `f8f4off` = `MEMRA_MMQ_F8F4=0` on the prefill,
  `dfast0` = `MEMRA_FAST=0` for the whole process.
- `analysis/`: per-layer comparisons (`xarm.py`, `stages.py`, `gain.py`, `norms.py`, `dim.py`,
  `ffngain.py`, `check_route.py` in `tools/`), `margins.tsv` (every row of every run).
- `tools/row-bisect.diff`: the diagnostic patch (the five commits of `bank/1203-row-bisect` on 96390ef0e4).
- Binaries (sha256 prefix to commit on `bank/1203-row-bisect`): `1037a1865db2` e9676a1026,
  `ef2c8395f42d` 6292c1d465, `1952d94b36a3` 695f346e5b, `8986e5f67aed` f0453bb32e. Built on
  the unit with `MEMRA_CUDA_ARCH=121a`, LTO off (host code only; the device code is the same).
- The per-layer taps (813 files, 1.2 GB of f32 rows, `taps/<program>-r<row>.{f32,idx}` per
  run) are kept off the tree, in the lab's evidence store as snapshot `192d27c2`.
