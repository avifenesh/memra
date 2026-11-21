# Inkling 2 Mini: the act-quant class of the routed experts

Lane D of the Inkling 2 Mini onboarding, 2026-11-21. Scope: the routed experts only, in the
checkpoint's declared act-quant program (`hf_quant_config.json`, `quant_method: inkling2`):
NVFP4 W4A4 with per-token amax and `fouroversix_mae` scale selection, and MXFP8 W8A8.
Nothing here is a support state: no pack, no serving route. Which layers run the class is a
per-layer plan choice that waits for per-layer KL receipts.

Card: one RTX 5090 Laptop GPU (sm_120a, 82 SMs, 64 MiB L2), driver 615.71.09, CUDA 13.5 nvcc.

The first round's receipts are in `receipts-20261121/`; its timed numbers were taken in a clamped
power state and are void (see [Microbench](#microbench-at-full-clocks)). This round, after the
independent review of memra#1396, is in `receipts-20261121b/`.

## What was built

| Piece | Where |
|---|---|
| CPU reference | `crates/memra-reference/src/inkling2_actquant.rs` |
| Golden vectors (31 records) and their generator | `crates/memra-reference/src/fixtures/inkling2-actquant-goldens.bin`, `gen_goldens.py` |
| Device capture of the golden GEMM records | `crates/memra-reference/src/fixtures/inkling2-actquant-sm120-capture.bin` |
| Kernels | `crates/memra-engine/cu/inkling2_actq.cu` |
| Engine entries, gate, bench | `crates/memra-engine/src/inkling2_actq.rs`, `ink2-actq-gate` |
| kernel-check cells | `ink2-actq-nvfp4`, `ink2-actq-mxfp8`, `ink2-cvt-hw`, `ink2-situ-actq-nvfp4`, `ink2-situ-actq-mxfp8`, `ink2-w4a4-gemm`, `ink2-w8a8-gemm`, `ink2-expert-block-nvfp4`, `ink2-expert-block-mxfp8` |

## Goldens: a port, not an import

The serving image's ops are triton kernels and do not import on CPU. `gen_goldens.py` is a
line-for-line torch CPU float32 port of the quantizers (`uv run --with torch --with numpy python3
gen_goldens.py OUT.bin`, torch 2.14 CPU). Every quantizer operation is one IEEE fp32 operation in
the source order. Two parts are not a port of the image and say so: the SiTU (next section) and the
GEMM accumulator (the goldens carry the index-order sum; the device chain is a separate model).

Coverage: random rows at K = 2304, 1152 and 64; adversarial rows (ties at every e2m1 threshold for
both candidate scales and one bf16 step either side, all-zero and -0 rows, a single maximum at
either end, denormal rows whose block scales are e4m3 subnormals, the largest finite bf16, rows
whose inverse overflows to FP32_MAX, block ladders over 2^-20..2^20, e4m3 scale halfway points);
for MXFP8 also amax exactly at 448 * 2^j and one step either side, e4m3 halfway values, exponent 0
and 254 edges; non-finite rows for both formats; two W8A8 GEMM records whose weights are 97% and
100% e4m3 subnormal codes. The first golden set let three wrong readings through; the generator
searches random rows for ones on which each of six readings changes the output and commits 48 rows
per reading.

Non-finite inputs follow the source operation by operation: `tl.max`, `tl.minimum` and
`tl.maximum` without `propagate_nan` are IEEE maxNum / minNum (a NaN operand is dropped, an
all-NaN reduction is NaN, made the quiet NaN `0x7fc00000`); the MXFP8 clamp propagates NaN; the
exponent-255 block (an all-NaN block) gets the source's NaN scale and zero inverse
(`mxfp8.py:23,25`, missing from the first round's port); e4m3 of NaN is `0x7f` and e2m1 of NaN is
`0x7` for either sign, what the sm_120 conversions the image uses return (measured, `ink2-cvt-hw`).
The first round's reference and kernel disagreed on these rows: e7204c97a0's reference fails the new
non-finite records (`receipts-20261121b/review-probes/red-old-reference-vs-new-goldens.log`).

## SiTU: the mathematical value, rounded once

The program is `bf16(cap_g tanh(g / cap_g) sigmoid(g) * cap_u tanh(u / cap_u) * gamma)`, the real
value rounded once to bf16, with `g` and `u` the bf16 gate and up values and `gamma` the fp32
routing weight (coordinator ruling for #1396: the correct program, not the library functions and
not the image's `ex2.approx` / `rcp.approx` / `tanh.approx` sequence). The inputs are bf16, so
each one-input factor is a table over all 65,536 bf16 bit patterns, in f64, built on the host from
the reference's f64 functions (`situ_tables`). The kernel multiplies `(TG[g] * TU[u]) * gamma`
in f64 (two IEEE multiplies, nothing to contract) and rounds once to bf16 through round-to-odd fp32,
so the device and `reference::situ_value` are the same arithmetic. Neither fp32 factor rounding
nor the library or approx functions remain; the measurement arm is deleted.

Held bit for bit, no tolerance (`ink2-situ-actq-*`): every bf16 pattern as a gate and as an up
value (65,536 outputs), every gate with exponent field 117..133 against every such up
(18,939,904 outputs), and a decode-like sample, each also through the quantizer. On all of them no
f64 value lies within 2^-40 (relative) of a bf16 rounding boundary, so the one rounding does not
depend on the last bits of the f64 factors. Red arm: the first round's kernel output on the same
18.9M window (the reviewer's capture) differs from this program on 39,066 outputs (library arm) and
61,714 (approx arm) (`receipts-20261121b/review-probes/situ_red.py`).

## The tensor-core accumulator

The checkpoint declares fp32 accumulation and no order. An fp32 sum in index order does not
reproduce the sm_120 tensor core. Two models do, bit for bit on the RTX 5090 Laptop:

- **W4A4, `kind::mxf4nvf4` m16n8k64**: one accumulator chained over ascending 64-wide K steps;
  each step adds the exact sum of its 64 products with a single rounding toward zero
  (`dot_chain_rz`).
- **W8A8, `kind::mxf8f6f4` m16n8k32**: per 32-wide step, every product and the accumulator are
  truncated toward zero to `2^(emax - 25)`, `emax` the largest of the accumulator's exponent and
  the products' operand-exponent sums over nonzero pairs; the truncated terms are summed exactly and
  truncated to fp32 (`dot_chain_trunc`, guard 25). The operand exponent is the hardware's: the e4m3
  field exponent plus the ue8m0 block exponent, so a subnormal code counts as -6
  (`mxfp8_operand_exponents`). The first round took the normalised exponent (-9..-7 for a
  subnormal) and was wrong on subnormal operands: the reviewer's device captures with mostly and
  all subnormal weights differ from it on 43 and 55 of 512 accumulators; the field-exponent model equals the device on all
  512 of both, and on the real layer 21 w13/w2, wide-ladder and K = 128 captures
  (`receipts-20261121b/review-probes/trunc_check.py`). On this round's committed capture the first
  round's model differs from the device on 61 and 35 of 512 accumulators of the two subnormal records
  (`review-probes/capture-red.log`). A unit test builds the case the readings
  disagree on (`chain_trunc_reads_the_field_exponent`), and the gate now runs subnormal-heavy
  device cases on every run: two golden records and the model's shapes with 97% subnormal weight
  codes and heavy-tailed activations.

The committed device capture (`chain_models_equal_the_sm120_capture`, CPU) now covers the subnormal
records. The chain models are measured on one card. `kernel-check` gates the model match only on the
devices listed in `INK2_CHAIN_MEASURED` (the RTX 5090 Laptop); on any other device, GB10 and the PRO
6000 included, it reports the match and gates the summation bound and the epilogue
(see [GB10](#gb10)).

## W8A8: which program

Two programs compute the checkpoint's W8A8 from the same e4m3 codes and ue8m0 scales:

- **mxf8f6f4** (this kernel): `kind::mxf8f6f4` block-scaled e4m3 MMA with the truncating chain above.
- **fp8 -> bf16, then a bf16 MMA** (the image's sm_120 path, `moe_sm120.py:128-150`): each operand
  dequantized to bf16 (exact on these inputs: 0 inexact of 2.3e9 weight and 1.1e6 activation
  values), then `mma.sync m16n8k16` bf16 with an fp32 accumulator chained over ascending K.

Both ran on the device on real data: the shipped layer 20..23 experts and 24 tokens of the lane E
tape `p2_window.actq_f32` (MoE inputs of slots 20..23, positions 8..1530, their top-8 experts and
routing weights; the w2 inputs are the SiTU of exact w13 sums). Error against the exact sum, as a
fraction of `sum |a_i w_i|` (`receipts-20261121b/w8a8-programs/`):

| layer / projection | elements | mxf8f6f4: mean / p99 / max err over sum\|p\| | bf16 MMA: mean / p99 / max | rms err mxf8f6f4 / bf16 MMA | bf16 outputs that differ between the programs |
|---|---|---|---|---|---|
| L20 w13 | 442368 | 3.01e-9 / 1.72e-8 / 1.05e-7 | 3.13e-9 / 1.78e-8 / 1.05e-7 | 8.66e-8 / 8.98e-8 | 0.0009% |
| L20 w2 | 442368 | 4.80e-9 / 2.86e-8 / 1.43e-7 | 5.10e-9 / 3.05e-8 / 1.43e-7 | 3.56e-8 / 3.80e-8 | 0.0005% |
| L21 w13 | 442368 | 2.79e-9 / 1.68e-8 / 7.70e-8 | 2.90e-9 / 1.73e-8 / 7.91e-8 | 7.47e-8 / 7.74e-8 | 0.0002% |
| L21 w2 | 442368 | 5.09e-9 / 3.08e-8 / 2.17e-7 | 5.44e-9 / 3.28e-8 / 3.17e-7 | 3.11e-8 / 3.36e-8 | 0.0007% |
| L22 w13 | 442368 | 2.97e-9 / 1.78e-8 / 1.41e-7 | 3.09e-9 / 1.85e-8 / 1.41e-7 | 9.42e-8 / 9.77e-8 | 0.0005% |
| L22 w2 | 442368 | 6.49e-9 / 4.18e-8 / 2.63e-7 | 6.95e-9 / 4.52e-8 / 3.26e-7 | 1.39e-7 / 1.50e-7 | 0.0005% |
| L23 w13 | 442368 | 3.08e-9 / 1.91e-8 / 1.39e-7 | 3.20e-9 / 1.97e-8 / 1.59e-7 | 1.02e-7 / 1.06e-7 | 0.0005% |
| L23 w2 | 442368 | 6.20e-9 / 3.86e-8 / 4.80e-7 | 6.65e-9 / 4.17e-8 / 5.90e-7 | 8.66e-8 / 1.02e-7 | 0.0007% |

mxf8f6f4 has the lower error on every layer and projection (mean 4 to 7% lower, rms 3 to 17% lower,
max equal or lower). It stays; the bf16-dequant program loses on error and on prefill speed (its
MMA peak on this card is 116 TF against 432-443 TF for block-scaled FP8) and is not built. Decode
is bandwidth-bound either way. The two programs' bf16 outputs differ on 2 to 9 per million
elements, so no KL receipt is needed to choose between them; the per-layer KL of the act-quant
class itself is lane F's.

## Microbench at full clocks

The first round's numbers (`receipts-20261121/`) were taken with nvidia-powerd latched in a low
power state (SM about 1.1 GHz, memory switching between 9001 and 11001 MHz, the instrument and the
kernel in different clock states). They are void, and so are its roofline fractions and its
"net of the call overhead" column, which subtracted launch cost that overlaps execution.

This round: two `gpulock --perf` leases, 08:29-08:31 and 09:12-09:18 (Jerusalem), AC power,
platform profile performance, enforced power limit 164-165 W, driver 615.71.09. `clampcheck.sh` before
each (lease 1: unsure, one sample with SW Power Cap at idle; lease 2: OK). 250 ms telemetry
(`smi.csv`, cut per phase by `telemetry.csv`): every loaded sample of every phase at P0 and
14001 MHz memory, SM 1.6-2.7 GHz (power-limited), up to 88 C; no take voided. The kernel's three takes
are lease 1's bench and lease 2's two arms of the same binary (`ink2-actq-gate` sha256
`46e2ae7b...`, `r3.sha256`), which is also the binary the committed tree builds.

Method: every call reads a different set of experts out of 128, so the weights come from DRAM; time
per call is host wall time over 200 back-to-back asynchronous calls, best of 5 per warps-per-block
arm, best arm kept; GB/s is weight bytes over that time, nothing subtracted. The same-size read
roofline is `tools/bwslice.cu`: the ceiling lane's streaming-read kernel launched back to back on a
different slice of a 3 GiB pool per call, timed the same way, at exactly each row's weight bytes,
before and after the bench in both leases (median of 4). The card's large-buffer read roofline is
865 GB/s; the MMA peaks at full clocks are FP4 block-scaled 910 TF and FP8 block-scaled 432-443 TF
(ceiling lane).

| GEMM | tokens (experts read) | weight MB | us/call, median (range) | GB/s | same-size read roofline, GB/s | of that | of 865 GB/s | TFLOP/s (of the MMA peak) | best wpb per take |
|---|---|---|---|---|---|---|---|---|---|
| W4A4 w13 | 1 (8) | 23.9 | 57.5 (57.5-57.7) | 415 | 831 | 50% | 48% | 1.48 (0.2%) | 1,1,1 |
| W4A4 w13 | 2 (16) | 47.8 | 108.0 (107.7-108.6) | 442 | 849 | 52% | 51% | 1.57 (0.2%) | 1,1,1 |
| W4A4 w13 | 4 (32) | 95.6 | 229.1 (225.7-236.4) | 417 | 856 | 49% | 48% | 1.48 (0.2%) | 1,1,1 |
| W4A4 w13 | 8 (64) | 191.1 | 372.5 (370.9-372.8) | 513 | 860 | 60% | 59% | 1.82 (0.2%) | 1,1,1 |
| W4A4 w13 | 256 (128) | 382.2 | 814.1 (798.8-863.9) | 469 | 863 | 54% | 54% | 26.71 (2.9%) | 4,4,4 |
| W4A4 w2 | 1 (8) | 11.9 | 22.4 (22.3-22.5) | 533 | 775 | 69% | 62% | 1.90 (0.2%) | 4,4,4 |
| W4A4 w2 | 2 (16) | 23.9 | 42.4 (42.2-42.7) | 563 | 831 | 68% | 65% | 2.00 (0.2%) | 1,1,1 |
| W4A4 w2 | 4 (32) | 47.8 | 72.5 (72.5-73.1) | 659 | 849 | 78% | 76% | 2.34 (0.3%) | 1,1,1 |
| W4A4 w2 | 8 (64) | 95.6 | 137.0 (134.8-140.0) | 697 | 856 | 81% | 81% | 2.48 (0.3%) | 1,1,1 |
| W4A4 w2 | 256 (128) | 191.1 | 332.2 (326.6-332.8) | 575 | 860 | 67% | 67% | 32.72 (3.6%) | 4,8,8 |
| W8A8 w13 | 1 (8) | 43.8 | 96.5 (96.4-96.5) | 454 | 848 | 54% | 52% | 0.88 (0.2%) | 1,1,1 |
| W8A8 w13 | 2 (16) | 87.6 | 182.1 (181.7-183.0) | 481 | 856 | 56% | 56% | 0.93 (0.2%) | 1,1,1 |
| W8A8 w13 | 4 (32) | 175.2 | 352.8 (351.9-353.7) | 497 | 860 | 58% | 57% | 0.96 (0.2%) | 1,1,1 |
| W8A8 w13 | 8 (64) | 350.4 | 692.1 (685.9-700.9) | 506 | 863 | 59% | 59% | 0.98 (0.2%) | 1,1,1 |
| W8A8 w13 | 256 (128) | 700.7 | 2042.5 (1921.4-2043.9) | 343 | 865 | 40% | 40% | 10.65 (2.5%) | 4,4,4 |
| W8A8 w2 | 1 (8) | 21.9 | 49.4 (49.4-49.6) | 443 | 826 | 54% | 51% | 0.86 (0.2%) | 1,1,1 |
| W8A8 w2 | 2 (16) | 43.8 | 94.1 (93.3-94.4) | 465 | 848 | 55% | 54% | 0.90 (0.2%) | 1,1,1 |
| W8A8 w2 | 4 (32) | 87.6 | 174.1 (174.1-175.7) | 503 | 856 | 59% | 58% | 0.98 (0.2%) | 1,1,1 |
| W8A8 w2 | 8 (64) | 175.2 | 341.0 (328.2-341.4) | 514 | 860 | 60% | 59% | 1.00 (0.2%) | 1,1,1 |
| W8A8 w2 | 256 (128) | 350.4 | 951.2 (930.1-956.3) | 368 | 863 | 43% | 43% | 11.43 (2.6%) | 4,4,4 |

Per token at M = 1, one NVFP4 layer's experts (w13 + w2, 35.8 MB) take 79.9 us (448 GB/s, 52% of
865) and one MXFP8 layer's (65.7 MB) 145.9 us (450 GB/s, 52%), plus two quantizer calls.

The decode GEMM reaches 48 to 60% of the read roofline on w13 and 51 to 81% on w2, not the roofline:
at full clocks the kernel, not DRAM, sets the time. The first round's "97 to 113%" was an artifact of
the clamp. Prefill runs at 2.5 to 3.6% of the block-scaled MMA peaks: the prefill body is the decode
body with more B tiles, and a tile that reuses one weight read across many rows is not built.

Rejected this round (lease 2, `bench-2.txt` and `bench-3.txt` against `bench-1.txt` and
`bench-4.txt`, both orders): staging only the largest group count of activation rows (less shared
memory per block, more resident warps) together with issuing the weight ring's prologue before the
staging. Slower on every decode row (W4A4 w13 M = 1: 65.1 against 57.6 us; W8A8 w13 M = 1: 143.6
against 96.5 us) and on prefill. Deleted.

### Quantizers

| quantizer | rows x k | us/call, median of 3 takes |
|---|---|---|
| NVFP4 | 1 x 2304 | 4.9 |
| NVFP4 | 8 x 2304 | 4.7 |
| NVFP4 | 8 x 1152 (+ SiTU) | 6.7 |
| NVFP4 | 64 x 1152 (+ SiTU) | 7.2 |
| NVFP4 | 2048 x 1152 (+ SiTU) | 53.2 |
| MXFP8 | 1 x 2304 | 4.6 |
| MXFP8 | 8 x 2304 | 4.6 |
| MXFP8 | 8 x 1152 (+ SiTU) | 7.0 |
| MXFP8 | 64 x 1152 (+ SiTU) | 7.1 |
| MXFP8 | 2048 x 1152 (+ SiTU) | 51.2 |

The review found 16-way (NVFP4) and 32-way (MXFP8) shared-memory bank conflicts in the per-thread
block reads. The row now sits in shared memory with one pad word per quantization block (17 or 33
words per block), so the 32 lanes, each reading element i of its own block, hit 32 banks. The
profiler counters are not readable on this rig (`ncu`: ERR_NVGPUCTRPERM), so the fix is measured by
time. Same lease, arms in the order e7204c97a0, padded, swizzled, old layout, old layout, swizzled,
padded, e7204c97a0 (`full-clocks-lease2/quant-ab.txt`, `tools/qbench.cu`; us per call, two takes;
the SiTU tables are zero-filled in this harness, so these SiTU rows time the table path, not real
data):

| quantizer, rows | e7204c97a0 | old layout, new NaN semantics, exact SiTU | padded (kept) | XOR swizzle, 16-byte reads |
|---|---|---|---|---|
| NVFP4, 1 | 3.08 / 3.06 | 3.51 / 3.52 | 3.46 / 3.46 | 3.47 / 3.48 |
| NVFP4, 8 | 3.07 / 3.06 | 3.52 / 3.52 | 3.46 / 3.46 | 3.47 / 3.47 |
| NVFP4 + SiTU, 8 | 3.78 / 3.76 | 5.47 / 5.47 | 5.30 / 5.30 | 5.45 / 5.45 |
| NVFP4 + SiTU, 64 | 4.08 / 4.07 | 5.96 / 5.97 | 5.58 / 5.57 | 5.92 / 5.95 |
| NVFP4 + SiTU, 2048 | 34.42 / 34.33 | 58.44 / 58.45 | 52.32 / 52.32 | 58.24 / 58.42 |
| MXFP8, 1 | 2.91 / 2.87 | 3.15 / 3.15 | 3.01 / 3.11 | 3.03 / 3.04 |
| MXFP8, 8 | 2.91 / 2.98 | 3.15 / 3.15 | 3.01 / 3.11 | 3.03 / 3.02 |
| MXFP8 + SiTU, 8 | 4.06 / 4.01 | 4.44 / 4.43 | 4.36 / 4.42 | 4.47 / 4.46 |
| MXFP8 + SiTU, 64 | 4.11 / 4.04 | 4.64 / 4.70 | 4.55 / 4.61 | 4.64 / 4.65 |
| MXFP8 + SiTU, 2048 | 27.86 / 27.24 | 48.82 / 48.82 | 48.60 / 48.62 | 48.46 / 48.42 |

- The padding is worth 0.05 to 0.14 us at 1 to 8 rows and 6 us (10%) at 2048 rows with SiTU on
  NVFP4. An XOR swizzle read as 16-byte chunks is no faster and loses 6 us at 2048 NVFP4 rows. Kept:
  padded.
- The non-finite semantics (NaN-start reductions, the NaN codes, the exponent-255 path) cost 0.4 us
  (NVFP4) and 0.25 us (MXFP8) per call at 1 to 8 rows.
- The exact SiTU costs 1.5 us at 8 rows and 18 to 21 us at 2048 rows over the first round's fp32
  library functions: two f64 multiplies and two 8-byte table reads per element, at the card's f64
  rate.
- At 1 to 8 rows a quantizer call is 3.0 to 3.5 us in this harness and 4.6 to 4.9 us in the engine
  bench, against a back-to-back launch cost of 1.5-2.5 us; at M = 1 the two calls of a layer are
  about 12 us beside 80 to 146 us of GEMM.

## GB10

Not measured. Both Sparks were under other leases for the whole lane (`sparklock status` at 07:27,
08:40 and 09:27 Jerusalem: spark-b23d held for "Hy4-flash decode tuning", spark-cb33 for "29b GB10
record landing", each about two hours left and renewed). So the gate does not hold an unmeasured
chain: `INK2_CHAIN_MEASURED` lists the RTX 5090 Laptop only, and on any other device (GB10, sm_121a,
and the RTX PRO 6000 alike) `ink2-w4a4-gemm`, `ink2-w8a8-gemm` and `ink2-expert-block-*` report the
chain-model match and gate the summation bound and the epilogue. Measuring GB10 is one
`ink2-actq-gate` run on a free unit; a match adds its device name to the list.

## Smaller review items

- Grid: groups are on grid.x (up to 2^31 - 1) and a group's N blocks on grid.y, so prefill chunks
  past 65,535 groups launch; the launcher refuses an N-block count above 65,535 (refusal 7).
- `Engine::ink2_expert_gemm_into` takes an `Ink2Groups` built on the host (`ink2_groups_upload`)
  and checks every buffer length against `experts x n x k` and `rows x k`, the table's experts and
  rows against the operands, the shapes the kernel needs and the launch shape before any launch.
- The reference has the expert block with the device's accumulators
  (`expert_block_w4a4` / `expert_block_w8a8` with `Accumulation::Sm120`), and the gate runs the
  whole block on the device against it (`ink2-expert-block-*`: gate_up, SiTU, down and the combine,
  0 differences).
- The NaN scale and zero inverse of the exponent-255 block are ported (section above).
- `lib.rs`: `inkling2_actq` has its own comment block.

## Gate (this round)

`ink2-actq-gate` on the RTX 5090 Laptop: ALL OK, 48 cells (`full-clocks-lease1/gate.log`):
quantizers 0 bytes differ on 13 + 5 golden records (non-finite rows included); the hardware e4m3
and e2m1 casts equal the software rounding on 1,027 + 110 values and on NaN; SiTU 0 differences on
the three sweeps above; W4A4 and W8A8 chain models bit-equal on every accumulator (golden records,
subnormal records, model shapes, subnormal-weight model shapes); both expert blocks 0 differences.
`cargo test -p memra-reference --lib inkling2_actquant`: 10 passed.

## Open

- The decode GEMM at 48 to 81% of the read roofline: find what the kernel waits on at full clocks
  (profiler counters are not available here) and close it. The default warps per block for a
  decode group is 2; one warp per block is faster on 15 of the 16 decode rows above.
- The prefill body (2.5 to 3.6% of the MMA peaks).
- GB10 and the PRO 6000 chain models (above).
- A capture from the serving image for the GEMM itself, to confirm its chain is the same K-step
  chain (its operands sit the other way round: activations on A).
- Wiring: no plan operation, pack or route reaches these kernels yet.
