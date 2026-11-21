# Inkling 2 Mini: the act-quant class of the routed experts

Lane D of the Inkling 2 Mini onboarding, 2026-11-21. Scope: the routed experts only, in the
checkpoint's declared act-quant program (`hf_quant_config.json`, `quant_method: inkling2`):
NVFP4 W4A4 with per-token amax and `fouroversix_mae` scale selection, and MXFP8 W8A8.
Nothing here is a support state: no pack, no serving route. Which layers run the class is a
per-layer plan choice that waits for per-layer KL receipts.

Card: one RTX 5090 Laptop GPU (sm_120a, 82 SMs, 64 MiB L2), driver 615.71.09, CUDA 13.5 nvcc.

## What was built

| Piece | Where |
|---|---|
| CPU reference | `crates/memra-reference/src/inkling2_actquant.rs` |
| Golden vectors (27 records) and their generator | `crates/memra-reference/src/fixtures/inkling2-actquant-goldens.bin`, `gen_goldens.py` |
| Device capture of the golden GEMM records | `crates/memra-reference/src/fixtures/inkling2-actquant-sm120-capture.bin` |
| Kernels | `crates/memra-engine/cu/inkling2_actq.cu` |
| Engine entries, gate, bench | `crates/memra-engine/src/inkling2_actq.rs`, `ink2-actq-gate` |
| kernel-check cells | `ink2-actq-nvfp4`, `ink2-actq-mxfp8`, `ink2-cvt-hw`, `ink2-situ-actq-nvfp4`, `ink2-situ-actq-mxfp8`, `ink2-w4a4-gemm`, `ink2-w8a8-gemm` |

## Goldens: a port, not an import

The serving image's ops are triton kernels and do not import on CPU. `gen_goldens.py` is a
line-for-line torch CPU float32 port of them (`uv run --with torch --with numpy python3
gen_goldens.py OUT.bin`, torch 2.14 CPU). Every quantizer operation is one IEEE fp32 operation
in the source order.

Coverage: random rows at K = 2304, 1152 and 64; adversarial rows (ties at every e2m1
threshold for both candidate scales and one bf16 step either side, all-zero and -0 rows, a
single maximum at either end, denormal rows whose block scales are e4m3 subnormals, the
largest finite bf16, rows whose inverse overflows to FP32_MAX, block ladders over 2^-20..2^20,
e4m3 scale halfway points); for MXFP8 also amax exactly at 448 * 2^j and one step either
side, e4m3 halfway values, exponent 0 and 254 edges. The e4m3 encoder agrees with torch's own
`float8_e4m3fn` cast on 1020 grid values.

The first golden set let three wrong readings through (sequential error sum, `encoder / 6`
instead of `encoder * (1/6)`, `amax / 1536` instead of `amax * (1/1536)`). The generator
now searches random rows for ones on which each of six readings changes the output and
commits 48 rows per reading. Ten mutations of the Rust reference then each fail its tests.

## Bit-exactness

| Claim | Result |
|---|---|
| Rust reference = torch port, quantizers | bit-identical, 27 records (`cargo test -p memra-reference --lib inkling2_actquant`: 7 passed) |
| NVFP4 quantizer kernel = goldens | 428 rows, 0 value bytes, 0 scale bytes, 0 amax differ |
| MXFP8 quantizer kernel = goldens | 44 rows, 0 bytes differ |
| `cvt.rn.satfinite.e4m3x2.f32` = round to nearest even | 0 of 1027 grid values differ (every code, every halfway point and its neighbours, both signs, the saturating inputs) |
| `cvt.rn.satfinite.e2m1x2.f32` = the software e2m1 | 0 of 110 differ |
| SiTU + quantizer kernel (library exp2f/tanhf) vs the f64-function reference | 0 of 18432 bf16 values differ; the quantization of the device's own rows is bit-identical |
| SiTU with the PTX approx instructions (measurement arm) | 12 of 18432 bf16 values differ, by one bf16 step |
| GEMM epilogue (row scale, bf16) from the device accumulators | 0 differ on every case |

The e4m3 open fact is closed: the hardware cast rounds to nearest even with subnormals and
saturates. The kernels use the software rounding, which is the same function and compiles on
every target.

The approx-math open fact is measured, not closed: the image's SiTU uses `tanh.approx`,
`ex2.approx` and `rcp.approx`. Our kernel defaults to the fp32 library functions, which match
the reference on every tested value; the approx arm stays as a kernel argument for the
measurement. Which one the checkpoint's numbers were calibrated under is not known from the
sources.

## The tensor-core accumulator

The checkpoint declares fp32 accumulation and no order. An fp32 sum in index order does not
reproduce the sm_120 tensor core: on the model's shapes 32 to 59% of W4A4 accumulators and 1.5
to 3.7% of W8A8 accumulators equal it. Two models do, bit for bit:

- **W4A4, `kind::mxf4nvf4` m16n8k64**: one accumulator chained over ascending 64-wide K steps;
  each step adds the exact sum of its 64 products with a single rounding toward zero
  (`dot_chain_rz`). Equal on 518,016 of 518,016 accumulators the gate runs.
- **W8A8, `kind::mxf8f6f4` m16n8k32**: per 32-wide step, every product and the accumulator are
  truncated toward zero to `2^(emax - 25)`, `emax` the largest of the accumulator's exponent and
  the products' operand-exponent sums; the truncated terms are summed exactly and truncated to
  fp32 (`dot_chain_trunc`, guard 25). Equal on 517,376 of 517,376. The W4A4 chain does not fit
  this model at guard 25 (98.8 to 99.9%), and the W8A8 chain does not fit the exact-sum model
  (7 to 15%).

The models were fitted on the device capture and then held on shapes they were not fitted on.
`chain_models_equal_the_sm120_capture` re-checks them on CPU against the committed capture
(1,280 accumulators) and asserts that the capture differs from the index-order sum.

Error against the exact sum, measured over the same cases: at most 1.6e-7 of `sum |a_i w_i|`
for W4A4 and 1.3e-6 for W8A8; the stated bound is `(K - 1) * 2^-23` of it (2.7e-4 at K = 2304).
In bf16 the device output differs from the index-order reference on 0 to 0.01% of W4A4
elements (one step, up to 9 steps where a sum cancels) and up to 0.08% of W8A8 elements (up to
237 steps where a sum cancels). The gate holds the accumulators and the epilogue, not that
distance.

## Kernel design

The weights sit on the MMA's 16-row A operand and the routed rows of one expert on its
8-column B operand, so a decode step (1 to 8 rows per expert) wastes nothing on the weight
side. Each warp streams its 16 weight rows in the stored row-major layout through a private
8-stage `cp.async` ring (16-byte copies) and `ldmatrix` reads a stage back as the A fragment
with no repack. NB B tiles share one weight read, for a prefill group of up to 8 * NB rows.

Rejected, with receipts in `receipts-20261121/`:
- direct 4-byte global loads into the A fragment, four K steps ahead in registers: same
  numbers as the ring (`bench-direct-load-body.log`), and more load instructions. Deleted.
- a 16-stage ring: no win over 8 at any decode or prefill shape (`bench-ring8-vs-ring16.log`).
  Deleted.

## Microbench (single window, N = best of 5 x 200 calls, weights from DRAM)

Conditions: platform profile quiet, power profile power-saver, GPU mostly P4 at 1117 MHz
SM / 9001 MHz memory clock (`receipts-20261121/conditions.txt`, `smi.csv`). Every repetition
reads a different set of experts out of 128, so the weight bytes do not come from L2.
The streaming-read roofline measured in the same lease, before and after the bench:
**271 GB/s** (512 MiB read, `bw-pre.txt`, `bw-post.txt`). The call overhead of the harness
is 4.4 us.

| GEMM, decode M = 1 (8 experts) | bytes | us/call | GB/s (net of call overhead) | of the same-window read roofline |
|---|---|---|---|---|
| W4A4 w13 gate/up [2304 x 2304] | 23.9 MB | 90.4 | 264 (278) | 97% (103%) |
| W4A4 w2 down [2304 x 1152] | 11.9 MB | 39.1 | 305 (344) | 113% (127%) |
| W8A8 w13 | 43.8 MB | 188.5 | 232 (238) | 86% (88%) |
| W8A8 w2 | 21.9 MB | 78.3 | 279 (296) | 103% (109%) |

Above 100% means the roofline instrument is a floor on the card, not that the kernel beats
the memory. The card's read roofline in the performance profile was measured by the roofline
lane at 455 to 486 GB/s (P2, 11001 MHz memory clock); this bench has not been run in that
state, so the fraction of that number is not known.

| Shape | W4A4 w13 | W4A4 w2 | W8A8 w13 | W8A8 w2 |
|---|---|---|---|---|
| M = 2 (16 experts) | 259 GB/s | 259 | 219 | 222 |
| M = 4 (32 experts) | 221 | 301 | 179 | 203 |
| M = 8 (64 experts) | 197 | 236 | 167 | 193 |
| prefill, 256 tokens (2048 rows, 128 experts) | 136 GB/s, 7.8 TFLOP/s | 144, 8.2 | 108, 3.4 | 110, 3.4 |

Quantizers: 12 to 23 us per call for 1 to 64 rows, 65 to 83 us for 2048 rows (one block per
row; at these sizes the call is launch-bound).

Per token at M = 1 the experts of one NVFP4 layer visit read 35.8 MB (8 x (w13 + w2), codes
and scales) and of one MXFP8 layer visit 65.7 MB. At the same-window roofline that is 132 us
and 242 us; measured 129.5 us and 266.8 us for the two GEMMs.

## Open

- The bench in the performance power profile, beside the roofline lane's instrument in the
  same window. Until then the M = 1 fraction is known only for the quiet profile.
- M = 4 and 8 fall to 73 to 87% of the same-window roofline, and the prefill group to about
  half: at 1117 MHz the per-step instruction cost shows. The prefill body is a decode body
  with more B tiles (2 to 8 TFLOP/s against a 319 TFLOP/s FP4 MMA peak); a tile that reuses a
  weight read across 128 rows is not built.
- The accumulator models are measured on one card. sm_121a and the PRO 6000 are not checked.
- Whether the checkpoint's serving numbers assume the approx SiTU instructions.
- A capture from the serving image for the GEMM itself, to confirm its chain is the same
  K-step chain (its operands sit the other way round: activations on A).
- Wiring: no plan operation, pack or route reaches these kernels yet.
