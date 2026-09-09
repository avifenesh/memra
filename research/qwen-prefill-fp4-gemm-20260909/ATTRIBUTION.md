# Bounded INT8 attribution and final candidate

Nsight Compute 2025.3.1 is installed in CUDA 13.0. The single-launch profile was refused with `ERR_NVGPUCTRPERM`, including when run as container root. `ncu-denied/console.log` preserves the error. No GPU performance counters were collected. Hardware tensor utilization, hardware stall ranking, bank-conflict counts and achieved occupancy are therefore unavailable; the previous occupancy rows are static resource bounds only. No host driver settings were changed. The authorized fallback is the component-removal microbench.

## Component-removal results

Same source base, one binary, M=1024, current 128x128 pipeline and RP layout. Six interleaved samples, 20 calls/sample, both orders, q8 quantization included. Each diagnostic arm removes only the named component from steady-state work. They deliberately compute a wrong numerical program and cannot be selected for serving.

| Arm | K=5120 N=17408 ms | GEMM tok/s | Throughput delta | K=17408 N=5120 ms | GEMM tok/s | Throughput delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Current | 0.740314 | 1,383,196 | reference | 0.744343 | 1,375,710 | reference |
| Unpack only first weight tile, then reuse it | 0.681557 | 1,502,443 | +8.62% | 0.683703 | 1,497,726 | +8.87% |
| Replace per-group FP32 fold with INT32 collapse | 0.669533 | 1,529,425 | +10.57% | 0.673826 | 1,519,679 | +10.47% |
| Load q8 activation planes once, omit later global staging | 0.697419 | 1,468,270 | +6.15% | 0.703283 | 1,456,028 | +5.84% |

Unpack removes repeated shared-to-shared LUT expansion, while retaining weight cp.async staging and the tensor/fold program. The fold arm is the existing `MEMRA_MMQ_FOLD_CEILING` research control, applied at compile time only; its INT32 collapse combines unequal-scale groups and is intentionally invalid. Activation staging keeps the first two initialized shared planes and their shared-to-register operand loads; it removes subsequent global-to-shared copies, not every activation-related instruction. The initial activation planes are fully waited before reuse. No uninitialized memory is used as an operand.

Scale folding is the largest measured component-removal effect, but deleting its numerical work buys only about 10.5% throughput. These effects are not additive: removing work changes scheduling and overlap. They do not prove that all of the 51% register-roof gap is fold overhead or identify a particular MIO/LG/barrier stall counter. The unavailable counter attribution is not invented. The older laptop no-fold result (+3.17% end to end, `research/prefill-gemm-20260806/VERDICT.md`) is retained as a different device/protocol; these direct desktop GEMM rows are the new evidence.

## One exact candidate

Target only the fold. Keep the two exact INT8 K16 dots. If adjacent decoded weight scales are positive and exactly equal, add their INT32 results before converting once and multiplying by the common scale. Otherwise evaluate the original expression verbatim. Do not hoist the activation scale across rounded FP32 operations.

The proof for an eligible pair is narrow: abs(INT32 K16 dot) <= 16*12*128 = 24576; their sum is <=49152. E4M3 weights have a four-bit significand, so the products and sum at a common power-of-two exponent fit exactly in FP32. Positive scales avoid a changed negative-zero result when opposite integer dots cancel. The final activation-scale multiplication and running FP32 accumulation keep the original order. The CUDA bitwise oracle, not this argument alone, decides acceptance.

Two seeded fixtures per shape: the original varied finite E4M3 scales, and all scales equal as an eligibility upper-bound control. The latter is not substituted for the original acceptance fixture. No K32 MMA is introduced: merging distinct per-16 scales changes the program, and keeping the current fold with a single K32 result is not possible.
