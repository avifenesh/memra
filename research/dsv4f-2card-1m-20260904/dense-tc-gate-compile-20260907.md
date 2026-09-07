# Dense FP8 to BF16 MMA gate r3, 2026-09-07

## Feasibility

The loader already proves the transport premise for the candidate: stored E4M3
codes multiplied by the E8M0 power-of-two block scale are checked against the
host dequantized f32 tensor, then `f32_to_bf16_exact` refuses any value whose
low 16 bits are nonzero. BF16 has the same 8-bit exponent width as f32, so this
does not introduce a range reduction for finite model weights.

The candidate is nevertheless a named numeric class, not a bit-identity arm.
The current FP8 control accumulates each output through its own serial per-thread
chain and 128-leaf reduction. The BF16 tensor-core arm uses `mma.sync`
`m16n8k16.f32.bf16.bf16.f32`, changing the reduction order. An output-neuron
m=1 tile also replicates the one input vector across eight N columns, so it
does eight times the useful output work. Peak-rate metadata is not a measured
bound for this kernel. The tool therefore records a falsifiable control/
candidate result and makes no speed or quality claim.

## Gate

`tools/dsv4-dense-tc-gate.cu` contains:

- the actual current `memra_dsv4_gemv_fp8_m` C-ABI control entry by including
  `crates/memra-engine/cu/dsv4_gpu.cu`;
- an exact host BF16-shadow representability check; no BF16 weight mirror is
  uploaded or resident;
- a BF16 MMA output-neuron candidate named
  `dsv4_dense_bf16_mma_f32acc_m1`;
- identical input/weight data, matched cache flushes, CUDA-event timings,
  finite-output checks for every scored row, bitwise within-arm repeatability,
  and max absolute/relative control deltas for every scored candidate arm;
  the terminal receipt is explicit `PASS`.
- the in-tree `load_ldmatrix_A` and `load_ldmatrix_A_trans` helpers, with the
  latter's x0/x2 first-n=8 selection; all 16x16 B shared-memory elements are
  initialized before the x4.trans load. The candidate's B source staging is
  now natural row-major `[K][N]`, matching the validated attention V path.
- `--basis` runs independent K=16 and K=32 single-warp tests, writes all 16x8
  MMA outputs, and compares every row/column against a host BF16-input oracle.
  Full-shape timing is not admitted until these basis cells pass.

Default shape is the matched q_b class, rows 32768 and K 1024. Optional
arguments are `rows k flush_mb abs_tol rel_tol`. The executable runs three
ABBA cycles. FP8 codes/scales remain resident and the candidate decodes each
BF16 weight tile on the fly. There is no new model or activation quantization.

The default flush is queried from the active device's L2 attribute and uses
2× reported L2 bytes; a positive `flush_mb` argument overrides that query.
The flush allocation is explicitly zeroed before the first read. One warmup
run is executed for each arm before the three scored ABBA cycles (six scored
runs per arm).

## Compile

```sh
/usr/local/cuda-13.1/bin/nvcc -std=c++17 -O3 -fmad=false \
  -Xcompiler=-ffp-contract=off \
  -gencode arch=compute_120a,code=sm_120a \
  -lcublasLt -lcublas \
  -o target/dsv4-dense-tc-gate-r3 tools/dsv4-dense-tc-gate.cu
```

The original compile-only stage is preserved below; paired GPU results follow.
No production integration was made.

Compile result on the current CUDA 13.1 toolchain: PASS for `sm_120a`, with
the BF16 MMA path using the in-repo inline `mma.sync.aligned.m16n8k16` tile
primitive rather than the incomplete WMMA 16x8x16 specialization. The current
engine TU emits its pre-existing `shd` linkage warning twice; no new warning
was introduced by the standalone candidate. The r2 GPU numerical failure is
preserved in `dense-tc-gate-r2-failure-20260907.md`.

Final compile hashes for
`/home/avifenesh/projects/wt-dsv4f-2card-1m-20260904/target/dsv4-dense-tc-gate-r3`:

```text
binary 02d651e66619f603a224a2f482321d1ef90b002605c0caa866717e03dc6ace26
source 1e0d271e411a105d3af687bcf8c45c72482c5a176466da8f78122fd58170c94a
```

## Paired GPU verdict

Do not integrate this layout. Both devices pass the independent K16/K32 basis
test with zero observed error and the full-shape memcheck with zero sanitizer
errors. Full-shape output is within the declared, unchanged numeric band:
max absolute 0.0341796875, max relative 0.000980377197. That is not bit identity
to the current GEMV and is not a model-quality gate.

Uninstrumented cold-cache timing, 3 ABBA cycles / six observations per arm:

| Device | Current FP8 GEMV, us | Candidate BF16 MMA, us | Candidate/current latency |
|---|---:|---:|---:|
| GPU 0 | 43.067 | 202.176 | 4.6945 |
| GPU 1 | 42.971 | 203.232 | 4.7296 |

The layout fix resolved a real B-tile orientation error; it did not produce a
performance win. Instrumented timings are excluded. Raw namespaces retained
by the controller: `dense-tc-{basis,memcheck,rate}-20260907-r3`. No engine
dispatch arm or runtime flag was added.

Primary interface checks used for the prototype:

- CUDA 13.2 Programming Guide BF16/BF16/f32 MMA shape table,
  <https://docs.nvidia.com/cuda/cuda-programming-guide/pdf/cuda-programming-guide.pdf>.
- Existing SM120 BF16 MMA implementation and rate note in
  `crates/memra-engine/cu/flash_attn.cu`.
