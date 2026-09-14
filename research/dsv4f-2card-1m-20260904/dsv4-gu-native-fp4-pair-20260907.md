# Native FP4 pair-store candidate, compile-only, 2026-09-07

## Candidate

`tools/dsv4-gu-native-fp4-pair-gate.cu` includes the geometry fixture and
current GU/M1+half2 launchers, then adds a standalone native pair-store GU
kernel. It preserves the ModelOpt nibble order and current helper semantics:

- nibble 8 (`-0`) is normalized to the helper's zero code;
- native `cvt.rn.f16x2.e2m1x2` produces the true E2M1 half2;
- a half2 multiply by 2 reconstructs the doubled `g_kvalues_mxfp4` codebook;
- the unchanged half2 scale multiply follows;
- A staging, K order, MMA order, gate/up order, and epilogue remain unchanged.

The `r9_fp4_pair_basis_kernel` compares current `kq_store_variant<108,true>`
against the native pair conversion over all 256 packed code bytes x all 256
signed-E4M3 scale bytes (65,536 cells), including nibble order and zero/sign
edges. The source also retains both current non-M1 and M1+half2 launcher
symbols from the geometry fixture for the eventual exact ABBA cell.

## Compile receipt

```sh
/usr/local/cuda-13.1/bin/nvcc -std=c++17 -O3 -fmad=false \
  -Xcompiler=-ffp-contract=off -gencode arch=compute_120a,code=sm_120a \
  -lcublasLt -lcublas -lcuda \
  -o target/dsv4-gu-native-fp4-pair-gate \
  tools/dsv4-gu-native-fp4-pair-gate.cu
```

```text
binary sha256 14acc3eef3c37ca6b1d067e65b4d4473e6be6c05714869152732d0c0608b7303
source sha256 4263e2379dc637401588b580e3157090a83892c169d798b89195dbab9e2dd873
native GU resource 138 registers/thread, 26128 shared bytes
current GU resource 150 registers/thread, 26128 shared bytes
basis resource 29 registers/thread, 2048 shared bytes
```

SASS contains `F2FP.F16.E2M1.UNPACK_B` in the native GU candidate. No GPU
execution, exhaustive basis launch, or production/library/Cargo edit was made.

Primary syntax/target reference: [PTX ISA 9.3 conversion
instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html),
which lists `cvt.rn.f16x2.e2m1x2` for `sm_120a`.
