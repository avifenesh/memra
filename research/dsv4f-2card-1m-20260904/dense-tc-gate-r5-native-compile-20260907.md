# Dense FP8 to BF16 native packed conversion R5, compile-only, 2026-09-07

## Result

CUDA 13.3.73 PTX 9.3 successfully compiled the native packed conversion for
both requested family/architecture targets:

```ptx
cvt.rn.bf16x2.e4m3x2 d, a;
```

The generated SASS is present in both cubins as:

```text
F2FP.BF16.E4M3.UNPACK_B R7, R2, 4.5736980577097704378e-41.H0
```

The 13.3 toolchain is under
`target/cuda13.3-ptx/usr/local/cuda-13.3`; no CUDA 13.3 cudart or PTX JIT is
part of the loader path.

## Artifacts

```text
target/dsv4-dense-tc-gate-r5-sm120f.cubin
  sha256 1ea70e07c0e87986f291f28a26ff6e32227760c2fa3960eda0b8decb281f22a4
  target sm_120f, function r5_packed_cvt_basis

target/dsv4-dense-tc-gate-r5-sm120a.cubin
  sha256 67befaab9d1eda508ece2a416e003572897ed2771ea1d3a1591c68047cfc109e
  target sm_120a, function r5_packed_cvt_basis

target/dsv4-dense-tc-gate-r5-driver
  sha256 82e569472f5dad0dde9dda82835f85edc781828b51c25f614a3ed7f8bac0c14b

source hashes:
  tools/dsv4-dense-tc-gate-r5.cu
    ccbd2d26bd1cdfed1e1423fd87f014ac9a33aa08431e3c0d366f428e68390cbf
  tools/dsv4-dense-tc-gate-r5-driver.cpp
    341fa20e94ca73c048c1ffb444c982d3995dc6bf367f39a6ce9881b9b7c47e30
```

The driver shim is plain C++ compiled against the CUDA 13.1 driver header and
`libcuda` only. `readelf -d` shows `libcuda.so.1` and no `libcudart`; the shim
uses `cuModuleLoad` on the cubin, `cuModuleGetFunction`, `cuLaunchKernel`, and
driver memory copies. A cubin contains SASS, so this path does not request PTX
JIT.

Build commands:

```sh
target/cuda13.3-ptx/usr/local/cuda-13.3/bin/nvcc -std=c++17 -O3 \
  -fmad=false -Xcompiler=-ffp-contract=off \
  -gencode arch=compute_120f,code=sm_120f -cubin \
  -o target/dsv4-dense-tc-gate-r5-sm120f.cubin tools/dsv4-dense-tc-gate-r5.cu

target/cuda13.3-ptx/usr/local/cuda-13.3/bin/nvcc -std=c++17 -O3 \
  -fmad=false -Xcompiler=-ffp-contract=off \
  -gencode arch=compute_120a,code=sm_120a -cubin \
  -o target/dsv4-dense-tc-gate-r5-sm120a.cubin tools/dsv4-dense-tc-gate-r5.cu

/usr/bin/g++ -std=c++17 -O2 -Wall -Wextra \
  -I/usr/local/cuda-13.1/include -L/usr/local/cuda-13.1/lib64 \
  -Wl,-rpath,/usr/local/cuda-13.1/lib64 \
  -o target/dsv4-dense-tc-gate-r5-driver \
  tools/dsv4-dense-tc-gate-r5-driver.cpp -lcuda
```

## ABI and basis vectors

The cubin exports the stable C kernel:

```text
r5_packed_cvt_basis(const uint32_t* packed, uint32_t* out, int n)
```

The CUDA 13.1 driver shim passes these eight packed E4M3x2 vectors and checks
the exact BF16x2 words independently. The low byte is the low BF16 half and
the next byte is the high BF16 half; E8M0 exponent adjustment is zero in this
basis:

```text
input       expected BF16x2
0x00003838  0x3f803f80
0x0000bf80  0xbff00000
0x00007e38  0x43e03f80
0x00007f00  0x00000000
0x00003f38  0x3ff03f80
0x00007e7e  0x43e043e0
0x00000000  0x00000000
0x0000ff38  0x00003f80
```

This receipt is compile/SASS/ABI-only. Neither cubin nor shim has been loaded
or run on a GPU in this lane. R4 and the R5 CUDA 13.1 rejection provenance
remain frozen.
