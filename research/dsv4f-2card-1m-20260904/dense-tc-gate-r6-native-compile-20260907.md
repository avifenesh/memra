# Dense FP8 to BF16 native packed conversion R6, 2026-09-07

> SUPERSEDED semantic note: the former R6 engine-normalized arm mapped
> `mag==0` signed-zero bytes to `+0`. Current `dsv4_e4m3` preserves negative
> zero and only maps `mag==0x7f` NaNs to `+0`. R9 is the corrective engine arm;
> R6 raw characterization and cubin provenance remain valid.

## Split semantics

R5's raw native conversion is retained as a characterization arm, but it is
not the engine oracle for E4M3 byte values with mag `0` or `0x7f`:

- `r6_raw_packed_cvt_basis` feeds every byte unchanged to native
  `cvt.rn.bf16x2.e4m3x2`. For all finite codes and signed zero, the driver
  checks exact BF16 bits. For `0x7f`/`0xff`, it checks the PTX-valid BF16 NaN
  class rather than assuming a payload/sign encoding.
- `r6_engine_packed_cvt_basis` maps `(code & 0x7f)==0` and
  `(code & 0x7f)==0x7f` to byte `0x00` before native conversion. This matches
  the current `dsv4_e4m3` serving contract, so its output is checked exactly
  for all 65,536 packed byte pairs, including asymmetric lane order and both
  signed-zero/NaN positions.

## Artifacts

CUDA 13.3.73 compiled both SASS targets:

```text
target/dsv4-dense-tc-gate-r6-sm120f.cubin
  sha256 7f4c9212555c1b44b3ad7bd56656b3c33dfab36c3e5b2a99df85ce232d763f3b

target/dsv4-dense-tc-gate-r6-sm120a.cubin
  sha256 91b8345669434e71835704542647d8b9b5dbf8febf0d3f522b82bc2f3dcb9cc6

target/dsv4-dense-tc-gate-r6-driver
  sha256 3e1bc5b151b0643ff6a6304183c88c2474f731835f17ccdcb6a0969a9ec11b01
```

SASS contains `F2FP.BF16.E4M3.UNPACK_B` in both raw and normalized kernels.
Each function uses 10 registers and zero shared memory.

The driver shim is plain C++ compiled with CUDA 13.1 headers and `libcuda`
only. It checks both kernels' three-argument ABI using
`cuFuncGetParamInfo`:

```text
param[0] offset=0  size=8   const uint32_t* packed
param[1] offset=8  size=8   uint32_t* out
param[2] offset=16 size=4   int n
```

`readelf -d` shows `libcuda.so.1` and no libcudart. The shim uses cubin
`cuModuleLoad`; it does not load PTX or request JIT. No cubin or shim was
loaded or run on a GPU in this lane.

Source hashes:

```text
tools/dsv4-dense-tc-gate-r6.cu
  65433480d55d36fed6c6ab5dba51286571117946ad77d58735ad3359df2e588c
tools/dsv4-dense-tc-gate-r6-driver.cpp
  2d2b06c74e6c839af5e830dde6b4a5f1486164dd22296c3ec5df88b13f290760
```

## Basis contract

The driver enumerates inputs `0x00000000` through `0x0000ffff`, where the low
byte is the lower BF16 lane and the next byte is the upper BF16 lane. It runs
raw first, then engine-normalized, and reports separate PASS lines. This is a
compile/ABI/oracle receipt only; R5 artifacts and source remain preserved.

The PTX semantics used here are documented in the [PTX ISA 9.3 conversion
section](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html):
E4M3 NaNs are `0x7f`/`0xff`, E4M3x2 converts upper/lower bytes to upper/lower
BF16 halves, and native conversion is `cvt.rn.bf16x2.e4m3x2`.

## Paired execution

Both target cubins loaded and passed all 65,536 raw and engine-normalized
pairs on both RTX PRO 6000 cards, with the existing 595.71.05 driver and
zero memcheck errors. No CUDA 13.3 runtime or driver update was installed.
Raw namespaces: `packed-conversion-{load,basis}-20260907-r6`.
This proves native conversion and the normalization contract, not a dense
projection or full-model performance gain.
