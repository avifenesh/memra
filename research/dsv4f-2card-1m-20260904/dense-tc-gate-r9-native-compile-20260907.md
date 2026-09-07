# Dense FP8 R9 negative-zero-correct native GEMV, compile-only, 2026-09-07

## Correction

`dsv4_gpu.cu:67-75` preserves negative zero: `mag==0` passes through
`sign * raw`, while only `mag==0x7f` NaN encodings map to `+0`. R9 supersedes
the R6/R7/R8 notes' incorrect claim that engine normalization maps signed zero
to `+0`; those older engine-equivalence claims are superseded, while their raw
characterization/artifact receipts remain valid.

R9 uses the corrected normalizer and preserves the original F32 GEMV order:
all `i0` products are accumulated before all `i1` products in the double-stride
path. The candidate remains a native E4M3 pair conversion followed by the
same F32 scale/input multiply and 128-leaf reduction.

## Control basis and oracle

The CUDA 13.1 launcher includes the actual current `dsv4_e4m3` helper in a
control kernel `r9_control_e4m3x2_basis`. It compares the native cubin's
`r9_packed_cvt_basis` against that control kernel for all 65,536 packed pairs.
A separate CPU IEEE oracle independently computes E4M3, preserving negative
zero and mapping only `mag==0x7f` to `+0`; all three outputs must match exactly.
The basis checks the native 3-argument ABI and GEMV 7-argument ABI.

## Artifacts

```text
target/dsv4-dense-tc-gate-r9-sm120f.cubin
  sha256 67717929120a1cee717fa1f1a6f05b937788aba997bb6d43158623c99c7bf8b8

target/dsv4-dense-tc-gate-r9-sm120a.cubin
  sha256 58c1f9cc92e4230dab07191a2806cbc14c16c4d1398740c744a1d8167be0b95d

target/dsv4-dense-tc-gate-r9-driver
  sha256 90ebde03ce24c75fc88e8fc6df9046a54b846b8b659219ce7d8802047166b71b
```

Source hashes:

```text
tools/dsv4-dense-tc-gate-r9.cu
  40d403db161ccad9e1b4ea8aac58cdc7f5eec90252e731993f693013b7cb25ce
tools/dsv4-dense-tc-gate-r9-driver.cu
  74de8566cac6175f96a8a3c48593eb76e35511697e06037979d6cf3ccda47824
```

The launcher retains the R8 warmup, 2x-L2 flush, guard, bit-repeatability,
full output bit-identity, K=128/256/2048/4096/8192 padded basis, signed-zero /
NaN / ±4-scale edge case, and explicit CPU cancellation schedules (ordered 7,
pair-interleaved 13, element-interleaved 14). No R9 cubin or launcher was run
on a GPU in this lane.
