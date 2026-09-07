# DSV4 attention existing-GEMV component

This is a correctness component gate for the first rank-local attention split. It is not a
runtime integration, throughput result, or full-model TP qualification.

The gate calls the existing `memra_dsv4_gemv_fp8_m` FFI for the real DSV4 shapes:

- `Q_b`: 32768 x 1024, with each logical rank taking one exact 16384-row half;
- `wo_a`: 8192 x 4096, with eight distinct 1024-row groups, four per logical rank;
- `wo_b`: 4096 x 8192, with each logical rank taking and repacking one 4096-column half to
  the physical stride expected by the existing GEMV.

`Q_b` and `wo_a` compare full-width outputs with the rank-local row outputs by bits, including
signed zero. The `wo_b` host-packed and device-packed rank halves also compare by bits, with
finite outputs, immutable input/weight planes, and output canaries. The host rank-order FP32
sum is exposed under the named numeric class
`dsv4_attention_wo_b_input_split_f32_rank_reduce`.

The full-width `wo_b` diagnostic was `false`, as expected for the changed input-split reduction
class. This component does not claim native GPU rank join, tolerance-based equivalence, or a
speed win. The test does not select a serving path or change any default.

## Bound receipt

The remote run used source `17772f148e33cc3d8afdd90b5f66c16b16c561ce`, Rust 1.97.1, CUDA
13.1.115, and `sm_120a` on the two RTX PRO 6000 Blackwell devices. The matching local consumer
source commit is `5efb311bd6e19ae19bf6f841f74fbd81917988b2`; the exercised module SHA256 is
`5743c782ab3d55aa70877590c90fe79975b372a82dd6f3fcf623d48005449d4e`.

The ignored gate is
`dsv4_attention_split::tests::cuda_attention_split_existing_fp8_gemv_component_gate`.
It passed normally and under memcheck on both physical devices. Both memcheck runs reported
zero errors. The test binary SHA256 is
`51bc1be92a46a67e7d90a7165493669e857de7dcc55baa48edc6c00f72e433c4`.

The complete raw CPU/build/GPU/memcheck receipt, source, binary, scripts, metadata, and hashes
are retained under the private ops namespace:
`research/dsv4f-devpair-20260905/receipts/ondemand/attention-pack-f72dc-gemv-20260907-gpu/`.
The pinned PR327 packer binary remains separately preserved and is not reused as evidence for
this GEMV gate.

No runtime attention caller, native rank reduction, model-level rate, serving qualification,
or production artifact was changed by this component.
