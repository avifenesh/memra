# MLA CUDA forward — design decisions (increment 4)

## Form: ABSORBED everywhere

`mla.rs` proves two equal forms. The kernels run the absorbed one for prefill AND decode:

* the latent plane is the only KV state, so no arm needs expanded per-head K/V — the naive
  form would decompress kv_rank -> n_head*(d_nope + d_v) per cached token, materializing the
  exact tensor MLA exists to avoid;
* one code path serves prefill, chunked prefill and decode, so both gates test the same math;
* it mirrors `mla_attend_absorbed` line for line, making the gate a direct comparison.

Accepted cost: absorbed scores are `kv_rank + d_rope` wide (576 GLM-5.2 / 512 glm5_next)
against the expanded form's `d_nope + d_rope` (256) — ~2.2x score FLOPs at prefill shapes,
where expanded is cheaper. Correctness first; the fused GEMM-shaped decode kernel and an
expanded prefill arm are DESIGN.md increment 5.

## Cache layout

`memra_kv::LatentKvLayer`: f32 rows `[max_ctx][width]`, `width == kv_rank + d_rope ==
StatePlan::LatentKvCache { width } == MlaGeom::latent_dim` (576 GLM-5.2, 512 glm5_next NoPE).
One row per token for ALL heads (MQA). NO V plane — V is the first `kv_rank` elements of the
same row. `len`/`len_d` follow `KvLayer`'s lock-step convention.

f32 and unquantized on purpose: the gate is maxdiff against the f32 oracle, whose `c_kv` is
f32; a q8_0 plane would fork the thing under test from its truth. DESIGN.md §3.2's eventual
quantized row (576 = 18 blocks, V boundary 512 = 16 blocks, both 32-aligned) is a later
increment. Consequence: the KVQUANT `% 32 == 0` head-dim assert does not apply to this plane —
it was function-wide in `Cache::new_inner` and is now scoped to the quantized arms, because a
latent row is not head-shaped (the micro fixture's width 24 tripped it).

## RoPE: interleaved directly, permutation seam kept

GLM-5.2 is `rope_interleave: true` (llama.cpp NORM); memra's `rope_neox` pairs (j, j+half).
`mla.rs::norm_to_neox_perm` can make the NEOX kernel compute the interleaved rotation if the
rope rows of wq_b/wkv_a are permuted at load. This lane applies the interleaved rotation
directly in `memra_mla_rope_interleaved_kernel` instead, so checkpoint bytes stay unmutated
and the fixture's CPU projection chain is compared against the same rotation it computes.
Both are proven equal for dot-product consumers (`rope_norm_equals_permuted_neox`), so the
load-time permutation remains available as an optimization.

Angle: each lane evaluates the closed form `pos * base^(-2j/d_rope)` independently; the oracle
walks a `theta *= theta_scale` recurrence. Equal to f32 rounding, covered by the rope gate —
asserted, not assumed. Accurate `sincosf`/`expf` are used, not the `__sincosf`/`__expf` SFU
intrinsics, whose error grows with the argument (rope theta == pos at j=0).

glm5_next is NoPE (`rope_head_dim` 0): the rope launcher returns early rather than issuing a
zero-extent grid, and the branch lives in the Rust wrapper so no empty slice is dereferenced.

## Sparse-index (DSA / kpool) seam

Dense core only. Every cache walk takes its horizon from `visible`, computed once per
(query, head) block. A sparse arm replaces that contiguous `0..visible` walk with a gathered
position list (`const int* idx, int n_idx`); the score / softmax / accumulate body is
unchanged, because nothing assumes cache rows are adjacent except the loads themselves.
No selection logic, no top-k, no indexer cache exists here.

## Wired paths vs named stops

Wired (mirroring the KDA lane's scoping in the same checkout): the stateless `forward` /
`forward_last`, the stateful `prime_layers`, and the eager T=1 decode.

Named stops via `mla_path_unimplemented(path)`: core-split prime (it hands back a pre-wo
activation plus the out-GEMM weight for the caller to fuse; the MLA arm owns its own wo GEMM),
captured-graph prime, batched cache prime, norm-fused decode, device-counter decode, captured
device-counter decode, decode_step_h/chain/lockstep variants, MTP head forward, speculative
verify, TP attention, batched PP decode. Each carries state and dispatch discipline no MLA
parity gate covers; DESIGN.md puts graph capture, the batched tick and the spec/MTP route in
increment 7.

Decode is wired at THREE sites, matching where the KDA lane wired its own: the unfused
`attn_in_norm_mixer` fallback, and the two `add_rms_norm` (fuse_add_norm) arms in
`decode_step_h` and `decode_step_chain`. The q8_1-fused twins of those arms stay named stops.

`mixer_in_q8_1_fast` stays `false` for MLA deliberately — the first GEMM is wq_a/wkv_a off an
f32 hidden and no gate covers a q8_1 activation path, so every MLA decode routes through the
unfused arm, which is the gated one.

## Open risks (carried into later increments)

1. **Quantized real weights are a hard prerequisite for serving.** `mla_split_operand` refuses
   anything but the Float arm for the 3D `attn_k_b`/`attn_v_b`. Real GLM-5.2 / GLM-5.3-Flash
   checkpoints are quantized, and the generic 2D Quant path mis-derives `row_bytes` from
   `ne[1]` for a 3D tensor. The micro fixture rides F32, so NO current gate covers this. This
   is the largest gap between "gates green" and "the model serves": the loader must split those
   tensors per head (or flatten `ne[1]*ne[2]`) before the absorb/decompress kernels can consume
   a real checkpoint.
2. **`CacheSnapshot` / rewind does not carry latent lengths.** A spec-decode rollback on an MLA
   model would leave the latent plane's `len`/`len_d` desynced from the restored KV lens.
   Unreachable today only because every speculative path is a named stop; it must land with
   increment 7 (spec/MTP), not after it.
3. Perf shape: the core is warp-dot / scalar, not tensor-core, and the static shared arrays are
   sized at `MLA_MAX_RANK` regardless of actual `kv_rank`, which caps occupancy. DESIGN.md
   increment 5 (fused GEMM-shaped decode) is where that is addressed. No timing number was
   taken here — rig law forbids it on this box.
4. Sparse index (DSA / kpool) is absent; the dense core is its own oracle for the T <= top_k
   bit-identity gate DESIGN.md increment 6 specifies.
5. The latent plane is f32; a real long-context deployment needs the quantized row (§ Cache
   layout) or the cache footprint advantage MLA exists for is partly given back.
