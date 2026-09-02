# b200-q8-fuse-20260902: fold `quantize_q8_1` into its producer on the glm5_next decode path

Owner order (2026-09-02): DECODE priority — "hardly improve the decode on these cards."
Branch `lane/b200-q8-fuse-20260902` off `lane/glm5-b200-20260902`, worktree
`/home/avifenesh/projects/wt-b200-q8fuse`. No GPU access to the 2x B200 SXM box (owned by the
spawning session) — all correctness gates below ran on the local RTX 5090 laptop rig, which is
exactness-only by repo law (`docs/*` / CLAUDE.md "Rig GPU exactness-only": correctness gates OK,
timing numbers never a performance claim).

## The measurement this lane answers

nsys, 2x B200 SXM, GLM-5.3-Flash NVFP4, resident PP2, plain decode, ~224 tokens: per token
~2,900 kernel launches at ~2.2 us each is a ~6 ms launch floor inside ~24 ms/token.
`quantize_q8_1` alone is 416 launches/token (2.0 us each, 93,232 total; cu/qmatvec.cu ~L585),
issued before every q8-activation matvec on the glm5 decode path.

## What was actually wrong, and what fixes it

The `quantize_q8_1` call sites are NOT one flat list of independent producer/consumer pairs —
most of the 416/token are buried inside `Engine::matmul`'s internal m==1 dispatch arm
(`crates/memra-engine/src/lib.rs`, the `quantize_q8_1` call right before `qmatvec_mmvq`), which
is **another lane's territory** (the matvec-dispatch lane working nearby, per the owner's
scope note) and is used by essentially every weight matmul in the model — touching it there
is out of scope and would be a much wider, riskier change than intended.

The scope that IS this lane's to fuse, and that is real: **explicit, visible
`quantize_q8_1(&z, ...)` calls inside `hybrid_forward.rs`/`kda.rs` that a caller issues against
a `z` some OTHER memra kernel just finished writing.** Auditing those (`rg -n
'quantize_q8_1\('`, ~40 call sites) against glm5_next's actual decode call graph
(`hyper_range_decode` / `hyper_range_decode_ws_body` -> `hyper_ffn_branch` ->
`moe_ffn_il_zq8` -> `moe_ffn_glm5_ep`/the sigmoid-router MoE arm) found:

1. The zq8 (pre-quantized `(zq, zd)` pair) plumbing ALREADY EXISTS through the whole MoE
   dispatch chain — `moe_ffn_il_zq8`, `moe_ffn_inner`, `moe_ffn_glm5_ep`, `moe_shexp_add`,
   `shexp_gate_up_t1` all already accept an `Option<&(CudaSlice<i8>, CudaSlice<f32>)>` and take
   the pre-quantized pair when it is `Some`, falling back to their own `quantize_q8_1(z, ...)`
   when it is `None`. `moe_ffn_il_zq8`'s own doc comment says as much: "Decode-path twin with a
   PRE-QUANTIZED z (from `add_rms_norm_zq8`): threads (zq, zd) into the t=1 dev arm so the
   per-layer standalone quantize_q8_1 launch folds away." This plumbing was built for a
   DIFFERENT call site (one that already had an `a+b` residual add to fuse into a norm).
2. glm5_next's own mHC (hyper-connection) decode walk never produces that pair. It computes
   the FFN-input norm with a plain `e.rms_norm(&y, layer.post_attn_norm.float_data(), &mut z,
   n_embd, 1, eps)` and passes `None` for zq8 into `hyper_ffn_branch`, so `moe_ffn_il_zq8` (and
   everything downstream of it) re-derives `(zq, zd)` from `z` with a standalone
   `quantize_q8_1` launch every layer, every token.
3. A dual-output fused kernel (rms_norm emitting BOTH f32 z — still consumed by the MoE router
   logits GEMV and an ungated shexp arm — AND its q8_1 quantization) did not exist for the
   PLAIN (non-residual) case; only the `a+b`-residual twin `add_rms_norm_zq8` did.

So this lane is exactly the "missing producer" the existing plumbing was built to accept: a new
kernel `rms_norm_zq8_f32` (`crates/memra-engine/cu/kernels.cu`, next to `rms_norm_q8_1`) plus
wiring the two glm5_next T=1 mHC decode call sites to call it and pass the result through.

## Kernel identity argument

`rms_norm_zq8_f32(x, w, ncols, eps)`:
- Pass 1 (sum of squares -> `scale = rsqrt(sum/ncols + eps)`): copied VERBATIM from
  `rms_norm_f32`/`rms_norm_q8_1`'s reduction (same per-thread stride, same `__shfl_down_sync`
  block-reduce tree, same shared `s[32]`).
- Pass 2 (`z[i] = (x[i]*scale)*w[i]`, then per-32-block `amax` -> `d = amax/127` ->
  `id = 1/d` -> `q[i] = round(z[i]*id)`): the epilogue is `add_rms_norm_zq8`'s warp-per-block
  form, copied VERBATIM, MINUS that kernel's `a+b` residual add (this kernel reads `x` directly
  instead of a freshly-summed `r`). `add_rms_norm_zq8`'s own header already carries the
  bit-identity argument for that epilogue against `quantize_q8_1`'s reduction (order-independent
  `__shfl_xor_sync` max).

**A real bug the gate caught, and the fix**: the first draft hardcoded `block_dim = 1024` in
the Rust wrapper (matching `rms_norm_q8_1`/`add_rms_norm_zq8`'s convention). That is WRONG in
general — the actual call site's shipped chain runs `e.rms_norm(...)` at `rms_block()`
(`crates/memra-engine/src/lib.rs`), which defaults to **256**, not 1024 (1024 is a per-model
override only gemma4's loader sets — `RMS_BLOCK_DEFAULT.store(1024, ...)` in `hybrid.rs`, never
touched for glm5_next). A different blockDim means a different per-thread reduction stride and
a different `__shfl_down_sync` tree depth, i.e. a DIFFERENT (if usually close) `scale` value —
not bit-identical. `q8_fuse_gate` at `ncols=1536` (the candidate MoE expert-ff width) caught
this immediately: `z`/`d` byte mismatch (first z: chain=7.966741920e-1 vs
fused=7.966742516e-1 — a real ULP-class divergence, not a rounding artifact of the test). The
`ncols=4096` (glm5_next n_embd, the shape this lane actually wires) and `ncols=8192` shapes
happened to match at 1024 vs 256 for that random seed, which would have been a silent,
seed-dependent bit-identity hole if `ncols=1536` had not been in the gate's shape list — the
kernel body itself is already blockDim-generic (same structure as `rms_norm_f32`), so the fix
is `block_dim: (rms_block(), 1, 1)` in the Rust wrapper instead of a hardcoded constant, tracking
whatever blockDim `rms_norm` actually uses at the call site (256 by default, 1024 under a
gemma4-style loader override). Re-ran the gate after the fix: all three shapes byte-identical
(see receipts below). This is the reason the gate carries more than one `ncols` — a single-shape
gate would have shipped the bug.

## Sites fused

| Call site | File | Producer before | Consumer after | Door-gated |
|---|---|---|---|---|
| mHC FFN-input norm, plain T=1 decode | `hybrid_forward.rs`, `hyper_range_decode` | `e.rms_norm(y, post_attn_norm, ...)` (256-thread block, glm5 default) | `hyper_ffn_branch` -> `moe_ffn_il_zq8` -> `moe_ffn_glm5_ep`/sigmoid-router MoE (was: separate `quantize_q8_1(z, 1, n_embd)`) | yes |
| mHC FFN-input norm, persistent-workspace T=1 decode (`MEMRA_HC_DECODE_WS=1`) | `hybrid_forward.rs`, `hyper_range_decode_ws_body` | same, `ws.y` -> `ws.z` | same | yes |

`hyper_ffn_branch` gained one parameter, `zq8: Option<&(CudaSlice<i8>, CudaSlice<f32>)>`,
threaded straight to `moe_ffn_il_zq8` (which already had the parameter). Every OTHER caller of
`hyper_ffn_branch` (the two prefill walks, the batched-decode dense-only per-row walk) now
passes `None` explicitly — byte-unchanged, because `moe_ffn_il_zq8`'s `None` branch is the
exact unfused call it always made.

**Launches removed per token when the door is ON**: one `quantize_q8_1` launch per hc-decode
layer (glm5_next has 45 trunk layers) = up to 45 of the ~2,900 launches/token the nsys census
measured — an arithmetic claim from the dispatch change, not a re-measured census (no B200
access this lane; see Open items).

**Not fused, and why** (explicitly out of scope per the task's exclusion list and this lane's
own scope):
- The attn-norm producer (`e.rms_norm(&y, layer.attn_norm.float_data(), &mut h, ...)`) feeding
  the KDA mixer — the KDA mixer's own internal q8 quantizes (`kda_proj_fused6`,
  `qmatvec_view`/`matmul`'s internal dispatch) are a separate, denser chain the KDA/matvec
  lanes own; `cu/qmatvec.cu` and the matvec dispatch in `lib.rs::matmul` were explicitly
  off-limits for this lane.
- KDA's own gated-norm (`kda_gated_rmsnorm`, sigmoid-gated, cu/kda.cu) feeding the `wo`
  out-projection — this chain's consumer is `e.matmul(&la.wo, &gated, t)`, which routes through
  the SAME off-limits `lib.rs::matmul` internal quantize; fusing the producer here would still
  need a way to skip that internal call without touching `matmul` itself, which this lane did
  not build (see Open items).
- Verify rows (t=2..15) — `moe_ffn_il_zq8_vrows` and the batched verify walk have their own
  zq8-shaped plumbing but this lane wired only the two T=1 mHC decode entry points named above.
- The MoE activation (`silu(gate)*up`) -> q8_1 site already has fused kernels wired in most
  arms (`mmq_iq_fused_act_quant` behind `MEMRA_MOE_FUSE_ACTQ`, default ON) — audited, already
  fused, nothing to do there.

## Gate

`cargo run -p memra-engine --bin q8_fuse_gate -- 5` (no model checkpoint needed — pure
synthetic kernel-shape gate). Compares the shipped two-launch chain (`rms_norm` then
`quantize_q8_1`) against the fused `rms_norm_zq8_f32` on three shapes: `ncols=4096` (glm5_next
n_embd, the site this lane wires), `ncols=8192` (glm5_next KDA qkv width, 64 heads * 128 —
candidate width for a future fusion, not wired here), `ncols=1536` (glm5_next MoE
`expert_ff_length` — same). Asserts byte-identical `z` (f32), `q` (int8), `d` (f32 per-32
scale); prints N=5 per-launch us per arm (rig numbers, not a performance claim — see the
top-of-file note).

### Receipts (RTX 5090 laptop rig, `flock /tmp/memra-5090.lock`, `sm_120a`, 2026-09-02)

Before the `rms_block()` fix (bug the gate caught):
```
[q8-fuse-gate] glm5_next-n_embd ncols=4096 z_bytes_match=true q_bytes_match=true d_bytes_match=true -> PASS
[q8-fuse-gate] glm5_next-kda_qkv ncols=8192 z_bytes_match=true q_bytes_match=true d_bytes_match=true -> PASS
[q8-fuse-gate] glm5_next-moe_ff_exp ncols=1536 z_bytes_match=false q_bytes_match=true d_bytes_match=false -> FAIL
  first z mismatch at i=0: chain=7.966741920e-1 fused=7.966742516e-1
  first d mismatch at i=0: chain=4.558592290e-2 fused=4.558592662e-2
```

After the fix (`block_dim: (rms_block(), 1, 1)` instead of a hardcoded 1024):
```
[q8-fuse-gate] glm5_next-n_embd ncols=4096 z_bytes_match=true q_bytes_match=true d_bytes_match=true -> PASS
[q8-fuse-gate] glm5_next-n_embd ncols=4096 N=5 chain_us=[17, 10, 9, 8, 8] fused_us=[15, 13, 13, 14, 13]
[q8-fuse-gate] glm5_next-kda_qkv ncols=8192 z_bytes_match=true q_bytes_match=true d_bytes_match=true -> PASS
[q8-fuse-gate] glm5_next-kda_qkv ncols=8192 N=5 chain_us=[10, 10, 9, 11, 9] fused_us=[20, 21, 20, 20, 20]
[q8-fuse-gate] glm5_next-moe_ff_exp ncols=1536 z_bytes_match=true q_bytes_match=true d_bytes_match=true -> PASS
[q8-fuse-gate] glm5_next-moe_ff_exp ncols=1536 N=5 chain_us=[10, 9, 9, 10, 10] fused_us=[10, 10, 9, 10, 10]
[q8-fuse-gate] ALL SHAPES PASS (byte-identical fused vs chain)
```
The rig's per-launch us are NOT a performance claim (5090 laptop throttles; correctness gates
only, per repo law) — they say nothing about which arm is faster on a real card, only that both
arms complete and produce identical bytes.

### Whole-model regression receipts (same rig/lock, `sm_120a`)

`hyper_connections_gpu` (6 tests, exercises `hyper_range_decode` end-to-end against a full
recompute on a mini glm5_next hc fixture) and `hc_decode_ws_gpu` (2 tests, exercises
`hyper_range_decode_ws_body`, WS-on-vs-off byte identity), run both with the door unset
(default OFF, the pre-existing baseline) and with `MEMRA_GLM5_Q8_FUSE=1`:

```
MEMRA_GLM5_Q8_FUSE unset:  hyper_connections_gpu 6 passed; hc_decode_ws_gpu 2 passed
MEMRA_GLM5_Q8_FUSE=1:      hyper_connections_gpu 6 passed; hc_decode_ws_gpu 2 passed
MEMRA_GLM5_Q8_FUSE=1 + MEMRA_HC_DECODE_WS=1: hc_decode_ws_gpu 2 passed
```
`hyper_prime_then_decode_matches_a_full_recompute` and
`hyper_two_chunk_prime_then_decode_matches_a_full_recompute` both compare the T=1 decode step's
logits against an independent full recompute — with the door ON these still match exactly,
which is a real-model (mini fixture) byte-identity receipt on top of the synthetic gate above.

## Build/lint receipts

- `MEMRA_CUDA_ARCH=100a cargo build -p memra-engine` — clean.
- `MEMRA_CUDA_ARCH=120a cargo build -p memra-engine` — clean.
- `cargo fmt --all -- --check` — clean.
- `tools/check-flags.sh` — `MEMRA_GLM5_Q8_FUSE` resolves against `docs/FLAGS.md`, 0 uncovered.
- `cargo test -p memra-engine --lib --no-run` — clean.

## Open items (not this lane, named for the next one)

1. **No B200 receipt.** This lane has no access to the 2x B200 SXM box (owned by the spawning
   session). The door is default OFF pending an interleaved x5 fresh-boot A/B there (real
   glm5_next NVFP4 artifact, plain decode + vendor-default sampled twin per the "never serve
   greedy" law) and an nsys re-census confirming the launch-count drop this LANE.md claims
   arithmetically.
2. **KDA's own gated-norm -> `wo` matvec chain** (`kda_gated_rmsnorm` in `cu/kda.cu`) is still
   two launches (norm, then `matmul`'s internal quantize). Fusing it needs either a bypass of
   `lib.rs::matmul`'s internal dispatch at the `wo` call site (i.e., a producer that emits q8_1
   directly and a caller that skips `matmul` for the `wo` projection specifically) or a new
   `kda_gated_rmsnorm_q8_1` kernel in `cu/kda.cu` plus that bypass — deliberately not attempted
   here because `matmul`'s internal dispatch is another lane's territory and a half-built bypass
   is worse than leaving the chain alone.
3. **Attn-norm -> KDA stage-1 projections.** Same shape of problem, same reason not touched.
4. **Verify rows (t=2..15).** `moe_ffn_il_zq8_vrows` accepts the same zq8 shape but this lane's
   two wired call sites are T=1 only.
