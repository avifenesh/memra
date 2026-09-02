# mHC sinkhorn chain fusion — decode, GLM-5.3-Flash NVFP4, 2x B200

Branch `lane/b200-sinkhorn-fusion-20260902`, off `lane/glm5-b200-20260902`. Worktree
`/home/avifenesh/projects/wt-b200-sinkhorn`.

## The ask

Fuse the per-site mHC (manifold-constrained hyper-connections) decode chain —
`dsv4_rowsq_scale_kernel -> dsv4_hc_sinkhorn_m_kernel -> dsv4_hc_collapse_kernel ->
dsv4_hc_post_kernel` — into ONE kernel launch per site, values staying in
registers/shared memory the whole way, bit-identical to the unfused four-kernel chain,
behind a new door `MEMRA_HC_FUSED_CHAIN` (default OFF).

## The measurement that opened this lane

nsys, 2x B200 SXM (sm_100a), GLM-5.3-Flash NVFP4, resident PP2, plain decode, ~224
tokens, both devices summed, per token:

| kernel | launches | us/launch | ms/token | share |
|---|---|---|---|---|
| `dsv4_hc_sinkhorn_m_kernel` | 130 | 20.3 | 2.64 | 8.8% of GPU time |
| `dsv4_rowsq_scale_kernel` | 130 | 4.8 | 0.62 | |
| `dsv4_hc_post_kernel` | 130 | 2.4 | 0.31 | |
| `dsv4_hc_collapse_kernel` | 130 | 1.8 | 0.23 | |
| `l2_norm_f32` | 98 | 2.0 | 0.20 | (not part of this chain — see Scope note below) |

Plus ~2.2 us launch gap per kernel. The launch-diet lane
(`research/glm53-flash-bringup-20260827/launch-diet-20260830/LANE.md`, item 2)
had already named "mHC pre-chain fusion (rowsq+sinkhorn+collapse, one launch/site): 3.3
ms/token of GPU time … sinkhorn 18.3 us at t=1 (20 serial iterations)" as the top
remaining mHC increment on an earlier box's census. This lane's job was to go one
kernel further and also fold in `dsv4_hc_post_kernel`.

## Finding: the four-kernel single-launch fusion is not shippable

`dsv4_hc_post_kernel` cannot join a single launch with `rowsq_scale` + `sinkhorn` +
`collapse` for the same site, because its `f` operand — the site's attention or FFN
branch output — does not exist yet when collapse finishes. The mHC per-site program
(`crates/memra-engine/src/hyper.rs:10-20`, module doc, cited verbatim):

```text
mixes[t, :]   = fn_w · x[t, :, :]                     (rows = (2+streams)*streams)
mixes[t, :]  *= rsqrt(mean(x[t]^2) + eps)             (over the whole streams*hidden slab)
pre/post/comb = sinkhorn(mixes[t, :], scale, base)    (per token, per site)
y[t, :]       = Σ_c pre[t, c] · x[t, c, :]            (collapse streams -> 1)
f             = branch(rms_norm(y))                   (the mixer or the FFN, unchanged)
x'[t, k, :]   = post[t, k] · f[t, :] + Σ_j comb[t, j, k] · x[t, j, :]
```

`branch` is the layer's actual attention core (QKV projection, RoPE, KV-cache read,
flash/MLA/KDA attention) or its FFN/MoE core (routing, expert GEMMs, SwiGLU) —
whichever site this is. That is a full sub-layer of GPU work, not a few extra
instructions, and it runs as its own multi-kernel program strictly between the
`collapse` write of `y` and the `hc_post` read of `f`. Confirmed at both call sites in
this checkout that this lane's kernels are drawn from:

1. **glm5_next persistent T=1 decode walk** (`crates/memra-engine/src/hybrid_forward.rs`,
   `hyper_range_decode_ws_body`, lines 2271-2300 — comment at 2247-2249 states the
   invariant directly: *"same kernels, same order (pre -> rms_norm -> mixer -> post ->
   pre -> rms_norm -> ffn -> post)"*):

   ```text
   pre_t1_ws(attn site)   # rowsq_scale + sinkhorn_m + collapse (or MEMRA_HC_FUSED_PRE, 1 launch)
   rms_norm
   mixer                  # full_attn / linear_attn / mla / kda — many kernel launches
   post_t1_ws(attn site)  # hc_post
   pre_t1_ws(mlp site)    # starts with the mixes GEMM (linear_t1_into), THEN rowsq/sinkhorn/collapse
   rms_norm
   ffn                    # dense or MoE FFN branch — many kernel launches
   post_t1_ws(mlp site)   # hc_post
   ```

2. **dsv4-native verify-batch path** (`crates/memra-engine/src/dsv4_gpu.rs`, the function
   that calls `hc_pre_batch_dev` for the FFN site, lines 9252-9356): the attention core's
   own o-projection GEMM (`gemm_m_dev`) runs, THEN `hc_post` (attn, line 9294), THEN
   `hc_pre_batch_dev` for the FFN site (mixes GEMM via `dots_m_dev` + rowsq_scale +
   sinkhorn_m + collapse, lines 8358-8436), THEN `rmsnorm`, THEN the FULL MoE branch
   (`moe_verify_dev`, routing + expert GEMMs), THEN `hc_post` (ffn, line 9344).

In both paths every occurrence of `hc_post` has a full attention or FFN sub-layer on one
side and — at the very next site's pre-chain — a mixes GEMM on the other. Neither
neighbor is something this lane can pull into `dsv4_hc_post_kernel`'s launch without also
inlining the attention or FFN math:

- Inlining the attention/FFN branch is a different, much larger kernel (a full mixer or
  MoE core) than "mHC glue fusion," is owned by other lanes (attention, MoE, matvec/GEMM
  kernels — see "Do not touch cu/qmatvec.cu" in this lane's brief and the do-not-touch
  list in `launch-diet-20260830/LANE.md`: "The mixes GEMM stays `Engine::linear` under
  the pre_exact law … untouched"), and is out of scope for a kernel-fusion lane scoped to
  the four named mHC glue kernels.
- Even the smaller neighbor — the mixes GEMM between a site's `hc_post` and the NEXT
  site's `rowsq_scale` — is explicitly a do-not-touch: `hyper.rs`'s own doc says the
  mixes GEMM rides `Engine::linear`/`linear_t1_into` (cuBLASLt f32) specifically because
  the dsv4 f64 "dots" island kernel would be "the wrong shape" and a different reduction
  program for this decode-exact contract. Replacing it with a hand-written epilogue-fused
  GEMV to bridge `hc_post` and `rowsq_scale` would trade a proven decode-exact reduction
  for an unproven one — exactly the "different numeric class" this lane's brief said to
  stop and document rather than ship.

So: no ordering of these four kernels is ever launch-adjacent without either (a) a full
attention/FFN sub-layer or (b) a GEMM sitting between them. **Per the lane's own
instruction — "if any step's reduction order cannot be kept, stop and document why rather
than shipping a different numeric class" — this lane stops here rather than shipping a
kernel that either fakes the fusion (silently dropping/reordering the intervening branch
compute, which would be a correctness bug, not a kernel fusion) or reaches into another
lane's kernels to make the launch adjacency real.**

No new door was added. A `MEMRA_HC_FUSED_CHAIN` flag that could not do more than the
existing `MEMRA_HC_FUSED_PRE` flag already does would be redundant naming, which the
flags doctrine (`CLAUDE.md` "Flags doctrine") argues against directly: "When an
experiment concludes negative or flat, kill its flag and dispatch arm." Shipping a
same-behavior flag under a different name is the mirror image of that same waste.

## What's actually fusable, and its status

The three-kernel pre-chain `rowsq_scale -> hc_sinkhorn_m -> hc_collapse` (stages 1-3 of
`hyper.rs::pre_finish_into`, `crates/memra-engine/cu/dsv4_gpu.cu:975-1085,2963-3011`) IS
launch-adjacent with no intervening compute, and this fusion was already shipped in
`lane/glm5-decode-diet` (2026-08-31), unmodified by this lane:

- Kernel: `dsv4_hc_pre_fused_kernel` (`cu/dsv4_gpu.cu:3080-3216`) — verbatim per-stage
  bodies (rowsq's 8-wide f64 reduction at the pinned blockDim=128; Sinkhorn on shared
  operands with a bit-preserving stationarity exit; collapse's per-element expression),
  proven bit-identical to the unfused chain by construction and asserted bytewise in
  `crates/memra-engine/tests/hc_fused_pre_gpu.rs`.
- Door: `MEMRA_HC_FUSED_PRE` (default OFF), read per call in
  `hyper.rs::pre_finish_into` — wired into BOTH decode paths this lane inspected
  (`pre_t1_ws` for the T=1 persistent walk, and `pre`/`pre_exact` for the allocating and
  batched-verify walks, all sharing `pre_finish_into`).
- Status per `docs/FLAGS.md`: bit-identity gate green on the rig (5090, exactness-only);
  **default OFF specifically because no box throughput receipt existed** ("NOT
  default-ON: no throughput receipt (the rig is exactness-only; the ~2.5-3 ms/token
  expected value is census arithmetic, WINDOW-20260830.md §3.2)").

Against this lane's own fresh B200 census, that existing fusion already collapses:

```
sinkhorn 2.64 ms + rowsq 0.62 ms + collapse 0.23 ms = 3.49 ms of the 3.77 ms
four-kernel total = 92.6% of the named chain's GPU time, at 3 launches -> 1 per site
(130 -> ~43-44 sites' worth of the fusable three, net -260 launches/token on this
sub-chain alone).
```

`hc_post` (0.31 ms, 7.4% of the four-kernel total, 130 launches unchanged) is the part
this lane could not move, for the structural reason above.

## Gate: `hc-fused-gate`

`crates/memra-engine/src/bin/hc_fused_gate.rs` (new bin, `hc-fused-gate`). Scoped
honestly to what's real: it does NOT test a `MEMRA_HC_FUSED_CHAIN` door (none exists). It
re-proves `dsv4_hc_pre_fused_kernel`'s bit-identity at the pinned GLM-5.3-Flash shape
(streams=4, n_embd=4096, sinkhorn_iterations=20, eps=1e-6 — read from the checkpoint
config at `model_plan.rs:2119` (`g5.hc_sinkhorn_iters`); the same constants are pinned in
the glm5_next test fixture at `hf_mapping.rs:1201-1204` and in `hc_fused_pre_gpu.rs`'s
`ITERS`/`EPS` constants, `n_embd` per this lane's task brief) at t in {1, 4, 8} (decode t=1 plus small-t
MTP-verify shapes, K+1 <= 8 within the `hyper_batch_cap()=15` bound named in
`docs/FLAGS.md`'s `MEMRA_GLM5_SPEC` row), and adds the box throughput receipt that
row says is missing: N=5 device-synchronized timings per arm (unfused 3-launch chain,
fused 1-launch kernel, and `hc_post` alone for census-completeness, explicitly labeled as
NOT part of any fusion).

Bit-identity check: `to_bits()` equality on all four outputs (`pre`, `post`, `comb`, `y`)
between the unfused chain and `dsv4_hc_pre_fused_kernel`, same operand bytes, per shape.

Run (box, under the fleet lock):

```
MEMRA_GPU_LOCK=/tmp/memra-gpu.lock flock /tmp/memra-gpu.lock \
  cargo run --release -p memra-engine --bin hc-fused-gate -- 0
```

(Pass a device index as the one CLI arg; defaults to device 0.) Exit 0 = PASS at every
tested shape. Timings print as a table plus one `HC_FUSED_GATE_JSON [...]` line for
machine parsing.

**Receipt pending**: this worktree has no GPU (see task brief — the B200 box belongs to
the session that spawned this lane). The gate has not been executed here; it compiles
clean under both `MEMRA_CUDA_ARCH=100a` and `MEMRA_CUDA_ARCH=120a`. The parent session
runs it on the box and banks the output under this directory (e.g.
`hc-fused-gate-b200-<date>.log`).

## Docs touched

- `docs/KERNELS.md`: added a `cu/dsv4_gpu.cu` section (previously absent from this file
  despite the file existing and being compiled) covering the mHC glue family this lane
  investigated — `rowsq_scale`, `hc_sinkhorn`/`_m`, `hc_collapse`, `hc_pre_fused`,
  `hc_post`, `hc_head_pre`/`_m` — with line numbers and the same finding as above on
  `hc_post`'s non-fusability. Explicitly scoped as a partial inventory of a ~140-symbol
  file, not a claim of full coverage.
- `docs/FLAGS.md`: **not changed.** No new flag was added (see Finding above); the
  existing `MEMRA_HC_FUSED_PRE` row already documents the fusable subset accurately and
  this lane did not modify its kernel, door, or dispatch site.

## Launch count, per site, before/after

| arm | launches/site | of the 4 named kernels |
|---|---|---|
| today (both doors OFF) | 4 (`rowsq_scale`, `hc_sinkhorn_m`, `hc_collapse`, `hc_post`) | baseline |
| `MEMRA_HC_FUSED_PRE=1` (existing, unmodified by this lane) | 2 (`hc_pre_fused`, `hc_post`) | -50% launches, -92.6% of this chain's GPU time per the B200 census above |
| requested `MEMRA_HC_FUSED_CHAIN` (all 4 in one launch) | not achievable | — |

## Open items (for the parent session / flag owner, not this lane)

1. **Run `hc-fused-gate` on the B200 box** under `MEMRA_GPU_LOCK=/tmp/memra-gpu.lock`
   and bank the output here. This is the missing evidence named in `MEMRA_HC_FUSED_PRE`'s
   FLAGS.md row ("NOT default-ON: no throughput receipt").
2. **Flip decision for `MEMRA_HC_FUSED_PRE`** is not this lane's call (per its own FLAGS.md
   row: "Flip condition: interleaved x3 fresh-boot A/B on the serving card class … greedy +
   vendor-default sampled twin, engagement announce in BOTH arms" — a serving-shape
   decision, owned by whoever runs that battery). This lane only supplies the per-kernel
   timing piece of that evidence, at the pinned production shape, on B200.
3. **`hc_post` reduction stays open as a genuinely different, larger increment**: the only
   way to shrink its launch count further is to fuse it with the attention or FFN branch
   it brackets (an epilogue on the o-projection GEMM on one side, or the FFN/MoE combine
   on the other) — that is GEMM/attention/MoE-kernel work, a different lane's authority,
   and a different (larger, riskier) numeric-class question than this lane was scoped to
   answer. Not attempted here; named for whoever owns those kernels next.
4. **dsv4-native `hc_pre_batch_dev` has no fused-door option at all** (`dsv4_gpu.rs:8338-8437`
   always runs the three-kernel unfused chain, unlike `hyper.rs::pre_finish_into` which
   checks `MEMRA_HC_FUSED_PRE`). Out of scope for GLM-5.3-Flash (which routes through
   `hyper.rs`, not this dsv4-native path) but worth a note for whoever next touches the
   native dsv4 (DeepSeek-V4-Flash) decode/verify path.

## Session

Claude-Session: https://claude.ai/code/session_01YAY8kDvonKLAS5HumdBbWP
