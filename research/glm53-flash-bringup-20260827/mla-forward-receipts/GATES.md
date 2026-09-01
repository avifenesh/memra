# MLA CUDA forward — increment 4 gate receipts

Box: local 5090 laptop, LOCK-SERIALIZED correctness only (rig law: exactness gates OK,
timing numbers never). All runs under `flock /tmp/memra-5090.lock`.

Oracle: `crates/memra-engine/src/mla.rs` (`mla_attend_naive` == `mla_attend_absorbed`).

## 0. Oracle pinned at NoPE FIRST

`mla.rs` had never been exercised at `d_rope == 0`, so it could not be trusted as the GPU
arm's truth for the glm5_next door. Pinned before any kernel was written:

    cargo test -p memra-engine --lib mla::
    test mla::tests::naive_equals_absorbed_nope_rope_zero ... ok
    test mla::tests::nope_scale_is_qk_head_dim ... ok
    (6 passed)

`naive_equals_absorbed_nope_rope_zero` covers shrunk NoPE shapes (decode / pure prefill /
chunked, 3 seeds each) plus full `MlaDims::GLM5_NEXT` (64 heads, nope 256, rope 0, v 256,
rank 512). `nope_scale_is_qk_head_dim` pins that the NoPE softmax scale is 1/sqrt(d_nope)
== 1/16 — the same 1/16 GLM-5.2 reaches as 1/sqrt(192+64), NOT 1/sqrt(kv_rank).

## 1. BEFORE — the panic, through an existing entry point

`HybridModel::forward` on the glm-dsa micro fixture, at lane head d1bdd58863:

    thread 'gpu_glm_dsa_micro_block_forward_runs' panicked at
      crates/memra-engine/src/hybrid_forward.rs:1040:34:
    Mixer::Mla has no forward arm yet — glm-dsa is loader-only in increment 2;
    the CUDA forward lands in increment 4 (research/mla-bringup-20260801/DESIGN.md §4)

    test result: FAILED. 0 passed; 1 failed

## 2. AFTER — five gates green (final run, verbatim)

    flock /tmp/memra-5090.lock cargo test -p memra-engine --test mla_gpu_forward \
      -- --ignored --test-threads=1

    test gpu_glm_dsa_micro_block_forward_runs ................... ok
    test gpu_mla_cached_prime_decode_matches_stateless_forward .. ok
    test gpu_mla_decode_stepwise_parity_vs_cpu_oracle ........... ok
    test gpu_mla_prefill_parity_vs_cpu_oracle ................... ok
    test gpu_mla_rope_interleaved_matches_cpu ................... ok
    test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out;
      finished in 33.14s

    flock /tmp/memra-5090.lock cargo test -p memra-engine --test mla_fixture_load_gpu \
      -- --ignored
    test gpu_load_glm_dsa_micro_fixture ... ok   (the increment-2 loader gate, still green)

Bounds: 1e-5 relative at shrunk shapes, 1e-4 at the production geometries (GLM52 and
GLM5_NEXT) — the kernel's tiled online softmax reorders accumulation against the oracle's
single left-to-right pass, so parity is a maxdiff bound, never bit-identity.

Prefill gate covers t_q == t_kv, the chunked shape (queries a suffix of a populated cache),
and T=17 (spans several 8-timestep softmax tiles). Decode gate appends through the real
append kernel — prefill block then one row per step — and compares EVERY step against the
oracle's full-sequence recompute at that position; it also asserts the appended plane is
bit-equal to the host-built `[c_kv | k_pe]` rows, so an append-layout bug cannot hide as a
soft maxdiff downstream.

## 3. Mutation checks — the gates bind

A gate that cannot fail is not a gate. Two deliberate kernel mutations, each reverted:

| mutation | expected | observed |
|---|---|---|
| `visible = t_kv` (causal mask removed) | prefill parity fails | FAILED, maxdiff 1.055e0 vs tol 1e-5 (rel 1.055) |
| rope term dropped from the score | rope dims fail, NoPE dims unaffected | FAILED: prefill maxdiff 1.079e-1, decode step 0 maxdiff 6.952e-2, both at rope-64-ratio dims; `gpu_mla_rope_interleaved_matches_cpu` stayed ok |

Both mutations were reverted and the full suite re-run green — the verbatim output is §2.

Honest note on mutation 2: `gpu_glm_dsa_micro_block_forward_runs` PASSED under it. That gate
is a smoke gate — it asserts finite, non-degenerate logits, not parity — so it cannot catch a
wrong-but-plausible score. The parity gates are the ones that bind; the smoke gate exists to
prove the block runs end to end at all (it is the before-receipt's counterpart).

## 4. Surrounding suites (kept green)

    cargo test -p memra-engine --test mla_fixture_forward  -> 3 passed
    cargo test -p memra-engine --lib mla::                 -> 6 passed (re-run AFTER the kernels)
    cargo test -p memra-kv                                 -> 15 passed
    cargo check --workspace                                -> clean (only pre-existing warnings,
                                                              none in the MLA files)

`memra-kv` matters here because this lane moved a load-bearing assert: the KVQUANT
`head_dim % 32 == 0` check was function-wide in `Cache::new_inner` and is now scoped to the
quantized-plane arms. The latent plane is not head-shaped (the micro fixture's width 24 trips
the old form), so the assert had to become per-arm rather than per-model.

## 5. Formatting

`cargo fmt` is NOT used: this checkout is shared with the concurrent KDA lane, and a tree-wide
format would rewrite that lane's in-flight code. `rustfmt` was run on the MLA-owned files only;
every shared file this lane touched was verified rustfmt-clean without modification.

## 6. What gate 4 cost, and why its bar differs

Gate 4 (cached prime+decode vs stateless forward) failed three times before passing, and each
failure was worth having:

1. `prime_cache` asserts `T >= PRIME_MIN_T` (16); the harness primed 5 tokens. Harness bug.
2. The length assert looped over ALL latent planes, including the MTP block's — which the
   trunk prime never touches (block_count = n_trunk + nextn). The assert now checks trunk
   planes == primed length AND the MTP plane == 0, so the MTP invariant is pinned rather than
   worked around.
3. `decode_step_h` does not route through `attn_in_norm_mixer`; it has its own dispatch, and
   the eager path is the `add_rms_norm` (fuse_add_norm) arm. Only after wiring that — plus the
   same arm in `decode_step_chain` — did the decode actually run. Three decode sites are wired,
   matching where the concurrent KDA lane wired its own.

Then the numeric bar itself was wrong: the gate used the 1e-4 kernel-level bound and measured
maxdiff 7.315e-4 (rel 4.29e-4). The two arms are NOT the same floating-point computation —
stateless runs each projection as one T=20 GEMM, cached runs T=19 then T=1, and a cuBLAS GEMM's
reduction order depends on M; that drift enters at wq_a/wkv_a/wo and every FFN+MoE GEMM and
compounds over the stack. It is prefill-vs-decode divergence, not MLA error — the thing
`bin/t2probe` exists to measure. The gate now uses the glm_dsa pack's OWN declared
`CheckpointParityGate` (max_abs 0.005, max_rel 0.005, require_argmax), the house bar for a
whole-stack comparison, and asserts argmax equality on top. The MLA core stays held to 1e-4
against the CPU oracle by gates 1-2, including through the real append path.
