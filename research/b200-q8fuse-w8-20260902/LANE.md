# b200-q8fuse-w8-20260902: extend the q8_1 activation-quantize fuse to the W8 mirror producers

Follow-up to `research/b200-q8-fuse-20260902/LANE.md` (PR #84, lane/b200-q8-fuse-20260902).
Branch `lane/b200-q8fuse-w8-20260902`, based on `origin/lane/glm5-b200-int2-20260902`
(carries PR #86's `MEMRA_GLM5_W8` and everything else on the int2 train). Worktree
`/home/avifenesh/projects/wt-b200-q8fuse-w8`. No GPU access to the 2x B200 SXM box (owned
by the spawning session) — all correctness gates below ran on the local RTX 5090 laptop rig,
exactness-only by repo law.

## The measurement this lane answers

Serving posture moved to `MEMRA_GLM5_W8=1` (PR #86, the q8_0 mirror of the bf16 KDA/MLA
trunk, +10% plain). The W8-posture nsys census (int9 fc4b71ef7, doors on incl.
`MEMRA_GLM5_Q8_FUSE=1` from the prior lane, ~224 tokens) shows `quantize_q8_1` at 118,084
launches x 2.0 us = ~527 launches/token (up from ~70,720 launches total, ~316/token, on the
bf16 posture with the mHC-norm door alone; ~93,232 total, ~416/token, with every door off).
The W8 path quantizes its activations to q8_1 through standalone launches the prior lane's
door does not reach: the q8_0 trunk matvecs (`qmatvec_q8_0_mmvq_rp`-class kernels, 211
launches/token) each consume a q8_1 activation the W8 mirror arm produces per projection.

## What was actually happening

`MEMRA_GLM5_W8`'s mechanism (`matvec_bf16_via_q8_mirror` / `matvec_bf16_via_q8_mirror_t`,
`lib.rs`) builds a q8_0 mirror of a bf16-resident weight on first use, then quantizes the
ACTIVATION fresh, every call, keyed only by `in_f` (not by which `x` was passed) — so N
sibling weight matrices that all read the SAME input vector each independently re-quantize
that identical vector. Two such groups exist in glm5_next's decode trunk:

- **KDA's wq/wk/wv** (`kda_core_gated`'s stage-1 group, `kda.rs`) all read the layer's
  attn-normed `x`. Under `MEMRA_GLM5_W8=1` + `MEMRA_BF16_MMV=1` these three are the ONLY
  bf16-resident members of the six-projection group (`f_a`/`g_a`/`b_proj` are plain f32
  `Float` weights with no W8 mirror and nothing to fold) — so the real win here is 3
  quantizes down to 1, not 6 down to 1.
- **MLA's wq_a/wkv_a** (`mla_attn_core_pre_wo`, `hybrid_forward.rs`) both read the layer's
  attn-normed `h`. `wq_b`/`wk_b`/`wv_b`/`wo` consume DIFFERENT downstream vectors (`q_an`,
  the post-softmax attention output) and are out of scope for this same-input trick.

The existing `MEMRA_KDA_FUSED_PROJ` door's bf16 arm (`kda_proj_fused6`) already collapses
the six-projection group into one launch, but its bit-identity claim is against the
UNMIRRORED bf16 program (`matvec_bf16_f32acc_x4_rows`) — it explicitly declines whenever
`glm5_w8_on()` is true (pre-existing code, unrelated to this lane), because fusing against
the wrong numeric class would be a silent correctness bug, not a launch-count win. So under
`MEMRA_GLM5_W8=1` the six-projection group falls through to the per-weight `matmul_group`/
`matmul_rows_exact` dispatch, and each of wq/wk/wv independently re-quantizes `x`.

## What this lane adds

Two new `_pre` twins in `lib.rs` — `matvec_bf16_via_q8_mirror_pre` (t==1) and
`matvec_bf16_via_q8_mirror_t_pre` (t in 2..=32) — that are VERBATIM copies of their
non-`_pre` originals MINUS the internal `quantize_q8_1_into` block: the mirror-build path
(lock, `contains_key`, `encode_q8_0_from_bf16`, `build_q8_rp4_raw`) is untouched, and the
caller's own `(aq, ad)` is used directly. BIT-IDENTICAL by construction: `quantize_q8_1` is
a pure, deterministic function of `(x, in_f)`, so computing it once and reusing the bytes
across three calls produces the SAME bytes three separate calls would have produced.

Two new call-site helpers wire these into the real dispatch, gated on the EXISTING door
`MEMRA_GLM5_Q8_FUSE` (extended, not a new flag) plus `MEMRA_GLM5_W8` and `MEMRA_BF16_MMV`:

- `kda::Engine::kda_proj_qkv_qmirror3(la, x, t, rows_exact) -> Option<[CudaSlice<f32>; 3]>`
  — quantizes `x` once, calls the `_pre` twins for wq/wk/wv, wired into
  `kda_core_gated`'s stage-1 dispatch (both the t==1 decode arm via `matmul_group` and the
  t in 2..=32 verify-rows arm via `matmul_rows_exact`). `f_a`/`g_a`/`b_proj` keep their
  existing per-weight dispatch unchanged either way.
- `HybridModel::mla_proj_qa_kv_qmirror2(e, mla, h, t, rows_exact) ->
  Option<(CudaSlice<f32>, CudaSlice<f32>)>` — the same pattern for MLA's wq_a/wkv_a, wired
  into `mla_attn_core_pre_wo` ahead of the existing `mm(&mla.wq_a, h)` / `mm(&mla.wkv_a,
  h)` calls (which remain the fallback on any shape the fused path declines).

Both return `None` (falling through to the prior unfused per-weight dispatch, byte-unchanged)
on any shape either underlying W8 mirror arm would itself refuse — every precondition is
copied from the mirror arms' own guards in `lib.rs`, so a shape this code accepts is a shape
the unfused arm would have engaged on anyway.

Three distinct engagement counters (the prior lane's `GLM5_Q8_FUSE_DISPATCHES` for the mHC
norm site is untouched):

- `GLM5_W8_QFUSE_KDA_DISPATCHES` — `[glm5-q8-fuse] engaged W8-mirror KDA qkv ...`
- `GLM5_W8_QFUSE_MLA_DISPATCHES` — `[glm5-q8-fuse] engaged W8-mirror MLA wq_a/wkv_a ...`

each announced once per process, so a box census can attribute launches removed to the
right producer instead of inferring engagement from a green diff.

## Scope: what is fused and what is not

| Site | in this lane | why / why not |
|---|---|---|
| KDA wq/wk/wv, t==1 (decode) | YES | `kda_proj_qkv_qmirror3(rows_exact=false)`, live-tested |
| KDA wq/wk/wv, t in 2..=32 (verify-rows) | YES | `kda_proj_qkv_qmirror3(rows_exact=true)`, gate-tested only (no live verify-rows fixture) |
| MLA wq_a/wkv_a, t==1 and t in 2..=32 | YES | `mla_proj_qa_kv_qmirror2`, gate-tested only (no bf16-resident MLA fixture exists) |
| KDA wq/wk/wv, prefill t>1 non-rows-exact | NO | `matmul_group`'s per-weight dispatch reaches `matvec_bf16_via_q8_mirror_t` through a different call shape this lane did not thread a shared quantize through; named follow-up |
| KDA's `wo`, f_a/g_a/b_proj | NO | `wo` consumes `gated`, a different vector, no sibling to share with; f_a/g_a/b_proj are plain f32 with no W8 mirror at all |
| MLA's wq_b/wk_b/wv_b/wo | NO | each consumes a distinct downstream vector (`q_an`, attention output), no shared input to fold |
| The mHC-norm producer itself | NO (already covered by the PRIOR lane) | `hyper_range_decode`'s `e.rms_norm` producing `y`/`h`/`z` is untouched here; this lane only changes the W8-mirror CONSUMER side |

## Identity argument

`quantize_q8_1` is a pure function: same `x` bytes, same `in_f`, same kernel -> same output
bytes, every time, by construction (no RNG, no accumulated state). Computing it once and
reusing the `(aq, ad)` pair for N sibling matvecs is therefore trivially bit-identical to N
independent calls — there is no reduction-order or blockDim subtlety here (unlike the prior
lane's `rms_norm_zq8_f32`, which had to match `rms_norm`'s exact blockDim). The `_pre` twins'
mirror-build and matvec kernel dispatch are VERBATIM copies of the non-`_pre` originals.

## Gates

`cargo run -p memra-engine --bin q8_fuse_gate -- 5` — extended with
`run_w8_kda_shape(label, in_f, t)`: three synthetic bf16-resident weights (row-major
`[out_f, in_f]`, representative wq/wk/wv-shaped out widths 576/576/4096, all >= 64) compared
chain (`matmul`/`matmul_rows_exact`, which internally re-quantizes `x` under the W8 doors)
vs fused (quantize once, `matvec_bf16_via_q8_mirror[_t]_pre`). Doors `MEMRA_GLM5_Q8_FUSE`,
`MEMRA_GLM5_W8`, `MEMRA_BF16_MMV` forced on for the whole gate process.

### Receipts (RTX 5090 laptop rig, `flock /tmp/memra-5090.lock`, `sm_120a`, 2026-09-02)

```
[q8-fuse-gate] glm5_next-n_embd ncols=4096 z_bytes_match=true q_bytes_match=true d_bytes_match=true -> PASS
[q8-fuse-gate] glm5_next-kda_qkv ncols=8192 z_bytes_match=true q_bytes_match=true d_bytes_match=true -> PASS
[q8-fuse-gate] glm5_next-moe_ff_exp ncols=1536 z_bytes_match=true q_bytes_match=true d_bytes_match=true -> PASS
[glm5-w8] engaged t=1 in_f=4096 out_f=576 (q8_0 mirror, MEMRA_GLM5_W8=1)
[q8-fuse-gate] glm5_next-w8-kda-qkv t=1 in_f=4096 -> PASS
[q8-fuse-gate] glm5_next-w8-kda-qkv t=4 in_f=4096 -> PASS
[q8-fuse-gate] ALL SHAPES PASS (byte-identical fused vs chain)
```

The `t=1` and `t=4` W8-mirror shapes (new this lane) are byte-identical, confirming the
identity argument above holds on the real dispatch functions, not a reimplementation.

### Live-model regression receipt

`tests/kda_fused_proj_bf16_gpu.rs::whole_mixer_matches_across_the_door_and_does_not_worsen_vs_reference`
(a real glm5-shaped KDA fixture, wq/wk/wv admitted `FloatBf16` at >=2M elements each — the
same `bf16_mmv` residency threshold the serving recipe uses), re-run with `MEMRA_GLM5_W8=1
MEMRA_GLM5_Q8_FUSE=1` set alongside the test's own `MEMRA_BF16_MMV=1`:

```
[glm5-q8-fuse] engaged W8-mirror KDA qkv t=1 rows_exact=false in_f=256 (one quantize_q8_1 replaces three; MEMRA_GLM5_Q8_FUSE=1 + MEMRA_GLM5_W8=1)
[kda-fused6 bf16 receipt] mixer t=1: fused-vs-unfused 0.000e0; vs reference OFF 9.148e-3 / ON 9.148e-3
[kda-fused6 bf16 receipt] mixer t=7: fused-vs-unfused 0.000e0; vs reference OFF 1.194e-2 / ON 1.194e-2
[kda-fused6 bf16 receipt] mixer t=15: fused-vs-unfused 0.000e0; vs reference OFF 8.910e-3 / ON 8.910e-3
ok
```

The engagement line confirms `kda_proj_qkv_qmirror3` fired at t=1 (this test's `kda_attn`
entry is always the prefill/non-rows-exact walk, so t=7/15 correctly fall through to the
unfused `matmul_group` path per this lane's stated scope — not a bug, the test's own
`[glm5-w8] engaged t=7 ...` line at those widths is the pre-existing per-weight dispatch).
The whole-mixer-vs-reference band is UNCHANGED from the door-off receipt at every t
(9.148e-3 / 1.194e-2 / 8.910e-3 — the same bf16-operand floor both arms already sit at),
confirming the wiring did not move a bit relative to the pre-existing correctness bar.

One pre-existing test in the same file,
`bf16_door_is_bit_identical_on_bf16_rows_and_banded_on_f32_rows_at_t_1_to_15`, fails when run
under this same env: it asserts `MEMRA_KDA_FUSED_PROJ`'s OWN bf16 arm
(`kda_proj_fused6`) engages, and that arm already declines by design whenever
`glm5_w8_on()` is true (pre-existing code, unrelated to this lane — read `kda_proj_fused6`'s
own guard in `kda.rs`). That test was never written to run with `MEMRA_GLM5_W8=1`; the
failure is a known incompatibility between two independently-gated doors, not a regression.

### No live MLA regression receipt

No test fixture in this repo currently admits MLA's wq_a/wkv_a as `FloatBf16` (the
`bf16_mmv` residency threshold), so `mla_proj_qa_kv_qmirror2` has the synthetic-primitive
identity argument (the `_pre` twins are the SAME functions the KDA path already proved
byte-identical) but no live-model regression receipt. Named open item.

## Build/lint receipts

- `MEMRA_CUDA_ARCH=100a cargo build -p memra-engine` — clean.
- `MEMRA_CUDA_ARCH=120a cargo build -p memra-engine` — clean.
- `cargo fmt --all -- --check` — clean.
- `tools/check-flags.sh` — no new flag names introduced (this lane extends the semantics of
  three EXISTING flags: `MEMRA_GLM5_Q8_FUSE`, `MEMRA_GLM5_W8`, `MEMRA_BF16_MMV`), 816 runtime
  reads, 0 uncovered.
- `cargo clippy --release --all-targets -- -D warnings` — clean (fixed along the way: a
  `type_complexity` on `mla_proj_qa_kv_qmirror2`'s return type and two `map_entry` lints on
  the new `_pre` twins, whose non-`_pre` originals already carry the same allow).

## Open items (not this lane, named for the next one)

1. **No B200 receipt.** Same as the parent lane — no access to the box that produced the
   527-launch/token census. Door stays default OFF pending the A/B there.
2. **Prefill-width (t>1, non-rows-exact) KDA/MLA dispatch** still re-quantizes per weight —
   only decode (t==1) and verify-rows (t in 2..=32) are wired.
3. **KDA's `wo` and MLA's wq_b/wk_b/wv_b/wo** remain on their existing per-weight dispatch —
   each consumes a distinct downstream vector with no sibling to share a quantize with.
4. **No live MLA regression fixture** (see above) — only the synthetic-primitive gate.
5. **Exact launches-removed count** is per-layer-topology arithmetic (how many trunk layers
   are KDA vs MLA, and how many of those actually admit bf16 residency under the real
   artifact) rather than a re-measured nsys census; the 527 -> ~100 target in the task
   description is the aim this lane was built toward, not a number reproduced here.
