# glm5_next decode diet: q8_0 mirror for the BF16 KDA/MLA trunk (MEMRA_GLM5_W8)

Status: door, dispatch, and gate written; both arches build clean under CI-equivalent local
builds (`MEMRA_CUDA_ARCH=100a` / `120a`); synthetic gate RUN GREEN on the rig. NO B200 in this
worktree — the real-artifact argmax gate, the interleaved x5 speed cells, and the
vendor-default sampled twin with engagement receipts landed on the pair 2026-09-02 (darklanes research/glm5-b200-20260902/LANE.md 14:27Z/14:48Z; argmax gate PASS, tape identical, +10% plain, +2-4% spec); the paragraph below records what was asked of the session that owns the
2x B200 pair. Door `MEMRA_GLM5_W8` default OFF everywhere.

Branch: `lane/b200-glm5-w8-20260902` (worktree `wt-b200-w8`), from
`lane/glm5-b200-int2-20260902` (carries the per-device cuBLASLt fix and four decode doors:
`MEMRA_B200_MATVEC_ARM`, `MEMRA_HC_FUSED_PRE`, `MEMRA_B200_MLA_DECODE_ARM`,
`MEMRA_GLM5_Q8_FUSE`). None of those four doors' code was edited — this lane composes with
them (one refusal-condition line added to a fifth, unrelated door, `MEMRA_KDA_FUSED_PROJ`;
see "Composition" below).

## The measurement that motivated this lane

nsys, 2x B200, GLM-5.3-Flash NVFP4 mint, resident PP2, plain decode t=1: per token ~15 GB of
weight reads, of which the BF16-resident KDA/attention projections dominate — `MEMRA_BF16_MMV`
(W4A16 auto-on) admits them to raw bf16 residency, and the t=1 decode program
`matvec_bf16_f32acc_x4_rows` reads them at 211 launches/token, ~13.5 GB/token. The NVFP4
experts are ~3.6 GB/token (untouched by this door). At 8 TB/s that bf16 trunk alone is ~1.7
ms/token of the ~24 ms/token decode step — the biggest byte lever after the launch floor
(precedent: `research/b200-matvec-occupancy-20260902/`, which attacked the LAUNCH/occupancy
side of the same kernel; this lane attacks the BYTES side).

In-tree precedent for the exact lever: `MEMRA_STEP_TP_W8` (docs/FLAGS.md) ships a q8_0 mirror
of bf16-resident weights for the step37 family's decode-tier attention/GEMV projections,
built once per weight at first decode use (`encode_q8_0_rows_from_bf16` — the same 34-byte
`[half d][32 x i8]` block program `quant_K_block` writes for the KV cache), then routed
through the shipped q8_0 mmvq/rows kernels. Its hybrid half (`MEMRA_W8_HYBRID`) already ships
a GENERIC pointer-keyed mirror cache (`matvec_bf16_via_q8_mirror`/`_t`, `w8_mirrors`/`w8_act`
fields) that is not step37-specific in its mechanics — only in what gates its use. This lane
reuses that exact building block under a NEW, independent door for glm5_next.

## What is mirrored

`MEMRA_GLM5_W8` is wired at the SHARED dispatch level, not at individual KDA/MLA call sites:

- `Engine::matvec_bf16_rows_into` (lib.rs) — the function `matmul()`'s `FloatBf16` branch
  calls for every `m` in `1..=32` (decode t=1 and the small-t verify-rows tiers). Two new
  branches, modeled on `MEMRA_STEP_TP_W8`'s hybrid-half branches immediately above them: `t
  in 2..=32` routes through `matvec_bf16_via_q8_mirror_t`, `t == 1` through
  `matvec_bf16_via_q8_mirror`. Both are the identical calls the step37 door already makes;
  only the gating predicate (`glm5_w8_on()` instead of `step_tp_w8_on() && w8_hybrid_on()`)
  and the announce line differ.
- `Engine::matmul_rows_exact` (lib.rs) — the verify-rows-exact walk's own dispatcher (used by
  `kda_core_gated`'s `rows_exact` arm and `mla_attn_core_pre_wo`'s `mm` closure when
  `rows_exact=true`). A new branch for `m in 2..=32` takes precedence over the existing
  `matvec_bf16_tcols_into` branch when the door is on (that branch's own condition gained `&&
  !glm5_w8_on()` so the two classes cannot both claim the same call).

Because `matmul()`/`matmul_group()`/`matmul_rows_exact()` are the SAME dispatchers every
KDA/MLA projection call already goes through — confirmed by reading `kda_core_gated` (the six
stage-1 projections wq/wk/wv/f_a/g_a/b_proj via `matmul_group`, `la.f_b`/`la.g_b` and `la.wo`
via `matmul`/`matmul_rows_exact` directly) and `mla_attn_core_pre_wo`'s `mm` closure (wq_a,
wq_b, wkv_a) plus `mla_attn_cached_inner`'s `wo` dispatch — **zero KDA/MLA call sites needed
editing**. This also means engagement is honest about its own scope: the door is not
name-scoped to "KDA/MLA tensors", it engages wherever a bf16-resident weight satisfying
`in_f % 32 == 0 && out_f >= 64` reaches this shared path. On glm5_next's actual residency
census (`MEMRA_BF16_MMV` GLM5_NEXT ACCEPTANCE row, docs/FLAGS.md: 148 resident tensors per
boot — 34x kda_q/k/v/out at 33.5M elements, 12x indexer.attn_q_b at 6.3M elements, plus
embed/head members) that set is dominated by the KDA/MLA trunk, but the door does not itself
exclude the LM head or embed table if they are bf16-resident and reach the same call — the
box census (`[glm5-w8] engaged`/`[w8-mirror] built` under `MEMRA_W8_TRACE=1`) is the receipt
for the actual engaged set, not an a priori guarantee.

KDA/MLA projections NOT covered: the "absorbed" MLA core (`mla_absorb_q`/`mla_decompress_v`,
which consume `wk_b`/`wv_b` through bespoke per-head batched kernels, not `matmul()`) and the
KDA delta-rule scan/conv kernels are untouched — they are not bf16-resident GEMV weights in
the sense this door's building block addresses.

## Composition

`MEMRA_KDA_FUSED_PROJ`'s BF16 operand arm (`kda.rs`, `qmatvec_kda6_bf16f32`) claims
bit-identity against `matvec_bf16_f32acc_x4_rows` — the UNMIRRORED program. It already
declined when `MEMRA_STEP_TP_W8 && MEMRA_W8_HYBRID` reroute that target; it now also declines
when `MEMRA_GLM5_W8` does, for the identical reason (one added disjunct in its refusal guard,
`crates/memra-engine/src/kda.rs`). This is the only edit outside the two dispatch functions
above.

The four decode doors named in the task brief (`MEMRA_B200_MATVEC_ARM`, `MEMRA_HC_FUSED_PRE`,
`MEMRA_B200_MLA_DECODE_ARM`, `MEMRA_GLM5_Q8_FUSE`) operate on different stages entirely —
occupancy arms on the MoE/bf16-rows kernel selection (composed with downstream, unedited: my
new branches simply return before reaching `MEMRA_B200_MATVEC_ARM`'s kernel-name choice at
the bottom of `matvec_bf16_rows_into` when they engage), the hyper-connection pre-mix chain,
the absorbed-MLA attention core, and the rms_norm+quantize fusion, respectively. None of them
call into `matvec_bf16_rows_into`/`matmul_rows_exact` for the tensors this door mirrors, so no
composition edit was needed for any of the four; their code is untouched.

`MEMRA_STEP_TP_W8`+`MEMRA_W8_HYBRID` and `MEMRA_GLM5_W8` are unrelated doors (different model
families) that happen to share the `w8_mirrors`/`w8_act` caches. The cache is keyed on
`(pointer, in_f, out_f)` and idempotent, so setting both together just serves both from one
mirror — no correctness issue, not a tested combination either.

## Numeric-class statement

Same class as `MEMRA_STEP_TP_W8`: the per-row arithmetic becomes an int8 dp4a dot with per-32
scales (q8_0, `qmatvec_mmvq`/`_rows_t`/`_rows_tw`/`_rows_tw32`) instead of a bf16xf32 fma chain
(`matvec_bf16_f32acc_x4_rows`), so a bit-tape cannot apply — acceptance is an argmax gate, not
byte identity. This changes weight PRECISION (the q8_0 mirror is a lossy re-encode of the bf16
original), not merely reduction order, so the acceptance bar is the argmax-tape agreement plus
(pending, box-only) a quality battery, per the `MEMRA_STEP_TP_W8` precedent's own stated bar.

No new CUDA kernel was written. Every kernel this door reaches
(`encode_q8_0_rows_from_bf16`, `qmatvec_q8_0_rows_t`/`_tw`/`_tw32`, `qmatvec_mmvq`'s q8_0/rp
arm) already ships and is already exercised by `MEMRA_STEP_TP_W8`'s battery on step37. This
lane is a NEW DISPATCH SITE onto EXISTING kernels, not new numeric machinery — see
`docs/KERNELS.md`'s new row.

## Bytes/token: before vs after, by arithmetic (measured on the pair 2026-09-02: +10% plain, +2-4% spec, receipts in darklanes research/glm5-b200-20260902/LANE.md)

bf16 = 2 B/weight. q8_0 mirror block = 34 B per 32 weights = 1.0625 B/weight (0.53125x bf16).

Against the nsys census above: the BF16-resident KDA/attention trunk reads ~13.5 GB/token
(211 launches/token). Mirrored, that class projects to:

```
13.5 GB * 0.53125 = 7.171875 GB/token          (new bf16-trunk read bytes)
13.5 - 7.171875   = 6.328125 GB/token saved
6.328125 GB / 8 TB/s = ~0.791 ms/token saved   (at the measured 8 TB/s HBM3e, perfect
                                                 bandwidth-bound scaling — no launch/gap
                                                 overhead accounted for)
```

Against the trunk's own ~1.7 ms/token share of the ~24 ms/token decode step, this is roughly
halving that trunk's read cost (0.531x is close to 0.5x), which projects to ~3.3% of total
decode-step time from this door alone, under the perfect-bandwidth assumption. NOTE the task's
own headline total (~15 GB/token) and the sum of its two named components (13.5 + 3.6 = 17.1
GB/token) do not reconcile exactly — stated here as given, not silently forced to agree; the
arithmetic above is scoped to the 13.5 GB/token bf16-trunk component only, which this door
touches, not to whichever total is used as the anchor.

## VRAM cost: +1.0625 B/weight, ADDITIVE (not a replacement)

The mirror is extra storage: the bf16 original stays resident (every prefill/verify-arm class
that is NOT rerouted keeps the arithmetic it was qualified against), so a mirrored weight's
total footprint grows from 2 B/w to 2 + 1.0625 = 3.0625 B/w.

On the census's two itemized sets alone:

```
34 x kda_q/k/v/out  @ 33.5M elements : 34 * 33.5e6 * 1.0625 B ~= 1.21 GB added
12 x indexer.attn_q_b @ 6.3M elements: 12 * 6.3e6  * 1.0625 B ~= 0.080 GB added
```

Any other resident tensor this door reaches (embed/head members, per the same census) adds
the same 0.53125x its own bf16 bytes on top. The box run's `[glm5-w8] engaged` /
`[w8-mirror] built` (`MEMRA_W8_TRACE=1`) lines are the receipt for the actual engaged set and
total — this arithmetic is a projection from the two itemized shapes, not the full account.
`MEMRA_STEP_TP_W8`'s own precedent (docs/FLAGS.md) found its mirror does not fit alongside a
262144-token cache reservation on 96 GB; the same residency-vs-context tradeoff applies here
and needs its own box measurement before any serving recommendation.

## The door: `MEMRA_GLM5_W8`

`pub(crate) fn glm5_w8_on() -> bool` (lib.rs, next to `step_tp_w8_on()`/`w8_hybrid_on()`):

```rust
pub(crate) fn glm5_w8_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MEMRA_GLM5_W8").as_deref() == Ok("1"))
}
```

Strict boolean, unset/`0` = OFF (default, bf16 unchanged), `1` = ON. Deliberately NOT
step37-family-armed (`step37_door`/`arm_step37_serving_defaults`) and NOT gated behind
`MEMRA_W8_HYBRID` — a separate model family gets its own receipts, not a silent ride on
another lane's owner-ratified flip. Read once per process into a `OnceLock`, so the rollback
seam is exact: unset or `=0` leaves every call site byte-identical to pre-lane behavior.

FLAGS.md row: `docs/FLAGS.md`, immediately after the `MEMRA_KDA_FUSED_PROJ` row (the door it
composes with). States the numeric class, the bytes/token arithmetic above, the VRAM cost, the
composition notes, the rollback seam, and the gate invocation. `docs/KERNELS.md` gets one new
row noting the new dispatch site onto the existing q8_0 mirror kernel family (no kernel code
added).

## The gate

`crates/memra-engine/src/bin/glm5_w8_gate.rs` (bin name `glm5-w8-gate`). Two independent
probes:

1. **Synthetic per-shape probe** (always runs; the only thing that ran on this rig). No real
   checkpoint needed. For each of four census-representative shapes (`4096x8192`,
   `8192x4096` for the KDA q/k/v/out class; `4096x1536` for the MLA indexer.attn_q_b class;
   `2048x128` as a small gate/beta-class width), builds a random bf16 weight matrix and its
   q8_0 mirror via the SAME `encode_q8_0_from_bf16` + `build_q8_rp4_raw` calls the engine's
   own mirror cache makes, then for N random f32 activations runs both matvec classes
   (`Engine::matvec_bf16_into` and `quantize_q8_1_into` + `qmatvec_mmvq_into` with
   `QT_Q8_0`/`rp=true` — the exact call `matvec_bf16_via_q8_mirror` makes) and reports max abs
   error and argmax agreement over the N activations. This is a WIRING sanity check (random
   unstructured weights are not the trained distribution), not a quantization-tightness bound
   — it fails only on NaN/Inf or an error large enough to indicate a block-layout bug.
2. **Real-artifact 32-token greedy-tape probe** (`GLM5_ARTIFACT=<safetensors dir | .gguf>`,
   box-only — the real checkpoint does not fit the one-card rig). Because `MEMRA_GLM5_W8` is
   read into a per-process `OnceLock`, comparing door-off vs door-on within one process would
   not be a fresh boot per arm; the gate instead re-execs itself as two child processes (a
   `GLM5_W8_GATE_WORKER=1` internal mode), one per arm, each loading the model fresh and
   greedy-decoding 32 tokens from the SAME fixed prompt, then diffs the two tapes and reports
   argmax agreement — the pin-against-truth law (same prompt, fresh boot per arm).

Invocation:

```
# synthetic only (ran on this rig, exactness-only per the rig's role — never a timing run)
cargo run -p memra-engine --release --bin glm5-w8-gate -- 200

# + real-artifact tape (box only)
GLM5_ARTIFACT=/path/to/glm5-next-artifact \
  cargo run -p memra-engine --release --bin glm5-w8-gate -- 200
```

### Rig receipt (synthetic probe, 5090, exactness-only)

```
=== MEMRA_GLM5_W8 synthetic per-shape probe (N=200 random activations/shape) ===
shape=kda_qkvo_4096x8192 in_f=4096 out_f=8192 max_abs_err=5.549679e-1 argmax_agree=197/200 (98.5%) nan_or_inf=false
shape=kda_qkvo_8192x4096 in_f=8192 out_f=4096 max_abs_err=8.604097e-1 argmax_agree=198/200 (99.0%) nan_or_inf=false
shape=mla_indexer_attn_q_b_4096x1536 in_f=4096 out_f=1536 max_abs_err=6.035824e-1 argmax_agree=198/200 (99.0%) nan_or_inf=false
shape=small_gate_2048x128 in_f=2048 out_f=128 max_abs_err=3.440189e-1 argmax_agree=200/200 (100.0%) nan_or_inf=false
GLM5_ARTIFACT not set: skipping the real-artifact greedy-tape probe (box-only — the checkpoint does not fit a one-card rig). The synthetic probe above is the only receipt from this run.
glm5-w8-gate: PASS
```

No NaN/Inf, no wiring-sized error (bar is 50.0 abs) on any shape at N=200 — confirms the
encode/mirror/matvec plumbing this door reuses is wired correctly. Random-weight argmax
agreement (98.5-100%) is a wiring signal only, not a quality claim: real trained weights and
real activation distributions are what the box's real-artifact tape and quality battery must
speak to.

## Build/test receipts (this worktree, RTX 5090 rig)

- `cargo build -p memra-engine --lib --bin glm5-w8-gate` (sm_120a, auto-detected): green.
- `MEMRA_CUDA_ARCH=100a cargo build -p memra-engine --lib --bin glm5-w8-gate`: green.
- `cargo fmt --all -- --check`: green.
- `tools/check-flags.sh`: green (804 runtime literal reads, `MEMRA_GLM5_W8` covered).
- `cargo clippy -p memra-engine --lib --bin glm5-w8-gate --no-deps`: one pre-existing,
  unrelated warning (`hyper_ffn_branch` too-many-arguments); nothing from this lane's code.
- `cargo test -p memra-engine --lib`: 365 passed, 3 ignored (need CUDA — none touch this
  lane), 0 failed.
- `flock /tmp/memra-5090.lock cargo test -p memra-engine --test kda_fused_proj_gpu --test
  kda_fused_proj_bf16_gpu --test glm5_matvec_doors_gpu --test glm5_verify_batch_gpu --
  --ignored --test-threads=1`: 19/19 passed — confirms the doors this lane composes with
  (`MEMRA_KDA_FUSED_PROJ` both arms, the glm5 matvec-doors bit-gates, the verify-batch
  bit-gates) are unaffected with `MEMRA_GLM5_W8` unset (its default, so every existing gate
  ran its unchanged path).

## Open items (owner-visible, not silently dropped)

1. **The box run itself.** Nothing in this PR is a performance claim; the bytes/token numbers
   above are arithmetic against the nsys census that motivated the lane, not a measurement of
   this door. The session with 2x B200 access runs `glm5-w8-gate` with `GLM5_ARTIFACT` set
   (real-artifact argmax-tape probe), then the interleaved x5 speed cells and the
   vendor-default sampled twin with engagement receipts (greedy-is-the-instrument law), and
   the default only moves on those receipts.
2. **VRAM/context tradeoff not measured here.** `MEMRA_STEP_TP_W8`'s own precedent found its
   mirror does not coexist with a 262144-token cache reservation on 96 GB; whether the same
   bites on 2x B200 (192 GB combined, but PP2-split and carrying the NVFP4 expert banks too)
   is a box question.
3. **Engagement scope is generic, not name-scoped.** As stated above, the door engages
   wherever a qualifying bf16-resident weight reaches the shared dispatch, which may include
   the LM head/embed table if they are bf16-resident on the real artifact. The box's
   `MEMRA_W8_TRACE=1` census is the way to confirm the actual engaged set before citing a
   final bytes/token number.
4. **A fused multi-projection q8_0 kernel (the step37 fused-QKV precedent) was NOT written.**
   The task allowed one "if needed" for the KDA q/k/v(/gate/beta/a) group; the existing
   generic `matvec_bf16_via_q8_mirror`/`_t` machinery (one launch per projection, reusing
   `qmatvec_mmvq`/`_rows_t`) is what this lane wired, matching `MEMRA_STEP_TP_W8`'s own
   hybrid-half shape (which also does NOT fuse the LM head/shexp/dense-FFN mirrors it
   covers). If the box census shows per-launch overhead (not bytes) is now the binding
   constraint for this group, a fused q8_0 KDA6 kernel modeled on
   `qmatvec_kda6_bf16f32`/the step37 fused QKV kernel is the named follow-up — same shape as
   `MEMRA_KDA_FUSED_PROJ`'s own two arms, a third (q8-mirror) operand arm.
