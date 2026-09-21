# WP-B day 22: memra#427 named (the b16 batched-MMVQ tier at m = 16 for the GDN alpha/beta projections), the fix shape decided, memra#445 mapped past section 1

Repository: **avifenesh/memra**, branch `lane/spill-b-20260919`, on `origin/main` `1097d7450` (day 21 merged through
#611; the merge is `34c0ac903`). The merge brought `crates/memra-engine/build.rs` (#610) into the tree past the
committed qualification pointer, so every push of the day is refused `UNQUALIFIED` by the #589 hook on a plain push
and runs as `MEMRA_RELEASE_QUALIFICATION_MODE=development git push` (`UNQUALIFIED DEVELOPMENT`, logged in
`.git/memra-gate-skips.log`). **No qualification is claimed anywhere in this record**; every collector cell is
`executed-not-qualified`.

Rig discipline as on every prior day: every GPU command on the local RTX 5090 Laptop GPU goes through
`tools/tier-battery.py --rig rtx5090` (`/tmp/memra-5090.lock`), CPU-heavy work runs under
`systemd-run --user --scope -p CPUQuota=1200% -p MemoryMax=28G`, `nvidia-smi --query-compute-apps` is read before
and after every cell, and no process this lane did not start is signalled or inspected. The target card is not
needed for today's cells (the #427 digests were byte-identical across the two card classes on day 21, and the
kernel-naming question is a dispatch question, not a card question); nothing here is timed.

## 1. memra#427: pre-registration of the width walk (written before any cell ran)

### 1.1 What the code says before the run

Day 21 left the operation unnamed and listed, as still keyed on the call width, "the dense (non-quantized) matmuls
that reach cuBLASLt" and "the GDN prep/conv kernels' width tiling". Reading `Engine::matmul`
(`crates/memra-engine/src/lib.rs`, the `GEMM_M_THRESHOLD` block and the tiers below it) today names a third
candidate that separates exactly 16 from 17, which the two day-21 candidates do not:

- The prefill GEMM/MMQ arms are `m >= GEMM_M_THRESHOLD (16) && out_f >= GEMM_MIN_OUT_F (128)`. A quantized
  projection with `out_f < 128` skips them at EVERY m (the tiny-out_f guard, 2026-06-28: the tile grid starves the
  SMs for `ssm_beta`/`ssm_alpha`).
- Below them sits the batched weight-resident MMVQ tier: `(2..=16).contains(&m) && fast &&
  MEMRA_NO_BATCHED unset && (m <= 4 || b8_enabled())`, with the b16 admission `m <= 8 || qtype in {Q4_0, Q6_K,
  F8_E4M3, NVFP4, Q4_K, Q5_K, Q8_0}`. It calls `qmatvec_mmvq_batched` (mcols 16, the 32-thread warp reduce). The
  tier's own comment records that its kernels are "bit-identical per (token,row) to MMVQ's 32-thread warp reduce,
  NOT to the dp4a kernels' 128-thread two-level reduce".
- At m = 17 the same tensor falls through to `qmatvec_<qtype>_fast` / `qmatvec_dp4a_named`: the `grid.y = m`
  dp4a kernel, one column per token, the program a 48-row or 80-row chunk also runs.
- `matmul_pre` carries the same tier with the same `(2..=16)` bound.
- The `PRIME_MIN_T` doc comment (`hybrid_forward.rs`) says where these tensors land: "the dp4a matvec family taken
  at out_f < 128 (`qmatvec*_fast`/`qmatvec_dp4a_named`, which is where the GDN ssm_beta/ssm_alpha projections
  land)". That sentence was true when a prime was always m >= 16 and the batched tier stopped at 8; the b16 tier
  was admitted 2026-07-11 (spec K > 7) and widened into the exact-16 decode tier 2026-08-01, and nobody re-read the
  prime's m = 16 edge against it.

The prime path reaches these tensors through `linear_attn_prime` -> `matmul_group_prefill_xh(&[wqkv, wqkv_gate,
ssm_beta, ssm_alpha], ..)` -> (activation program present) `matmul_group_prefill` -> `matmul_prefill` per tensor ->
`matmul` for a tensor without an A4 stamp. `ssm_beta` and `ssm_alpha` are `[n_embd, num_v_heads]`, `out_f =
num_v_heads`, far below 128.

**Hypothesis H1 (pre-registered).** In a 16-row prime chunk, `ssm_beta` and `ssm_alpha` of every GDN layer take the
b16 batched-MMVQ program; in a 17-row (or wider) chunk they take the `grid.y = m` dp4a program. Every other
projection has `out_f >= 128` and rides the same GEMM/MMQ arm at both widths; norms, the fused GDN conv
(`grid.y = t`, per token), the GDN scan and attention are width-invariant per row. So the first operation whose
output differs between the two widths for the same rows is the alpha/beta projection of the FIRST GDN layer, the
difference is a rounding-order difference (below a token flip on `docs/SERVING.md`, day 21), and everything after
it differs by propagation.

### 1.2 The two cells, with predictions

**Cell W (the per-op width walk), `rtx5090-day22/width-walk/`, runner `run-day22-walk.sh`.** A new diagnostic
binary `qwen-a4-width-walk` (`crates/memra-engine/src/bin/qwen_a4_width_walk.rs`, Cargo `[[bin]]` entry; no engine
code changed, no flag read) loads the artifact, and for every layer and every projection tensor the prime walk
uses (attention `wq wk wv wo attn_gate`, GDN `wqkv wqkv_gate ssm_beta ssm_alpha ssm_out`, dense `ffn_gate ffn_up
ffn_down`, and the `output` head) builds ONE deterministic activation of `max(widths)` rows (LCG, values in
(-1, 1)), then calls the prime path's own entry `Engine::matmul_prefill(w, x[..m], m)` at the reference width 17
and at each other width (16, then 48 as the wide control), and compares the shared leading rows bitwise. It prints
per tensor `qtype in_f out_f rows_differ=k/n maxabs` and a digest of each output, plus a summary per width. No
server, no prefix cache, no drafter, no KV. Widths: reference 17; compared 16 and 48.

Predictions: at 16 vs 17, `ssm_beta` and `ssm_alpha` differ in every GDN layer (`rows_differ=16/16` or close to it,
`maxabs` at the 1e-6..1e-4 level for values of order 1) and every other tensor is `same`; at 48 vs 17 every tensor
is `same` (the `grid.y = m` program is per-row identical for every m above 16, as is every GEMM arm). If any
`out_f >= 128` tensor differs at 16 vs 17, H1 is incomplete and that tensor's dispatch is read next; if
`ssm_beta`/`ssm_alpha` are `same`, H1 is refuted and the day-21 candidates (cuBLASLt dense, conv tiling) are next.

**Cell S (the seam arm on the reproducer), `rtx5090-day22/seam/`, runner `run-day22-seam.sh`.**
`qwen-a4-continuation-gate` (unchanged since day 21) on the same artifact and prompt (`docs/SERVING.md`), with the
documented A/B reference flag `MEMRA_NO_BATCHED` (FLAGS.md: "per-m grid.y=m path for ALL m=2..8 [and the b16
tier], the batched-verify A/B reference"). Arms, in this order:

| arm | env | total, tails | prediction |
| --- | --- | --- | --- |
| S0 baseline | none | 9296: 16, 48 | one-call `14ab5f8b365dbd71`; `9280 + 16` DIFFERS `35bd15f063bfd5ba`; `9248 + 48` ok (day 21, byte for byte) |
| S1 | `MEMRA_NO_BATCHED=1` | 9296: 16, 48 | one-call `14ab5f8b365dbd71` (unchanged: the default schedule has no chunk of 2..=16 rows); `9280 + 16` **ok** with `14ab5f8b365dbd71`; `9248 + 48` ok |
| S2 | `MEMRA_NO_BATCHED=1 MEMRA_PRIME_CHUNK=32` | 9296: 16, 48 | one-call `14ab5f8b365dbd71` (day 21 read `35bd15f063bfd5ba` here with the tier on: the cold prime becomes chunk-invariant at width 16); both splits ok |
| S3 control | `MEMRA_NO_BATCHED=1` | 9297: 17 | one-call `e641952cac19e526`, `9280 + 17` ok (the flag is inert at 17: nothing in the range 2..=16 is dispatched) |

If S1 turns the 16-row split `ok`, the b16 tier is the ONLY width-keyed operation in the 16-row layer walk (with
it off, the two programs are bit-identical end to end), and cell W says which tensors it touched. If S1 still
DIFFERS, the tier is not the whole story and cell W decides.

The flag is an existing documented rollback/reference seam, set only in a cell's environment; it is not a fix and
is not proposed as one (it also moves the m = 2..8 verify tiers, which is a different program for serving).

Both cells are pass/fail digests, not timed; both run under the collector on the local card only.
