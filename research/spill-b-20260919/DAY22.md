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

### 1.3 Results of cells S and W (verbatim), and the named operation

Order of events, for the record: `DAY22.md` sections 1.1 and 1.2 were written and the build receipt taken
(`build-tip/`, 16:53:43Z to 16:56:20Z, exit 0) before cell S started (16:56:53Z); the pre-registration COMMIT landed
at 16:57:32Z (`99c939f6a`) because the pre-commit hook refused the first commit on the walk binary's formatting; the
formatting change moved no prediction and no logic. Cell W ran after that commit on a walk binary rebuilt from the
committed source (`build-walk-fmt/`).

**Cell S (`rtx5090-day22/seam/`, `run-day22-seam.sh`, exit 0, collector `executed-not-qualified`, 696 samples at
250 ms, 55..88 C, peak draw 181.25 W, peak `memory.used` 19,929 MiB, card alone before and after):**

```text
S0 baseline:                                one call over 9296 tokens: logits_sha=14ab5f8b365dbd71
                                              9280 + 16: logits_sha=35bd15f063bfd5ba DIFFERS (known: a 16-row final segment is not bitwise on either artifact)
                                              9248 + 48: logits_sha=14ab5f8b365dbd71 ok
S1 MEMRA_NO_BATCHED=1:                      one call over 9296 tokens: logits_sha=14ab5f8b365dbd71
                                              9280 + 16: logits_sha=14ab5f8b365dbd71 ok
                                              9248 + 48: logits_sha=14ab5f8b365dbd71 ok
S2 MEMRA_NO_BATCHED=1 MEMRA_PRIME_CHUNK=32: one call over 9296 tokens: logits_sha=14ab5f8b365dbd71
                                              9280 + 16: logits_sha=14ab5f8b365dbd71 ok
                                              9248 + 48: logits_sha=14ab5f8b365dbd71 ok
S3 MEMRA_NO_BATCHED=1, total 9297:          one call over 9297 tokens: logits_sha=e641952cac19e526
                                              9280 + 17: logits_sha=e641952cac19e526 ok
```

Every prediction of section 1.2 held byte for byte. The `[prime-row]` receipt of the 16-row call: baseline
`rows=16 ... logits_sha=35bd15f063bfd5ba top1=318 margin=1.369547e0`; under S1 `rows=16 ... logits_sha=14ab5f8b365dbd71
top1=318 margin=1.375317e0`, the one-call row's margin. With the batched tier off, the 16-row prime and the wide
prime are bit-identical end to end, and the day-21 `MEMRA_PRIME_CHUNK=32` digest (`35bd15f063bfd5ba`) becomes the
default schedule's digest: the cold prime is chunk-invariant at width 16 once the tier is out of the walk.

**Cell W (`rtx5090-day22/width-walk/`, `run-day22-walk.sh width-walk 1800 17 16 48`, exit 0, `executed-not-qualified`,
31 samples, 62..67 C, peak draw 80.64 W, peak `memory.used` 14,521 MiB):** 497 tensors walked (48 GDN layers x 5, 16
attention layers x 4, 64 x 3 dense FFN, the head; `attn_gate` is absent on this artifact, MLA/KDA/MoE not present).
Verbatim summary lines and the first GDN layer:

```text
WIDTH WALK width 16 vs 17: 96 of 497 tensors differ; tensor names: {"ssm_alpha", "ssm_beta"}; sites: ["00/ssm_beta", "00/ssm_alpha", "01/ssm_beta", ..., "62/ssm_beta", "62/ssm_alpha"]
WIDTH WALK width 48 vs 17: 0 of 497 tensors differ; tensor names: {}; sites: []
layer 00 ssm_beta  qtype=NVFP4   in_f=5120   out_f=48     width  16 vs 17: rows_differ=16/16 maxabs=1.490e-7 ref_sha=e40f0aeaaec76762 sha=0a984de6e2ed5fd0 DIFFERS
layer 00 ssm_alpha qtype=NVFP4   in_f=5120   out_f=48     width  16 vs 17: rows_differ=16/16 maxabs=4.768e-7 ref_sha=27a9e796e65eee50 sha=c9c9a51efce71c75 DIFFERS
layer 00 wqkv      qtype=NVFP4   in_f=5120   out_f=10240  width  16 vs 17: rows_differ=0/16 maxabs=0.000e0 ref_sha=92c8bbc0029fa2b7 sha=92c8bbc0029fa2b7 same
layer head output    qtype=Q5_K    in_f=5120   out_f=248320 width  16 vs 17: rows_differ=0/16 maxabs=0.000e0 ref_sha=4b4980f9e6b40c82 sha=4b4980f9e6b40c82 same
```

All 96 differing sites read `rows_differ=16/16`; `maxabs` spans 1.192e-7 to 5.066e-7 on rows of order 1. H1 held
exactly: the 96 are `ssm_beta` and `ssm_alpha` of every one of the 48 GDN layers (0, 1, 2, 4, 5, 6, ..., 62; layers
3, 7, ..., 63 are attention), and nothing else moves. The wide control (48 vs 17) is identical everywhere.

**The named operation.** `Engine::matmul` (`crates/memra-engine/src/lib.rs`) for the GDN `ssm_beta` and
`ssm_alpha` projections (NVFP4, `[5120, 48]`, `out_f = 48 < GEMM_MIN_OUT_F = 128`), reached from
`HybridModel::linear_attn_prime` -> `matmul_group_prefill_xh(&[wqkv, wqkv_gate, ssm_beta, ssm_alpha])` -> (no A4
stamp, no f16 mirror on this artifact) `matmul_group` -> `matmul`. Dispatch condition at m = 16: `out_f < 128`
skips the `m >= GEMM_M_THRESHOLD` MMQ/GEMM arms; then `(2..=16).contains(&m) && fast && MEMRA_NO_BATCHED unset &&
(m <= 4 || b8_enabled())` with the b16 qtype admission (NVFP4 listed) selects `qmatvec_mmvq_batched` (mcols 16, the
32-thread warp reduce). At m = 17 and above the same tensor takes `qmatvec_dp4a_named("qmatvec_nvfp4_dp4a")`,
grid `(out_f, m)`, the 128-thread two-level reduce. Rows: all 16 rows of the chunk. Same seam in `matmul_pre`. The
day-21 sentence "the batched-MMVQ b16 tier sits below [the GEMM threshold] and is not reached at m = 16" was wrong
for exactly the tensors the tiny-out_f guard routes past the GEMM arms.

### 1.4 The fix, pre-registered before its cells ran

Shape (a) from DAY21 2.4 is one bounded dispatch change in the pattern `matmul` already has for intent-keyed
dispatch (the `verify_exact` scope): a second RAII scope `prefill_rows` (`Engine::prefill_rows_scope`, the same
`ExactScope` guard type, restored on every exit), armed at the top of `HybridModel::prime_layers` (the per-chunk
layer walk every prime chunk and every PP stage goes through), and one admission clause `batched_tier_admits()
= verify_exact_on() || !prefill_rows_on()` on the batched tier in `matmul` and `matmul_pre`. Verify-exact keeps
precedence, so the t = 16 dflash verify (which runs under `exact_scope(true)`) still rides the tier. No new
`MEMRA_*` read, no new kernel, no new numeric program: a 16-row prime chunk now takes the `grid.y = m` dp4a program
its 17-row sibling takes. `matmul_decode_exact` and `matmul_decode_exact_pre` (the decode/verify-exact classes,
never on the prime walk) are untouched. The continuation gate's "known" exemption for a 16-row tail is removed
(a differing 16-row split now FAILS the gate: a tightening, not a relaxation). The walk binary gains a `scope=`
column and runs every tensor twice: `prime` (under the scope, what `prime_layers` runs) and `bare` (the
decode/verify class).

Bytes that move: every prime call of exactly 16 rows (a restored suffix of exactly PRIME_MIN_T, a cold prompt whose
schedule ends in 16, a 16-token prompt) on any model with a quantized `out_f < 128` projection in the batched
tier's qtype list; nothing at any other width, nothing in decode, verify or spec.

Predictions, written before the cells ran (fix build `build-fix/`):

| cell | prediction |
| --- | --- |
| F1 walk (`fix-walk/`), widths 17 vs 16, 48 | `scope=prime`: 0 of 497 differ at 16 vs 17 and at 48 vs 17; `scope=bare`: 96 of 497 (`ssm_alpha`, `ssm_beta`) at 16 vs 17, 0 at 48 vs 17 (the decode/verify class is unchanged) |
| F2 gate table (`fix-gate/`) | 9296: one-call `14ab5f8b365dbd71`, every split 16..208 `ok`; 9297: `e641952cac19e526`, 17 and 49 `ok`; 9311: `b61719f294866b05`, 31 and 63 `ok`; 9312: `736a88d7c7448dcb`, 32, 64, 96 `ok`; `MEMRA_PRIME_CHUNK=32`: one-call `14ab5f8b365dbd71` (was `35bd15f063bfd5ba`), splits `ok`; `MEMRA_PRIME_CHUNK=16`: a new one-call digest (every chunk now dp4a; off the 32 grid, so not comparable to the default), both splits `ok`; `MEMRA_NO_BATCHED=1`: identical to the default (the tier is already out of the prime walk); `A4 CONTINUATION GATE: PASS` on every arm |
| F3 `kernel-check` (`fix-kc/`), both manifests | `ALL GREEN (N cells, K skipped)`, K within local-ci's budget of 11 on this rig (no kernel changed) |
| F4 the #379 hit gate (`fix-hitgate/`), 9B trunk, fix `memra-server` | `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` (as day 19's corrected run on the pre-fix tree) |
