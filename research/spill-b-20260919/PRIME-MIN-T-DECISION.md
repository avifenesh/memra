# Decision: a prime chunk of exactly PRIME_MIN_T (16) rows runs the wide chunk's program (memra#427)

Lane `lane/spill-b-20260919`, day 22 (2026-09-21). Status: **implemented on the lane** as fix shape (a), commit
`89c693be0`, local RTX 5090 receipts in `DAY22.md` sections 1.3 to 1.5; the target-card battery and the merge are
the integ's. Nothing here claims qualification.

## The defect, in one paragraph

A 16-row prime call and a wider prime call over the same rows produced different bits (`qwen-a4-continuation-gate`:
`9280 + 16` DIFFERS, every other split `ok`; DAY21 section 2). Day 21 classified it as a 16-row-chunk program, cold or
restored (`MEMRA_PRIME_CHUNK=32` one-call digest equal to the 16-row suffix digest). Day 22 named it: in
`Engine::matmul` (and `matmul_pre`), a quantized projection with `out_f < GEMM_MIN_OUT_F (128)` skips the
`m >= 16` GEMM/MMQ arms and, at m = 16, lands in the batched weight-resident MMVQ tier `(2..=16).contains(&m)`
(`qmatvec_mmvq_batched`, mcols 16, the 32-thread warp reduce), while at m >= 17 it takes the `grid.y = m` dp4a
kernel (the 128-thread two-level reduce). On the Qwen3.8-27B artifact those projections are the GDN `ssm_beta` and
`ssm_alpha` (`NVFP4 [5120, 48]`) of all 48 GDN layers: `WIDTH WALK width 16 vs 17: 96 of 497 tensors differ; tensor
names: {"ssm_alpha", "ssm_beta"}`, `rows_differ=16/16` at every site, `maxabs` 1.2e-7 to 5.1e-7; `width 48 vs 17: 0
of 497 tensors differ`. With the tier off (`MEMRA_NO_BATCHED=1`, an existing A/B reference flag) the 16-row split is
`ok` at the one-call digest and the chunk-32 cold prime digests as the default schedule does.

The tier is right where it is for decode and verify: it is bit-identical per (token, row) to the m = 1 MMVQ warp
reduce, which is the decode-parity law the exact-16 batched-decode tier (B = 9..=16) and the K = 15 verify need.
The prime path is the odd caller: it enters at m = 16 only from a chunk of exactly PRIME_MIN_T rows, and a prime of
16 rows is a prefill, not a verify.

## Candidate (a): engine side, the 16-row chunk runs the wide program

**What changes.** A second RAII scope on the engine, `prefill_rows` (`Engine::prefill_rows_scope`, the same
`ExactScope` guard type as `verify_exact`, restored on every exit), armed at the top of `HybridModel::prime_layers`
(the per-chunk layer walk every prime chunk and every PP stage goes through), and one admission clause on the
batched tier in `matmul` and `matmul_pre`: `batched_tier_admits() = verify_exact_on() || !prefill_rows_on()`.
Verify-exact keeps precedence (the t = 16 dflash verify runs under `exact_scope(true)` and must keep the tier).
`matmul_decode_exact` and `matmul_decode_exact_pre` are untouched (decode/verify-exact classes, never on the prime
walk). No new `MEMRA_*` read, no new kernel, no new numeric program: the 16-row chunk takes the program its 17-row
sibling takes. The continuation gate's "known" exemption for a 16-row tail is removed (a tightening).

**Which bytes move.** Every prime call of exactly 16 rows, on any model with a quantized `out_f < 128` projection
in the tier's qtype list (Q4_0, Q6_K, F8_E4M3, NVFP4, Q4_K, Q5_K, Q8_0): a restored suffix of exactly PRIME_MIN_T
(the shape #427 names: P publishes `capture_len(P)` and the identical re-request restores 16 rows), a cold prime whose
schedule ends in a 16-row chunk (t = 16 mod chunk under the default 4096 schedule: 4112, 8208, ...; or under an
explicit `MEMRA_PRIME_CHUNK`), and a 16-token cold prompt. Nothing moves at any other width, in decode, in verify or
in spec; the b16 tier's other callers keep their program.

**Batteries it owes.** `kernel-check` (no kernel changed; owed anyway), the `run-gen` argmax gate on affected prompt
lengths (a 16-token prompt and a t = 16 mod 4096 prompt), `run-spec` K=1..8 (the spec run's prime, not its verify),
the #379 hit gate (a restored suffix of exactly 16 rows is the hit shape), the twin gate
(`prefix-newest-turn-fits-gate.py`, V5 restored render equals cache-off render on every turn), and the
continuation gate with the 16-row split as a real verdict. Run locally today (5090): the two-scope width walk, the
continuation table, `kernel-check`, the hit gate (DAY22 section 1.5). Owed to the integ: `run-gen`, `run-spec`, the
twin gate, and the same table on the target card. Not armed today, stated plainly: `prime_layers_gemma` (gemma's own
chunked walk) and the step35 batched prime (`step35_prime_cache_batch`) do not take the scope; if either model class
carries a quantized `out_f < 128` projection, its 16-row chunk still has the tier's program, and arming those walks
is a follow-up with its own gate, not a silent extension of this one.

## Candidate (b): schedule side, never emit a 16-row segment

**What changes.** The fold rule in `prime_cache` (`t - end > 0 && t - end < PRIME_MIN_T` folds the tail) becomes
`<= PRIME_MIN_T`; the worker's three capture sites (`seed_capture_boundary`, the LCP split, the spec `capture_at`)
leave MORE than PRIME_MIN_T prompt tokens behind a seed (the #602 capture law says "at least"); `DflashRestoreSuffix::
new` and `check_mtp_prime_walker` refuse a suffix of exactly 16 the way they refuse below 16; the continuation gate's
16-row split becomes a refusal rather than a verdict.

**Which bytes move.** Where a cold prime folds (a 16-row tail joins the previous chunk: every cold prime with
t = 16 mod chunk changes digest, the same set as (a) but to the OTHER program), and where entries publish (one grid
step earlier for prompts whose length leaves exactly 16 behind the aligned length), so `cached_tokens` and the
restored-suffix lengths move for those prompts. It needs the twin gate, the restore gate and the hit gate green on
both cards plus the chunkinv row at the new fold; it touches the server's capture law that #602/#606 just landed.

**What it does not do.** It leaves the engine's own continuation invariant broken at width 16: a direct engine user
(the CLI, a gate, a future caller) priming 16 rows still gets the tier's program and a different digest from the same
rows in a wider chunk. It is a serving-surface fix, not an engine fix.

## Which keeps "one numeric program per request" by construction

(a). After it, every prime call of 16 or more rows runs the same per-row program for these projections (the dp4a
`grid.y = m` kernel), so a restored suffix equals the cold prime for every suffix length the restore protocol admits,
and a cold prime is chunk-invariant at width 16, by construction of the dispatch rather than by a schedule that
avoids the width. The bit-identity gate is the continuation gate with the 16-row split as a verdict (all splits `ok`
at the one-call digest) plus the two-scope walk (`scope=prime`: 0 of 497 tensors differ at 16 vs 17). (b) only
forbids the transition in serving; its gate can only show the 16-row segment never happened, not that it would have
been identical.

## Decision

Land (a) on the lane (done, `89c693be0`): bounded, in the pattern `matmul` already uses for intent-keyed dispatch
(the `verify_exact` scope), no door (door hygiene, owner 2026-09-05: the fix is the naked default; the rollback is
`git revert`; `MEMRA_NO_BATCHED` stays what it was, the batched-verify A/B reference). Do not land (b): with (a) in
place a 16-row segment is bit-identical and the #602 capture law can stay at "at least PRIME_MIN_T". The target-card
continuation table, `run-gen`, `run-spec` and the twin gate on both cards are the integ's before this reaches
`main`; `docs/TESTING.md` should list the continuation gate's 16-row split as a verdict once it does.

Receipts: `research/spill-b-20260919/DAY22.md` (the pre-registration, the verbatim lines, the fix cells),
`rtx5090-day22/{seam,width-walk,fix-walk,fix-gate,fix-kc,fix-hitgate}/`, runners `run-day22-*.sh`, builds
`rtx5090-day22/{build-tip,build-walk-fmt,build-fix}/`.

## Superseded by #614 (2026-09-21, lead ruling 26)

Main merged #614 (`653c997f4`, 19:46Z, another session) while this lane's day 23 was running: the same defect fixed
by a different mechanism. `Engine::small_m_tier_max()` returns 16 inside the verify-exact scope and
`PRIME_MIN_T - 1` (`.min(16)`, the batched family's hard stop) outside it; `matmul` and `matmul_pre` use it as the
batched small-m tier's ceiling (`(2..=self.small_m_tier_max()).contains(&m)`); `matmul_decode_exact*` keep their own
`2..=16` tiers because they ARE the decode/verify class. Main's `qwen-a4-continuation-gate` counts a differing 16-row
tail as a failure and runs in `tools/local-ci.sh` (`MEMRA_CI_CONTGATE`, `MEMRA_CI_CONT_MODEL`).

**Both mechanisms.** The lane's shape (a) as built (`89c693be0`): a `prefill_rows` RAII scope armed at the top of
`HybridModel::prime_layers` and `batched_tier_admits() = verify_exact_on() || !prefill_rows_on()` on the tier in
`matmul` and `matmul_pre`. Main's: the tier's upper bound itself, written in terms of the prime floor. Both leave a
16-row prime call on the `grid.y = m` dp4a program its wider siblings run; both keep the tier for verify-exact and for
the decode-exact classes; neither adds a kernel, a flag or a numeric program.

**Why main's is the more general one.** The lane's scope covered exactly the callers that armed it. `prime_layers`
is the per-chunk layer walk of the qwen/hybrid prime, and the lane STATED the gap: `prime_layers_gemma` and
`step35_prime_cache_batch` do not go through it. Main's bound sits at the tier, so it holds for every caller of the
general entries, present and future, with no site to forget; and because the bound is `PRIME_MIN_T - 1`, the prime
floor and the tier ceiling cannot drift back onto each other. The scope was therefore a second mechanism for the same
behaviour and was removed on the lane in the merge (`180ca38aa`): the field, `prefill_rows_on`, `prefill_rows_scope`,
`batched_tier_admits`, the two admission conjuncts, the `hybrid_forward.rs` arm. `qwen-a4-width-walk` stays as the
per-operation 16-versus-17 digest diagnostic, one arm (the plain program: exactly the calls the prime walk makes).

**The scope gap, read at the two sites (for the record; closed by construction by #614).**

- `step35_prime_cache_batch` -> `step35_prime_batch_layers(.., ts, ..)`, `let total: usize = ts.iter().sum();`, then
  per attention layer `e.matmul_group(&[&fa.wq, &fa.wk, &fa.wv, gate_w], &h, total)` (or `matmul_group_xh` when
  `pp_f16_enabled() && total >= 16`; a weight without an f16 mirror falls to `matmul` there as well) ->
  `matmul(w, x, m = total)`. Every `ts[s] >= PRIME_MIN_T` is asserted, so at B = 1 a 16-token prompt or a 16-row
  continuation chunk gives `total = 16`. `attn_gate.weight` is `[n_embd, n_head_l]` with `n_head_l` = 64 (full) /
  96 (SWA) on Step-3.7-Flash (`hybrid.rs`, the field's doc), `out_f < GEMM_MIN_OUT_F = 128`: under the GEMM floor
  and, quantized in a b16-admitted qtype, in the batched tier at m = 16 before #614. So a 16-row prime chunk COULD
  reach the tier there, the lane's scope did not cover that site, and #614 does. No Step-3.7-Flash GGUF exists on
  either rig (locally only the HF safetensors `Step-3.7-Flash` and `-FP8`), so the step35 continuation split
  (16 versus 17 rows) is owed to a rig that holds one; the step35 `kernel-check` manifest is model-free and ran on
  both cards (DAY23.md).
- `prime_layers_gemma` -> `gemma4_attn_prime(e, fa, il, &h, pos_d, t, ..)` -> `e.matmul(&fa.wq, h, t)`,
  `e.matmul(&fa.wk, h, t)`, `e.matmul(&fa.wv, h, t)`, `e.matmul(&fa.wo, .., t)`; the FFN through
  `gemma4_layer_tail_add` -> `gemma4_layer_tail_core`: `matmul_q4_fused2_batched(ffn_gate, ffn_up, .., t)` or
  `matmul(ffn_gate, .., t)` + `matmul(ffn_up, .., t)`, then `matmul_pre(ffn_down, .., t)` / `matmul(ffn_down, .., t)`.
  m = t, and t = 16 is a legal chunk. The gemma4 packs' quantized projections are wq `[query_heads * head_dim]`,
  wk/wv `[kv_heads * head_dim]`, wo, gate/up/down `[intermediate]`; q_norm/k_norm are `FloatOnly` `[head_dim]`
  vectors; the MoE router is `[n_expert = 128]`. On the gemma-4 artifacts every quantized `out_f >= 128`, so at
  m = 16 the GEMM arm takes them and the tier is not reached at any width: the site was width-safe by the artifact
  census, not by construction. #614 makes it so by construction for any gemma variant with a small projection. A
  gemma continuation-split tool does not exist (the continuation gate is the qwen/hybrid prime path); not run.

**What the lane's gates guard now.** The width walk, the seven-arm continuation table, `kernel-check`, the #379 hit
gate and the twin gate re-ran on the merged tree (main's mechanism, no scope): `DAY23.md`. The day-22 receipts on the
lane's mechanism stand as evidence for the defect and its placement; they are not receipts for main's code.
