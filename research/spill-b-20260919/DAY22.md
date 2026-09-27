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

### 1.5 Resumed: what the dead run finished, and the fix cells (a) to (f) verbatim

**Resumed.** The agent process that ran the fix cells died mid-run (API outage; two later restarts died before
doing anything). Read back from the receipts before acting: the fix commit `89c693be0` landed at 17:12:29Z with
cells S and W and the fix build (`build-fix/`, 17:09:51Z to 17:10:41Z, exit 0). The dead run then ran F1 as
`fix-walk` and hit the canonical lock busy four times (`fix-walk-retries.log`: 17:13:01, 17:14:56, 17:16:52,
17:18:46, each a 90 s sleep; attempts 0..3 have `.exit` 2 and no `cell/`, the collector refused with
`[Errno 11] Resource temporarily unavailable`; the holder was not this lane's and was not inspected), then
`fix-walk-retry4` (17:20:16Z, exit 0), `fix-gate` (17:20:51Z, exit 0), `fix-kc` (17:28:48Z, exit 0) and
`fix-hitgate` (17:32:21Z to 17:32:49Z, exit 0). It had NOT written this section, had not committed the receipts or
`PRIME-MIN-T-DECISION.md`, had not run the twin gate, the base-versus-fix cold-prime check, or anything on the
target card. On resume (19:02Z): no process of the dead run alive (checked by cwd), the 5090 lock free, the card
alone; `origin/main 5804cac6a` (#612, integ23) merged as `051bbfee3` (no file under `crates/` moved, so the
`build-fix` binaries are the merged tree's engine byte for byte); the receipts and the decision note committed as
they were (`de4fb332b`, `wip:`), pushed in the announced development mode (`UNQUALIFIED DEVELOPMENT` at
`de4fb332b`, logged in `.git/memra-gate-skips.log`; no qualification claimed).

Every cell below ran on the `build-fix` binaries (SHA-256 `f7daf4a6...` continuation gate, `b3be1e62...` width
walk, `5a125d00...` kernel-check, `9d0eb4ff...` memra-server), the served `Qwen3.8-27B-NVFP4-Q5K-mtp.gguf`
(`1facf36c...`), prompt `docs/SERVING.md`, under the collector on the local RTX 5090 Laptop GPU
(`executed-not-qualified`), `nvidia-smi --query-compute-apps` empty before and after each.

**(a) The 16-row versus 17-row per-operation identity at the named site, fixed tree (`fix-walk-retry4/`, F1).**
Summary lines, verbatim:

```text
WIDTH WALK scope=prime width 16 vs 17: 0 of 497 tensors differ; tensor names: {}; sites: []
WIDTH WALK scope=prime width 48 vs 17: 0 of 497 tensors differ; tensor names: {}; sites: []
WIDTH WALK scope=bare width 16 vs 17: 96 of 497 tensors differ; tensor names: {"ssm_alpha", "ssm_beta"}; sites: ["00/ssm_beta", "00/ssm_alpha", "01/ssm_beta", ..., "62/ssm_beta", "62/ssm_alpha"]
WIDTH WALK scope=bare width 48 vs 17: 0 of 497 tensors differ; tensor names: {}; sites: []
```

The named site, first GDN layer, both scopes:

```text
scope=prime layer 00 ssm_beta  qtype=NVFP4   in_f=5120   out_f=48     width  16 vs 17: rows_differ=0/16 maxabs=0.000e0 ref_sha=e40f0aeaaec76762 sha=e40f0aeaaec76762 same
scope=prime layer 00 ssm_alpha qtype=NVFP4   in_f=5120   out_f=48     width  16 vs 17: rows_differ=0/16 maxabs=0.000e0 ref_sha=27a9e796e65eee50 sha=27a9e796e65eee50 same
scope=bare  layer 00 ssm_beta  qtype=NVFP4   in_f=5120   out_f=48     width  16 vs 17: rows_differ=16/16 maxabs=1.490e-7 ref_sha=e40f0aeaaec76762 sha=0a984de6e2ed5fd0 DIFFERS
scope=bare  layer 00 ssm_alpha qtype=NVFP4   in_f=5120   out_f=48     width  16 vs 17: rows_differ=16/16 maxabs=4.768e-7 ref_sha=27a9e796e65eee50 sha=c9c9a51efce71c75 DIFFERS
```

Under the prime's scope the 16-row output equals the 17-row output's leading rows at every one of the 497 tensors
(the `ref_sha` is the same at both scopes: the 17-row program did not move); the bare (decode/verify) class is
exactly the day-22 cell W result. F1's prediction held in every clause.

**(b) The #427 table on the fixed tree (`fix-gate/`, F2), all seven arms, verbatim (the `[prime-row]` receipts are
in `fix-gate/cell/arm-*.log`):**

```text
== arm F-9296 total=9296 tails=16 48 80 112 144 176 208 env:
one call over 9296 tokens: logits_sha=14ab5f8b365dbd71
  9280 + 16: logits_sha=14ab5f8b365dbd71 ok
  9248 + 48: logits_sha=14ab5f8b365dbd71 ok
  9216 + 80: logits_sha=14ab5f8b365dbd71 ok
  9184 + 112: logits_sha=14ab5f8b365dbd71 ok
  9152 + 144: logits_sha=14ab5f8b365dbd71 ok
  9120 + 176: logits_sha=14ab5f8b365dbd71 ok
  9088 + 208: logits_sha=14ab5f8b365dbd71 ok
A4 CONTINUATION GATE: PASS
== arm F-9297 total=9297 tails=17 49 env:
one call over 9297 tokens: logits_sha=e641952cac19e526
  9280 + 17: logits_sha=e641952cac19e526 ok
  9248 + 49: logits_sha=e641952cac19e526 ok
A4 CONTINUATION GATE: PASS
== arm F-9311 total=9311 tails=31 63 env:
one call over 9311 tokens: logits_sha=b61719f294866b05
  9280 + 31: logits_sha=b61719f294866b05 ok
  9248 + 63: logits_sha=b61719f294866b05 ok
A4 CONTINUATION GATE: PASS
== arm F-9312 total=9312 tails=32 64 96 env:
one call over 9312 tokens: logits_sha=736a88d7c7448dcb
  9280 + 32: logits_sha=736a88d7c7448dcb ok
  9248 + 64: logits_sha=736a88d7c7448dcb ok
  9216 + 96: logits_sha=736a88d7c7448dcb ok
A4 CONTINUATION GATE: PASS
== arm F-9296-chunk32 total=9296 tails=16 48 env: MEMRA_PRIME_CHUNK=32
one call over 9296 tokens: logits_sha=14ab5f8b365dbd71
  9280 + 16: logits_sha=14ab5f8b365dbd71 ok
  9248 + 48: logits_sha=14ab5f8b365dbd71 ok
A4 CONTINUATION GATE: PASS
== arm F-9296-chunk16 total=9296 tails=16 48 env: MEMRA_PRIME_CHUNK=16
one call over 9296 tokens: logits_sha=bafe0e0a09a3d0a4
  9280 + 16: logits_sha=bafe0e0a09a3d0a4 ok
  9248 + 48: logits_sha=bafe0e0a09a3d0a4 ok
A4 CONTINUATION GATE: PASS
== arm F-9296-nobatched total=9296 tails=16 48 env: MEMRA_NO_BATCHED=1
one call over 9296 tokens: logits_sha=14ab5f8b365dbd71
  9280 + 16: logits_sha=14ab5f8b365dbd71 ok
  9248 + 48: logits_sha=14ab5f8b365dbd71 ok
A4 CONTINUATION GATE: PASS
```

Every prediction of the F2 row held: the one-call digests of 9296, 9297, 9311 and 9312 are day 21's, the 16-row
split reads `ok` at the one-call digest with the gate now counting it, the chunk-32 one-call digest moved from
`35bd15f063bfd5ba` to the default schedule's `14ab5f8b365dbd71` (the cold prime is chunk-invariant at width 16), the
chunk-16 arm is a new one-call digest (`bafe0e0a09a3d0a4`, every chunk 16 rows on the dp4a program, off the 32 grid
so not comparable to the default) with both splits `ok`, and `MEMRA_NO_BATCHED=1` is identical to the default. On
"both artifacts": day 21 ran ONE artifact (the served mint) on both cards; the calibrated A4 artifact of #427's
table is on neither this rig nor the target box, so the table stands on the served mint, both cards (section 2 for
the target card).

**(c) `kernel-check` on the fixed tree (`fix-kc/`, F3), both manifests
(`--require-manifest tools/kernel-check-27b.cells --require-manifest tools/kernel-check-step35.cells`):**
`ALL GREEN (109 cells, 10 skipped)`, exit 0. The 10 skips are the rig's absent models
(`Qwen3.6-35B-A3B-UD-IQ4_XS.gguf` x7 cells, `gemma-4-12b-it-qat-q4_0.gguf`, `Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf`)
and `sigrouter-served-replay` (no capture set), within local-ci's budget of 11.

**(d) The #379 hit gate (`fix-hitgate/`, F4), `tools/spec-on-cache-hit-gate.sh qwen` on the fix `memra-server`
(`9d0eb4ff...`), 9B trunk `Qwen3.5-9B-NVFP4-MTP-GGUF.gguf`, 17:32:21Z to 17:32:49Z:**
`SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)`, exit 0. Recorded, not hidden: the external drafter local-ci attaches
(`draft-9b-owntrim-nvfp4head-q4blk.gguf`) is absent on this rig, so the gate ran as local-ci's WARNING branch
(`fix-hitgate/drafter.txt`), the same shape as day 19's run on the pre-fix tree. The gate takes the canonical lock
itself (`MEMRA_GPU_LOCK=/tmp/memra-5090.lock`), so it did not go through the collector.

**(e) The twin gate (`fix-twin/`, `tools/prefix-newest-turn-fits-gate.py` on the fix `memra-server`, default 8-turn
shape, LRU default, collector-locked, 19:06Z, exit 0), verbatim:**

```text
PREFIX-NEWEST-TURN-FITS: budget_bytes=1073741824 cohort_bytes=736755712 turns=8 cold_turns_after_1=0 cached_ok=7/7 lines_ok=8/8 evictions=9 cohort_evictions=3 self_evictions=0 refused_or_skipped=0 effective_free_ok=8/8 identity_ok=8/8 grid_ok=21/21 grid=32 off_grid_calls=0 V1=ok V2=ok V3=ok V4=ok V5=ok V6=ok -> PASS
```

Byte for byte day 21's line on the pre-fix tree (the same bytes, the same eviction counts; V5 is the restored render
equal to the cache-off render on every turn).

**(f) Cold-prime bit-identity, base tree versus fixed tree (`build-base/`, `base-gate/`; runner
`run-day22-base.sh`).** The base binary is the merged tree with exactly the fix's three source hunks reverted
(`git diff 89c693be0 99c939f6a` on `lib.rs`, `hybrid_forward.rs`, `qwen_a4_continuation_gate.rs`, applied and
recorded in `build-base/reverted-hunks.txt`: 3 files, 11 insertions, 51 deletions; nothing else), built 19:11:07Z to
19:11:41Z, SHA-256 `61184681...`; the tree was restored (`git checkout -- crates/`, clean) and the `build-fix`
binaries put back with their SHA-256 re-read equal. Same artifact, same prompt, same rig, three arms, verbatim:

```text
== arm B-9296 total=9296 tails=16 48 env:
one call over 9296 tokens: logits_sha=14ab5f8b365dbd71
  9280 + 16: logits_sha=35bd15f063bfd5ba DIFFERS (known: a 16-row final segment is not bitwise on either artifact)
  9248 + 48: logits_sha=14ab5f8b365dbd71 ok
== arm B-9296-chunk32 total=9296 tails=16 48 env: MEMRA_PRIME_CHUNK=32
one call over 9296 tokens: logits_sha=35bd15f063bfd5ba
  9280 + 16: logits_sha=35bd15f063bfd5ba ok
  9248 + 48: logits_sha=35bd15f063bfd5ba ok
== arm B-9297 total=9297 tails=17 env:
one call over 9297 tokens: logits_sha=e641952cac19e526
  9280 + 17: logits_sha=e641952cac19e526 ok
```

Read against `fix-gate/`:

| schedule | base one-call | fix one-call | reading |
| --- | --- | --- | --- |
| 9296, default chunk (2 x 4096 + 1104; no 16-row chunk) | `14ab5f8b365dbd71` | `14ab5f8b365dbd71` | identical: the fix moved nothing here |
| 9296, `MEMRA_PRIME_CHUNK=32` (290 x 32 + a 16-row chunk) | `35bd15f063bfd5ba` | `14ab5f8b365dbd71` | differs by design; the new value IS the wide-chunk digest |
| 9297, default chunk (17-row remainder) | `e641952cac19e526` | `e641952cac19e526` | identical |
| 9296 restored as 9280 + 16 | `35bd15f063bfd5ba` | `14ab5f8b365dbd71` | the restored 16-row suffix now digests as the one-call prime |

So the fix moved exactly the 16-row chunk program and only that: a schedule without a 16-row chunk is bit-identical
to the base, and the one with a 16-row chunk now equals the wide program. (a) to (f) all hold; the fix stands on the
lane as `fix:` (commit `89c693be0`), still `executed-not-qualified` everywhere, the target-card table in section 2.

## 2. Target card: the table and `kernel-check` on the fixed tree (one RTX PRO 6000 Blackwell, `pro-single-day22/`)

The box was free on resume (`ssh -O check` master running, `/tmp/memra-gpu.lock` free, no compute app). `/root/wt-b`
fetched and reset to the lane tip `de4fb332b` (the merged tree; no `crates/` file differs from `89c693be0`), built on
the box (`build-fix/`, 3 m 13 s, exit 0; the first attempt `build-fix-attempt1-no-cargo-path/` exited 127 because
the detached shell had no cargo on its PATH and is kept as-is): continuation gate SHA-256 `acbe9414...`,
`kernel-check` `d61a41b5...`, artifact `Qwen3.8-27B-NVFP4-Q5K-mtp.gguf` `1facf36c...` (the same bytes as the local
copy). Every cell through `tools/tier-battery.py --rig pro-single` (`executed-not-qualified`), pass/fail, not timed,
`nvidia-smi --query-compute-apps` empty before and after; nothing here is compared in time with the local card.

**The #427 table (`fix-gate/`, `run-day22-cont-box3.sh`, the seven local arms byte for byte, 291.8 s wall, exit
0).** Every line equals the local `fix-gate/` line of section 1.5 (b), digest for digest:

```text
F-9296:            one call 14ab5f8b365dbd71; 9280 + 16 ok; 9248 + 48 ok; 9216 + 80 ok; 9184 + 112 ok; 9152 + 144 ok; 9120 + 176 ok; 9088 + 208 ok; A4 CONTINUATION GATE: PASS
F-9297:            one call e641952cac19e526; 9280 + 17 ok; 9248 + 49 ok; PASS
F-9311:            one call b61719f294866b05; 9280 + 31 ok; 9248 + 63 ok; PASS
F-9312:            one call 736a88d7c7448dcb; 9280 + 32 ok; 9248 + 64 ok; 9216 + 96 ok; PASS
F-9296-chunk32:    one call 14ab5f8b365dbd71 (MEMRA_PRIME_CHUNK=32); 9280 + 16 ok; 9248 + 48 ok; PASS
F-9296-chunk16:    one call bafe0e0a09a3d0a4 (MEMRA_PRIME_CHUNK=16); 9280 + 16 ok; 9248 + 48 ok; PASS
F-9296-nobatched:  one call 14ab5f8b365dbd71 (MEMRA_NO_BATCHED=1); 9280 + 16 ok; 9248 + 48 ok; PASS
```

(The full lines, with `logits_sha=` on every split, are `fix-gate/command.log`; the `[prime-row]` receipts are
`fix-gate/cell/arm-*.log`.) Two card classes, one artifact, one fixed tree: every split of every arm at the one-call
digest, and the one-call digests byte-identical across the cards, as on day 21 for the pre-fix digests.

**`kernel-check` (`fix-kc/`, `run-day22-kc-box3.sh`, both manifests, `MEMRA_KC_MODELS_DIR=/root/artifacts`,
12.1 s wall, collector exit 0).** Stated plainly, not ALL GREEN: 362 cells `OK`, 0 `FAIL`, 15 `SKIP` (the box holds
only the two artifacts; skipped for want of `Qwen3.5-9B-NVFP4-MTP-GGUF.gguf` x10 cells, `ornith-1.0-35b-Q4_K_M.gguf`,
`Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.gguf`, `gemma-4-12b-it-qat-q4_0.gguf`, `gemma-4-26B_q4_0-it.gguf`,
`Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf`, plus `sigrouter-served-replay`), and the run ended
`MISSING REQUIRED CELL DUAL-BATCHED-AUX` / `Error: "1 required cell(s) missing"` / `exit=1`: the 27b manifest's one
required cell needs the 9B artifact, which is not on the box. That is the manifest doing its job on a box without the
model, not a red cell; the local run (section 1.5 (c)) has that cell green. A second cell requiring the step35
manifest alone (`fix-kc-step35`, `MANIFESTS=tools/kernel-check-step35.cells`, the runner parametrized for it) was
queued behind the table and never ran: the canonical lock was held by another lane from 19:20:59Z through the
runner's sixth attempt at 19:28:29Z (`fix-kc-step35-retries.log`, six `REFUSED: [Errno 11] Resource temporarily
unavailable` driver logs, no `cell/`); the holder was not inspected and the runner exited on its bound. The
target-card `kernel-check` verdict line is therefore owed to the integ (or to a rerun when the box is free), with the
9B artifact staged or the step35 manifest alone.

## 3. memra#445 past section 1: the map

Posted as a comment on #445 (no cell, no code). Section 1 is closed by #588 (`f4350c241`, gate `b351d7db9`, day 13).
Section 2 (prefix capture on the gemma spec route): open; #561/#564 gave gemma the generic continuation-capable
prime (memra#535 P1a), but the capture sites stay keyed on the eager-only class because gemma's prefix snapshot is
refused by the SWA flat-history layout (memra#151, open) and its plain-affinity resume has no gate
(`worker.rs`, the boundary-stop block); needs #151's ring-aware snapshot and restore with a bit-identity gate first,
then the three capture sites under the #602 capture law, then the hit gate and twin gate on a gemma trunk. Section 3
(spec under concurrency): the shared mechanism moved (#266 closed, the resumed-carrier K floor, the sampled wave,
#429 closed on qwen35), none of it measured with the gemma drafter and no INDEX row for the gemma4 full-serving lane;
needs one pre-registered c=1/4/8 ladder on the target card reading the `[spec-k]` source lines, N>=5 both orders,
plain batched arm beside it.

## 4. Close of day

Checks on the final tree: `cargo fmt --all -- --check` ok, `git diff --check` ok, `tools/check-flags.sh` (no
uncovered runtime names), `python3 tools/check-public-boundary.py check` (0 new), `cargo clippy -p memra-engine
--all-targets -- -D warnings` under the CPU quota (exit 0, `rtx5090-day22/checks/`). Scratch: the two comment
drafts under `/tmp` removed after posting; the reverse patch removed after the base build; `target/bins/day22-*`
hold the base and fix binaries for the receipts' SHAs (untracked build products, as `target/bins/{base,fix}` since
day 18). Nothing merged, no PR; the lead integrates. Budget: about 1.1 agent-hours on the resumed run (19:02Z to
about 20:05Z) on top of the dead run's roughly 0.75 h of fix cells.
