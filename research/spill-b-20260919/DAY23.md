# WP-B day 23: the memra#427 battery re-run on main's mechanism (#614), the lane's scope dropped, target-card `kernel-check`, `run-spec` and `run-gen` receipts

Repository: **avifenesh/memra**, branch `lane/spill-b-20260919`. Two merges today: `origin/main d6401ff70` (#613,
integ24; merge `aeb4d0771`, no `crates/` change) at the start, and `origin/main 653c997f4` (#614) after lead ruling 26
(merge `180ca38aa`, section 2). Every push of the day is refused `UNQUALIFIED` by the #589 hook on a plain push and
runs as `MEMRA_RELEASE_QUALIFICATION_MODE=development git push` (`UNQUALIFIED DEVELOPMENT`, logged in
`.git/memra-gate-skips.log`). **No qualification is claimed anywhere in this record**; every collector cell is
`executed-not-qualified`, pass/fail, not timed; no timing is compared across the two cards.

Rig discipline as on every prior day: every GPU command on the local RTX 5090 Laptop GPU goes through
`tools/tier-battery.py --rig rtx5090` (`/tmp/memra-5090.lock`; the #379 hit gate takes the same lock itself),
CPU-heavy work runs under `systemd-run --user --scope -p CPUQuota=1200% -p MemoryMax=28G`, `nvidia-smi
--query-compute-apps` is read before and after every cell (empty every time today), and no process this lane did not
start is signalled or inspected. On the target card (one RTX PRO 6000 Blackwell, the `pro-single` rig profile,
`/tmp/memra-gpu.lock`, socket `ssh -O check` first) every cell runs from `/root/wt-b` through the collector; the lane's
receipts live under `/root/spill-receipts/b-day23/` and are mirrored here under `pro-single-day23/`.

## 1. Before ruling 26 (on the lane's day-22 mechanism, tip `aeb4d0771`)

**The 9B artifact staged for the 27b manifest's one required cell.** Day 22's target-card `kernel-check` ended
`MISSING REQUIRED CELL DUAL-BATCHED-AUX` because the cell needs `Qwen3.5-9B-NVFP4-MTP-GGUF.gguf`, absent on the box.
A verified copy exists on this rig (`/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/`, 5.7 GB): its SHA-256
`52c9cceb190055e0591a9a30c21f7200572eaf3ff1c59f6e9a1eda838a8f39de` equals the value recorded in five receipts already
in the repo (`research/dualpp1-20260811/raw/*/correctness/SHA256SUMS`,
`research/b200-kernel-twins-dry-20260901/receipts/*/{artifact,q9-artifact}.sha256`). It was streamed over the existing
ssh master (20 MB/s, about five minutes) into this lane's own receipts dir `/root/spill-receipts/b-day23/artifacts/`
and hashed there: the same `52c9cceb...` (`Qwen3.5-9B-NVFP4-MTP-GGUF.gguf.sha256`). `/root/artifacts` was not touched:
`MEMRA_KC_MODELS_DIR` takes one directory, so `/root/spill-receipts/b-day23/models/` holds symlinks to the two
`/root/artifacts` files plus the staged 9B (`kernel-check` resolves a candidate with `Path::exists`, which follows
them). Neither manifest nor the checker was edited.

**Target-card `kernel-check`, day-22 binary (`d61a41b5...`, source `aeb4d0771`; runner
`pro-single-day23/run-day23-kc-box3.sh`, cells `kc-step35` and `kc-full`, both exit 0, both before the ruling
arrived, banked as-is).** Verbatim verdict lines:

```text
kc-step35 (MANIFESTS=tools/kernel-check-step35.cells):                      ALL GREEN (109 cells, 6 skipped)   exit=0
kc-full   (tools/kernel-check-27b.cells tools/kernel-check-step35.cells):   ALL GREEN (109 cells, 6 skipped)   exit=0
DUAL-BATCHED-AUX [NVFP4 rp] out=48 m=3: bit-bad=0/0 OK
```

The six skips, both cells: `sigrouter-served-replay` (no capture set), `f16g-kq-direct-ornith`
(`ornith-1.0-35b-Q4_K_M.gguf`), `iq4xs-mmq-real` (`Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.gguf` or the Step IQ4_XS
shards), `q4_0-mmq` (`gemma-4-12b-it-qat-q4_0.gguf`), `nvfp4-27b-shape` (`Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf`),
`q4_0-sk-arm` (`gemma-4-26B_q4_0-it.gguf`), all absent on the box. So the day-22 verdict line is closed: with the one
missing artifact staged the full manifest set is `ALL GREEN` on the target card. These two cells ran the lane's
day-22 mechanism; section 3 repeats them on main's.

**The two prompts.** `tok-check` (the memra tokenizer, `encode(text, bos = true)`, the same call the continuation
gate makes) on the first 100,000 bytes of `docs/SERVING.md` gives 27,632 ids (`rtx5090-day23/prompts/`,
`README.txt` carries the head's SHA-256). `p16-ids.txt` is the first 16 ids (`2 69208 1892 279 5097 15015 7099 321 279
35729 24303 271 1919 369 279 8418`: 16 tokens). `p4112-ids.txt` is the first 4112 ids (4112 = 4096 + 16: the cold
prime's default schedule is one 4096-row chunk and one 16-row chunk, the shape memra#427 named). Both are passed to
`run-gen` and `run-spec` as raw ids, so the token count is exact by construction.

## 2. Lead ruling 26: #614 on main, the lane's scope dropped

PR #614 (`653c997f4`, 19:46Z, another session) fixes memra#427 at the tier: `Engine::small_m_tier_max()` is 16 inside
the verify-exact scope and `PRIME_MIN_T - 1` outside it, and `matmul`/`matmul_pre` bound the batched small-m tier
with it; `matmul_decode_exact*` keep their own `2..=16`. It covers every caller of the general entries, so the scope
gap named on day 22 (`prime_layers_gemma`, `step35_prime_cache_batch`) is closed by construction. The lane's
`prefill_rows` scope was a second mechanism for the same behaviour: removed in the merge `180ca38aa` (the field,
`prefill_rows_on`, `prefill_rows_scope`, `batched_tier_admits`, the two admission conjuncts in `lib.rs`, the
`_prefill_rows_scope` arm in `hybrid_forward.rs`); `qwen-a4-continuation-gate` is main's version (it counts a
differing 16-row tail); `qwen-a4-width-walk` is reduced to the one-arm plain program (the calls the prime walk makes,
no scope) and kept as the per-operation diagnostic. Main's `docs/TESTING.md` paragraph on the continuation gate
gained one pointer to the width walk (a `memra-engine` bin, not a `tools/` script). Against `origin/main` the lane's
tracked non-research diff is exactly that: the width-walk bin (plus its `Cargo.toml` `[[bin]]` entry) and the
TESTING.md sentence. The full record, both mechanisms and the two sites read: `PRIME-MIN-T-DECISION.md`, section
"Superseded by #614".

Build receipts on the merged tree: local `rtx5090-day23/build-merged/` (exit 0; `run-spec 857da18b...`, `run-gen
8332b712...`, `qwen-a4-continuation-gate 9d7f83f6...`, `qwen-a4-width-walk 40b33f7e...`, `kernel-check a2759a1c...`,
`memra-server cbb7d4ad...`); target card `/root/spill-receipts/b-day23/build-merged/` (exit 0). Clippy on the touched
crate under the quota: `cargo clippy -p memra-engine --all-targets -- -D warnings` exit 0
(`rtx5090-day23/checks/clippy.log`).

## 3. Local RTX 5090, merged tree `5f2060381` (main's mechanism, no scope): the five day-22 cells and run-gen/run-spec

One serial chain (`rtx5090-day23/chain-day23-local.sh`, `chain.log`, 19:58:59Z to 20:17:52Z), every cell exit 0 on
the `build-merged` binaries, the served mint, prompt `docs/SERVING.md` where the gate takes one. (`gate-source.txt` in
a cell names the lane HEAD at the time the cell ran, which moved by docs commits during the chain; the binary SHA-256 in
each cell is the build at `5f2060381`, and no `crates/` file changed after it.)

**(a) Width walk (`walk/`, `run-day23-walk.sh walk 1800 17 16 48`, one arm, no scope), verbatim:**

```text
WIDTH WALK width 16 vs 17: 0 of 497 tensors differ; tensor names: {}; sites: []
WIDTH WALK width 48 vs 17: 0 of 497 tensors differ; tensor names: {}; sites: []
layer 00 ssm_beta  qtype=NVFP4   in_f=5120   out_f=48     width  16 vs 17: rows_differ=0/16 maxabs=0.000e0 ref_sha=e40f0aeaaec76762 sha=e40f0aeaaec76762 same
layer 00 ssm_alpha qtype=NVFP4   in_f=5120   out_f=48     width  16 vs 17: rows_differ=0/16 maxabs=0.000e0 ref_sha=27a9e796e65eee50 sha=27a9e796e65eee50 same
```

The prediction of ruling 26 held: 0 of 497 at 16 vs 17 with no scope (the `ref_sha` of the 17-row program is day
22's, so #614 moved no wide call), 994 `same` lines, 0 `DIFFERS`. The day-22 `scope=bare` arm that read 96 differing
tensors was the tier at m = 16; under #614 that tier is not admitted at m = 16 outside verify-exact, so there is no
second arm to run.

**(b) The seven-arm continuation table (`gate/`, `run-day23-gate.sh gate 2400`), verbatim:**

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

Digest for digest the day-22 fix table (DAY22.md 1.5 (b)) and the day-22 base table where the two agree
(1.5 (f)): main's mechanism and the lane's produce the same bits at every split, and the chunk-32 one-call prime
digests as the default schedule (`14ab5f8b365dbd71`, not the pre-fix `35bd15f063bfd5ba`).

**(c) `kernel-check`, both manifests (`kc/`, `run-day23-kc.sh kc 1800`):** `ALL GREEN (109 cells, 10 skipped)`,
exit 0; the same ten skips as day 22 (the rig's absent `Qwen3.6-35B-A3B-UD-IQ4_XS.gguf` x7, `gemma-4-12b-it-qat-q4_0.gguf`,
`Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf`, `sigrouter-served-replay`), within local-ci's budget of 11.

**(d) The #379 hit gate (`hitgate/`, `tools/spec-on-cache-hit-gate.sh qwen` on the merged `memra-server`
`cbb7d4ad...`, 9B trunk):** `ok: g1 spec==plain byte identity`, `ok: g2 spec==plain byte identity`,
`SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)`, exit 0. As on days 19 and 22 the external drafter local-ci attaches is
absent on this rig (`drafter.txt`; local-ci's WARNING branch), recorded, not hidden.

**(e) The twin gate (`twin/`, `tools/prefix-newest-turn-fits-gate.py`, default 8-turn shape, LRU default), verbatim:**

```text
PREFIX-NEWEST-TURN-FITS: budget_bytes=1073741824 cohort_bytes=736755712 turns=8 cold_turns_after_1=0 cached_ok=7/7 lines_ok=8/8 evictions=9 cohort_evictions=3 self_evictions=0 refused_or_skipped=0 effective_free_ok=8/8 identity_ok=8/8 grid_ok=21/21 grid=32 off_grid_calls=0 V1=ok V2=ok V3=ok V4=ok V5=ok V6=ok -> PASS
```

Byte for byte days 21 and 22.

**(f) `run-gen` argmax and `run-spec` K=1..8 (`genspec/`, `run-day23-genspec.sh genspec 3000`; the same arms as
section 4's target-card cell), verbatim:**

```text
run-gen std   (9419 11 1814 0):  prefill argmax=271  decode argmax=271  logit maxdiff=2.276e-1  MATCH
run-gen probe (90 tokens):       prefill argmax=15666  decode argmax=15666  logit maxdiff=5.047e-1  MATCH
                                 batched-prime argmax=15666  tokenwise argmax=15666  logit maxdiff=3.523e-1  MATCH
run-gen p16   (16 tokens):       prefill argmax=23314  decode argmax=23314  logit maxdiff=2.965e-1  MATCH
                                 batched-prime argmax=23314  tokenwise argmax=23314  logit maxdiff=2.392e-1  MATCH
run-gen p4112 (4112 tokens):     prefill argmax=84728  decode argmax=84728  logit maxdiff=4.292e-1  MATCH
                                 batched-prime argmax=84728  tokenwise argmax=84728  logit maxdiff=3.407e-1  MATCH
run-spec probe (90 tokens):  acceptance 12/19, 19/28, 19/36, 22/48, 21/55, 21/66, 21/77, 21/88 for K=1..8, each
                             `self-consistency: PASS (identical to plain target)`; `=== SELF-CONSISTENCY PASS ===`
run-spec p16 (16 tokens):    acceptance 14/18, 18/30, 21/39, 21/52, 21/65, 21/78, 21/91, 21/104 for K=1..8, each PASS; `=== SELF-CONSISTENCY PASS ===`
run-spec p4112 (4112 tokens): acceptance 14/17, 17/30, 18/42, 18/56, 18/70, 18/84, 18/98, 18/112 for K=1..8, each PASS; `=== SELF-CONSISTENCY PASS ===`
```

Every argmax, every accepted/drafted count and every verdict equals the target card's (section 4); the only figures
that differ between the cards are the `logit maxdiff` diagnostics on `p4112` (4.292e-1 / 3.407e-1 here against
7.075e-1 / 7.158e-1 there), which the gate does not read as severity (run-gen's own note) and which are not a
verdict. `nvidia-smi --query-compute-apps` empty before and after every cell.

## 4. Target card, merged tree `5f2060381` (one RTX PRO 6000 Blackwell; `pro-single-day23/m-*`, mirrored from `/root/spill-receipts/b-day23/`)

`/root/wt-b` reset to `5f2060381` and built on the box (`build-merged/`, exit 0; `kernel-check cc04dfca...`,
`qwen-a4-continuation-gate 11456b33...`, `run-spec`/`run-gen` per `binary.sha256`); the four cells ran back to back
(`chain.log`, 20:00:38Z to 20:10:36Z) through the collector, the card alone before and after each, artifact
`Qwen3.8-27B-NVFP4-Q5K-mtp.gguf` (`1facf36c...`, the same bytes as the local copy).

**Task 1, `kernel-check` on the merged tree (`m-kc-step35`, `m-kc-full`; runner `run-day23-kc-box3.sh`,
`MEMRA_KC_MODELS_DIR=/root/spill-receipts/b-day23/models`, the 9B staged as in section 1), verbatim:**

```text
m-kc-step35 (MANIFESTS=tools/kernel-check-step35.cells):                     ALL GREEN (109 cells, 6 skipped)   exit=0
m-kc-full   (tools/kernel-check-27b.cells tools/kernel-check-step35.cells):  ALL GREEN (109 cells, 6 skipped)   exit=0
DUAL-BATCHED-AUX [NVFP4 rp] out=48 m=3: bit-bad=0/0 OK
```

The same six skips as section 1 (absent models and the sigrouter capture). The 27b manifest's requirement is met by
the staged artifact, not by editing the manifest or the checker.

**Task 2, `run-spec` K=1..8 self-consistency with the MTP drafter (`m-genspec/cell/spec-*.log`; runner
`run-day23-genspec-box3.sh`; `MEMRA_SPEC_TEMP=0 MEMRA_NGEN=32`, the naked K=1..8 sweep as `tools/local-ci.sh`
runs it; no `MEMRA_MTP_DRAFT`: the mint's own NextN head is the drafter, `loaded qwen35 (65 layers, nextn=1)` in
every log).** Token counts: `probe` = `tools/fast-gate/prompts/probe.txt`, `text prompt (472 chars) -> 90 tokens`;
`p16` = 16 raw ids; `p4112` = 4112 raw ids (16 mod 4096). Verbatim, the per-K lines in order K=1..8:

```text
probe (90 tokens):
  acceptance: 12/19 = 63.2%   self-consistency: PASS (identical to plain target)
  acceptance: 19/28 = 67.9%   self-consistency: PASS (identical to plain target)
  acceptance: 19/36 = 52.8%   self-consistency: PASS (identical to plain target)
  acceptance: 22/48 = 45.8%   self-consistency: PASS (identical to plain target)
  acceptance: 21/55 = 38.2%   self-consistency: PASS (identical to plain target)
  acceptance: 21/66 = 31.8%   self-consistency: PASS (identical to plain target)
  acceptance: 21/77 = 27.3%   self-consistency: PASS (identical to plain target)
  acceptance: 21/88 = 23.9%   self-consistency: PASS (identical to plain target)
=== SELF-CONSISTENCY PASS ===
p16 (16 tokens):
  acceptance: 14/18 = 77.8%   self-consistency: PASS (identical to plain target)
  acceptance: 18/30 = 60.0%   self-consistency: PASS (identical to plain target)
  acceptance: 21/39 = 53.8%   self-consistency: PASS (identical to plain target)
  acceptance: 21/52 = 40.4%   self-consistency: PASS (identical to plain target)
  acceptance: 21/65 = 32.3%   self-consistency: PASS (identical to plain target)
  acceptance: 21/78 = 26.9%   self-consistency: PASS (identical to plain target)
  acceptance: 21/91 = 23.1%   self-consistency: PASS (identical to plain target)
  acceptance: 21/104 = 20.2%   self-consistency: PASS (identical to plain target)
=== SELF-CONSISTENCY PASS ===
p4112 (4112 tokens):
  acceptance: 14/17 = 82.4%   self-consistency: PASS (identical to plain target)
  acceptance: 17/30 = 56.7%   self-consistency: PASS (identical to plain target)
  acceptance: 18/42 = 42.9%   self-consistency: PASS (identical to plain target)
  acceptance: 18/56 = 32.1%   self-consistency: PASS (identical to plain target)
  acceptance: 18/70 = 25.7%   self-consistency: PASS (identical to plain target)
  acceptance: 18/84 = 21.4%   self-consistency: PASS (identical to plain target)
  acceptance: 18/98 = 18.4%   self-consistency: PASS (identical to plain target)
  acceptance: 18/112 = 16.1%   self-consistency: PASS (identical to plain target)
=== SELF-CONSISTENCY PASS ===
```

Every K on every prompt token-identical to the plain target with acceptance > 0 (both halves of the gate). The
`tok/s` figures in those logs are single runs on a pass/fail cell and are not measurements.

**Task 3, `run-gen` argmax gate (`m-genspec/cell/gen-*.log`, `MEMRA_NGEN=8`), verbatim:**

```text
std   (CONTRIBUTING's raw ids 9419 11 1814 0, 4 tokens):
      prefill argmax=271  decode argmax=271  logit maxdiff=2.276e-1  MATCH
probe (90 tokens):
      prefill argmax=15666  decode argmax=15666  logit maxdiff=5.047e-1  MATCH
      batched-prime argmax=15666  tokenwise argmax=15666  logit maxdiff=3.523e-1  MATCH
p16   (16 tokens):
      prefill argmax=23314  decode argmax=23314  logit maxdiff=2.965e-1  MATCH
      batched-prime argmax=23314  tokenwise argmax=23314  logit maxdiff=2.392e-1  MATCH
p4112 (4112 tokens):
      prefill argmax=84728  decode argmax=84728  logit maxdiff=7.075e-1  MATCH
      batched-prime argmax=84728  tokenwise argmax=84728  logit maxdiff=7.158e-1  MATCH
```

The `batched-prime` line is the one that walks the 16-row shape: on `p16` the whole prompt is one 16-row prime
chunk, on `p4112` the second chunk is; both equal the tokenwise reference. The base-versus-fixed run-gen comparison
of the original task 3 was not run: ruling 26 narrowed task 3 to the merged tree, and the bit-level statement it was
after (a 16-row prime equals the wide program) is the continuation table's, below and in section 3.

**The seven-arm continuation table on the merged tree (`m-gate`, runner `run-day23-cont-box3.sh`, exit 0),
verbatim, digest for digest the local table of section 3 and the day-22 fix table on both cards:**

```text
F-9296:            one call 14ab5f8b365dbd71; 9280 + 16 ok; 9248 + 48 ok; 9216 + 80 ok; 9184 + 112 ok; 9152 + 144 ok; 9120 + 176 ok; 9088 + 208 ok; A4 CONTINUATION GATE: PASS
F-9297:            one call e641952cac19e526; 9280 + 17 ok; 9248 + 49 ok; PASS
F-9311:            one call b61719f294866b05; 9280 + 31 ok; 9248 + 63 ok; PASS
F-9312:            one call 736a88d7c7448dcb; 9280 + 32 ok; 9248 + 64 ok; 9216 + 96 ok; PASS
F-9296-chunk32:    one call 14ab5f8b365dbd71 (MEMRA_PRIME_CHUNK=32); 9280 + 16 ok; 9248 + 48 ok; PASS
F-9296-chunk16:    one call bafe0e0a09a3d0a4 (MEMRA_PRIME_CHUNK=16); 9280 + 16 ok; 9248 + 48 ok; PASS
F-9296-nobatched:  one call 14ab5f8b365dbd71 (MEMRA_NO_BATCHED=1); 9280 + 16 ok; 9248 + 48 ok; PASS
```

Reading across the day: the lane's scope (day 22) and main's tier bound (#614) produce the same bits at every one of
these splits on both card classes; the two mechanisms are the same numeric program for the prime path. Nothing here
is red against #614.

## 5. Close of day

What the day leaves: memra#427 is fixed on main by #614 and guarded by the continuation gate in `tools/local-ci.sh`;
this lane's contribution that survives is the evidence (the width walk that named `ssm_beta`/`ssm_alpha` and the
batched tier, the seven-arm table on both card classes, the decision record) and two things in the tree: the one-arm
`qwen-a4-width-walk` diagnostic and the TESTING.md pointer to it. Nothing here is red against #614. The step35
continuation split (16 versus 17 rows) stays owed to a rig that holds a Step-3.7-Flash GGUF (none on either rig);
under #614 it is a confirmation, not a gap. The original task 3's base-versus-fixed `run-gen` comparison was not run
(ruling 26 narrowed it; section 4 states why the table already answers it).

Checks on the final tree: `cargo fmt --all -- --check`, `git diff --check`, `tools/check-flags.sh` (no uncovered
runtime names; no new `MEMRA_*` read today), `python3 tools/check-public-boundary.py check` (0 new), `cargo clippy -p
memra-engine --all-targets -- -D warnings` under the quota (exit 0, `rtx5090-day23/checks/`). Scratch under `/tmp`
(the tokenizer scratch, the comment draft) removed at close. On the box: the staged 9B (5.7 GB) and the symlink
directory stay under `/root/spill-receipts/b-day23/` (this lane's own dir; `/root/artifacts` untouched) for the
integ's re-runs. `target/bins/day22-*` and the `build/` dirs under the day 11/12 receipts stay untracked as before.
Nothing merged, no PR; the lead integrates. Budget: about 2.6 agent-hours (19:36Z to about 20:32Z of wall time with
both rigs running in parallel, counted once) against the 4-hour brief.
