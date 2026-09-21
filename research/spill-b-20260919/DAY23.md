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

PENDING: filled in as the chain (`rtx5090-day23/chain-day23-local.sh`, `chain.log`) completes.

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

PENDING.
