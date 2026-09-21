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

## 4. Target card, merged tree `5f2060381`

PENDING: filled in as the box chain (`/root/spill-receipts/b-day23/chain.log`) completes.

## 5. Close of day

PENDING.
