# cx-splitiso progress

## 2026-08-13 — lane opened

- Worktree: `/home/avifenesh/projects/wt-cx-splitiso`
- Branch: `lane/cx-splitiso`
- Base: `v0.81.3` (`7cf5fd842`)
- Scope: exactness-only isolation of the partial-prefix split-boundary divergence.
- Required mechanism flags: `MEMRA_PREFIX_PARTIAL_RESTORE=1` and
  `MEMRA_PREFIX_SPLIT_TRACE=1`.
- Guardrails: no eligibility-gate or suffix-prime changes; no scored/timed cells; no merge, tag,
  push, board edit, or live-serve-box access.
- Rig coordination: not yet attempted. Before any GPU cell, inspect active compute processes,
  respect the money-lane lock, and take only the designated second-card or local-5090 lock.
- Evidence status: no build or reproduction has run in this lane yet.
- Next: read the committed lcprestore receipt/harness and the adjacent eosclass progress, then map
  existing runtime trace surfaces before touching code.

## 2026-08-13T04:51:40+03:00 — upstream evidence and steering bound

- Read `/home/avifenesh/.lanectl/inbox/cx-splitiso.md` in full. It matches the task statement and
  adds no later steering.
- Read the merged lcprestore `PROGRESS.md`, `RESULTS.md`, `split_exactness.py`, and
  `verify_split_receipts.py`. The prior receipt establishes source-slice/restored K/V equality at
  64/512/2048/4374, byte failures at 512/2048, and no causal field.
- Read the adjacent eosclass progress through its restore-vs-batch discriminant checkpoint. That
  lane already owns the restored-hit EOS reproduction and broad state/decode-row instrumentation;
  this lane will not duplicate its EOS campaign. Exact batch width/row remains a required
  co-variable in the split-boundary cell.
- Imported the exact lcprestore merge delta as `1c7bf40a0`, including its committed harness and raw
  receipts, then imported the explicit default-off follow-up as `ab11ecc54`. No harness was
  reconstructed and the mechanism will be armed only by environment for exactness cells.
- No GPU work has started. Next: bind the concrete chunk/tile/stride/minimum-token constants and
  extend only the existing harness/diagnostic trace surfaces needed for a dense split sweep.

## 2026-08-13T05:12:00+03:00 — execution-shape hypothesis and checkpoint-1 diagnostics

- The old four-cell Gemma4 receipt does **not** compare the source slice with a cold computation
  ending at the same split. Request 1 primes all 4,860 tokens through `gemma4_prime`; a partial hit
  restores the slice and then feeds the suffix tokenwise through `decode_step`. The worker names
  this contract directly: fresh eager-only prompts prime whole, while carried suffixes use the
  tokenwise path (`worker.rs`, `prefill_tick`). The engine independently refuses carried
  `gemma4_prime` (`hybrid_forward.rs`, `gemma4_prime`). This is the first concrete mechanism to
  test, not yet a verdict.
- Bound comparison constants from code before measuring: `PREFIX_CACHE_MIN_TOKENS = 64`, scheduler
  `PREFILL_TICK_T = 1024`, general prefill `BLOCK_Q = 64` / `BK = 32`, and hd512 prefill
  `SP_M_ROWS = 16`. KV planes are token-linear allocations (`rows * token_bytes + 8`) with
  per-layer `k_tok_bytes`, `v_tok_bytes`, host `len`, and device `len_d`.
- Extended the committed lcprestore harness with an opt-in `--mode map`. Each split still uses its
  frozen A/B request pair, then adds (a) a genuinely cold full B request and (b) a cold request that
  ends exactly at the common-prefix boundary. Divergence is recorded per cell rather than aborting
  the dense sweep. Gate mode is unchanged.
- Added targeted receipts behind `MEMRA_PREFIX_SPLIT_DETAIL=1` plus an explicit boundary allowlist.
  They capture per-layer K/V hashes and `len_d`, canonical conv/SSM and the spare SSM buffer,
  boundary logits, `cache.pos` as the next RoPE position, sampler RNG/history, per-Engine
  `capture_keep_on` / `verify_exact`, and logits producer plus decode batch width/row. The detail
  reducer names the first differing field across source/restored, source/cold-boundary, and
  restored/cold-full comparisons.
- Static gates passed without formatting: `git diff --check`, Python byte-compilation for all three
  harness/reducer scripts, and `DOCS_RS=1 TMPDIR=/home/avifenesh/tmp-lanes cargo check -p
  memra-server`. Raw build output is in `raw/build-checkpoint1/cargo-check-docs-rs.log`.
- No GPU work has started. Next: checkpoint this restartable instrumentation, then live-check rig
  locks/processes before choosing box1 card 1 or the local 5090.

## 2026-08-13T05:14:00+03:00 — rig gate yielded; suffix-row receipt added

- Live rig check: box1 reported both PRO 6000s at `0 MiB`, but both `/tmp/memra-gpu.lock` and
  `/tmp/memra-gpu-1.lock` were held. This is the cachesize campaign's required two-lock exclusion,
  so idle-between-cells is **not** permission to co-run. Local GPU compute also had a live
  `memra-server` using `19714 MiB`, and `/tmp/memra-5090.lock` was held. No GPU work was attempted.
- Tightened the targeted live-cache receipt without another device readback: each layer now hashes
  the retained prefix, the first suffix row at the split, and the whole suffix separately. If the
  source/restored prefix agrees but cold and restored diverge on the first suffix row, the reducer
  can name that exact field and token row rather than only a whole-plane digest.
- Checkpoint-2 static gates again passed (`git diff --check`, Python byte-compilation, incremental
  `DOCS_RS=1 cargo check -p memra-server`); raw output is in
  `raw/build-checkpoint2/cargo-check-docs-rs.log`.
- Added a restartable box1 cell runner for the eventual opening. It takes both coordination locks
  non-blocking (yield code 75 if either owner is present), refuses any compute process, pins physical
  GPU 1/its UUID, runs only the one-server map path, stops the server before parsing, and releases
  between cells. `bash -n` and `shellcheck` are clean. This prevents an idle-looking gap inside the
  active cachesize process from being mistaken for an available rig.
- Added a deterministic map reducer that requires the 69-point grid (64-token steps through 4,352
  plus the prior 4,374 endpoint), emits the complete pass/fail table, and cross-tabulates outcomes
  against the named code geometries. For this frozen plain Gemma4 configuration the relevant
  first-suffix decode thresholds are global hd512 `MEMRA_FA512_MIN=512`, SWA window 1,024, dense
  defaults `FA_SP512_DEFAULT=32` and `FA_SPW_DEFAULT=64`, plus the big-rig split ladder. It also
  computes 4 KiB plane offsets using the expected FP8 row strides (global 512 B, SWA 2,048 B); the
  targeted receipts will confirm those strides from the live cache rather than assuming them in
  the verdict.
- Found a plausibly named local Gemma checkpoint at
  `/data/ai-ml/models/gemma-4-12b-it-qat/gemma-4-12b-it-qat-q4_0.gguf` and parameterized the same
  fail-closed cell for the local 5090. Its byte identity had not yet been checked at this checkpoint;
  the live preflight below correctly rejected it.

## 2026-08-13T05:26:00+03:00 — local smoke queued fail-closed

- Local 5090 remained actively owned by `cx-shmconflict`: its live `kernel-check` PID changed as the
  battery progressed and `/tmp/memra-5090.lock` stayed held. No splitiso GPU process has run.
- Added a queue wrapper that atomically waits on the local lock, verifies zero compute apps after
  acquisition, release-builds this committed source with disk-backed `TMPDIR`, then runs only the
  original 64/512/2048/4374 cells with all four detail boundaries. The inner cell understands an
  inherited lock, so there is no check-to-build or build-to-run race. Shell syntax, ShellCheck, and
  diff whitespace checks pass.
- Queue-time audit caught that the cell's original all-files cleanliness check would reject its own
  already-open raw evidence directory. Tightened the intended invariant to zero **tracked/index**
  drift (`--untracked-files=no`); untracked receipts may coexist, while every executable source file
  still has to be in the expected HEAD.

## 2026-08-13T06:03:00+03:00 — first local preflight excluded; exact artifact reconstructed

- The queued local smoke acquired `/tmp/memra-5090.lock` with zero compute applications and built
  the release server at committed source `9f66c3fb7` (server SHA-256
  `07455e7e70271e78c55e5966fbdb1f3e57681923e0e594fd4c54722a1250298c`). It then stopped before
  server launch: the selected local file was 6,975,877,728 bytes with SHA-256
  `faff1a63667fac17ac5e777f47114688fcefea96e220e211aaa8d62c2c4561f1`, not the frozen artifact's
  `93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b`. No GPU request ran and the
  cleanup receipt is empty. This attempt is excluded, not a reproduction result.
- The frozen box1 artifact is 1,568 bytes larger because its GGUF tensor-data section begins at
  15,823,488 rather than the local file's 15,821,920. The tensor layout and total file-size delta
  coincide exactly. While the cachesize campaign retained both GPU locks, fetched only the frozen
  15.8 MB GGUF header, combined it locally with the existing tensor-data blob, and accepted the
  result only after its whole-file SHA-256 reproduced the pinned `93567e57...` value. The exact local
  artifact is `gemma-4-12b-it-qat-q4_0-lcprestore-exact.gguf`; no cachesize GPU process or lock was
  touched. Coordination caveat: the initial remote identity check ran `sha256sum` over the full
  6.98 GB frozen file before this was recognized as an avoidable NVMe scan. It launched no GPU work,
  but the overlapping cachesize cell must be treated as potentially storage-contaminated. All later
  reconstruction transfer was limited to the 15.8 MB header.
- Hardened the cell to quote an expected/actual model-hash failure and made the local model and
  smoke-output paths overrideable. Next: commit this excluded preflight, then queue a fresh
  original-four cell against the verified artifact.

## 2026-08-13T06:18:00+03:00 — designated box1 card opened; release build complete

- The cachesize process exited and both `/tmp/memra-gpu.lock` and `/tmp/memra-gpu-1.lock` became
  acquirable; both PRO 6000 cards reported zero compute applications and 0 MiB used. Cancelled only
  splitiso's still-waiting local queue before it acquired the 5090 lock. Its `local-smoke2` receipt
  contains the queue line only: no build, server, or request ran.
- Materialized committed source `4fcd4bd1b` under `/opt/scratch/nvme/cx-splitiso/memra` through a
  verified Git bundle, then held both box1 coordination locks for the release build. The build
  completed on CUDA 13.2 / sm_120a in 3m54s; server SHA-256 is
  `fc94f06645cebcc483dc32b4dfc7a3f65050ca7c9d6f74a88d847fc121e8de95`. Raw output is under
  `raw/box1-build/`.
- Moved the expensive pinned-model hash behind both lock acquisitions and reuse that digest in the
  input manifest. This closes the preflight race and avoids hashing the 6.98 GB file twice per
  cell. Next: refresh the remote runner-only commit and run the detailed original four on physical
  GPU 1.

## 2026-08-13T06:24:00+03:00 — original four reproduced; first differing state named

- Exactness cell `raw/box1-smoke-original-four/` ran on physical GPU 1 only after acquiring both
  box1 locks with `compute_apps=none`. The pinned model hash, frozen workload hash, source commit,
  GPU UUID, and cleanup receipt are captured. No fatal marker or infrastructure failure occurred.
- Reproduced lcprestore byte-for-byte: split 64 PASS (`eb3e68a9...` both), split 512 FAIL
  (`bf81e8cb...` restored vs `719a43f4...` cold), split 2048 FAIL (`eb3e68a9...` vs
  `223618bf...`), and split 4374 PASS (`eb3e68a9...` both). Source-entry to immediate-restored state
  SHA-256 matches at all four splits.
- The added same-position cold boundary does **not** match the source entry at any of the four
  splits. The first named field is `kv.layer.0.k_sha256`; the detail reducer reports it at 64, 512,
  2048, and 4374. This proves the retained rows came from a different arithmetic execution shape:
  the 4,860-token monolithic `gemma4_prime` source slice is not byte-identical to a cold prime that
  ends at the split. Because this difference also occurs at both passing splits, it is mechanism
  evidence but not the pass/fail discriminator by itself.
- At the full 4,860-token first-sample boundary, restored and cold agree on `cache_pos` / RoPE next
  position, sampler seed/RNG/history, Engine atomics (both false), host sampling, and selected first
  token. Decode width/row are null for both because this single Gemma session runs the explicit
  eager path. They differ in device `len_d` (`[4860]` restored versus `[0]` immediately after cold
  monolithic prime), boundary-logit hashes, and suffix K/V hashes; the earliest suffix K/V mismatch
  after the common retained prefix is layer 0's suffix aggregate (layer 0's first suffix row still
  matches, while layer 1's first suffix row already differs). These same categories differ at all
  four splits, again separating the mechanism from the eventual token-hash flip.
- Verdict is not assigned yet: the required dense pass/fail pattern remains. Next: checkpoint this
  reproduced receipt, then run the 69-point 64-token map in restartable short segments with detail
  disabled.

## 2026-08-13T06:34:00+03:00 — dense segment 1 captured; normal EOS classification corrected

- `dense-seg01-attempt1` captured all ten requested splits (128, 192, 256, 320, 384, 448, 576,
  640, 704, 768), released both locks, and left zero compute applications. All restored requests
  reported the requested cached-token count and all split state traces were emitted.
- The runner stopped before post-parse because request 1 at split 448 ended normally on EOS after
  two completion tokens (`HTTP 200`, SSE `DONE`, request id present, `finish_reason=stop`). The
  frozen sellgate helper's scored-cell `ok` contract requires exactly 60 tokens, so map mode called
  this `split=448 case=request1-seed: None`. This is not a quoted runtime failure: the seed cache was
  inserted and its output is not the comparison oracle.
- Kept gate mode unchanged. Map mode now accepts only this narrow normal-stop case in addition to
  the frozen helper's strict success. Added a deterministic normalizer for the already-captured raw
  receipt; it preserves the original file and records the original verdict/failure in the derived
  summary. The derived receipt is `MAP-COMPLETE` and the independent state reducer is
  `MAP-VERIFIED` with no failures: PASS at 128/192/256/320/384/704/768 and FAIL at 448/576/640.
  Thus the first 64-token transitions are PASS 384 -> FAIL 448 and FAIL 640 -> PASS 704; 512 remains
  the prior confirmed failure. Next: checkpoint segment 1, then continue the dense cells.

## 2026-08-13T06:42:00+03:00 — dense segment 2 crosses SWA window

- `raw/dense-seg02/` is `MAP-COMPLETE` and independently `MAP-VERIFIED`; source→restored state
  matches at all eleven splits and there are no infrastructure failures. PASS:
  832/896/960/1088/1152/1216/1344/1472. FAIL: 1024/1280/1408.
- This rules out the Gemma `sliding_window = 1024` boundary as a one-sided threshold: the exact
  1024 point fails, both adjacent 64-token samples (960 and 1088) pass, and failures recur above the
  window. It also rules out a single monotonic `MEMRA_FA512_MIN=512` transition, since both outcomes
  occur on its high side. No geometry conclusion is promoted until the complete dense grid and
  targeted transition probes exist.

## 2026-08-13T06:48:00+03:00 — dense segment 3 brackets 2048

- `raw/dense-seg03/` is `MAP-COMPLETE` / `MAP-VERIFIED`, with source→restored equality at every
  split and no infrastructure failures. PASS: 1536/1856/1920/1984/2112/2176/2240. FAIL:
  1600/1664/1728/1792.
- Combining the prior detailed split 2048 failure gives PASS 1984 -> FAIL 2048 -> PASS 2112: the
  old 2048 point is an isolated failure at 64-token sampling resolution, not a one-sided change at
  the big-rig FA split-ladder boundary. The 1600–1792 failure island also lies wholly within the
  same <=2048 ladder class. Geometry correlations remain pending the full map.

## 2026-08-13T06:50:00+03:00 — dense segment 4 crosses 2560

- `raw/dense-seg04/` is `MAP-COMPLETE` / `MAP-VERIFIED`, with source→restored equality at all
  eleven splits, no infrastructure failures, and an empty cleanup compute-app receipt. PASS:
  2304/2368/2432/2496/2560/2624/2688/2880. FAIL: 2752/2816/2944.
- Both outcomes recur while all 64-token alignment residues remain fixed: PASS 2688 -> FAIL 2752,
  FAIL 2816 -> PASS 2880, and PASS 2880 -> FAIL 2944. The exact 2560 point passes. This continues
  to contradict a one-sided prefill-size or tile-alignment threshold; the complete grid and
  targeted transition probes remain required before assigning the lane verdict.

## 2026-08-13T06:55:00+03:00 — frozen-map content confound identified; controlled map added

- Audited the frozen constructor before interpreting its many pass/fail islands. For every split it
  builds the common prefix with `fixed_prompt_ids(split, 370)` and then **restarts** request B's
  suffix at position zero with `fixed_prompt_ids(total - split, 444)`. Therefore request B is not
  byte-identical between split cells: moving the split changes both execution geometry and prompt
  content. The original/frozen map remains required reproduction evidence, but its transition
  locations alone cannot establish an alignment boundary.
- Added a map-only `fixed-target` constructor without changing gate mode or the default
  `lcprestore` map. It holds request B byte-identical at all splits and changes only request A after
  the requested exact LCP. Constructor assertions cover 64/511/512/1024/2047/2048/2049/4374 and
  confirm identical B across splits. This controlled follow-up is necessary to distinguish a real
  split-position boundary from content-sensitive amplification of the prefill-vs-decode numeric
  difference already captured. Python compile, ShellCheck, and diff whitespace checks pass.

## 2026-08-13T06:56:00+03:00 — dense segment 5 is an all-pass band

- `raw/dense-seg05/` is `MAP-COMPLETE` / `MAP-VERIFIED`: all eleven splits from 3008 through 3648
  PASS, source→restored state matches at every split, there are no infrastructure failures, and
  cleanup reports no compute application.
- This 704-token all-pass span sits entirely above the big-rig 2048 split-ladder rung while the
  same code paths and 64-token alignment residues produced both outcomes below 3008. It is not an
  exact geometry discriminator. The final 3712–4352 segment remains before reducing the complete
  frozen-input grid.

## 2026-08-13T06:58:00+03:00 — first-suffix dispatch correlated to the arms actually reached

- Corrected the preliminary 2048-ladder interpretation by following Gemma's eager T=1 call sites,
  not just the generic `fa_split_keys()` function. Global layers switch to `fa_decode_rows` at
  `kvl.len >= MEMRA_FA512_MIN` (512) and use the dense-model `FA_SP512_DEFAULT=32`; SWA layers
  switch to `fa_decode_rows_w` above the 1024-token window and use
  `FA_SPW_DEFAULT=64`. Consequently the generic 188-SM `t_kv <= 2048 ? 16 : 64` rung is nominal
  but **not live** for either first-suffix arm at the 2048 split.
- The reducer now reports the actual global/SWA arm and partition, separately retains the nominal
  fallback ladder and whether it is live, and records the constant execution-shape facts: cold is
  one 4860-token `gemma4_prime`, restored suffix is tokenwise T=1 `decode_step`, and Gemma runs the
  per-session eager path with observed batch width/row null. On the 58 points currently captured,
  every live arm and actual partition contains both PASS and FAIL; exact discriminator remains
  none pending the last segment and controlled map.

## 2026-08-13T07:00:00+03:00 — first differing logical field reduced explicitly

- Added `FIELD-COMPARISON.json`, a compact reduction of the already-tee'd detailed original-four
  log. At 64/512/2048/4374, restored and genuine-cold full sessions agree on **every retained
  prefix K/V hash**, cache/RoPE position 4860, canonical/spare recurrent state (both absent for
  this transformer-only checkpoint), sampler seed/RNG/history, per-Engine flags, eager batch
  provenance (width/row both null), and the selected first token. Their boundary logit vectors
  differ at all four positions.
- The first differing captured **logical model-state** field is identical at all four positions:
  `kv.layer.1.first_suffix_k_sha256`. Layer 0's first suffix K/V row still matches; layer 1's first
  suffix K and V rows differ. This locates the divergence inside the first suffix token after layer
  0, where restored execution uses T=1 `decode_step` while cold execution uses the monolithic
  prefill graph. Whole suffix aggregates then differ on all 48 layers.
- The device mirror `kv.layer.0.len_d` is separately `[4860]` restored versus `[0]` immediately
  after cold prime. It is not promoted as the boundary-logit cause: cold `gemma4_prime` has already
  produced those logits using host logical lengths, and the eager rows calls sync `len_d` before
  later decode attention. The matrix keeps this observable discrepancy explicit rather than
  silently normalizing it away.

## 2026-08-13T07:01:00+03:00 — frozen-input 69-point map complete; no geometric discriminator

- Final segment `raw/dense-seg06/` is `MAP-COMPLETE` / `MAP-VERIFIED`. FAIL:
  3904/4096/4224. PASS: 3712/3776/3840/3968/4032/4160/4288/4352. All eleven
  source→restored receipts match and there are no infrastructure failures.
- `DENSE-MAP.md` / `.json` now contain all 69 requested points: every 64 tokens from 64 through
  4352 plus 4374. Result: **51 PASS / 18 FAIL**, 22 sampled outcome transitions, no missing cells,
  and **no exact discriminator** among prefix eligibility, worker prefill size, prefill tile
  residues, actual global/SWA first-suffix arms and partitions, nominal 188-SM split ladder,
  decode batch provenance, or global/SWA KV-plane page offsets.
- All 68 regular grid points share split mod 64/32/16 = 0 and both PASS and FAIL occur there; all
  share global/SWA plane page offset 0 and again both outcomes occur. Every actual attention class
  also contains both outcomes: global scalar-sp16 6/1, global rows-sp32 45/17, SWA kvmod-sp16
  11/4, and SWA rows_w-sp64 40/14 (PASS/FAIL). `PREFIX_CACHE_MIN_TOKENS=64` is true for all 69.
- This complete frozen-input map falsifies the opening alignment/tiling inference. Because the
  frozen constructor changes request B's suffix contents at each split, the 22 apparent
  transitions are content-sensitive output-basin flips over a persistent arithmetic divergence,
  not evidence of a split-position geometry boundary. Next: run targeted named/transition points
  and the byte-identical fixed-target control before assigning the verdict.

## 2026-08-13T07:03:00+03:00 — eosclass handoff re-read; classes are now separated

- Re-read `../wt-cx-eosclass/research/eosclass-20260813/PROGRESS.md` through its current 06:50
  checkpoint. That lane has now deterministically isolated the historical Q27 11-token EOS class:
  the target crossed from eager B=1 to generic batched width, EOS id 248046 became rank-1, and a
  one-program B=1/B>=2 default eliminated the failures across its controlled delay sweep. Its
  restored state still matched the seed.
- This lane does not duplicate that work: every detailed splitiso sample is the single-session
  Gemma eager path with decode batch width/row null. The splitiso field difference is instead the
  monolithic-prompt-prime versus tokenwise-carried-suffix arithmetic at
  `kv.layer.1.first_suffix_k_sha256`. Therefore partial restore is **not** promoted as the Q27 EOS
  discriminant, and splitiso will not claim to close the EOS class.

## 2026-08-13T07:06:44+03:00 — fixed-target pilot changes the old four-point pattern

- Ran the byte-identical-request control at 64/512/2048/4374 on box1 physical GPU 1 under both
  `/tmp/memra-gpu.lock` and `/tmp/memra-gpu-1.lock`; preflight saw `compute_apps=none`, and cleanup
  is empty. The genuinely-cold output is the same at all four cells
  (`eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df`).
- The controlled result is **FAIL 64; PASS 512/2048/4374**. In particular, the two frozen-input
  receipt failures at 512 and 2048 disappear when request B is held fixed, while the old 64 PASS
  flips to FAIL. Source-to-restored state still matches at every split and the verifier reports no
  infrastructure failure. This directly confirms that the old PASS/FAIL islands cannot be
  assigned to split geometry without controlling prompt identity.
- Detailed traces still show unequal boundary-logit vectors at every pilot split, including the
  three cells whose 60-token greedy output matches. The first differing captured logical field is
  again `kv.layer.1.first_suffix_k_sha256` at all four cells; prefix K/V, position, recurrent/spare
  state, sampler, Engine flags, and eager batch provenance remain equal. Output equality therefore
  measures whether the persistent arithmetic perturbation crosses a greedy decision boundary,
  not whether the underlying states are bit-identical.
- This pilot predates the prompt-hash receipt added at `ebee2a2c0`. The controlled worktree is now
  fast-forwarded to that commit; the full fixed-target grid will record and enforce one constant
  request-B token hash before this control is promoted to the final result.

## 2026-08-13T07:09:43+03:00 — reducer now enforces controlled-request identity

- The reducer now rejects a `fixed-target` map unless every input summary carries a target-prompt
  hash and the union contains exactly one hash. Frozen lcprestore receipts remain readable and are
  explicitly labeled `not-recorded`; they predate the hash field and are not used as the controlled
  prompt-identity proof.
- Transition follow-ups are now the two immediately adjacent unsampled values (`left + 1` and
  `right - 1`) plus the named 64/512/1024/2048 neighbors. Midpoints added no boundary resolution
  beyond the already-dense 64-token grid. The frozen map therefore names 47 exact targeted probes,
  covering every sampled transition and every requested code boundary without redundant cells.

## 2026-08-13T07:15:48+03:00 — fixed-target dense segment 1 complete

- `raw/fixed-dense-seg01/` is `MAP-COMPLETE` / `MAP-VERIFIED` for 64..768 in 64-token steps. PASS:
  128/192/256/320/448/512/576/640/704/768. FAIL: 64/384. All twelve source→restored state hashes
  match, infrastructure failures are empty, cleanup reports no compute application, and both locks
  were released after the bounded cell.
- Every cell reports exactly the same request-B token hash,
  `21ef4227fcb0993c341e03c4df6bf01b27f6012021c881fc4a8f451364495397`. Even with content held
  fixed, output equality is already non-monotonic (FAIL 64 -> PASS 128 and PASS 320 -> FAIL 384 ->
  PASS 448). A simple eligibility threshold or monotonic prefix-length boundary is therefore not
  the mechanism.

## 2026-08-13T07:22:49+03:00 — fixed-target segment 2 crosses all requested low boundaries

- `raw/fixed-dense-seg02/` is `MAP-COMPLETE` / `MAP-VERIFIED` for 832..1472. PASS:
  832/896/960/1024/1088/1152/1216/1280/1344/1408. FAIL: 1472. All eleven source→restored hashes
  match, the controlled target hash is unchanged, infrastructure failures are empty, and cleanup
  reports no compute application.
- The exact 1024 point and both sampled sides (960/1088) PASS. The global rows arm is already live
  throughout this segment, while SWA changes from its vector arm through the 1024 window to
  `fa_decode_rows_w` at 1088; both sides remain PASS. Thus neither the 512 global switch nor the
  1024 SWA-window/rows switch explains the controlled output failures.

## 2026-08-13T07:29:08+03:00 — fixed-target segment 3 excludes the nominal 2048 rung

- `raw/fixed-dense-seg03/` is `MAP-COMPLETE` / `MAP-VERIFIED` for 1536..2240. FAIL: 1728/1856.
  PASS: 1536/1600/1664/1792/1920/1984/2048/2112/2176/2240. All twelve source→restored hashes
  match, the controlled target hash is unchanged, infrastructure failures are empty, and cleanup
  is empty.
- 1984/2048/2112 all PASS under byte-identical request B. This directly excludes the nominal
  188-SM `t_kv <= 2048 ? 16 : 64` rung as the output boundary, in addition to the source audit
  showing that Gemma's live rows/rows_w arms do not consume it here. The two failures inside the
  unchanged rows/rows_w class remain content-sensitive greedy-output basin flips.

## 2026-08-13T07:34:04+03:00 — fixed-target segment 4 remains non-geometric

- `raw/fixed-dense-seg04/` is `MAP-COMPLETE` / `MAP-VERIFIED` for 2304..2944. Only 2304 FAILS;
  2368 through 2944 all PASS. All eleven source→restored hashes match, the controlled target hash
  remains identical, infrastructure failures are empty, and cleanup reports no compute
  application.
- This entire segment has one global rows-sp32 / SWA rows_w-sp64 dispatch shape and identical
  64/32/16 alignment residues. Its FAIL 2304 -> PASS 2368 transition therefore cannot be assigned
  to an attention-arm, tile, page-offset, decode-width, or eligibility transition.

## 2026-08-13T07:38:06+03:00 — fixed-target segment 5 has one isolated output flip

- `raw/fixed-dense-seg05/` is `MAP-COMPLETE` / `MAP-VERIFIED` for 3008..3648. Only 3136 FAILS;
  3008/3072 and 3200 through 3648 PASS. All eleven source→restored hashes match, the target hash is
  unchanged, infrastructure failures are empty, and cleanup is empty.
- PASS 3072 -> FAIL 3136 -> PASS 3200 occurs without changing any named execution class or regular
  grid residue. This is an isolated greedy-output flip over the persistent numerical divergence,
  not a one-sided storage, page, or attention threshold.

## 2026-08-13T07:41:26+03:00 — fixed-target 69-point map complete

- Final segment `raw/fixed-dense-seg06/` is `MAP-COMPLETE` / `MAP-VERIFIED`; only 3776 FAILS and
  the other eleven points PASS. All source→restored state hashes match, the target hash remains
  constant, infrastructure failures are empty, and cleanup is empty.
- `FIXED-TARGET-MAP.md` / `.json` reduce all 69 controlled cells: **61 PASS / 8 FAIL** at
  64/384/1472/1728/1856/2304/3136/3776, with 15 sampled outcome transitions. Every segment reports
  the one canonical request-B hash
  `21ef4227fcb0993c341e03c4df6bf01b27f6012021c881fc4a8f451364495397`; no summary lacks the
  receipt.
- There is no exact discriminator among `PREFIX_CACHE_MIN_TOKENS`, worker prefill size, prefill
  64/32/16 tile residues, actual global/SWA first-suffix arm or partition, nominal 2048 ladder,
  eager decode batch width/row, or global/SWA KV-plane page offset. The 36 immediate transition
  neighbors and named 512/1024/2048 neighbors are now frozen as the targeted controlled follow-up.

## 2026-08-13T07:48:00+03:00 — early fixed-target neighbors are all PASS

- `raw/fixed-targeted-seg01/` is `MAP-COMPLETE` / `MAP-VERIFIED`: 65/127/321/383/385/447/511/513/
  1023 all PASS. All nine source→restored hashes match, one target hash is retained,
  infrastructure failures are empty, and cleanup is empty.
- The dense FAIL points 64 and 384 are therefore single sampled-position failures at one-token
  resolution: split 64 is bracketed by PASS 65, and split 384 by PASS 383/385. The 512 global-arm
  switch is PASS at 511/512/513, and the lower side of the 1024 SWA boundary is PASS at 1023.

## 2026-08-13T07:53:23+03:00 — middle fixed-target neighbors refine the islands

- `raw/fixed-targeted-seg02/` is `MAP-COMPLETE` / `MAP-VERIFIED`. PASS:
  1025/1409/1471/1473/1535/1665/1729. FAIL: 1727/1791. All source→restored hashes match, the
  target hash is unchanged, infrastructure failures are empty, and cleanup is empty.
- 1023/1024/1025 all PASS across the SWA dispatch switch. Dense split 1472 is a single-position
  failure bracketed by PASS 1471/1473. The 1728 failure extends left to 1727 but ends at PASS 1729;
  the next dense transition is sharpened to FAIL 1791 -> PASS 1792. These one-token-scale changes
  cannot arise from the quoted rows/rows_w partition thresholds, which are constant throughout.

## 2026-08-13T07:58:15+03:00 — 2048 is clean; later islands remain token-scale

- `raw/fixed-targeted-seg03/` is `MAP-COMPLETE` / `MAP-VERIFIED`. PASS:
  1793/1855/1857/2047/2049/2241/2303. FAIL: 1919/2305. All nine source→restored hashes match, the
  target hash is unchanged, infrastructure failures are empty, and cleanup is empty.
- The 1856 failure is bracketed by PASS 1855/1857. The next transition is FAIL 1919 -> PASS 1920.
  Most importantly, 2047/2048/2049 all PASS, closing the nominal ladder correlation at one-token
  resolution. The 2304 failure extends right to 2305 but begins after PASS 2303, again without a
  dispatch or partition change.

## 2026-08-13T08:01:44+03:00 — fixed-target transition probing complete

- `raw/fixed-targeted-seg04/` is `MAP-COMPLETE` / `MAP-VERIFIED`. PASS:
  2367/3073/3135/3199/3713/3777/3839. FAIL: 3137/3775. All nine source→restored hashes match, the
  target hash is unchanged, infrastructure failures are empty, and cleanup is empty.
- `FIXED-TARGET-MAP.md` / `.json` now include the 69-point dense grid plus all 36 immediate and
  named probes: **105 cells, 91 PASS / 14 FAIL**, no missing dense cells, one target-prompt hash,
  and no exact named discriminator. The one-token-resolution failure islands are:
  64; 384; 1472; 1727–1728; 1791; 1856; 1919; 2304–2305; 3136–3137; and 3775–3776.
- Those islands cross arbitrary modulo/page residues and occur wholly within constant dispatch
  classes. The 512, 1024, and 2048 code boundaries are all PASS on `boundary-1`, `boundary`, and
  `boundary+1`. The byte-output failure set is therefore not an alignment/tiling boundary; it is
  where the persistent cold-prime versus restored-decode numeric difference crosses a greedy
  token decision during the 60-token completion.

## 2026-08-13T08:08:57+03:00 — frozen-constructor targeted segment 1

- `raw/frozen-targeted-seg01/` is `MAP-COMPLETE` / `MAP-VERIFIED`. PASS:
  65/385/511/703/961/1023. FAIL: 447/513/641/1025. All ten source→restored hashes match,
  infrastructure failures are empty, and cleanup is empty.
- The summary correctly contains ten distinct target-prompt hashes for ten splits. This is the
  frozen lcprestore constructor's content-changing behavior, not a controlled geometry sweep; its
  transition receipts are retained for exact reproduction coverage but remain separate from the
  single-hash fixed-target causal table.

## 2026-08-13T08:15:47+03:00 — frozen-constructor targeted segment 2

- `raw/frozen-targeted-seg02/` is `MAP-COMPLETE` / `MAP-VERIFIED`. PASS:
  1217/1279/1343/1345/1407/1409/1471/1537. FAIL: 1087/1281. All ten source→restored hashes match,
  the summary carries ten distinct target hashes, infrastructure failures are empty, and cleanup
  is empty.
- Adjacent frozen-constructor outcomes can flip while their request-B hashes also change. This is
  useful reproduction detail but cannot be interpreted as a split-only alignment transition; the
  fixed-target board remains the causal control.

## 2026-08-13T08:20:48+03:00 — frozen-constructor targeted segment 3

- `raw/frozen-targeted-seg03/` is `MAP-COMPLETE` / `MAP-VERIFIED`. PASS: 1599/2049/2111. FAIL:
  1793/1855/1985/2047/2689/2751. All nine source→restored hashes match, the summary carries nine
  distinct target hashes, infrastructure failures are empty, and cleanup is empty.
- Around the old 2048 point, the frozen constructor yields FAIL 2047 / FAIL 2048 / PASS 2049 while
  simultaneously changing target content at every position. The controlled constructor instead
  yields PASS at all three. This is a direct counterexample to attributing the old three-cell
  pattern to the nominal 2048 split ladder.

## 2026-08-13T08:24:18+03:00 — frozen-constructor targeted segment 4

- `raw/frozen-targeted-seg04/` is `MAP-COMPLETE` / `MAP-VERIFIED`. PASS:
  2879/2945/3007/3841/3905. FAIL: 2817/2881/2943/3903. All nine source→restored hashes match, the
  summary carries nine distinct target hashes, infrastructure failures are empty, and cleanup is
  empty.
- As in the lower ranges, adjacent frozen outcomes alternate while target content changes. These
  receipts complete the dense transition neighborhoods through 3905 without creating a valid
  geometry discriminator.

## 2026-08-13T08:28:39+03:00 — frozen-constructor transition probes complete

- `raw/frozen-targeted-seg05/` is `MAP-COMPLETE` / `MAP-VERIFIED`. PASS:
  3967/4095/4097/4159/4161/4225/4287. FAIL: 4033/4223. All nine source→restored hashes match, the
  summary carries nine distinct target hashes, infrastructure failures are empty, and cleanup is
  empty.
- The box1 exactness cell acquired both `/tmp/memra-gpu.lock` and `/tmp/memra-gpu-1.lock`
  non-blocking, saw no compute applications, used physical GPU 1 UUID
  `GPU-2b4cf166-fd33-f161-8536-ca04bc72280c`, and released both locks. No scored or timed work was
  run. All requested frozen-constructor immediate neighbors and named 512/1024/2048 probes are now
  present; the next checkpoint reduces the full 116-cell frozen table.

## 2026-08-13T08:29:27+03:00 — frozen-constructor 116-cell reduction sealed

- `FROZEN-TARGETED-MAP.md` / `.json` reduce the 69-point every-64 sweep plus all 47 requested
  immediate/named transition probes: **116 cells, 80 PASS / 36 FAIL**, no missing dense cells, no
  reducer failures, and no exact discriminator among the named geometry/execution fields.
- The 47 newly instrumented probes have 47 distinct request-B hashes; the seven older dense
  segment summaries predate prompt hashing. Therefore this table reproduces and brackets the
  frozen lcprestore receipt, but is explicitly not a single-input split-only experiment.
- Its code thresholds are contradicted locally as causal output boundaries: the frozen constructor
  is PASS/FAIL/FAIL at 511/512/513, FAIL/FAIL/PASS at 2047/2048/2049, and PASS/FAIL/PASS at
  4095/4096/4097, while target content changes at each point. The single-hash fixed-target control
  is PASS at all nine corresponding named-boundary cells.

## 2026-08-13T08:35:36+03:00 — actual global threshold bracket closed

- Source re-check caught the coordinate distinction between split and first-suffix KV length:
  `fa_decode_rows` begins at `kvl.len >= 512` after the suffix row is appended, so the exact split
  transition is 510 -> 511. Added fixed-target split 510 under both box1 locks; it is PASS,
  source→restored state matches, the one controlled target hash matches, infrastructure failures
  are empty, and cleanup is empty.
- `FIXED-TARGET-MAP.md` / `.json` now contain **106 cells, 92 PASS / 14 FAIL**. The exact global
  bracket is split 510 PASS (`first_tkv=511`, `fa_decode_kvmod-scalar-sp16`) -> 511 PASS
  (`first_tkv=512`, `fa_decode_rows-sp32`) -> 512 PASS. Together with PASS triples across the SWA
  1024 and nominal 2048 boundaries, every named code boundary is now present and none is an output
  discriminator.
- Corrected the reducer's named-boundary coordinate and expanded its source-backed prefill-tile
  description: cold-only SWA uses 64x32 (or paired 32x32), globals use 16x32 single-pass (32x32
  fallback); the split does not select those tiles because every cold control primes the same
  4,860-token request monolithically.

## 2026-08-13T08:46:05+03:00 — result drafted: BOUNDARY-IDENTIFIED

- `RESULTS.md` assigns **BOUNDARY-IDENTIFIED**. The named boundary is not a token alignment: it is
  the source-quoted `eager_mono && carried` program fork. Fresh Gemma runs one 4,860-token
  `gemma4_prime`; a restored suffix runs T=1 `decode_step`. Prime attends from transient Q/K/V
  operands while decode attends through quantized cache rows. That transient state/program is not
  restored.
- The first captured persistent restored-vs-full-cold model-state difference is
  `kv.layer.1.first_suffix_k_sha256` at every detailed split. Layer 0's quantized first-suffix K/V
  still match, bounding the first uncaptured floating-point difference to layer 0's
  prime-versus-decode attention block without claiming whether Q, pre-quantized K/V, or attention
  output differs first. `len_d` remains explicitly reported as a separate non-causal mirror
  difference.
- The controlled failure set is 14 content-sensitive positions in 11 one/two-token islands. Every
  named threshold (actual global 511, SWA 1024, nominal 2048), tile/stride/page class, batch
  provenance, and eligibility class is contradicted as an exact output discriminator. The fixed
  request still has unequal detailed boundary logits at all four pilot positions, including three
  60-token output PASS cells.
- cx-eosclass's final Q27 result remains separate: its discriminant is an eager-B1 -> batched-width
  transition with rank-1 EOS; splitiso's width/row are null. No claim that this lane closes Q27 is
  made.
- No small fix is proposed. A safe fix requires one canonical Gemma numerical contract across
  cold prime and resumed decode, not a gate toggle. Eligibility, suffix-prime behavior, partial
  restore's default-off posture, release state, and generated boards remain unchanged. Final static
  and artifact validation is next, followed by the requested lane-only commit and stop.

## 2026-08-13T08:48:22+03:00 — final artifact audit PASS

- The first final audit completed every artifact/state/server/Cargo/perf-board check, then stopped
  at the staged whitespace gate with the quoted message
  `research/splitiso-20260813/RESULTS.md:3: trailing whitespace.` The failed audit is retained
  byte-exact as `raw/final-validation-attempt1.log.gz` (compressed because its verbatim offending
  line would itself fail Git's whitespace gate); the Markdown whitespace was removed without
  changing the result.
- `raw/final-validation.log` is the complete green rerun. It byte-compiles and loads every Python
  entrypoint; passes `bash -n` and ShellCheck for all three runners; enforces all map counts,
  prompt-hash invariants, field-matrix invariants, and reducer verdicts; independently verifies all
  116 frozen and 106 fixed-target source→restore receipts; and scans every accepted server log for
  clean compute preflight/cleanup plus the runner's fatal CUDA/runtime markers.
- `DOCS_RS=1 TMPDIR=/home/avifenesh/tmp-lanes CARGO_BUILD_JOBS=1 cargo check -p memra-server`
  passes. `python3 tools/update-perf-board.py --check` reports `perf board is up to date`.
  `git diff --cached --check` passes. No `cargo fmt`, GPU cell, score, merge, tag, push, or board
  edit was performed during final validation.
- The lane deliverable is complete. Next action is only the requested final lane commit, followed
  by STOP.
