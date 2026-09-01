# cx-coldfix progress

## 2026-08-12T16:57:13Z — lane opened

- Branch/worktree: `lane/cx-coldfix` in `/home/avifenesh/projects/wt-cx-coldfix`, based on `main` at `9b43b556b`.
- P0: reproduce and fix Q35-A3B cold-miss completions ending before the requested 60 tokens under the naked coldhol continuation-capable prime-batch default.
- Source finding: Q35 mixed `c=4` cold misses returned `finish_reason=stop` at 26/60 tokens on `584ed0af0`; Q27 was clean on the same default. The sealed upstream evidence manifest is `bd6b819e3e784af1b59b960515866f52346e067dfc7478206889738e76e55bf2`.
- Initial hypotheses remain unproven: continuation-slot stop/EOS leakage, violation of the Qwen35Moe batched-trunk pin, or stale sequence-position stop evaluation.
- Success criteria: capture a minimal local 5090 repro with raw logs; identify the mechanism from code/runtime evidence; add a Q35 `c=4` mixed exact-token regression cell; implement the smallest correct fix; pass cargo tests plus `kernel-check`, both-model `run-gen`, `run-spec` K=1..8, `serve-smoke`, and `c=64` stress; write `RESULTS.md` with a release recommendation.
- Constraints in force: no merge, tag, origin push, board update, formatting sweep, rustup, nsys artifact, other worktree, `--no-verify`, or uncoordinated GPU use.

## 2026-08-12T17:00:24Z — pre-reproduction evidence

- GPU coordination is clear: the local RTX 5090 reported 0% utilization and 157 MiB used, no compute applications, and no live battery/server process. The newest `/tmp/battery-*.log` is the completed probeready battery ending `ALL GREEN`; it is not held open.
- The pinned Q35 artifact is present and hashes to the expected `df27a780435b7b45c2597536112ea3cb091f8544c3d0c3318d9f4258b31f7adf`.
- The exact scheduler delta is `b37d77c6f`: it broadened prime-batch eligibility from a complete fresh prompt to one bounded `cold_chunk` per tick, including carried caches where `cache.pos == fed.len()`, and repeatedly calls `prime_cache_batch` until fewer than two candidates remain.
- The prior Q35 fix is a decode-only model-class pin: `b1_fast_arch_eligible()` excludes `Arch::Qwen35Moe` so B=1 stays on the batched decode trunk. The new coldhol eligibility has no equivalent MoE/architecture guard. This is a code-path distinction to test, not yet the root-cause verdict.
- Next: add a focused wrapper around the frozen sellgate workload, reproduce the Q35 mixed `c=4` cell under the naked default, then repeat with `MEMRA_PRIME_BATCH=1` as the rollback control.

## 2026-08-12T17:07:15Z — local baseline reproduced

- Added `tools/q35-cold-mixed-gate.py`, a test-only wrapper over the frozen sellgate workload: Q35 mixed90 `c=4`, 20 requests, 18 full-cache hits, two cold misses, and an exact 60-token requirement.
- On the unmodified naked default, the gate failed exactly as reported: both cold misses (indices 1 and 3) returned HTTP 200 and `[DONE]` but stopped at 26/60 with `finish_reason=stop` and identical text hash `1ecc62ea...730f`; all 18 hits reached 60/60 with `finish_reason=length`.
- Server evidence shows the two misses took five cross-request prime batches: fresh `B=2 tokens=2048 carried=0 partial=2`, three `carried=2 partial=2` batches, then `carried=2 partial=0`. Client and engine both reported 1,132 output tokens; cached/prompt counters had zero drift; admission defers, VRAM defers, and step-OOM parks were all zero.
- The request supplies no stop strings. In `advance_sample_emit`, the only pre-text stop producing this shape is `s.params.eos.contains(&next)`, which records the EOS token then finishes `StopReason::Eos`; budget exhaustion would instead produce 60 tokens and `length`. This rules out a stale HTTP finish mapping and establishes that the changed prime path makes Q35 select a real EOS at generated token 26.
- Raw baseline receipts are under `research/coldfix-20260812/raw/baseline/`.

## 2026-08-12T17:13:51Z — mechanism isolated and fix staged

- `MEMRA_PRIME_BATCH=1` is a clean rollback control: the same frozen cell completed 20/20 and both cold misses reached 60/60 with `finish_reason=length`.
- A forced whole-fresh control (`MEMRA_PREFILL_TICK=8192`, `MEMRA_PRIME_BATCH_MAX_T=8192`, 100 ms formation hold) executed one live `[prime-batch] B=2 tokens=9720 carried=0 partial=0` and also completed 20/20 at 60/60. Thus prime batching in general is not the failing condition; repeated carried batches are.
- Necessary/sufficient path evidence for this repro: naked repeated carried batches fail; disabling all batches passes; retaining a live whole-fresh batch while removing continuation calls passes. The two failed sessions were cold, newly admitted, uncached sessions with independent per-session stop state, so continuation-pool EOS leakage is not involved. Worker and engine position bookkeeping advances the exact drained chunk length and rechecks `cache.pos == fed.len()` before the next carried batch; no stale-position or accounting symptom was observed.
- Small ship fix staged: `carried_prime_batch_eligible()` keeps routed-MoE architectures on serial continuation primes while leaving whole-fresh batching unchanged. Dense Q27 remains eligible, preserving the coldhol knee win. The follow-up door is explicit: re-enable MoE only after the engine gate covers repeated real-size chunks followed by the serving batched-decode trunk.
- The frozen Q35 mixed-c4 exact-token cell is implemented as `tools/q35-cold-mixed-gate.py` and wired into `tools/serve-smoke.sh`; unit coverage pins routed-MoE versus dense eligibility.

## 2026-08-12T17:22:20Z — focused post-fix gate green

- The targeted Rust eligibility test passed (`1 passed; 0 failed`).
- A fresh release binary from the staged worker tree (`a833f21f...`) passed the naked-default focused cell: 20/20 requests, 18 hits and two cold misses, all exactly 60 tokens with `finish_reason=length`, 1,200 client/engine output tokens, zero counter drift, zero admission/OOM events, and zero routed-MoE carried prime-batch log entries.
- The fixed cold-miss text hash matches the serial rollback control (`b723be26...be1`), while the failing carried path's hash was `1ecc62ea...730f` before it selected EOS. This is output-path evidence, not a performance comparison.
- `research/coldfix-20260812/run-local-battery.sh` now freezes the remaining required local gates and captures their raw logs under one dual-lock 5090 hold.

## 2026-08-12T17:32:46Z — first full-battery attempt failed closed on model discovery

- The fresh release build and complete `cargo test` suite passed, including the new routed-MoE eligibility unit test (219/219 server tests).
- `kernel-check` then exited nonzero because the first driver revision set `MEMRA_KC_MODELS_DIR=/data/ai-ml/hf-models`. That override is an authoritative *flat* basename directory, while this rig's artifacts live in nested subdirectories, so model-backed cells were loudly skipped and required `DUAL-BATCHED-AUX` was missing.
- This is a battery-protocol failure, not a product failure. The failed receipts remain sealed under `raw/battery/`; the driver now follows `tools/local-ci.sh` and uses kernel-check's normal 5090 artifact resolver. The clean rerun will write a distinct `raw/battery-r2/` directory.

## 2026-08-12T17:43:29Z — corrected full local battery green

- `kernel-check`: `ALL GREEN (106 cells, 1 skipped)` with both required manifests satisfied; the sole skip is the optional unconfigured sigrouter served replay.
- `run-gen`: Q27 and Q35 both report prefill/decode argmax MATCH and batched-prime/tokenwise argmax MATCH.
- `run-spec`: Q27 and Q35 each pass self-consistency at every K=1..8.
- `serve-smoke`: 0 failed. The new Q35 mixed-c4 cell completed 20/20 requests at exactly 60 tokens and confirmed zero routed-MoE carried prime-batch lines.
- c=64 stress: 64/64 streams complete and well formed, worker alive, server log clean.
- Full `cargo test`: aggregate 440 passed, 0 failed, 2 GPU-explicit ignored tests; the new eligibility test is green.
- Raw evidence is sealed by `raw/MANIFEST.sha256` (`cf5d4f9c...e073401`). `RESULTS.md` recommends v0.81.1, subject to the repository's designated Vast 2x PRO 6000 pre-release battery; this lane does not merge, tag, or push.

## 2026-08-12T18:07:00Z — requal2 steering added Q27 sold-tail acceptance

- Requal2's final N=5 evidence adds a second acceptance condition: Q27 mixed-c4 full-cache-hit
  TTFT p95 regressed from the 21.6 ms sold envelope to 269.1 ms while p50 stayed 18.3 ms.
- The five >100 ms hit rows are 269-299 ms and each starts while an uncached 4,860-token request
  is being synchronously primed. Server order shows the hit is admitted only around the cold
  seed insertion; there is no OOM, admission defer, or false cache-role classification.
- A scheduler fence is staged: a newly admitted full-cache hit gets its first decode before
  unrelated cold prefill, and a completed interactive request opens the existing 4 ms batch-
  formation grace before cold prefill resumes. Cold-only continuation batching remains intact,
  so the Q27 c16 knee should survive; this is a target-rig hypothesis until the frozen box1 N=5
  replay proves it.
- The earlier local battery predates this fairness addition and will be rerun on the final tree.
  Box1 acceptance remains pending: Q35 40/40 at c4, Q27 hit-TTFT p95 in the sold envelope class,
  and the full requal2 exactness/performance campaign.

## 2026-08-12T18:14:18Z — final-tree local battery green

- The final-tree rerun is sealed under `raw/battery-r3/` (manifest
  `d3df2fe39dc9ced8aa3b17c642347544942ad73a3265dba98448c1744c05517c`).
- Full `cargo test` aggregates to 441 passed, 0 failed, and two GPU-explicit ignores. The extra
  passing test is the cache-hit first-token fence predicate added with the Q27 fairness fix.
- `kernel-check` is `ALL GREEN (106 cells, 1 skipped)`; both Q27/Q35 `run-gen` checks report
  prefill/decode and batched-prime/tokenwise argmax `MATCH`; both `run-spec` checks pass K=1..8.
- `serve-smoke` reports zero failures, including 20/20 exact 60-token Q35 mixed-c4 responses and
  zero routed-MoE carried batches. The c=64 stress gate completed 64/64 well-formed streams with
  the worker alive and no captured fatal marker.
- `tok-check` resolves Q35 token id 248046 as `<|im_end|>`, and the server declares the same id as
  EOS. Thus the repeated-carried path's constant 26th generated token was the real turn-end/EOS
  token, not a 26-token scheduler chunk boundary; the false selection is the regression.

## 2026-08-12T18:22:23Z — box1 driver assertion corrected, full rerun started

- The first box1 attempt passed the standard gates plus Q27 rep 1 and Q35 rep 1. Q27 c4 was
  20/20 exact with hit-TTFT p95 18.939 ms and its clean throughput rose through c16. Q35 was
  exact through the complete c1..48 grid (16/16 cells, 460/460 requests), with zero carried
  prime-batch entries; its mixed throughput rose through c40 and declined at c48.
- The upstream requal2 driver then exited after the clean Q35 verdict because its post-run line
  231 required every model to emit at least one `[prime-batch]` line. That assertion predates the
  hotfix and is stale for a routed-MoE workload whose carried batches are intentionally disabled.
  No model/runtime failure marker was captured; the incomplete attempt remains raw evidence and
  is not used for acceptance.
- A sealed one-hunk driver copy changes only that post-run assertion: dense Q27 must retain a
  positive prime-batch count, while Q35 must have zero `carried>0` prime batches (whole-fresh
  batching remains allowed). Frozen replay `91eac7...`, workload `85597a...`, helpers, artifacts,
  source commit `065a705d6`, and runtime binaries are unchanged.
- A fresh full N=5 campaign acquired the single box1 GPU lock at 18:21:54Z. Acceptance will use
  only this uninterrupted rerun.

## 2026-08-12T19:15:28Z — uninterrupted box1 N=5 campaign sealed

- The corrected driver completed all ten alternating model boots and recorded
  `REQUAL2_COMPLETE` plus `DRIVER_EXIT rc=0`. The single GPU lock ran from 18:21:54Z to
  19:15:28Z; post-run compute applications, ports 18427/18435, and the lock were clear.
- Q27 completed 70/70 cells and 1,400/1,400 requests at exactly 60 tokens. Mixed-c4 pooled
  hit-TTFT is p50 18.497 / p95 19.820 ms, restoring the sold 18.573/21.565 ms envelope; its
  clean median-throughput path rises through c16 and declines at c20. Dense continuation
  batching remained live with 825 recorded calls.
- Q35 completed 80/80 cells and 2,300/2,300 requests at exactly 60 tokens, including 200/200
  c4 cold+mixed requests across N=5. All 40 required base cells are clean and all five server
  logs report zero prime-batch calls, hence zero routed-MoE carried entries.
- All 3,700 requests are full length; the 150 cells are clean; serial-cache exactness passes;
  admission-session defers, admission-VRAM defers, and step-OOM parks sum to zero. The thermal
  regime peaked at 68 C, 2,422 MHz, 525.13 W, and 77,845 MiB.
- The copied remote manifest verifies locally at
  `5804c57af75bd5b738a77a0ba175eb92f4e2684da1fe3b93f058b18ab2d8b727`.

## 2026-08-12T19:18:00Z — frozen reduction and release boundary

- Frozen analyzer `f4526b...`, old sold summary `e152c4...`, reducer `eb2319...`, and workload
  `85597a...` produced Q27 `SELLABLE` and Q35 `SELLABLE`. Every target criterion passes.
- The overall reducer verdict remains `P0_REGRESSION` for one strict comparison: Q27 mixed-c4
  output median 144.245 versus 144.462 tok/s (`-0.217`, `-0.150%`). The new median lies inside
  the old N=5 raw range, and the old/new runs are not a same-window interleaved A/B, so no
  causal throughput claim is made and the reducer's red flag is not waived.
- This shipped-defect repair is a v0.81.1 patch, not v0.82.0, but this lane does not authorize a
  tag. The orchestrator must resolve the strict output comparison and run the designated Vast
  2x RTX PRO 6000 pre-release exactness battery before merge/tag.
- `analysis.json` hashes to `092c8e42489009984c363afe992482f4ef59799797ada00ed4d11a0eea41c2e3`.
  The complete lane raw seal covers 256 files and hashes to
  `263d3db1f02b0efce19532936b6104a47841165c4f43083f36e847070d3e3153`.

## 2026-08-12T22:46:05+03:00 — v0.81.1 coldship lane opened

- Started `lane/cx-coldship` at
  `83f77180dddf193dbf22e586e3d100ac5ec60cee`. The coldfix P0 ship decision and PATCH
  classification are accepted as directed; the Q27 c4 throughput observation is non-gating.
- Scope locked: merge `lane/cx-coldfix`, bump the workspace and internal dependency pins to
  0.81.1, keep the generated perf board unchanged, run the full box1 battery under
  `/tmp/memra-gpu.lock`, commit the raw evidence, and stop without tag, push, or main merge.
- Status: merge, version audit, local release build, box1 deployment, battery, tenant-clean
  shutdown, and evidence commit pending.

## 2026-08-12T22:51:58+03:00 — merge and local release gate green

- Merged `lane/cx-coldfix` at `33975c134dca9efac91e09af793af4439a79e333` through merge
  commit `1a210b38e`. The only conflict was this add/add journal; the complete coldfix history and
  existing `RESULTS.md` were retained, then this coldship section was appended.
- Bumped `[workspace.package]` and all eight exact intra-workspace dependency pins from 0.81.0
  to 0.81.1. `cargo metadata --locked` resolves all nine publishable `memra-*` packages as
  0.81.1, with no residual 0.81.0 entry in any workspace Cargo manifest or `Cargo.lock`.
- Local `cargo build --release` passed in 2m36s with CUDA 13.1 auto-detecting sm_120a. The
  generated lockfile resolves cleanly, and `python3 tools/update-perf-board.py --check` reports
  `perf board is up to date` with no board file changed.
- Status: box1 preflight, deployment/build, full two-model battery, raw-log seal, tenant-clean
  shutdown, and final evidence commit pending.

## 2026-08-12T23:09:46+03:00 — box1 pre-release battery stopped at kernel-check

- Deployed clean source `7538f044d4080f075bdd20bccb70392ea34cd5da` to the designated
  box1 host `<private-host-redacted>`. The fresh CUDA 13.2 / sm_120a release build passed in 4m01s,
  and all nine publishable workspace packages resolved as 0.81.1.
- Fresh binary SHA-256 values: `kernel-check`
  `e9ed5fef3043aac576d445d0f363f707c1f376513b92154f730ef173a9092f44`, `run-gen`
  `6aa44a1abda8b6458f43b131e016f69a4ca073ef94a7f0bb5c003f5582a5b8a7`, `run-spec`
  `6809049f6048afc8d5b790e5085faa7d2b856f1fc567d497362bf728e36d7a89`, and
  `memra-server` `b7e3e6fecca80e88c467d1d84ee2b9b495ae3eebc0c091b96b52226355d280b7`.
- The first correctness gate exited 1, so the battery stopped and no later gate ran. Captured
  output, verbatim:

  ```text
  SKIP DUAL-BATCHED-AUX (missing model Qwen3.5-9B-NVFP4-MTP-GGUF.gguf under MEMRA_KC_MODELS_DIR=/opt/scratch/nvme/cx-requal/models)
  MISSING REQUIRED CELL DUAL-BATCHED-AUX
  Error: "1 required cell(s) missing"
  ```

- Failure-run thermal sample counts were N=380 per GPU. GPU0 covered 26--32 C, 172--2,415 MHz,
  at most 96.47 W / 21,493 MiB / 22% utilization; idle GPU1 covered 26--27 C, 172--180 MHz,
  at most 34.16 W / 0 MiB / 0% utilization.
- Tenant-clean verification passed after the stop: the flock was released and reacquired, both
  GPUs were back at 0 MiB / 0% with no compute apps, no `memra-server`, `kernel-check`,
  `run-gen`, or `run-spec` process remained, and ports 8177/8179/18427/18435 were clear.
- Partial raw evidence is sealed at `raw/coldship-box1/`; its manifest hashes to
  `b90004f8fd96c2f678efe6e4066e51a7cd003f607aeb0c40b5d7b8c736cdee48`. Release verdict:
  **FAIL / blocked at kernel-check**. No retry, tag, push, main merge, or perf-board change was
  performed.

## 2026-08-12T23:33:03+03:00 — box1 retry r2 full pre-release battery green

- Re-deployed the exact lane head `65264b1095caa548f0b4e7f9ba71e9ff320c4830` and ran the
  committed driver from the top into the distinct `raw/coldship-box1-r2/` directory. Before the
  run, the repaired `Qwen3.5-9B-NVFP4-MTP-GGUF.gguf` symlink resolved to `/scratch-models/` and
  hashed to `52c9cceb190055e0591a9a30c21f7200572eaf3ff1c59f6e9a1eda838a8f39de`.
- The fresh CUDA 13.2 / sm_120a release build passed in 4m04s; all nine publishable workspace
  packages resolved as 0.81.1. Fresh binary SHA-256 values: `kernel-check`
  `46ffb4886c0b72ec334c1ee994b50b15e96caea4ca94fac63cbe55dcf1aebb34`, `run-gen`
  `44214c73c2b41380e0e9736cbad62123d3d0b6e96f120155651643d6de68d4cb`, `run-spec`
  `8d072d1e9641991c7c4a44f6ba76e0da12731c87e71871a9afc33af08127a77b`, and
  `memra-server` `22c925549ee9d5caeae10662f96f29a879d22487e6cfbb903a114ef1d6530046`.
- Full battery N=1 passed every required gate: `kernel-check` reported `ALL GREEN (100 cells,
  5 skipped)` with both required manifests satisfied and `DUAL-BATCHED-AUX` OK; Q27 and Q35
  `run-gen` each reported prefill/decode and batched-prime/tokenwise argmax `MATCH`; Q27 and Q35
  `run-spec` each passed self-consistency for all K=1..8 (one run per K).
- `serve-smoke` reported 0 failed. Its Q35 mixed-c4 regression cell completed 20/20 requests at
  exactly 60 tokens (1,200 completion tokens total), and the routed-MoE server log contained zero
  `carried>0` prime-batch lines. The c=64 stress completed 64/64 well-formed streams with the
  worker alive and server log clean; the captured wall-time distribution was p50 41.4s, p95
  45.3s, max 45.7s. The explicit server failure-signature scan is empty.
- Thermal sampling recorded N=519 one-second samples per GPU. Active GPU0 covered 26--50 C,
  172--2,415 MHz, at most 405.80 W / 60,181 MiB / 100% utilization; reserved idle GPU1 covered
  26--27 C, 172--180 MHz, at most 34.41 W / 1 MiB / 0% utilization.
- Tenant-clean verification passed independently after shutdown: the flock released and
  reacquired, both GPUs were at 0 MiB / 0% with no compute apps, no `memra-server`,
  `kernel-check`, `run-gen`, or `run-spec` process remained, and ports 8177/8179/18427/18435
  were clear.
- The 37 raw artifacts verify against `raw/coldship-box1-r2/MANIFEST.sha256`, whose SHA-256 is
  `7b7bc5e513051df66f398f93f4c080042a6a62f588290d3ab32d076b09725a89`. Release verdict:
  **PASS / ready for orchestrator merge and tag**. This lane did not tag, push, merge to main,
  or change the perf board.

## 2026-08-12T20:42:22Z — carried-prime default-DENY follow-up opened

- Started `lane/cx-fencehard` in `/home/avifenesh/projects/wt-cx-fencehard` at
  `b8fc24dbe9bab2e0a414b4b3c99d6aa719b9d92a`, exactly the local and live-origin `v0.81.1`
  tag. The worktree was clean before this progress entry.
- Finding accepted: the shipped `!arch.is_moe()` fence is fail-open for both a new named `Arch`
  variant omitted from the MoE allowlist and an unknown GGUF architecture parsed as
  `Arch::Other`. This follow-up hardens classification; it does not change any current named
  architecture's behavior.
- Design: add an exhaustive, no-wildcard carried-prime eligibility method beside the existing
  `Arch` classifications. `Qwen3`, `Qwen35`, and `Llama` explicitly opt in; `Qwen3Moe`,
  `Qwen35Moe`, `Olmoe`, `MinimaxM3`, `Hy3`, `Gemma4`, `GlmDsa`, `Step35`, and `Other(_)`
  explicitly deny. This is cleaner than deriving scheduler safety from `is_moe()`: architecture
  bring-up must classify a new enum variant before the crate can compile, while unknown strings
  take the safe default. Q27 parses as `Qwen35`, so its shipped carried-prime c=16 knee remains
  eligible.
- Gates: replace the worker regression test with explicit assertions for every current named
  architecture plus parsed/direct `Other`; run cargo tests locally; then run the full locked box1
  battery with raw logs under `raw/fencehard-box1/` and verify tenant-clean shutdown. No board
  edit, tag, push, or merge is authorized.

## 2026-08-12T20:48:46Z — implementation and local tests green

- Added `Arch::carried_prime_batch_eligible()` as the exhaustive opt-in classification and made
  the worker fence consume it. The match has no wildcard: all 11 current named variants are
  classified, `Other(_)` is denied, and a future enum variant is a compile error until bring-up
  makes an explicit decision.
- Replaced the old two-list regression with explicit assertions for all three current dense
  arches, all eight current MoE arches, Q27's production `qwen3next` parse alias, an unknown
  parsed string, and direct `Arch::Other`. The focused test passed 1/1.
- Full workspace `cargo test` passed with exit 0; `memra-server` passed 220/220 and `memra-gguf`
  passed 85/85. The complete local console log is retained at
  `/tmp/cx-fencehard-cargo-test.log` with SHA-256
  `116945dfb7e6289479c71ce3ca5ff85d3737d829e105e45e9ef39bffb04d02a5`.
- Live box1 preflight resolved the cloud instance `<instance-id>` at `<rented-box-ip>` and used its
  pinned `<keypair>` key with `IdentitiesOnly=yes`. Both RTX PRO 6000 GPUs reported 0 MiB,
  0% utilization; the global lock was immediately acquirable, the frozen model directory is
  present, and the dedicated `/opt/scratch/nvme/cx-fencehard` target is absent. No GPU gate has run
  yet.

## 2026-08-12T21:02:58Z — box1 full battery ALL GREEN

- Deployed the exact local config/worker blobs to clean box1 snapshot `e731d45d` and built from
  scratch with Rust 1.97.1 and CUDA 13.2 under one `/tmp/memra-gpu.lock` hold. The snapshot parent
  `65264b109` and tagged `v0.81.1` have byte-identical Cargo, crate, tool, workflow, and gate-prompt
  inputs; the tested config and worker blobs are `27fd884d...` and `2f9788a1...`, identical to this
  lane. Battery shape is N=1 correctness, not a performance claim.
- `kernel-check`: `ALL GREEN (100 cells, 5 skipped)` with both required manifests satisfied.
  Q27 and Q35 `run-gen` each passed prefill/decode and batched-prime/tokenwise argmax `MATCH`.
  Q27 and Q35 `run-spec` each passed all eight K=1..8 self-consistency rows.
- `serve-smoke`: 0 failed. The Q35 mixed-c4 regression cell completed 20/20 requests at exactly
  60 tokens, 1,200 completion tokens total, and its server log contained zero routed-MoE
  carried-prime batches. The Q27 c=64 stress completed 64/64 well-formed length-finished streams
  with the worker alive and its server-failure signature scan empty.
- Thermal capture contains N=518 one-second samples per GPU. Active GPU0 covered 26--50 C,
  172--2,415 MHz, at most 413.44 W / 60,181 MiB / 100% utilization; reserved GPU1 covered
  26--27 C, 172--180 MHz, at most 33.94 W / 0 MiB / 0% utilization.
- Tenant-clean shutdown passed in the driver before and after unlock, then passed an independent
  lock reacquisition at `2026-08-12T21:01:35Z`: both GPUs were 0 MiB / 0%, with no compute apps,
  runtime processes, or listeners on ports 8177/8179/18427/18435.
- The 38 raw artifacts under `raw/fencehard-box1/` verify against `MANIFEST.sha256`; the manifest
  hashes to `86bed9be272254186a72e870911471b0072b450a2afd677b3178903f4b34d81b`. The generated perf
  board check remains current and no board file changed. Final lane commit is the only remaining
  action; no tag, push, or merge will be performed.

## 2026-08-13T00:34:38+03:00 — qwen3next alias runtime-belt lane opened

- Started `lane/cx-fencealias` in `/home/avifenesh/projects/wt-cx-fencealias` at
  `18885ec479d8`, exactly the local `v0.81.2` tag. The worktree was clean before this progress
  entry.
- Premise to verify before shaping the implementation evidence: enumerate public GGUFs carrying
  the `qwen3next` architecture string and inspect their GGUF expert-count metadata to distinguish
  dense aliases from the original hybrid-MoE family.
- Fix boundary is belt-and-braces regardless of the public-artifact result: retain
  `Arch::carried_prime_batch_eligible()` as the compile-time default-deny gate, add the parsed
  model config's expert presence as a runtime deny at the worker call site, and warn without
  rejecting when `qwen3next` resolves through `Arch::Qwen35` while the config carries experts.
- Gates: pin the combined predicate truth table, run the full cargo test suite, then run the full
  box1 battery under `/tmp/memra-gpu.lock` with raw evidence in `raw/fencealias-box1/` and prove
  tenant-clean shutdown. No board edit, tag, push, or merge is authorized; stop after the lane
  commit for the orchestrator's v0.81.3 patch release.

## 2026-08-13T00:39:42+03:00 — public GGUF premise verified from pinned headers

- The initial live Hugging Face GGUF search returned 73 repositories whose indexed
  `general.architecture` is `qwen3next`. Range reads of one pinned GGUF header per repository
  found a positive `*.expert_count` in 71/73 readable production/model fixtures: counts span
  4--512, with 40 repositories at 512. The two exceptions are a deliberately malformed
  modulo-zero PoC with no expert-count key and one currently listed file whose download returns
  HTTP 401. No readable production `qwen3next` artifact in the census was dense.
- Direct `gguf_dump.py --no-tensors` inspection of Qwen's official Instruct file at revision
  `4c8630cf7af926a9c5095cb4bbbbc65d36e20f77` reports
  `general.architecture = 'qwen3next'`, `qwen3next.expert_count = 512`, and
  `qwen3next.expert_used_count = 10`; the official Thinking header reports the same class and
  count. The 32 MiB pinned Instruct prefix hashes to
  `099533daf8ac552fac07e33807f9d14e387ee8bfb166eb57cb359f6184d6f995`.
- Dense control: Unsloth's public Qwen3.5-27B Q4_K_M at revision
  `3221f178a6b842d04f1fb42f1c413534adcc0a6a` reports
  `general.architecture = 'qwen35'` and has no expert-count metadata. Its 32 MiB pinned prefix
  hashes to `1d2dff2006ace4c4ca9bfefebe0cb1bba3566a7c983279713959d2c78fe47687`.
- Implementation consequence from the tagged source: GGUF `ModelConfig::from_gguf` currently
  constructs `cfg.moe` only when the already-aliased `Arch` says MoE, unlike the HF-config path
  which also respects explicit expert metadata. A worker-only `cfg.moe` check would therefore
  remain fail-open for the verified official header. The minimal runtime belt must first make
  positive GGUF `expert_count` authoritative for `cfg.moe`, then combine the existing
  exhaustive Arch opt-in with `expert_count == 0`; this also gives the load warning a truthful
  parsed-config signal.

## 2026-08-13T00:48:37+03:00 — runtime belt implemented, local suite green

- GGUF parsing now retains an explicit expert configuration whenever `expert_count > 0`, even if
  the parsed Arch enum is a dense alias. An expert-bearing literal
  `qwen3next` emits a non-fatal load warning naming the 512-expert fixture, the
  `Arch::Qwen35` dense-enum alias, and the risk to subsystems keyed only on Arch.
- The worker's carried-prime predicate keeps `Arch::carried_prime_batch_eligible()` as its first
  exhaustive gate and additionally denies any parsed config whose `moe.expert_count > 0`.
  Its unit truth table covers dense/no-experts (including explicit zero) as eligible,
  dense/512-experts as denied, all eight named MoE arches as denied, and parsed/direct `Other`
  as denied.
- Focused parser and worker tests each passed 1/1. Full `cargo test` passed with 442 tests, zero
  failures, and the two CUDA-explicit tests ignored as designed; the complete console log is
  `/tmp/cx-fencealias-cargo-test-final.log`, SHA-256
  `78ddad7445e8a8267a8c247185a884eab3e2e50a304efea48b6c89949e71f23d`.
- `python3 tools/update-perf-board.py --check` reports `perf board is up to date`, `git diff
  --check` passes, and no board file changed. Box1 deployment and the full locked battery remain.

## 2026-08-13T00:54:30+03:00 — box1 preflight green; driver adaptation failed before lock

- Box1 preflight at `<rented-box-ip>` found both RTX PRO 6000 GPUs at 0 MiB / 0%, no compute
  applications or gate listeners, `/tmp/memra-gpu.lock` immediately acquirable, and the dedicated
  `/opt/scratch/nvme/cx-fencealias` target absent. The isolated snapshot is based exactly on
  `v0.81.2`; its config and worker blobs are `5753fe4f...09e9` and `3401648c...c91c`, identical
  to the locally tested files.
- The first driver invocation exited 1 during provenance, before lock acquisition, build, or any
  GPU gate. Its mechanically adapted copy still contained the regex-escaped `0\.81\.1` assertions
  on lines 155--156, so `set -e` stopped at the first unmatched grep. There was no emitted gate
  failure line; the verbatim final output was `## box1/cx-fencealias`. Both GPUs remained at
  0 MiB / 0%, the compute-app list was empty, and the lock probe passed afterward.
- The three-file partial receipt is preserved as `raw/fencealias-box1-attempt1/`. The assertion is
  corrected to escaped `0\.81\.2`, both grep checks now match Cargo.toml, and the clean snapshot
  was amended to `cfc43630713c9a504ca2de84e1418526ff84eddf`. A top-to-bottom rerun into the
  required fresh `raw/fencealias-box1/` directory is next.

## 2026-08-13T01:05:34+03:00 — box1 full battery ALL GREEN

- Deployed the exact locally tested config and worker blobs to isolated box1 snapshot
  `cfc43630713c9a504ca2de84e1418526ff84eddf`, whose parent is exactly `v0.81.2` at
  `18885ec479d897a3e8c42b0d408a71fa3edaa708`, then built release binaries from scratch under a
  single `/tmp/memra-gpu.lock` hold. The battery is N=1 correctness evidence, not a performance
  claim.
- `kernel-check` reported `ALL GREEN (100 cells, 5 skipped)`. Q27 and Q35 `run-gen` each passed
  prefill/decode and batched-prime/tokenwise argmax `MATCH`; Q27 and Q35 `run-spec` each passed
  all eight K=1..8 self-consistency rows.
- `serve-smoke` reported 0 failed. Its Q35 mixed-c4 cell completed 20/20 requests at exactly 60
  tokens (1,200 completion tokens total), and the Q35 server log contained zero routed-MoE
  carried-prime batches. The c=64 stress completed 64/64 well-formed streams, kept the worker
  alive, and left the server-failure signature scan empty.
- Thermal capture contains N=498 one-second samples per GPU. Active GPU0 covered 26--50 C,
  172--2,415 MHz, at most 407.66 W / 76,919 MiB / 100% utilization; reserved GPU1 covered
  26--27 C, 172--180 MHz, at most 34.27 W / 0 MiB / 0% utilization.
- Tenant-clean shutdown passed before and after driver unlock, then an independent probe at
  `2026-08-12T22:04:31Z` found both GPUs at 0 MiB / 0%, no compute applications, runtime
  processes, or listeners on ports 8177/8179/18427/18435, and successfully reacquired the lock.
- A final live HF refresh at `2026-08-12T22:08:39Z` found the index had moved from 73 to 74
  `qwen3next` repositories while this lane was running: the new Schackay3 artifact also carries
  512 experts. Pinned range-header reads now show positive expert counts in 72/74 rows, including
  41 at 512; the only non-positive rows remain the malformed modulo-zero PoC and the HTTP 401
  file. The five retained research payloads under `raw/fencealias-hf/` verify against a manifest
  hashing to `9e9a29dbfe5c93843a05d5ba97d8662c85b44705bbf2a2eb12fad2a36ffe48da`.
- The 38 payload artifacts plus `MANIFEST.sha256` under `raw/fencealias-box1/` verify; the
  manifest hashes to `78bcdbca0372ecf7dfec52f58b86907616cf3dc511307aa7d441f221517760f6`.
  The pre-lock driver-adaptation receipt under `raw/fencealias-box1-attempt1/` is separately
  sealed by manifest hash `51caa9cfda9848c46cbb7898315592c13e665cbf46ebcb080262c636251302b8`.
  The generated perf board remains current and no board file changed. The lane commit is the
  only remaining action; no tag, push, or merge will be performed.
