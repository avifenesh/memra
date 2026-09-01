# cx-eosclass progress

## 2026-08-13T02:59:26+03:00 — lane opened

- Branch/worktree: clean `lane/cx-eosclass` in `/home/avifenesh/projects/wt-cx-eosclass`, at exact tag `v0.81.3` (`7cf5fd842ebc76f6e8a82910a8e6d4b864b6b42d`).
- P0: reproduce and root-cause the shared early-EOS numerical class without adding another trigger fence. The first handle is Q27: seed the same prefix working set sequentially and in concurrent batches of eight, then compare cache-hit exactness and the persisted/restored snapshot state for an identical prompt. The second handle is Q35 mixed `c=4` with `MEMRA_MOE_GROUPED=1`.
- Accepted upstream observation, pending a fresh local reproduction in this lane: cachesize attempt 3 produced one full 4,860-token Q27 cache hit that selected EOS after 11/60 tokens with exact request/cache/output counters and no captured CUDA/OOM/panic/fatal marker; the same boot passed 80/80 after sequential seeding. This is evidence of a real failure, not yet evidence for a particular mechanism.
- Starting hypothesis to falsify: prefix snapshots produced under different prime batch configurations are not numerically equivalent. The first comparison target is serialized snapshot bytes and metadata for the same token prefix; if bytes differ, identify the exact field and trace its producer/consumer. If bytes do not differ, capture logits/sampler inputs at the first divergent token and distinguish genuine EOS preference from stale or uninitialized sampler state.
- Reproduction criteria: tee raw server/client logs before parsing; state N and observed failure rate; require HTTP 200 plus `finish_reason=stop` and fewer than 60 completion tokens; resolve the selected token against the server EOS set; retain exact request id, cache role/tokens, counters, process/GPU state, and failure-signature scan.
- Fix criteria: a deterministic regression cell fails on this exact tag and passes after the smallest class-level repair; the repair must preserve the cached-vs-recomputed numerical contract across sequential and concurrent priming. Routed-MoE carried-prime, grouped-MoE, and batched working-set seeding remain fenced/off until separate gates authorize them.
- Required final gates for affected surfaces: `cargo test`, `kernel-check` ALL GREEN, Q27/Q35 `run-gen` argmax MATCH, Q27/Q35 `run-spec` K=1..8 PASS, and `serve-smoke` with zero failed including the Q35 mixed-c4 cell. GPU work uses `flock /tmp/memra-5090.lock`; the existing 210--1,200 MHz thermal cap stays untouched. No box1 scored work, live-serve changes, merge, tag, push, perf-board edit, `cargo fmt`, or `--no-verify`.
- Steering check: no mail/inbox connector is configured in this session and no lane-specific inbox file exists in the worktree. Live user steering remains authoritative.

## 2026-08-13T03:36:00+03:00 — recovered checkpoint and bound new evidence

- Read `~/.lanectl/inbox/cx-eosclass.md` in full after the orchestrator checkpoint. Box1 scored
  work is globally serialized by `/tmp/memra-gpu.lock` because the two PRO 6000s share a PIX path;
  this lane will not use box1 without the orchestrated slot. `cargo fmt` remains forbidden.
- Recovered commit `1212d87e7ab35a13de2c901332da580d0a5128b2`: the Q27 eight-row
  concurrent-seed/full-hit harness and diagnostics-only snapshot/logit receipts are intact. The
  killed nvcc process left only an orphaned scratch directory; no build or server process survived.
- Bound the merged lcprestore evidence at `6249b0096`. It proves source-slice -> restored K/V state
  equality at splits 64/512/2048/4374, but it does **not** compare that state with a genuinely cold
  computation of the same prefix. The candidate suffix-prime path also took 22--34 seconds at the
  failing 512/2048 splits versus about 0.95 seconds genuinely cold, so a different execution shape
  is established while a shared mechanism with early EOS is not. Do not collapse the two classes
  without first-divergence logits/state evidence.
- Native `cargo check -p memra-server` did not reach Rust type checking: nvcc aborted qmatvec with
  the captured line `LLVM ERROR: out of memory`. At the post-failure snapshot no cargo/nvcc process
  remained. `DOCS_RS=1 cargo check -p memra-server` then passed, establishing that the checkpointed
  Rust tracing type-checks; this is not a CUDA-build or runtime gate.

## 2026-08-13T04:40:58+03:00 — restore-vs-batch discriminant checkpoint

- Re-read the recovered cachesize attempt-7 cell rather than repeating either completed passing
  boot. Three sequentially seeded restored Q27 hits in the same c=4 cell produced three different
  60-token/60-token/11-token streams (`prefix_id=85/86/87`); the identical cold prompt also
  produced multiple 60-token hashes in that cell. The `prefix_id=87` hit is still the only EOS
  crossing and still matches attempt 3, but these receipts do not establish restore as the sole
  variable. Exact decode width/row transitions must be captured alongside state bytes.
- Resolved the lcprestore request-3 control-shape anomaly as a harness expectation error, not a
  lookup defect: request 2 cold-primes the full prompt and publishes it, so request 3 correctly
  takes the resulting 4,860-token whole-entry hit. This is separate from the real candidate-vs-cold
  byte mismatches at splits 512/2048.
- Added a trace-free focused reproduction arm: one serial full-length seed followed by one restored
  hit released with three genuinely cold peers. Added diagnostics-only receipts for the restored
  destination's logical K/V, `len_d`, canonical conv/SSM, spare SSM, boundary logits, and the exact
  decode batch width/row that produced each next sample. The reducer now joins seed and restore
  state digests and reports EOS logit rank/source with batch provenance.
- `python3 -m py_compile`, both Python `--help` loads, `bash -n`, and `git diff --check` pass.
  `DOCS_RS=1 CARGO_BUILD_JOBS=1 cargo check -p memra-server` passed; raw log is under
  `raw/build-checkpoint1/`. This is Rust type-check evidence only. The local 5090 is currently held
  by the conforming slrucache battery under `/tmp/memra-5090.lock`; no competing GPU work started.

## 2026-08-13T05:00:01+03:00 — numeric-program A/B queued

- Source review found the class-level program transition already described by the engine:
  `decode_step_batch` deliberately sends eligible B=1 rows through the eager fused trunk while
  B>=2 uses the generic batched trunk; the programs are not bit-identical. The historical
  `research/iso-gap-20260807/` receipts already attribute load-dependent bytes to this transition:
  default solo and loaded streams diverged, while `MEMRA_SERVE_B1FAST=0 MEMRA_SERVE_GS=0` made
  them byte-identical and the loaded default stream equalled the pinned stream. Q35's earlier
  eager-B1 exception similarly removed early EOS at 15/17/25 by keeping both widths on one trunk.
- Attempt-7 timing is consistent with this discriminant without proving it: the restored Q27 hit
  began from persisted boundary logits, was admitted with three cold peers, and could take solo
  decode ticks around their long primes before switching widths as they became decode-ready. The
  focused restore-mix arm therefore records the exact sample-width transition rather than treating
  restore itself as the independent variable.
- Built one release binary for a one-variable trace-free A/B; build completed successfully from
  `dfa94b54e`, binary SHA-256
  `51eb0636b2e15810a203d07c0d5a835c736777d06b715facdb8d7f3cf089c31b`, with raw provenance in
  `raw/build-b1-ab/`. The first launch never loaded a model because optional `env -u` operands were
  ordered after assignments (`env: '-u': No such file or directory`); it is retained and explicitly
  excluded under `raw/pre-b1-default-r5/`, and the runner ordering is fixed.
- The replacement default-B1 run `pre-b1-default-r5b` is queued behind
  `flock /tmp/memra-5090.lock` with a bounded 7,200-second wait. It has not acquired the lease or
  created a measurement directory yet; no duplicate will be started. Once complete, the same binary
  gets the `MEMRA_SERVE_B1FAST=0` control before diagnostics are enabled. GraphSession is inert in
  this 60-token local cell (`MEMRA_GS_MIN` defaults to 384), but remains part of the class-level
  default-policy review because it is another solo-only program that degrades when peers arrive.

## 2026-08-13T05:21:36+03:00 — deterministic Q27 discriminant and trace receipt

- The trace-free one-variable A/B completed on the same exact release binary
  (`51eb0636b2e15810a203d07c0d5a835c736777d06b715facdb8d7f3cf089c31b`). Under the then-default
  eager B=1 policy, five restored target hits produced two hashes while all 15 cold peers produced
  the baseline hash (`raw/pre-b1-default-r5b/`). With `MEMRA_SERVE_B1FAST=0`, all five restored
  targets produced one hash despite varying co-residence (`raw/post-b1off-r5/`). Both arms exited
  zero with no OOM, Xid, panic, or protocol failure; commits `3b2b7f556` and `a02035877` seal the
  raw receipts.
- A controlled width-transition harness removed cold prefill and cache publication from the
  variable: it serially seeded the target and three peers as four distinct full-prefix entries,
  ran a solo target control, then started the restored target before its already-restored peers.
  Across 25 delays from 0 through 600 ms, all 101 post-seed requests were 4,860-token hits, with
  zero evictions, defers, OOM parks, or protocol failures. The target produced five hashes, and at
  50 and 225 ms reproduced the historical failure exactly: HTTP 200, `finish_reason=stop`, 11
  completion tokens, SHA-256
  `ddbbcf35ae93821874be64e15f63feecb2471bbe6ea0c23b63fa00b1e98a9b73`. Commit `ee67811ef`
  seals `raw/pre-widthflip-default-d25/`.
- The diagnostics-enabled replay (`raw/pre-widthflip-trace-d25/`) shifted the timing window and
  produced no early EOS, so it is not counted as another failure trial. It still localizes the
  numeric transition. All 13 target cells ran the persisted boundary sample, one eager B=1 tick,
  then 58 generic B=4 ticks; every target moved from the solo-control hash to one identical loaded
  hash. All 53 restores matched the seed in token count/hash, position, boundary-logit hash, all 48
  recurrent-layer logical hashes, and all 16 populated KV-layer logical hashes. The only aggregate
  KV digest difference is the empty layer-64 representation (one-byte allocation sentinel in the
  source snapshot versus zero logical bytes in the restored cache, both `len=0`), not model state.
  Client exit was zero; the sole failure-signature scan match is the startup gpu-watch banner that
  lists fatal Xid numbers, not an observed Xid. The run used the old exact binary deliberately so
  the trace described the failing program class rather than the source repair.
- The class-level source repair is checkpointed at `7cd4561a6`: generic batched decode is now the
  default at B=1 and B>=2; eager B1 and GraphSession require exact explicit `=1`. Q35-MoE's eager
  exclusion remains defense in depth. Config gates now pin the generic B=1-vs-B=N default, while
  strict/PP eager checks explicitly opt in so the historical diagnostic program stays covered.
  Rust type checking under `DOCS_RS=1`, shell syntax, and diff whitespace checks passed. A native
  post-fix release binary and live A/B remain pending.

## 2026-08-13T05:40:06+03:00 — repaired default passes the Q27 width gate

- Built the repaired release server natively (SHA-256
  `17a222026e08b65f9344407ba9108cb554688c0431365932bb4e78de1033597d`) and the release
  `decode-batch-gate`, `kernel-check`, `run-gen`, and `run-spec` binaries. Commits `5fa016ed6` and
  `e854600a6` seal the build logs and hashes. The pure B1/GraphSession policy tests pass; the
  supported library/unit invocations pass 83 engine tests (one GPU-only ignore) and 221 server
  tests. A broader `DOCS_RS=1 cargo test -p memra-engine -p memra-server` invocation is excluded:
  it tried to link CUDA gate binaries without CUDA objects and stopped on quoted undefined CUDA
  FFI symbols before running tests. Commits `0c1cbbf31` and `66aa2b3c6` retain both the passing
  batteries and that excluded invocation.
- Updated the engine-facing docs and gate comments to state the one-program serving default;
  generated perf-marker blocks and `current-board.json` were not edited.
  `python3 tools/update-perf-board.py --check` passes. The diagnostic eager and GraphSession doors
  remain available only through exact explicit `=1` values.
- The deterministic post-fix width sweep covered 25 peer-arrival delays from 0 through 600 ms.
  All 25 restored target requests completed the full 60 tokens and matched the solo control's sole
  SHA-256 (`5790654979cb98bfacf6d3593b6a5d3def7a5f4bd2a1b8b65e4a6fabe1a72f66`). The
  run admitted and completed all 105 requests, including 101 post-seed full-prefix hits, with zero
  early-EOS targets, protocol failures, cache evictions, admission defers, or OOM parks. Commit
  `57a0249cd` seals `raw/post-widthflip-default-d25/`.
- A diagnostics-only canary is queued behind `/tmp/memra-5090.lock`, not duplicated. It explicitly
  restores the retired eager B1 program, forces host sampling, and sweeps later 300--1,200 ms peer
  arrivals to try to capture the EOS token's actual logit rank. `MEMRA_SERVE_GS=1` is also recorded,
  but GraphSession remains ineligible at this 60-token budget under `MEMRA_GS_MIN=384`; the canary
  therefore isolates the eager-to-batched transition. It is not a repaired-default gate, and its
  instrumentation may perturb the failure window as the earlier trace did.

## 2026-08-13T06:18:12+03:00 — EOS rank captured; grouped class separated

- The diagnostics-only eager canary completed 37 target delays from 300 through 1,200 ms under one
  local GPU lease. It reproduced the exact historical 11-token hash at 450, 600, 625, and 675 ms.
  All four terminal receipts selected EOS id `248046` at generated index 10 from a full 248,320-value
  host logit vector; EOS was rank 1/top 1 with positive margins (1.1117076874 once, 0.4370250702
  three times). Each failure crossed from eager B1 to a batched width before EOS. The trace verifier
  joined 8,748/8,748 emitted-token receipts with no errors. Commit `6dfaf5073` seals the run.
- The current repaired binary then ran the frozen Q35 grouped A/B under one lease. Grouped ON failed
  all 8 serial seeds and all 20 mixed-c4 requests at exactly 25/60 tokens with one hash. Grouped OFF
  passed 8/8 seeds and 20/20 mixed requests at 60/60 with one hash. Both arms reconciled counters,
  had zero defers/OOM parks/carried-prime violations, and had no observed error signature. This
  falsifies the proposed carry-over: grouped dispatch is a separate live correctness class, not
  fixed by the global eager-B1 policy. Commit `10869c5b4` seals the A/B; grouped stays off.
- Next: remove the lane-specific runtime tracing now that its raw receipts are sealed, rebuild the
  clean shipping binary, and run the full two-model local battery on that exact binary. No box1
  work was initiated; target-PRO validation remains an orchestrator-owned shipment gate.

## 2026-08-13T06:35:48+03:00 — diagnostic code retired; clean candidate built and unit-tested

- Removed the lane-only snapshot/logit tracing after sealing its evidence. The resulting source
  diff retains only the default-off eager-B1 and GraphSession policy seams, their tests, and the
  associated gates/docs; commit `7d41113dd` is the cleanup checkpoint.
- Rebuilt the release candidate from the cleaned source. Its `memra-server` SHA-256 is
  `06b264df7ee7c1e4b1982508f573c7ef299d4ed95bc98efc2a4d3e6c322527d9`; commit `2009c15c3`
  seals the authoritative successful build log and all gate-binary hashes.
- Re-ran the supported CPU/unit battery after cleanup: engine 83 passed with one GPU-only ignore;
  server 221 passed. Commit `4760771f8` seals those logs.
- The full Q27+Q35 local GPU battery is running once under `/tmp/memra-5090.lock` against those
  exact cleaned binaries. Q27 `kernel-check` has completed `ALL GREEN` (107 cells, 3 skipped);
  Q35 `kernel-check` is in progress. No box1 work has been initiated.

## 2026-08-13T06:50:15+03:00 — cleaned local battery complete

- The one uninterrupted local-5090 battery completed with `overall.exit=0`; all eleven individual
  stage receipts are zero. Q27/Q35 kernel checks are `ALL GREEN` (107 cells/3 skips and 113/1),
  default config gates pin B1 fast globally/effectively OFF, the explicit eager diagnostic gate is
  green, both `run-gen` arms have two argmax matches, and both `run-spec` arms pass K=1 through K=8.
- Integrated `serve-smoke` rebuilt the cleaned server before use and reported `0 failed` across the
  public API, cache accounting, spec/plain, sampled truncation, affinity, Gemma4, and Q35 mixed-c4
  surfaces. The Q35 cell passed 20/20 at 60 tokens (18 full hits, 2 cold misses), with no defers,
  evictions, OOM parks, golden mismatches, integrity failures, or carried-prime activation.
- Commit `cf9662003` seals the raw logs, every exit receipt, the Q35 JSONL, exact executed hashes,
  and `raw/post-local-gates-clean/VALIDATION.md`. The failure scan's three matches are literal
  gpu-watch/test-description strings, not observed failures.
- The mandatory smoke rebuild's exercised server SHA-256 is
  `e63f9fad6553820a7944687dcf1a8a45326ece039f3384536964b6c560e3594f`, embedding cleanup source
  fingerprint `memra-7d41113dd3f4`. Native rebuild ELF hashes vary in these receipts, so the raw
  per-phase hashes are authoritative. A later post-run restoration rebuild of the untracked local
  target artifact is not part of the evidence battery.
- Local lane work is complete. The Vast 2x RTX PRO 6000 pre-release battery remains an
  orchestrator-owned shipment gate under the box1-global `/tmp/memra-gpu.lock`; this lane has not
  used box1 and will not merge, tag, push, or edit the generated board.
