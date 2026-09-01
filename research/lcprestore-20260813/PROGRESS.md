# Immediate partial-node prefix reuse — progress

Date: 2026-08-13  
Branch: `lane/cx-lcprestore`  
Base: `v0.81.3` (`7cf5fd842ebc76f6e8a82910a8e6d4b864b6b42d`)

## Scope

Restore the longest reusable prefix at an LCP split boundary for the current request, then prime
only its unmatched suffix. The first arm is deliberately narrow: transformer-only entries, plus
hybrid entries whose recurrent state is already captured exactly at the split. Hybrid mid-entry
splits remain unsupported and must take the existing miss path with an observable refusal. Routed
MoE remains disabled unless exact split reuse is proven for that class.

This lane owns the prefix lookup/restore path and its split-boundary exactness/latency receipts. It
does not change cache-budget policy, SLRU policy, generated performance boards, product surfaces,
or live serving infrastructure.

## Required gates

- [x] Cite and bind the feasibility contract and current cache/KV implementation anchors.
- [x] Add a split-boundary cell that compares partial restore with a genuinely cold request,
      byte-for-byte, without borrowing correctness from whole-entry reuse.
- [x] Reject hybrid mid-entry splits observably; gate routed-MoE unless exactness is established.
- [x] `cargo test` green.
- [ ] `kernel-check` ALL GREEN.
- [ ] `run-gen` argmax MATCH.
- [ ] `run-spec` K=1..8 self-consistency PASS.
- [ ] `serve-smoke` 0 failed, including the Q35 c=4 cell; c=64 stress passes.
- [ ] Sequential shared-prefix request-2 TTFT measured before/after, interleaved N>=5 in one lock
      hold with arms alternated; mixed c=4 and knee checked for regression.
- [x] Raw logs captured with `tee` before parsing, including physical GPU identity, foreign-process
      preflight, sample count, and thermal regime.
- [x] Tenant-clean shutdown: lane lock released, assigned GPU at 0 MiB, ports clear.

## Checkpoints

### 2026-08-13 — lane opened

- Verified clean worktree, correct branch, and exact `v0.81.3` base.
- No implementation or measurement claim yet.

### 2026-08-13 — source contract bound and host implementation green

- The cited door is exact: `research/prefixdoors-20260811/FEASIBILITY.md:105-118` names the
  request-2 gap and makes hybrid mid-entry reuse N/A; `:130-140` limits slicing to position-addressed
  transformer K/V and excludes SWA; `:173-195` defines HIRADIX-EXACT-ISO and forbids borrowing a
  whole-entry result.
- The concrete storage contract is `crates/memra-kv/src/lib.rs:213-245`: each full-attention K/V
  plane is a context-linear row array with explicit bytes/token, absolute `len`, device `len_d`,
  and a separately identified optional ring. The recurrent counterexample is the committed-state
  law at `crates/memra-server/src/worker.rs:1073-1084` (the task's older line references resolve to
  the same block on the pinned tag).
- The old admission fallback is preserved at `worker.rs:6743` and the boundary capture at `:7743`.
  The new path selects the longest LCP entry (`:2501`), validates identity/version/plane geometry
  before copying (`:2832`), and logs explicit hybrid/MoE refusals (`:6656`).
- HIRADIX item 3 has device-free corruption fixtures for wrong namespace/model identity, version,
  row layout, truncated planes, wrong entry length, and undersized destinations. HIRADIX item 4
  has a fail-closed support-matrix fixture for hybrid recurrent, routed-MoE, and no-suffix splits.
- Added a live split-boundary cell for 64/512/2048/4374-token boundaries. It hashes the cold
  boundary K/V+position state, source slice, and restored cache separately, then requires request-2
  and request-3 output bytes to equal the genuinely cold control. Boundary logits are explicitly
  not consumed by the supported non-empty-suffix arm; a request ending at an interior split is
  refused rather than sampling from unavailable logits.
- `cargo test --workspace`: all runnable tests passed (**2 GPU-only tests explicitly ignored**);
  `memra-server` itself is **224 passed, 0 failed**. Device evidence is still pending; no
  performance or end-to-end exactness claim is made yet.

### 2026-08-13 — box1 protocol frozen

- The scored transformer cell uses the official dense Gemma-4-12B QAT Q4_0 GGUF, pinned by
  revision and SHA-256, with its default flat K/V allocation (`MEMRA_SWA_RING` remains off) and
  split boundaries 64/512/2048/4374. Each candidate partial restore is checked against a
  genuinely cold request in a fresh namespace; source-slice and restored-cache state hashes are
  paired independently so the result cannot inherit the whole-entry control's correctness.
- The sequential request-2 timing arm is feature-off/on, N=5, alternated by cell under one
  uninterrupted `/tmp/memra-gpu-1.lock` hold. It runs on fresh trace-disabled servers so the
  diagnostic D2H state hashes cannot contaminate TTFT. Q27 mixed-serve replays are likewise
  alternated across five complete frozen repetitions; Q27 hybrid and Q35 routed-MoE refusal
  paths are live negative controls.
- The remote runner checks the source/model hashes before work, records physical card 1 and UUID,
  rejects foreign compute processes before every gate, captures stdout/stderr before reduction,
  and treats the final 0 MiB/ports-clear/lock-release state as part of the result.

### 2026-08-13 — first device attempt stopped before requests

- The initially selected official Qwen3-4B GGUF could not enter `memra-server`, whose pinned
  serving runtime owns `HybridModel` only. The captured failure was `panicked ... not a hybrid
  arch` followed by `FATAL: worker init failed: worker died during init`; no request or score ran.
- Cleanup returned physical GPU 1 to 0 MiB, cleared the lane ports, and released the lock. The
  positive arm moved to the already-supported dense Gemma4 class: it has no recurrent planes,
  uses flat context-linear K/V while the Step35-only ring door is off, and is not a routed-expert
  artifact. Q35-A3B remains explicitly gated by its nonzero expert count.

### 2026-08-13 — exactness STOP; verdict NO-GO

- The dense Gemma4 run reached the split-boundary cell on physical GPU 1
  (`GPU-2b4cf166-fd33-f161-8536-ca04bc72280c`) under the lane lock with no foreign compute
  process. Source-slice and restored-cache hashes matched at all four boundaries (64, 512, 2048,
  and 4374 tokens), so the bounded device copy itself reproduced the selected entry bytes.
- The independently cold request refuted end-to-end byte exactness at two boundaries. The 512-token
  candidate hash was `bf81e8cb4ffc94c306d31d47159bb6a2ef9eb65b519bf41f122e5ae82f1fe525`
  versus cold `719a43f41b407364130580b2f12a8c09e78da460dc25ada2f1781dd436780079`;
  the 2048-token candidate hash was
  `eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df`
  versus cold `223618bfd84e4f30bb454fb7383f139753011e918926af620cf047dda7c136c2`.
- Per the lane's stop rule, the campaign stopped immediately after exactness reduction. The N>=5
  trace-free timing cell, mixed c=4/knee replay, `kernel-check`, `run-gen`, `run-spec`,
  `serve-smoke`, and c=64 stress were **not run**. The trace-enabled N=1 samples are diagnostic
  only and are not a performance result.
- The runner cleanup returned the assigned card to 0 MiB and cleared its ports; a post-run probe at
  `2026-08-12T23:59:57Z` found no compute app, the lane lock available, and no lane listener.
  Full receipts and the verbatim failure strings are in `RESULTS.md` and `raw/`.
