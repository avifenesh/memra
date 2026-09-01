# cx-sellgate progress — 2026-08-12

## Objective

Qualify the private-first Q35 + Q27 pair at the provisional external cap of four
concurrent requests per model on the eu-west 2x RTX PRO 6000 rig. The week-1 gate
passes when at least one sold model shape satisfies exactness, cache accounting,
and latency at `c=4`; Q122 is not a fallback in this lane.

## Frozen gates

- Pin both artifact hashes, runtime commit, and prompt-template hash.
- On this build and pair: `kernel-check` ALL GREEN, `run-gen` argmax MATCH, and
  `run-spec` K=1..8 PASS for Q35 and Q27.
- Replay the frozen 81:1 prompt:completion workload with prefix cache enabled at
  cold and 90% token-weighted-hit shapes for `c=1/2/4/8`.
- At `c=4`, require cache-hit p95 below 2 seconds, all-traffic p50 below 2
  seconds, at least 25% capacity headroom above the cap, and zero cached-token
  accounting drift.
- Report cache hits and misses separately, plus honest overall p90/p99, while
  both models serve simultaneously.
- Commit raw JSONL and a per-model `SELLABLE` / `NOT at c=4` result envelope.

## Execution constraints

- Reuse `research/percard-20260812/` staging, manifests, flock discipline, and
  eu-west harness; reuse the `research/prefixmoney-20260812/` battery patterns.
- Size prefix-cache budget for the frozen workload prefix set (approximately
  343 MB for the 4k-prefix class).
- Launch long work detached from minute one and retrieve receipts frequently
  because the rig is spot-backed.
- No merge, tag, push, perf-board edit, formatting sweep, or hook bypass.

## Start state

- Branch: `lane/cx-sellgate`
- Base/runtime source commit: `79c3c0b2779101c7de89d6f822b9392d03e71702`
- Worktree: `/home/avifenesh/projects/wt-cx-sellgate`
- Steering checked: `~/.lanectl/inbox/cx-sellgate.md`
- Status at open: lane opened; harness audit and live-rig preflight pending.

## Timeline

- 2026-08-12: owner GO received; branch/base/steering verified and immutable
  gate contract recorded before remote execution.
- 2026-08-12: live `origin/main` re-fetched and confirmed unchanged at the
  starting runtime source. The eu-west pair was reachable, GPU-idle, lock-idle,
  and retained both sha-manifested per-card artifacts on local NVMe.
- 2026-08-12: froze `workload.lock.json`: 4,860 prompt tokens and 60 completion
  tokens are exactly 81:1; each scored mixed cycle contains nine full-prompt
  hits and one cold miss, so token-weighted cache coverage is exactly 90% and
  its all-traffic tails contain the real miss population. Eight hot templates
  plus cold churn run under a 4,096 MiB cache budget per model.
- 2026-08-12: added the dual-endpoint replay and eu-west runner. Before scoring,
  the runner also applies the existing prefix-money exactness pattern at 4,374
  cached plus 486 computed prompt tokens, N=3 under the serial cache oracle. The scored
  ladder uses N=5, rotated width order, alternating cold/mixed arms, exact
  response-versus-engine cache-token reconciliation, and same-boot conditional
  extension while mixed throughput rises.
- 2026-08-12: local gates pass: Python compilation, shell syntax, ShellCheck,
  workload invariants, and an 80-request dual fake-server contract replay with
  zero usage/counter drift. Current-source remote build and GPU gates remain
  pending.
- 2026-08-12: incorporated appended owner steering: raw rental is not fallback
  language; a failed card returns to research/SOTA training, while listing may
  continue on a passing card. Commercial context uses the corrected $126/day
  year-1 and $63/day expansion bars and the `tiyuvta.ai/inference` surface.
- 2026-08-12: staged an isolated eu-west checkout at current main
  `79c3c0b2779101c7de89d6f822b9392d03e71702`, reusing the percard artifacts
  without changing its checkout or receipts. The detached CUDA 13.2/sm_120a
  build passed; binary SHA-256 values are retained in
  `raw/setup/runtime-binaries.sha256`.
- 2026-08-12: the first independent current-build kernel battery completed on
  physical GPU 0 with `ALL GREEN (95 cells, 13 skipped)`. Its stable log and
  provenance were copied home while the same battery continued on GPU 1.
- 2026-08-12: the sealed two-card correctness gate passed and its remote
  manifest verified locally. Both physical GPUs reported `ALL GREEN (95 cells,
  13 skipped)`; Q27 and Q35 each reported prefill/decode and batched-prime/
  tokenwise argmax MATCH; both external-drafter runs reported eight K=1..8
  self-consistency PASS rows plus the overall PASS sentinel.
- 2026-08-12: launched the dual-server cache-on campaign detached immediately
  after the gate, under the same `/tmp/memra-gpu.lock` discipline.
- 2026-08-12: attempt 1 stopped before scoring at the inherited prefix-money
  canary. Q27's synthetic `range(...)` prompt completed 60 tokens but emitted no
  visible text for one leg. Q35 passed all serial cold-to-hit byte checks and
  exact cache-token counts, but the canary compared a c=4 batched decode against
  its c=1 cold output and rejected the known cross-config numeric class. No
  CUDA/OOM/panic event occurred and no latency cell was scored. The complete
  failed receipt is retained under `raw/campaign-attempt1/`.
- 2026-08-12: corrected the canary without changing the sell gate: prompts now
  use the repository's established visible-output synthetic family, and cache
  exactness compares cold versus partial/full hits under the same serial decode
  composition. Concurrent output hashes remain recorded, while the documented
  batched-prime near-tie class is not mislabeled as cache corruption. The
  campaign remains passable only when at least one model independently clears
  every required base cell; per-model latency and capacity verdicts stay
  separate.
- 2026-08-12: attempt 2 launched detached under the GPU flock and both models
  passed the serial partial/full prefix-cache gate before scoring. Per model,
  N=3 reconciled 27,702 cached tokens exactly in both `cached_tokens_in` and
  `prefix_cache_hit_tokens`, with six hits, six misses, and byte-identical
  cold-versus-hit outputs. Stable receipts were copied home immediately while
  the dual-model replay continued.
- 2026-08-12: attempt 2 was deliberately terminated after six cells exposed an
  invalid scored fixture. The replay's generator resembled `safe-c8-v1` but did
  not preserve its locked eight variants; arbitrary offsets caused early EOS
  and sub-60-token completions on both models. Cache accounting remained at
  zero drift, but those cells are not the frozen 81:1 shape and are excluded.
  The complete aborted receipt is sealed under `raw/campaign-attempt2/`.
- 2026-08-12: added a pre-score prompt pilot rather than selecting from latency
  results. Fixed offset 105 is the only attempt-2 identity already observed to
  reach the full 60-token cap on both models at c=1 and c=2; it must now pass
  cold c=1/2/4/8, N=3, simultaneously on both cards before it can be frozen for
  scoring. The pilot rejects any early EOS, cache credit, response/counter
  drift, or OOM park and does not emit a sellability measurement.
- 2026-08-12: fixed offset 105 passed the prompt pilot: 90/90 simultaneous
  cold requests across Q27 and Q35, c=1/2/4/8, N=3, each consumed exactly
  4,860 prompt tokens and emitted exactly 60 completion tokens. All 24 cells
  reconciled request and engine counters with zero cache credit and zero OOM
  parks. The sealed qualification manifest and canonical prompt-ID SHA-256 are
  now pinned in `workload.lock.json`; eight isolated hot-cache namespaces carry
  that one qualified prompt identity, while unique namespaces force cold
  misses. This fixes content across cache arms and avoids early-EOS length bias.
- 2026-08-12: the final frozen campaign began and both models again passed the
  serial cache exactness gate. Q27's first c=1/c=2 cold and mixed90 cells are
  clean with exact cached-token reconciliation. Q35 is clean at c=1, but one
  c=2 cached request stopped before 60 tokens and its cell also exposed a
  one-token response-versus-engine output-count mismatch; the cell is invalid
  and will not be blended into a sellable envelope. A live receipt checkpoint
  was copied home under `raw/campaign-scored/` while the per-model sweep
  continued.
- 2026-08-12: added a header-only GGUF metadata hasher so RESULTS can pin the
  exact embedded `tokenizer.chat_template` bytes for both artifacts. It reads
  metadata only and will run after the latency sweep, avoiding storage traffic
  during a scored cell.
- 2026-08-12: retrieved a second live campaign checkpoint at 64/80 base
  model-cells. Q27 remains 32/32 clean. Q35 is 28/32 clean: its c=2 cached
  early-stop/output-counter fault has reproduced in four repetitions, while
  its completed c=1/c=4/c=8 cells remain retained as measurements rather than
  being relabeled as a passing envelope.
- 2026-08-12: completed and retrieved the required 80-cell base ladder. Q27 is
  40/40 clean. Q35 is 35/40 clean because the same mixed c=2 fault reproduced
  in all five repetitions; its c=4 and c=8 measurements are clean. Both
  models' clean mixed medians rose from c=4 to c=8, so the frozen extension
  rule opened c=12. The checkpoint also includes the first c=12 cell pair.
- 2026-08-12: completed and retrieved c=12. Q27 remained 10/10 clean and its
  mixed median rose over c=8, opening c=16. Q35 had another invalid cached
  completion in the fifth c=12 mixed window, so it cannot drive the extension
  decision; c=16 opened solely on Q27's clean rise. The checkpoint includes
  the first c=16 cell pair.
- 2026-08-12: the final campaign sealed PASS after 120 cells at
  c=1/2/4/8/12/16. Q27 is 60/60 clean and its mixed median rose through c=12,
  then fell at c=16; c=24 was therefore not run. Q35 is 54/60 clean and NOT at
  c=4 because all five required mixed c=2 repetitions failed response/output
  integrity. The campaign manifest verified on-box and after retrieval.
- 2026-08-12: pinned the two exact embedded GGUF chat-template strings in a
  separate sealed receipt. Q27's 7,764-byte template is
  `e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259`;
  Q35's 8,057-byte template is
  `55d4931433fe502b794226ee7f4d206a6bdd436ac9f80eb7d8ebb4c639f9ea0c`.
- 2026-08-12: deterministic reduction and independent raw assertions agree:
  overall GO, Q27 SELLABLE at c=4 with a c=12 knee (200% width headroom), and
  Q35 NOT at c=4. Q27 completed 1,200/1,200 scored requests at 60 tokens and
  reconciled all 2,624,400 scored cached tokens in client and both engine
  counters. `RESULTS.md` now carries the customer envelope and evidence limits.

## Final status

- Completed on branch `lane/cx-sellgate`; no merge, tag, push, perf-board edit,
  README number change, formatting sweep, or hook bypass was performed.
- Passing offer surface: Q27 at c=4/model. The proposed Q35+Q27 pair is not yet
  qualified; a two-card private offer now requires the separate Q27-replicas
  gate.
- Canonical derived result: `summary.json`. Canonical scored raw receipt:
  `raw/campaign-scored/`.
