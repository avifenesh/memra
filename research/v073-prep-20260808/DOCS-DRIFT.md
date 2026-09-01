# v0.73 documentation and board audit

Frozen merged train: `v0.72.0..cbe25b75`

Audit surfaces: `README.md`, `docs/PERFORMANCE.md`, `docs/FLAGS.md`, `docs/SERVING.md`, the
directly affected RunPod usage contract, source read sites for merged runtime flags, merged lane
reports/raw summaries, and every generated perf-board surface.

## Confirmed drift fixed in this worktree

| final location | pre-audit drift | resolution |
|---|---|---|
| `README.md:144` | Step35 was still described as bring-up that had not crossed the bar, with chunk dependence and served-spec wiring open. | States the serve-ready trial result, closed/gated execution surfaces, and the still-blocked grouped default. |
| `README.md:206` | The chunk-invariance contract still singled out Step35 as a live exception. | States the current per-architecture chunk and caller-tick gates (`chunkinv35` / `tickinv35`) and keeps the scope narrower than tokenwise-oracle identity. |
| `README.md:259` | The Known gaps list repeated the fixed Step35 chunk defect. | Replaces it with the real open qualification: Lever C is opt-in after the local proof-rig rejection; PP-blind residency and 256K remain open. |
| `README.md:166` | The serving summary described only later prefix-cache hits, not merged same-window fanout or entry pinning. | Adds exact tenant-scoped fanout, pinned lifetime, cached-token accounting, and isolation. |
| `docs/PERFORMANCE.md:539` | The Step note said no throughput existed and listed chunk invariance, served spec, tokenizer byte-check, and reasoning control as open. | Replaces the stale ledger with the v0.73 measured pipeline/dynamic/grouped/serve cells, closed gates, opt-in caveat, and remaining 128K/256K plus residency boundary. |
| `docs/FLAGS.md:8` | Flag coverage counts were from 2026-08-07 and below the frozen tree. | Recounts 442 distinct literal Rust `env::var`/`env::var_os` names and 419 names mentioned in the catalog, including the graveyard. |
| `docs/FLAGS.md:489` | The sm90 diagnostic table duplicated `MEMRA_PRIME_CHUNK` with a `0 (monolithic)` default, contradicting the canonical PP-2 auto/dynamic contract. | Removes the duplicate; the canonical rows remain the single authority for `MEMRA_PRIME_CHUNK`, `MEMRA_PRIME_CHUNK_SCHED`, and Step batching. |
| `docs/SERVING.md:331` | The serving runbook had the placement-aware spec policy but none of the merged PP-2 prime pipeline, dynamic schedule, Step primebatch, Lever-C opt-in, TTFT, serve-ready, or saturation truth. | Adds a separated-mechanism table, trial-config requirements, exact N/protocol labels, cache-budget finding, and the concurrent-prefill refutation. |
| `docs/SERVING.md:829` | Session allocation was documented as evicting every prefix entry, which became false when in-flight leases landed. | Limits emergency eviction to unpinned entries and documents last-release LRU reinsertion. |
| `docs/SERVING.md:833` | Same-window fanout dedup and its tenant/salt security boundary were absent. | Documents the one-compute/N-1-copy path, pinning, exact accounting, rollback, isolation, and N=8 TTFT receipt. |

Adjacent direct-release drift was also pure documentation and fixed:

- `deploy/runpod/API-USAGE.md:90` no longer claims a fixed K=3 serving default; it names the
  request-conditioned K=0/K=2/K=3 policy for the launch shape.
- `deploy/runpod/API-USAGE.md:97` retains the merged privacy rule: serving boxes never enable
  `MEMRA_CONFIDENCE_TRACE` or `MEMRA_DEBUG_SPEC` because the code does not enforce that boundary.

## Flag/dead-doc audit

- Every new production or rollback flag in the frozen train is cataloged: `MEMRA_PRIME_PIPE`,
  `MEMRA_PRIME_CHUNK_SCHED`, `MEMRA_STEP35_PRIME_BATCH`, `MEMRA_MOE_GROUPED`,
  `MEMRA_PREFIX_DEDUP`, `MEMRA_SPEC_K`, and `MEMRA_PREFILL_TICK`. New diagnostic-only
  `MEMRA_TTFT_TRACE`, `MEMRA_TICK_TRACE`, `MEMRA_CONFIDENCE_TRACE`, and `MEMRA_DEBUG_SPEC` are
  also named and scoped.
- No new active runtime flag in the v0.73 train is documented without a read site. Apparent
  zero-read entries found by the string comparison are shorthand families or intentional rows in
  the removed/refuted ledger, not live controls.
- The archkit consolidation is current: `docs/ONBOARDING.md` is canonical and the Qwen 3.8
  runbooks are pointers/worked-example surfaces rather than competing maintained procedures.

## Board check

Verdict: **no missing board-moving number identified; do not edit `current-board.json`.**

Evidence:

- `research/tune-data/current-board.json:2` remains the 2026-08-02 tracked board. The v0.73 train
  does not change that file or either SVG card.
- `python3 tools/update-perf-board.py --check` reports `perf board is up to date` after the docs
  fixes. README's generated blocks remain `README.md:102-142`; the generated PERFORMANCE blocks
  remain `docs/PERFORMANCE.md:222-245` and `docs/PERFORMANCE.md:365-377`. No audit edit entered
  those ranges.
- The only post-v0.72 `research/tune-data/` changes are twelve `perf-ci.jsonl` rows at
  `research/tune-data/perf-ci.jsonl:439-450`. Every row has `window_clean:false`; they are raw
  regression samples, not board promotion evidence.
- The new numeric evidence is Step PP-2/serve-trial, request-policy, prefix-fanout, or scheduler
  anatomy. It does not replace a tracked bare-CLI Qwen/Gemma or H100 comparison cell. Lever C is
  explicitly default-off, dynamic pp4096 is flat, the K-policy mixed workload is neutral, and
  concurrent-prefill lands no production mechanism.
- Step remains a trial-validation paragraph outside the generated supported-model block because
  its bar-passing receipt requires explicit `MEMRA_MOE_GROUPED=1`, whose local default-flip gate
  is red. Promoting Step into `supported_models` would be a separate owner decision, not a
  mechanical consequence of this audit.

## Verification run in this lane

- `git diff --check`: pass.
- `python3 tools/update-perf-board.py --check`: pass.
- All six generated `PERF-*` marker blocks are byte-identical to the frozen target; the board JSON
  and both SVG cards are unchanged.
- Local Markdown targets resolve across all nine changed/new documentation files.
- `bash -n tools/changelog.sh`: pass.
- `bash tools/changelog.sh v0.72.0 cbe25b75`: pass; exact stdout hash recorded in
  `CHANGELOG-DRAFT.md`.
- No GPU job, Rust toolchain action, tag, merge, origin push, or board edit was performed.
