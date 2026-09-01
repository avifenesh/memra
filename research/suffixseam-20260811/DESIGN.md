# Server-Side Suffix-Proposer Seam — Design Document

**Date:** 2026-08-11  
**Lane:** suffixseam-20260811  
**Status:** Design study (read-only, no code changes)

---

## Executive Summary

Model-free speculative candidates from suffix trees over prior outputs, fed into the existing lossless verify path. Suffix proposers win on repeated schema traffic (tool-calling, structured outputs, agent loops) where lexical patterns recur. Complements MTP drafter (which wins on novel text). NeurIPS 2025 + vLLM production evidence: up to 5.3x on SWE-Bench class, 1.8-4.5x Snowflake production (research/spec-landscape-20260810/SURVEY.md §5e).

Core value proposition for memra: zero draft-training cost, zero drafter VRAM, no added load at peak traffic — compatible with c≥4-plain batching, and the one spec class whose economics IMPROVE with repeated-schema traffic. Composes with the adaptive-trim escape loop (DRAFT-REGIME.md): suffix proposals are real prior text, hence in-distribution, and their misses are exactly the escape events the adaptive trim learns from.

---

## 1. Candidate Entry Seam (Losslessness Preserved)

### 1.1 Current Verify Path

Speculative candidates today come from the MTP drafter head (spec.rs:1077-1200 `mtp_head_forward_dev`), producing a K-token draft chain. The verify/accept orchestrator (spec.rs:6300-6330) runs the exactness walk:

1. **Verify forward** produces target logits for the pending + draft tokens (spec.rs:2107-2266 `verify_stage0_issue` + `verify_stage1_finish`).
2. **Accept walk** (spec.rs:6304-6330): compare draft[j] to argmax(verify_logits[j]) for j in 0..K. The first mismatch truncates acceptance at position j; accepted tokens are consumed; bonus = verify_logits[j] is the correction.
3. **Grammar constraint** (if present): truncates acceptance at the first grammar-illegal token (spec.rs:102-149, `SpecConstraint` trait), recomputes that slot as the masked argmax.

The contract: verify distribution is UNCHANGED — eligibility and acceptance rules untouched. Draft tokens are proposals only; the target's own verify logits arbitrate every position bit-identically to plain greedy decode. This is the losslessness guarantee (spec.rs:2-3 header comment).

### 1.2 Suffix Candidate Injection Point

Suffix-tree proposals enter at the SAME seam as MTP drafts: they are token-id sequences handed to the verify/accept orchestrator. The injection point is BEFORE the draft chain loop (spec.rs:6200-6300), where today `draft[0..k_round]` is populated by `mtp_head_forward_dev` calls.

Proposed new arm in `generate_spec` (pseudo-Rust):

```rust
let draft: Vec<u32> = if let Some(suffix_match) = suffix_proposer.match_at(pos, &committed, K) {
    // Suffix tree found a continuation of length suffix_match.len() in the pool.
    // Use it as the draft proposal instead of running the MTP chain.
    suffix_match.tokens
} else {
    // No suffix match or match length < K_min: fall back to MTP drafter.
    mtp_draft_chain(e, mtp, last_token, h_seed, K, ...)
};
// Verify + accept walk proceeds identically — no code change below this point.
```

The verify forward (stage0/stage1) and accept walk are OBLIVIOUS to the proposal source. This preserves losslessness: suffix proposals are verified bit-identically to MTP drafts, and the target's argmax corrections override any suffix mismatch exactly as they do for MTP mismatches.

### 1.3 Composition with MTP

Three strategies for suffix vs MTP selection:

**A. Suffix-first with MTP fallback** (simplest, zero training overhead):
- On every round, query the suffix tree for a match of length ≥ K_min (e.g. K_min=2).
- If found: use suffix tokens as the draft, skip MTP chain.
- If not found: run MTP chain as today.
- Measured by vLLM: suffix decoding "no extra draft model; no added workload at peak traffic" (SURVEY.md §5e).

**B. Length-weighted selection** (Snowflake production strategy):
- Suffix match found AND len(suffix_match) ≥ len(expected_mtp_chain): use suffix.
- Otherwise: use MTP.
- Rationale: longer drafts amortize verify cost better on MoE/PP-2 shapes (marginal verify column ≈ 7.5-8.7 ms on Step-3.7 PP-2, SURVEY.md §0).

**C. Dual-source proposals** (increment 2+, deferred):
- Run BOTH suffix and MTP, merge proposals into a tree structure (suffix spine + MTP alternatives), verify with tree attention.
- Requires tree-masked verify kernel (SURVEY.md §6, shelf design research/gemma4-bringup/TREE-DRAFT-DESIGN.md).
- Not increment 0: tree verify on PP-2 is anti-leverage (7.5-8.7 ms per column vs +0.05-0.11 tokens/round for depth-1 siblings, SURVEY.md §6 arithmetic).

**Increment 0 adopts strategy A**: suffix-first with MTP fallback, zero new verify-path code.

---

## 2. Pool Structure

### 2.1 Global + Per-Request Suffix Trees

Two-tier pool, following vLLM's `method=suffix` design:

**Global tree** (cross-request):
- Accumulated from committed outputs of ALL requests in the same (model, cache_ns) pool.
- Bounds: cap at N_global tokens (e.g. 256k-512k tokens, ~1-2 MB index overhead).
- Eviction: LRU at the tree-node level (least-recently-matched branches pruned first), or fixed sliding window (drop nodes older than T requests).
- Lookup: given the last M committed tokens (context window, M ≤ 64), find the longest suffix-tree continuation.

**Per-request tree** (session-local):
- Built from THIS request's own committed tokens (turn-by-turn in a conversation).
- No eviction within the request; dropped at request retire.
- Lookup: same suffix-match algorithm as global, scoped to this request's tree.
- Composition: query per-request tree first (exact conversation repeat), then global tree (cross-request pattern reuse).

**Memory bound enforcement** (server-side):
- Global pool sized in `worker.rs` state, per PoolKey (model, cache_ns) — same isolation tier as continuation pool (worker.rs:2354-2356 `Session::pool_key`).
- Per-request trees held in `Session` struct (worker.rs:2229-2350), allocated at admit, dropped at retire.
- Honest capacity gate: if global pool + active per-request pools exceed VRAM budget, defer new admits (same class as step-OOM park, worker.rs:2265-2274).

### 2.2 Tenant Isolation (PC-ISO)

Suffix pools obey the same cache-namespace (cache_ns) isolation as every other cross-request KV reuse tier (worker.rs:236-237 `Request::cache_ns`, worker.rs:2235 `Session::cache_ns`). The isolation contract:

- `cache_ns = ""` (default): single-tenant namespace, byte-identical to pre-PC-ISO behavior.
- `cache_ns = "tenant-X"`: isolated pool, no cross-tenant suffix leakage.
- Suffix tree keys: `(model, cache_ns)` → separate global trees per namespace.
- vLLM's `cache_salt` design (PC-ISO lane reference, 2026-08-02) applies verbatim: the salt scopes prefix cache, continuation pool, and spec pool. Suffix trees are the fourth reuse tier, scoped identically.

**Implementation**: add a `suffix_pools: HashMap<PoolKey, SuffixTree>` to the worker's per-model state (worker.rs:76-91 `LoadedModel` or adjacent). At request admit, `Session` gets a reference to `suffix_pools[session.pool_key()]` for the global tree, plus its own `session_suffix: SuffixTree` for per-request state.

### 2.3 Re-Key Readiness (Oilbird Prep)

Increment 2 target: hidden-state re-keying (arXiv 2608.03839 "Oilbird"). Observation from the paper: on tool-calling traffic, ~half of what exact-match suffix drafters miss is IN the pool but unreachable by lexical addressing. A second draft source re-keys the SAME pool by the verifier's already-computed hidden states at committed tokens, merged into the lexical tree: +24-29% accepted length at matched budget, 4.4x API-Bench.

**Design requirement**: the suffix tree index must support DUAL lookup modes:

1. **Lexical lookup** (increment 0/1): given token sequence [t₀, t₁, ..., t_M], return longest suffix-tree continuation.
2. **Semantic lookup** (increment 2): given hidden-state vector h (from verify forward at a committed position), return nearest-neighbor suffix-tree node by cosine similarity, continue from that node.

**Index structure** (increment 2 prep, no increment 0 blocker):
- Store both token ids AND hidden-state vectors at tree nodes (hidden states from verify forward, spec.rs:6401-6407 `vh_seed` extraction from verify `vx` buffer).
- Lexical index: standard trie (token-id edges).
- Semantic index: quantized vector index over stored hidden states (HNSW / IVF-PQ class, or simpler: bucketed cosine clusters if the pool is small).
- At admit, the worker allocates both indices; increment 0 populates only the lexical trie.

**Increment 0 decision**: suffix tree holds token ids only (pure lexical trie). Hidden-state storage deferred to increment 2, so increment 0 pays zero hidden-state memory overhead and zero semantic-lookup complexity. The trie structure itself (token-id nodes + child pointers) is re-key-ready: adding hidden-state vectors is an additive field, not a rebuild.

---

## 3. Increment Plan

### Increment 0: Single-Card A/B on Dogfood Traffic

**Goal**: Measure suffix-proposer acceptance and overhead on memra's own dogfood traffic (agent loops, repeated tool schemas).

**Scope**:
- Single-card only (no PP-2 yet).
- Strategy A (suffix-first with MTP fallback).
- Lexical trie only (no hidden-state re-key).
- Global pool + per-request pools, bounded at 256k tokens global / 64k per-request.
- No serve-surface flag: run as an internal A/B via `MEMRA_SUFFIX_POOL=1` (off by default).

**Implementation bill**:
- Trie data structure (Rust): `struct SuffixTree { root: Node, size_tokens: usize }` with `fn longest_match(&self, context: &[u32], max_len: usize) -> Option<Vec<u32>>`.
- Worker integration: `HashMap<PoolKey, SuffixTree>` in per-model state; `Session.session_suffix: SuffixTree`.
- Draft-source switch in `generate_spec` (spec.rs:6200-6300): query suffix tree before MTP chain, fall back if no match.
- Pool update hook: on commit (spec.rs:6365-6385), append accepted tokens + bonus to both global and per-request suffix trees.
- VRAM accounting: count suffix pool bytes in the worker's capacity gate (same tier as continuation pool).

**Measurement protocol** (board-protocol rules apply):
- Corpus: memra's own dogfood traffic (agent loops calling memra, recorded request/response pairs from ~/.hermes/sessions or live serve logs). Corpus must include tool-call schemas, multi-turn conversations, and repeated JSON structures.
- Metrics: acceptance rate (suffix-proposed tokens accepted by verify), tokens/round, e2e tok/s, suffix-hit rate (fraction of rounds where suffix matched ≥ K_min).
- Baseline: MTP-only (current main branch).
- A/B: suffix+MTP vs MTP-only, interleaved N≥2, power pinned, thermal-stable window validated (research/benchmarks.md rules).
- Kill criterion (see §4): if suffix hit rate < 10% OR suffix acceptance < 50% of MTP acceptance OR e2e tok/s regression > 2%, abandon the lane.

**Gates before increment 1**:
1. `run-spec` K=1..8 self-consistency PASS (suffix proposals must not break exactness).
2. Acceptance > 0 on dogfood corpus (suffix-hit rate > 0).
3. E2e tok/s ≥ baseline on at least one measured workload class (tool-call heavy).

### Increment 1: Serve-Surface Integration

**Goal**: Expose suffix proposer as a serve-time option, flag-gated, for external workloads to measure.

**Scope**:
- Single-card + PP-2 (PP-2 inherits the Rank-1 stage-idle fill dependency, see §3.3).
- Global + per-request pools, configurable bounds (`MEMRA_SUFFIX_POOL_GLOBAL` / `MEMRA_SUFFIX_POOL_SESSION`).
- Per-request opt-in via body field `suffix_pool: bool` (default false) OR environment `MEMRA_SUFFIX_POOL=1` (default for all requests).
- Acceptance telemetry: extend `SpecUsage` (worker.rs:206-211) with `suffix_hit_rounds: u64, suffix_accepted: u64` so clients can measure suffix contribution in `usage.spec`.

**Implementation bill**:
- Environment variable parsing + per-request body field.
- Telemetry counters in `SpecTelemetry` (spec.rs:272-291): add `suffix_proposed`, `suffix_accepted` fields.
- Worker metrics endpoint (`/metrics`): expose per-model suffix pool size, hit rate, acceptance rate.

**Receipt before default**: measure on ≥3 external workloads (SWE-Bench class if accessible, tool-calling benchmarks, multi-turn chat). If e2e tok/s ≥ 1.1x baseline on tool-heavy workloads AND no regression on freeform generation, promote to default-on (remove flag, suffix pool always enabled). Otherwise, keep flag-gated or kill.

### Increment 2: Hidden-State Re-Key (Oilbird)

**Goal**: +24-29% accepted length via semantic lookup (arXiv 2608.03839 evidence).

**Scope**:
- Dual-index suffix tree (lexical trie + semantic HNSW/IVF-PQ over hidden states).
- Lookup strategy: lexical lookup first (O(M) trie walk); if no match or match length < K_min, semantic lookup (O(log N) ANN query).
- Hidden-state extraction: from verify forward's `vx` buffer (spec.rs:6401-6407), store per committed token.

**Implementation bill**:
- Hidden-state storage in suffix tree nodes: `struct Node { token: u32, children: HashMap<u32, Node>, hidden: Option<Vec<f32>> }`.
- Semantic index build: quantize hidden states (e.g. PQ with m=8 subquantizers), build HNSW graph over quantized vectors.
- Lookup function: `fn semantic_match(&self, query_hidden: &[f32], max_len: usize) -> Option<Vec<u32>>`.
- VRAM overhead: ~4 bytes/token (quantized hidden) + HNSW graph (~16 bytes/node).

**Measurement**: same board protocol as increment 0/1. Kill criterion: if semantic lookup does NOT lift acceptance by ≥15% over lexical-only (below the Oilbird +24% floor), or if VRAM overhead exceeds 10% of model footprint, abandon semantic re-key and keep lexical-only as the default.

### 3.3 PP-2 Explicitly Deferred Behind Rank-1 Stage-Idle Fill

Suffix proposers on PP-2 inherit the SAME Rank-1 dependency as every other spec method: the stage-idle problem (one stage verifies, the other waits — 95.13% of a PP-2 K=1 round is verify, spec.rs SURVEY.md §0). Suffix proposals do NOT fix the idle-stage bubble; they only change the draft source. Therefore:

**PP-2 suffix serving blocked until**: stage-resident multi-session pipelining (fills the idle stage with another session's verify, SURVEY.md §9 Rank-1, lane/specmech in development) + batched spec prefill + confidence-scheduled verify length land. Only after Rank-1's 1.89x round-phase upper bound is realized can suffix proposals lift PP-2 throughput further (by raising acceptance quality under the fixed stage-fill schedule).

**Single-card is the increment 0/1 target**: no idle-stage bubble, suffix hit rate is the direct lever, e2e tok/s is the honest verdict.

---

## 4. Kill Criteria (Honest Overhead/Acceptance Thresholds)

Suffix proposers are ADOPTED only if they beat the house MTP method e2e on the target workload class. The kill thresholds (measured under board protocol, interleaved N≥2, thermal-stable):

### Increment 0 Kill Criteria (Dogfood A/B)

1. **Suffix hit rate < 10%**: If fewer than 10% of spec rounds find a suffix match ≥ K_min, the pool is not capturing enough repeated structure to justify the index overhead. → Kill, suffix proposers do not fit memra's traffic.
2. **Suffix acceptance < 50% of MTP acceptance**: If suffix-proposed tokens are accepted at < 50% of the rate MTP drafts are accepted, the suffix pool is proposing low-quality continuations. → Kill, lexical matching is too coarse.
3. **E2e tok/s regression > 2%**: If suffix+MTP fallback is slower than MTP-only by > 2% on ANY measured workload, the trie lookup overhead dominates the acceptance gain. → Kill, suffix proposers are net-negative.

### Increment 1 Kill Criteria (Serve-Surface)

1. **No workload with ≥1.1x e2e gain**: If no external workload measures ≥1.1x e2e tok/s over MTP-only baseline, suffix proposers are a wash. → Keep flag-gated as a specialist option, do NOT promote to default.
2. **Regression on freeform generation > 5%**: If suffix pool (even with low hit rate) slows freeform generation by > 5%, the lookup overhead is too expensive for mixed traffic. → Kill, suffix proposers are traffic-class-specific and memra serves mixed.

### Increment 2 Kill Criteria (Oilbird Semantic Re-Key)

1. **Acceptance lift < 15%**: If semantic lookup does NOT lift accepted length by ≥15% over lexical-only (below the Oilbird +24% floor), the hidden-state re-key does not pay its complexity. → Abandon increment 2, keep lexical-only as the default.
2. **VRAM overhead > 10% of model footprint**: If hidden-state storage + semantic index exceed 10% of the model's VRAM footprint (~40-50 GB for Step-3.7, so kill threshold is ~4-5 GB suffix pool), the capacity cost dominates the throughput gain. → Abandon increment 2.

**Doctrine applied**: Winners are defaults (docs/FLAGS.md). If suffix proposers pass increment 1 gates, they become the default proposer for tool-heavy traffic (no flag needed). If they fail any kill criterion, the lane is terminated and suffix proposers are documented as "measured negative, not pursued."

---

## 5. Implementation Seams (File:Line Citations)

### 5.1 Verify Path Entry (Losslessness Preserved)

- **Draft proposal generation**: `crates/memra-engine/src/spec.rs:6200-6300` (inside `generate_spec` orchestrator, before verify forward). Today: MTP chain loop populates `draft[0..k_round]`. Suffix arm: replace with `suffix_proposer.longest_match(...)` when hit, else MTP fallback.
- **Verify forward**: `spec.rs:2107-2266` (`verify_stage0_issue` + `verify_stage1_finish`). No change: suffix-proposed token ids are verified identically to MTP drafts.
- **Accept walk**: `spec.rs:6304-6330`. No change: draft source is opaque to accept logic. Acceptance = draft[j] == argmax(verify_logits[j]).

### 5.2 Worker Session State

- **Per-model global pool**: add to `crates/memra-server/src/worker.rs:76-91` (`LoadedModel` or adjacent worker state): `suffix_pools: HashMap<PoolKey, SuffixTree>`.
- **Per-request pool**: add to `worker.rs:2229-2350` (`Session` struct): `session_suffix: Option<SuffixTree>` (Some when suffix pool enabled).
- **Pool key isolation**: `worker.rs:2354-2356` (`Session::pool_key` returns `(model, cache_ns)`). Suffix trees keyed identically: `suffix_pools[(model, cache_ns)]`.

### 5.3 Cache Namespace Isolation

- **Isolation contract**: `worker.rs:236-237` (`Request::cache_ns`), `worker.rs:2235` (`Session::cache_ns`). Suffix pools obey the same PC-ISO rules as continuation pool (lane/pc-iso, 2026-08-02).
- **vLLM precedent**: `cache_salt` design (SURVEY.md §5e, vLLM ships `method=suffix` under cache-namespace isolation).

### 5.4 Adaptive-Trim Composition

- **Trim escape loop**: `docs/DRAFT-REGIME.md:120-133`. Suffix proposals are real prior text, hence in-distribution for the trim. Suffix misses (verify corrections) are the same escape events the adaptive trim learns from. No conflict: suffix pool updates (on commit) and trim escape writes (on correction) compose naturally.
- **Gemma adaptive trim**: `DRAFT-REGIME.md:127-135` (512 spare head slots, `<ranks>.learned` persistence). Suffix pool + adaptive trim run in parallel: suffix pool captures token-level patterns, adaptive trim captures vocab-level coverage gaps.

---

## 6. Deferred Questions (Not Increment 0 Blockers)

1. **Suffix pool persistence**: Should global suffix trees persist across server restarts? vLLM does NOT persist (in-memory only); memra's continuation pool also does not persist (worker.rs, no disk write). Increment 0: in-memory only, dropped at shutdown. Revisit if hit-rate measurements justify persistence overhead.
2. **Cross-model suffix sharing**: Can a suffix pool span multiple models on the same tokenizer (e.g. Qwen3.5-9B and Qwen3.5-27B)? Tokenizer identity is necessary but not sufficient (different models emit different distributions). Increment 0: per-model pools only. Revisit if capacity pressure justifies cross-model dedup.
3. **Suffix length schedule**: Should K_min (minimum suffix match length) be dynamic per request, or fixed at server start? Increment 0: fixed K_min=2 (suffix match must beat a single-token MTP draft). Increment 1: expose as body field `suffix_min_length` if measurements show workload-dependent optima.

---

## References

- research/spec-landscape-20260810/SURVEY.md §5e (suffix decoding: NeurIPS 2025 spotlight, Snowflake production 1.8-4.5x, vLLM `method=suffix`)
- research/spec-landscape-20260810/SURVEY.md §9 Rank-3 (suffix decoding ranked for PP-2 serving, Rank-1 stage-idle fill dependency)
- crates/memra-engine/src/spec.rs:2107-2266 (verify_stage0_issue / verify_stage1_finish)
- crates/memra-engine/src/spec.rs:6200-6330 (draft/verify/accept orchestrator in generate_spec)
- crates/memra-engine/src/spec.rs:102-149 (SpecConstraint trait, grammar-constrained accept truncation)
- crates/memra-server/src/worker.rs:2229-2350 (Session struct, per-request state)
- crates/memra-server/src/worker.rs:76-91 (LoadedModel, per-model worker state)
- crates/memra-server/src/worker.rs:236-237, 2235, 2354-2356 (cache_ns PC-ISO isolation)
- docs/DRAFT-REGIME.md (adaptive-trim escape loop, law 1: per-model own-gen ranks)
- arXiv 2608.03839 "Oilbird" (hidden-state re-keying: +24-29% accepted length, 4.4x API-Bench)
