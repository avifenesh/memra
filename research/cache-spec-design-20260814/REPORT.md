# Cache + speculative decoding coexistence — cross-engine survey and port design

Date: 2026-08-14. Method: source reads of SGLang (commit c20aceeb) and vLLM
(commit e078a223) plus web-verified TRT-LLM / LMDeploy / MLC mechanics and the
load-bearing papers. Motivation: memra's measured 4x loss — spec-on at c=16 on
the shared-prefix sold shape = 2.14 req/s @ 18.4% cache hit vs spec-off
8.50 req/s @ 99.5% hit (`research/canonflip-20260813/`). Owner directive
2026-08-14: study how the main engines run cache concurrent with (masked) MTP
and port it.

## The universal invariant (every production engine)

1. **One shared KV pool.** Draft/verify tokens write into per-request tail
   slots past the committed length — never a separate scratch pool, never over
   published cache entries.
2. **Commit-gated publication.** KV becomes visible to the shared prefix cache
   ONLY for verified-accepted tokens. Speculative KV is physically in the pool
   but logically invisible until commit.
3. **Rollback = counter, not data.** Rejection never touches the cache: the
   committed-length counter simply doesn't advance; rejected slots are
   overwritten in place next round (chain drafts make them a contiguous tail).
4. **The real design fork: the draft-head KV boundary.** EAGLE/MTP draft KV at
   position i depends on token i+1 (hidden(i) + embed(i+1)), so cached draft KV
   validity depends on the NEXT token, and the drafter needs a target hidden
   state at the reuse boundary.

## Per-engine mechanics (condensed; full refs at bottom)

- **SGLang**: RadixAttention with **bigram keys** when spec is on
  (`RadixKey.maybe_to_bigram_view`, PR #13714) — key units become
  (tok_i, tok_i+1), so cached draft KV is valid by construction and a hit
  always leaves ≥1 tail token to recompute (regenerating the drafter's seed
  hidden state). MTP-layer KV lives as extra layers on the SAME slot indices
  (`mtp_draft_device_pools`). Per-request `kv_committed_len` vs allocated;
  radix insertion only ≤ committed at request boundaries; over-allocation
  freed back. Adaptive spec: batch-size buckets — bs1 {1,3,7} … **bs≥64 {0}**,
  and at steps=0 the draft-extend STILL runs each iteration to keep draft KV
  warm for resumption. FR-Spec token map = draft LM-head row slice + index
  remap before verify — zero KV interaction.
- **vLLM v1**: SD × APC = fully compatible (v0 limitation gone). Publication
  gate caps caching at `request.num_tokens` (kv_cache_manager.py:559);
  rollback `num_computed_tokens -= num_rejected` (scheduler.py:1846).
  Boundary: EAGLE/MTP KV groups **match one extra block then drop it** on hit
  (kv_cache_coordinator.py) + never end a prefill chunk within the MTP
  lookahead. Measured failure mode: the one-block drop × coarse alignment —
  issue #38182 (Qwen3.5-35B MTP-1: hit rate 92%→71%), vllm-ascend #9247
  (16K-LCM alignment truncates a 32K hit to 16K → MTP = net TTFT loss). The
  official Qwen3.5 recipe splits: throughput = APC without MTP; latency =
  MTP with APC off. Dynamic SD buckets [[1,64,3],[65,128,1],[129,512,0]].
- **TRT-LLM**: block radix reuse works with all spec models now; explicit
  `rewindKVCache(requestId, lens)` outside CUDA-graph capture; separate
  draft-KV manager for two-model EAGLE; `max_concurrency` +
  `draft_len_schedule={100:4,200:3,300:2}` gates drafting by load.
- **LMDeploy** (PR #4688, cleanest boundary fix): "one-block overlap
  recompute" — matched-but-recomputed overlap blocks stay PRIVATE/WRITABLE
  during trie allocation, so spec never writes into shared cached blocks
  (`recompute_blocks = 1`).
- **MLC**: radix as sequence forks; EAGLE prefill forks at offset−1
  (`RollBackSequence(id,1)`); `PopN` chain rollback; adaptive default —
  draft len 4 if running<10 … 0 if ≥30.
- **Qwen MTP shape**: both big engines run the single MTP layer as an
  EAGLE-style chain drafter re-applied autoregressively (SGLang NEXTN,
  steps 3 / draft budget 4, accept ≈3.2-3.3; vLLM num_speculative_tokens 1-2
  with module reuse). Hybrid GDN models additionally need mamba-state commit
  after verify (SGLang `commit_mamba_states_after_verify`).
- **Papers**: FR-Spec (2502.14856) — draft-head-only trim, no KV implications,
  composes with everything. EAGLE-3 (2503.01840) stays throughput-positive to
  bs64. Batch-spec-done-right (2510.22876) — per-request committed counters ARE
  the fix for ragged acceptance. SuffixDecoding (2411.04975) — model-free
  drafting, zero draft KV, cache-compatible by construction, strongest on
  agentic/shared-prefix traffic (memra's sold shape).

## Port designs for memra, ranked

**A. Commit-gated publication + counter rollback + one-page boundary recompute
(vLLM/LMDeploy class). LOWEST COST — do first.** Per-request committed vs
allocated lengths; spec writes only into own tail slots; publish ≤ committed;
on hit, recompute the last small page into request-private slots (never dedup
back to shared; regenerates the MTP seed hidden state). Guard: keep recompute
granularity ONE small page and never round hits to a coarser alignment (the
#38182/#9247 amplification).

**B. Bigram-keyed cache (SGLang class). MEDIUM COST, best steady-state hits.**
Everything in A + bigram key units when spec on (insert caches len−1; exactly
one tail token recomputed per hit) + MTP-layer KV as extra layers on the same
slot indices + draft-extend after every prefill/verify to keep draft KV warm.

**C. Eliminate draft KV.** C1 frozen-KV MTP head (drafter reads target KV
read-only — retraining decision, not a serving patch; stock Qwen MTP heads
have their own KV). C2 model-free drafting (suffix/n-gram) as the
high-concurrency arm — zero draft KV, cache-compatible, strong on exactly the
shared-prefix agent shape.

**Scheduling policy (any design):** batch-size-bucketed draft steps with 0 at
high load, AND draft-extend keeps running at steps=0 so resumption doesn't pay
a re-prime. At the sold shape's measured 99.5% hit / 8.5 req/s, throughput-
optimal may genuinely be steps ∈ {0,1} — fix the cache interaction, then let
the adaptive table decide instead of assuming spec-on.

**Masked head note:** FR-Spec trimming (d2t) is orthogonal to KV and cache in
every implementation surveyed — keep it enabled in all designs; it is not a
suspect for the cache regression.

## Key references

SGLang: `srt/mem_cache/{radix_cache,common,allocation}.py`,
`srt/speculative/{eagle_worker_v2,eagle_worker_common,adaptive_spec_params}.py`,
PR #13714, issues #19796, #8726. vLLM: `v1/core/{kv_cache_manager,
kv_cache_coordinator,single_type_kv_cache_manager}.py`, `v1/core/sched/
scheduler.py`, issue #38182, vllm-ascend #9247, docs features/README +
speculative_decoding/*. TRT-LLM: `kvCacheManager.h` (rewindKVCache), MTP tech
blog, `_torch/speculative/drafter.py`. LMDeploy: PR #4688,
`strategies/ar_spec/sequence.py`. MLC: `engine_actions/{eagle_new_request_
prefill,batch_verify,auto_spec_decode}.cc`, TVM `kv_state.h`. Papers:
arXiv 2502.14856, 2503.01840, 2510.22876, 2411.04975.
