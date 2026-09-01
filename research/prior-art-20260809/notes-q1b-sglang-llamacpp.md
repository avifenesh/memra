# Prior art Q1b — SGLang RadixAttention & llama.cpp server cache vs rewritten agent histories

Survey date: 2026-08-09. All sources fetched and read this session (code from raw.githubusercontent,
issues/PRs via GitHub API/pages). Focus: how each engine handles rewritten conversation histories,
entry granularity, matching, eviction, VRAM budgeting, host-tier spill, huge contexts — and what
broke in production (real issue numbers). memra context: prompt-end checkpoints + "identity
nominates, bytes decide" resume.

---

## 1. SGLang (RadixAttention)

### 1.1 Granularity: token-level radix tree, no block hashing on the hot path

**Claim.** The core `RadixCache` keys on raw token-id sequences (`array("q")`), with an optional
`extra_key` namespace. Matching is a byte-compare of token arrays, not hash lookup. Page
alignment (`page_size > 1`) is a rounding constraint layered on top, and SHA256 page hashes
(`hash_page`, `get_hash_str`) exist only for KV-event emission / storage tiers, computed lazily —
they do not drive prefix matching.

> "In our system, we utilize a radix tree to manage a mapping between sequences of tokens, and
> their corresponding KV cache tensors. These KV cache tensors are stored in a non-contiguous,
> paged layout, where the size of each page is equivalent to one token."

Source: SGLang paper, arXiv 2312.07104 (https://arxiv.org/html/2312.07104v2, §3).

Code (read in full): `python/sglang/srt/mem_cache/radix_cache.py`
(https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/mem_cache/radix_cache.py)

> ```python
> def match(self, other: RadixKey, page_size: int = 1) -> int:
>     """Logical-unit prefix length shared with ``other``. Result is rounded down to ``page_size``."""
>     ...
>     # Exponential search for the first diverging token: gallop in doubling
>     # windows (one C-level slice compare each), then binary-search the window
>     # holding the divergence -- no per-token Python loop on long shared prefixes.
> ```

Note the galloping compare: token-exact divergence detection on 100k-token keys costs
O(log n) slice compares, not O(n) per-token Python loops.

**Transfer to memra:** validates "bytes decide" at token granularity as the SOTA choice; a galloped
compare (doubling windows + binary search on the diverging window) is the right shape for finding
the divergence point in a 100k-token resent history cheaply.

### 1.2 Mid-prefix edit handling: match stops at first divergence, node is split there

**Claim.** `match_prefix` walks children by a `child_key` (first page of the remaining key), and if
the match ends *inside* a stored node, the node is split at the exact boundary
(`_split_node`). There is no skip-and-resync: everything after the first diverging token is a
miss and gets re-prefilled. A mid-history edit by an agent client therefore invalidates the whole
tail, exactly like memra's observed problem.

> "If the lookup ends inside a stored segment the node is split once to expose a precise
> boundary; this structural refinement improves subsequent match efficiency and does not
> duplicate data." — `match_prefix` docstring, radix_cache.py

**Transfer to memra:** SGLang gives no answer for mid-prefix edits beyond "keep everything before
the divergence." The value it adds is that the *tree* retains the old branch, so if the client
flip-flops between two history variants, both are cached (until evicted) — an entry-per-snapshot
behavior memra's single-snapshot cache lacked. memra's checkpoint design can get the same effect
with a bounded set of prompt-end checkpoints instead of a full tree.

### 1.3 Insert on completion + in-flight lock refs

**Claim.** KV enters the tree at request completion (`cache_finished_req`: inserts
`origin_input_ids + output_ids`) and incrementally during chunked prefill
(`cache_unfinished_req`). In-flight requests pin their matched path with a per-node
`lock_ref` counter walked up to the root; `evictable_size_`/`protected_size_` are maintained as
the two accounting buckets.

> ```python
> def inc_lock_ref(self, node: TreeNode) -> IncLockRefResult:
>     ...
>     while node != self.root_node:
>         if node.lock_ref == 0:
>             self.evictable_size_ -= len(node.key)
>             self.protected_size_ += len(node.key)
> ```

Source: radix_cache.py (read).

Paper backs the design:

> "each node maintains a reference counter indicating how many running requests are using it. A
> node is evictable if its reference counter is zero. Note that we do not preallocate a
> fixed-size memory pool as a cache. Instead, we let the cached tokens and the currently running
> requests share the same memory pool." — arXiv 2312.07104 §3

**Transfer to memra:** the two-bucket accounting (protected = pinned by running request,
evictable = cache) plus "cache and running requests share one pool, cache yields under load" is
the correct VRAM-budget doctrine — directly relevant after memra's c=64 admission work: cached
prefixes should be a *soft* pool the admission gate can reclaim, counted in `free + pool_cached`
just like the retired-KV pool fix.

### 1.4 Eviction: heap over evictable leaves, leaf-first LRU, parent becomes leaf next

**Claim.** `evict(num_tokens)` heapifies the current `evictable_leaves` set by the pluggable
eviction strategy's priority (default LRU on `last_access_time`; a priority-aware policy also
exists — `TreeNode.priority`, propagated max-wise up the path on insert), pops leaves until the
token quota is freed, and re-pushes a parent when its last child disappears.

> "we introduce a simple LRU eviction policy that evicts the least recently used leaf first. By
> evicting leaves first, we enable the re-use of their common ancestors until those ancestors
> become leaves and are also evicted." — arXiv 2312.07104 §3

**Transfer to memra:** leaf-first eviction preserves shared prefixes longest — for memra's
checkpoint list per session, the analog is: evict deepest/newest checkpoints of cold sessions
first, keep the shortest common-prefix checkpoint (e.g. the system-prompt-end one) until last.

### 1.5 Cache-aware scheduling: longest-prefix-match ordering, with escape hatches

**Claim.** The scheduler sorts the waiting queue by matched prefix length (`LPM` policy, default),
approximating a DFS over the request radix tree; it falls back to FCFS when the queue exceeds 128
(matching cost), and de-prioritizes requests whose prefix hit is only against *other waiting
requests* (in-batch prefix caching, so one of the twins runs first and the rest hit its KV).

> ```python
> class CacheAwarePolicy(Enum):
>     LPM = "lpm"  # longest prefix match
>     DFS_WEIGHT = "dfs-weight"  # depth-first search weighting
> ...
> def _determine_active_policy(self, waiting_queue: List[Req]) -> Policy:
>     if self.policy == CacheAwarePolicy.LPM and len(waiting_queue) > 128:
>         # Turn off the expensive prefix matching and sorting when the #queue is large.
>         return CacheAgnosticPolicy.FCFS
> ```

Source: `python/sglang/srt/managers/schedule_policy.py` (read;
https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/managers/schedule_policy.py).
Paper §3: "longest-shared-prefix-first order... can lead to starvation. We leave its integration
with other fair scheduling methods [as future work]."

**Transfer to memra:** at memra's concurrency (c<=64, single model) LPM ordering of the admission
queue is cheap and buys hit-rate; the starvation caveat is acknowledged-and-unsolved upstream, so
memra shouldn't over-engineer fairness either. The in-batch dedup trick matters for n>1 sampling
and parallel subagent fan-out (same history, k lanes) — run one, let k-1 hit.

### 1.6 extra_key: identity as namespace, not as trust

**Claim.** `RadixKey.extra_key` (LoRA id, cache salt) partitions the tree: same tokens under
different extra_key never share nodes. Identity *narrows* the match domain; bytes still decide
within it.

> "Entries that share identical leading token ids but have *different* ``extra_key`` values are
> intentionally kept disjoint and never share prefix nodes." — `match_prefix` docstring

**Transfer to memra:** exact precedent for "identity nominates, bytes decide" — and the LoRA
salting is the lesson llama.cpp learned the hard way (issue #26207 below): anything that changes
the KV *function* (adapter, cache version) must be part of the key, not just the token bytes.

### 1.7 HiCache: host tier + storage tier over the same tree

**Claim.** HiCache extends the radix tree to a HiRadixTree where each node records which tier(s)
(GPU L1 / host L2 / storage L3) hold its KV; host tier uses a page-first layout decoupled from the
GPU layer-first layout for I/O efficiency; L2→GPU loads overlap layer-wise with compute; L3 has
prefetch policies (best_effort / wait_complete / timeout with token-linear timeout) and write
policies (write_through / selective-by-hit-count / write_back-on-evict). Nodes carry
`host_value`, a separate `host_ref_counter` (`protect_host`/`release_host`), and
`write_through_pending_id` — visible in radix_cache.py's TreeNode.

> "HiRadixTree extends this idea: each node corresponds to the KV cache of a span of consecutive
> tokens and records where that KV cache is stored—whether in local GPU memory, CPU memory, L3
> storage, or multiple of these tiers." — HiCache design doc,
> https://docs.sglang.io/advanced_features/hicache_design.html

> "When a cache miss happens on the GPU but hits the CPU memory... we apply a layer-wise
> overlapping mechanism to load the data. This enables concurrent KV cache loading for layer N+1
> while layer N is executing." — LMSYS HiCache blog,
> https://www.lmsys.org/blog/2025-09-10-sglang-hicache/

Reported gains (vendor-quoted in blog): TTFT −56% and hit rate 40%→80% on a Qwen3-Coder-480B
coding-agent workload (Novita); up to 6x throughput / −80% TTFT in LMSYS's own multi-turn bench.

**Transfer to memra:** the host tier is what makes multi-session agent serving survive VRAM
pressure — memra's checkpoint entries are naturally host-spillable (they're already state blobs);
layer-wise H2D overlap on restore is the same shape as memra's existing prefetch/overlap spill
doctrine and belongs on the CUDA owner thread. Page-first host layout (one contiguous blob per
page across layers) is the right on-host format for a future storage tier.

### 1.8 What SGLang got wrong (verified issues)

**(a) #22819 — KV corruption at radix block boundary, `prefix_len == block_size`.**
https://github.com/sgl-project/sglang/issues/22819 (2026-04, open at fetch time)

> "Identical prompts at `temperature=0` produce completely different output sequences across
> runs. The key trigger: a request whose `prefix_len` is exactly equal to the KV block size (64
> tokens) consistently gets corrupted... 10/10 runs... `--enable-deterministic-inference` does
> NOT fix it — this is not an overlap scheduler race... consistent with a stale or misassigned
> KV block being deterministically reused from a prior request's prefix."

Timing-sensitive to the first burst window when the block is first allocated. Lesson: the
page-boundary edge (match length exactly one page) is the hottest correctness corner of any
paged prefix cache.
**Transfer to memra:** gate the resume path with an exact-boundary test battery: match length ==
page/checkpoint boundary, ±1 token, under concurrent first-touch allocation.

**(b) #22373 — thinking tokens pollute the tree with unreachable branches.**
https://github.com/sgl-project/sglang/issues/22373 (2026-04)

> "The core issue is that what gets cached and what gets sent in future requests are decided
> independently... SGLang's own chat completion API drops `reasoning_content` from assistant
> messages during prompt construction... yet the caching layer stores all thinking tokens."
> "Turn 2 arrives with [Q1, A1, Q2] (thinking stripped)... match stops at Q1... A1 is stranded:
> its KV is in the tree but unreachable behind T1... Each turn leaves a dead [Ti, Ai] branch."
> "~5000 thinking tokens per turn... 50 concurrent conversations with 1 recent dead branch each:
> 65-80 GB of dead KV cache."

This is the *same failure class as memra's rewritten-history problem*: the client's next prompt is
not `previous prompt + output`; anything cached beyond the divergence is dead weight AND blocks
reuse of what comes after it.
**Transfer to memra:** cache what the client will *resend*, not what the engine computed — i.e.
insert/checkpoint keyed on the prompt-visible byte stream (strip non-resent reasoning), which is
precisely the prompt-end-checkpoint framing. Also: measure dead-entry mass as a first-class metric.

**(c) #19796 / #32459 — speculative decoding × radix reuse: crash, then silent reuse collapse.**
https://github.com/sgl-project/sglang/issues/19796 (EAGLE V2 NaN in verify when a batch's KV was
partially radix-populated, SM120/PRO 6000): "the bug is in Eagle V2 verify path when processing a
batch where KV cache was partially populated from radix cache prefix."
https://github.com/sgl-project/sglang/issues/32459 (2026-07):

> "Enabling EAGLE speculative decoding collapses radix prefix reuse for multi-turn agentic
> traffic — at any draft length... TP8, no spec: 97% cache hit (deep >=20K-tok prompts); TP8 +
> EAGLE steps=5/draft=6: 53%; steps=3/draft=4: 40%... looks like either a defensive bypass of
> radix-populated KV on the spec path or silent recompute."

**Transfer to memra:** MTP spec state must be part of the checkpoint contract from day one
(llama.cpp got this right — see checkpoint `data_spec` below). A resume path that works for plain
decode but silently bypasses on the spec path recreates memra's TTFT cliff while looking "green."

**(d) #5525 — `cached_tokens` misreported under parallel sampling** (closed;
https://github.com/sgl-project/sglang/issues/5525) and #20451 (cached_tokens = 0 under spec) —
observability of reuse is itself bug-prone; both issues were only findable because the counter
existed.
**Transfer to memra:** ship a per-request `cached_tokens`/reused-length field in the serve
surface at the same time as the resume feature; it's the instrument every one of these upstream
bugs was diagnosed with.

**(e) #26577 — SWA prefix reuse is capped by window validity.** SGLang only reuses a prefix when
the sliding-window KV for the suffix boundary is still valid; `match_prefix_for_req` in
schedule_policy.py caps match length ("a reused prefix carries stale SWA. Cap" — code comment,
read). Same physics as llama.cpp's SWA checkpoint story.

---

## 2. llama.cpp server

Server was split from monolithic `server.cpp` into `tools/server/server-context.cpp` (slot logic,
5434 lines), `server-task.cpp` (incl. `server_prompt_cache`), `server-common.cpp` (tokens).
All read from master, 2026-08-09.

### 2.1 Slot system + token-level common-prefix trim

**Claim.** Each of `n_parallel` slots owns one KV sequence (`slot.id` = seq id) and remembers its
full token history (`slot.prompt.tokens`). On a new request with `cache_prompt: true` (default
**true** per README), the server computes the longest common prefix between the slot's cached
tokens and the incoming tokens — plain token-by-token loop — keeps it, and `seq_rm`'s the rest.

> ```cpp
> if (slot.task->params.cache_prompt) {
>     // reuse any previously computed tokens that are common with the new prompt
>     n_past = slot.prompt.tokens.get_common_prefix(input_tokens);
> ```
> — tools/server/server-context.cpp (~line 3219)

> ```cpp
> size_t server_tokens::get_common_prefix(const server_tokens & b) const {
>     const size_t max_idx = std::min(tokens.size(), b.tokens.size());
>     if (!has_mtmd) {
>         for (size_t i = 0; i < max_idx; ++i) {
>             if (tokens[i] == b.tokens[i]) continue;
>             return i;
>         }
>         return max_idx;
> ```
> — tools/server/server-common.cpp:471 (mtmd chunks compared by chunk id + token count)

Then the trim + guarantee-one-token rules:

> ```cpp
> // [TAG_PROMPT_LOGITS]
> if (n_past == slot.task->n_tokens() && n_past > 0) { n_past--; }  // must eval >= 1 token
> slot.prompt.tokens.keep_first(n_past);
> ...
> slot.mem.seq_rm(slot.id, p0, -1);   // truncate KV beyond the kept prefix
> ```

README (`tools/server/README.md`):

> "`cache_prompt`: Re-use KV cache from a previous request if possible. This way the common
> prefix does not have to be re-processed... Because (depending on the backend) the logits are
> not guaranteed to be bit-for-bit identical for different batch sizes... enabling this option
> can cause nondeterministic results. Default: `true`"

**Transfer to memra:** the one-slot-one-history model is memra's current model; the piece memra
lacked is exactly `get_common_prefix` + trim-don't-freeze. Note also the honest documentation
that prefix reuse changes batching and therefore logits — worth mirroring in memra's docs given
the argmax-gate discipline.

### 2.2 Mid-prefix edit: `--cache-reuse` KV-shifting (chunk resync past the divergence)

**Claim.** Uniquely among the two engines, llama.cpp can reuse cache *after* a mid-prefix edit:
with `n_cache_reuse > 0` it scans past the divergence for matching chunks >= N tokens and
*shifts* their KV to the new positions (RoPE re-rotation via `seq_add`), instead of recomputing.

> ```cpp
> // reuse chunks from the cached prompt by shifting their KV cache in the new position
> if (can_cache_reuse && n_cache_reuse > 0) {
>     ...
>     while (head_c < slot.prompt.tokens.size() && head_p < input_tokens.size()) {
>         size_t n_match = 0;
>         while (... slot.prompt.tokens[head_c + n_match] == input_tokens[head_p + n_match]) n_match++;
>         if (n_match >= (size_t) n_cache_reuse) {
>             const int64_t kv_shift = (int64_t) head_p - (int64_t) head_c;
>             slot.mem.seq_rm (slot.id, head_p, head_c);
>             slot.mem.seq_add(slot.id, head_c, head_c + n_match, kv_shift);
> ```
> — server-context.cpp (~lines 3240-3285). Gated on `llama_memory_can_shift()` and no mtmd.
> Default 0 (off) — README: "`--cache-reuse N`: min chunk size to attempt reusing from the cache
> via KV shifting, requires prompt caching to be enabled (default: 0)".

This is an *approximation*: shifted KV was computed with different absolute positions attending to
now-deleted tokens; only position embedding is corrected. It's off by default for a reason.
**Transfer to memra:** tempting for the "history edited mid-stream" case (deleted tool result,
compacted message), but it trades exactness for TTFT — incompatible with memra's argmax-gate
discipline unless gated as an explicitly-blocked experimental door with its own quality gate.
Adopt the *concept* (divergence-point resync is possible) but not as default.

### 2.3 Slot selection: LCP-similarity routing (`slot.similarity` ancestor)

**Claim.** Multi-slot routing picks the idle slot whose cached tokens share the largest LCP
fraction with the incoming prompt (`--slot-prompt-similarity`, default 0.10), falling back to LRU.
And critically: before recycling a slot whose contents would be mostly destroyed (`f_keep < 0.5`),
it *saves the outgoing state to the host-memory prompt cache*.

> ```cpp
> // fraction of the Longest Common Prefix length with respect to the input prompt length
> const size_t lcp_len = tokens.get_common_prefix(task.tokens);
> const float f_sim_cur = float(lcp_len) / task.tokens.size();
> ...
> if (f_sim_cur > f_sim_best && f_sim_cur > slot_prompt_similarity) { ... ret = &slot; }
> ...
> // if we are about to lose a large portion of the existing context - save it in the prompt cache
> if (f_keep < 0.5f) { update_cache = true; }
> ```
> — server-context.cpp `get_available_slot()` (~lines 1589-1700)

**Transfer to memra:** this IS "identity nominates, bytes decide" without the identity — llama.cpp
has no session id at all, similarity does everything. memra's session-id nomination is strictly
stronger (O(1) candidate lookup vs O(slots × prompt_len) LCP scans), with bytes as the same final
arbiter. Keep llama.cpp's save-before-recycle rule: never destroy a >50%-losable context without
first spilling it to the host cache.

### 2.4 Host-memory prompt cache (`--cache-ram`, PR #16391)

**Claim.** Merged 2025-10-09. Evicted/recycled slot states (full `llama_state_seq` blobs +
checkpoints + token list) go to a host-RAM deque acting as "extra slots"; on new tasks the server
compares the incoming tokens against cached entries by LCP and hot-swaps the best one in
(`llama_state_seq_set_data_ext`). Selection requires *both* better `f_sim` (fraction of new prompt
covered) and better `f_keep` (fraction of the cached entry that survives), with a floor
`f_keep >= 0.25` ("don't trash large prompts"). Eviction is FIFO-oldest under two budgets: MiB
(`--cache-ram`, default 8192) and token count (default = n_ctx, dynamically raised if bytes/token
allow). Entries fully contained in a newly saved prompt are dropped as obsolete; a failed
`bad_alloc` shrinks the limit to 0.4× current size.

> "The host-memory prompt cache acts as 'extra slots' with which we can calculate prefix
> similarity and decide to hot-swap them into the `llama_context` if it would reduce the
> processing." — PR #16391 body, https://github.com/ggml-org/llama.cpp/pull/16391

> ```cpp
> // don't trash large prompts
> if (f_keep_cur < 0.25f) continue;
> if (f_keep_best < f_keep_cur && f_sim_best < f_sim_cur) { ... it_best = it; }
> ```
> — server-task.cpp `server_prompt_cache::load()` (~line 1746)

Also `--cache-idle-slots` (default enabled): idle slots are proactively saved to the prompt cache
and cleared when using unified KV, freeing VRAM cells for the active request.
**Transfer to memra:** a deque of full-state host entries with dual (bytes, tokens) budgets and
the two-ratio acceptance test is a simple, shippable host tier — much less machinery than
HiCache, fits memra's single-node scope. The `f_keep` floor encodes a real lesson: resuming a tiny
prefix from a huge entry costs a huge restore for little reuse (restore bandwidth is the hidden
denominator). memra's "bytes decide reusable length" should also weigh restore cost, not just
matched length.

### 2.5 Context checkpoints — llama.cpp converged on memra's design (user-turn checkpoints)

**Claim.** For models where trimming isn't possible or the window slides (SWA, recurrent/hybrid,
partial-`seq_rm`-bounded), llama.cpp keeps per-slot *checkpoints* — small state blobs
(`--ctx-checkpoints N`, default 32; `--checkpoint-min-step`, default 8192 tokens) — and on a new
request finds the last checkpoint at-or-before the divergence point and restores it. Evolution,
each step driven by an agent-workload failure:

- PR #15293 (2025-08): SWA checkpoints introduced. "The server now makes checkpoints of the SWA
  memory in order to minimize the amount of context reprocessing... the size is relatively small
  (proportional to the SWA window)."
- PR #20288 (2026-03): "make 2 checkpoints near the end of the prompt" — offsets `{4 + n_ubatch, 4}`
  tokens before prompt end — "In some cases, reprocessing the last 512 tokens of the prompt could
  be too slow. In other cases it is necessary in order to allow **mutating the last user message**."
- PR #22929 (merged 2026-05-25): parse `message_spans` from the chat template, split the prefill
  batch at the last user message, checkpoint there. Author: "This is another chapter in my journey
  toward fixing `forcing full prompt re-processing due to lack of cache data`. My main goal is to
  increase the 'responsiveness' of agentic coding in llama.cpp."
- PR #23814 (closed in favor of) → PR #24176 (merged 2026-06-23): "Create checkpoints at the start
  of **every** user message, as opposed to only the last message" — because "#22929 creates a
  context checkpoint only before the last user message, so prompts with a stable prefix and
  content that changes between turns lose all checkpoint cache hits (the surviving checkpoints sit
  past the divergence point)."

Restore-side selection (server-context.cpp ~3352):

> ```cpp
> // search for a context checkpoint
> const auto it = std::find_if(slot.prompt.checkpoints.rbegin(), ..., [&](const auto & cur) {
>     if (cur.pos_max > pos_next) return false;         // must not extend past divergence
>     return cur.pos_min < pos_min_thold || cur.pos_min == 0;
> });
> if (!do_reset) {
>     it->load_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
>     it->load_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
>     common_speculative_set_state(spec.get(), slot.id, it->data_spec);   // draft/spec state too
> ...
> // erase any checkpoints with pos_max > pos_next   (invalidated by the edit)
> ```

Checkpoints are created *before* `llama_decode` of the batch that starts a user message
(`spans.is_user_start(n_tokens_start)`), skipped after mtmd chunks, bounded by min-step except at
the last user message, and stored inside the host prompt-cache entry when the slot is saved.
**Transfer to memra:** this is the strongest possible prior-art confirmation of memra's design in
flight — llama.cpp independently converged on prompt-end/user-turn checkpoints for exactly the
Claude-Code-style rewritten-history workload, and needed three iterations to learn: (1) checkpoint
near prompt end (the mutating last message), (2) checkpoint at *message-boundary semantic
positions* not fixed token strides, (3) at EVERY user boundary because agents edit *mid*-history,
not just the tail. Also: checkpoint the draft/spec state (`data_spec`) — the thing SGLang's EAGLE
path is still broken over. Note their derivation of boundaries from chat-template delimiters
(`find_message_spans`) — memra can do the same from its per-arch templates, or accept
client-declared boundaries.

### 2.6 Context shift: off by default, incompatible with the exact path

**Claim.** `--context-shift` (discard oldest tokens + `seq_add` shift when generation exceeds
n_ctx) is **default disabled** (README: "whether to use context shift on infinite text generation
(default: disabled)") and structurally impossible for SWA/recurrent memory ("advanced cache
operations such as removing tokens or shifting their positions are not possible when using SWA
cache, because token information becomes lost when the window slides" — PR #13194 body). Issue
#16983 (2025-11): after the `server: remove n_past` refactor (#16818), a full slot returned
"context shift is disabled" errors to *new* conversations — regression from entangling slot
recycling with shift; fixed in #17000.
**Transfer to memra:** position-shifting mutation of live KV is a permanent source of exactness
bugs and model-class incompatibility; memra's choice (reject/re-prefill rather than shift) matches
where llama.cpp landed after trying the other way.

### 2.7 What llama.cpp got wrong (verified issues)

**(a) #26207 — prompt cache reused across different per-request LoRA; silent contamination.**
https://github.com/ggml-org/llama.cpp/issues/26207 (2026-07, open at fetch)

> "KV computed while adapter A was active is reused for a request selecting adapter B whenever
> the prompt prefix matches. The request succeeds, nothing is logged, and the output is
> partially influenced by the wrong adapter... Suggested fix: treat a change in the resolved
> per-request lora config like a prompt mismatch: invalidate the slot's cached prefix (or key
> the cache on the lora config)."

Bytes-only matching is *insufficient* when anything besides the token stream conditions the KV.
SGLang's `extra_key` is the fix llama.cpp lacks here.
**Transfer to memra:** the candidate key must include every KV-conditioning parameter — model,
adapter, KV dtype, template/reasoning-format flags — before bytes get a vote. "Identity nominates"
should mean *full identity*, not just session id.

**(b) #21831 — hybrid/recurrent + SWA models: full re-processing every turn.**
https://github.com/ggml-org/llama.cpp/issues/21831 (2026-04)

> "On the second request to the same slot, it forces a full prompt re-processing, logging:
> `forcing full prompt re-processing due to lack of cache data (likely due to SWA or
> hybrid/recurrent memory...)`. As a result, the model 'forgets' the conversation history."

Reproduced with 45-token contexts on Qwen3.5-MoE/Gemma4-MoE. The checkpoint machinery (2.5) is
the mitigation; before checkpoints landed at user boundaries this was the default agent
experience on SWA models (also #22746 for Qwen 3.6 27B — memra's own release-target arch).
**Transfer to memra:** any non-full-attention arch memra adds (SWA layers, hybrid) breaks
"trim to divergence and continue" — the resume design should carry, per-arch, the max rollback
depth (llama.cpp: `pos_min_thold = pos_next - n_swa`), and fall back to nearest checkpoint.

**(c) #12253 — garbage output after KV-cache defragmentation (CPU backend, closed/fixed).**
https://github.com/ggml-org/llama.cpp/issues/12253 (2025-03)

> "using the API or the WebUI to make the model generate large outputs on two slots at once, I
> get garbage output (which stays until server restart) as soon as KV-cache defragmentation
> occurs once."

Defrag moved cells while slot bookkeeping still pointed at old locations — the llama.cpp analog of
the misassigned-block class in sglang #22819.
**Transfer to memra:** if memra ever compacts KV pages, the compaction must be atomic w.r.t. every
index that names cells (session table, checkpoint table, spec state) — same "one owner thread
publishes" doctrine as memra's H2D/cache publication rule.

**(d) Claude-Code-specific field report — cache killed by the *client*, not the server.**
Blog: https://www.mykolaaleksandrov.dev/posts/2026/06/claude-code-llamacpp-prompt-cache-fix/

> "The problem was Claude Code adding an attribution block at the beginning of the system
> prompt... That block can include details like the client version and prompt fingerprint...
> Prompt cache reuse depends heavily on stable token prefixes. If the beginning of the prompt
> changes, even slightly, the server may fail to reuse the existing context."
> Fix: `CLAUDE_CODE_ATTRIBUTION_HEADER=0`. Logs flipped from "forcing full prompt re-processing"
> to "restored context checkpoint... prompt eval time = 511.40 ms / 212 tokens".

Corroborated by PR #22929's author: "`preserve_thinking` really helps, without it, the prompt
history changes, so there is always some reprocessing."
**Transfer to memra:** two levers live outside the engine: (1) document the client-side stability
knobs (attribution header, preserve_thinking) in memra's serve-compat notes for the pill/dogfood
setup; (2) a head-of-prompt mutation defeats *any* prefix scheme — only mid-history checkpointing
(2.5) or chunk-resync (2.2) survive it. TTFT telemetry should distinguish "diverged at token <100"
(client problem) from "diverged mid-history" (engine's job).

**(e) Disk persistence exists but is manual-only.** `--slot-save-path` + `POST
/slots/{id}?action=save|restore|erase` (README lines ~1085-1125) — byte dump of a slot's KV to a
named file; no automatic keying/eviction. The automatic path stops at host RAM. UNVERIFIED beyond
README: real-world usage of these endpoints appears rare.

---

## Both engines: direct answers to the cross-cutting questions

- **Granularity.** SGLang: token (page_size=1 default; page-rounding when >1; HiCache L3 moves
  pages). llama.cpp: token, per-slot linear history (no tree, no pages at server level).
  Neither hashes blocks for the hot match path (vLLM-style block hashing is the odd one out);
  both compare token ids directly.
- **Mid-prefix edit.** SGLang: keep-before-divergence only; the old branch stays in the tree
  (usable if the client flip-flops back, dead weight otherwise — #22373). llama.cpp: trim to
  divergence + optional KV-shift resync (`--cache-reuse`, approximate, off by default) + restore
  nearest checkpoint at-or-before divergence and erase checkpoints past it (exact).
- **Multi-user fairness/eviction.** SGLang: shared pool, lock_ref-protected in-flight paths,
  leaf-first LRU (or priority strategy), LPM scheduling with acknowledged starvation risk and a
  >128-queue FCFS fallback. llama.cpp: fairness = slot partitioning (or unified-KV contention);
  eviction = slot recycling by LCP-then-LRU with save-before-destroy into a FIFO host deque with
  byte+token caps.

## Strongest transferable mechanisms (ranked for memra)

1. **User-turn/prompt-end checkpoints at every boundary** (llama.cpp PRs #15293 → #20288 → #22929
   → #24176): independent convergence on memra's in-flight design, including the two failure modes
   that forced iteration (tail-only checkpoints die on mid-history edits; fixed-stride checkpoints
   miss the semantic edit points). Checkpoint MUST include spec/draft state (`data_spec`) —
   sglang #32459/#19796 show what its absence costs (97%→40% reuse or NaN crash).
2. **Full-identity cache keys** (sglang `extra_key`; llama.cpp #26207 as the counterexample):
   session id + model + adapter + KV dtype + template flags nominate; bytes decide within the
   namespace. Cheap to do now, silent-corruption class if skipped.
3. **Save-before-recycle + two-ratio acceptance (`f_sim`/`f_keep` with 0.25 floor) host deque**
   (llama.cpp #16391): the minimal host tier that fits a single-node engine; weigh restore cost,
   not just matched length. Dual budgets (MiB + tokens), FIFO eviction, drop-contained-entries.
4. **Two-bucket VRAM accounting** (sglang lock_ref: protected vs evictable, cache shares the
   request pool and yields under load): plugs directly into memra's admission-gate
   `free + pool_cached` model.
5. **Galloped divergence search** (sglang `RadixKey.match`): O(log n) compares to find the edit
   point in a resent 100k-token history.
6. **Boundary-exact test battery** (sglang #22819, llama.cpp #12253): resume-length == page/
   checkpoint boundary ±1, under concurrent first-allocation and after any compaction; plus
   determinism check (same prompt, temperature=0, N runs) as the corruption detector.
7. **Reuse observability day-one** (`cached_tokens` per request + divergence-position histogram):
   every upstream bug above was found through these counters; also separates client-caused
   head-of-prompt churn from engine-side misses.
8. **Cache what will be resent, not what was computed** (sglang #22373): strip non-resent
   reasoning bytes from the checkpoint key; track dead-entry mass.

## Open uncertainties

- **KV-shift resync quality** (llama.cpp `--cache-reuse`): no upstream quality measurements found
  of shifted-KV output vs recompute on modern RoPE models; it survives as an off-by-default flag.
  If memra ever wants mid-edit resync, it needs its own argmax-gate evidence. UNVERIFIED beyond
  code + README.
- **sglang #22819 root cause**: still open at fetch time; whether it's allocator first-touch,
  page-boundary off-by-one in match, or block misassignment in the pool is not concluded upstream.
  The *repro shape* (prefix_len == block_size, first burst) is the transferable artifact.
- **LPM starvation**: paper explicitly leaves fairness integration as future work; no upstream
  incident found quantifying it. Low risk at memra's concurrency.
- **HiCache multi-rank sync costs** (`all_reduce(min)` per prefetch decision): single-node PP-2
  memra likely doesn't need it, but if the two PRO 6000s ever run TP-style shared cache, the
  consistency protocol is non-trivial — not evaluated here.
- **llama.cpp `slot.similarity` naming**: the current code uses `slot_prompt_similarity` (server
  param) + local `f_sim_*`; a `slot.similarity` member existed in older monolithic server.cpp.
  Behavior verified in current code; the exact historical field name UNVERIFIED.
- **cache_prompt default flip date**: default is `true` in current README; the PR that flipped it
  from false was not chased down. UNVERIFIED (low value).
