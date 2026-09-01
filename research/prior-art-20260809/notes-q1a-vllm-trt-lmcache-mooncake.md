# Prior art Q1a: rewritten-history prefix caching in vLLM / TensorRT-LLM / LMCache / Mooncake

Survey date: 2026-08-09. All quotes fetched live from the cited URLs on this date (no memory-only
claims). Context: memra's live problem — agent clients rewrite conversation history each turn,
memra's prefix cache froze at the first snapshot, TTFT grew 5.85x over 10 turns. Design in flight:
prompt-end checkpoints + "identity nominates, bytes decide" resume.

---

## 1. vLLM — Automatic Prefix Caching (APC, v1 engine)

### 1.1 Cache entry granularity + hash scheme: chained per-block hash, full blocks only

Claim: vLLM hashes at KV-block granularity (default block size 16 tokens); each block's hash is
`hash(parent_hash, block_token_ids, extra_keys)` — a chained hash, so a hash at position k
uniquely fingerprints the *entire prefix* up to k. Only full blocks are cached.

> "we can build the block hash of `hash(tuple[components])`, where components are: Parent hash
> value: The hash value of the parent hash block. Block tokens: A tuple of tokens in this block.
> The reason to include the exact tokens is to reduce potential hash value collision. Extra
> hashes: Other values required to make this block unique, such as LoRA IDs, multi-modality input
> hashes ... and cache salts to isolate caches in multi-tenant environments." ... "Note 1: We only
> cache full blocks."

Source: https://docs.vllm.ai/en/latest/design/prefix_caching/ (docs/design/prefix_caching.md in
vllm-project/vllm)

Code (fetched from raw.githubusercontent.com, vllm/v1/core/kv_cache_utils.py @ main):

> ```python
> def hash_block_tokens(
>     hash_function,
>     parent_block_hash: BlockHash | None,
>     curr_block_token_ids: Sequence[int],
>     extra_keys: tuple[Any, ...] | None = None,
> ) -> BlockHash:
>     ...
>     if not parent_block_hash:
>         parent_block_hash = NONE_HASH
>     curr_block_token_ids_tuple = tuple(curr_block_token_ids)
>     return BlockHash(
>         hash_function((parent_block_hash, curr_block_token_ids_tuple, extra_keys))
>     )
> ```

`BlockHash = NewType("BlockHash", bytes)`; group id is packed into the key as
`BlockHash + group_id.to_bytes(4, "big")` (`make_block_hash_with_group_id`). Extra keys are
LoRA name + multimodal feature identifiers + `cache_salt` (first block only) + prompt-embeds
hash (`generate_block_hash_extra_keys`, same file). `get_request_block_hasher` walks the request
token ids block-by-block, chaining `prev_block_hash_value` — and early-stops at the first
non-full block ("We only hash full blocks").

Source: https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/core/kv_cache_utils.py

Transfer to memra: the chained-hash structure is exactly the "bytes decide" half of memra's
design — a per-block chain hash makes "how much of the rewritten history is byte-identical"
answerable in O(matched blocks) lookups with no per-token scan past the first divergence, because
a miss at block k implies misses at all k+.

### 1.2 The partial-match walk: divergence at block k invalidates k+ automatically

Claim: on a new request, the scheduler hashes the full prompt and walks block hashes from the
start; the first missing hash terminates the walk (chained hashes make later hits impossible).
A history edited mid-prefix therefore silently degrades to "reuse up to the block containing the
edit, recompute everything after" — no explicit invalidation is ever performed; the stale blocks
just stop being reachable under the new chain and age out via LRU.

Code (fetched from raw.githubusercontent.com, vllm/v1/core/single_type_kv_cache_manager.py @ main,
`FullAttentionManager.find_longest_cache_hit`):

> ```python
> # Phase 1: longest run of cached full blocks from the start. A missing
> # block implies every later block misses too (chained hashes).
> for block_hash in itertools.islice(full_block_hashes, max_length // block_size):
>     cached_block = block_pool.get_cached_block(block_hash, kv_cache_group_ids)
>     if not cached_block:
>         break
> ```

There is also a "Phase 2 (fine-grained only)" that probes *interior* hash boundaries of the first
non-full block "high-to-low (longest hit first)" when the hash granularity is finer than the
block size — i.e., vLLM now supports hash_block_size < physical block size to reduce the
tail-waste of the "full blocks only" rule.

Source: https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/core/single_type_kv_cache_manager.py

Transfer to memra: memra's byte-compare resume should mirror this: never "invalidate" on edit,
just walk the chain and stop; and consider a finer checkpoint granularity than the reuse
granularity so an edit near a boundary doesn't forfeit a whole large block.

### 1.3 Eviction: ref-counted blocks + LRU free queue, reverse-order free, tail-first eviction

Claim: every block carries a `ref_cnt`; blocks with ref_cnt=0 sit in a doubly-linked free queue
but *stay cached* (hash still mapped) until they're popped for reallocation — eviction happens
lazily at allocation time. Freed request blocks enter the queue tail in reverse order so deepest
(least reusable) blocks evict first.

> "the freed blocks are added to the tail of the free queue in the *reverse* order. This is
> because the last block of a request must hash more tokens and is less likely to be reused by
> other requests. As a result, it should be evicted first."
> ... "'Touch' the computed blocks. It increases the reference count of the computed block by one,
> and removes the block from the free queue if the block wasn't used by other requests. This is
> to avoid these computed blocks being evicted."

Source: https://docs.vllm.ai/en/latest/design/prefix_caching/

`KVCacheBlock` (kv_cache_utils.py): `block_id`, `ref_cnt`, `_block_hash`,
`prev_free_block`/`next_free_block` (intrusive doubly-linked free list, O(1) mid-queue removal),
plus `is_null` for never-cached placeholder blocks. All blocks are pre-allocated into a pool at
init ("avoids Python object creation overheads").

Transfer to memra: "free ≠ evicted" is the key trick — an agent session that pauses on a tool
call keeps its blocks *reusable* at zero cost, they only die if a new allocation actually needs
the memory. memra's checkpoint entries should be exactly this: unreferenced-but-indexed until
capacity pressure, with depth-aware (tail-first) eviction order within a session.

### 1.4 What they got WRONG (a): builtin hash() → predictable collisions → SHA-256 default

Claim: vLLM originally hashed blocks with Python's builtin `hash()`. Python 3.12 made
`hash(None)` a predictable constant, making cross-request collision attacks feasible; an attacker
could poison the cache so another user's prompt reused wrong-content KV. Fixed in stages:
randomize the seed (PR #12621, advisory GHSA-rm76-4mrf-v9r8 / CVE-2025-25183), then default to
SHA-256 in v0.11.

> "Maliciously constructed prompts can lead to hash collisions, resulting in prefix cache reuse,
> which can interfere with subsequent responses and cause unintended behavior. ... vLLM's prefix
> caching makes use of Python's built-in hash() function. As of Python 3.12, the behavior of
> hash(None) has changed to be a predictable constant value."

Source: https://github.com/vllm-project/vllm/security/advisories/GHSA-rm76-4mrf-v9r8 (fix PR:
https://github.com/vllm-project/vllm/pull/12621 ; SHA-256 option PR:
https://github.com/vllm-project/vllm/pull/15297)

> "In previous versions, the hash key was not guaranteed to be collision-free. As of v0.11, the
> default hashing algorithm is `sha256`, which addresses collision risks."

Source: https://docs.vllm.ai/en/latest/design/prefix_caching/ — which also documents the
`--prefix-caching-hash-algo` choices: `sha256` (pickle serialization, NOT reproducible across
Python/vLLM versions), `sha256_cbor` (canonical, cross-language reproducible), `xxhash`,
`xxhash_cbor`. `init_none_hash` in kv_cache_utils.py seeds NONE_HASH from `os.urandom(32)` when
PYTHONHASHSEED is unset, and warns that CBOR hashing is non-reproducible without it.

Transfer to memra: memra's "bytes decide" design side-steps this whole class — if resume is
decided by comparing actual token bytes rather than trusting a hash match, a collision costs a
wasted candidate check, never wrong KV. Keep byte-verification even if a hash index is added for
speed; hash = nomination, bytes = decision. (This is precisely the lesson vLLM paid for.)

### 1.5 What they got WRONG (b): extra-key omissions — the recurring correctness bug family

Claim: every input that changes KV but is not in the token ids must be an extra hash key, and
vLLM has repeatedly shipped gaps here:

- LoRA: block hash keyed on `lora_name`, not `lora_int_id` — two adapters with the same name but
  different weights share KV blocks:
  > "Two `LoRARequest` objects with the same `lora_name` but different `lora_int_id` are
  > considered equal ... This function is called in `_gen_extra_hash_keys()` ... and contributes
  > to the block hash used for prefix caching. Two configs with the same `lora_name` but
  > different weights will have identical cache hashes."
  Source: https://github.com/vllm-project/vllm/issues/30931 (open at fetch time; code confirmed
  in current main: `_gen_lora_extra_hash_keys` returns `[request.lora_request.lora_name]`).
- Multimodal: "[Bug]: Prefix caching ignores visual input, causing incorrect multimodal outputs
  under concurrency" — https://github.com/vllm-project/vllm/issues/20261 (qwen2.5-vl, garbled
  output under high concurrency with --enable-prefix-caching).
- Perf + garbage under batch: "[Bug]: Prefix caching with larger batch-sizes and large prompts is
  much slower and occasionally outputs garbage" — https://github.com/vllm-project/vllm/issues/22808
  (Qwen2.5-VL, temperature=0, 32-batch, ~2K-token shared system prompt).
- Numerics, not hashing: on ROCm MI355X, cache-hit vs cache-miss paths produced *different
  outputs* for identical prompts (first request full-prefill differs from later cached requests)
  — https://github.com/vllm-project/vllm/issues/33123. Cache reuse changes which kernel path
  computes the prefix, and kernels are not bitwise-identical across paths.

Transfer to memra: enumerate every KV-affecting input outside the token stream (model+quant
revision, template/BOS handling, spec-decode drafter state if it colors KV, RoPE scaling config,
KV cache dtype) and fold them into the cache-entry identity; and gate the resume path in the
battery with an argmax cross-check between cold-prefill and resumed-prefill outputs — vLLM's ROCm
issue shows reuse can be hash-correct and still numerically divergent.

### 1.6 Preemption interaction + duplicated blocks (v1 append-only block table)

Claim: under KV pressure vLLM preempts by *dropping* the request's blocks and recomputing later —
which is exactly where prefix caching pays back (recompute becomes a cache walk if blocks
survived). Docs: "Preempted requests are recomputed when sufficient KV cache space becomes
available again" (https://docs.vllm.ai/en/stable/configuration/optimization/ — the
`PreemptionMode.RECOMPUTE` warning). Also, v1's block table is append-only, so a same-content
block computed concurrently by two requests is cached twice until request free:

> "the block table in vLLM v1 is append-only ... As a result, we will have duplicated blocks for
> the hash key E-H. This duplication will be eliminated when the request is freed."

Source: https://docs.vllm.ai/en/latest/design/prefix_caching/

Also relevant: TRT-LLM has the same first-request race (see §2.4); vLLM partially avoids it by
caching full blocks *immediately when they fill*, "so that the block can be reused by other
requests in the same batch" (same doc, allocation step 4).

Transfer to memra: publish checkpoint blocks as soon as they complete, not at request end —
agent turns often overlap (turn N+1 arrives while N still streams in other sessions), and
end-of-request publication forfeits intra-flight reuse.

### 1.7 Cascade/prefix-sharing compute limitation

Claim: prefix *caching* saves prefill; it does not by itself make decode attention over the
shared prefix cheaper. vLLM's cascade attention only kicks in when ALL requests in the batch
share one common prefix (single tree), and cannot handle different prefixes per group.

> "Currently, V1 only uses cascade attention when all requests in the batch share the same prefix
> (i.e., a single tree). We want to extend this to support a forest (multiple trees)."

Source: https://github.com/vllm-project/vllm/issues/14729 ; see also
https://github.com/vllm-project/vllm/issues/12080 ("When it comes to vllm's cascade attention, it
cannot support different common prefixes for different requests in one batch").

Transfer to memra: single-model single-user memra rarely batches divergent sessions, so this
limitation costs little — but if darklanes serving batches multiple agent sessions on card 1,
shared-system-prompt decode is a separate (compute) optimization from the (memory) prefix cache,
and vLLM's experience says don't conflate them.

### 1.8 CPU offloading / KV connector API + the layout lesson

Claim: vLLM 0.11 added an OffloadingConnector (async, pluggable backend, CPU tier bundled);
transfers use cudaMemcpyAsync DMA. The decisive perf factor was *KV layout*: vLLM's default
per-layer (and per-K/V) fragmentation made the effective transfer block a few KB; a layout change
making one contiguous physical block per logical block (all layers) multiplied block size by
2*num_layers and sped offloading by ~10x.

> "This fragmentation is meaningless for model computation performance, but is devastating for KV
> offloading ... we recently upstreamed a change in vLLM's KV cache layout which creates one
> contiguous physical block including the KV data of all layers. This change effectively increased
> the physical block size by a factor of 2*num_layers, and this in turn increased the throughput
> of the offloading connector by an order of magnitude."

and: "loading KV values from the CPU reduces TTFT by X2-X22, depending on the prompt size" ...
"DMA achieves 83.4 GB/s" bidirectional on H100; "using the offloading connector has minimal
effect on TTFT for cache misses" (offload is async, off the critical path). New layout physical
blocks: ~0.5–2 MB for common 1B–70B models at 16-token blocks.

Source: https://vllm.ai/blog/2026-01-08-kv-offloading-connector (vLLM blog, 2026-01-08; layout PR
https://github.com/vllm-project/vllm/pull/27743). Known follow-up bugs the blog itself lists:
preempted requests couldn't load back from CPU (PR #29870) and a race between offloading and model
computation (PR #31341) — both fixed post-0.12.

Transfer to memra: if memra adds a host tier for checkpoints, store a checkpoint as one
contiguous all-layer slab per block (or per checkpoint) so H2D restore is a single DMA per unit —
memra owns its KV layout, so it can bake this in from day one instead of retrofitting like vLLM.
On PCIe 5 the restore of a multi-hundred-MB checkpoint is tens of ms — far cheaper than the 5.85x
TTFT regrowth. Keep D2H off the owner-thread critical path (matches memra's existing CUDA
owner-thread discipline).

### 1.9 The agentic-eviction RFC: LRU is the wrong policy for tool-call pauses (memra's exact workload)

Claim: vLLM/llm-d filed an RFC (March 2026) describing precisely the Claude-Code-style failure
mode and proposing token-range retention priorities.

> "Agentic workloads break prefix caching under concurrent load. Over 90% of tokens in a typical
> agent turn are prefixes reused verbatim from the previous turn. With prefix caching, this is a
> hit. But 40–60% of session wall time is spent paused on tool calls, and during those pauses the
> agent's blocks are unreferenced. Under concurrent load, other agents evict them via LRU. When
> the session resumes: cache miss, full recomputation of the entire context — which grows to
> 70K–200K tokens by session end."

Proposed: `RetentionDirective {start, end, priority 0-100, duration}` per token range +
`retention_scope` (session id), two-tier evictor (plain LRU queue for unprioritized blocks,
min-heap for prioritized). Cites evidence that LRU loses to structure-aware eviction: 10% of KV
blocks account for 77% of reuses (arXiv:2506.02634); TTL retention 1.12–3.66x delay reduction on
SWE-Bench (Continuum, arXiv:2511.02230); cost-aware eviction up to 34.4x hit rate (MARCONI,
arXiv:2411.19379).

Source: https://github.com/vllm-project/vllm/issues/37003 (RFC, open; "We have a working
implementation" per the issue).

Transfer to memra: this is independent confirmation of memra's session-id "identity nominates"
half — the session id is the retention scope. For single-node memra: give session-tail
checkpoints a TTL-style protected class (agent tool-call pauses are minutes, not hours) and evict
cross-session by LRU only within the unprotected class.

### 1.10 Multi-turn reuse works only if history is byte-stable — templating gotchas

Claim: multi-turn APC issues in the tracker repeatedly root-cause to the *client or template*
changing bytes at the head of the prompt (or hitting non-cachable paths), not the cache itself:
e.g. https://github.com/vllm-project/vllm/issues/4917 ("Automatic Prefix Caching in multi-turn
conversations" — no observed benefit) and
https://github.com/vllm-project/vllm/issues/31920 ("Prefix cache hit rate remains 0 in
multi-round conversation with history of identical prompts", ROCm, Jan 2026). UNVERIFIED: I did
not fetch the resolution comments of these two issues; cited for existence of the failure class,
not for their individual root causes.

Transfer to memra: memra's serve surface should log, per request, the matched-prefix length vs
prompt length (vLLM exposes hit-rate counters for exactly this triage). A single dashboard number
("reused/total tokens this turn") would have caught the frozen-snapshot bug immediately.

---

## 2. TensorRT-LLM — KV cache block reuse

### 2.1 Radix-tree over blocks; reuse on by default; partial (in-block) reuse with copy-on-partial-reuse

Claim: filled blocks enter a radix search tree keyed by tokens; new requests search the tree and
share matched blocks. Unlike vLLM's "full blocks only", TRT-LLM supports partial reuse of the
last block, with an optional copy so multiple requests can share a partially-matched block.

> "Blocks containing KV state computed for previous requests are stored in a radix search tree as
> soon as they are filled. A search is performed when a new request is added, and matched blocks
> are reused instead of calculated."
> ... "Partial reuse of a block can happen when some but not all tokens are matched. It is enabled
> by default ... The property `copy_on_partial_reuse` specifies whether a block should be copied
> or not in order to allow partial reuse. If copying is disabled, a partially matched block can
> only be reused if no other request is using it. If copying is enabled, partially matched blocks
> are not reused directly, instead a new block is created and the matched tokens are copied into
> the new block."

Source: https://nvidia.github.io/TensorRT-LLM/latest/features/kvcache.html ("KV Cache System").
Enable/disable: `KvCacheConfig(enable_block_reuse=...)` (default true);
`scheduler_config.enable_prefix_aware_scheduling` additionally lets the scheduler defer duplicate
first-chunk context requests based on *estimated* reusable tokens.

Transfer to memra: partial-block reuse + copy-on-partial-reuse is the exact mechanism for the
"history edited mid-block" case: reuse the matched head of the divergent block by copying it into
a fresh block, so the old entry stays valid for the old chain and the new turn gets the partial
credit. vLLM only recently approximated this (fine-grained hash lookup); TRT-LLM has had it as a
first-class token-level radix walk.

### 2.2 Block size trade-off + the "reusable only after terminating" race

Claim: default 128 tokens/block (vs vLLM's 16); NVIDIA explicitly documents that big blocks kill
reuse; and blocks only become reusable when the producing request *terminates*, which starves
reuse for concurrent same-prefix launches.

> "Only full blocks can be shared by multiple requests, thus the block size matters. ... larger
> block size may improve efficiency of compute kernels, but it reduces the likelihood of kv cache
> state reuse. The block defaults to 128 tokens" (`trtllm-build --tokens_per_block 32`, power of 2)
> ... "KV cache state only becomes reusable after the request that computed the state terminates.
> If you have a shared system prompt, the first request will compute kv cache state for the system
> prompt, the second request will reuse it, but only if the second request launches after the
> first request completed."

Source: https://nvidia.github.io/TensorRT-LLM/advanced/kv-cache-reuse.html ("KV cache reuse",
legacy-advanced docs page)

Transfer to memra: two concrete numbers to sweep — memra's checkpoint granularity should be
nearer vLLM's 16–64 than TRT's 128 for edit-heavy agent histories (every edit wastes on average
half a block); and publish-on-fill, not publish-on-terminate (see §1.6).

### 2.3 Priority-based eviction + retention config (the productized version of vLLM's RFC)

Claim: eviction is prioritized LRU: priorities 0–100, all lowest-priority blocks evicted before
any higher-priority block, LRU within a class; per-token-range retention with expiring durations,
separate decode-block priority; default priority 35; leaf-only eviction (radix leaves) is a
documented current limitation.

> "The core eviction scheme is prioritized LRU. All blocks are assigned a priority between 0 and
> 100 (100 being most important). All blocks of the lowest priority must be evicted before any
> blocks of the next priority can be evicted. If all blocks have the same priority, the least
> recently used block is evicted." ... "One caveat in the current code is that only leaf blocks
> can be evicted (leaves are blocks with no descendants in the radix tree)."

Source: https://nvidia.github.io/TensorRT-LLM/latest/features/kvcache.html

> "a request with a 500-token system prompt can set the token range [0, 500) to the maximum
> priority. This way, the cache blocks corresponding to these tokens will only be evicted if
> absolutely necessary." ... "This new implementation also biases toward blocks further from the
> root, which leads to a small performance improvement, even when not setting priority levels.
> Our internal benchmarks show priority-based eviction increasing cache hit rate by around 20%."

Source: https://developer.nvidia.com/blog/introducing-new-kv-cache-reuse-optimizations-in-nvidia-tensorrt-llm/
(also documents the KV cache event API: StoredData/RemovedData events with blockHash + parentHash,
consumed by external routers — the "identity" layer externalized).

Transfer to memra: a tiny version of this is enough for memra: two or three priority classes
(system-prompt blocks / live-session tails / everything else) + duration-based decay, instead of
a full 0–100 API. The "biases toward blocks further from the root" default matches vLLM's
reverse-order free — both engines converged on depth-aware eviction independently.

### 2.4 Host (secondary) offloading

Claim: evicted-from-GPU blocks are copied to a pinned host buffer and *stay in the radix tree*;
they onboard (H2D) on rehit; eviction policy in host tier mirrors the GPU tier; a priority
threshold (`secondary_offload_min_priority`, default 35) filters which blocks are worth the PCIe
traffic.

> "When a block is evicted from primary memory, its KV state is copied to a block in secondary
> memory. The secondary memory block remains in the search tree, so the block remains reusable
> until it is evicted from secondary memory." ... "Offloading is controlled with property
> `host_cache_size` ... The default is 0." ... "Blocks with lower priority than a certain
> threshold are not offloaded; they are evicted directly from GPU memory to reduce traffic."

Source: https://nvidia.github.io/TensorRT-LLM/latest/features/kvcache.html ; older page adds:
"this buffer is pinned memory, allocating a lot of pinned memory on x86 machines can take a
substantial amount of time (10s of seconds). This is a one-time cost." and "This cost is
negligible on Grace-Hopper machines, and small enough to yield a net benefit for many use cases
on x86 machines with Hopper GPUs."
(https://nvidia.github.io/TensorRT-LLM/advanced/kv-cache-reuse.html)

Transfer to memra: the unified index over GPU+host tiers ("the block remains in the search tree")
is the important design bit — one lookup structure, per-entry location tag, demand-driven
onboarding. memra already has bounded pinned host buffers in the spill path; a KV host tier can
reuse that allocator discipline. Also budget the pinned-alloc startup cost into serve bring-up.

### 2.5 What they got WRONG

- FP8 KV cache + block reuse produced wrong outputs (200 identical prompts diverge wildly with
  reuse on, correct with reuse off; FP16 KV unaffected): "With kvcache reuse, the outputs are
  different and almost totally wrong. ... FP16 kvcache reuse seems don't have this problem."
  Source: https://github.com/NVIDIA/TensorRT-LLM/issues/2699 (TRT-LLM 0.16.0, RTX 4090,
  Qwen2.5-7B, `--use_fp8_context_fmha enable`). Quantized-KV reuse is a distinct correctness
  surface from FP16 reuse — per-block scale/dequant state has to survive the reuse path.
- p-tuning aliasing: virtual-token ids larger than vocab collide across requests with different
  prompt tables; TRT-LLM requires user-supplied uint64 `extra_ids` to disambiguate — i.e. the
  "extra hash key" burden is pushed onto the client: "different requests may use same fake input
  ids ... That may lead to incorrect kv cache reuse, since TRT-LLM could not distinguish these
  requests only by input ids."
  Source: https://nvidia.github.io/TensorRT-LLM/advanced/kv-cache-reuse.html
- Eviction raced an async disaggregated-serving transfer: "KV cache block evicted from reuse tree
  before async CacheSender completes transfer, causing silent deadlock" — assertion "Couldn't
  find the requested block in the reuse tree".
  Source: https://github.com/NVIDIA/TensorRT-LLM/issues/12542 (March 2026).
- VSWA models (Gemma-3-style sliding-window/global mix): "kv_cache_enable_block_reuse=True
  produces no prefix caching" on the TRT engine path.
  Source: https://github.com/NVIDIA/TensorRT-LLM/issues/12563. Interacts with the documented
  leaf-only-eviction caveat ("works well for full attention layers, but not for limited attention
  layers").
- Guided decoding (xgrammar) + block reuse segfaulted:
  https://github.com/NVIDIA/TensorRT-LLM/issues/2660.

Transfer to memra: three memra-relevant hazards: (1) if memra ever reuses quantized-KV blocks,
add an explicit quantized-KV-reuse arm to the battery (FP16-green does not imply FP8/NVFP4-green);
(2) any async consumer of a cache entry (offload writer, PP-2 stage transfer) must hold a
refcount — eviction must be impossible mid-transfer, not just unlikely; (3) attention-window
variety (if memra serves SWA models) breaks naive prefix reuse — scope the first implementation
to full-attention KV.

---

## 3. LMCache

### 3.1 Architecture: 256-token chunks, chained prefix hash, key includes model+world-size+dtype

Claim: LMCache indexes KV in chunks (default 256 tokens), each chunk keyed by a chained prefix
hash (same fold as vLLM: `hash((prefix_hash, tokens_tuple, extra_keys))`), wrapped in a
`CacheEngineKey` that also carries model name, world size, worker id, and KV dtype. Unfull-chunk
saving is configurable (`save_unfull_chunk`, default false in config docs, true in code default).

Code (fetched from raw.githubusercontent.com, LMCache/LMCache @ dev, lmcache/v1/token_database.py):

> ```python
> class ChunkedTokenDatabase(TokenDatabase):
>     ...
>     self.chunk_size = config.chunk_size   # default 256
>     def _prefix_hash(self, token_chunks):
>         prefix_hash = self._get_init_hash()
>         for token_chunk in token_chunks:
>             prefix_hash = self._hash_tokens(token_chunk, prefix_hash)
>             yield prefix_hash
> ```

and the key: `CacheEngineKey(model_name, world_size, worker_id, chunk_hash, kv_dtype,
request_configs)`. Notable wart in the same file: "# Ignore extra keys for now / Extra keys are
for multi-modal inputs and request specific metadata (e.g., LoRA ID)." — LMCache's chunk hash
currently drops the very extra-keys that vLLM's bug history (§1.5) shows are load-bearing.
Also: byte digests are truncated to 8 bytes ("the first eight bytes of a digest as a big-endian
int") to fit msgpack serialization — a deliberate 64-bit collision surface on top of vLLM's
256-bit hashes.

Source: https://raw.githubusercontent.com/LMCache/LMCache/dev/lmcache/v1/token_database.py

Config reference (chunk_size 256; local_cpu default true, max_local_cpu_size 5 GB;
max_local_disk_size; cache_policy "LRU"/"LFU"/"FIFO"; pre_caching_hash_algorithm default
"builtin"; remote_serde "naive" or "cachegen"): https://docs.lmcache.ai/api_reference/configurations.html
The docs and code repeatedly warn that with the builtin hash, "For production environments
... you MUST set PYTHONHASHSEED" or cross-process sharing breaks ("This will cause incorrect KV
cache transfer" for PD disaggregation).

Transfer to memra: 256-token chunking is tuned for transfer amortization (fewer, bigger objects
for CPU/disk/remote tiers), not for edit granularity — a mid-history edit costs up to 255 tokens
of lost credit per divergent chunk. For memra: small blocks near the live tail (where edits
happen), large chunks for the frozen head (where transfer efficiency dominates) — a two-scale
scheme neither vLLM nor LMCache implements.

### 3.2 Storage tiers + eviction

Claim: tiers are GPU → pinned CPU DRAM → local disk/NVMe-GDS → remote (Redis/Valkey, Mooncake,
InfiniStore, S3, NIXL); async offload off the inference thread; LRU default eviction, pluggable
LFU/FIFO; controller API adds Pin/Unpin, Move, Compress (CacheGen), Clear, Lookup.

> "LMCache implements a hierarchical storage system with three distinct tiers ... **CPU DRAM**:
> Acts as a 'hot cache' for recently used KV chunks, using pinned memory for efficient GPU-CPU
> transfers ... **Asynchronous Offloading**: Offloading / loading the KV cache chunks in an
> asynchronous manner to avoid blocking inference threads and GPU cycles."

Source: https://docs.lmcache.ai/developer_guide/architecture.html

Standalone-daemon claim (survives engine restarts — relevant to memra restarts losing all cache):
"Engine-independent deployment: LMCache, as a standalone daemon process, manages KV cache
independently from the inference engine process, so that KV cache will not be lost even if the
inference engine crashes." Source: https://docs.lmcache.ai/

Transfer to memra: the memra-sized subset is GPU + pinned-CPU only; the transferable idea is
`min_retrieve_tokens` (skip retrieval when the hit is too small to beat recompute — LMCache
config) — memra's resume should have the same floor: below N matched tokens, cold prefill is
faster than restore.

### 3.3 CacheGen / CacheBlend (non-prefix reuse)

Claim: CacheGen is KV compression/streaming for cheaper storage/transfer (SIGCOMM'24,
https://dl.acm.org/doi/10.1145/3651890.3672274). CacheBlend fuses *non-prefix* cached chunks:
reuse precomputed KV of a chunk appearing at a different position, then selectively recompute a
small token subset to repair cross-attention.

> "We present CacheBlend, a scheme that reuses the pre-computed KV caches, regardless prefix or
> not, and selectively recomputes the KV values of a small subset of tokens to partially update
> each reused KV cache."

Source: https://arxiv.org/html/2405.16444v3 (CacheBlend paper, arXiv:2405.16444). LMCache wiring:
`enable_blending`, `blend_recompute_ratios` (default 0.15), `blend_check_layers` (default 1),
`blend_special_str` separator (https://docs.lmcache.ai/api_reference/configurations.html);
`SegmentTokenDatabase` in token_database.py splits on the separator tokens and hashes each segment
*without* prefix chaining (position-independent keys).

IMPORTANT caveat for memra: CacheBlend is *approximate* — recomputing only ~15% of tokens does
not reproduce full-prefill logits. For an edited mid-history turn, blend-style reuse of the
post-edit suffix would change outputs. memra's argmax-gate discipline rules this out as a default;
it could only ever be an explicitly-flagged door. The prefix-checkpoint design (exact) is the
right default; CacheBlend defines the quality-tradeoff frontier beyond it.

### 3.4 What they got WRONG / overclaimed

- Overhead when the workload has no reuse: "Performance worse with VLLM + LMCache then with VLLM
  (plain) on given benchmark" — a shared-prefix benchmark (70 docs x 15 prompts, H100,
  granite-3-8b) measured *higher* latency with LMCache than plain vLLM: "We can clearly see that
  the latency is higher with LMCache. Also other indicators show degraded performance."
  Source: https://github.com/LMCache/LMCache/issues/1812 (open, Oct 2025). Related:
  "[bug] TTFT was 2x slower than using single vllm" https://github.com/LMCache/LMCache/issues/1938.
- The lookup/copy layer is not free: retrieval competes with GPU prefill; when the GPU-side
  prefix cache would have hit anyway, LMCache's CPU round-trip is pure loss. (Same lesson as
  vLLM's `min_retrieve_tokens` knob existing at all.)
- Backend variance is huge: LMCache-over-Mooncake-store measured *slower than no cache at all*
  (29.0s retrieve vs 8.47s baseline) while local DRAM was 1.77s, from a single 32x10k-token test:
  table in https://github.com/kvcache-ai/Mooncake/issues/467 (also cites Mooncake store's
  internal 4MB page size causing "highly inefficient memory allocation and copying" for >4MB
  reads).

Transfer to memra: measure the *no-reuse* overhead of the checkpoint path as a first-class gate
number (memra CLAUDE.md discipline already demands this): the checkpoint write, index update, and
nomination probe must be ~free on cache-miss turns, or the feature is a regression for
non-agent workloads. LMCache shipped without that gate and collected the issues above.

---

## 4. Mooncake (Moonshot/Kimi)

### 4.1 Architecture: KVCache-centric disaggregation; 512-token chunk prefix hashes

Claim: Mooncake separates prefill and decode clusters and builds a distributed KV cache
("Mooncake Store") out of the cluster's idle CPU DRAM/SSD, moved by a GPUDirect-RDMA "Messenger";
a global scheduler ("Conductor") routes each request to the prefill node holding the longest
reusable prefix, balancing reuse against load and TTFT SLO. Dedup/identity uses chained
prefix-block hashes at 512-token granularity, remapped to global ids.

> "In CPU memory, KVCache is stored as paged blocks. Depending on the request patterns, it can
> use cache eviction algorithms such as LRU (Least Recently Used), LFU (Least Frequently Used),
> or algorithms based on request characteristics."
> "Each block is attached with a hash value determined by both its own hash and its prefix for
> deduplication." (Fig. 3 caption)
> "It is generated by hashing token blocks (with a block size of 512) into prefix hash values
> that include both the current and all preceding blocks ... Identical hash IDs indicate that a
> block of tokens, along with preceding tokens, are the same."

Source: https://arxiv.org/html/2407.00079v3 — "Mooncake: A KVCache-centric Disaggregated
Architecture for LLM Serving" (arXiv:2407.00079; FAST'25 version:
https://www.usenix.org/system/files/fast25-qin.pdf, not fetched — PDF; abstract page verified).
Open-source runtime: https://github.com/kvcache-ai/Mooncake (transfer engine + store; also the
open trace: 23,608 entries with `hash_ids` fields).

Honest scoping: prefill/decode disaggregation, Conductor scheduling, RDMA transfer, and hot-block
replication are all multi-node mechanisms with no direct transfer to a single-node 2-GPU engine.
The transferable residue is below.

### 4.2 Transferable idea 1: cache-aware admission/scheduling — reuse length is a scheduling input

Claim: the request carries "the block IDs of the prefix cache that can be reused" computed
*before* prefill starts; incremental prefill stores new KV back as it goes, layer-by-layer,
overlapped with compute.

> "1) KVCache Reuse: The selected prefill node (group) receives a request that includes the raw
> input, the block IDs of the prefix cache that can be reused, and the block IDs of the full
> cache allocated to the request." ... "2) Incremental Prefill: The prefill node (group) completes
> the prefill stage using the prefix cache and stores the newly generated incremental KVCache
> back into CPU memory." ... "the load and store operations of the KVCache layer are performed
> layer-by-layer and in parallel with the prefill computation to mitigate transmission overhead"

Source: https://arxiv.org/html/2407.00079v3 §3

Transfer to memra: resolve the nomination + byte-match *at admission time* (before allocating
prefill work), so the scheduler/admission gate (memra's c=64 admission work) can charge the
request only for its uncached suffix — this is exactly TRT-LLM's `enable_prefix_aware_scheduling`
too. And layer-wise overlapped restore: while layer L prefills, H2D layer L+1's checkpoint.

### 4.3 Transferable idea 2: measured reuse skew + eviction-policy evidence

Claim: from the Kimi production trace: block popularity is extremely skewed and plain LRU was the
*best* of the simple policies on their chat workload.

> "Increasing the cache capacity from 1,000 to 50,000 blocks boosts the cache hit ratio from 30%
> to 50%. Further capacity increases show minimal improvement." ... "over 50% of cache blocks
> remaining unused while certain blocks are accessed tens of thousands of times" ...
> "LRUCache performs best under this dataset's patterns, likely due to the temporal proximity in
> request utilization."

Source: https://arxiv.org/html/2407.00079v3 §4.2 (Table 1, Fig. 6)

Transfer to memra: two usable numbers — (a) hit-rate saturates with capacity: a modest host-tier
budget captures most of the value, don't over-provision pinned memory; (b) LRU-with-session-
protection is a defensible default (temporal proximity dominates in conversational traffic); the
fancy policies (LFU, cost-aware) only pay off under multi-tenant contention (cf. §1.9's counter-
evidence for *agentic* concurrent load — the difference is contention, not workload).

### 4.4 What they got WRONG / rough edges

- Mooncake Store's 4MB internal page size caused pathological alloc/copy for larger reads when
  used as an LMCache backend; end-to-end retrieve 29.0s vs 8.47s no-cache baseline in that test
  ("Mooncake turned out to be unexpectedly slow ... When handling reads larger than 4MB, it
  incurs highly inefficient memory allocation and copying, which can take several times longer
  than the actual data transfer.")
  Source: https://github.com/kvcache-ai/Mooncake/issues/467 (June 2025).
- The paper itself concedes static capacity partitioning is suboptimal (in TRT-LLM's words for
  the same problem: "The fraction to assign to each pool is determined during initialization and
  is static. This is not optimal" — nvidia.github.io/TensorRT-LLM/latest/features/kvcache.html);
  Mooncake's equivalents are the prediction-based early-rejection ("wasted computational
  resources" under load fluctuation, §Conclusion of the paper — UNVERIFIED exact wording, section
  not fully fetched).

Transfer to memra: tier-internal granularity mismatch (engine block size vs store page size) is a
silent perf killer — if memra's host tier stores checkpoints, the storage unit should be exactly
the transfer unit (one contiguous slab per checkpoint block), no repacking on either side.

---

## Strongest transferable mechanisms (ranked for memra)

1. **Chained per-block hash as the resume index** (vLLM `hash_block_tokens`, Mooncake 512-token
   variant, LMCache 256-token variant): hash(parent, tokens, extra) chained from the prompt
   start. Gives O(1)-per-block longest-prefix match and makes mid-history edits self-invalidating
   (divergence at block k kills k+ automatically, no explicit invalidation). Pair it with memra's
   byte-compare so hash is nomination-only — vLLM's CVE (GHSA-rm76-4mrf-v9r8) is the argument for
   never letting the hash alone decide.
2. **Free ≠ evicted: ref-count + lazy LRU eviction at allocation time** (vLLM KVCacheBlock free
   queue; TRT-LLM radix tree). Session pauses keep entries reusable at zero cost. Add depth-aware
   ordering (vLLM's reverse-order free / TRT's "biases toward blocks further from the root").
3. **Partial-block reuse with copy-on-partial-reuse** (TRT-LLM): recover the matched head of the
   block containing the edit by copying into a fresh block — directly reduces per-edit waste from
   ~block_size/2 to ~0.
4. **Session-scoped retention priority over plain LRU** (TRT-LLM KvCacheRetentionConfig, vLLM RFC
   #37003): protect live-session tails and the system prompt with a small priority/TTL class;
   the RFC documents memra's exact tool-call-pause eviction failure mode with production
   evidence that LRU alone loses.
5. **Contiguous all-layer KV slabs for the host tier** (vLLM PR #27743, ~10x offload throughput;
   Mooncake store's 4MB-page mismatch as the negative example): storage unit == transfer unit ==
   one DMA. memra owns its layout — bake this in.
6. **Prefix-aware admission**: compute matched-prefix length before scheduling and charge only
   the uncached suffix (Mooncake Conductor step 1, TRT-LLM enable_prefix_aware_scheduling) —
   plugs straight into memra's existing admission gate.
7. **Publish checkpoints on block-fill, not request-end** (vLLM allocation step 4; TRT-LLM's
   terminate-first rule as the anti-pattern).
8. **Observability as a correctness gate**: per-request reused-tokens/prompt-tokens counter
   (vLLM hit-rate metrics, TRT-LLM KV event API). The frozen-snapshot 5.85x bug is exactly what
   this number catches on turn 2.
9. **A no-reuse overhead gate**: LMCache #1812/#1938 are what shipping without one looks like.
   The checkpoint path must be measurably ~free on miss-only workloads.

## Open uncertainties

- vLLM's fine-grained hash lookup (hash_block_size < block_size, `BlockHashListWithBlockSize`,
  Phase 2 interior probing) — verified in code at main, but I did not pin which release
  introduced it or its default on/off state; the elided middle of
  single_type_kv_cache_manager.py (bytes 19.7K) wasn't fully read.
- The LoRA lora_name-vs-lora_int_id cache-corruption issue (#30931) was open at fetch time;
  whether a fix has landed since is unverified.
- TRT-LLM FP8-KV-reuse wrong-output issue (#2699): resolution status and root cause (scale
  handling vs kernel path) unverified — only the report and repro were read.
- vLLM issues #4917 and #31920 (multi-turn zero-hit reports): cited for the failure class; their
  individual root-cause comment threads were not fetched.
- Mooncake FAST'25 PDF not fetched (binary); all Mooncake quotes are from the arXiv v3 HTML,
  which may differ in details from the FAST'25 camera-ready.
- LMCache "extra keys ignored for now" (token_database.py comment): whether multimodal/LoRA
  correctness bugs analogous to vLLM's exist in LMCache deployments is inferred risk, not a
  documented incident.
- The exact interaction of vLLM APC with speculative decoding (EAGLE/MTP drop-last-matched-block
  behavior seen in `find_longest_cache_hit`'s `drop_eagle_block` arg — the last matched block is
  recomputed to recover hidden states for the draft head) — verified the mechanism exists in the
  signature/docstring, but not its cost profile. Directly relevant to memra's MTP: a resume that
  is 100% cache-hit still needs the last block recomputed (or hidden states checkpointed) to
  re-arm the drafter.
