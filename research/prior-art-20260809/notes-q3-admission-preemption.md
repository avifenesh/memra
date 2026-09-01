# Prior art Q3: admission control, preemption, swapping, session retirement/reclaim

Survey date: 2026-08-09. All quotes fetched and verified from the cited source on that date
unless marked UNVERIFIED. Context: memra's parked-session admission fix (spec-shrink reserve
charged at admission; gate reads free + pool_cached; parked sessions evictable/deferrable).

---

## 1. vLLM

### 1.1 Admission gate (v0): `can_allocate` + watermark in the block manager

Claim: v0 admission is a per-request block check against free GPU blocks minus a **watermark**
(default 1% of GPU blocks) that exists explicitly to avoid thrashing ("frequent cache
eviction"). Three-way verdict: `OK` / `LATER` (defer) / `NEVER` (reject as unschedulable).

> ```python
> watermark: float = 0.01,
> ...
> self.watermark_blocks = int(watermark * num_gpu_blocks)
> ...
> # Use watermark to avoid frequent cache eviction.
> if (self.num_total_gpu_blocks - num_required_blocks < self.watermark_blocks):
>     return AllocStatus.NEVER
> if num_free_gpu_blocks - num_required_blocks >= self.watermark_blocks:
>     return AllocStatus.OK
> else:
>     return AllocStatus.LATER
> ```

Source: `vllm/core/block_manager.py` (`SelfAttnBlockSpaceManager.can_allocate`), tag v0.6.6 —
https://github.com/vllm-project/vllm/blob/v0.6.6/vllm/core/block_manager.py

Transfer to memra: the watermark is exactly the "reserve charged at the gate" idea, but as a
flat fraction rather than a mechanism-specific quantity — memra's SPEC_SHRINK_RESERVE is
strictly more precise (charged only on spec-capable models, sized from the measured live/parked
delta + capture-arena transient). The three-way `OK/LATER/NEVER` verdict (defer vs.
never-fits-reject) is worth copying: memra's defer path exists, but an explicit "NEVER at this
config" fast-reject for requests that can't fit even on an empty pool avoids infinite deferral.

### 1.2 Preemption modes: RECOMPUTE vs SWAP, and the cost warning

Claim: when KV space runs out mid-decode, v0 preempts the lowest-priority running sequence
group. Default mode is RECOMPUTE (free blocks, re-prefill later); SWAP (copy blocks to CPU
"swap space", copy back later) is used automatically for multi-sequence groups (beam search)
or when the user forces `--preemption-mode swap`. Every 50th preemption logs a warning.

> ```python
> # We use recomputation by default since it incurs lower overhead than
> # swapping. However, when the sequence group has multiple sequences
> # (e.g., beam search), recomputation is not currently supported. In
> # such a case, we use swapping instead.
> ...
> logger.warning(
>     "Sequence group %s is preempted by %s mode because there is "
>     "not enough KV cache space. This can affect the end-to-end "
>     "performance. Increase gpu_memory_utilization or "
>     "tensor_parallel_size to provide more KV cache memory. "
>     "total_num_cumulative_preemption=%d", ...)
> ```

Source: `vllm/core/scheduler.py` `_preempt()`, tag v0.6.6 —
https://github.com/vllm-project/vllm/blob/v0.6.6/vllm/core/scheduler.py

Docs confirm the user-facing framing and the same warning text ("Sequence group 0 is preempted
by PreemptionMode.RECOMPUTE mode because there is not enough KV cache space...") —
https://docs.vllm.ai/en/latest/configuration/optimization/ (§Preemption).

Swap space knob: `--swap-space` = "CPU swap space size (GiB) per GPU", default 4 GiB
(`vllm/engine/arg_utils.py`, v0.6.6, `swap_space: float = 4  # GiB`).

Transfer to memra: memra's park-then-evict is closer to vLLM SWAP semantics (state survives,
resume cheap) but memra keeps parked KV **on-GPU** until pressure — a third point on the
spectrum neither vLLM mode occupies. The always-on cumulative-preemption warning (rate-limited
to every 50th) is directly copyable for memra's defer/park events: a single greppable line with
a cumulative counter is what made vLLM preemption thrash diagnosable in the field.

### 1.3 v1: recompute-only preemption, no swap

Claim: the v1 scheduler dropped SWAP entirely. On allocation failure it pops the last (or
lowest-priority) running request, frees its blocks, resets `num_computed_tokens = 0`, and
prepends it to the waiting queue — pure recompute. There is no `blocks_to_swap_*` anywhere in
the v1 scheduler. Prefix caching softens the recompute cost (freed blocks retain their hashes
until actually evicted, so a preempted request can re-hit its own blocks — see
`cache_full_blocks` comment: "cache it in the request in case it will be preempted in the
future", `vllm/v1/core/block_pool.py`, v0.10.0).

> ```python
> preempted_req = self.running.pop()
> self.kv_cache_manager.free(preempted_req)
> preempted_req.status = RequestStatus.PREEMPTED
> preempted_req.num_computed_tokens = 0
> ...
> self.waiting.prepend_request(preempted_req)
> ```

Source: `vllm/v1/core/sched/scheduler.py`, tag v0.10.0 —
https://github.com/vllm-project/vllm/blob/v0.10.0/vllm/v1/core/sched/scheduler.py

Docs: "In vLLM V1, the default preemption mode is `RECOMPUTE` rather than `SWAP`, as
recomputation has lower overhead in the V1 architecture."
(https://docs.vllm.ai/en/latest/configuration/optimization/)

Host offload in v1 moved into the **KV connector** layer instead:
`OffloadingConnector` ("extends the prefix cache by offloading completed KV blocks to slower
but larger tiers (CPU host memory, plus optional secondary tiers) as they are produced. Hits in
the offload tiers are promoted back to GPU on demand... Transfers ... use DMA
(`cudaMemcpyAsync`) and run asynchronously"), with an explicit host budget
`cpu_bytes_to_use` (required, "Total bytes of host memory reserved for the CPU tier across all
workers") and `eviction_policy: lru|arc`. Source:
https://docs.vllm.ai/en/latest/features/kv_offloading_usage/ ; motivation blog explicitly
frames it as the anti-preemption mechanism: "The cost for re-computing the KV values can be
avoided by offloading the KV cache to a larger tier (such as CPU DRAM) before the request is
preempted" (https://vllm.ai/blog/2026-01-08-kv-offloading-connector). The connector had a
preemption-interaction bug fixed in PR #29870 ("fixes the OffloadingConnector to fail any
stores for a preempted request") — evidence the offload path and the preemption path compose
non-trivially.

Transfer to memra: v1's verdict — with prefix caching, recompute beats block-granular
GPU↔CPU swap, and host memory is better spent as an async *prefix-cache extension* than as a
synchronous swap target. If memra ever spills parked sessions to host, the vLLM v1 lesson is:
do it as an async write-behind of completed KV (so eviction is free later), not as an eviction-
time synchronous copy.

### 1.4 Spec-decode growth at admission (lookahead)

Claim: both generations charge speculative growth at allocation time. v0 block manager has
"lookahead slots": "Speculative decoding uses lookahead slots to store KV activations of
proposal tokens" and `can_allocate(seq_group, num_lookahead_slots)` includes them in
`num_required_blocks`. v1 passes `num_lookahead_tokens=self.num_lookahead_tokens` (set to
`num_speculative_tokens` for EAGLE) into every `allocate_slots` call.

> ```python
> self.num_spec_tokens = self.num_lookahead_tokens = 0
> if speculative_config:
>     self.num_spec_tokens = speculative_config.num_speculative_tokens
>     if speculative_config.use_eagle():
>         ...
>         self.num_lookahead_tokens = self.num_spec_tokens
> ...
> new_blocks = self.kv_cache_manager.allocate_slots(
>     request, num_new_tokens, num_lookahead_tokens=self.num_lookahead_tokens)
> ```

Sources: v0 `vllm/core/block_manager.py` (docstring + `can_allocate` signature) v0.6.6; v1
`vllm/v1/core/sched/scheduler.py` v0.10.0.

Transfer to memra: **direct prior art for the spec-shrink reserve.** vLLM charges per-step
lookahead KV (K draft tokens per scheduling step); memra charges a per-session one-time reserve
for the spec-capable arena delta. Different quantities, same principle: the admission/allocation
gate must be spec-aware, and the reserve is scoped to spec-capable requests only (in vLLM it is
zero when `speculative_config is None` — exactly matching memra's "plain path untolled").

### 1.5 Reclaimable-cache accounting in the gate

Claim: v1's `BlockPool` counts prefix-cached-but-unreferenced blocks as FREE. The
`free_block_queue` "stores the free blocks in eviction order" and explicitly includes "eviction
candidates when caching is enabled"; `get_num_free_blocks()` returns that queue's length, and
`get_new_blocks()` lazily evicts a cached hash when it pops such a block
(`_maybe_evict_cached_block`).

> "Free block queue that constructs and manipulates a doubly linked list of free blocks
> (including eviction candidates when caching is enabled)."

Source: `vllm/v1/core/block_pool.py`, tag v0.10.0 —
https://github.com/vllm-project/vllm/blob/v0.10.0/vllm/v1/core/block_pool.py

Transfer to memra: this is the same fix as memra's "gate reads free + pool_cached" — vLLM
solved it structurally (one queue where cached-reclaimable IS free, evicted lazily on pop)
rather than arithmetically (summing two counters). The structural version can't regress by
forgetting a counter; worth considering when memra's pinned pool gets a second consumer.

### 1.6 Known vLLM failure modes (issue receipts)

- **#27951 [RFC]: Fixing the inaccurate memory profiling** (2025-11-03, open): "vLLM's Memory
  profiling is becoming more and more inaccurate... Imbalanced MoE layers means a popular
  expert can directly cause OOM. And CUDA graphs require a pretty significant pool of reserved
  memory. In practice ... people adjust gpu memory utilization to make the model not OOM.
  However, the original meaning of GPU memory utilization is like the following: Say we set
  the utilization to 70%. vLLM's memory usage should never exceed 70%... But now, we set the
  utilization to 70% to use the other 30% for uncaptured CUDA Graph/Activation memory. This is
  a confusing and wrong interface." — https://github.com/vllm-project/vllm/issues/27951
  (Direct analogue of memra's capture-arena transient finding: the profiling-time snapshot
  misses runtime transients — CUDA-graph capture and activation spikes — so the "budget"
  interface silently becomes a fudge factor.)
- **#25538 [Bug]: performance regression caused by frequently preempting and resuming a
  request** (2025-09-24): a request whose prefill exceeds `max_num_batched_tokens` can be
  resumed and preempted again next step — preemption livelock/thrash shape. Title + linked
  scheduler line verified; body not readable via fetch (GitHub JS) — reproduction details
  UNVERIFIED. https://github.com/vllm-project/vllm/issues/25538
- **#5051 [Feature]: Add num_requests_preempted metric** (2024-05-25): "The metrics
  num_requests_running and gpu_cache_usage_perc had to be correlated to understand that the
  requests are getting thrashed." — https://github.com/vllm-project/vllm/issues/5051
  (Observability lesson: preemption thrash was invisible without a dedicated counter.)
- **#15783 [Bug]: vllm fails to calculate gpu_memory and lead to OOM** (2025-03-31): profiling
  measured on the wrong/other GPU state leads to OOM at load. Snippet verified via search;
  full body UNVERIFIED. https://github.com/vllm-project/vllm/issues/15783
- **#26300 [Feature]: Improve vLLM CUDA Memory Utilization and Estimation** (2025-10-06):
  "we do CUDA graph with memory allocation for KV cache, 2) we run the torch.compile. Since we
  may do memory profiling in torch.compile which may consume a lot of memory..." —
  https://github.com/vllm-project/vllm/issues/26300
- **#49674 [Bug]: Deferred KV block frees cause zero-progress preemption cascades with async
  KV consumers** (2026-07): "a single allocation failure can cause many requests to be
  preempted and recomputed even though the scheduler already has the fence state needed to know
  that those frees are deferred." Snippet from search; body UNVERIFIED.
  https://github.com/vllm-project/vllm/issues/49674 — (async-freed memory invisible to the
  gate → over-preemption: the mirror image of memra's "retires returned KV to the pinned pool
  invisible to free" bug, found in vLLM too.)

---

## 2. SGLang

### 2.1 Admission: PrefillAdder budgets against available + evictable, discounted future decode

Claim: SGLang admits prefills against `rem_total_tokens = available_size() + evictable_size()
- rem_total_token_offset` — i.e. the gate **counts radix-cache evictable memory as available**
(cached-but-reclaimable in the admission arithmetic), and pre-charges every running and
incoming request's *estimated* future decode length (`max_new_tokens` clipped to 4096, scaled
by `new_token_ratio`). Radix nodes in active use are lock-protected (`inc_lock_ref`) so
evictable never includes in-flight prefixes.

> ```python
> @property
> def rem_total_tokens(self):
>     return (
>         self.token_to_kv_pool_allocator.available_size()
>         + self.tree_cache.evictable_size()
>         - self.rem_total_token_offset
>     )
> ```

Source: `python/sglang/srt/managers/schedule_policy.py` (`PrefillAdder`), tag v0.4.9 —
https://github.com/sgl-project/sglang/blob/v0.4.9/python/sglang/srt/managers/schedule_policy.py

The `new_token_ratio` starts at 0.7 (`SGLANG_INIT_NEW_TOKEN_RATIO`), decays toward
`0.7 * 0.14` over 600 steps, and is **bumped back up whenever a retraction happens** (feedback
loop: over-admission → retraction → become conservative again). Constants:
`python/sglang/srt/global_config.py` (v0.4.9): `default_init_new_token_ratio = 0.7`,
`default_min_new_token_ratio_factor = 0.14`, `default_new_token_ratio_decay_steps = 600`,
`retract_decode_steps = 20`.

Transfer to memra: SGLang's gate is *probabilistic* (discounted max_new_tokens with an adaptive
ratio) where memra's is *worst-case-with-measured-constants*. The adaptive-ratio feedback
(each retraction returns the estimator to conservative) is an elegant self-tuning admission
throttle memra could adopt if the fixed SPEC_SHRINK_RESERVE ever proves too conservative at
high c: charge less than worst case, and let park events push the charge back up.

### 2.2 Retraction: the preemption mechanism (`retract_decode`)

Claim: before each decode step, `check_decode_mem` requires one page per active request
(times a spec buf multiplier); if short, it first **evicts the radix tree**
(`self.tree_cache.evict(tokens_required)`), and only if still short the scheduler calls
`retract_decode`: victims are chosen with most-output-generated/shortest-input first (i.e.
kill the requests that wasted the most would-be-recomputed work last — sort is
`(len(output_ids) DESC, len(origin_input_ids) ASC)`), their uncached KV is freed, cached
prefixes are unlocked back into the evictable pool, and the request goes back to the waiting
queue to be **re-prefixed/re-prefilled** (RECOMPUTE-style; the radix cache makes the recompute
partially free). Retraction reserves headroom of `retract_decode_steps` (20) tokens per
surviving request, plus an explicit **spec-decode headroom term**:

> ```python
> def get_required_tokens(num_reqs: int):
>     headroom_for_spec_decode = 0
>     if server_args.speculative_algorithm:
>         headroom_for_spec_decode += (
>             num_reqs * server_args.speculative_eagle_topk
>               * server_args.speculative_num_steps
>             + num_reqs * server_args.speculative_num_draft_tokens
>         )
>     return num_reqs * global_config.retract_decode_steps + headroom_for_spec_decode
> ```

and the scheduler logs:

> ```python
> logger.info(
>     "KV cache pool is full. Retract requests. "
>     f"#retracted_reqs: {len(retracted_reqs)}, "
>     f"#new_token_ratio: {old_ratio:.4f} -> {self.new_token_ratio:.4f}")
> ```

Sources: `python/sglang/srt/managers/schedule_batch.py` (`check_decode_mem`,
`retract_decode`), `python/sglang/srt/managers/scheduler.py` (`update_running_batch` retract
call site), tag v0.4.9 —
https://github.com/sgl-project/sglang/blob/v0.4.9/python/sglang/srt/managers/schedule_batch.py

Note the corner-case guard: with one request left it `assert available_size() > 0, "No space
left for only one request"` — SGLang crashes rather than deadlocks when even bs=1 can't fit
(same failure class as memra's c=64 red, handled by assert not by admission).

Transfer to memra: (a) the spec headroom inside the retraction target is the second engine
(after vLLM lookahead) that charges spec growth explicitly — but SGLang charges it at the
*retract/keep* decision, not at admission; memra charging it at admission is earlier and
stricter. (b) Eviction ladder "radix-evict first, retract second" mirrors memra's
"evict parked before deferring/parking active" — the shared principle is *reclaim passive
state before punishing active work*. (c) The victim-sort (retract the request that has
generated the most, since re-prefilling it recovers the most VRAM per victim... while wasting
the most work) is a policy memra should NOT copy blindly — with parked-session
instant-resume as the product promise, memra's inverse ordering (evict longest-idle parked
first) is right.

### 2.3 Failure receipts: retraction and pool accounting bugs

- **#4602 [Bug] sglang decode out of memory** (2025-03-20): "The SGLang server crashes with
  the error 'Decode out of memory' when the --page-size parameter is not set to 1. As shown in
  the log below, there is sufficient space (2048 tokens) available" — page-granularity vs
  token-granularity accounting mismatch in the gate.
  https://github.com/sgl-project/sglang/issues/4602
- **#11581 [Bug] GLM4.5-air-fp8 token_to_kv_pool_allocator memory leak detected** (2025-10-14):
  "Decode out of memory. Try to lower your batch size. Try to allocate 4 tokens. Available
  tokens: 3 (available_size=3 + evictable_size=0) ... Scheduler hit an exception" — pool leak
  drains the token pool until even 4-token decode fails; the "available_size + evictable_size"
  printout in the error is the gate's own arithmetic.
  https://github.com/sgl-project/sglang/issues/11581
- **#14972 [Bug] Overlap schedule + retract cause alloc fail** (2025-12-12): "Available
  tokens: 64 (available_size=64 + evictable_size=0) ... Decode out of memory ... Try to
  allocate 6 tokens" — retraction racing the overlap scheduler still allocates for retracted
  requests. https://github.com/sgl-project/sglang/issues/14972
- **#6857 [Bug] PD Disaggregation benchmark hang after Decode out of memory** (2025-06-04):
  retract-triggered hang in PD mode. https://github.com/sgl-project/sglang/issues/6857
  (Titles/snippets verified via search; full bodies UNVERIFIED — GitHub issue bodies render
  via JS.)

Transfer to memra: SGLang's OOM error line prints the gate's own accounting
(`available_size=X + evictable_size=Y`) at failure time — cheap and directly copyable: memra's
admission-refusal/park log should always carry `free=X pool_cached=Y reserve_charged=Z` so
every field report is self-diagnosing.

### 2.4 HiCache: host-RAM parking of KV with explicit budgets

Claim: SGLang's hierarchical cache treats GPU as L1, **host memory as L2** with an explicit
per-rank budget, and storage as L3. Host budget knobs: `--hicache-ratio` ("ratio of the size
of host KV cache memory pool to the size of device pool... must be greater than 1") and
`--hicache-size` ("size of host KV cache memory pool in gigabytes... for each rank"; overrides
ratio). Write-back policies `write_through | write_through_selective | write_back`; CPU→GPU
promotion overlaps transfer with compute per layer and has dedicated I/O kernels ("up to 3x
higher transfer speed"). The admission path is HiCache-aware: `PrefillAdder.add_one_req`
subtracts `req.host_hit_length` from the charged input tokens and calls
`tree_cache.init_load_back(...)` to promote host-cached prefix before prefill (verified in
schedule_policy.py v0.4.9).

> "`--hicache-size HICACHE_SIZE`: The size of host KV cache memory pool in gigabytes. ...
> allocates 30GB ... for the host memory pool **for each rank**."

Source: https://docs.sglang.io/advanced_features/hicache_design.html ; blog
https://www.lmsys.org/blog/2025-09-10-sglang-hicache/

Also: in PD-disaggregation decode mode, `retract_decode` calls `req.offload_kv_cache(...)`
before freeing — retracted sessions are parked to host rather than dropped (verified in
schedule_batch.py v0.4.9, `if server_args.disaggregation_mode == "decode":`).

Transfer to memra: the strongest existing "park to host RAM with a budget" design. If memra's
parked sessions ever exceed what the spec-shrink-reserve admission can hold resident, HiCache's
shape is the reference: explicit host budget, layer-overlapped promotion, and admission that
*discounts* host-cached prefix instead of treating it as a miss.

---

## 3. TGI (HuggingFace text-generation-inference)

### 3.1 Admission: token-budget validation, semaphore, warmup-measured VRAM budget

Claim: TGI does **static admission**, no preemption at all. Three nested gates:

1. Concurrency semaphore: `--max-concurrent-requests` (default 128) — "Having a low limit will
   refuse clients requests instead of having them wait for too long and is usually good to
   handle backpressure correctly." Over the limit → immediate `InferError::Overloaded`
   ("Model is overloaded", counted as `tgi_request_failure err=overloaded`); implemented as
   `Semaphore::try_acquire_owned()` — verified in `router/src/infer/mod.rs` v3.3.4.
2. Per-request validation: `--max-input-tokens`, `--max-total-tokens` ("the most important
   value to set as it defines the 'memory budget' of running clients requests").
3. Batch formation: the queue's `next_batch` walks entries and for each does a real block
   allocation against `max_batch_total_tokens`; failure pushes the entry back to the queue
   front and stops batching:

> ```rust
> let tokens = entry.request.input_length
>     + entry.request.stopping_parameters.max_new_tokens
>     + self.speculate - 1;
> ...
> let block_allocation = match block_allocator.allocate(tokens, input_ids).await {
>     None => {
>         // Entry is over budget
>         // Add it back to the front
>         tracing::debug!("Over budget: not enough free blocks");
>         self.entries.push_front((id, entry));
>         break 'entry_loop;
>     }
> ```

Source: `backends/v3/src/queue.rs`, tag v3.3.4 —
https://github.com/huggingface/text-generation-inference/blob/v3.3.4/backends/v3/src/queue.rs

Note `+ self.speculate` in the charged tokens: **TGI charges the speculation depth
(`--speculate`, Medusa/n-gram draft length) into every request's admission-time token budget.**
Third engine confirming spec-aware admission (here: worst-case, per-request, at admission —
the closest shape to memra's reserve).

The VRAM budget itself is measured, not guessed: at startup the router warms up the shards and
"Flash attention models return their max supported total tokens", overriding any user value —
"`--max-batch-total-tokens` is deprecated for Flash Attention models. ... Inferred max batch
total tokens: {max_supported_batch_total_tokens}" (`backends/v3/src/lib.rs`, v3.3.4,
`connect_backend`).

Launcher docs (all flags): https://huggingface.co/docs/text-generation-inference/basic_tutorials/launcher

### 3.2 waiting_served_ratio: batch-join pressure, not eviction

Claim: `--waiting-served-ratio` (default 0.3) governs when the running batch is *paused for one
prefill pass* to merge waiters in — the running requests are delayed, never evicted:

> "This represents the ratio of waiting queries vs running queries where you want to start
> considering pausing the running queries to include the waiting ones into the same batch.
> `waiting_served_ratio=1.2` Means when 12 queries are waiting and there's only 10 queries left
> in the current batch we check if we can fit those 12 waiting queries into the batching
> strategy, and if yes, then batching happens delaying the 10 running queries by a `prefill`
> run."

plus `--max-waiting-tokens` (default 20) forcing a merge after N generated tokens.
Source: launcher docs (above).

### 3.3 Session retirement: none — TGI holds no cross-request state

Claim: a finished request frees its blocks; there is no parked/session concept, no swap, no
retraction. Requests that would exceed the budget wait in the queue (or are rejected at the
semaphore). Radix-trie prefix caching (`backends/v3/src/radix.rs`) is the only cross-request
KV retention and it lives inside the same block allocator. OOM avoidance is entirely by
worst-case admission (input + max_new_tokens + speculate charged up front) against a
warmup-measured capacity.

Transfer to memra: TGI is the "pure admission, zero preemption" endpoint of the design space —
it proves you can serve safely with worst-case charging IF the budget is measured by a real
warmup pass on the exact config. Its cost is utilization (worst-case charging strands VRAM
that decode never uses). memra sits between TGI (worst-case charge, no reclaim) and
vLLM/SGLang (optimistic admit, forcible reclaim): parked sessions give memra a reclaimable
buffer TGI lacks, while the spec reserve keeps the TGI-style hard safety property.

---

## 4. llama.cpp server

### 4.1 Slot model: hard cap, per-slot context split

Claim: concurrency = fixed slots: `-np, --parallel N` "number of server slots (default: -1,
-1 = auto)". Each slot gets `n_ctx_slot = llama_n_ctx_seq(ctx)` — the context is divided
across slots at startup (unless `--kv-unified` pools it: "use single unified KV buffer shared
across all sequences (default: enabled if number of slots is auto)"). Admission is trivially
the slot count; no token-level gate at all.

> ```cpp
> SRV_INF("initializing, n_slots = %d, n_ctx_slot = %d, kv_unified = '%s'\n",
>         params_base.n_parallel, n_ctx_slot, params_base.kv_unified ? "true" : "false");
> ```

Sources: `tools/server/README.md` (flag table) and `tools/server/server-context.cpp` (master,
fetched 2026-08-09) — https://github.com/ggml-org/llama.cpp/tree/master/tools/server

### 4.2 Slot selection = parked-session reuse: LCP similarity, then LRU eviction

Claim: llama.cpp *is* a parked-session engine — an idle slot keeps its full KV, and
`get_available_slot()` routes a new request to the idle slot whose cached prompt shares the
longest common prefix (threshold `--slot-prompt-similarity`, default 0.10), so the next turn
of a conversation resumes on its own KV without re-prefill. Only if no similar slot exists does
it take the least-recently-used idle slot and overwrite (evict) its state — saving the evicted
prompt into a host-RAM prompt cache first when a large portion would be lost:

> ```cpp
> // find the slot that has at least n% prompt similarity
> ...
> const size_t lcp_len = tokens.get_common_prefix(task.tokens);
> const float f_sim_cur = float(lcp_len) / task.tokens.size();
> ...
> // if we are about to lose a large portion of the existing context - save it in the prompt cache
> if (f_keep < 0.5f) { update_cache = true; }
> ...
> // find the slot that has been least recently used
> ...
> SLT_INF(*ret, "selected slot by LRU, t_last = %" PRId64 "\n", t_last);
> ```

When all slots are busy the task is **deferred** (queued), never preempted:

> ```cpp
> if (slot == nullptr) {
>     // if no slot is available, we defer this task for processing later
>     SRV_DBG("no slot is available, defer task, id_task = %d\n", id_task);
>     queue_tasks.defer(std::move(task));
> ```

(exposed as the `llamacpp:requests_deferred` gauge). Source: `server-context.cpp`
`get_available_slot()` / main task loop; README metrics table.

### 4.3 Host-RAM prompt cache + idle-slot clearing (parking to RAM with a budget)

Claim: `-cram, --cache-ram N` "set the maximum cache size in MiB (default: 8192, -1 - no
limit, 0 - disable)" (PR #16391 "server : host-memory prompt caching", ggerganov) gives
evicted/idle slot states a budgeted host-RAM home; `--cache-idle-slots` "save idle slots to
the prompt cache on new task, and clear them when using unified KV (default: enabled, requires
cache-ram)". With unified KV, `try_clear_idle_slots()` purges idle slots' VRAM one at a time
under pressure; without unified KV the code explicitly documents that clearing frees no
reusable room so it only publishes the RAM copy:

> ```cpp
> // without a unified KV cache, clearing a slot frees no reusable room, so we only
> // publish a RAM-cache copy of idle slots (their KV stays in VRAM) [TAG_IDLE_SLOT_CLEAR]
> ```

and the roadmap comment on `try_clear_idle_slots` reads:

> ```cpp
> // TODO: improve logic
> //       - smarter decision which slot to clear (LRU or longest prompt?)
> //       - move slot to level 2 cache instead of removing?
> //       - instead of purging, try to store and resume later?
> ```

Source: `tools/server/server-context.cpp` (master); PR
https://github.com/ggml-org/llama.cpp/pull/16391

### 4.4 Explicit parking to disk: `/slots/{id}?action=save|restore|erase`

Claim: the server exposes REST endpoints that serialize a slot's full KV state to a file and
restore it later — parking to disk as a *user-driven* API (requires `--slot-save-path`):

> "### POST `/slots/{id_slot}?action=save`: Save the prompt cache of the specified slot to a
> file. ... `filename`: Name of the file to save the slot's prompt cache. ... response:
> `{"id_slot": 0, "filename": "slot_save_file.bin", "n_saved": 1745, ...}`"
> and the matching `?action=restore` / `?action=erase`.

Source: `tools/server/README.md` (master) —
https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md

Transfer to memra: llama.cpp is the only surveyed engine whose *native* session model matches
memra's parked sessions (idle slot = parked KV, similarity-routed resume = instant next turn).
Its whole reclaim ladder — (1) reuse by prefix similarity, (2) save-to-RAM-cache before
overwrite, (3) purge idle slots under unified-KV pressure, (4) explicit save-to-disk API — is
the closest prior art to memra's park/evict/defer design. What it lacks is any *memory-charged
admission*: slots are count-capped, not byte-capped, and the TODO comment shows tiered
store-and-resume is aspiration, not implementation. memra's admission gate + measured reserve
is ahead of llama.cpp here; llama.cpp's LCP-similarity slot routing and budgeted host prompt
cache are ahead of memra.

---

## 5. Cross-cutting answers

**Does anyone charge a reserve for speculative-decoding growth at admission?**
Yes — all three token-accounting engines, at different points:
- vLLM: lookahead slots/tokens in `can_allocate` / `allocate_slots` (per-step draft KV).
- TGI: `+ self.speculate` inside the queue's admission token math (worst-case, per request —
  closest to memra's shape).
- SGLang: spec headroom (`eagle_topk * num_steps + num_draft_tokens` per request) — but inside
  the *retraction* target, not admission.
None charges a *model-arena* shrink/capture reserve like memra's SPEC_SHRINK_RESERVE — the
others' spec reserves are KV-token quantities; memra's covers allocator/arena effects (live-vs-
parked 1.49x + capture transient) that block-pool engines don't have because their pools are
pre-carved at startup.

**Does anyone count cached-but-reclaimable (pool) memory in the admission gate?**
Yes, two:
- SGLang, arithmetically: `available_size() + evictable_size()` is the gate's supply term
  (and the OOM message prints both).
- vLLM v1, structurally: prefix-cached unreferenced blocks live in the free queue itself.
memra's `free + pool_cached` fix independently rediscovered the SGLang form.

**Does anyone park full contexts on host RAM with explicit budgets?**
Yes, three mechanisms:
- vLLM: v0 `--swap-space` (GiB per GPU, default 4) for preempted-sequence blocks; v1
  `OffloadingConnector` `cpu_bytes_to_use` (required explicit byte budget, lru/arc eviction) as
  an async prefix-cache tier.
- SGLang HiCache: `--hicache-ratio` / `--hicache-size` (GB per rank) host pool, write-through/
  write-back policies, admission-integrated promotion (`host_hit_length` discount); PD-decode
  retraction offloads KV to host before freeing.
- llama.cpp: `--cache-ram` MiB budget host prompt cache + `--cache-idle-slots` + explicit
  `/slots?action=save` disk parking.

---

## Strongest transferable mechanisms (ranked for memra)

1. **Self-diagnosing pressure logs** (vLLM's rate-limited cumulative-preemption warning;
   SGLang's `available_size=X + evictable_size=Y` in the OOM line). Cheapest win: every memra
   defer/park/refuse event should print the gate's full arithmetic and a cumulative counter.
2. **Three-way admission verdict** (vLLM `OK/LATER/NEVER`): add a "can never fit at this
   config" fast-reject distinct from "defer", so an oversized request errors immediately
   instead of starving in the defer queue.
3. **Adaptive admission discount with retraction feedback** (SGLang `new_token_ratio`): if the
   fixed spec reserve costs too much concurrency, charge a decaying fraction and let park
   events reset it to conservative — a control loop instead of a constant.
4. **Budgeted host tier for parked sessions** (SGLang HiCache shape: explicit per-rank byte
   budget, async write-behind while parked-resident, layer-overlapped promotion on resume;
   llama.cpp `--cache-ram` as the minimal version). This converts memra's park/evict binary
   into park-resident / park-host / evict — eviction stops costing a full re-prefill.
5. **Prefix-similarity slot routing** (llama.cpp LCP threshold): when a parked session must be
   evicted, prefer evicting one whose prefix overlaps the incoming request least; and on
   admission, a partially-matching parked session is worth resuming even for a "new" request.
6. **Warmup-measured capacity** (TGI): derive the admission budget from a real max-shape
   forward pass on the exact config at startup rather than static arithmetic — this is what
   memra's admission constants (292 vs 286 MiB/session cross-check) already approximate;
   making it a boot-time self-measurement removes constant drift.

## What memra's parked-session design has that these lack

- **Instant resume with zero re-prefill and zero transfer**: parked KV stays in VRAM. vLLM v1
  preemption recomputes (prefix cache softens, doesn't eliminate); SGLang retraction
  re-prefills; HiCache/OffloadingConnector resume costs a host→GPU copy; llama.cpp idle slots
  are the only equivalent — and llama.cpp will silently overwrite a parked slot on LRU
  collision with no admission-level accounting of what parking costs.
- **Parked state charged at the admission gate**: no surveyed engine accounts "sessions we are
  keeping warm" as a first-class admission quantity — vLLM/SGLang account only running
  requests + anonymous cache blocks; llama.cpp accounts nothing but slot count. memra's gate
  seeing parked sessions as evictable-but-charged is genuinely novel among these four.
- **Arena-level spec reserve**: others reserve draft-token KV; none models the allocator-level
  live-vs-parked footprint delta (1.49x) or capture-arena transient, because their block pools
  are pre-carved. Any engine that (like memra) allocates KV dynamically from a shared arena
  needs memra's kind of reserve; there is no prior art to copy for it.
- **Quoted-failure evidence discipline**: vLLM #27951/#25538 and SGLang #11581/#14972 are all
  post-hoc discoveries of gate/accounting drift in the field; memra's forced-tiny-reserve
  teeth test (verdict inversion 11/64) is a stronger standing gate than anything the surveyed
  engines run in CI.

## Open uncertainties

- vLLM issue bodies for #25538, #15783, #49674 and SGLang issue bodies (all four cited) were
  not readable via fetch (GitHub renders via JS); titles + search snippets verified, full
  reproduction details UNVERIFIED.
- vLLM v1: whether any *residual* swap path exists behind flags in the newest tree (checked
  v0.10.0 scheduler: none) — later releases add KV-connector-based "swap-like" flows
  (#29870); the exact current interaction of OffloadingConnector with preemption ordering is
  UNVERIFIED beyond the PR title.
- SGLang: whether `retract_decode`'s host offload (`offload_kv_cache`) is used outside
  PD-decode disaggregation mode in newer versions — in v0.4.9 it is decode-disagg-only.
- llama.cpp: `try_clear_idle_slots` semantics verified on master (2026-08-09); the
  unified-KV default flipped recently (`default: enabled if number of slots is auto`) and slot
  behavior around it is still moving — re-check before citing in a design doc.
- TGI v3 vs older v2 backend differ; all claims here are from the v3 backend (`backends/v3`,
  tag v3.3.4). TGI maintenance status (HF has deprioritized TGI in favor of other stacks) not
  assessed.
