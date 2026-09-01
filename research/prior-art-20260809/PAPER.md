# Prior art in other engines for memra's hardening problems

Survey date: 2026-08-09. Raw findings with verbatim quotes and URLs:
`notes-q1a-vllm-trt-lmcache-mooncake.md`, `notes-q1b-sglang-llamacpp.md`,
`notes-q2-swa-depth-degradation.md`, `notes-q3-admission-preemption.md`,
`notes-q4-batched-specdec.md` (this directory). memra-side grounding: `NOTES.md`.
Every external claim below traces to a quote and URL in a notes file; the notes files
mark unverifiable claims UNVERIFIED, and this paper excludes or flags them.

## Executive summary: ranked transferable mechanisms

The survey covered six engines (vLLM, SGLang, llama.cpp, TensorRT-LLM, LMCache, Mooncake)
plus HuggingFace transformers and TGI across memra's four live hardening problems. Four
headline results:

1. **llama.cpp independently converged on memra's in-flight prefix-resume design.** A
   four-PR arc (#15293 -> #20288 -> #22929 -> #24176, merged 2026-06-23) arrived at
   checkpoints at every user-message boundary, selected at restore time as the last
   checkpoint at-or-before the divergence point, with invalidated checkpoints erased and
   the draft/spec state restored alongside the KV. The same Claude-Code-style
   rewritten-history workload memra measured forced each iteration. This is the strongest
   available validation of prompt-end checkpoints + bytes-decide resume, and it supplies
   the two lessons memra has not yet paid for: tail-only checkpoints die on mid-history
   edits, and the checkpoint must carry speculative-decoder state.
2. **"Identity nominates, bytes decide" is the industry's converged correctness posture,
   learned through a CVE.** vLLM's builtin-hash prefix cache produced predictable
   collisions an attacker could exploit (GHSA-rm76-4mrf-v9r8); the fix chain ended at
   SHA-256 defaults. SGLang never trusted hashes on the hot path (token-byte compares,
   with `extra_key` namespaces). llama.cpp's bytes-only matching without full identity
   produced silent LoRA cross-contamination (#26207). The stable design: full identity
   (session, model, adapter, KV dtype, template flags) narrows the candidate set; exact
   byte comparison decides reuse length.
3. **Every engine that shipped prefix reuse without per-request reuse telemetry paid in
   undiagnosable bugs.** SGLang #22373 (65-80 GB dead KV) and LMCache #1812/#1938
   (net-loss deployments) were diagnosed through `cached_tokens`-class counters; vLLM's
   tracker carries a standing class of silent zero-hit multi-turn reports (#4917,
   #31920, cited for the failure class; root causes unfetched). memra already ships
   this surface (`/metrics` lcp_histogram, cached_tokens); the survey confirms it is
   load-bearing.
4. **Depth degradation into token soup is an engine-bug signature, never documented as
   a model property.** The verbatim cross-lingual-soup receipts trace to engine defects
   (stale CUDA graphs at long context, llama.cpp #21640; quantized-K corruption,
   #8853), and the adjacent depth-corruption receipts share the mechanism that predicts
   soup: corrupted or NaN logits sampling a multilingual vocab (half-precision
   accumulator saturation, #23577; recycled uninitialized SWA blocks, vLLM #47574).
   Section 2.4 ranks eight diagnoses
   for memra's 9k bug, led by MTP rollback crossing the SWA ring's write frontier, the
   operation llama.cpp's cache architecture declares structurally impossible on a
   pruned rolling window.

Ranked mechanisms worth adopting (details and citations in the per-problem sections):

| rank | mechanism | source | memra fit |
|---|---|---|---|
| 1 | Checkpoints at EVERY user boundary, not just prompt end; restore = last checkpoint at-or-before divergence, erase later ones | llama.cpp #24176 | Direct upgrade to the in-flight design |
| 2 | Checkpoint carries spec/draft state | llama.cpp `data_spec`; SGLang #32459 (97%->40-53% reuse collapse without it) | memra's spec tier currently BYPASSES the prefix cache; this is the fix shape |
| 3 | Free != evicted: ref-count + lazy eviction; parked entries stay indexed until allocation pressure | vLLM free queue; TRT-LLM radix tree; SGLang lock_ref two-bucket accounting | Extends the admission gate's `free + pool_cached` model to cache entries |
| 4 | Retention priority over plain LRU for agent tool-call pauses | vLLM RFC #37003; TRT-LLM priority 0-100 + TTL | Two or three classes suffice: system-prompt, live-session tails, rest |
| 5 | Host tier as a save-before-recycle deque with two-ratio acceptance (f_sim/f_keep, 0.25 floor) and dual byte+token budgets | llama.cpp #16391 `--cache-ram` | The minimal single-node host tier; far less machinery than HiCache |
| 6 | Contiguous all-layer slab per checkpoint (storage unit == transfer unit == one DMA) | vLLM PR #27743 (~10x offload throughput); Mooncake #467 as the negative | memra owns its layout; bake in from day one |
| 7 | Prefix-aware admission: resolve match length BEFORE scheduling, charge only the uncached suffix | Mooncake Conductor; TRT-LLM prefix-aware scheduling | Plugs into the existing VRAM admission gate |
| 8 | Boundary-exact resume test battery (match length == boundary, +/-1, under concurrent first-touch) | SGLang #22819 (10/10 KV corruption at prefix_len == block_size); llama.cpp #12253 | One new gate arm, cheap |
| 9 | Galloped divergence search (doubling windows + binary search) | SGLang RadixKey.match | O(log n) compares on 100k-token resent histories |
| 10 | swa-full debug lever + ring slack >= draft depth + fp32-accumulator audit for the depth bug | llama.cpp #13194/#13833/#22660/#23577; vLLM #40898 | Ranked diagnosis list for the 9k token-soup bug in section 2.4 |

What memra already does that the surveyed engines lack: bit-identity gates on every reuse
tier (vLLM's ROCm #33123 shows hash-correct-but-numerically-divergent reuse shipping);
quoted-failure discipline; and a parked-session resume that restores GDN recurrent state,
which no surveyed engine handles (llama.cpp #21831 shows hybrid/recurrent models forcing
full re-prime every turn until checkpoints landed).

---

## 1. Growing-conversation prefix reuse under rewritten histories

memra's baseline: three tiers (continuation pool, cross-request prefix cache, session
affinity) with the affinity tier already measuring 20-24x TTFT on rewritten turns
(`research/session-affinity-20260805/RESULTS.md`). The question here is what the mature
engines do that the hardening pass should add. Full citations:
`notes-q1a-vllm-trt-lmcache-mooncake.md`, `notes-q1b-sglang-llamacpp.md`.

### 1.1 Entry granularity and hash schemes

| engine | granularity | match scheme |
|---|---|---|
| vLLM v1 APC (Automatic Prefix Caching) | 16-token blocks, full blocks only (fine-grained interior probe added later) | chained per-block hash: `hash(parent_hash, block_tokens, extra_keys)`; SHA-256 default since v0.11 |
| SGLang | token (page rounding optional) | radix tree, direct token-array compare with galloped divergence search; SHA-256 page hashes exist only for KV events/storage tiers |
| llama.cpp | token, per-slot linear history | plain longest-common-prefix loop (`get_common_prefix`), then trim (`seq_rm`) |
| TensorRT-LLM | 128-token blocks (default), partial reuse of the last block | radix tree over blocks; copy-on-partial-reuse |
| LMCache | 256-token chunks | chained prefix hash truncated to 64 bits; key adds model/world-size/dtype |
| Mooncake | 512-token chunks | chained prefix-chunk hashes remapped to global ids |

Two design consequences. First, the chained hash makes mid-history edits
self-invalidating: a miss at block k implies misses at all k+, so no explicit
invalidation logic exists anywhere in vLLM (`find_longest_cache_hit` just breaks at the
first missing hash). Second, granularity trades edit tolerance against transfer
efficiency: NVIDIA documents that a larger block size "reduces the likelihood of kv
cache state reuse," while LMCache's 256 and Mooncake's 512 target host/remote transfer
amortization. An edit-heavy agent workload wants small units near the live tail and large
units for the frozen head; no surveyed engine implements that two-scale split, and
memra's checkpoint design can.

memra's affinity tier operates at a coarser unit than any of these: one checkpoint per
turn boundary, byte-compared. That is the right unit for the rewritten-history case
(llama.cpp's arc, section 1.3), and section 1.4's eviction evidence says the entry count
stays small enough that per-entry byte comparison beats block-hash machinery at memra's
concurrency.

### 1.2 What broke: the correctness bug families

Three recurring families, each with a direct memra defense:

- **Hash-trust bugs.** vLLM's builtin `hash()` collisions became CVE-2025-25183
  (GHSA-rm76-4mrf-v9r8): "Maliciously constructed prompts can lead to hash collisions,
  resulting in prefix cache reuse, which can interfere with subsequent responses."
  LMCache still truncates digests to 64 bits and defaults to the builtin hash. memra's
  bytes-decide rule closes this class by construction; keep it even if a hash index is
  added for nomination speed.
- **Identity-omission bugs.** Anything that conditions KV but is outside the token stream
  must be in the key: vLLM keyed LoRA blocks on `lora_name` not the adapter id (#30931),
  ignored visual inputs under concurrency (#20261); llama.cpp reused KV across different
  per-request LoRA adapters silently (#26207); TRT-LLM pushes p-tuning disambiguation
  onto the client via `extra_ids`. LMCache's own code comments "Ignore extra keys for
  now." memra's key already carries (model, cache_salt); the hardening pass should
  enumerate the rest: quant revision, template/BOS handling, RoPE config, KV dtype.
- **Numerics-of-reuse bugs.** vLLM on ROCm produced different outputs for cache-hit vs
  cold-prefill paths of the same prompt (#33123) because reuse changes which kernel
  computes the prefix. TRT-LLM's FP8-KV block reuse produced garbage while FP16 was fine
  (#2699). memra's cached-vs-fresh bit-identity gate
  (`research/prompt-cache-20260802/gate-exact.jsonl`) is exactly the defense; the
  transferable addition is a quantized-KV-reuse arm if memra ever reuses non-FP16 KV, and
  the tick-seg lane's extent-class residual is memra's own instance of this family.

### 1.3 The rewritten-history mechanism: llama.cpp's convergence

llama.cpp reached memra's design through four PRs, each forced by an agent-workload
failure (all quotes in notes-q1b section 2.5):

1. #15293: SWA checkpoints introduced, because SWA models cannot trim-and-continue.
2. #20288: two checkpoints near prompt end, "necessary in order to allow mutating the
   last user message."
3. #22929: checkpoint at the last user message, parsed from chat-template
   `message_spans`; author's stated goal was "the responsiveness of agentic coding."
4. #24176: checkpoints at EVERY user boundary, because "prompts with a stable prefix and
   content that changes between turns lose all checkpoint cache hits (the surviving
   checkpoints sit past the divergence point)."

Restore selects the last checkpoint at-or-before the divergence, erases checkpoints past
it, and restores target KV, draft KV, and speculative state (`data_spec`) together.
llama.cpp has no session identity at all; slot nomination is longest-common-prefix
(LCP) similarity routing
(`--slot-prompt-similarity`, default 0.10) plus a save-before-recycle rule: the server
spills a slot to the host cache before overwriting more than half its context. memra's explicit
session-id nomination is strictly cheaper (O(1) candidate lookup vs per-slot LCP scans)
with the same bytes-decide arbiter.

SGLang, by contrast, has no answer for mid-prefix edits beyond keep-before-divergence.
Its radix tree retains the pre-edit branch, which helps flip-flopping clients but
otherwise strands dead KV: issue #22373 quantifies 65-80 GB of unreachable branches when
clients strip thinking tokens, and states the general law: "what gets cached and what
gets sent in future requests are decided independently." The lesson for memra: checkpoint
the prompt-visible byte stream (what the client resends), and meter dead-entry mass.

Client-side head-of-prompt churn defeats any engine-side scheme: a field report traced
llama.cpp cache misses to Claude Code's attribution header alone
(`CLAUDE_CODE_ATTRIBUTION_HEADER=0` fixed it). TTFT telemetry should therefore
distinguish "diverged at token < 100" (client problem) from "diverged mid-history"
(engine's job); memra's decline-logging with offsets already carries this signal.

### 1.4 Eviction and VRAM budgeting

All three GPU-tier designs converge on two ideas. First, free != evicted: vLLM's
ref_cnt=0 blocks sit in a free queue but stay hash-indexed until the memory is actually
reallocated; TRT-LLM's radix entries persist until pressure; SGLang keeps two accounting
buckets (protected = lock_ref-pinned by running requests, evictable = everything else)
over ONE shared pool, so cache yields to sessions automatically. This composes directly
with memra's admit-oom gate: checkpoint bytes belong in the `pool_cached` term of
`free + pool_cached`, reclaimable by admission. Second, depth-aware ordering: vLLM frees
request blocks in reverse so deepest blocks evict first; TRT-LLM "biases toward blocks
further from the root"; SGLang evicts leaf-first so shared ancestors survive. The
checkpoint analog: evict cold sessions' newest checkpoints first, keep the
system-prompt-end checkpoint until last.

Beyond LRU: vLLM's RFC #37003 documents memra's exact workload: "40–60% of session wall
time is spent paused on tool calls, and during those pauses the agent's blocks are
unreferenced. Under concurrent load, other agents evict them via LRU. When the session
resumes: cache miss, full recomputation of the entire context", and proposes
session-scoped retention priorities with TTL. TRT-LLM already ships the productized
version (priorities 0-100, per-token-range retention, ~20% hit-rate gain in NVIDIA's
benchmarks). Counter-evidence: Mooncake's production trace found plain LRU best on chat
traffic ("temporal proximity in request utilization") and hit rate saturating with modest
capacity. The synthesis: LRU is the right default; a small protected class for
live-session tails and the system prompt is the agent-workload insurance; a full priority
API is overkill for a single-node engine.

### 1.5 Host-tier spill and multi-hundred-k contexts

The scaled designs (SGLang HiCache: GPU/host/storage tiers over one HiRadixTree,
layer-wise H2D overlap, page-first host layout; TRT-LLM secondary offloading: evicted
blocks stay in the search tree with a location tag) share one structural idea worth
keeping: ONE index over all tiers, per-entry location, demand-driven onboarding. The
memra-sized implementation is llama.cpp's `--cache-ram` deque: full-state host entries,
FIFO eviction under dual byte+token budgets, acceptance requiring both better f_sim
(fraction of new prompt covered) and better f_keep (fraction of entry surviving, floor
0.25, "don't trash large prompts"), drop entries fully contained in newer ones, shrink
on bad_alloc. The f_keep floor encodes the real economics: restore bandwidth is the
hidden denominator, so resume decisions should weigh restore cost, not just matched
length. LMCache's `min_retrieve_tokens` is the same floor from the other side: below N
matched tokens, cold prefill beats restore.

Transfer performance is a layout property: vLLM's offload connector was ~10x faster after
one contiguous physical block per logical block across all layers (PR #27743, "This
fragmentation is meaningless for model computation performance, but is devastating for KV
offloading"); Mooncake-as-backend measured slower than no cache at all when its 4MB store
page mismatched the read size (#467, a single 32x10k-token test). memra checkpoints
should be stored as one contiguous slab each, sized so H2D restore is a single DMA.

LMCache's negative results are the cautionary gate: two deployments measured net loss
over plain vLLM (#1812: higher latency on a shared-prefix benchmark; #1938: TTFT 2x
worse), because the lookup/copy layer competed with GPU prefill for wins the GPU-side
cache would have gotten anyway. The checkpoint path needs a measured ~zero overhead on
miss-only workloads as a first-class gate number.

### 1.6 Speculative decoding is the open flank

memra's prefix cache currently bypasses spec sessions entirely (SERVING.md: a trunk-only
restore would leave draft state unprimed). The survey shows this bypass is where the
other engines are failing today. SGLang's EAGLE integration collapsed radix reuse from 97% to
40-53% on multi-turn agentic traffic (#32459) and NaN-crashed on partially
radix-populated batches (#19796, reported on SM120/PRO 6000 hardware). vLLM's
`find_longest_cache_hit` carries a `drop_eagle_block` argument that recomputes the last
matched block on a 100% hit purely to recover drafter hidden states. llama.cpp is the
one engine that got it right: checkpoints store `data_spec` and restore it with the KV.
For memra's MTP tier, the checkpoint contract must include draft state (or accept a
last-block recompute to re-arm the drafter). Otherwise the TTFT win silently excludes
every spec session, which is the default serving path.

---

## 2. Long-generation degradation at depth (SWA window=512, ~9k generated tokens)

memra's open bug: ~9k+ generated tokens on an interleaved-SWA model (window=512, MTP
live) degrades into cross-lingual token soup. The survey question: do other engines
document rolling-KV/SWA bugs at depth, and what were the fixes. Answer: every major
engine does, the receipts map cleanly onto memra's symptom, and no engine document
surveyed attributes cross-lingual soup to a MODEL property at depth: every soup receipt
traced to an engine defect. Full citations: `notes-q2-swa-depth-degradation.md`.

### 2.1 The structural law: rollback on a pruned rolling SWA cache is impossible

llama.cpp's interleaved-SWA (iSWA) split-cache PR (#13194) states it as design doctrine: "advanced cache
operations such as removing tokens or shifting their positions are not possible when
using SWA cache, because token information becomes lost when the window slides", with
`--swa-full` (full-size cache, mask-only windowing) as the escape hatch. The
spec-decoding corollary is a convergent industry fix. llama.cpp sizes the SWA ring
`n_swa + n_ubatch` explicitly to permit only SHORT rollbacks (#13833: "Allow short SWA
rollbacks") and auto-forces `swa_full` for SWA draft models (#22660). vLLM landed the
same shape independently (PR #40898: full draft KV allocation for SWA layers "so
target-prewritten context K/V is not evicted by masked draft-block tokens"; also #46032,
#50169). The check for memra: after a rejected MTP draft, is every slot the draft
overwrote still recoverable, and does the ring carry >= draft-depth slack beyond the
window? A wrap during a speculative batch that then rejects is unrecoverable without
that slack; a rare race here matches the "hours fine, then permanent soup" shape.

### 2.2 The five documented depth-failure classes

1. **Window applied to the wrong layers.** llama.cpp #15808/#15811: gpt-oss
   (interleaved SWA) "goes off the rails into unrelated topics" mid-generation because
   "when we mask the attention we should not use `hparams.swa_type` for all layers -
   only for the SWA layers." Exact memra symptom shape and exact model shape.
2. **fp16 accumulator saturation at depth.** llama.cpp #23577 (open): correct for
   hours, then permanent `////` collapse; the in-thread quantitative diagnosis found the
   FA tile keeping softmax max/denominator in F32 while accumulating the V-weighted
   numerator in half2: "the half2 numerator overflows on the ninth contribution and the
   FA output becomes Inf... eventually making all 248,320 target logits NaN. MTP
   acceptance then collapsed to zero." Contribution count grows with context, so
   saturation has a characteristic onset depth; NaN/garbage logits sample near-uniformly
   over a multilingual vocab, producing exactly the token-soup symptom. Corroborated by #23606 (all logits
   NaN at 80k+, title-verified). Note the MTP entanglement: the visible symptom included
   draft-acceptance collapse to zero.
3. **Cache maintenance corrupting state.** llama.cpp #12253: "garbage output (which
   stays until server restart) as soon as KV-cache defragmentation occurs once",
   narrowed to quantized-KV byte math; #14059/#14189 defrag bookkeeping races. vLLM
   #47574: freed sliding-window blocks recycled into an fp8 group and read back
   uninitialized, "NaN / all-zero output once the input exceeds the sliding window";
   #42273: SWA eviction leaves stale physical block ids, "reproducible as real output
   corruption."
4. **Window mask arithmetic off-by-one.** SGLang PR #32087: the Triton decode kernel
   consumed a radius (`config.sliding_window - 1`) as a count, so "on every decode step
   past the window size it silently drops the oldest in-window key." HuggingFace
   transformers has four independent instances of mask/cache wrong exactly at or past
   the boundary (#37574 wrong HybridCache update when length reaches the window; #35681
   "mask slicing was wrong in all cases when the sequence length is larger than the
   sliding window"; #30127 SDPA ignored sliding_window for six months; #32945 Phi-3
   long-prompt gibberish from a dropped mask term). Reference semantics to test
   against: llama.cpp's STANDARD mask is `masked iff p1 - p0 >= n_swa`.
5. **Stale CUDA-graph reuse at depth.** llama.cpp #21640/#21635: the graph-reuse
   equality check missed `node->src->data` pointers, so when KV addresses changed at
   long context the captured graph read stale memory, producing infinite multilingual
   garbage on Gemma4 26B. The reporter's sample is verbatim cross-lingual soup, as is
   #8853 ("Mh giàu され rodas reliablyacheteurδε Są" from quantized-K on Gemma-2).

Two posture datapoints: vLLM and FlashInfer historically REFUSED interleaved SWA rather
than half-support it ("Disabling sliding window and capping the max length", #6220,
#20865), and llama.cpp disabled context shift by default (#15416) after years of
footgun reports. Both engines chose refusal over approximation, matching memra's
exactness doctrine.

### 2.3 Attention sinks: relevant but conditional

StreamingLLM (arXiv 2309.17453): "Window attention... fails when the text length
surpasses the cache size... keeping the KV of initial tokens will largely recover the
performance." The caveat the notes file carries: this applies to models TRAINED with
full attention then windowed at inference. Gemma-style interleaved models train with
the window on SWA layers, so sink eviction inside the trained window only bites if
the engine windows a layer trained full-attention (class 1 above) or the model carries
learned sink parameters the engine must honor (gpt-oss does). A cheap discriminating
probe: pin the first N tokens in the SWA ring and see if quality recovers.

### 2.4 Ranked diagnoses for memra's 9k soup

From the notes file's ranked list, with the discriminating experiments:

1. **MTP rollback crossing the SWA ring's write frontier after wrap** (section 2.1).
   Highest prior; matches stochastic onset and MTP entanglement.
2. **Interleave map wrong** (window on a full-attn layer or vice versa); audit per-layer
   is_swa against the model config.
3. **Half-precision accumulator saturation**; audit every fp16/bf16 accumulator in the
   attention path; test by forcing fp32 accumulation and watching the onset move.
4. **Allocation-boundary trigger near 9k**: 9k is not a window boundary (17.5 wraps of
   512) but may be memra's first defrag/pool-recycle/arena boundary; compute what fires
   first near 9k generated tokens.
5. **Window convention off-by-one**: predicts GRADUAL degradation from pos=512, not a
   9k cliff; test quality at 1-2k to rule it out.
6. **Stale CUDA-graph keys**: verify capture keys cover ring base pointers and wrap
   offsets.
7. **Sink eviction / learned sinks** (conditional, section 2.3).
8. **Per-layer RoPE mix-up** (vLLM #39914: p-RoPE on global vs local layers fails at
   the transition; llama.cpp refuses K-shift for per-layer-RoPE archs).

The two most discriminating experiments, both cheap: (a) a memra `swa-full` debug mode
(full-length KV on SWA layers, mask-only windowing), the community's standard A/B lever
(#25751's reporter isolated with "This does NOT happen with swa-full enabled", though
that case ultimately resolved as a multi-GPU build issue); soup
disappearing indicts cache handling, soup persisting indicts mask arithmetic or kernel
numerics. (b) Determine whether onset is deterministic-at-a-position (mask/arithmetic)
or stochastic-around-a-depth (race/numeric drift); prior art divides cleanly on this
axis, as does permanent-for-the-process (corrupted persistent state) vs per-request
(arithmetic).

---

## 3. VRAM admission with parked sessions

memra's baseline: the admit-oom fix (SPEC_SHRINK_RESERVE charged on spec-capable models,
gate reads `free + pool_cached`, step-OOM parks with bounded front-requeue; 64/64 x3,
gated with teeth). The survey question: where does this sit against the mature engines'
admission/preemption designs, and what do they add. Full citations:
`notes-q3-admission-preemption.md`.

### 3.1 The design space has four occupied points

- **TGI: pure admission, zero preemption.** A concurrency semaphore rejects over-limit
  requests immediately ("overloaded"); the queue then charges every request
  `input + max_new_tokens + speculate - 1` tokens worst-case against a capacity the
  router MEASURES at warmup ("Flash attention models return their max supported total
  tokens", overriding the user flag). Safe by construction, pays in stranded VRAM.
- **vLLM: optimistic admit, forcible reclaim.** v0: `can_allocate` against free blocks
  minus a 1% watermark ("avoid frequent cache eviction"), preemption RECOMPUTE by default
  with SWAP for beam search (`--swap-space`, 4 GiB/GPU default). v1 dropped swap
  entirely: preemption pops the request, frees blocks, resets computed tokens to zero,
  prepends to waiting; host memory moved into an async `OffloadingConnector` with an
  explicit `cpu_bytes_to_use` budget. The v1 verdict: with prefix caching, recompute
  beats synchronous block swap, and host RAM is better spent as an async prefix-cache
  extension written behind while the request is alive than as an eviction-time copy.
- **SGLang: probabilistic admit, retraction feedback.** The gate's supply term is
  `available_size() + evictable_size()` (radix-cache reclaimable counts as available,
  printed in the OOM line); the demand term discounts each request's future decode by an
  adaptive `new_token_ratio` (starts 0.7, decays, and RESETS UP on every retraction: a
  control loop where over-admission events re-tighten the estimator). Under decode
  pressure it evicts the radix tree first, then `retract_decode` sends victims
  (most-output-first) back to waiting for re-prefill.
- **llama.cpp: the parked-session native.** Idle slots ARE parked sessions: full KV
  retained, next request routed to the slot with the longest prompt LCP
  (`--slot-prompt-similarity`), LRU overwrite otherwise with save-to-host-cache first,
  busy = defer never preempt, plus explicit disk parking
  (`/slots/{id}?action=save|restore`). Admission is slot-count only; the code's own TODO
  ("instead of purging, try to store and resume later?") marks tiered store-and-resume
  as unimplemented.

memra sits at a fifth point: worst-case charging like TGI, but with parked sessions as a
reclaimable buffer TGI lacks, and resident-in-VRAM resume that vLLM/SGLang preemption
cannot offer (their reclaim always costs a re-prefill or a host copy).

### 3.2 Spec-aware admission: three engines confirm the principle, none the mechanism

All three token-accounting engines charge speculative growth explicitly: vLLM as
lookahead slots/tokens in every allocation (zero when spec is off, matching memra's
"plain path untolled"); TGI as `+ speculate` in the admission token math (worst-case
per-request, the closest shape to memra's); SGLang as
`eagle_topk * num_steps + num_draft_tokens` per request, though inside the retraction
target rather than admission. None charges an arena-level reserve like
SPEC_SHRINK_RESERVE: their spec reserves are KV-token quantities over pre-carved block
pools, so the allocator-level effects memra measured (live-vs-parked 1.49x, ~1.3 GiB
capture transient) have no upstream analog. Any engine allocating KV dynamically from a
shared arena needs memra's kind of reserve; there is no prior art to copy for it.

### 3.3 Reclaimable-pool accounting: memra rediscovered the SGLang form

memra's `free + pool_cached` fix has two upstream twins. SGLang does it arithmetically
(the two-term supply above). vLLM v1 does it structurally: prefix-cached unreferenced
blocks live IN the free queue and evict lazily on pop, so the gate cannot regress by
forgetting a counter. The structural form is worth considering when memra's pinned
pool gains a second consumer. The failure mode also has an upstream twin: vLLM #49674
("Deferred KV block frees cause zero-progress preemption cascades", title-verified) is
async-freed memory invisible to the gate, the mirror image of memra's
retires-to-pinned-pool bug, found independently in vLLM. And vLLM RFC #27951 documents the same transient-accounting
drift memra hit: "CUDA graphs require a pretty significant pool of reserved memory...
people adjust gpu memory utilization to make the model not OOM... a confusing and wrong
interface". The profiling-time snapshot misses capture and activation transients, so
the budget knob degrades into a fudge factor.

### 3.4 What to adopt

1. **Self-diagnosing pressure logs.** SGLang's OOM line prints the gate's own arithmetic
   (`available_size=X + evictable_size=Y`); vLLM's preemption warning is rate-limited and
   carries a cumulative counter (added because thrash was invisible without it, #5051).
   Every memra defer/park/refuse event should print `free/pool_cached/reserve_charged`
   plus a cumulative counter.
2. **Three-way admission verdict** (vLLM `OK/LATER/NEVER`): an explicit
   "never fits at this config" fast-reject, distinct from defer, so an oversized request
   errors immediately instead of starving in the never-rejected FIFO.
3. **Budgeted host tier for parked sessions** (SGLang HiCache: explicit byte budget,
   admission discounts `host_hit_length`, layer-overlapped promotion; llama.cpp
   `--cache-ram` as the minimal version; vLLM's write-behind-while-alive lesson). This
   turns memra's park/evict binary into park-resident / park-host / evict, so eviction
   stops costing a full re-prime.
4. **Adaptive reserve with park-event feedback** (SGLang `new_token_ratio`): only if the
   flat reserve ever costs too much concurrency at high c: charge a decaying fraction
   and let park events reset it to conservative.
5. **Warmup-measured capacity** (TGI): memra's admission constants already come from
   measurement (292 vs 286 MiB/session from independent instruments); promoting that to
   a boot-time self-measurement removes constant drift across model configs.

Eviction-order note: SGLang retracts the request that generated the MOST output
(recovers the most VRAM per victim, wastes the most work). memra should keep its inverse
(evict longest-idle parked first): with instant resume as the product promise, punishing
the deepest session is the wrong ordering.

### 3.5 What memra has that these lack

Parked sessions as a first-class, charged-but-evictable admission quantity with zero
re-prefill and zero transfer on resume. llama.cpp's idle slots come closest but account
nothing beyond slot count and will silently LRU-overwrite a parked slot. The
teeth-forced admission gate (reserve forced tiny inverts the verdict 11/64) is a
stronger standing CI property than anything surveyed: vLLM #27951/#25538 and SGLang
#11581/#14972 are all post-hoc field discoveries of gate drift.

---

## 4. Cross-request batched draft/verify speculative decoding

memra's baseline: per-request MTP verify (a q_len=K+1 forward per sequence); the batched
tick chunks plain decode but spec sessions step alone, and the PP-2 door is eager-decode
only. The survey question: who ships multi-sequence verify batching and with what
geometry. Full citations: `notes-q4-batched-specdec.md`.

### 4.1 Four geometries, one winner

| geometry | who | verify batch shape | cost |
|---|---|---|---|
| Batch expansion | vLLM v0 (dead) | sum(k_i+1) single-query rows | k+1x rows and KV reads; vLLM's own docstring: "strictly less efficient than MQA scoring" |
| Varlen multi-query rows | vLLM v1, SGLang ragged_verify, llama.cpp submission side | one forward, total tokens = sum(k_i+1), CSR offsets (`cu_num_draft_tokens` / `qo_indptr`) | near-optimal FLOPs; needs decode attention with q_len > 1 per row; graph-shape churn |
| Tree attention | SGLang EAGLE-2/3, TRT-LLM Medusa/EAGLE | dense [bs, num_draft_tokens] + per-request tree mask | higher acceptance per draft budget; uniform capacity per request; SGLang errors on tree + overlap scheduler |
| Uniform-K padded | TRT-LLM (all modes) | [bs, max_draft_len] always | fixed shapes, graph-friendly; docs concede "There is currently no way to dynamically disable speculation, thus speed ups are only observable at low batch sizes" |

The field converged on varlen rows. vLLM v1's `SpecDecodeMetadata` is the reference: a
flattened token tensor, per-request `num_draft_tokens` (ragged by design),
`cu_num_draft_tokens` cumulative offsets, and a Triton rejection kernel with one program
per request emitting a dense [bs, K+1] output with -1 placeholders. Optimistic advance
(assume all accepted) plus GPU-side correction from the accept-count vector handles
ragged NEXT-step positions; the continuous-batching scheduler absorbs the raggedness
with no re-padding. SGLang's `ragged_verify.py` adds the production trick
memra's graph lane will want: CUDA-graph capture keyed on a rounded-up TOTAL-TOKEN bucket
instead of (bs, K), so per-request K variation does not multiply graph count. Tree
attention turns out to be orthogonal to the cross-request question: SGLang batches trees
per-request-dense anyway, so trees buy acceptance rate, not batching.

llama.cpp, memra's closest architectural cousin, landed cross-request drafting and
batched verify SUBMISSION in May 2026 (PR #22838: one shared draft context over all
slots, all slots' sampled+draft tokens packed into one llama_batch) but the accept phase
stays per-slot sequential host sampler calls with checkpoint-restore rollback. Two of its
scars transfer directly: a session's K+1 verify rows must not straddle a sub-batch
boundary (issue #24840, currently a thrown error), and checkpoint-restore rollback is the
expensive alternative to position-counter correction.

### 4.2 K scheduling under load

vLLM's binary `speculative_disable_by_batch_size` (PR #4592) is gone, replaced by a
K-schedule keyed on concurrency (`num_speculative_tokens_per_batch_size`, e.g. K=3 for
bs 1-64, K=1 for 65-128, off past 129), because "as BS increases, the effective BS
becomes BS*K... beyond a critical BS then SD negatively impacts the decode speed."
SGLang ships adaptive spec (topk=1 only). TRT-LLM has no pressure valve at all. The
transferable shape: K(c) as a table lookup at schedule time, with K=0 rows legal in the
verify kernel (they already are in vLLM's geometry: bonus-only row). memra's crossover
on the PRO 6000 pair is a measurement, not a guess.

### 4.3 Spec under pipeline parallelism: an open corner memra can own

Nobody ships verified batched-spec-under-PP. vLLM's docs: "Pipeline parallelism is not
composable with speculative decoding as of vllm<=0.15.0"; the open RFC #44697 pins the
exact failure: accept counts exist only on the last rank (where the sampler lives), so
non-last ranks over-advance positions after every rejection: "Rope/KV ends up off by one
and verification goes wrong." SGLang documents pp_size==1 for its DFLASH method;
TRT-LLM's status is UNVERIFIED either way. The invariant is simple: every stage must
apply the same per-session accepted-count vector to its KV/position state. memra owns
both stages in one process, so the "broadcast" that vLLM needs across ranks is an
accept-count tensor handed to stage-0's bookkeeping. The gate is forced-rejection argmax
equality across PP-2 (SGLang's `SIMULATE_ACC_LEN` seam, which forces deterministic
ragged accept lengths to test the plumbing, is the test technique to copy).

### 4.4 MTP specifics

All three engines drive MTP heads (DeepSeek, Step-3.x) through the same EAGLE-family
batched verify path; nobody has an MTP-specific verify geometry, so memra's MTP tier
maps onto the varlen geometry unchanged. The recommendation assembled from the notes:
uniform K across the batch first (one graph shape, TRT-LLM-proven), varlen rows with
per-session K later, batch expansion only as an oracle path for gating the varlen kernel
(the MEMRA_FAST=0 pattern), never checkpoint-restore rollback (accept-count correction
of per-session write cursors suffices when attention length is authoritative), and the
packer enforcing no-straddle for a session's verify rows.

---

## 5. Steal vs already-better

| area | steal (source) | memra already does better |
|---|---|---|
| Prefix resume | Checkpoints at every user boundary + erase-past-divergence (llama.cpp #24176); spec state in the checkpoint (`data_spec`) | Bit-identity gates on every reuse tier; GDN recurrent-state snapshots (no surveyed engine resumes hybrid state: llama.cpp #21831) |
| Cache identity | Full-identity keys: adapter, KV dtype, template flags, RoPE config (SGLang extra_key; llama.cpp #26207 as counterexample) | Bytes-decide already closes the hash-collision class vLLM paid a CVE for; cache_salt tenancy already shipped |
| Eviction | Free != evicted lazy eviction; depth-aware order; small protected class for session tails (vLLM free queue, TRT-LLM priorities, RFC #37003) | Sessions-always-win-over-cache policy already stated and enforced |
| Host tier | llama.cpp `--cache-ram` deque shape: two-ratio acceptance, f_keep >= 0.25 floor, dual budgets; contiguous all-layer slabs (vLLM PR #27743) | Bounded pinned-buffer discipline and CUDA owner-thread H2D publication already in the spill doctrine |
| Reuse telemetry | Divergence-position histogram split client-vs-engine; dead-entry mass metric (SGLang #22373) | `/metrics` cached_tokens + lcp_histogram + per-tenant rows already shipped |
| Admission | OK/LATER/NEVER three-way verdict (vLLM v0); gate arithmetic in every pressure log (SGLang); warmup-measured capacity (TGI) | Arena-level spec reserve (no upstream analog); parked sessions as charged-but-evictable admission quantity; teeth-forced gate in CI |
| Preemption/reclaim | Budgeted host park tier (HiCache shape) making evict cheaper than re-prime | Zero-transfer VRAM-resident resume; quoted-OOM-only park rule; front-requeue bounded retries |
| Batched spec verify | Varlen rows + CSR offsets + [bs]-grid rejection kernel (vLLM v1); total-token graph bucketing (SGLang ragged_verify); K(c) schedule; forced-accept test seam (SIMULATE_ACC_LEN) | Draft-file regime (byte-verbatim extracted heads, acceptance-parity proven) is ahead of all surveyed draft-management; K=1..8 self-consistency gate |
| Spec x PP-2 | The invariant from vLLM RFC #44697: same accept-count vector applied on every stage | Single-process two-stage ownership makes the fix bookkeeping, not a cross-rank broadcast protocol; nobody in OSS ships this; open corner to own |
| Spec x prefix cache | llama.cpp restores draft state with checkpoints; SGLang #32459/#19796 show the cost of bypassing | memra's spec-tier bypass is documented and pool-covered where SGLang's collapse was silent; the bypass itself is the gap to close |
| SWA at depth | swa-full debug lever (llama.cpp `--swa-full`, the community's standard A/B); ring slack >= draft depth for spec rollback (#13833, #22660, vLLM #40898); window-boundary unit test (decode across window-1/window/window+1 vs full-cache oracle, HF #37574) | Tick/chunk segmentation invariance already gated bit-exact (tickinv35 + off-grid resume arms); refusal-over-approximation posture already matches where vLLM/llama.cpp landed |
| Depth numerics | fp32 promotion audit of attention accumulators (llama.cpp #23577: half2 numerator saturation -> NaN logits -> soup) | Argmax/bit-identity gates catch this class at gated depths; the gap is coverage at 9k+ generation depth, not discipline |

### Closing note on method

The strongest pattern across all four problems: the engines that shipped mechanisms
without teeth (LMCache's no-reuse overhead, SGLang's silent spec-cache collapse, vLLM's
watermark-drift RFCs) paid in field debugging; the mechanisms that survived are the ones
whose failure modes print their own diagnosis (vLLM's preemption counter, SGLang's
gate-arithmetic OOM line, llama.cpp's divergence logging). Each "steal" row above lands
as a gated lane with before/after receipts, not a port.
