# Prior art: rolling-KV / SWA bugs at depth in other engines

Lane: prior-art-20260809, question 2.
Target symptom in memra: at ~9k+ **generated** tokens output degrades into cross-lingual token
soup. Model: interleaved SWA/full-attention (gemma-style), SWA window = 512 on the SWA layers,
MTP speculative decoding live.

Method: GitHub issue archaeology via the GitHub search API (`gh api search/issues`) + direct
fetches of issue/PR bodies and comment threads, llama.cpp master source fetched raw, StreamingLLM
abstract fetched from arXiv. Everything quoted below was read directly unless marked UNVERIFIED.
All quotes verbatim (trimmed with `...`). Date of survey: 2026-08-09.

---

## 1. llama.cpp — the richest record

### 1.1 The iSWA split-cache design itself documents why rolling SWA + cache ops is dangerous

**PR #13194 — "kv-cache : add SWA support"** (merged 2025-05-20, ggerganov)
https://github.com/ggml-org/llama.cpp/pull/13194

The PR that introduced `llama_kv_cache_unified_iswa` (two internal caches, one full-attn, one
SWA sized `PAD(n_swa*n_seq_max + n_batch)`). The design text states the core hazard explicitly:

> "Note that advanced cache operations such as removing tokens or shifting their positions are
> not possible when using SWA cache, because token information becomes lost when the window
> slides. For such cases, we can 'fallback' to the old implementation by expanding the SWA
> cache size to the full context and disabling the SWA token pruning. ... See the `swa_full`
> flag for more info."

And from the author's own early doubts (outdated section of same PR):

> "The reason we cannot do context caching with SWA enabled is because when the window slides,
> we 'forget' the old KV stuff and there is no way to recover it without recomputing it."

Relevance to memra's 9k-depth soup: llama.cpp's own architect declared rollback/shift on a
pruned rolling SWA cache **structurally impossible** and built a full-size-cache escape hatch
(`--swa-full`). memra's MTP rollback on a rolling window=512 cache is exactly the operation
this design forbids — any rejected-draft rewind past a slot the ring has overwritten silently
reads or recomputes wrong K/V.

### 1.2 The SWA-mask-applied-to-all-layers bug — mid-generation "off the rails" (exact memra shape)

**Issue #15808 — "Eval bug: gpt-oss incoherent output"** → fixed by **PR #15811 —
"kv-cache : fix SWA checks + disable cacheless iSWA"** (merged 2025-09-05)
https://github.com/ggml-org/llama.cpp/issues/15808
https://github.com/ggml-org/llama.cpp/pull/15811

Reporter (interleaved-SWA gpt-oss-20b, short prompt, long generation):

> "the story begins normally, but eventually goes off the rails into unrelated topics with
> incorrect grammar and inconsistent formatting."

ggerganov's fix diagnosis in #15811:

> "the problem is that when we mask the attention we should not use `hparams.swa_type` for all
> layers - only for the SWA layers. This was handled by the KV cache and that is why it had its
> own `swa_type` to differentiate from the one in `hparams`."

Relevance to memra: an interleave-pattern bug (windowing a layer that should be full-attention,
or vice versa) produces exactly this signature — clean start, progressive degradation once
positions exceed the window, no crash. First thing to audit: memra's per-layer is_swa pattern
against the model config, and that the 512-window mask is *never* applied on full-attn layers.

### 1.3 SWA + speculative decoding: a whole sub-lineage of correctness work

- **Issue #13747** — "Feature Request: --swa-extra parameter needed to restore speculative
  decode function with SWA" (https://github.com/ggml-org/llama.cpp/issues/13747). Documents
  that naive spec decode on a rolling SWA cache loses information; asks to extend the window by
  the max draft length so speculative tokens can be masked out instead of destroying state.
- **PR #13833** — "llama : use n_swa + n_ubatch cells for SWA cache" (merged 2025-05-31,
  ggerganov) (https://github.com/ggml-org/llama.cpp/pull/13833). Checklist includes:
  > "Enable SWA speculative decoding" and "Allow short SWA rollbacks (avoids cache
  > recalculations caused by whitespace truncation of the last response)"
  i.e. the SWA ring is deliberately over-allocated by `n_ubatch` cells specifically so that a
  *bounded* rollback is safe. Rollbacks longer than the slack are still unsafe.
- **PR #22660** — "speculative : auto-enable swa_full for SWA draft models"
  (https://github.com/ggml-org/llama.cpp/pull/22660):
  > "For SWA draft models, the rotating SWA cache can handle speculative cache operations
  > within the sliding window, but it cannot reuse prefixes outside of that window."
  Cross-references vLLM PR #40898 (same fix shape in vLLM: keep full KV allocation for the SWA
  drafter while SWA is still used for compute).

Relevance to memra: the industry consensus fix for spec-decode × rolling SWA is "give the SWA
cache slack cells >= max draft length, or full allocation" — check that memra's SWA ring has
>= MTP-depth spare slots beyond the window and that rejected-draft rewind never crosses the
ring's write frontier after a wrap.

### 1.4 KV defrag corrupting output / asserting under context shift

- **Issue #12253 — "Eval bug: garbage output right after kv-cache defragmentation for CPU
  backend"** (https://github.com/ggml-org/llama.cpp/issues/12253):
  > "using the API or the WebUI to make the model generate large outputs on two slots at once,
  > I get garbage output (which stays until server restart) as soon as KV-cache defragmentation
  > occurs once."
  Thread narrows it to **quantized KV cache** ("the bug does not occur with KV-cache
  quantization disabled") — defrag moved quantized blocks with wrong byte math.
- **Issue #14059 — "Misc. bug: KV defrag bug: nf != nh"**
  (https://github.com/ggml-org/llama.cpp/issues/14059): sporadic
  `GGML_ASSERT(nf == nh && "KV defrag bug: nf != nh")` under parallel slots; reporter confirms
  > "Yes, it doesn't occur with it [--no-context-shift], so that's it then."
  i.e. defrag bookkeeping went inconsistent specifically when the context-shift path ran.
- **PR #14189 — "kv-cache : fix use-after-move of defrag info"** (merged 2025-06-15)
  (https://github.com/ggml-org/llama.cpp/pull/14189):
  > "This bug caused defrags to be performed only when a shift is needed."
  A use-after-move silently coupled two unrelated cache maintenance ops.

Relevance to memra: cache-maintenance passes (defrag/compaction/shift) are where "worked for
hours, then permanently garbage" comes from, and the quantized-KV × defrag interaction is a
proven corruption pair. If memra has any KV compaction/defrag pass, check whether the first one
fires near 9k.

### 1.5 SWA checkpoint/restore discarding context (server layer, not kernel)

**Issue #21769 — "Eval bug: Gemma-4: SWA checkpoint restoration discards mid-conversation
context, causing context regression across turns"** (closed)
https://github.com/ggml-org/llama.cpp/issues/21769

> "the server restores a checkpoint anchored early in the conversation ... and only processes a
> narrow window of tokens forward from there (~1200-1400 tokens actually evaluated against a
> full prompt of 11,000-18,000 tokens), silently discarding the intervening conversation."

Related: **Issue #21831** ("forcing full prompt re-processing due to lack of cache data (likely
due to SWA or hybrid/recurrent memory)", https://github.com/ggml-org/llama.cpp/issues/21831),
**PR #21749** ("server: ensure prompt caching for SWA models" — `pos_min_thold` was computed as
`pos_next - n_swa` even under `--swa-full`, https://github.com/ggml-org/llama.cpp/pull/21749),
**Issue #24587** (SWA checkpoint invalidation loops with RAG prompts,
https://github.com/ggml-org/llama.cpp/issues/24587),
**Issue #25751** ("SWA on Gemma 4 forgets key details" — resolved as an NCCL/multi-GPU build
interaction, but the reporter's key diagnostic was
> "This does NOT happen with swa-full enabled."
https://github.com/ggml-org/llama.cpp/issues/25751).

Relevance to memra: `--swa-full` is the community's standard A/B lever for "is the rolling SWA
cache guilty?" memra should grow the same lever: a debug mode that keeps full-length KV on SWA
layers (mask-only windowing, no pruning). If soup disappears → cache handling; if it stays →
mask arithmetic or kernel numerics.

### 1.6 The long-session `////` collapse — NaN logits at depth, MTP entangled (closest cousin to memra's bug)

**Issue #23577 — "Eval bug: MTP with Qwen3.6 27B outputs repeated //// after long session"**
(open at time of survey) https://github.com/ggml-org/llama.cpp/issues/23577

OP: server correct for hours, then permanent `////` loop; restart required; recurs on resume.
Multiple backends (CUDA, ROCm, Vulkan), with and without MTP. Two diagnostic gems in-thread:

1. Rogn4r: > "The logs show that draft acceptance drops to 0.00000, causing the main model and
   MTP draft to desync during context checkpoint restoration."
2. redthing1's investigated root cause on gfx1100 (long, quantitative comment): the FA tile
   kernel kept softmax max/denominator in F32 but accumulated the V-weighted numerator in
   `half2`:
   > "the half2 numerator overflows on the ninth contribution and the FA output becomes Inf"
   > "With V=1 and TK=8192, the half2 numerator stops advancing at 256 after roughly 2048
   > contributions while the F32 denominator continues."
   > "one target hidden row became non-finite, eventually making all 248,320 target logits NaN.
   > MTP acceptance then collapsed to zero and the sampler emitted / or repeated slashes."
   Fix candidate: promote the VKQ accumulator to float2 (~0.35% wall-time cost).

Also title-level corroboration (UNVERIFIED, search-result title only):
**Issue #23606 — "Eval bug: All logits NaN at ~80K+ context on Qwen3.6-35B-A3B (single GPU,
CUDA)"** https://github.com/ggml-org/llama.cpp/issues/23606

Relevance to memra: this is the canonical *depth-triggered numeric* failure — fp16 accumulator
saturation is a function of **how many contributions** the attention reduction sums, which grows
with context on full-attn layers and crosses saturation thresholds at a characteristic depth.
memra is a custom CUDA engine on Blackwell; audit every fp16/bf16 accumulator in the attention
path (esp. any fast-fp16 tile) for numerator-in-half + denominator-in-float splits. NaN/garbage
logits → near-uniform sampling over a multilingual vocab → cross-lingual token soup.

### 1.7 CUDA-graph reuse regression → infinite multilingual garbage at long context

**Issue #21640 — "Eval bug: Regression: CUDA graphs props check optimization causes infinite
generation with Gemma4 26B on long context"** (closed; fixed by **PR #21635 — "CUDA: also store
`node->src->data` ptrs for equality check"**, merged 2026-04-08)
https://github.com/ggml-org/llama.cpp/issues/21640
https://github.com/ggml-org/llama.cpp/pull/21635

> "When running Gemma4 26B ... with long context, the model generates infinite garbage tokens
> (e.g., `<unused>` tokens and other malformed output). This regression was introduced by
> commit c5ce4bc22." [the reporter's own garbage sample is cross-lingual: "As an user, I've,
> **I'm, VPN-configuration is not a wonderful, I'm, ** la ..."]

The bug: the CUDA-graph reuse check compared graph properties but not source-tensor data
pointers, so when KV-cache tensor addresses changed at depth (cache growth/rotation) the stale
captured graph kept reading old pointers.

Relevance to memra: if memra uses CUDA graphs for decode, verify graph-reuse keys include every
KV/ring pointer and ring offset that changes at large positions — a stale graph is a
depth-triggered, permanent-corruption mechanism with exactly this token-soup symptom.

### 1.8 Cross-lingual token soup precedent — engine bug, not "model at depth"

**Issue #8853 — "Bug: Gemma 2 incoherent output when using quantized k cache without Flash
Attention"** (closed stale) https://github.com/ggml-org/llama.cpp/issues/8853

> "Output like 'Mh giàu され rodas reliablyacheteurδε Są' happens when using quantized K cache,
> CUDA, with Gemma 2."

Relevance to memra: verbatim cross-lingual token soup from a pure engine cause (quantized-K
path on an interleaved-SWA model). Strong prior that memra's soup is an engine bug: soup =
high-entropy sampling over a multilingual BPE vocab when logits are corrupted, and no engine
doc surveyed attributes this symptom to a *model* property at 9k depth. (Interleaved-SWA models
are trained with the rolling window; depth alone should not produce soup below the trained
context length.)

### 1.9 Context shift: deprecated as a footgun; ring-shift bookkeeping bugs

- **PR #15416 — "server : disable context shift by default"** (merged 2025-08-19)
  https://github.com/ggml-org/llama.cpp/pull/15416:
  > "Context shift was a useful feature in the past with pre-trained models and the raw
  > /completions API. But today, it is causing a lot of confusion, so it is better to disable
  > it by default."
- **Issue #18409** — llama.android `shift_context()` forgot to maintain
  `stop_generation_position` → infinite generation after a shift
  (https://github.com/ggml-org/llama.cpp/issues/18409). Trivial but instructive: every counter
  that references absolute position must be updated on shift.
- Source (fetched from master, `src/llama-kv-cache.cpp`): K-shift is implemented as a RoPE
  re-rotation graph (`build_graph_shift` / `set_input_k_shift`), and `get_can_shift()` returns
  false for per-layer-RoPE archs:
  > "// Step35 uses per-layer RoPE dims; K-shift assumes a single global n_rot."
  Also: `LLAMA_SWA_TYPE_STANDARD` mask arithmetic in `src/llama-hparams.h`:
  `if (p1 - p0 >= (int32_t) n_swa) return true;` — i.e. key at p0 is visible iff
  `p1 - p0 < n_swa` (window *includes* the current token, strictly `n_swa` keys total).
- **Issue #3440** — the original StreamingLLM feature request
  (https://github.com/ggml-org/llama.cpp/issues/3440); `main`'s legacy context swap keeps
  `n_keep` initial tokens (attention-sink preservation) — llama.cpp never shipped naive
  sink-evicting shift for that reason.
- **Issue #12637** — "Feature Request: Interleaved sliding window attention support for gemma 2
  and 3" (found via search; UNVERIFIED body) — the memory-cost motivation for the iSWA split.

Relevance to memra: (a) pin down memra's window convention against gemma's reference —
llama.cpp's is `p1 - p0 < n_swa` inclusive-of-current; an inclusive/exclusive mismatch is the
SGLang bug below; (b) if memra ever RoPE-re-rotates shifted keys, quantized-K re-rotation and
per-layer RoPE are known hazards.

---

## 2. HuggingFace transformers — mask/cache off-by-ones at the window boundary

### 2.1 Wrong HybridCache update exactly when seq length reaches the window

**Issue #37574 — "Wrong KV cache update for sliding-window attention (SWA) layers when total
sequence length reaches window size"** (closed; proper fix in **PR #38046 — "Fix cache
update!"**, merged 2025-05-09; intermediate PR #37972 "broke it more, unintentionally" per
maintainer Cyrilvallez)
https://github.com/huggingface/transformers/issues/37574
https://github.com/huggingface/transformers/pull/38046

Maintainer comment on the fix:

> "This was not properly handled in the PR mentioned above: proper fix is #38046! ...
> [generations] still appear better, especially with regards to special tokens, at least with
> Gemma2"

Relevance to memra: the exact transition step where `pos == window` (and the first ring
overwrite) is the highest-risk instruction in a rolling cache; write a unit test that decodes
across positions window-1, window, window+1 and diffs against a full-cache oracle.

### 2.2 Mask slicing wrong whenever seq length > sliding window

**PR #35681 — "Fix mask slicing for models with HybridCache"** (merged 2025-01-28)
https://github.com/huggingface/transformers/pull/35681

> "mask slicing was wrong in all cases when the sequence length is larger than the sliding
> window. It is currently broken and leads to garbage generation when using padding."

### 2.3 SDPA path silently ignored sliding_window for ~6 months

**Issue #28980 → PR #30127 — "Fix SDPA sliding window compatibility"** (merged 2024-04-17)
https://github.com/huggingface/transformers/pull/30127

> "This bug dates back to #26572 where `sliding_window` was not properly accounted for in the
> `_prepare_4d_causal_attention_mask_for_sdpa` method."

Companion: **PR #34093** "Fix FA2 attention for models supporting sliding window" and
**PR #33586** "Phi3: fix attn for sliding window" (titles verified via search).

### 2.4 Phi-3 long-prompt gibberish = dropped sliding-window mask term

**Issue #32945 — "Regression in generating text with Phi-3-mini-4k-instruct with a long prompt
(gibberish in v4.42+)"** (closed)
https://github.com/huggingface/transformers/issues/32945

Maintainer (zucchini-nlp):

> "I think the degradation is related to the `self.config.sliding_window` which is no longer
> used when constructing the attention mask."

Relevance of 2.2-2.4 to memra: three independent transformers incidents where the *mask
construction* for positions beyond the window was wrong while everything below the window was
fine — the precise "fine until depth D" signature. memra's SWA mask arithmetic at
pos >> window (and its interaction with batch offsets during MTP verify, where q positions in
one pass differ) is a prime suspect.

### 2.5 Off-by-one family in window boundary math (recent, streaming model)

**PR #47010 — "Fix off-by-one in sliding_window_mask_function right window boundary"** (+
follow-ups #46310, #47028, #47231; issue #46305: "right_window_size=4 attends to 3 future
frames instead of 4") https://github.com/huggingface/transformers/pull/47010
Relevance: window-boundary off-by-ones keep recurring even in 2026 code; treat the ±1 as
guilty-until-tested.

---

## 3. vLLM — interleaved SWA support gaps and block-reuse corruption

### 3.1 gemma-2 launch: interleaved SWA simply not supported; window capped

**Issue #6220 — "Gemma2 supports 8192 context with sliding window, but vllm only does 4196 or
fails if try 8192"** (closed) https://github.com/vllm-project/vllm/issues/6220

vLLM's own warning quoted in-thread:

> "Gemma 2 uses sliding window attention for every odd layer, which is currently not supported
> by vLLM. Disabling sliding window and capping the max length to the sliding window size
> (4096)."

Still the same posture for gemma-3 + FlashInfer a year later — **Issue #20865** (closed):
> "gemma3_text has interleaved attention, which is currently not supported by the FLASHINFER
> backend. Disabling sliding window and capping the max length to the sliding window size
> (1024)."
https://github.com/vllm-project/vllm/issues/20865

Relevance to memra: two major engines chose "refuse/disable" over "run interleaved SWA
slightly wrong" — evidence that getting interleaved SWA right at depth is genuinely hard and
half-support is worse than none.

### 3.2 SWA eviction leaves stale physical block ids → reachable output corruption

**Issue #42273 — "[Bug]: SWA eviction leaves stale block table entries"** (open)
https://github.com/vllm-project/vllm/issues/42273

> "Sliding window attention (SWA) eviction can leave stale physical block ids in the
> worker/GPU block table for an already-running request. ... if an attention backend gets the
> SWA exclusion wrong, stale entries can point to freed/reused physical KV blocks."

Third-party comment (woosebastian) upgrades it from theoretical:

> "This is reproducible as real output corruption, not only a robustness concern. ... It shows
> up on a hybrid model with `int8_per_token_head` KV and prefix caching off"
(companion bugs: #50702, #50749 — Gemma-4 hybrid first-token corruption, titles verified).

### 3.3 Freed SWA blocks reallocated to a quantized full-attn group → NaN once input > window

**PR #47574 — "[Bugfix] Zero new KV blocks for quantized + sliding-window hybrid caches"**
https://github.com/vllm-project/vllm/pull/47574

> "produces NaN / all-zero output once the input exceeds the sliding window. ... The
> sliding-window group frees blocks mid-request; those pages get reallocated to the fp8 group
> and read back including slots not written this step. As bf16 that leftover is harmless
> garbage, but as fp8 it decodes to NaN/Inf and corrupts attention."

Relevance of 3.2-3.3 to memra: rolling-window *eviction* creates a class of use-after-free /
stale-mapping bugs whose symptom is exactly "correct until the window starts rolling, then
corrupt." If memra shares one pool between SWA-layer and full-attn-layer KV (or between target
and MTP-draft KV), audit recycle paths and consider zero-fill-on-realloc as a diagnostic.

### 3.4 Interleaved attention + dual RoPE prefill hang (gemma-4 era)

**Issue #39914 — "[Bug]: Gemma 4: Engine hang during large prefill caused by Interleaved
Attention and p-RoPE implementation"** (closed) https://github.com/vllm-project/vllm/issues/39914

> "The hang occurs specifically when the prefill process hits the global attention layers. ...
> the transition from local sliding-window attention to global attention (utilizing p-RoPE)
> fails during large-batch prefill."

Relevance: position-embedding handling differs per layer type in modern interleaved models; a
per-layer-type RoPE mix-up is depth/shape-triggered.

### 3.5 Spec decode × SWA in vLLM

- **PR #40898** — DFlash drafter SWA support (verified body):
  > "Without this, SWA layers in the drafter lose their windowed-attention configuration and
  > run as full attention ... Keep DFlash SWA visible to attention metadata while allocating
  > full draft KV for those layers, so target-prewritten context K/V is not evicted by masked
  > draft-block tokens."
  https://github.com/vllm-project/vllm/pull/40898
- **PR #46032** — "[Bugfix] Add support for SWA draft models in speculative decoding" and
  **PR #50169** — "Fix KV cache allocation for sliding-window drafters and local-attention pool
  sizing" (titles verified via search; bodies UNVERIFIED).

Relevance: same convergent fix as llama.cpp #22660 — spec decoding forces the SWA cache to keep
more than the window (draft tokens must be evictable/rollback-able without destroying
in-window history).

---

## 4. SGLang — the cleanest off-by-one and eviction-boundary receipts

### 4.1 Triton decode gathers one key too few — silent quality loss at every step past window

**PR #32087 — "Fix Triton SWA decode window dropping the oldest in-window key"** (open/unmerged
at survey time) https://github.com/sgl-project/sglang/pull/32087

> "`sliding_window_size` is stored as `config.sliding_window - 1` (a radius ...) and the window
> is two-side inclusive: a query at position p attends keys [p - sliding_window_size, p], i.e.
> sliding_window_size + 1 of them. The Triton decode kernel ... consumes the gathered window
> verbatim with no per-query masking, so the buffer must already hold the full window. ... The
> Triton path is one token short, so on every decode step past the window size it silently
> drops the oldest in-window key. ... This degrades attention quality (no crash) for every
> SWA-layer model on the Triton backend (Gemma2 / Gemma3 / ...), in both the eager and
> cuda-graph decode paths."

Relevance to memra: THE textbook window-arithmetic bug: radius-vs-count convention mismatch
between config, gather, and kernel. Also note the failure *shape*: an off-by-one gives gradual
quality degradation, not instant soup — so if memra's soup is abrupt at ~9k, an off-by-one
alone is probably not the whole story, but a larger convention mismatch (e.g. windowing 511 vs
512 vs "everything since last wrap") could be.

### 4.2 SWA radix-cache eviction boundary bugs

**PR #22469 — "Fix SWA eviction boundary and page-align chunked prefill"** (unmerged at survey)
https://github.com/sgl-project/sglang/pull/22469

> "Fix `_evict_swa` over-eviction when `page_size > sliding_window_size` ... Fix
> `_insert_helper` missing boundary case where `swa_evicted_seqlen == total_length` (all
> remaining tokens evicted) ... Page-align chunked prefill `trunc_len` to prevent chunk
> boundary drift."

Relevance: eviction frontier vs. allocation granularity mismatches (page/block size vs window
size) — memra should check its ring/page granularity against window=512 (512 is suspiciously
aligned with common block sizes; an off-by-one-page over-eviction would kill in-window keys).

### 4.3 Softcap silently dropped (gemma-2) — quality-only engine bug

**Issue #33915** — "[Bug] FlashInfer backend drops attn_logit_softcapping: the cap is passed to
the deprecated forward() instead of plan(), so gemma-2 and grok-1 run uncapped" (title verified
via search; body UNVERIFIED). https://github.com/sgl-project/sglang/issues/33915
Relevance: model-specific attention scalars (softcap, per-layer scale) silently not reaching the
kernel is another "no crash, output subtly wrong, worse at depth" class.

---

## 5. StreamingLLM / attention sinks — when eviction itself breaks the model

**Paper: "Efficient Streaming Language Models with Attention Sinks" (Xiao et al., ICLR 2024),
arXiv:2309.17453** — abstract fetched and verified:

> "Window attention, where only the most recent KVs are cached, is a natural approach -- but we
> show that it fails when the text length surpasses the cache size. We observe an interesting
> phenomenon, namely attention sink, that keeping the KV of initial tokens will largely recover
> the performance of window attention."
> "the emergence of attention sink is due to the strong attention scores towards initial tokens
> as a 'sink' even if they are not semantically important."

(The perplexity-explosion-on-first-token-eviction figure is the paper's Figure 1/3 claim; the
abstract's "fails when the text length surpasses the cache size" is the verified formulation.)

IMPORTANT caveat for memra: StreamingLLM's failure mode applies to models **trained with full
attention** then windowed at inference. Gemma-style interleaved-SWA models are *trained* with
the rolling window on SWA layers, so sink eviction inside the trained window is not, by itself,
expected to break them — **unless** the engine (a) applies the window to layers trained
full-attention (see llama.cpp #15811, exactly that), or (b) the model relies on a learned sink
token that the engine's ring handles specially upstream but memra evicts. gpt-oss-style models
even carry explicit learned attention-sink parameters per layer (llama.cpp PR #15091 era);
verify whether memra's target model family has such sinks and whether they're honored.

---

## 6. Ranked: most-likely-transferable diagnoses for memra's 9k soup

1. **MTP rollback × rolling SWA ring** — highest prior. llama.cpp declared rollback on a pruned
   SWA cache impossible (#13194), sized the ring `n_swa + n_ubatch` specifically to allow only
   *short* rollbacks (#13833), and auto-forces full-size SWA cache for spec drafters (#22660;
   vLLM #40898/#46032 converged on the same). Check: after a rejected draft at position p, does
   memra's SWA ring restore its write pointer AND is every slot the draft overwrote still
   recoverable? A wrap during a speculative batch (draft tokens overwrite the oldest in-window
   real keys, then the draft is rejected) is unrecoverable unless the ring has >= draft-depth
   slack. With window=512 the ring wraps ~17 times by 9k — but corruption probability per wrap
   is small if the race is rare, which matches "hours fine, then permanent soup" (#23577).
2. **Wrong layer gets the window (interleave pattern / swa_type applied to all layers)** —
   llama.cpp #15808/#15811 is the exact symptom (starts fine, goes off the rails during long
   generation) and the exact model shape (interleaved). Audit memra's per-layer is_swa mapping
   and that mask type is per-layer, not per-model.
3. **fp16/half accumulator saturation at depth in the attention kernel** — #23577 (half2 VKQ
   numerator overflow/stall → NaN logits → token collapse), #23606 (all logits NaN at 80k+).
   Contribution counts grow with context on full-attn layers; saturation has a characteristic
   onset depth. Soup (near-uniform multilingual sampling) is precisely what corrupted logits
   produce (#8853, #21640). Audit every half-precision accumulator in memra's SDPA/FA path;
   test: force fp32 accumulation, see if soup onset moves/disappears.
4. **Buffer-boundary rather than window-boundary trigger** — 9k with window=512 is NOT a window
   boundary, but it may be an *allocation* boundary: first wrap of an over-allocated ring,
   first defrag/compaction trigger, first KV-block-pool exhaustion + recycle
   (vLLM #47574/#42273: freed window blocks reused and read-back → NaN with quantized KV;
   llama.cpp #12253: first defrag → permanent garbage with quantized KV). Compute what memra
   allocates/pads for the SWA cache and what fires first near 9k generated tokens.
5. **Window mask arithmetic convention at pos > window** — radius vs count, inclusive vs
   exclusive of current token (SGLang #32087 dropped the oldest in-window key every step;
   HF #37574/#35681/#30127/#32945/#47010 all wrong exactly past the boundary). Reference
   semantics to test against: llama.cpp `is_masked_swa` STANDARD = mask iff `p1 - p0 >= n_swa`.
   Note this predicts *gradual* degradation from pos=512 onward, not a 9k cliff — useful to
   *rule out* by testing generation quality at 1-2k.
6. **CUDA-graph reuse with stale KV/ring pointers** — llama.cpp #21640/#21635: graph equality
   check missed `node->src->data`; long-context cache growth changed addresses; captured graph
   read stale memory → infinite multilingual garbage. If memra captures decode graphs, verify
   the capture key covers ring base pointers and wrap offsets.
7. **Attention-sink eviction / learned sink mishandling** — lower prior for a
   trained-with-SWA model (StreamingLLM applies to full-attn-trained models), but cheap to
   check: does quality recover if the first N tokens are pinned (never evicted) in the SWA
   ring? If yes, either the interleave map is wrong (see #2) or the model family has learned
   sinks the engine must keep.
8. **Position-embedding per-layer-type mix-up at large positions** — vLLM #39914 (p-RoPE on
   global layers vs standard RoPE on local layers; fails at the local→global transition);
   llama.cpp `get_can_shift()` refuses K-shift for per-layer-RoPE archs. If memra's model has
   different RoPE bases/scales for SWA vs full layers, check which one each layer actually
   gets, and any fp16 rounding of large position values in the RoPE tables at pos ~9k.

## 7. Open uncertainties

- Why 9k specifically? None of the surveyed bugs put a cliff at 17.5x the window. Candidates:
  an internal allocation boundary (hypothesis 4), a slow numeric drift crossing a threshold
  (hypothesis 3), or a rare race whose expected first-hit time lands near 9k (hypothesis 1).
  Measuring whether onset is deterministic-at-a-position vs stochastic-around-a-depth is the
  single most discriminating experiment.
- Whether memra's degradation is *permanent for the process* (like llama.cpp #12253/#23577 —
  points at corrupted persistent state: cache bytes, graph capture) or *per-request* (points at
  mask/position arithmetic). Prior art divides cleanly on this axis.
- sgl PR #32087 and #22469 were unmerged at survey time — the diagnoses are detailed and
  self-consistent but not yet maintainer-confirmed.
- llama.cpp #23606 (NaN at 80k) and vLLM #46032/#50169, sgl #33915 cited on title only
  (UNVERIFIED bodies).
- The StreamingLLM perplexity-explosion figure was not re-read beyond the abstract's claim.
- Did not find any engine documenting cross-lingual soup as a *model* property at depth within
  trained context — every soup receipt found traces to an engine defect (quantized-KV paths,
  stale CUDA graphs, NaN logits). Absence of evidence, but consistent: treat memra's soup as an
  engine bug until a full-KV oracle run at 9k+ reproduces it.
