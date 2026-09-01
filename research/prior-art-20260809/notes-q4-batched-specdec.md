# Prior art: cross-request batched speculative draft/verify

**Question:** who ships multi-sequence verify batching (verifying speculative tokens for MANY
concurrent sequences in one target forward pass, with ragged per-sequence accept lengths), and
what geometry do they use?

Survey date: 2026-08-09. All quotes verbatim from fetched sources; anything not fetched is
marked UNVERIFIED. memra context: single-model, single-node, Rust+CUDA, sm_120a, 2x RTX PRO
6000, PP-2, per-request MTP spec decode already shipping; the gap is cross-request batched
draft/verify.

---

## 1. vLLM

### 1.1 v0: batch expansion (each spec position becomes its own row)

**Claim:** vLLM v0 verified spec tokens by *expanding* the batch — every sequence with k draft
tokens became k+1 single-query rows (prefix chains `[]`, `[t0]`, `[t0,t1]`, ...), so the verify
batch size was `sum over seqs of (k+1)`. This existed precisely to avoid needing an MQA
(query-len>1 decode) kernel.

**Quote** (class docstring, `BatchExpansionTop1Scorer`):
> "Batch expansion converts a list of sequences and multiple query positions to a new batch of
> sequences, each with a single query position. This allows for MQA-like scoring in speculative
> decoding without requiring an MQA kernel. It is strictly less efficient than MQA scoring. It
> only supports scoring the top1 proposal tokens of the proposer, instead of topk/tree."

Also in `_create_single_target_seq_group_metadata`:
> "This is a hack. Technically, spec decoding should compute num_lookahead slots at one shot,
> but instead, it expands the batch and evaluate one by one right now."

- URL: https://raw.githubusercontent.com/vllm-project/vllm/v0.6.4/vllm/spec_decode/batch_expansion.py
  (repo file `vllm/spec_decode/batch_expansion.py`, tag v0.6.4 — file no longer exists on main;
  the whole `vllm/spec_decode/` v0 tree is gone, replaced by `vllm/v1/spec_decode/`).
- v0 also could NOT do per-sequence ragged proposal lengths — quoted comment in `_expand_batch`:
  > "vLLM currently only supports proposal lens equal to zero or the batch proposal len. This
  > adds some complexity (splitting the batch into spec and non spec sequences) and should be
  > removed in the future. It can be done by supporting per-sequence proposal lens."
  i.e. v0 handled raggedness by *splitting the batch into spec (uniform k) and non-spec (k=0)
  groups*, not true per-seq k.
- Design doc link ("Information on batch expansion") is still referenced from the docs page:
  https://docs.vllm.ai/en/v0.9.2/features/spec_decode.html (Google doc
  1T-JaS2T1NRfdP51qzqpyakoCXxSXTtORppiwaj5asxA; UNVERIFIED content — Google-doc, not fetched).
- Independent confirmation of why v0's approach died (arXiv 2510.22876v2, "Batch Speculative
  Decoding Done Right"): "A variant approach, employed by vLLM's v0 engine, avoids truncation
  through *batch expansion*: each sequence is duplicated K times with progressively longer draft
  prefixes ... While this preserves accepted tokens, it incurs K× redundant computation and
  memory overhead" and "vLLM deprecated this in v1 due to batch expansion memory overhead."
  https://arxiv.org/html/2510.22876v2

**Transfer to memra:** batch expansion is the zero-new-kernel fallback — memra's existing
single-query decode path could verify c concurrent spec sequences today at (k+1)x row cost; it's
a correctness bridge, not the destination (vLLM itself abandoned it).

### 1.2 v1: flattened varlen rows + cu_num_draft_tokens + Triton rejection kernel

**Claim:** vLLM v1 dropped batch expansion. The verify pass packs each request's (k_i draft +
1 bonus) positions as *extra query positions of the same request row* in one flattened varlen
forward (total tokens = sum(k_i)+bs). Raggedness is carried as CSR-style cumulative offsets, and
accept/reject runs as a Triton kernel with one program per request walking its own segment.

**Evidence (fetched code, main branch):**
- `vllm/v1/spec_decode/metadata.py` — `SpecDecodeMetadata` is the whole geometry:
  ```
  draft_token_ids: torch.Tensor        # [num_tokens]   (flattened across requests)
  num_draft_tokens: list[int]          # [batch_size]   (per-request k_i — ragged by design)
  cu_num_draft_tokens: torch.Tensor    # [batch_size]   (cumsum → CSR row offsets)
  target_logits_indices: torch.Tensor  # [num_tokens]
  bonus_logits_indices: torch.Tensor   # [batch_size]
  logits_indices: torch.Tensor         # [num_tokens + batch_size]
  ```
  https://github.com/vllm-project/vllm/blob/main/vllm/v1/spec_decode/metadata.py
- `vllm/v1/sample/rejection_sampler.py` — logits arrive as
  "[num_tokens + batch_size, vocab_size] ... probabilities from different requests are flattened
  into a single tensor"; `rejection_greedy_sample_kernel[(batch_size,)]` loads
  `cu_num_draft_tokens[req_idx-1]..[req_idx]` and walks `for pos in range(num_draft_tokens)` with
  an early `rejected` flag; output buffer is dense `[batch_size, max_spec_len+1]` filled with
  `PLACEHOLDER_TOKEN_ID = -1` for rejected slots, filtered on CPU in `parse_output`. Padded/
  invalid drafts are `-1` and auto-rejected ("-1 is used for padded draft token ids that should
  be rejected"). `MAX_SPEC_LEN = 128` per request.
  https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/rejection_sampler.py
- The drafter side pads by default but can go ragged: `SpeculativeConfig.disable_padded_drafter_batch`
  — "Disable input padding for speculative decoding. If set to True, speculative input batches
  can contain sequences of different lengths, which may only be supported by certain attention
  backends. This currently only affects the EAGLE method of speculation."
  https://docs.vllm.ai/en/stable/api/vllm/config/speculative/
  So even vLLM treats ragged draft-side batches as backend-dependent; verify-side raggedness via
  cu_num_draft_tokens is unconditional.
- V1 spec-decode design RFC: vllm-project/vllm issue #14719 "[RFC][V1][Spec Decode] V1 Spec
  Decode Eagle Support" — body: "This doc summarizes the algorithm and design of V1 spec decode"
  (design doc is a Google doc 1xRix4gZqmKAPe6c9mw7B883-vxhlErpQkIBpRgcNZtg; UNVERIFIED content).
  https://github.com/vllm-project/vllm/issues/14719
- Attention kernel requirement: the varlen verify needs "query length > 1 decode" support.
  FlashInfer's ragged layout is exactly this ("a (qo_len, num_heads, head_dim) q tensor, and a
  qo_indptr tensor demarcating the boundaries") and FlashInfer's TRT-LLM-gen decode explicitly
  exposes "mask ... causal attention mask for xqa speculative decoding. max_q_len ... The maximum
  query sequence length across all requests when using variable-length queries."
  https://docs.flashinfer.ai/generated/flashinfer.decode.trtllm_batch_decode_with_kv_cache.html
  (I did not locate a single canonical "FlashAttention speculative-verify kernel" issue —
  UNVERIFIED which exact FA PR added it; the FlashInfer API above is verified.)

**Transfer to memra:** this is the reference geometry for memra's cross-request verify: one
flattened forward, `cu_num_draft_tokens` CSR offsets, per-request rejection walk, dense
[c, K+1] output with -1 placeholders. It maps directly onto a varlen decode-batch kernel with
per-row q_len = k_i+1 and needs no batch inflation.

### 1.3 Ragged NEXT step (post-accept positions)

**Claim:** vLLM v1 handles differing accepted lengths by advancing `num_computed_tokens`
per-request and just building the next flattened batch from the new per-request lengths — the
continuous-batching scheduler absorbs raggedness; there is no re-padding step. The V1 scheduler
"treats both prompt and output tokens the same way by using a simple dictionary (e.g.,
{request_id: num_tokens}) to dynamically allocate a fixed token budget per request"
(https://docs.vllm.ai/en/v0.8.5/getting_started/v1_user_guide.html). The rejected tokens' KV
entries are simply overwritten/ignored since attention reads only up to the accepted length.
Confirmed indirectly by the PP RFC (#44697, below): "num_computed_tokens is advanced
optimistically (assuming every draft is accepted) and then corrected after the forward by a GPU
kernel that's gated on valid_sampled_token_count_gpu" — i.e. optimistic advance + GPU-side
correction, not host-side re-layout.

**Transfer to memra:** memra's session table should do the same: advance KV write positions
optimistically by K+1 during verify, then correct per-session from the accept-count vector; no
compaction of KV needed if attention length is authoritative.

### 1.4 Disabling / shrinking speculation under batch load

**Claim (v0):** `speculative_disable_by_batch_size` existed — "disables speculative decoding
for new incoming requests if the number of enqueued requests exceeds this value" (SqueezeBits
blog, https://blog.squeezebits.com/vllm-vs-tensorrtllm-11-speculative-decoding-37301). Landed
as PR #4592 "[Dynamic Spec Decoding] Auto-disable by the running queue size": "we allow users to
set a threshold, in terms of the number of requests in the current running queue, to disable
speculative decoding for new incoming requests."
https://github.com/vllm-project/vllm/pull/4592
It was buggy at times — issue #25112 "[Bug]: [Spec Decode] Spec decoding is not disabled
at/after configured batch size" (https://github.com/vllm-project/vllm/issues/25112). The field
is GONE from today's `SpeculativeConfig` (verified: no `disable_by_batch_size` in
https://docs.vllm.ai/en/stable/api/vllm/config/speculative/).

**Claim (current):** replaced by a K-schedule keyed on concurrency —
`num_speculative_tokens_per_batch_size: list of [start_bs, end_bs, optimal_K]`. Doc rationale is
the acceptance-vs-batch tradeoff in one sentence:
> "SD methods need to verify K tokens for each sequence during decoding. As BS increases, the
> effective BS becomes BS*K which increases the compute requirement during verification. When
> this BS*K goes beyond a critical BS then SD negatively impacts the decode speed (TPOT)."
Example schedule: K=3 for bs 1-64, K=1 for 65-128, K=0 (off) for 129-512.
https://docs.vllm.ai/en/latest/features/speculative_decoding/dynamic_speculative_decoding/
Limitations stated there: "Not compatible with data parallelism ... ranks can pick different K
values, causing DP collective divergence and deadlocks."
Roadmap (issue #39749) lists "Dynamic speculation based on batch size [V1][Spec Decode] Add
Dynamic SD #32374" and "Optimized attention kernels for heterogeneous speculation within a
batch" — i.e. per-request-K kernels are still an open vLLM roadmap item, not shipped.
https://github.com/vllm-project/vllm/issues/39749
Older automation RFC: #4565 "[RFC]: Automate Speculative Decoding"
(https://github.com/vllm-project/vllm/issues/4565); per-sequence K request: #17984
"[Feature]: Per-sequence speculative decoding" — "the current vLLM batch size processing method
can only handle static SL that the K (num_speculative_tokens) set before inference"
(https://github.com/vllm-project/vllm/issues/17984, snippet-verified only).

**Transfer to memra:** a batch-size→K step schedule is trivially cheap (a table lookup at
schedule time) and is what the strongest OSS engine converged on after abandoning a binary
disable; memra should bake K(c) into the admission/scheduler layer from day one, and K=0 rows
must be legal in the verify kernel (they already are in vLLM's geometry: num_draft_tokens=0 →
bonus-only row).

### 1.5 Spec decode + pipeline parallelism (matters: memra is PP-2)

**Claim:** vLLM shipped spec-decode+PP as *incompatible* for its entire v0 and most of v1 life,
and it is still broken for MTP today.
- v0.9.2 docs: "Currently, speculative decoding in vLLM is not compatible with pipeline
  parallelism." https://docs.vllm.ai/en/v0.9.2/features/spec_decode.html
- Latest docs, "Known Feature Incompatibility": "Pipeline parallelism is not composable with
  speculative decoding as of vllm<=0.15.0."
  https://docs.vllm.ai/en/latest/features/speculative_decoding/
- Feature asks: #6911 (2024) "Combine pipeline parallelism with speculative decoding", #10615,
  #14044 ("once PP is enabled, speculative decoding is no longer supported").
- The live RFC #44697 "[RFC]: MTP speculative decoding under pipeline parallelism (PP>1)"
  (2026-06) is the best public analysis of WHY it's hard, and it is exactly the accounting
  problem, quote:
  > "The reason is that the speculative-decode token accounting is computed on the last PP rank
  > only (where the sampler lives) and never makes it to the non-last ranks:
  > num_computed_tokens is advanced optimistically (assuming every draft is accepted) and then
  > corrected after the forward by a GPU kernel that's gated on valid_sampled_token_count_gpu,
  > which only the sampler produces. On the non-last ranks that correction is skipped, so
  > positions over-advance after every rejection. Rope/KV ends up off by one and verification
  > goes wrong."
  Proposed fix: "a typed, width-agnostic broadcast of the sampler's per-request tokens/counts to
  the non-last ranks (pp_spec_broadcast.py)" + non-last-rank input reconstruction + drift
  correction. https://github.com/vllm-project/vllm/issues/44697

**Transfer to memra:** the single hard invariant for PP-2 + batched spec is: *every stage must
apply the same per-session accepted-count vector to its KV/position state*. memra owns both
stages in one process with one CUDA owner thread per device, so the "broadcast" is just an
accept-count tensor handed to stage-0's bookkeeping — structurally far easier than vLLM's
multi-process ranks, but the failure mode (stage-0 KV over-advance after rejection) is identical
and needs an explicit gate test (argmax equality spec-on vs spec-off across PP-2 with forced
rejections).

---

## 2. SGLang

### 2.1 EAGLE-2/3 tree drafting, batched across requests

**Claim:** SGLang batches multiple requests' draft *trees* into a single verify forward. The
per-request tree is linearized to a fixed `num_draft_tokens` slots per request; batch layout is
dense [bs, num_draft_tokens]; the tree topology enters through a custom mask + retrieve arrays
built by a CUDA kernel `build_tree_kernel_efficient` (one program per batch item), and greedy
verify runs as `verify_tree_greedy` (also one program per request).

**Evidence (fetched code, main branch, `python/sglang/srt/speculative/eagle_utils.py`):**
- `build_tree_kernel_efficient(bonus_tokens, parent_list, top_scores_index, draft_tokens,
  seq_lens, seq_lens_sum, topk, spec_steps, num_verify_tokens, ...)` builds, for the whole
  batch at once:
  - `tree_mask`: FULL_MASK shape is flattened
    `seq_lens_sum * num_verify_tokens + num_verify_tokens^2 * bs` — i.e. each request's
    draft-token rows attend to its own prefix (ragged via seq_lens) plus its own qlen x qlen
    tree block. There is also `QLEN_ONLY` (qlen x qlen only) and `QLEN_ONLY_BITPACKING`
    (packed uint8/16/32) — "if use_partial_packed_tree_mask is True, tree_mask:
    num_draft_token (flattened, packed)".
  - `positions`: per draft token, "where each token belongs to. e.g. if depth of each draft
    token is [0, 1, 1, 2] and the prompt length is 7 then, positions = [7, 8, 8, 9]" — tree
    depth → RoPE position, batched.
  - `retrieve_index/retrieve_next_token/retrieve_next_sibling`: [bs, num_verify_tokens] tree
    walk tables consumed by verify.
- `verify_tree_greedy_kernel_triton[grid]` with `grid = (batch_size,)` — one program per
  request walks its own tree and emits `accept_index` [bs, max_tree_depth] and
  `accept_token_num` [bs] — the ragged accept lengths as a dense per-request count vector.
- Non-greedy verify: `tree_speculative_sampling_target_only` /
  `chain_speculative_sampling_triton` (sgl-kernel CUDA ops) with per-request coins; TP-rank
  divergence handled by broadcasting predict/accept from rank 0.
- Files: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/speculative/eagle_utils.py
  (note: `eagle_worker.py` was renamed/split — current tree is `eagle_worker_v2.py`,
  `eagle_worker_common.py`, `eagle_info.py`, `spec_utils.py`; directory listing verified via
  GitHub API 2026-08-09).

**Params** (docs, verified): `--speculative-num-steps` (tree depth of autoregressive drafting),
`--speculative-eagle-topk` (branching per step), `--speculative-num-draft-tokens` ("Maximum
parallel verification capacity ... If topk=1, it is adjusted to num_steps + 1"). Auto defaults:
5/4/8 for Llama/Grok, 3/1/4 for many others.
https://docs.sglang.io/advanced_features/speculative_decoding.html

### 2.2 The KEY artifact: `ragged_verify.py` — per-request verify lengths in CUDA-graph tiers

**Claim:** SGLang now ships an explicit ragged-verify layout: per-request `verify_lens` (>=1,
each request must at least verify the anchor), device `qo_indptr` built by a kernel, total
tokens rounded UP to a CUDA-graph bucket grid (`round_up_grid`), and a
"capped padded variant" for dense consumers (mamba/KDA). This is cross-request ragged verify as
a first-class serving structure, gated behind `SGLANG_RAGGED_VERIFY_MODE`
(static | cap-accept | compact).

**Quotes** (`python/sglang/srt/speculative/ragged_verify.py`, fetched):
> "class RaggedVerifyMode(str, Enum): STATIC = 'static'; CAP_ACCEPT = 'cap-accept';
> COMPACT = 'compact'"
> "every request must verify the anchor (verify_len >= 1)"
> "Per-row upper bound (capped padded variant); rows never exceed it, so dense [bs, cap]
> consumers stay in bounds. None = full-coverage variant."
> `build_ragged_target_verify_geometry`: "cache_seqlens_int32 = (seq_lens +
> layout.verify_lens); cu_seqlens_q = layout.qo_indptr_device; cu_seqlens_k = pad(cumsum(...))"
— i.e. the target verify attention consumes exactly FlashAttention-varlen style
(cu_seqlens_q / cu_seqlens_k) geometry with per-request q lengths.
> `compute_target_verify_graph_key`: graph key = `graph_num_tokens` (bucketed total), asserted
> `<= num_draft_tokens * bs` — ragged verify replaces the dense bs x K graph with a
> total-token-bucketed graph.
https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/speculative/ragged_verify.py

**Transfer to memra:** this is the most directly transferable design in the whole survey:
(a) per-session verify_lens with a hard `>=1` anchor invariant, (b) qo_indptr built on-device,
(c) CUDA-graph capture keyed on a rounded-up total-token bucket rather than (bs, K) — that last
trick is how you keep graph count sane when K varies per request. memra's graph-decode lane
should adopt total-verify-token bucketing if per-session K ever diverges.

### 2.3 Overlap scheduling + spec, and feature limits

- **Overlap scheduler:** "Speculative decoding runs the V2 speculative workers (e.g.
  StandaloneWorkerV2, EAGLEWorkerV2) with the overlap scheduler enabled by default. ... The
  overlap scheduler currently only supports --speculative-eagle-topk 1; set it explicitly. If
  you explicitly set --speculative-eagle-topk > 1, the server will error." — trees and overlap
  don't compose yet; linear (topk=1) drafts do.
  https://docs.sglang.io/advanced_features/speculative_decoding.html
- **Feature incompatibilities** (same doc, method table): DFLASH — "No --enable-dp-attention;
  pp_size == 1; disables overlap scheduler & mixed chunked prefill"; STANDALONE — "does not
  support --enable-dp-attention"; NGRAM — "CUDA-only; no --enable-dp-attention; disables the
  overlap scheduler and mixed chunked prefill". Note the explicit `pp_size == 1` on DFLASH —
  SGLang documents PP incompatibility per-method rather than globally (EAGLE+PP status:
  UNVERIFIED; SGLang's PP support is itself narrow).
- **Radix-cache interactions are a live bug farm:** issue #8726 "Incorrect topk_index in
  Speculative Decoding with RadixCache Enabled" — "any prompt that is a repeat of a previous one
  in the same batch receives corrupted topk_index values"; issue #19796 "Eagle V2 speculative
  decoding crashes with NaN in logits when radix cache prefix hit occurs (SM120 / 8x RTX PRO
  6000 Blackwell)" — "the bug is in Eagle V2 verify path when processing a batch where KV cache
  was partially populated from radix cache prefix"; issue #32459 "EAGLE speculative decoding
  defeats radix prefix reuse for multi-turn traffic ... silent 97%→40-53% reuse collapse".
  (Snippet-verified; #19796 is on memra's exact GPU class.)
  https://github.com/sgl-project/sglang/issues/8726 ,
  https://github.com/sgl-project/sglang/issues/19796 ,
  https://github.com/sgl-project/sglang/issues/32459
- **Adaptive spec:** SGLang has an adaptive speculative decoding mode (docs page
  `adaptive_speculative_decoding`, "Workload acceptance changes over time ... on top of EAGLE
  with --speculative-eagle-topk 1") plus `adaptive_spec_params.py` / `adaptive_runtime_state.py`
  in the tree — per-workload K adaptation shipped, tree-mode excluded.
- **Simulation hooks:** `SIMULATE_ACC_LEN` / `SGLANG_SIMULATE_ACC_TOKEN_MODE` in eagle_utils.py
  — they test batched-verify plumbing under forced accept lengths. Worth copying as a gate
  technique (force ragged accepts deterministically, assert bookkeeping).

**Transfer to memra:** (1) tree verify is a per-request-mask problem, not a cross-request one —
cross-request batching stays a varlen packing problem even with trees; (2) prefix-cache + spec
interactions produced silent correctness bugs in the best OSS implementation, on memra's own GPU
class — any memra KV-reuse feature must be in the spec gate battery; (3) forced-accept-length
simulation is a cheap, high-value test seam.

---

## 3. TensorRT-LLM

### 3.1 Modes and batching geometry

**Claim:** TRT-LLM (current PyTorch backend) supports draft/target, EAGLE-3 (linear chain by
default, optional dynamic tree), NGram, MTP (DeepSeek + "other architectures that ship native
MTP modules (including Step-3.x)"), PARD, DFlash, SA (suffix automaton), and user-provided
drafters. Batching geometry is *uniform-K padded*: every request in the batch carries exactly
`max_draft_len` draft tokens, and speculation cannot be turned off dynamically.

**Quote** (docs/source/features/speculative-decoding.md, main, fetched):
> "For all speculation algorithms, when speculation is enabled, a single sequence of draft
> tokens with length max_draft_len is created for every request. There is currently no way to
> dynamically disable speculation, thus speed ups are only observable at low batch sizes."
https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/speculative-decoding.md

- Dynamic tree mode (EAGLE-3): `use_dynamic_tree`, `dynamic_tree_max_topK`,
  `max_total_draft_tokens` ("Must satisfy max_draft_len <= max_total_draft_tokens <=
  dynamic_tree_max_topK * max_draft_len"); "the dynamic tree CUDA buffers are pre-allocated
  based on the LLM's max_batch_size" — again fixed-capacity per request, batch-preallocated.
  Excluded for "sliding window attention or MLA ... such as DeepSeek and gpt-oss models."
- MTP: `MTPDecodingConfig(max_draft_len=N)`, `num_nextn_predict_layers` "Currently must match
  max_draft_len"; relaxed-acceptance knobs for thinking phase (`use_relaxed_acceptance_for_thinking`,
  `relaxed_topk`, `relaxed_delta`) — a *lossy* acceptance widening unique to TRT-LLM among the
  surveyed engines.
- Legacy (TRT-engine) doc confirms spec + inflight batching shipped there too: "ReDrafter
  supports both Inflight Fused Batching runtime and Python static batching runtime"; Medusa tree
  is "a runtime parameter" with the packed sparse mask ("applying attention with a sparse mask
  that represents the various paths").
  https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/legacy/advanced/speculative-decoding.md
- The underlying batched verify kernel surface is visible through FlashInfer's TRT-LLM-gen
  wrapper: `trtllm_batch_decode_with_kv_cache(..., mask=causal attention mask for xqa
  speculative decoding, max_q_len=...variable-length queries)` — XQA-style decode with q_len>1
  per request is the NVIDIA-lineage verify kernel.
  https://docs.flashinfer.ai/generated/flashinfer.decode.trtllm_batch_decode_with_kv_cache.html
- Spec + PP in TRT-LLM: UNVERIFIED. No doc statement found either way for the PyTorch backend;
  release notes mention "pipeline parallelism with attention DP support" (0.19) but never
  spec+PP together. Treat as unknown, do not cite TRT-LLM as prior art for spec-under-PP.

**Transfer to memra:** TRT-LLM proves uniform-K padded batching is shippable and fast at low
batch — and its own docs concede the cost ("speed ups are only observable at low batch sizes",
no dynamic disable). memra can start uniform-K (simplest kernels, one CUDA graph) but should
keep the vLLM-style K(c) schedule as the pressure valve TRT-LLM lacks.

---

## 4. llama.cpp

### 4.1 Per-slot drafts, shared draft context, parallel drafting since PR #22838

**Claim:** llama-server historically ran speculative decoding per-slot sequentially; since
PR #22838 (merged 2026-05-11, ggerganov, "spec : parallel drafting support") the *draft* model
generates drafts for multiple slots in parallel through one shared `common_speculative` context.

**Quote** (PR #22838 body, fetched via GitHub API):
> "The draft context can generate speculative drafts for multiple sequences in parallel ...
> Single common_speculative context for all slots, capable of handling multiple sequence ids
> ... Extract the drafting logic from server_slot::update_batch() and parallelize it across the
> active slots."
https://github.com/ggml-org/llama.cpp/pull/22838

### 4.2 Cross-request verify: batched submission, per-slot sequential accept — CONFIRMED from source

Verified by reading `tools/server/server-context.cpp` (master, fetched 2026-08-09):
- Draft phase: slots are gathered into `drafting`, params written per slot, then ONE call
  `common_speculative_draft(spec.get())` drafts all slots (the #22838 parallelism).
- Verify submission IS cross-request batched: each generating slot appends its sampled token +
  its `spec_draft` tokens into the SAME `server_batch` (`handle_last_sampled_token`: "add sampled
  token of this slot to the batch, optionally add the speculative draft tokens if any" — slot
  records `spec_i_batch` row indices), and the whole thing decodes "in chunks of params.n_batch"
  through the target model. So the target forward over multiple slots' draft tokens is one
  llama_batch — geometry is flattened rows with per-token (seq_id, pos), the ggml equivalent of
  varlen packing.
- Accept phase is per-slot host-side and sampler-driven, not a batched GPU rejection kernel:
  `iterate(slots, ...)` → `common_sampler_sample_and_accept_n(slot.smpl, slot.ctx_tgt,
  slot.spec_i_batch, slot.spec_draft)` per slot, then per-slot rollback. Rollback on partial
  acceptance can be heavyweight: for context types without partial seq_rm it restores a full
  state checkpoint — "partial acceptance is not supported by the context -> truncate the draft
  and restore the state" (`spec_ckpt.load_tgt(...)`, `slot.prompt.tokens.keep_first(...)`).
- Known limitation left as a thrown error: "TODO @ngxson : it's tricky to make sub-batch
  compatible with common_sampler_sample_and_accept_n, so for now we will throw an error in this
  case: https://github.com/ggml-org/llama.cpp/issues/24840" — a slot's spec rows must not be
  split across sub-batches (ubatch boundary), i.e. their packing has a hard constraint memra
  should note: each session's k+1 verify rows must land in one kernel launch.
- File: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server-context.cpp
  (grep anchors: `handle_last_sampled_token`, `spec_i_batch`,
  `common_sampler_sample_and_accept_n`, `issues/24840`).
- Server flags: `--spec-draft-n-max` (default 3), `--spec-draft-n-min` (default 0),
  `--spec-draft-p-min`; spec types chainable (`--spec-type ngram-mod,mtp`; "If a draft model is
  combined with a draftless decoding the draftless decoding has higher precedence"); `ngram-mod`
  hash pool "is shared across all server slots, so different requests can benefit from each
  other." https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md
- MTP/NextN in llama.cpp: PR #22673 "llama + spec: MTP Support" ("steady-state acceptance of
  around 75% with 3 draft tokens, ... >2x speed-up") — snippet-verified.
  https://github.com/ggml-org/llama.cpp/pull/22673

**Transfer to memra:** llama.cpp is the closest architectural cousin (single node, slots, GGUF)
and it landed cross-request *drafting* + batched verify *submission* only in May 2026, while
keeping accept/rollback per-slot on the host. memra can beat this by doing the accept phase as
one GPU kernel over the packed batch (vLLM/SGLang style) instead of c sequential
host sampler calls, and by making per-session rollback a position-counter update instead of a
state-checkpoint restore.

---

## 5. Papers / geometry for ragged accept

- **"Batch Speculative Decoding Done Right"** (arXiv 2510.22876v2, fetched) — names the core
  problem "the ragged tensor problem: sequences in the same batch accept different numbers of
  draft tokens, desynchronizing position IDs, attention masks, and KV-cache state", claims "all
  existing batch speculative decoding implementations [in the HF-style eager setting] violate"
  output equivalence, formalizes sync invariants, and reports alignment overhead "grows
  superlinearly and consumes up to 40% of computation" in their EQSPEC, mitigated by EXSPEC
  "cross-batch scheduling that dynamically groups same-length sequences" (up to 3x at bs=8).
  Crucially for engine builders it concedes: "Continuous batching systems (e.g., vLLM, SGLang)
  process requests using variable-length packing, sidestepping I1 [contiguous position IDs],
  but still require scatter-gather across requests and rollback both position-ID and KV-cache
  for rejected tokens." Also: "vLLM deprecated this in v1 due to batch expansion memory
  overhead, and SGLang only supports EAGLE-family drafters" [for batch spec with external
  drafts — dated; SGLang added STANDALONE since]. It also dismisses the padding/masking
  alternative: "sequences accumulate padding in various positions (middle and right), forming
  non-contiguous position IDs that standard Transformer implementations handle poorly."
  https://arxiv.org/html/2510.22876v2
- **SmartSpec / goodput** (arXiv 2406.14066) — "SmartSpec dynamically determines [the]
  best speculation length ... based on a new metric called goodput, which characterizes the
  current observed load of the entire system and the speculation accuracy." This is the
  intellectual ancestor of vLLM's dynamic-SD work (vLLM RFC #4565 credits the same authors'
  direction). https://arxiv.org/abs/2406.14066
- **DeepSeek MTP as EAGLE-like head in serving engines:** vLLM — `method: "mtp"`, "MTP is a
  speculative decoding method where the target model includes native multi-token prediction
  capability. Unlike draft-model-based methods, you do not need to provide a separate draft
  model." (https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp/). SGLang — "We
  support MTP ... by using speculative decoding" with the same
  num_steps/topk/num_draft_tokens knobs, i.e. the MTP module is driven through the EAGLE worker
  machinery (docs, §Multi Token Prediction). TRT-LLM — `MTPDecodingConfig`, K MTP modules
  chained (`num_nextn_predict_layers == max_draft_len`). All three treat MTP heads as an
  EAGLE-family drafter behind the same batched verify path — nobody has an MTP-specific verify
  geometry.

---

## 6. Comparison of batching geometries

| Geometry | Who | Batch shape for verify | Ragged accept handling | Kernel need | Cost |
|---|---|---|---|---|---|
| **Batch expansion** | vLLM v0 (dead) | sum_i(k_i+1) single-query rows | contract step maps rows back [bs, k+1] | none beyond plain decode | k+1x rows, k+1x KV reads of same prefix, k+1x sampler metadata; "strictly less efficient than MQA" (vLLM's own docstring) |
| **Varlen multi-query rows (flattened + CSR offsets)** | vLLM v1; SGLang ragged_verify; llama.cpp verify submission (ggml seq_id/pos packing) | one forward, total tokens = sum_i(k_i+1); per-request q_len via cu_seqlens_q / qo_indptr | per-request accept count vector from a [bs]-grid rejection kernel; positions advanced optimistically then corrected (vLLM) or rolled back per slot (llama.cpp) | decode attention with q_len>1 per row (FA varlen / FlashInfer qo_indptr / XQA spec mask) | ~optimal FLOPs; scheduler complexity; CUDA-graph shape churn (SGLang answers with total-token bucket grid) |
| **Tree attention (dense per-request block + custom mask)** | SGLang EAGLE-2/3; TRT-LLM Medusa/EAGLE dynamic tree | dense [bs, num_draft_tokens] rows + per-request tree mask (full: seq_lens_sum*V + V^2*bs; or qlen-only, optionally bitpacked) + positions-by-depth | tree-walk verify kernel (grid=(bs,)) emits accept_index + accept_token_num per request | tree-mask-aware attention + build_tree kernel + tree verify kernel | higher acceptance per draft budget; V^2 mask cost; uniform V per request (capacity padded); overlap-scheduler conflicts (SGLang errors on topk>1) |
| **Uniform-K padded rows** | TRT-LLM (all modes) | [bs, max_draft_len] always, buffers preallocated at max_batch_size | acceptance trims per request downstream; no dynamic disable | fixed-shape kernels, graph-friendly | wasted compute for low-acceptance requests; "speed ups are only observable at low batch sizes" |
| **Same-length grouping (EXSPEC)** | paper only (2510.22876) | group sequences with equal lengths into aligned sub-batches | avoids ragged tensors by scheduling | none special | scheduling latency; "alignment overhead ... up to 40%" when ungrouped |

### Which fits a single-model, 2-GPU PP-2 engine (memra)

**Recommendation: varlen multi-query rows (vLLM-v1/SGLang-ragged geometry), uniform K first,
per-session K later.**

1. memra already has per-request MTP verify (a q_len=K+1 forward for one sequence). The
   cross-request step is packing c of those into one forward with `cu_seqlens_q`-style offsets —
   it composes with the existing decode-batch lane rather than requiring a tree-mask attention
   rewrite. Tree attention buys acceptance rate, not batching; it can layer on later and is
   orthogonal to the cross-request question (SGLang batches trees per-request-dense anyway).
2. Batch expansion is the fallback bridge: it works with today's kernels and is worth having
   only as an oracle path for gating the varlen kernel (same trick as MEMRA_FAST=0), not as the
   shipped path — every engine that had it removed it.
3. Uniform K across the batch (TRT-LLM style) is the right v1: one graph shape per (bucketed
   total tokens), K(c) step schedule at admission (vLLM's `num_speculative_tokens_per_batch_size`
   semantics: K=K0 below c1, smaller above, 0 past the crossover — find memra's crossover on the
   PRO 6000 by measuring, since BS*K vs TPOT is the tradeoff every engine documents).
4. PP-2 specifics: the verify forward spans both stages; the accept decision materializes only
   after stage-1's logits. The one invariant (from vLLM RFC #44697's postmortem): stage-0 must
   apply the same per-session accepted counts before the next microbatch touches its KV. In
   memra's single-process design this is an accept-count vector handed back with the
   pipeline bubble already paying the latency — the correction is bookkeeping, not an extra
   sync. Gate it with forced-rejection argmax-equality across PP-2. Nobody in OSS ships
   verified batched-spec-under-PP today (vLLM: incompatible ≤0.15.0, RFC open; SGLang: DFLASH
   documented pp_size==1, EAGLE+PP unverified; TRT-LLM: unknown) — this is a genuinely open
   corner memra can own.
5. KV/rollback: never checkpoint-restore (llama.cpp's cost); rejected tokens' KV slots are
   dead-on-arrival if attention length is authoritative — accept-count correction of per-session
   write cursors is sufficient (vLLM/SGLang both do this).
6. Sub-batch constraint from llama.cpp issue #24840: a session's K+1 verify rows must not
   straddle a kernel-launch boundary; make the packer enforce it.

### Open uncertainties

- vLLM v1 spec-decode design details live in a Google doc (RFC #14719) I could not fetch;
  the geometry above is reconstructed from shipped code (metadata.py, rejection_sampler.py),
  which is authoritative anyway.
- Which FlashAttention version/PR added spec-verify (decode with q_len>1 + custom mask) —
  UNVERIFIED; FlashInfer's qo_indptr ragged API and trtllm_batch_decode's
  `mask`/`max_q_len` spec params are the verified kernel-surface evidence.
- TRT-LLM spec + pipeline parallelism: no statement found either way (UNVERIFIED).
- SGLang EAGLE + PP: not explicitly documented (only DFLASH carries `pp_size == 1`); SGLang's
  PP support overall is limited, so treat as unsupported until proven.
- vLLM `speculative_disable_by_batch_size` removal commit/PR number — the field's absence from
  current SpeculativeConfig is verified, but I did not chase the exact removal PR.
- SGLang `SGLANG_RAGGED_VERIFY_MODE` default and maturity (static vs compact) — the file is
  shipped on main but I did not verify which mode production defaults to.
- Acceptance-rate-vs-batch-size *measurements* (not mechanisms): the SqueezeBits vLLM-vs-TRT-LLM
  post (#11, Dec 2024) has curves but was only snippet-read; vLLM's dynamic-SD doc gives the
  mechanism ("BS*K beyond a critical BS then SD negatively impacts TPOT") without numbers.
