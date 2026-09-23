# Open-output survey: the bound, the charge and the finish reason when a chat request omits its output cap

WP-B day 31, companion to `DAY31.md` (rule R4 reads this file). Read only: every surface below was read from source
at a release tag, and none was built, installed or run. This is a research record of what each surface does when a
Chat Completions request carries neither `max_tokens` nor `max_completion_tokens`. It is not a comparison, carries no
ratio, and states no preference. Retrieved 2026-09-23.

For each surface: (i) the output bound applied when the field is omitted, and any generation-config override; (ii)
how admission reserves or charges KV for such a request, and what happens when KV runs short; (iii) the
`finish_reason` returned at the bound. A claim without a tag or SHA plus file:line (or a dated URL) is marked
**unpinned**.

## Pins

| surface | repository | tag | commit |
|---|---|---|---|
| vLLM | vllm-project/vllm | `v0.30.0` | `ced6857afa0ea7b2e3f0846a62e1394e90f15607` |
| SGLang | sgl-project/sglang | `v0.5.20` | `94602c9c2b7cbdb8efd5c52802dac6a1c180089e` |
| llama.cpp `llama-server` | ggml-org/llama.cpp | `v0.4.1` | `b29c606e28a01b1bc8c1351026a0fa6e616bf6c4` |
| TensorRT-LLM `trtllm-serve` | NVIDIA/TensorRT-LLM | `v1.2.1` (stable) | `376f7e1bd8ed543f75014309e3fd4b237e9b0e73` |
| TensorRT-LLM, pre-release check | NVIDIA/TensorRT-LLM | `v1.3.0rc27` | `6e1cc953c071b8a9055b03ef2ae4ee0bc4c645c4` |
| OpenAI contract (SDK types) | openai/openai-python | `v3.18.0` | `b56e6d89d877309c479cb36fd4f077ce72baf3af` |
| OpenAI contract (reference page) | https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create (redirected from https://platform.openai.com/docs/api-reference/chat/create) | retrieved 2026-09-23 | n/a |
| memra (reference row) | this repo | lane tip | engine source of `2fd5d8d82` (the origin/main `5f1b0eda4` merge); `ee53f7117` and later add research files only |

Tags were resolved with `gh api repos/<repo>/commits/<tag> --jq .sha` on 2026-09-23. File paths below are relative
to each repository root; line numbers are at the pinned commit.

## Table

| surface | (i) bound when omitted | generation-config override | (ii) admission charge for an omitted-cap request | KV shortage while running | (iii) finish at bound |
|---|---|---|---|---|---|
| vLLM v0.30.0 | remaining context, `max_model_len - P` | yes: a `max_new_tokens` in the checkpoint's `generation_config.json` becomes the omitted default (`--generation-config auto`, the default); a folder path or `--override-generation-config max_new_tokens` makes it a server-wide limit | prompt only: the admission gate checks that the prompt's blocks fit; no output tokens are reserved; watermark 0.0 by default | preempt the lowest-priority running request (FCFS: the last one), free its blocks, recompute later | `"length"` |
| SGLang v0.5.20 | remaining context or KV pool, whichever is smaller: `min(max_req_len - P - 1, pool - ceil_page(P) - page - 1)`, `max_req_len = min(context_len - 1, pool - 1)`; optional `SGLANG_MAX_NEW_TOKENS_LIMIT` (unset by default) | no for the output bound: the chat path passes `max_new_tokens` as `None` and does not read it from `generation_config` (other sampling fields do fall back to it) | estimate, decoupled from the stop: a new request is charged `P_extend + min(remaining, 4096) + page`; each running request is charged `min(remaining, 4096) x new_token_ratio`, ratio 0.7 decaying to 0.098 over 600 steps | retract running requests (release, requeue) until decode fits; the last one is aborted with an out-of-memory error | `"length"` |
| llama.cpp v0.4.1 `llama-server` | no token limit (`n_predict = -1`) unless the server sets `-n`; generation stops when the slot's context is full (context shift off by default); slot context = `n_ctx` (default: the trained context) when the KV buffer is unified, which the default auto slot count (4) turns on | no | none: a request waits for a free slot (4 by default); no KV is reserved for its output | on a failed decode: clear idle slots or halve the batch and retry; at batch size 1, every processing slot gets the error "Context size has been exceeded." and is released | `"length"` (internal `limit`) |
| TensorRT-LLM v1.2.1 `trtllm-serve` | remaining sequence, `max_seq_len - P` (v1.3.0rc27: same) | no | full reservation: default policy `GUARANTEED_NO_EVICT` reserves `ceil((P + max_new) / tokens_per_block)` blocks to completion, and for an omitted cap `max_new = max_seq_len - P`, so the whole window | none by construction: an admitted request already holds its blocks to completion; a request that does not fit is not scheduled | `"length"` |
| OpenAI Chat Completions | **unpinned**: the reference page marks the field optional and states no value for the omitted case | n/a | **unpinned** (closed service) | **unpinned** | `"length"` when "the maximum number of tokens specified in the request was reached" |
| memra, door OFF (reference) | served context: `ctx_cap = MEMRA_CTX` (or `P + MEMRA_CTX` when `P + 16 > MEMRA_CTX`), clamped by the model context; budget `ctx_cap - P` | yes, the registry path: a model row with `max_output_length` applies `default_output_length` to an omitted cap, which makes the request bounded | full reservation: the same `ctx_cap` is the admission charge and the allocation (booked `ctx_cap + 64`) | FIFO wait at admission | `"length"` |
| memra, door ON (reference) | `v + 8` generated tokens: `ctx_cap = min(P + v + 8, model_ctx)` and budget `ctx_cap - P` | same registry path | the same `ctx_cap` is charged and allocated: charge and output bound are one number | bounded defer (`MEMRA_ADMIT_DEFER_BUDGET_MS`, 8000 ms), then 429 with `Retry-After` in 1..=60 s | `"length"` |

## vLLM v0.30.0 (`ced6857afa0ea7b2e3f0846a62e1394e90f15607`)

**(i) Bound.** Both fields default to `None`:

```
vllm/entrypoints/openai/chat_completion/protocol.py:222-227
    max_tokens: int | None = Field(
        default=None,
        deprecated="max_tokens is deprecated in favor of "
        "the max_completion_tokens field",
    )
    max_completion_tokens: int | None = None
```

The chat server computes the bound through `get_max_tokens`, with the model's diff sampling parameters and an
override:

```
vllm/entrypoints/openai/chat_completion/serving.py:172-178
        self.default_sampling_params = self.model_config.get_diff_sampling_param()
        mc = self.model_config
        self.override_max_tokens = (
            self.default_sampling_params.get("max_tokens")
            if mc.generation_config not in ("auto", "vllm")
            else getattr(mc, "override_generation_config", {}).get("max_new_tokens")
        )
vllm/entrypoints/openai/chat_completion/serving.py:310-319
            max_tokens = get_max_tokens(
                max_model_len,
                request.max_completion_tokens
                if request.max_completion_tokens is not None
                else request.max_tokens,
                self._extract_prompt_len(engine_input),
                self.default_sampling_params,
                self.override_max_tokens,
                truncate_prompt_tokens=request.truncate_prompt_tokens,
            )
```

```
vllm/entrypoints/serve/utils/api_utils.py:188-205
    model_max_tokens = max_model_len - input_length
    platform_max_tokens = current_platform.get_max_output_tokens(input_length)
    fallback_max_tokens = (
        max_tokens
        if max_tokens is not None
        else default_sampling_params.get("max_tokens")
    )

    return min(
        val
        for val in (
            model_max_tokens,
            fallback_max_tokens,
            override_max_tokens,
            platform_max_tokens,
        )
        if val is not None
    )
vllm/platforms/interface.py:217-218
    def get_max_output_tokens(self, prompt_len: int) -> int:
        return sys.maxsize
```

The generation-config source and the `max_new_tokens` mapping:

```
vllm/config/model.py:330-336
    generation_config: str = "auto"
    """The folder path to the generation config. Defaults to `"auto"`, the
    generation config will be loaded from model path. If set to `"vllm"`, no
    generation config is loaded, vLLM defaults will be used. If set to a folder
    path, the generation config will be loaded from the specified folder path.
    If `max_new_tokens` is specified in generation config, then it sets a
    server-wide limit on the number of output tokens for all requests."""
vllm/config/model.py:1750-1767
        available_params = [
            "repetition_penalty",
            "temperature",
            "top_k",
            "top_p",
            "min_p",
            "max_new_tokens",
        ]
        ...
            # Huggingface definition of max_new_tokens is equivalent
            # to vLLM's max_tokens
            if "max_new_tokens" in diff_sampling_param:
                diff_sampling_param["max_tokens"] = diff_sampling_param.pop(
                    "max_new_tokens"
                )
```

Reading: with the default `generation_config = "auto"`, an omitted cap is `min(max_model_len - P, gc)` where `gc` is
the checkpoint's `generation_config.json` `max_new_tokens` when present, else `max_model_len - P`. The platform term is
`sys.maxsize` on the base platform. How `max_model_len` is derived when not set is not pinned here.

**(ii) Charge.** New requests pass the admission gate with `full_sequence_must_fit`; the gate counts the tokens the
request has now (the prompt), not its future output:

```
vllm/config/scheduler.py:172-183
    scheduler_reserve_full_isl: bool = True
    """If True, the scheduler checks whether the full input sequence length
    fits in the KV cache before admitting a new request, rather than only
    checking the first chunk. Prevents over-admission and KV cache thrashing
    with chunked prefill."""

    watermark: float = Field(default=0.0, ge=0.0, lt=1.0)
    """Fraction of total KV cache blocks to keep free (the watermark) when
    admitting waiting or preempted requests into the running queue. This headroom
    helps avoid frequent KV cache eviction and the resulting repeated preemption
    of requests when GPU memory is scarce. Must be in the range [0.0, 1.0); 0.0
    (the default) disables the watermark."""
vllm/v1/core/sched/scheduler.py:1159-1171
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    ...
                    full_sequence_must_fit=self.scheduler_reserve_full_isl,
                    reserved_blocks=reserved_blocks,
                    has_scheduled_reqs=bool(self.running),
                )
vllm/v1/core/kv_cache_manager.py:508-524
        if full_sequence_must_fit:
            # First check and fail if the full request sequence won't fit.
            full_num_tokens = min(request.num_tokens, self.max_model_len)
            ...
            required_blocks = num_blocks_to_allocate + watermark_blocks
            if required_blocks > self.block_pool.get_num_free_blocks():
                return None
```

`reserved_blocks` is nonzero only for an asynchronous KV load (`scheduler.py:1151-1157`, "An async load holds its
blocks for the whole transfer"), and it covers in-flight prefills, not output.

KV shortage for a running request:

```
vllm/v1/core/sched/scheduler.py:745-753
                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request.
                    if self.policy == SchedulingPolicy.PRIORITY:
                        preempted_req = max(
                            self.running,
                            key=lambda r: (r.priority, r.arrival_time),
                        )
                    else:
                        preempted_req = self.running[-1]
vllm/v1/core/sched/scheduler.py:1493-1497
        self._free_request_blocks(request)
        self.encoder_cache_manager.free(request)
        self._inflight_prefills.discard(request)
        request.status = RequestStatus.PREEMPTED
        request.num_computed_tokens = 0
vllm/config/scheduler.py:141
    policy: SchedulerPolicy = "fcfs"
```

**(iii) Finish.**

```
vllm/v1/core/sched/utils.py:110-114
    if (
        request.num_tokens >= max_model_len
        or request.num_output_tokens >= request.max_tokens
    ):
        request.status = RequestStatus.FINISHED_LENGTH_CAPPED
vllm/v1/request.py:398-402
_FINISHED_REASON_MAP = {
    RequestStatus.FINISHED_STOPPED: FinishReason.STOP,
    RequestStatus.FINISHED_LENGTH_CAPPED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ABORTED: FinishReason.ABORT,
    RequestStatus.FINISHED_IGNORED: FinishReason.LENGTH,
vllm/v1/engine/__init__.py:35
FINISH_REASON_STRINGS = ("stop", "length", "abort", "error", "repetition")
```

## SGLang v0.5.20 (`94602c9c2b7cbdb8efd5c52802dac6a1c180089e`)

**(i) Bound.** Both fields default to `None`, and the chat path forwards `None` for the output cap while other
sampling fields fall back to the model's generation config:

```
python/sglang/srt/entrypoints/openai/protocol.py:861-870
    max_tokens: Optional[int] = Field(
        default=None,
        deprecated="max_tokens is deprecated in favor of the max_completion_tokens field",
        description="The maximum number of tokens that can be generated in the chat completion. ",
    )
    max_completion_tokens: Optional[int] = Field(
        default=None,
        description="The maximum number of completion tokens for a chat completion request, "
        "including visible output tokens and reasoning tokens. Input tokens are not included. ",
    )
python/sglang/srt/entrypoints/openai/protocol.py:1108-1130
        """
        Convert request to sampling parameters.
        Priority: user value > model generation_config > OpenAI defaults
        """

        def get_param(param_name: str):
            value = getattr(self, param_name)
            if value is None:
                return model_generation_config.get(
                    param_name, self._DEFAULT_SAMPLING_PARAMS[param_name]
                )
            return value
        ...
        sampling_params = {
            "temperature": get_param("temperature"),
            "max_new_tokens": self.max_completion_tokens or self.max_tokens,
```

The request dict is merged last over any server-preferred parameters, so its explicit `None` stands:

```
python/sglang/srt/managers/tokenizer_manager.py:1456-1457
        if self.preferred_sampling_params:
            sampling_kwargs = {**self.preferred_sampling_params, **obj.sampling_params}
python/sglang/srt/sampling/sampling_params.py:123
    max_new_tokens: Optional[int] = 128
```

The struct default 128 is therefore not reached from the chat path (reading of the lines above; not exercised). The
scheduler turns `None` into the remaining context or pool:

```
python/sglang/srt/managers/scheduler.py:2510-2542
    def init_req_max_new_tokens(self, req):
        input_len = len(req.origin_input_ids)
        max_new_tokens = (
            req.sampling_params.max_new_tokens
            if req.sampling_params.max_new_tokens is not None
            else 1 << 30
        )
        if self.max_new_tokens_limit is not None and self.max_new_tokens_limit > 0:
            ...
            max_new_tokens = min(max_new_tokens, self.max_new_tokens_limit)
        ...
        paged_input_len = -(-input_len // self.page_size) * self.page_size
        req.sampling_params.max_new_tokens = max(
            0,
            min(
                max_new_tokens,
                self.max_req_len - input_len - 1,
                self.max_total_num_tokens * get_parallel().attn_dcp_size
                - paged_input_len
                - self.page_size
                - 1,
            ),
        )
python/sglang/srt/managers/tp_worker.py:543-547
        max_req_len = min(
            self.model_config.context_len - 1,
            self.model_runner.effective_max_total_num_tokens * self.ps.attn_dcp_size
            - 1,
        )
python/sglang/srt/environ.py:589
    SGLANG_MAX_NEW_TOKENS_LIMIT = EnvInt(None)
```

A caller value above the server context is refused unless auto-truncate is on (`serving_chat.py:1010-1020`,
"max_completion_tokens is too large"); this applies only to a given value.

**(ii) Charge.** The admission estimate is clipped at 4096 output tokens and, by its own comment, does not change the
stop condition:

```
python/sglang/srt/managers/schedule_policy.py:76-81
# This can prevent the server from being too conservative.
# Note that this only clips the estimation in the scheduler but does not change the stop
# condition. The request can still generate tokens until it hits the unclipped max_new_tokens.
CLIP_MAX_NEW_TOKENS = int(
    os.environ.get("SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION", "4096")
)
python/sglang/srt/managers/schedule_policy.py:1357-1373
        max_new = min(
            max(req.sampling_params.max_new_tokens - len(req.output_ids), 0),
            CLIP_MAX_NEW_TOKENS,
        )
        cand_extend_input_len = len(req.full_untruncated_fill_ids) - len(
            req.prefix_indices
        )
        total_tokens = cand_extend_input_len + max_new + self.page_size
        ...
        if total_tokens >= self.rem_total_tokens:
            return AddReqResult.NO_TOKEN
```

Running requests are charged a decaying fraction of the same clipped remainder:

```
python/sglang/srt/managers/schedule_policy.py:613-620
        if running_batch is not None:
            # Estimate the offset in the remaining token space
            self.rem_total_token_offset += sum(
                [
                    self._get_running_request_total_token_offset(r)
                    for r in running_batch.reqs
                ]
            )
python/sglang/srt/managers/schedule_policy.py:710-717
    def _get_running_request_total_token_offset(self, req: Req) -> int:
        return (
            min(
                (req.sampling_params.max_new_tokens - len(req.output_ids)),
                CLIP_MAX_NEW_TOKENS,
            )
            * self.new_token_ratio
        )
python/sglang/srt/environ.py:584-588
    SGLANG_INIT_NEW_TOKEN_RATIO = EnvFloat(0.7)
    SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR = EnvFloat(0.14)
    SGLANG_NEW_TOKEN_RATIO_DECAY_STEPS = EnvInt(600)
    SGLANG_RETRACT_DECODE_STEPS = EnvInt(20)
    SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION = EnvInt(4096)
python/sglang/srt/managers/scheduler_components/new_token_ratio_tracker.py:22-35
        init = min(
            envs.SGLANG_INIT_NEW_TOKEN_RATIO.get()
            * get_schedule().schedule_conservativeness,
            1.0,
        )
        min_ratio = min(
            init * envs.SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR.get(),
            1.0,
        )
        decay = (init - min_ratio) / envs.SGLANG_NEW_TOKEN_RATIO_DECAY_STEPS.get()
        ...
        self.current = max(self.current - self.decay, self.min)
python/sglang/srt/arg_groups/fields/schedule.py:141-144
    schedule_conservativeness: A[
        float,
        "How conservative the schedule policy is. ...",
    ] = 1.0
```

With the default conservativeness 1.0 the ratio starts at 0.7 and floors at 0.7 x 0.14 = 0.098.

KV shortage while decoding:

```
python/sglang/srt/managers/scheduler.py:4111-4124
        if (kv_full_retract_flag := not batch.check_decode_mem()) or (
            ...
            retracted_reqs, new_token_ratio, reqs_to_abort = batch.retract_decode()
python/sglang/srt/managers/schedule_batch.py:3176-3206
        while first_iter or (
            not self.check_decode_mem(selected_indices=sorted_indices)
        ):
            if len(sorted_indices) == 1:
                # Always keep at least one request
                break
            ...
            # release memory and don't insert into the tree because we need the space instantly
            if self.release_req(idx, len(sorted_indices)):
                retracted_reqs.append(req)
python/sglang/srt/managers/schedule_batch.py:3221-3231
        if len(sorted_indices) <= 1 and not self.check_decode_mem(
            selected_indices=sorted_indices
        ):
            # Even the last remaining request cannot fit in memory.
            # Instead of crashing the scheduler, gracefully abort it.
            ...
            last_req.to_finish = FINISH_ABORT(
                "Out of memory even after retracting all other requests "
                "in the decode batch. Aborting the last request.",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
```

**(iii) Finish.**

```
python/sglang/srt/managers/schedule_batch.py:1874-1878
        if len(self.output_ids) >= self.sampling_params.max_new_tokens:
            self.finished_reason = FINISH_LENGTH(
                length=self.sampling_params.max_new_tokens
            )
python/sglang/srt/managers/schedule_batch.py:305-309
    def to_json(self):
        return {
            "type": "length",  # to match OpenAI API's return value
            "length": self.length,
        }
python/sglang/srt/entrypoints/openai/serving_chat.py:1839
                finish_reason_type = finish_reason["type"] if finish_reason else None
```

## llama.cpp v0.4.1 `llama-server` (`b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`)

**(i) Bound.** The request fields alias one parameter whose default is "no limit":

```
tools/server/server-schema.cpp:44-47
    add((new field_num("n_predict", params.n_predict))
        ->set_hard_limits(-1, INT32_MAX)
        ->add_alias("max_completion_tokens")
        ->add_alias("max_tokens")
common/common.h:449-450
    int32_t n_predict             =    -1; // max. number of new tokens to predict, -1 == no limit
    int32_t n_ctx                 =     0; // context size, 0 == context the model was trained with
tools/server/server-context.cpp:1817-1818
        // the per-request limit takes priority over the global one
        slot.n_predict_max = task.params.n_predict != -1 ? task.params.n_predict : params_base.n_predict;
tools/server/server-context.cpp:457-464
    // returns -1 if the generation is limitless
    int32_t n_remaining() const {
        return n_predict_max == -1 ? -1 : n_predict_max - (int32_t) stats.n_gen;
    }

    bool has_budget() const {
        return n_predict_max == -1 || n_remaining() > 0;
    }
```

With context shift off (the default), the slot's context is the bound:

```
common/common.h:571
    bool ctx_shift         = false; // context shift on infinite text generation
tools/server/server-context.cpp:1885-1889
        // if context shifting is disabled, make sure that we don't run out of context
        if (!params_base.ctx_shift && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
            slot.truncated      = true;
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;
```

The slot context under the server defaults (auto slot count, which turns on the unified KV buffer):

```
common/arg.cpp:1399-1400
    } else if (ex == LLAMA_EXAMPLE_SERVER) {
        params.n_parallel = -1;     // auto by default
tools/server/server.cpp:152-156
        if (params.n_parallel < 0) {
            SRV_TRC("%s", "n_parallel is set to auto, using n_parallel = 4 and kv_unified = true\n");

            params.n_parallel = 4;
            params.kv_unified = true;
src/llama-context.cpp:290-294
    if (cparams.kv_unified) {
        cparams.n_ctx_seq = cparams.n_ctx;
    } else {
        cparams.n_ctx_seq = cparams.n_ctx / cparams.n_seq_max;
        cparams.n_ctx_seq = GGML_PAD(cparams.n_ctx_seq, 256);
tools/server/server-context.cpp:4027-4034
    int n_ctx_slot() const {
        int res = llama_n_ctx_seq(ctx_tgt);

        if (params_base.kv_unified_per_slot > 0) {
            res = std::min(res, params_base.kv_unified_per_slot);
        }

        return std::min(res, llama_model_n_ctx_train(model_tgt));
```

Reading: under the defaults each of the 4 slots may grow to the whole `n_ctx`, and all 4 share one buffer of that
size. Whether a later release step (for example an automatic fit of `n_ctx` to free memory) changes `n_ctx` before
this point is not pinned here. The Anthropic `/v1/messages` conversion fills `max_tokens = 4096` when omitted
(`tools/server/server-chat.cpp:571-576`, "required in Anthropic, but we're permissive"); that is not the Chat
Completions path.

**(ii) Charge.** None per request. A request waits for a free slot:

```
tools/server/server-context.cpp:2398-2407
                    server_slot * slot = get_available_slot(task);
                    ...
                    if (slot == nullptr) {
                        // if no slot is available, we defer this task for processing later
                        SRV_DBG("no slot is available, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
```

KV shortage during decode:

```
tools/server/server-context.cpp:3692-3695
                if (n_batch == 1 && ret == 1) {
                    // TODO: try to terminate only the largest active slot/sequence and continue with the rest
                    //       need to remove the tokens from the current batch too
                    err = "Context size has been exceeded.";
tools/server/server-context.cpp:3712-3716
                    for (auto & slot : slots) {
                        if (slot.is_processing()) {
                            send_error(slot, err);
                            slot.release();
tools/server/server-context.cpp:3728-3733
            // retry with half the batch size to try to find a free slot in the KV cache
            if (!try_clear_idle_slots()) {
                n_batch /= 2;
            }

            SRV_WRN("failed to find free space in the KV cache, retrying with smaller batch size, off = %d, n_batch = %d, ret = %d\n", off, n_batch, ret);
```

**(iii) Finish.**

```
tools/server/server-task.cpp:414-425
json server_task_result_cmpl_final::to_json_oaicompat_chat() {
    std::string finish_reason = "length";
    ...
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = msg.tool_calls.empty() ? "stop" : "tool_calls";
    }
tools/server/server-task.cpp:255
        case STOP_TYPE_LIMIT: return "limit";
```

## TensorRT-LLM v1.2.1 `trtllm-serve` (`376f7e1bd8ed543f75014309e3fd4b237e9b0e73`)

**(i) Bound.** One field, `max_tokens` as its alias, default `None`, forwarded as `None`:

```
tensorrt_llm/serve/openai_protocol.py:546-547
    max_completion_tokens: Optional[int] = Field(default=None,
                                                 validation_alias='max_tokens')
tensorrt_llm/serve/openai_protocol.py:655
            max_tokens=self.max_completion_tokens,
tensorrt_llm/sampling_params.py:128,206
        max_tokens (int): The maximum number of tokens to generate. Defaults to 32.
    max_tokens: int = 32
```

The dataclass default 32 is not reached from the OpenAI path, which passes `None` explicitly. The worker fills it:

```
tensorrt_llm/executor/base_worker.py:463-464
            # deduce max_tokens when it's not set by user
            max_tokens = request.sampling_params.max_tokens
tensorrt_llm/executor/base_worker.py:496-508
            splited_prompt_len = int(len(prompt_token_ids) / cp_size)
            default_max_tokens = max_seq_len - splited_prompt_len - query_token_len
            ...
            # default_max_tokens is the biggest available value
            if max_tokens is None:
                return default_max_tokens
tensorrt_llm/executor/base_worker.py:525
                max_tokens=_deduce_max_tokens(
```

`tensorrt_llm/serve/` carries no `generation_config` or `max_new_tokens` read at this tag (searched both files of the
directory). How `max_seq_len` is derived when not set is not pinned here.

**(ii) Charge.** The default capacity policy reserves blocks to completion:

```
tensorrt_llm/llmapi/llm_args.py:1467-1469
    capacity_scheduler_policy: CapacitySchedulerPolicy = Field(
        default=CapacitySchedulerPolicy.GUARANTEED_NO_EVICT,
        description="The capacity scheduler policy to use")
tensorrt_llm/_torch/pyexecutor/_util.py:855-860
    capacity_scheduler = BindCapacityScheduler(
        scheduler_capacity,
        kv_cache_manager.impl if kv_cache_manager is not None else None,
        peft_cache_manager.impl if peft_cache_manager is not None else None,
        scheduler_config.capacity_scheduler_policy,
        two_step_lookahead=mapping.has_pp())
cpp/tensorrt_llm/batch_manager/capacityScheduler.cpp:222-225
    // If a request is already in progress, include it
    // If it's been allocated, it had resource to run to completion
    // Also keep track of blocks needed to drive all in-progress requests to completion
    auto reservedBlocks = kv_cache_manager::NoEvictScheduledBlocksManager(kvCacheManager);
cpp/tensorrt_llm/batch_manager/capacityScheduler.cpp:303-313
                    bool enoughBlocks = reservedBlocks.enoughAvailableBlocks(*req);
                    ...
                    if (enoughBlocks && enoughCrossBlocks && neededPeftPages <= availablePeftPages)
                    {
                        scheduledRequests.emplace_back(req);
                        reservedBlocks.decrementReservedBlocks(*req);
cpp/tensorrt_llm/batch_manager/scheduledBlocksManager.h:44,54
            availableBlocks -= mKvCacheManager.getRemainingBlocksToCompletion(req, windowSize);
                auto const neededBlocks = mKvCacheManager.getRemainingBlocksToCompletion(req, windowSize);
cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp:2267,2285-2287
SizeType32 KVCacheManager::getRemainingBlocksToCompletion(LlmRequest const& req, SizeType32 windowSize) const
    SizeType32 const numTotalBlocksPerBeam = tc::ceilDiv(
        std::min(req.mPromptLen + req.mMaxNewTokens, windowSize + temporaryAttentionWindow) + mSinkBubbleLength,
        getTokensPerBlock());
```

Reading: an omitted cap sets `mMaxNewTokens = max_seq_len - P`, so the reservation is the whole `max_seq_len` window
(bounded by the attention window). The other policies (`MAX_UTILIZATION`, `STATIC_BATCH`,
`tensorrt_llm/llmapi/llm_args.py:1421-1423`) are not read here.

**(iii) Finish.**

```
tensorrt_llm/executor/result.py:342-343
            elif finish_reasons[src_idx] == tllm.FinishReason.LENGTH:
                output.finish_reason = 'length'
```

**Pre-release check, v1.3.0rc27 (`6e1cc953c071b8a9055b03ef2ae4ee0bc4c645c4`).** The same shape:

```
tensorrt_llm/serve/openai_protocol.py:964-965
    max_completion_tokens: Optional[int] = Field(default=None,
                                                 validation_alias='max_tokens')
tensorrt_llm/executor/base_worker.py:461-472
            splited_prompt_len = int(len(prompt_token_ids) / cp_size)
            default_max_tokens = max_seq_len - splited_prompt_len
            ...
            # default_max_tokens is the biggest available value
            if max_tokens is None:
                return default_max_tokens
tensorrt_llm/llmapi/llm_args.py:3667-3669
    capacity_scheduler_policy: CapacitySchedulerPolicy = Field(
        default=CapacitySchedulerPolicy.GUARANTEED_NO_EVICT,
        description="The capacity scheduler policy to use")
```

## OpenAI Chat Completions contract

**(i) Bound.** The SDK types and the reference page describe the field and mark it optional; neither states the value
applied when it is omitted.

```
openai-python v3.18.0, src/openai/types/chat/completion_create_params.py:117-122
    max_completion_tokens: Optional[int]
    """
    An upper bound for the number of tokens that can be generated for a completion,
    including visible output tokens and
    [reasoning tokens](https://developers.openai.com/api/docs/guides/reasoning).
    """
src/openai/types/chat/completion_create_params.py:130-132
    This value is now deprecated in favor of `max_completion_tokens`, and is not
    compatible with
    [o-series models](https://developers.openai.com/api/docs/guides/reasoning).
```

Reference page (URL in Pins, retrieved 2026-09-23): "max_completion_tokens : optional number or null An upper bound
for the number of tokens that can be generated for a completion, including visible output tokens and reasoning
tokens."

The omitted-case bound is **unpinned**. Admission and KV charging are **unpinned** (closed service).

**(iii) Finish.**

```
src/openai/types/chat/chat_completion.py:40-45
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter", "function_call"]
    """The reason the model stopped generating tokens.

    This will be `stop` if the model hit a natural stop point or a provided stop
    sequence, `length` if the maximum number of tokens specified in the request was
    reached, `content_filter` if content was omitted due to a flag from our content
```

## memra reference row (engine source of `2fd5d8d82`)

```
crates/memra-server/src/lib.rs:3428-3431
    /// Omitted (gap-scan F2) => context-bounded (session ctx - prompt, model-capped), the
    /// OpenAI default-when-omitted semantics ...
    #[serde(default, alias = "max_completion_tokens")]
    max_tokens: Option<usize>,
crates/memra-server/src/lib.rs:8031
        max_new: req.max_tokens.unwrap_or(worker::MAX_NEW_CTX_BOUNDED),
crates/memra-server/src/worker.rs:50
pub const MAX_NEW_CTX_BOUNDED: usize = usize::MAX;
```

The registry path (`apply_model_request_limits`, `lib.rs:8897`) replaces an omitted cap when the model row carries
`max_output_length`:

```
crates/memra-server/src/lib.rs:8927-8931
    if let Some(max_output) = max_output {
        if request.params.max_new == worker::MAX_NEW_CTX_BOUNDED {
            request.params.max_new = metadata
                .default_output_length
```

On the naked path the cap, the charge and the allocation are one value:

```
crates/memra-server/src/worker.rs:27139-27157
    let requested = match (max_ctx, max_new) {
        ...
        (None, MAX_NEW_CTX_BOUNDED) => match open_output_tokens {
            // Door ON: charge the output this request will plausibly emit, not the envelope.
            ...
            Some(open) => {
                return crate::admit_memory::charged_ctx_tokens(prompt_len, None, open, model_ctx);
            }
            // Door OFF: with no output bound, use the server context default. ...
            None => {
                let mut cap = server_ctx;
                if prompt_len.saturating_add(16) > cap {
                    cap = prompt_len.saturating_add(server_ctx);
                }
                cap
            }
        },
crates/memra-server/src/worker.rs:27302
    let budget = req.params.max_new.min(ctx_cap - prompt_len);
crates/memra-server/src/admit_memory.rs:132-150
pub(crate) fn charged_ctx_tokens(
    prompt_tokens: usize,
    max_output_bound: Option<usize>,
    open_output_tokens: usize,
    model_ctx: usize,
) -> usize {
    let output = max_output_bound.unwrap_or(open_output_tokens);
    let charged = prompt_tokens
        .saturating_add(output)
        .saturating_add(CTX_SLACK)
        .max(prompt_tokens.saturating_add(CTX_SLACK));
    if model_ctx > 0 {
        charged.min(model_ctx)
    } else {
        charged
    }
}
crates/memra-server/src/admit_memory.rs:288-290
pub(crate) fn clamp_retry_after_s(hint: Option<u64>) -> u64 {
    hint.unwrap_or(RETRY_AFTER_MIN_S * 5)
        .clamp(RETRY_AFTER_MIN_S, RETRY_AFTER_MAX_S)
crates/memra-server/src/lib.rs:4280-4284
fn stop_reason_to_finish(r: &str) -> &'static str {
    match r {
        "Eos" | "Callback" => "stop",
        "MaxNew" | "ContextFull" => "length",
```

`CTX_SLACK` is 8, so door ON allocates `min(P + v + 8, model_ctx)` and the budget is `v + 8` tokens whenever
`P + v + 8 <= model_ctx`.

## R4 (DAY31.md 1.7)

Omitted-cap output bound, per surface:

| surface | bound |
|---|---|
| vLLM | remaining context (`max_model_len - P`), lowered by a checkpoint `generation_config.json` `max_new_tokens` when one is shipped |
| SGLang | remaining context or KV pool, whichever is smaller |
| llama.cpp | remaining slot context (no token limit) |
| TensorRT-LLM | remaining sequence (`max_seq_len - P`) |
| OpenAI | unpinned, no vote |

Four of the four pinnable surfaces bound an omitted cap by the remaining context; none applies a fixed token count
of its own. **R4 = "context"**, which selects `off` for the output bound.

The charge is a separate question and the surfaces split on it: TensorRT-LLM reserves the whole remaining window (the
shape memra door OFF books), SGLang charges a clipped estimate (4096 at admission, a decaying fraction while running)
that it states does not change the stop condition, and vLLM and llama.cpp reserve no output at all and handle
shortage after the fact (preempt and recompute; clear, halve, or error every processing slot). No majority exists
on the charge, and this file picks none.

## Observations for the lead (recorded, nothing changed)

1. memra door ON is the only surveyed surface whose admission charge for an omitted cap is also its output bound.
   SGLang is the only other surface that charges a fixed-size estimate, and it decouples the two explicitly
   (`schedule_policy.py:77-78`).
2. `crates/memra-server/src/admit_memory.rs:48-51` says 8192 "is the `default_output_length` the fleet's registries
   already pin". The registries pin 32768 on every chat row (DAY31.md 1.7 R3). The sentence is stale; this lane does
   not edit it.
3. `crates/memra-server/src/lib.rs:3428-3429` calls the context-bounded omitted cap "the OpenAI default-when-omitted
   semantics". OpenAI publishes no omitted-case value (above), so that attribution is unpinned. The engine behaviour
   it describes matches the four pinnable surfaces.

## Unpinned

- OpenAI: the bound applied when `max_completion_tokens` is omitted; admission and KV charging.
- vLLM `max_model_len`, SGLang `context_len` and TensorRT-LLM `max_seq_len` derivation when not set on the command
  line.
- llama.cpp: any step that resizes `n_ctx` before slot setup at this tag.
- TensorRT-LLM: the `MAX_UTILIZATION` and `STATIC_BATCH` charge paths (not read).
