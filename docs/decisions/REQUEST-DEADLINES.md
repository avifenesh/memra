# Every request has a deadline, and a time cut is not a length cut (2026-08-23)

**Status:** amended. Adopted 2026-08-23: `timeout_ms` on every generation surface, a default
equal to the response mode's maximum, and a zero debit for a request the server fails to answer
in time. Amended 2026-08-26 (an infeasible non-streaming request is refused at admission, and a
deadline that lands mid-generation delivers the tokens produced), 2026-11-01 (the streaming and
non-streaming defaults on unified-memory devices) and 2026-11-02 and 2026-11-06 (an extended stream commits its
response while queued, and as one group when `n` > 1). Code: `TIMEOUT_MS_MIN`/`_MAX`/`_DEFAULT`,
`nonstream_timeout_ms_max`, `unified_nonstream_timeout_ms`, `nonstream_deadline_gate`, `deadline_fitting_max_tokens` and `DEADLINE_INFEASIBLE_MARGIN_PCT` in
`crates/memra-server/src/lib.rs`. Flags: `MEMRA_NONSTREAM_DEADLINE_GATE`,
`MEMRA_STREAM_TTFT_MS_MAX`, `MEMRA_SSE_PREFILL_COMMIT_MS`, `MEMRA_TIMEOUT_MS_MAX`
([docs/FLAGS.md](../FLAGS.md)). The current contract is in [docs/SERVING.md](../SERVING.md).
Related records: [COMPLETE-RESULT-PATH-V1.md](COMPLETE-RESULT-PATH-V1.md) and
[BACKGROUND-DELIVERY-DEFAULT.md](BACKGROUND-DELIVERY-DEFAULT.md) cover work that cannot fit a
synchronous window at all.

The rationale below moved out of the serving doc on 2026-11-12. The old text is kept in
[docs/archive/SERVING-2026-11-12.md](../archive/SERVING-2026-11-12.md#request-deadlines-and-the-billing-promise-timeout_ms)
and its section
["Long non-streaming requests"](../archive/SERVING-2026-11-12.md#long-non-streaming-requests-what-the-deadline-does-now-2026-08-26).

## Chosen

- `timeout_ms` is an integer number of milliseconds on `/v1/chat/completions`, `/v1/completions`,
  `/v1/messages` and `/v1/responses`. Non-streaming accepts 1,000 to 90,000 on a discrete card;
  on a unified-memory device the maximum is derived (below). An out-of-range or non-integer value
  is a named 400 stating the effective range for that response mode. Nothing clamps.
- An absent field takes the response mode's maximum, so every request has a deadline whether or
  not the client knows the parameter exists.
- Non-streaming bounds the complete response. Streaming bounds the time to the first token; after
  it the parameter is spent and the stream runs to completion. Admission wait counts against the
  deadline.
- On a discrete card the 90 s non-streaming maximum reserves margin under a fronting proxy's
  response timeout: a blocking response cannot send headers or a heartbeat before it finishes. The
  proxy's own limit belongs to the deployment.
- Since 2026-11-01, on a unified-memory device (`CU_DEVICE_ATTRIBUTE_INTEGRATED`) the
  non-streaming maximum and default is derived: the streaming first-token default (one full served
  context prefilled, below) plus 8,192 open output tokens at 4 tok/s. That is 3,045,311 ms on GB10
  at 262,144 tokens. `MEMRA_TIMEOUT_MS_MAX` or `MEMRA_SSE_PREFILL_COMMIT_MS=0` restores the
  operator or shipped value, and the `[deadline]` boot line states the result.
- Metering: a request that misses its deadline with nothing delivered settles the zero-debit
  outcome `deadline_exceeded` (HTTP 408 before the response commits). A client that walks away
  after receiving tokens is the existing `abandoned` outcome, debited for what was generated.
- Since 2026-08-26, for non-streaming:
  - a request whose pessimistic estimate needs more than 1.5x its deadline is refused at
    admission with 400 `nonstream_deadline_infeasible`, naming the `max_tokens` that fits and the
    streaming alternative. The estimate prices the prompt at 2,000 tok/s and generation at
    60 tok/s (`MEMRA_PREFILL_FLOOR_TOK_S`, `MEMRA_DECODE_FLOOR_TOK_S`). It costs no slot, no
    receipt and no GPU time. A request with no `max_tokens` is never gated: context is then its
    only limit;
  - a deadline that lands mid-generation on the OpenAI-dialect surfaces (`/v1/completions`,
    `/v1/chat/completions`) answers 200 with the tokens produced, `finish_reason: "error"`,
    `native_finish_reason: "deadline_exceeded"` and an `error` object, under the billable
    outcome `deadline_partial`. Zero tokens still answers 408.
- A streaming deadline above 90 s needs `MEMRA_STREAM_TTFT_MS_MAX` paired with a positive
  `MEMRA_SSE_PREFILL_COMMIT_MS` below it; boot refuses the first without the second. At the
  commit threshold the response commits HTTP 200 and the 5 s SSE comment keepalive starts. A
  timeout or refusal after the commit arrives as the dialect's terminal error event, settles the
  same zero-debit outcome and cancels the worker.

## Rejected

- **Discarding the generated tokens at a non-streaming deadline** (the behaviour before
  2026-08-26). The reported case: a 30,278-token non-streaming request answered 408 after 90.7 s
  with every generated token discarded, while the same work streamed 200 in 93.8 s. The
  partial-delivery half has no flag, because a posture that throws away tokens the server already
  computed is not worth keeping.
- **Refusing only after the deadline.** An infeasible request burned its whole 90 s before the
  408. The admission refusal answers at once.
- **`finish_reason: "length"` for a time cut.** No provider's finish-reason enum has a time value:
  OpenAI, Anthropic, Google and the hosted resellers all mean the token budget by `length` or
  `MAX_TOKENS`. Reporting a deadline as `length` tells a caller to ask for more tokens when it must
  stream.
- **Partial delivery on `/v1/messages` and `/v1/responses`.** Anthropic's `stop_reason` has no time
  value, and OpenAI defines `incomplete_details.reason` as `max_output_tokens` or
  `content_filter` only. Labelling a time cut as either would be the same misreport, so both keep
  the 408. Turning the feasibility gate off is therefore strictly worse on those two surfaces.
- **Raising the non-streaming ceiling on a discrete card.** A higher engine limit does not raise
  the fronting proxy's limit, and a blocking response has no heartbeat. Long work streams, or uses
  background delivery. Unified memory is the exception: there the 90 s refused work the device
  serves (Measurement).
- **A pre-header 408 for a stream still queued at the commit threshold** (until 2026-11-02), and
  the same for an `n` > 1 group whose prefill ran past it (until 2026-11-06). The queue shed then
  judged its wait against the commit threshold instead of the whole first-token deadline. It was
  one candidate for an uncaptured HTTP 429 in the first DGX Spark c=32 agentic cell; the
  body-admission, queue, per-key and vision 429s now log a `[shed]` line so the next one is
  captured.

## Measurement

- **The feasibility boundary.** At the reported 30,278-token prompt the non-streaming ladder ran
  4,096 output tokens in 52.0 s, 5,120 in 61.9 s and 6,144 in 71.5 s (200), and 8,192 in 90.7 s and
  16,384 in 91.5 s (408). The gate's floors and 150% margin are pinned to that ladder by
  `the_feasibility_gate_boundary_matches_the_measured_ladder`: it allows through 6,144 and refuses
  8,192 and above. The raw runs are not in this repository; the
  [`MEMRA_NONSTREAM_DEADLINE_GATE` section](../FLAGS-HISTORY.md#memra_nonstream_deadline_gate) of
  docs/FLAGS-HISTORY.md carries the ladder.
- **The prompt-size estimate.** A bytes/4 proxy read the same 30,278-token prompt as 51,277
  tokens (that text runs about 6.8 characters per token), a 69% over-count in the false-refusal
  direction. The character fallback divides by 6 so it under-counts: an under-count lets a doomed
  request through to partial delivery, an over-count refuses work that would have finished
  (`CHARS_PER_TOKEN_FLOOR`).
- **Unified memory, streaming (2026-11-01).** The first DGX Spark run (Qwen3.8-27B, agentic c=4) answered 408
  to four 14k-42k-token prompts prefilling together inside 90 s. On a device with
  `CU_DEVICE_ATTRIBUTE_INTEGRATED` the streaming default is now the time to prefill one full served
  context at the measured RTX PRO 6000 pp512 rate for the 27B NVFP4 class (4,118 tok/s on 188 SMs),
  scaled by the device's SM count and shared by the four concurrent prefills of the failed cell,
  paired with a 60 s commit: 997,311 ms on GB10 (48 SMs) at 262,144 tokens. Discrete cards keep
  90 s. Arithmetic:
  [`research/gb10-defaults-20261101/`](../../research/gb10-defaults-20261101/).
- **Unified memory, non-streaming (2026-11-01).** The shipped 90 s refused work GB10 serves: one
  131k-token prompt, or four concurrent 32k-token prompts, cannot prefill inside it (32,768 cold
  tokens took 45.2 s alone), and vLLM and SGLang have no such deadline. The derived bound adds a
  generation budget to the streaming value: 8,192 open output tokens (admission's open-output
  charge at the time) at a per-stream decode floor of 4 tok/s, the measured per-user median under
  concurrent prefill in the same agentic c=4 cell (4.59 tok/s) rounded down. That is 2,048,000 ms
  of generation, so 3,045,311 ms at 262,144 tokens and 2,546,656 ms at 131,072
  ([`derive.txt`](../../research/gb10-defaults-20261101/derive.txt)). Pinned by
  `unified_memory_nonstream_deadline_covers_full_prefill_plus_open_generation`.

## What would change it

A provider enum that gains a time value would let the matching surface deliver partials. A
measured ladder that moves the floor rates (a slower card in the served set, a longer context)
re-pins the gate's boundary test.
