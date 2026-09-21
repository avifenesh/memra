# Request lifecycle diagnostics

`MEMRA_TTFT_TRACE=1` preserves the existing TTFT line and emits bounded `[request-lifecycle]` JSON for the OpenAI completion and chat-completion routes. It is a diagnostic opt-in; normal requests allocate no trace state when disabled.

An optional `x-memra-trace-id` header carries exactly 32 lowercase hexadecimal characters. This key joins a client attempt to the server-generated request ID even when cancellation occurs before admission has produced response headers. It provides no authentication or request authority. Invalid values are not echoed into the log and make the diagnostic history invalid without changing request handling.

Each record includes the process ID, trace ID, ordered sequence number, elapsed nanoseconds from that trace's start, request/model identity, worker-run generation and dispatch route. Capture tooling must additionally bind the process boot/start identity and the actual source and binary. The worker generation is process-local.

The HTTP wrapper distinguishes a dropped pending response, a dropped SSE body, and normal body EOF. Queue overflow and receiver drop are separate worker observations; neither alone proves a client cancellation. The instrumented main-worker queue and active-abort paths record retirement only after the request's owned state and admission/prefix-pin resources are released. Panic and OOM replay do not produce a clean retirement record. Non-SSE response transport is unchanged.

Actual prime/decode observations are separate from the zero-duration prime timestamps used in legacy cached-prompt TTFT reporting. The quantum API records an actual advance's start and return using the trace's own clock; unknown remaining work stays unknown. Its scheduler call sites remain a separate integration step.

Consumers must reject incomplete bindings, invalid ordering, event suppression, and missing final records. At most 256 ordinary records, one overflow record and one final record are emitted per trace. A structured record does not itself qualify a model or a serving scenario.

Current validation is CPU-only: 18 producer controls and six actual Axum middleware/body controls with a fake backend. Full server/worker checks, independent review and real serving/native qualification remain pending. Constrained-compiler and dedicated DSv4 lifecycles, normal-completion retirement, and unwired quantum paths are not qualified by this change. See [the preserved CPU evidence](../research/serving-release-20260920/lifecycle-producer-20260921/README.md).
