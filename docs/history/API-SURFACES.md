# API-SURFACES history

Records moved out of [docs/API-SURFACES.md](../API-SURFACES.md) on 2026-11-12, verbatim except for the removal
of private deployment detail. The current contract is in the main doc; this file is not.

## `/v1/messages`: Anthropic Messages API

Three rows of the supported-fields table, with the dated notes on what each field did before
(the current rows are under Reasoning controls in the main doc):

| Field | Handling |
|---|---|
| `thinking` | `enabled` → model-native thinking on, `disabled` → off, `adaptive` → the model's own default (Anthropic's "the model decides", which is what that arm does). Any other `type` → 400. **`budget_tokens` → 400**: reasoning tokens are output tokens under the single `max_tokens` budget and no lever can cap a segment, so accepting it would promise a spend cap this server cannot keep (lane/reasoning-schema-20260823; it used to be accepted, never read, never enforced) |
| `output_config.effort` | the SAME canonical set as `reasoning_effort` on the chat surface and `reasoning.effort` on `/v1/responses` (`none`/`minimal`/`low`/`medium`/`high`; `xhigh`/`max`/`ultra` clamp to `high`; Claude Code sends `xhigh` by default on current models): `none`/`minimal` suppress thinking, invalid values 400. When `thinking.type` is also present it wins the on/off switch (the documented Anthropic lever); the effort is still validated and still supplies the level for level-consuming templates. Before issue #31 this field was silently dropped |
| `output_config.*` other than `effort` | → 400 naming the key (previously accepted and ignored) |

## Accounting parity (all three dialects)

One admission path (`surfaces::admit_translated` mirrors `chat_completions` exactly):
canonical model resolution → tenant auth → lane → **`timeout_ms` validation** →
tenant admission / prepaid reservation → ledger receipt (`route` records which dialect:
`/v1/messages`, `/v1/responses`) → rate-limit slot → **deadline-aware backpressure** →
meter line → worker submit → deadline-bounded admission wait.
A request deadline, its 408, its zero-debit ledger outcome and the admission sheds are
therefore identical on all four dialects; see SERVING.md "Request deadlines and the
billing promise" for the contract and the outcome→debit census. Receipt discipline on the response side is the chat
surface's: prompt usage recorded at admission, one completion record per token, terminal
complete/reject synced before the response finishes. Per-tenant capture (when armed)
stores the **translated** internal messages array, the exact prompt the template
rendered, with the same [redacted: product detail] as chat completions.

## Bounded choices on chat and text completions

Responses carry one indexed `choices` entry per row. Streaming may interleave indices; each row has one finish reason. `[DONE]` ends every stream: after all terminal rows and successful parent accounting, or after the error event that ends a failed stream (since 2026-11-06, the same as an OpenAI-shape `n` = 1 stream; until then a failed `n` > 1 stream closed without `[DONE]`). With `stream_options.include_usage=true`, choice chunks carry null usage and one final empty-choice chunk carries the sum. Otherwise the final choices-bearing finish chunk carries aggregate usage. Dropping the group stream closes every child receiver; a worker failure cannot become a successful partial choice list.

Capture, when enabled by the deployment, retains one prompt with n and indexed JSON-line raw completion deltas. Consent/redaction and finalization remain owned by the existing metering implementation. No production billing backend, model support state, compiler/numerical default or release gate is changed by this API capability.

Exact qualification evidence and its fixture limits belong in `research/n-choices-528-20261003/`. A code declaration or CPU result alone is not native model qualification.
