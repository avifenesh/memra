# OpenRouter provider application packet

State: **Q27 + Q35-A3B SUBMITTED 2026-08-12; REVIEW / SLACK CONNECT FOLLOW-UP PENDING**.

Staged 2026-08-13 replacement: **NOT LIVE AND NOT FILED**. The maintenance-window config removes
Qwen3.6-27B, keeps Qwen3.6-35B-A3B, and retains Qwen3.8-27B plus Gemma 4 26B-A4B as non-emitting
plans until their artifacts/measurements exist. The owner performs both the live flip and form
update after the apifix gates.

Use these exact answers for the current 15-field two-model application. Both exact ids are live at
the same base URL and have public protocol/accounting receipts.

| # | Field | Answer |
|---:|---|---|
| 1 | Company / operator | `Avi Fenesh (individual operator; Tiyuvta is not incorporated)` |
| 2 | Website | `https://tiyuvta.ai` |
| 3 | Company email | `hello@tiyuvta.ai` |
| 4 | Display name | `Tiyuvta` |
| 5 | Desired slug | `tiyuvta` |
| 6 | Distinguishing features | `Low latency`; `High throughput`; `Low pricing`; `Unique infrastructure` |
| 7 | Infrastructure / offer | Use the exact narrative below. |
| 8 | `/models` URL | `https://api.tiyuvta.ai/models?schema=openrouter` |
| 9 | API base URL | `https://api.tiyuvta.ai/v1` |
| 10 | Privacy policy | `https://tiyuvta.ai/privacy/` — in force, effective 2026-08-12 |
| 11 | Terms of service | `https://tiyuvta.ai/terms/` — in force, effective 2026-08-12 |
| 12 | Data policy | Use the exact disclosure below. |
| 13 | Output modalities | `Text` |
| 14 | Inference countries | `Canada` — current test endpoint is in Ontario; production region remains pending the cloud-provider hunt. |
| 15 | Headquarters | `Israel` |

## Submitted infrastructure / offer narrative (historical; do not reuse after cutover)

```text
Tiyuvta is an independently operated, first-party OpenAI-compatible endpoint powered by memra's
own native sm_120a Blackwell engine. The current offers are qwen/qwen3.6-27b at $0.28 per 1M input
tokens,
$0.07 per 1M cached input tokens, and $2.69 per 1M output tokens, and
qwen/qwen3.6-35b-a3b at $0.12 per 1M input tokens, $0.03 per 1M cached input tokens, and $1.03 per
1M output tokens. Cached prompt tokens are reported separately and billed at 25% of the ordinary
input-token price.

The launch surface supports streaming and non-streaming Chat Completions, native tools, structured
JSON output, exact prompt/completion/cached-token usage, request ids, bounded prefix caching, and
early 429 overload rejection. The qualification envelope is concurrency 4 per model. Q27 and
Q35-A3B each passed 40/40 required qualification cells and exact standard/serial and cached-token
reconciliation. Each model also passed a fresh 21-check public protocol/accounting gate with zero
failures immediately before this application. The public status page is https://status.tiyuvta.ai/
and probes the API every five minutes.

Qualification receipt:
https://github.com/avifenesh/memra/blob/main/research/requal-20260812/RESULTS.md
```

## Exact data-policy disclosure

```text
Tiyuvta does not durably store prompt or completion bodies and does not use them for training or
evaluation. It retains a content-free billing and settlement ledger containing identifiers,
timestamps, model, token counts, request status, and account/tenant for at most 12 months.
Operational logs contain no prompt or completion bodies and are retained for at most 30 days.
Cloudflare processes connection information at the public edge, and the current rented inference
host processes request content transiently to serve the request. Deletion and privacy requests may
be sent to hello@tiyuvta.ai, subject to the public policy.
```

Do not claim zero processing, HIPAA, ZDR certification, incorporation, a permanent Canadian
production region, a cache discount beyond the listed 25% cached-input rate, or capacity beyond
the evidenced pair envelope.

## Staged maintenance-window narrative (not live or filed)

```text
Tiyuvta is an independently operated, first-party OpenAI-compatible endpoint powered by memra's
native sm_120a Blackwell engine. The staged active offer is qwen/qwen3.6-35b-a3b at $0.0931 per 1M
input tokens, $0.0652 per 1M cached input tokens, and $0.9025 per 1M output tokens. Cached prompt
tokens are reported separately and billed at approximately 70% of the ordinary input-token price
(a 30% cache discount; per-million prices are rounded to four decimal places).

The provider document declares independent 262,144-token prompt and output ceilings, with the
combined request always bounded by the model's 262,144-token trained context. It supports streaming
and non-streaming Chat Completions, native tools, structured JSON output, exact
prompt/completion/cached-token usage, request ids, bounded prefix caching, and clean pre-header 429
admission when a request cannot fit the available KV budget.

Qwen3.6-27B is not offered. qwen/qwen3.8-27b is planned at $0.2745/$0.1922/$2.2800 per 1M
input/cached-input/output tokens when its exact public artifact lands and passes qualification.
google/gemma-4-26b-a4b-it is planned at $0.0665/$0.0466/$0.3230 pending cx-gemlist quality and
capacity numbers. Planned models are not returned by the endpoint and are not routable.
```

## Submitted-packet pricing authorization — historical

On 2026-08-12 the owner ordered: **"price - a slight under market for begining"** and
**"rest do yourself im on remote"**. OpenRouter's live effective-pricing feed was rechecked before
that deployment. Until the owner executes the staged apifix cutover, the public OpenRouter and
OpenModels feeds still carry those historical price triples and `is_ready=true` for both submitted
models. The 2026-08-13 5%-under-market / 70%-cached owner rule is staged above and is not live.

## Submitted-packet technical blockers — historical

**None for the 2026-08-12 packet.** The off-host restore drill passed in `c1f630e72` and is
published on `origin/main`; its
Q27 and Q35 batteries each passed 21/21 with exact accounting while the live endpoint returned
72/72 successful before/during/after monitor requests.

The owner-authorized submission lane rechecked fields 8–12, current `shape=v7` weighted prices,
authenticated streaming/non-streaming usage, tools, structured output, cache accounting, and early
429 behavior immediately before sending. Both models passed fresh 21/21 public gates with exact
usage reconciliation. OpenRouter's live form displayed `Thanks for submitting the form.` and no
confirmation id; the redacted receipt is under
`raw/openrouter-submission-20260812T1553Z/`. The next step is to watch `hello@tiyuvta.ai` for the
review response and Slack Connect invitation.

The replacement packet has complete local apifix evidence. It remains blocked on the owner-run
maintenance cutover, real public-surface verification, and owner form amendment. `cx-3card` still
owns the final measured capacity replacement, but its latest allocation failed before container
creation and produced no scored result; `cx-gemlist` still owns Gemma activation evidence.
