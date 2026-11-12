# Model metadata (`MEMRA_MODEL_METADATA`)

This page is the reference for the optional per-model metadata file: its TOML schema, the model
catalog views the server builds from it, and the vendor sampling defaults it can declare. It is
for operators who publish a model list or set per-model request defaults. The operator guide is
[Serving](../SERVING.md); the request schemas are in the [HTTP API reference](../API-SURFACES.md).

## Loading the file

`MEMRA_MODEL_METADATA=/path/models.toml` names one TOML file. Without it the server serves every
model with no metadata, and the catalog views carry only what the loaded model reports.

- The file is read and validated before the GPU worker starts. Any error is a boot FATAL: a TOML
  syntax error, an unknown key (every table is `deny_unknown_fields`), a value outside its range
  below, or a `[models]` alias that is not in `MEMRA_MODELS`.
- The boot log prints the model count and the SHA-256 of the file bytes.
- A deployment binary can reload the file at runtime through `RuntimeHandles::metadata_reload`
  ([deployment binaries](deployment-binary.md#runtime-handles-on_ready)). A reload re-reads the
  boot path, runs the same validation and swaps the set atomically; a failed reload keeps the old
  set, and in-flight requests keep the set they were admitted under. The stock server has no
  reload route.
- Roll out a new binary before a config that uses keys the old binary does not know. Because
  unknown keys are fatal, an older binary refuses to boot on the newer file.

## Schema

The file has three top-level tables:

```toml
[models."<alias>"]          # one table per served alias; the alias must be in MEMRA_MODELS
[planned_models."<alias>"]  # validated at boot, never listed or served
[provider]                  # optional operator identity for GET /v1/models
```

A `[planned_models]` entry uses the same keys and the same validation, and it may not share an
alias with `[models]`. To activate it, move it under `[models]` and load the alias through
`MEMRA_MODELS`.

### Per-model keys

| Key | Value and validation | What it changes |
|---|---|---|
| `description` | string | catalog rows |
| `hugging_face_id` | string | the router feed |
| `created` | integer, Unix seconds | catalog rows; required by the flat feed |
| `owned_by` | string | `/v1/models` `owned_by` (default: the alias up to its first `/`) |
| `quantization` | one of `int4`, `int8`, `fp4`, `mxfp4`, `nvfp4`, `fp6`, `fp8`, `mxfp8`, `fp16`, `bf16`, `fp32` | the router feed |
| `surface` | `chat` (default), `embedding`, `rerank`, `transcription` | the row's type, endpoints, modalities and chat capabilities ([below](#what-the-server-derives)) |
| `input_modalities` | list of `image`, `video` | extra input modalities beyond text |
| `max_prompt_length` | integer > 0 | a rendered prompt longer than this is refused as a context-length error |
| `max_output_length` | integer > 0 | an explicit `max_tokens` above it is a 400 (`param: "max_tokens"`) |
| `default_output_length` | integer > 0, at most `max_output_length`, which it requires | the bound for a request that omits `max_tokens`; without it, `max_output_length` is that bound |
| `default_reasoning_effort` | `none`, `minimal`, `low`, `medium`, `high` | an unset reasoning control resolves as if the client sent this value ([Reasoning controls](../API-SURFACES.md#reasoning-controls)) |
| `default_temperature` and six more sampling keys | see [Vendor sampling defaults](#vendor-sampling-defaults) | omitted sampling fields |
| `non_thinking_sampling` | table, see [Per-mode arms](#per-mode-arms-non_thinking_sampling) | omitted sampling fields on a thinking-off request |
| `is_ready`, `is_free` | boolean | the feeds; required by the flat feed |
| `discount_to_user` | finite number below 1 | the flat feed; required there |
| `openrouter_slug` | non-empty string | the router feed's `openrouter.slug` |
| `datacenters` | list of `{country_code, region?}`; `country_code` is two uppercase ASCII letters | the router feed |
| `zdr`, `hipaa` | boolean | the router feed's `compliance` object |
| `pricing` | table, see below | catalog prices |
| `capacity` | table, see below | the router feed's capacity entries |
| `lifecycle` | table: `status`, `deprecation_at`, `retirement_at`, `replacement_model_id` (RFC 3339 times) | `/v1/models` `lifecycle`; `status` defaults to `active` |
| `reliability` | table: `first_token_timeout_seconds`, `completion_timeout_seconds`, `stream_idle_timeout_seconds`, `capacity_scope` | `/v1/models` `reliability`; defaults 120, 900, 60 and `model_region` |
| `decode_deterministic`, `decode_strategy`, `decode_beam_size`, `decode_temperature_ladder`, `serves_partials` | the speech decode contract | required on `surface = "transcription"` and refused on every other surface; rules in [Speech](../SPEECH.md) |

When both `max_prompt_length` and `max_output_length` are set, a request's `max_ctx` above their
sum plus 8 tokens (capped at the trained context) is a 400, and every catalog view publishes
`min(trained context, max_prompt_length + max_output_length)` as the model's context.

`pricing` takes per-unit USD decimal strings (digits with an optional fraction, never negative):
`prompt`, `cached_prompt`, `cache_write`, `completion` and `internal_reasoning` per token, and
`request` per request. `capacity` takes positive integers: `prompt_tpm`, `cached_prompt_tpm`,
`completion_tpm`, `request_rpm` and `concurrency`. Integers above 2^53 - 1 are refused. An
undeclared price or capacity is omitted from every view; the server never turns an absent price
into `"0"`.

`[provider]` takes `id` (a non-empty slug), `status_url`, `support_contact` and
`incident_contact` (each a URI such as `mailto:...` or `https://...`), and a `regions` list.

### Example

Replace the placeholders; a placeholder left in place fails validation.

```toml
[models."org/model-name"]
description = "An example model served by memra."
quantization = "nvfp4"
max_prompt_length = 131072
max_output_length = 32768
default_output_length = 8192
is_ready = true

[models."org/model-name".pricing]
prompt = "<usd-per-token>"
cached_prompt = "<usd-per-token>"
completion = "<usd-per-token>"

[models."org/embedder-name"]
surface = "embedding"   # served on /v1/embeddings, listed as an embedding model
```

```bash
MEMRA_MODELS="org/model-name=/models/model.gguf,org/embedder-name=/models/embedder.gguf" \
MEMRA_MODEL_METADATA=/etc/memra/models.toml \
memra-server
curl -s http://127.0.0.1:8080/v1/models
```

## Gateway listing surface

Four model-list bodies exist, because each consumer's schema rejects fields the others need.

| Request | Body |
|---|---|
| `GET /models` or `?schema=openai` | `{"object":"list","data":[{"id":"<alias>","object":"model"}]}`, nothing else |
| `GET /v1/models`, `GET /v1/models/{id}` | one row per model: `id`, `name`, `object`, `owned_by`, `type`, `context_length`, `max_output_tokens`, `endpoints`, `input_modalities`, `output_modalities`, `capabilities`, `pricing`, `lifecycle`, `reliability`; the list carries `contract_version: "2.0"` and, when declared, a `provider` block |
| `GET /models?schema=openrouter` | the router feed, schema version 2.4: typed input and output modality objects with their pricing and capacity nested on the modality, `supported_parameters` per model, and only the operator fields the file declares |
| `GET /models?schema=openmodels` | the flat feed: modality arrays, `context_length`, `max_output_length`, `currency: "USD"`, `pricing` (`cached_prompt` appears as `input_cache_read`), `supported_features`, `is_ready`, `is_free`, `discount_to_user` |

Any other `schema` value is a 400 with `param: "schema"`. The flat feed is also a 400 unless every
served model has a loaded context and declares `created`, `max_output_length`,
`pricing.prompt`, `pricing.completion`, `pricing.cached_prompt`, `is_ready`, `is_free` and
`discount_to_user`, so it never publishes a partial document.

`/v1/models` prices are per-million-token decimal strings converted from the per-token strings
by an exact decimal shift (`"0.00000038"` becomes `"0.38"`), `null` when undeclared;
`minimum_request` is the `request` price, `"0"` when undeclared. The `provider` block adds a fixed
error contract next to the declared identity: the `x-request-id` header, 429 for rate limits, 503
for overload, the `Retry-After` header and the account-quota error codes.

### What the server derives

These fields come from the loaded model and the live HTTP surface, never from the file:

- `id` and `name` are the exact `MEMRA_MODELS` alias; `tokenizer` and the trained context come
  from the checkpoint.
- `capabilities` on `/v1/models`: `streaming`, `max_choices`, `tools` (the template has a tools
  branch), `forced_tool_calls` (a constrained tool-call dialect exists), `parallel_tool_calls`,
  `structured_output`, `reasoning` and `prompt_caching`. The DeepSeek-V4-Flash route reports
  `structured_output` and `prompt_caching` as false, because it refuses `response_format` and
  keeps no prefix cache.
- The router feed's `supported_parameters` list the sampling fields, `stop`, `seed` and
  `max_tokens` (with `max_output_length` as its `max`), plus `json_mode` and
  `structured_outputs`, `tools` and `tool_choice`, `reasoning` and `reasoning_effort` only where
  the template serves them. `reasoning` is listed only where an explicit off-request works, and
  the ladder templates list their own rungs (`xhigh`, `medium`, `low` on Qwen3.8; `low`, `high`,
  `max` on GLM-5.3-Flash).
- A non-chat `surface` drops every chat affordance from every view: no streaming, tools,
  reasoning, structured output, generation parameters or output ceiling. It is declared rather
  than inferred, because whether a model can serve embeddings is known only at runtime, after the
  catalog row is built.

## Vendor sampling defaults

A request that omits a sampling field gets the model vendor's published recommendation for that
field when the file declares one, and the API-standard value otherwise
([decision](../decisions/VENDOR-SAMPLING-DEFAULTS.md)). Seven optional keys, validated at boot:

| Key | Range | Off value |
|---|---|---|
| `default_temperature` | `(0, 2]`; zero is refused | none |
| `default_top_p` | `(0, 1]` | 1.0 |
| `default_top_k` | integer >= 0 | 0 (keep all) |
| `default_min_p` | `[0, 1)` | 0.0 |
| `default_presence_penalty` | `[-2, 2]` | 0.0 |
| `default_frequency_penalty` | `[-2, 2]` | 0.0 |
| `default_repetition_penalty` | `> 0` | 1.0 |

- Each key fills exactly one omitted field. A client value always wins, and an explicit
  `temperature: 0` is still greedy.
- `/v1/completions`, `/v1/chat/completions`, `/v1/messages` and `/v1/responses` share one
  resolver (`resolve_sampler_config` in `crates/memra-server/src/lib.rs`), pinned by
  `vendor_sampling_defaults_are_identical_on_every_surface`.
- An undeclared key keeps the API-standard default: temperature 1.0, top_p 1.0, top_k 0, min_p
  0.0, penalties off. Declare nothing where the vendor recommends nothing.
- `default_temperature = 0` is refused because a zero default would make greedy decoding the
  behavior for every client that omits temperature.
- Some model packs carry their vendor's API defaults for temperature and top_p: Step-3.7-Flash
  (`step35`) 0.5 and 0.9, Hy3 0.9 and 1.0. A metadata key outranks the pack default field by
  field.
- Put the vendor citation in a TOML comment next to the numbers, so a later cleanup does not
  "simplify" a deliberate value.

```toml
[models."google/gemma-4-31b-it"]
# Vendor source: the gemma-4-31B-it model card, "Best Practices", "Sampling Parameters".
default_temperature = 1.0
default_top_p = 0.95
default_top_k = 64
# The card recommends nothing for min_p or the penalties: left at the API standard.
```

### Per-mode arms: `non_thinking_sampling`

Some vendors publish one recommendation for thinking mode and another for non-thinking mode. The
flat `default_*` keys are the primary arm; an optional nested table, with the same seven keys
unprefixed and the same ranges, declares the second arm. An empty table is refused.

```toml
[models."qwen/qwen3.8-27b".non_thinking_sampling]
# Vendor source: the Qwen3.8-27B model card, non-thinking mode.
temperature = 0.7
top_p = 0.8
top_k = 20
presence_penalty = 1.5
```

- The request's resolved thinking mode picks the arm (`ModelSamplingDefaults::for_mode`). Every
  spelling that resolves to thinking off takes the non-thinking arm: `reasoning_effort` `none` or
  `minimal`, `enable_thinking: false` in either spelling, `reasoning.enabled: false`,
  `include_reasoning: false`, Anthropic `thinking.type: "disabled"`, a `default_reasoning_effort`
  of `none` on an unset request, and `response_format` on a template whose thinking switch it
  turns off. Thinking on and the template's own default take the primary arm.
- The arms never blend: a field the non-thinking arm leaves out falls to the API-standard default,
  not to the primary arm's value.
- A model without the table uses one arm for every mode. `/v1/completions` has no thinking
  control and always takes the primary arm.

### Performance note

Vendor recommendations that include truncation filters (`top_p` below 1, a positive `top_k`)
move the default request out of the pure-temperature sampling regime. On the MTP route a
filtered request still drafts on the captured sampled graph with the filter inside it
(`MEMRA_SPEC_GRAPH_FILTERED`, on by default); a penalized request drafts on the eager chain,
because a graph cannot carry the per-round penalty history. Exactness does not change on any
route, since verification emits the target's own filtered draw, but the throughput profile of the
default request does. A client can return to the pure-temperature regime with an explicit
`top_p: 1.0` and `top_k: 0`. `vendor_defaults_leave_the_pure_temp_sampled_spec_regime` pins the
regime change for the Qwen3.8 and Gemma 4 vendor numbers.
