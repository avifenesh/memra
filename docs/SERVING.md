# Serving

`memra-server` serves the models named in `MEMRA_MODELS` over an OpenAI-compatible HTTP API. This
guide is for the person who runs it on hardware they own: how to start it, and what each route,
default, limit, refusal and health signal means.

Request and response schemas are in [API surfaces](API-SURFACES.md). Every `MEMRA_*` variable named
here has its row in [Flags](FLAGS.md), and copy-paste commands per model and card are in the
[Cookbook](COOKBOOK.md). The reference pages under [docs/serving/](serving/) hold the detail this
guide summarizes: [prompt cache](serving/prompt-cache.md),
[speculative decoding](serving/speculative.md), [exactness laws](serving/exactness.md),
[metrics](serving/metrics.md), [request log](serving/request-log.md),
[model metadata](serving/model-metadata.md) and [deployment binaries](serving/deployment-binary.md).
A number here belongs to the card it was measured on
([rig labels](PERFORMANCE.md#rigs-what-was-measured-on-what)); the boards are in
[Performance](PERFORMANCE.md). The previous version of this guide, with every lane receipt and
measurement it held, is kept as [a record](archive/SERVING-2026-11-12.md).

## Start a server

```bash
MEMRA_MODELS="chat=/path/to/model.gguf" memra-server

curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"chat","messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

The server binds `127.0.0.1:8080` (`MEMRA_ADDR`). Binding anything else needs an API key
([Authentication and tenants](#authentication-and-tenants)). One process runs one GPU worker
thread; it owns the CUDA context and every loaded model, and the HTTP handlers send it work over a
channel. The batched scheduler is the only scheduler: `MEMRA_SERVE_BATCH`, which used to select
another, is a boot FATAL under any value.

### Models: `MEMRA_MODELS`

`MEMRA_MODELS` is a comma-separated list of `name=source[+draft]` entries. `name` is the served
model id, the `model` field of a request.

| Source | What loads |
|---|---|
| `/path/model.gguf` | a GGUF file |
| `/path/checkpoint_dir` | an HF safetensors checkpoint: `config.json` plus `model.safetensors` or `model.safetensors.index.json`; the chat template comes from the checkpoint's tokenizer config, and a template-less directory answers chat with a 400 that points to `/v1/completions` |
| `/path/repack_dir` | a memra repack directory with a `manifest.json` |
| `hf:owner/repo`, `hf:owner/repo:<file substring>` | a Hugging Face repo, fetched once through the `hf` CLI into `$MEMRA_MODELS_DIR/hf/owner--repo` (default `~/.cache/memra/models/hf/`); a cached spec resolves offline, and the substring picks one GGUF from a multi-quant repo |
| `<source>+/path/draft.gguf` | the model plus a NextN/MTP draft GGUF file (an `hf:` spec works here too) |

Every entry is checked before any weight loads. A missing path, a directory that is not a
checkpoint, a malformed entry, a set but empty value, or a `+draft` that is missing or is not a
file stops the boot with `[server] FATAL: model "<name>": ...`. A wrong drafter path never
degrades to plain decode. Unset, `MEMRA_MODELS` falls back to a hard-coded development pair named
`main` and `judge`, so set it on every other machine.

Several entries serve several models from one process, for example a chat model beside an
embedding model. A model with no trained NextN block loads plain beside a speculating one, even
with `MEMRA_FRSPEC_TRIM` set. Official FP8 block-128 safetensors checkpoints dequantize on the GPU
at load and speculate on the checkpoint's own MTP head
([receipt](../research/fp8ship-20260804/official/), [decision](decisions/SAFETENSORS-DECISION.md)).

A request names a model by its exact id, or by a unique suffix after the last `/`: with
`vendor/model` loaded, `model` resolves to it. An exact id always wins, a request that contains a
`/` is never suffix-matched, and an ambiguous suffix (`a/m` and `b/m` loaded, `m` requested) stays
a 400 `model_not_found`. Responses report the canonical id, and `/v1/models` lists canonical ids
only.

### Drafters

Speculative decoding (`MEMRA_SERVE_SPEC`, on by default; `0` turns it off) runs when a model has a
drafter. The routes, in the order a reader usually meets them:

1. an MTP/NextN head embedded in the GGUF, used with no flags;
2. a per-model `+draft.gguf` in `MEMRA_MODELS`;
3. a global `MEMRA_MTP_DRAFT=<draft.gguf>`; vocabulary-trimmed (FR-Spec) heads come from
   `MEMRA_FRSPEC_TRIM` and are recognized by their `d2t` map;
4. a DSpark/DFlash2 drafter: `MEMRA_DSPARK_SPEC=1` with `MEMRA_DSPARK_DRAFT=<export dir or hf:
   spec>`; it replaces MTP for that model and refuses to combine with PP stages, Step TP/EP, GLM5
   TP, `MEMRA_SPEC_DFLASH` or `MEMRA_GEMMA4_SPEC`;
5. a Gemma 4 assistant drafter for a dense Gemma 4 model other than E4B:
   `MEMRA_DRAFT=<assistant.gguf>`; `MEMRA_GEMMA4_SPEC` defaults to K=5 when `MEMRA_DRAFT` is set
   (`0` turns it off). A request speculates only while no other session is active; sampled
   requests without penalties are admitted (`MEMRA_GEMMA_SPEC_SAMPLED`, on);
6. a GLM-5 MTP block: `MEMRA_GLM5_MTP=1` with `MEMRA_GLM5_SPEC=1`, off by default.

GLM5 TP serving refuses every speculative program. `tools/assert-drafter-attached.sh` checks the
attach line a boot prints. The full attach table: [speculative decoding](serving/speculative.md#drafter-attach). How each model's draft is built: [Draft regime](DRAFT-REGIME.md). How
the server schedules speculation: [Speculative decoding in the server](#speculative-decoding-in-the-server).

### Context, sessions and response shape

- `MEMRA_CTX` sets the served context. Unset, it is the checkpoint's declared context. Zero, a
  non-integer, a value past the engine ceiling, or an undeclared context with `MEMRA_CTX` unset is
  refused at the boundary that reads it; memra never substitutes a window the checkpoint does not
  declare.
- `MEMRA_MAX_SESSIONS` (default 64) caps the interactive lane's resident sessions. Requests past it
  queue ([Admission and backpressure](#admission-and-backpressure)).
- `MEMRA_COMPAT=openai` makes `/v1/completions` answer in OpenAI shape. Any other value keeps the
  native `{text, tokens, stop_reason, ...}` shape, and unset, OpenAI shape follows
  `MEMRA_API_KEY` being set. `/v1/chat/completions` is always OpenAI shape. The example unit sets
  `MEMRA_COMPAT=openai`.
- The decode chunk width is per model: 16 rows where every matmul has a bit-exact 16-row kernel
  class, 32 where a boot probe finds one 32-row wave faster than two 16-row waves (bit-identical
  per row; never under a pipeline split), 15 on GLM-5.3 hyper trunks, 8 otherwise
  ([table](serving/exactness.md#the-exact-16-decode-chunk-tier)). `MEMRA_DECODE_BATCH_CAP` is the
  measurement door.

### Command line

The stock binary takes exactly these arguments; any other token, positionals included, exits 2
naming the token, before any environment read or device work.

```
memra-server --version                 # also -V
memra-server --gen-key <tenant> [--lane interactive|batch] [--rate-limit N] [--keys <path>]
memra-server --revoke-key <key-prefix> [--keys <path>]
```

The key commands are under [API keys](#api-keys). A deployment binary that calls `serve_with`
owns its own command line and may call `memra_server::argv::validate` for the stock set.

<a id="the-build-fingerprint-lanereal-system-fingerprint-2026-09-01"></a>

### The build fingerprint

`system_fingerprint` is `memra-<crate version>-<content id>`, for example
`memra-0.123.0-6371ca8a0af4`: 12 hex characters of an FNV-1a-128 digest over the workspace's
compiled inputs (root `Cargo.toml` and `Cargo.lock`, and the `*.rs`, `*.toml`, `*.cu`, `*.cuh` and
`*.h` files under `crates/`). Two builds of the same source carry the same id, and a history rewrite
does not move it; it is a build identity, not a tamper seal. The git sha is a separate field. No
flag turns the fingerprint off. Why a content id, with no timestamp:
[decision](decisions/BUILD-FINGERPRINT.md).

**Reading it without a GPU.** `memra-server --version` prints the identity and exits before any
engine, GPU or model work, so an artifact can be identified on any machine:

```
$ memra-server --version
memra-server 0.123.0
system_fingerprint memra-0.123.0-6371ca8a0af4
build_id_src source-tree
git_sha <this build's short sha, or unknown>
```

Every boot prints the identity as its first line:
`[server] build: memra-0.123.0-6371ca8a0af4 (id: source-tree, git: <short sha>)`. When the source
tree was unreadable at build time (a vendored or packaged crate), the id falls back to a digest of
the package identity, the build emits `cargo:warning=memra build identity DEGRADED: <reason>`, and
every boot prints `[server] WARNING: build identity is DEGRADED: <reason>`.

### Boot checks and boot lines

- Every `MEMRA_*` name in the environment is audited before anything reads it. A retired door, or
  an unknown name inside a family the code owns, is a boot FATAL naming the ledger; a name outside
  every family warns once. `MEMRA_ENV_AUDIT=warn` downgrades refusals to warnings, `=0` turns the
  audit off. The registry is generated from [Flags](FLAGS.md).
- `MEMRA_REQUEST_LEDGER`, `MEMRA_TENANT_BUDGETS`, `MEMRA_ADMIN_ADDR`, `MEMRA_ADMIN_TOKEN_FILE` and
  `MEMRA_CAPTURE_DIR` are boot FATALs in the stock binary: they configure surfaces only a
  deployment binary carries ([below](#embedding-memra-server-in-a-deployment-binary)).
- Per-card defaults state themselves: `[kv-vmm] door=ON|OFF`, `[admit-oom] batch OOM recovery:`,
  `[admit-mem] door=ON|OFF (<reason>)`, `[prefix-cache] on: ...` with the derived budget, and on
  unified memory `[deadline] unified memory ...`.
- Each serve route prints `[route-contract] model= route= capacity= implemented=[..] refused=[..]`.
  A policy the operator armed that no route in the process honors (`MEMRA_REWRITE_BUNDLE` in a
  DeepSeek-V4-only process) refuses the boot with the refusing route named.
- A live PP split prints `[pp] cross-device transport: stage0=dev0 stage1=dev1`; a config that did
  not split prints nothing of the kind.

### Vision and audio

Images arrive as `image_url` parts: data URIs always, `http(s)` URLs only with
`MEMRA_FETCH_URLS=1` (optionally `MEMRA_FETCH_URLS_ALLOWED_HOSTS`), fetched with SSRF checks on
every resolved address. At most 8 images per request. One vision family loads per process: the
Qwen tower (`MEMRA_VISION_DIR`), Gemma 4 (`MEMRA_GEMMA_VISION=1` with `MEMRA_GEMMA_MMPROJ`), GLM-5.3
(on when the artifact carries `model.visual.*`) or Step-3.7 (`MEMRA_STEP_VISION_DIR`); the rows are
in [Flags](FLAGS.md). The `/v1/audio/*` routes exist, but no speech pack can be resident yet:
a valid transcription or session open answers 503 `engine_unbound`, the session list answers 200
with the operator view, and frames or close answer 404 `unknown_session`, since no session can
open ([Speech](SPEECH.md)).

## Routes

`router()` in `crates/memra-server/src/lib.rs` is the authority; schemas are in
[API surfaces](API-SURFACES.md#routes).

| Method | Path | Auth | Operator note |
|---|---|---|---|
| GET | `/health`, `/livez` | open | inference liveness, one handler ([phases](#worker-phases)) |
| GET | `/readyz` | open | whether to route traffic here |
| GET | `/models` | open | OpenAI-style list; `?schema=openrouter` and `?schema=openmodels` render the [listing feeds](#gateway-listing-surface) |
| GET | `/v1/models`, `/v1/models/{id}` | open | catalog with per-model capabilities; the id may carry a raw or `%2F` slash |
| GET | `/v1/auth/check` | key | 204 when the key is accepted (Bearer or `x-api-key`) |
| POST | `/v1/chat/completions` | key | always OpenAI shape |
| POST | `/v1/completions` | key | native or OpenAI shape, by `MEMRA_COMPAT` |
| POST | `/v1/messages` | key | Anthropic Messages over the chat core (Bearer or `x-api-key`) |
| POST | `/v1/responses` | key | OpenAI Responses, stateless, over the chat core |
| GET, POST | `/v1/responses/{id}`, `/v1/responses/{id}/cancel` | key | [background](#background-delivery-for-long-non-streaming-requests-background-true) poll and cancel |
| GET, POST | `/v1/jobs/{id}`, `/v1/jobs/{id}/cancel` | key | background chat and text jobs |
| POST | `/v1/embeddings`, `/v1/rerank` | key | prefill-only [capture surfaces](#embeddings-and-rerank) |
| POST | `/v1/tokenize`, `/v1/detokenize` | key | CPU only, no GPU work |
| POST | `/v1/audio/transcriptions`, `/v1/audio/sessions` | key | 503 `engine_unbound` after validation: no speech pack is resident |
| GET | `/v1/audio/sessions` | key | 200, the operator view: drive, caps, shed counters |
| POST | `/v1/audio/sessions/{id}/frames`, `/v1/audio/sessions/{id}/close` | key | 404 `unknown_session`, since no session can open |
| GET | `/metrics` | metrics | JSON, or Prometheus text by `Accept` ([metrics](#metrics)) |
| GET | `/yield/metrics` | metrics | per-lane QoS counters and step p50/p99 |

The stock server has no `/admin/*` route; a deployment binary adds its own. Auth reads headers only
and runs before the body is read. Connection limits, fixed in code:

- at most 1,024 concurrent connections; one accepted past that is closed at once, with no response;
- HTTP/1 request headers must arrive within 10 s, at most 64 of them;
- an HTTP/2 connection carries at most 128 concurrent streams;
- a connection takes no new request after 300 s, while a response in flight, a long stream
  included, runs to its end. Clients and proxies reconnect.

## OpenAI compatibility

- **Envelope.** Every OpenAI-shape completion and stream chunk carries `id` (`chatcmpl-...`,
  `cmpl-...`), `created` and `system_fingerprint`; the id is echoed as `x-request-id`. The first
  stream delta carries `role: "assistant"`. Errors are `{"error":{"message","type","param","code"}}`.
  A failure after the 200 is committed arrives as a final `data:` error chunk and `[DONE]`. SSE
  comment keepalives flow every 5 s, so a long prefill is never a silent stream.
- **Native streams.** On a native-shape server (no `MEMRA_COMPAT`, no `MEMRA_API_KEY`) a streaming
  `/v1/completions` ends with named `event: error` and `event: done` events and no `data: [DONE]`.
  The validation harnesses read that shape. An OpenAI SDK pointed at a bare server sees it as a hang;
  set `MEMRA_COMPAT=openai`.
- **Omitted fields.** `temperature` omitted is 1.0 and `seed` omitted draws fresh entropy per
  request; explicit values are honored exactly, `temperature: 0` (greedy) and `seed: 0` included.
  A model's metadata file may declare the vendor's own sampling defaults for omitted fields
  ([below](#vendor-sampling-defaults)). `max_tokens` omitted means a context-bounded budget (served
  context less the prompt), or the open-output charge where memory-shaped admission is the card's
  default ([send `max_tokens`](#config-recommendation-send-max_tokens)).
- **What a `seed` pins.** On plain decode and on the MTP speculative route, a seeded sampled request
  writes one text: the token at generated position p is the draw at Philox event p from the
  target's own filtered row, whatever the draft length, batch companions, resume point or a
  mid-request demotion to plain decode
  ([decision](decisions/SAMPLED-VERIFY-POSITION-KEYED.md), `tools/seeded-invariance-gate.py`).
  Not covered: the DFlash2, GLM5, DSpark and Gemma assistant-drafter routes, whose verify is still
  the rejection walk (a seed pins the text for one serving shape only); and a penalized sampled
  request where the device penalty arm is off (`MEMRA_SERVE_DEVPENALTY`), whose plain tokens draw
  on the host stream.
- **Refused parameters.** `logit_bias`, `logprobs`/`top_logprobs`, `best_of` other than 1 and
  unknown `response_format` types are a 400 naming the parameter. `n` up to 8 is served
  ([multiple choices](API-SURFACES.md#multiple-choices-n)). Cosmetic fields such as `user` are
  accepted. Penalties apply over the last 8,192 tokens on every path. Streams hold back stop-string
  text exactly as non-stream responses do. Up to 16 `stop` strings (1 KiB each, 4 KiB in total)
  and 16 `stop_token_ids` are accepted. The model's end-of-generation tokens (`<|im_end|>`,
  `<end_of_turn>`, `<turn|>` and the rest of its set) stop generation and never reach the client
  as text.
- **Disconnect abort.** A client that hangs up retires its session at the next tick on every serve
  path, and usage counts what was generated to that point (the `[abort]` line records prompt,
  cached and generated tokens). A request still queued or still in prefill when its client leaves
  is dropped before the GPU spends more on it.

**Request body ceiling.** Every inference route accepts bodies up to 192 MiB; a larger one is an
OpenAI-shape 413 before admission. The ceiling is sized from the advertised maxima (a 262,144-token
`prompt_ids` body, 8 base64 images and 2 GIF videos, and envelope headroom). Raise a fronting
proxy's body limit to match, or the proxy becomes the real ceiling. Body reads are admitted
through two bounded pools: bodies declared at 1 MiB or less use a 32-slot pool, larger or
unknown-length bodies a 4-slot pool, so a slow upload cannot block ordinary calls. A body read
gets 90 s plus 2 MiB/s of declared length, capped at 180 s, and fails after 10 s idle. A full pool
answers 429 `body_admission_busy` with `Retry-After: 1`; a closed pool answers 503
`body_admission_unavailable`.

<a id="request-deadlines-and-the-billing-promise-timeout_ms"></a>
## Request deadlines (`timeout_ms`)

Every request has a deadline, whether or not the client sends `timeout_ms`. It is an integer in
milliseconds on the four generation surfaces, embeddings and rerank. Non-streaming accepts
1000 through 90000, default 90000, on a discrete card; on a unified-memory device the maximum and
default are derived (3,045,311 ms on GB10 at 262,144 tokens). Streaming defaults to and caps at
the deployment's first-token ceiling: 90000 unless `MEMRA_STREAM_TTFT_MS_MAX` raises it (it needs
`MEMRA_SSE_PREFILL_COMMIT_MS` beside it, or the boot refuses) or the device has unified memory,
where both are derived ([Unified-memory devices](#unified-memory-devices-gb10-dgx-spark)). Out of
range or not an integer is a 400 stating the range for that response mode; nothing clamps.
`MEMRA_TIMEOUT_MS_MAX` is the non-streaming measurement override.

| | non-streaming | streaming |
|---|---|---|
| what the deadline bounds | the complete response | the time to the first generated token |
| cannot fit before it starts | 400 `nonstream_deadline_infeasible`, naming a `max_tokens` that fits | never gated |
| missed after some tokens | chat and text completions: 200 with the tokens so far, `finish_reason: "error"`, `native_finish_reason: "deadline_exceeded"` and an `error` object (`code: "deadline_exceeded"`); `/v1/messages` and `/v1/responses`: 408, output discarded | cannot happen: the first token spends the deadline |
| missed before any token | 408 `deadline_exceeded`, generation cancelled, no usage | 408 before headers; after a committed stream, the dialect's terminal error event under HTTP 200 |

- **Feasibility gate.** A non-streaming request with an explicit `max_tokens` whose pessimistic
  estimate (prompt at `MEMRA_PREFILL_FLOOR_TOK_S`, default 2,000 tok/s; output at
  `MEMRA_DECODE_FLOOR_TOK_S`, default 60 tok/s) needs more than 1.5 times its remaining deadline is
  refused at once. It takes no slot and runs no GPU work. `MEMRA_NONSTREAM_DEADLINE_GATE=0` lets
  such requests run.
- **Partial delivery** happens only on the OpenAI-dialect surfaces. The Anthropic `stop_reason` and
  the Responses `incomplete_details.reason` have no time value, and `finish_reason: "length"` is not
  used for a time cut, because it would tell the caller to ask for more tokens rather than to
  stream.
- **Streaming commit.** With an extended streaming ceiling, a stream still queued or prefilling at
  `MEMRA_SSE_PREFILL_COMMIT_MS` commits its headers and starts the 5 s comment keepalive, so a
  fronting proxy sees bytes. The deadline keeps running, and a later refusal or timeout arrives as
  the dialect's terminal error under HTTP 200 (an OpenAI `data:` error then `[DONE]`, Responses
  `response.failed`, Anthropic `event: error`). An `n` > 1 stream commits as one group. Keepalive
  comments are not output; a first-token measurement must ignore them.
- Admission wait counts against the deadline. A miss carries no invented `Retry-After`.

Why non-streaming stops at 90 s on a discrete card and partials are delivered:
[decision](decisions/REQUEST-DEADLINES.md);
the delivery contract past the window: [decision](decisions/COMPLETE-RESULT-PATH-V1.md).

### Background delivery for long non-streaming requests (`background: true`)

Background delivery is a per-request opt-in on `/v1/responses`, `/v1/chat/completions` and
`/v1/completions`; there is no server switch ([decision](decisions/BACKGROUND-DELIVERY-DEFAULT.md)).
`background: true` without `stream` returns `status: "queued"` with the request id. Poll
`GET /v1/responses/{id}` or `GET /v1/jobs/{id}`; cancel with the matching `.../cancel` route.
Request and result shapes: [API surfaces](API-SURFACES.md#background-jobs-background).

- A background request keeps normal authentication, capacity, queue and model limits but skips
  the synchronous deadline and the feasibility gate. With `stream: true` it is a 400.
- Poll and cancel resolve the authenticated tenant on every call; another tenant's id is the same
  404 as an unknown one. Cancelling a terminal job is a 409, and polls do not consume a result.
- The stock store is in memory and process-local: a terminal result lives
  `MEMRA_BACKGROUND_JOB_TTL_SECS` (900) under an approximate cap of `MEMRA_BACKGROUND_JOB_MAX_BYTES`
  (64 MiB). Output that cannot fit fails the job visibly (`background_storage_failed`, no usage). A
  restart loses every job.
- A deployment may supply a shared store through `ServerWiring::with_job_store`; a cancel must
  still reach the process that owns the generation
  ([job store](serving/deployment-binary.md#background-job-store-jobstore)).

### Request outcomes

Every request settles its receipt exactly once through the [metering seam](#embedding-memra-server-in-a-deployment-binary).
The stock server attaches no accounting; a deployment binary's implementation sees these
outcomes:

| How the request ended | Receipt call | Usage it carries |
|---|---|---|
| delivered in full | `complete` | all tokens |
| `timeout_ms` cut a non-streaming OpenAI-dialect response, partial delivered | `complete_deadline_partial` | the delivered tokens |
| background job cancelled after tokens | `complete_deadline_partial` | the produced tokens |
| the client left mid-generation | receipt dropped (abandoned) | tokens generated to the abort |
| deadline passed before any token | `settle_unbilled("deadline_exceeded")` | none |
| admission shed | `settle_unbilled("shed_deadline" \| "shed_queue" \| "shed_queue_wait")` | none |
| background job cancelled before tokens | `settle_unbilled("cancelled")` | none |
| background result could not be stored | `settle_unbilled("background_storage_failed")` | none |
| refused or failed, before or during generation | `reject(status, code)` | none |
| still in flight at the drain deadline | `Metering::drain_kill`, then dropped | the implementation settles these as server faults |
| a handler panicked while holding the receipt | dropped while the thread unwinds | none |

A worker failure mid-stream reaches the client as the same error object the blocking path returns,
never as a stream that simply stops.

## Admission and backpressure

<a id="multi-tenant-qos--the-x-lane-slo-gate-laneqos-p95-2026-08-02"></a>

### Lanes (`x-lane`)

The `x-lane` header picks a service class: `interactive` (the default, protected), `judge`
(prefill-shaped) or `harvest` (bulk decode). An unknown value is a 400 `invalid_lane`, and a
batch-class key asking for `interactive` is a 403. Interactive requests queue past
`MEMRA_MAX_SESSIONS`. Judge and harvest admit only while the measured interactive decode-step p99
stays under their share (`MEMRA_SHED_JUDGE` 1.00, `MEMRA_SHED_HARVEST` 0.90) of `MEMRA_SLO_P99_MS`
(default 50 ms), and otherwise answer an immediate 429 with `Retry-After`; the engine never queues
them. They also have their own resident-session caps, `MEMRA_LANE_MAX_JUDGE` (4) and
`MEMRA_LANE_MAX_HARVEST` (8), and shed at them. Inside a tick, interactive decode rows batch first
and dark-lane prefill runs within the measured SLO headroom (`MEMRA_PREFILL_JUDGE` and
`MEMRA_PREFILL_HARVEST`, 256 tokens a tick each). `/yield/metrics` exports the per-lane counters.

`MEMRA_SLO_P99_MS` is the dial between interactive latency and bulk throughput. The gate can only
order a queue it sees: a fronting proxy that holds requests in its own queue hides them from it,
so let engine admission own the queue. Measured on 8 H100 replicas of Qwen3.5-9B Q8_0 (2026-08-02,
N=3): at 25 ms, contended interactive latency was statistically equal to running alone, at 67% less
bulk throughput ([research/qos-p95-20260802/](../research/qos-p95-20260802/)).

### Shedding

At submission, and only while the interactive lane is at capacity, three gates can answer 429 with
the `X-RateLimit` trio and `Retry-After`. An admitted request is never shed.

1. **Deadline:** the estimated queue wait exceeds the request's remaining deadline (`shed_deadline`).
   The estimate is mean tokens per request times the p50 step latency, scaled by the waves of
   queued work ahead: a hint, not a promise.
2. **Queue bound:** `MEMRA_MAX_QUEUE_DEPTH` (default 4 times the lane cap) is reached (`shed_queue`).
3. **Queue-wait ceiling,** opt-in: `MEMRA_QUEUE_WAIT_CEILING_S=N` sheds when the estimate exceeds N
   seconds even for a patient caller (`shed_queue_wait`). Default 0, off.

A per-key `rate_limit` adds a fourth refusal, before worker admission: a tenant already holding
every configured slot gets 429 `rate_limit_exceeded`. Circuit breaking belongs to the router in
front; the engine's part is prompt, honest 429s. Body-admission, queue, per-key and vision 429s log
a `[shed]` line.

### Memory-aware admission

- Admission charges each request its KV ladder (see [send `max_tokens`](#config-recommendation-send-max_tokens))
  and defers it while effective free memory (driver free plus the pool's cached bytes) is below its
  cost plus the reserve, logging `[admit-oom] VRAM defer: ... queueing (FIFO)`. On an idle server
  the refusal follows the request's size. A request that does not fit the server's post-load
  capacity (read once after the boot calibration: the prefix cache empty, nothing else holding
  memory, less what memra itself has come to hold at rest since) is a 400
  `context_length_exceeded` naming the largest context that capacity admits; retrying cannot help
  it (`[admit-oom] capacity reject:`). A request that fits that capacity but not the free memory at
  the time (host memory on unified memory, another process on the card) is a 503 `overloaded` with
  `Retry-After` (`[admit-oom] idle memory short:`). Both messages name the largest context that
  fits the free memory at the time (the 400 when it is under the capacity's figure): memory
  another process holds for good does not come back to a retry, and a caller that cannot wait can
  send that size. The boot line `[admission] idle capacity` states the capacity. Record:
  [`decisions/IDLE-REFUSAL-CLASS.md`](decisions/IDLE-REFUSAL-CLASS.md).
- Speculation-capable models carry a flat 1.5 GiB admission reserve for the draft-graph transient,
  which is not proportional to session count. Plain models carry none.
- A step that hits a quoted CUDA OOM despite admission parks its session at the front of the queue
  and re-primes it, up to `MEMRA_STEP_OOM_RETRIES` (default 3) times, only for a session that has
  emitted nothing, so a streamed prefix is never replayed. Past the budget it is a 503 `overloaded`.
- `MEMRA_ADMIT_DRIVER_HEADROOM_MB` (2048) keeps a driver floor in front of every admission.
- 64 staggered streams with speculation on, on one 24 GB RTX 5090 Laptop, finished well formed in
  3 of 3 runs, peak 23.1 of 24.5 GB (2026-08-06, [receipt](../research/admit-oom-20260806/));
  `tools/serve-stress-gate.sh` holds that, and its `--teeth` arm must invert it.

### Prefill order and fairness

- **A memory-deferred request holds the queue.** When the VRAM gate defers an interactive request
  that would fit once its device's sessions retire and its evictable prefix entries go, later
  requests on that device wait behind it; dark-lane ones shed. Refusals that need no memory (slot
  capacity, template, the first-token gate) still answer at once. An interactive request in a
  settle wait (below) holds the queue the same way for that pass. `/metrics`
  `admission_vram_hol_holds` counts the held requests.
- **A DFlash request that does not fit with its drafter is served plain.** A speculating request is
  charged its session, its draft state and the reserve. On the MTP route a request that fits only
  without the draft state is admitted and its draft-graph capture is refused later
  (`[admission] eager-arm admit:`). On the DFlash route the draft state is the drafter KV plane,
  allocated with the drafter session, so the gate refuses the request its drafter
  (`[admission] drafter refused: ... arm A`) and prices it again in the same pass as a plain
  request, in its queue position: beside peers when the charge without the plane fits or when no
  retirement can fund the full charge, and on an idle card, after its drained re-read, whenever
  the full charge still does not fit. The refusal holds for the request's life. A drafter session
  exists only for a request whose plane the gate charged, so a step-OOM replay runs plain too. A
  greedy request writes the same text either way; a seeded sampled one writes its plain stream
  (memra#1154). `/metrics` `admission_drafter_refusals` counts them. Record:
  [`decisions/DRAFTER-ADMISSION-REFUSAL.md`](decisions/DRAFTER-ADMISSION-REFUSAL.md).
- **A session admitted without its state yet keeps its charge booked.** A DFlash, gemma-spec or
  GLM cold prime builds its caches and drafter plane in the tick after `admit`, so until then the
  live reading does not show them. Every later reading is reduced by the charge such a session
  was admitted on, until the tick builds it; before this, a second arrival in the same pass (or a
  step-OOM replay's pass) was admitted on the same bytes (memra#1255). An arrival that the
  booking alone keeps out (it does not fit the booked reading but fits the reading without it)
  waits for the next pass (`[admission] settle wait:`): no reclaim, eager admission, drafter
  refusal or capacity reject is decided on a reading the tick is about to change, and later
  requests on its devices wait behind it for that pass as they do behind a VRAM defer. An
  arrival that fits neither goes on to those verdicts at once, on the booked reading.
  `[admission] settled:` names the passes and milliseconds it waited, printed once `admit` has
  built the session; `/metrics` `admission_settle_waits` and `admission_settle_wait_ms` count
  them. Single-device deployments only (`is_multi_device_deployment`). Record:
  [`decisions/ADMISSION-UNBUILT-SESSION-BOOKING.md`](decisions/ADMISSION-UNBUILT-SESSION-BOOKING.md).
- **A prefill at risk goes first.** Prefilling sessions normally advance one chunk per tick in turn.
  When a session would miss its first-token deadline under that order but would make it served
  alone, the tick advances sessions in earliest-deadline order until one chunk of tokens is used.
  `[prefill-order]` logs each switch and `prefill_deadline_ticks` counts the ticks.
- **The prefix cache yields the transient reserve.** Before a decode phase, a discrete card whose
  cache published since the last check evicts the oldest unleased entries when effective free
  memory falls below the reserve admission charges (`[prefix-cache] reserve: ...`,
  `prefix_headroom_evictions`).
- **A long prime never holds the tick.** The prime walker (`MEMRA_PRIME_YIELD`, on) advances one
  chunk (`MEMRA_PRIME_CHUNK`, 4096) per tick and gives each admitted peer one bounded quantum (a
  prime chunk, a committed spec round or a decode step) in between; the bytes are the same either
  way. The serial plain prime is bounded by `MEMRA_PREFILL_TICK` (1024 tokens), widened to 8192 for
  a sole fresh interactive request. Boot prints `[prime-walk] model=... route=... supported=true|false`
  for each speculative route, naming the terms that decided it; the DeepSeek-V4 route primes each
  request in one call ([decision](decisions/PRIME-FAIRNESS-DEFAULT.md)).
- **A cache hit goes before cold prefill.** A full-prompt hit takes its first decode before
  unrelated cold prefill.
- **A short hit suffix primes first, bounded.** A resumed row (a prefix-cache hit, a pool or
  affinity resume) whose remaining suffix fits one prime call primes it first, and cold prefill
  waits one tick. Fitting means at least `PRIME_MIN_T` tokens, at most the tick budget and no
  capture boundary inside (`hit_suffix_fits_one_call`). This hit-suffix fence covers plain
  interactive rows only: not speculative or GLM-5 routes, vision, multi-choice requests, eager-only models or a
  run with the confidence trace on.

  The fence is bounded. Once its consecutive deferred ticks add up to one prefill chunk's time at
  the measured rate, it stands down for a tick and the suffixes prime beside the cold chunks
  (`hit_suffix_fence_yields`). A steady stream of hits therefore never holds cold prefill back
  longer than one chunk's time plus one fenced tick. The fence also stands down while the
  deadline-aware prefill order is on, and on a tick where a whole-prompt hit waits for its first
  decode ([decision](decisions/PREFIX-PROMPT-END-TAIL.md)).
- **A freed slot gets a refill grace.** When an interactive completion frees a slot while cold
  prefill remains, `MEMRA_PRIME_BATCH_HOLD_MS` (4 ms) gives the replacement a short window to join
  the next prime call.

Why the hold, the deadline order and the reserve yield, measured at c=8 on a 24 GB card:
[decision](decisions/C8-SERVING-24GB.md); the workspace charge on the RTX 5090 Laptop GPU:
[decision](decisions/PREFILL-WORKSPACE-HELD.md).

### Rate-limit headers

The generation surfaces, tokenize and detokenize send `x-ratelimit-limit`, `-remaining` and
`-reset` (lowercase on the wire; key a case-sensitive parser on that). They have concurrency-slot
meaning: the binding cap, the free slots (the tighter of the tenant and lane gauges), and the
estimated seconds until a slot frees. A slot rides a stream to its end, so `remaining` is true for
the whole stream. With no traffic signal, `reset` falls back to `MEMRA_RL_RESET_S` (default 2).
`remaining: 0` does not mean the request waits: the [sheds](#shedding) above may answer 429.

### Config recommendation: send `max_tokens`

Admission sizes each session from the request's own bound. An explicit `max_ctx` is authoritative;
otherwise a request with `max_tokens` is charged `prompt_tokens + max_tokens + 8`, and only a
request that omits `max_tokens` falls back to the full `MEMRA_CTX` envelope. Where memory-shaped
admission is the card's default (below), an open request is charged
`prompt_tokens + 32768 + 8` instead (`MEMRA_ADMIT_OPEN_OUTPUT_TOKENS`), never more than the
envelope, and that charge less the prompt and 8 rows is its output budget. At `MEMRA_CTX=32768` the
fallback stranded 6.3% of a 96 GB card at c=16 and 12.6% at c=32 on a 9B model, while right-sized
requests stranded about 0% ([receipt](../research/serving-density-20260806/VERDICT.md)). Send an
explicit `max_tokens` from clients and serve configs, and size `MEMRA_CTX` for the unbounded
requests you expect, not the maximum.

<a id="serving-defaults-keyed-to-the-rtx-pro-6000-workstation-edition-2026-11-04"></a>

### Serving defaults keyed to the RTX PRO 6000 Workstation Edition

The full-power RTX PRO 6000 Blackwell Workstation Edition carries single-card receipts other cards
do not, so these mechanisms are its defaults (`HardwareTarget::is_pro6000_workstation`; the Max-Q
and Server editions share the class, not the receipts). A multi-card deployment of that card keeps
the on-demand planes off and memory-shaped admission unarmed (`worker::is_multi_device_deployment`).
None has an environment switch; each states itself at boot.

- **On-demand KV planes.** Full-attention K/V planes of a covered session reserve a fixed virtual
  range and map granules as rows grow; admission subtracts the reserved-but-unbacked bytes. Boot:
  `[kv-vmm] door=ON ... (pro6000-ws-receipt)`; elsewhere `door=OFF`
  ([decision](decisions/KV-VMM-SERVING-DEFAULT.md)).
- **Batched-chunk OOM retry.** A batched decode chunk whose CUDA OOM came before any session-state
  write takes one reclaim rung and runs once more, so each request keeps one numeric program. Since
  2026-11-08 the RTX 5090 Laptop GPU has it too. Boot: `[admit-oom] batch OOM recovery: ON|OFF`
  ([decision](decisions/BATCH-OOM-RECOVER-DEFAULT.md)).
- **Memory-shaped admission.** The open-output charge above; the reclaim demotes device prefix
  entries to the host tier before dropping them; a memory defer past `MEMRA_ADMIT_DEFER_BUDGET_MS`
  (8 s) answers 429 with `Retry-After`; headroom readings book what priming sessions still owe. Boot:
  `[admit-mem] door=ON|OFF (<reason>)` ([decision](decisions/ADMIT-BY-MEMORY-DEFAULT.md),
  [workspace release](decisions/ADMIT-W-RELEASE-DEFAULT.md)).
- **Direct handoff export.** The host-tier handoff export writes with `O_DIRECT` and `fdatasync`;
  other cards write through the page cache. Same bytes; the export line says `io=direct` or
  `io=buffered` ([decision](decisions/HANDOFF-DIRECT-EXPORT.md)).

## Unified-memory devices (GB10, DGX Spark)

A unified-memory device shares system DRAM between the GPU and the host. memra detects it from
`CU_DEVICE_ATTRIBUTE_INTEGRATED`, never from the device name, and only these defaults take a
unified arm; discrete cards read none of them. GB10 values and the measurements behind each row:
[DGX Spark](rigs/dgx-spark.md); arithmetic:
[research/gb10-defaults-20261101/derive.py](../research/gb10-defaults-20261101/derive.py).

| Default | Discrete card | Unified memory |
|---|---|---|
| free memory for sizing (admission and its driver-headroom rung, prefix-cache boot budget, MoE residency, the pre-capture guard, the pooled MTP and DFlash2 verify widths) | `cuMemGetInfo` free | the larger of `cuMemGetInfo` free and `/proc/meminfo` MemAvailable |
| free memory for a delta (both ends of the per-session draft-state measurement) | `cuMemGetInfo` free | `MemFree + Buffers + Cached + SReclaimable` from one `/proc/meminfo` read: page-cache filling and reclaim move pages between its terms, so they cancel in it ([UNIFIED-MEMORY-READERS](decisions/UNIFIED-MEMORY-READERS.md)) |
| `MEMRA_PREFIX_CACHE_MB` | the larger of two full-context entries and boot free less the 1.5 GiB reserve, the driver headroom and the session peak, clamped to boot free less the reserve | the larger of two entries and half of (boot free less the reserve, the driver headroom and any host KV tier) |
| `MEMRA_STREAM_TTFT_MS_MAX` | 90,000 ms | one full served context at the derived prefill rate, at least 90,000 ms |
| `MEMRA_SSE_PREFILL_COMMIT_MS` | off | 60,000 ms |
| non-streaming `timeout_ms` maximum and default | 90,000 ms | the streaming value plus 8,192 open output tokens at 4 tok/s |
| prefix-cache insert budget | the configured budget | clamped on every insert to what the cache holds plus allocatable free memory less the driver headroom and the reserve; an entry that copies (a snapshot, a DFlash or GLM-5 boundary publication with its drafter tail, a host promotion) reserves that room before it allocates |
| admission booking of cache growth | none, except after a partial admission reclaim | the cache's allowed growth is booked as committed while a session is active; an idle server books none |

Of the readers memra#1307 went through, three stay on raw `cuMemGetInfo` free on unified memory, where
it can only read lower: the 256 MiB graph-launch floor, the bound a failed draft-graph capture
records as its appetite (a proven bound; an estimate there would stick as an over-charge), and the
DeepSeek-V4 route's memory door, which loads only on two discrete cards
([unified_memory.rs](../crates/memra-engine/src/unified_memory.rs)). The deepseek-v4.1 route reads
raw free memory in its own code (`dsv41_gpu`: `device_memory`, `device_mem`); it loads on one
device, and only its size keeps it off a GB10 today.
The driver headroom's 2 GiB floor, the crash guards, the async pool's release threshold, the host KV tier (it
adds copies, not capacity, on shared DRAM; armed, the derived device share subtracts it), the
pinned spill budget and the MoE slot cache keep their discrete values. Rollback: an explicit
`MEMRA_PREFIX_CACHE_MB` is authoritative; `MEMRA_SSE_PREFILL_COMMIT_MS=0` restores the 90 s
windows; an explicit `MEMRA_STREAM_TTFT_MS_MAX` or `MEMRA_TIMEOUT_MS_MAX` takes over its ceiling.
The `[prefix-cache] on:` and `[deadline] unified memory` boot lines state each value.

<a id="prompt-caching-cross-request-prefix-cache--2026-08-02"></a>

## Prompt caching

Three tiers serve prompt tokens without recomputing them. Every counter, refusal line and
per-device default: [prompt cache](serving/prompt-cache.md).

1. **Continuation pools** (`MEMRA_KV_REUSE`, on): a retiring session parks its whole state, and a
   prompt that exactly extends it resumes. `MEMRA_REUSE_POOL` (2) entries per `(model, namespace)`
   and pool, at most `MEMRA_REUSE_POOL_GLOBAL_CAP` (16) across all of them.
2. **Cross-request prefix cache** (`MEMRA_PREFIX_CACHE_MB`; `0` turns it off): device snapshots of
   primed state, keyed by the exact token-id prefix within a model and namespace, reusable by any
   number of sessions; entries start at 64 tokens. Request 1 seeds its prompt, request 2 primes
   split at the longest common prefix and inserts it, request 3 onward hit. Unset, the budget is
   derived at boot: at least two full served-context entries of the largest model, clamped to
   post-load free memory less the 1.5 GiB serving-transient reserve, and on a discrete card the
   larger of that and what the session peak leaves ([decision](decisions/PREFIX-CACHE-DISCRETE.md)).
   PP host bounce turns it off.
3. **Session affinity** ([below](#session-affinity)) resumes a conversation the client rewrote.

**Exactness.** An entry stores the bytes of whichever prime program ran, so a hit is bit-identical
to the run that computed the prefix ([receipt](../research/prompt-cache-20260802/gate-exact.jsonl)).
Against a cold run of another prime program it inherits the
[first-token near-tie class](#first-token-cross-config-drift).

**Capture and eviction law.** Every boundary the server chooses lands on the
[prime grid](#the-prime-grid-law). An entry for a prompt of P tokens is `capture_len(P)` long: P
when P is a multiple of 32, else the largest multiple of 32 below P that leaves at least 16 tokens
behind it. A grid-entry hit reports `cached_tokens == capture_len(P_previous)`. The entry also
carries the prompt-end tail of the prompt that published it. While that tail is resident, an exact
re-send of an off-grid prompt restores all P tokens and reports `cached_tokens == P`
([detail](serving/prompt-cache.md#prompt-end-tails)). A prompt with no on-grid entry above the floor publishes nothing (`[prefix-cache] seed REFUSED (grid)`), and a recurrent entry that
would leave an off-grid suffix is skipped at the hit site; both are counted. Eviction is plain
global LRU: the oldest unleased entry goes first and an entry is never its own victim, so the newest
turn of a growing conversation always fits ([decision](decisions/PREFIX-CACHE-POLICY.md)). Sessions
win over cache residency: a failed session allocation evicts every unleased entry and retries.

**Plain prefix-hit leases** end after final carrier selection and the root and TP-peer restore
streams, before deeper publication, so the previous turn's entry is evictable when the next turn
publishes ([detail](serving/prompt-cache.md#plain-prefix-hit-leases)).

<a id="cache-salt-isolation"></a>

**Per-tenant isolation (`cache_salt`).** Every reuse tier keys on `(model, cache namespace)`. The
namespace is the optional `cache_salt` body field: requests share cached prefixes only with the
same salt, so `cached_tokens` reflects only the caller's own namespace. With a keyring the namespace
becomes `t:<tenant>` plus `\x1f` plus the salt, so tenants never share and a salt cannot forge
another tenant. No salt is the default `""` namespace; the byte budget stays global. A gateway that
multiplexes many end users through one key should set a salt per end user and bound its fanout.

**Speculative pools compare samplers.** A resume whose sampler differs from the parked session's is
refused on every probe tier and on the DSpark pool, and serves cold:
`[worker] spec-reuse REFUSED: parked session's sampler differs on <field>; serving COLD spec ...`,
counted as `spec_pool_sampler_refusals`. `seed` is not compared: the sampled draft graph is already
re-keyed on it, and comparing it would refuse every seed-omitting conversation.

**Accounting.** `usage.prompt_tokens_details.cached_tokens` counts prompt tokens resumed from any
tier. Tools requests cache like any other request, keyed on the rendered prompt's token ids.

<a id="session-affinity--resuming-a-rewritten-conversation-lanesession-affinity-2026-08-05"></a>

### Session affinity

Agent clients often rewrite history instead of extending it (stripping `<think>` blocks from prior
turns), so neither prefix tier matches. Affinity asks whether this is the same conversation:

- **Identity nominates.** Explicit: `session_id` or `user` in the body, or the `x-session-id`
  header, in that order (`metadata.user_id` on Messages, `prompt_cache_key` on Responses).
  Implicit: a fingerprint of the prompt's control-token segments, which needs three shared
  segments.
- **Bytes decide.** The session resumes only if the new prompt reproduces its committed tokens up
  to its checkpoint, at the last turn-marker control token on the plain and speculative tiers
  alike; otherwise it declines (`history diverged at N of checkpoint M`) and primes in full.
- Scoped per `(model, cache namespace)`; constrained requests never resume. `MEMRA_AFFINITY=0`
  turns it off ([detail](serving/prompt-cache.md#session-affinity)).

### Host tier and tenant lifecycle

- `MEMRA_KV_HOST_MB` (default 0, off) adds a pinned host-RAM tier behind the device cache: an
  evicted entry demotes verbatim and a later exact match restores through the normal path. It is
  clamped to MemAvailable times 0.6. `MEMRA_KV_HOST_TENANT_PCT` (50) caps one tenant's share; set
  100 on a single-tenant server.
- On key revocation or tenant deletion, a deployment calls `PurgeHandle::purge_tenant(tenant)`: it
  removes the tenant's host entries across all its salts and its unleased device entries, and
  reports leased ones as `device_pinned_left` instead of aborting admitted requests.
- On the RTX 5090 class and the full-power RTX PRO 6000 Workstation Edition, a parked plain-pool
  session compacts to its committed length ([decision](decisions/KV-PARK-COMPACT-DEFAULT.md)).

<a id="measured-numbers-qwen35-9b-q8_0-receipts-in-research"></a>

## Speculative decoding in the server

Speculation is a low-concurrency latency tier. Speculative sessions run their bursts outside
batched plain decode (several on one model share one verify forward per round as a pooled cohort),
and plain batching overtakes them between c=2 and c=4. Mechanism, counters and refusal lines:
[speculative decoding](serving/speculative.md).

- **Admission gate** (`MEMRA_SPEC_GATE`, on). Two deciders share it. The concurrency band admits a
  request to speculation while the projected wave (live sessions, this request, the queue and the
  requests still in flight from the HTTP layer) is at most `LOW` (2 on one card), and demotes live
  speculative sessions into batched plain decode at `active >= HIGH` (4); in between, a live session
  keeps speculating and a new arrival serves batched. The cost model (`MEMRA_SPEC_GATE_MODEL`, on)
  measures verify and plain step costs for each MTP-served model at boot (`[spec-cost]` lines) and,
  once the model has 64 rounds of acceptance, picks the K (0 is plain) with the highest predicted
  aggregate tok/s at the projected concurrency, for admission and demotion alike. The band decides
  instead for an operator pin, the PP-2 default, explicit `MEMRA_SPEC_GATE_LOW` or `_HIGH`,
  DSpark-served models and models with no verdict yet. On a sharded cross-device PP-2 placement the
  band is 0 and 1: no request enters speculation. `MEMRA_SPEC_GATE=0` speculates always;
  `MEMRA_SPEC_GATE_MODEL=0` restores the band everywhere.
- **Demotion** is one-way per session and carries the session's cache, so a demoted greedy stream
  is byte-identical to one batched from the start, and a sampled one keeps its Philox position.
  GLM5 sessions demote greedy only; constrained and Gemma assistant-drafter sessions do not demote.
  On an RTX 5090 Laptop with Qwen3.5-9B (N=5 interleaved, 2026-08-06) the band-gated server measured
  251.2 tok/s at c=1-2 and 504.7 tok/s at c=4-8, tracking the faster arm at each width
  ([receipt](../research/spec-gate-20260806/), [decision](decisions/SPEC-ADMISSION-GATE.md)).
- **Depth.** The automatic policy picks K per request: K=3 for cold prompts; K=2 for a cold prompt
  of 1,024 or more tokens on the RTX 5090 Laptop GPU and the RTX PRO 6000 Max-Q (by device name),
  K=3 there on every other card ([decision](decisions/SPEC-K-COLD-LONG-K2.md),
  [rejected short rung](decisions/SPEC-K-COLD-SHORT-5090.md)); K=2 for a greedy request that
  resumes at least 1,024 cached tokens (K=5 with a trimmed head). A sampled request takes its cold
  twin's rung. The cost model may then pick a lower K. `MEMRA_SPEC_K` pins K for every request
  (`0` is plain) and turns off the gate, the cost model and demotion; each rung has its own
  override.
- **Cache hits.** A whole-entry prefix hit re-arms a warm speculative session, greedy or sampled.
  Where the cost model owns the gate, a sampled restore is admitted exactly when the model would
  speculate at the current demand; with no verdict it is admitted only when the request is alone
  (`MEMRA_SPEC_RESTORE_LOAD_GUARD`; the refusal line is
  `[spec-restore-guard] sampled restore REFUSED: ...`). Any hit that cannot convert serves plain,
  and the downgrade line names the reason.
- **Bursts and streaming.** A burst runs up to `MEMRA_SPEC_BURST` (32) tokens. Text streams at
  every round commit. A waiting admission ends the burst at the next round boundary
  (`MEMRA_ADMIT_YIELD`, on); a session yields at a burst's entry with nothing emitted at most once
  in a row ([decision](decisions/SPEC-ENTRY-YIELD-BOUND.md)); the burst ends at the round that
  commits EOS or a stop id, and when the client is gone; a pooled member closes at the round its
  stream ended ([decision](decisions/POOLED-MEMBER-CLOSE.md)). A sampled burst carries its pending
  token into the next one ([decision](decisions/SAMPLED-PENDING-CARRY.md)).
- **Constrained requests** speculate with draft-side grammar masking, always on: drafts are legal
  by construction.
- **Telemetry.** Each speculative response carries `usage.spec` (its own rounds, drafted, accepted
  and acceptance rate). The operator `/metrics` view adds the cumulative per-model `spec` block and
  the rolling 30 s `spec_tau` and `spec_accept_by_position` ([metrics](serving/metrics.md)).

On a shared-prefix serving shape the prefix cache can be worth more than speculation: one RTX PRO
6000 measured 8.50 req/s with speculation off against 2.14 with it on at c=16
([Performance](PERFORMANCE.md#speculative-decoding-and-the-prefix-cache)).

## Structured output, tools and reasoning

Request fields are in API surfaces:
[structured output](API-SURFACES.md#structured-output-response_format),
[tool calls](API-SURFACES.md#tool-calls) and [reasoning controls](API-SURFACES.md#reasoning-controls).

### Constrained decoding (`response_format`)

`response_format` `{"type":"json_object"}` and `{"type":"json_schema",...}` compile to an
[llguidance](https://github.com/guidance-ai/llguidance) grammar. Each step's token mask bans
disallowed tokens on the device, ahead of the same sampling, graph and speculative paths
unconstrained sessions use.

- **Compile isolation.** Each model owns a bounded eight-job CPU compile queue, so compilation stays
  off the scheduler tick. Schemas over 512 KiB, 64 JSON levels or 32,768 JSON values are a 400
  before admission. A compile has 5 s, then a retryable overloaded-class error. While four abandoned
  compile workers are still running, constrained compilation for that model fails closed
  (`[constraint] ... compiler fail-closed`, `/metrics` `constraint_compiler_fail_closed`); it rearms
  when they finish (`compiler rearmed`)
  ([watchdog](../research/sec5-20260811/RESULTS.md), [cap](../research/sec6-20260811/RESULTS.md)).
- **Cost.** Plain constrained greedy ran at 99.4% of unconstrained, and the speculative lane at 79%
  (Qwen3.5-9B NVFP4 on an RTX 5090 Laptop, N=3, 2026-08-03): the gap is draft acceptance under a
  tight grammar, not mask overhead ([receipt](../research/constrained-full-20260803/)).
- **Thinking.** On a template with a think switch, a constrained request forces it off. On a
  think-forced template with a token-level close (Step-3.7's `</think>`), the request decodes in two
  phases: thinking runs unconstrained with end-of-generation banned, and the grammar clamps every
  token from the close on; such requests never speculate. `MEMRA_POSTTHINK_CEILING=<tokens>` closes
  a think that never ends. A budget, stop or context bound that ends generation inside the think
  phase is a 400 naming the field, never a 200 with empty content.
- `MEMRA_CONSTRAIN_HOST=1` is the host-side oracle arm.

<a id="openai-tools-surface-serve-tools-lane-2026-08-02"></a>

### Tool calls

Tool schemas render through each model's own template tools branch, reproduced byte for byte from
the vendor template; a template without a supported tools branch rejects `tools` with a 400 at
admission. Emitted tool-call blocks parse into OpenAI `tool_calls` (streaming deltas and
`message.tool_calls`, deterministic ids, `finish_reason: "tool_calls"`), with argument values
coerced to the declared JSON-schema types. A block that does not parse is returned verbatim as
content, never dropped. `tool_choice` `"required"` or a named function forces a call through a
tool-call grammar on Qwen and Gemma templates; `parallel_tool_calls: false` limits the reply to
one call; the DSML, Hy3 and GLM dialects refuse forced and single-call forms by name. `/v1/models`
reports `forced_tool_calls` and `parallel_tool_calls` per template
([receipt](../research/tool-choice-20261003/)).

**The isolation law.** Non-tools traffic bypasses the tools renderer and the tool-call parser, so
its streams are byte-identical to a server without tools support; tools traffic is generation-identical
for the same rendered prompt. One exception: a template that carries the Qwen3.8 reasoning-effort
ladder routes every chat through the tools-capable renderer, because only that renderer writes the
vendor's default rung.

### Reasoning controls

One effort table serves every surface: `none` and `minimal` mean off, `low`, `medium` and `high`
mean on at that level, `xhigh`, `max` and `ultra` map to the highest rung the template
distinguishes, and anything else is a 400. Absent, the model's own default applies, or the
`default_reasoning_effort` its metadata entry declares. Level strings reach only templates that
read one (probed from the template, never from the family name); on a binary-switch template a
graded level means reasoning on, rendered byte-identically to `reasoning: {"enabled": true}`. The
mapping table, per template class:

| Template | Native control | Absent | `none`, `minimal` | `low` | `medium` | `high` | `xhigh`, `max`, `ultra` |
|---|---|---|---|---|---|---|---|
| Qwen3.5/3.6 class | `enable_thinking` | on | off | on | on | on | on |
| Qwen3.8 | `enable_thinking` plus an effort sentence | `xhigh` | off | `low` | `medium` (no sentence) | `xhigh` | `xhigh` |
| Ornith | `enable_thinking`, no ladder | on | off | on | on | on | on |
| Gemma 4 | `enable_thinking`, default off | off | off | on | on | on | on |
| Hy3 | `reasoning_effort` `no_think`, `low`, `high` | `no_think` | `no_think` | `low` | `low` | `high` | `high` |
| Step-3.7-Flash | `Reasoning: <level>` line, think tail always open | no line | 400 | `low` | `medium` | `high` | `high` |
| GLM-5.3-Flash | `Reasoning Effort: Low\|High\|Max` line, think tail always open | `Max` | 400 | `Low` | `High` | `High` | `Max` |

- **`minimal` is off**, a deliberate divergence from Qwen's hosted API, which maps it to a short
  reasoning rung: this server's no-reasoning side is real. On a template that cannot turn reasoning
  off, a client-explicit `none` or `minimal` is a 400 (`cannot disable reasoning`); an operator
  `default_reasoning_effort = "none"` never refuses a caller who sent nothing.
- **Qwen3.8's unset case renders the vendor's own `xhigh` default.** A deployment that wants the
  older no-instruction bytes declares `default_reasoning_effort = "medium"`, the vendor's
  zero-steering rung.
- **Every reasoning field this server cannot act on is a named 400, on every surface**: unknown
  `reasoning` keys (`max_tokens` included), wrong types, two switches that disagree,
  `chat_template_kwargs` keys other than `enable_thinking` and `preserve_thinking: true` (the
  vendor default; `false` is refused), `thinking.budget_tokens` on `/v1/messages`.
- **Reasoning is always delivered.** `include_reasoning: false` and `reasoning.exclude: true` turn
  reasoning off rather than hide it. Reasoning text arrives as `message.reasoning`
  (`delta.reasoning`), a Messages `thinking` block, or a Responses `reasoning` item. `max_tokens` is
  one budget for reasoning and content, so a small budget on a thinking request can end
  `finish_reason: "length"` with empty content.
- Gemma 4 `<|channel>thought ... <channel|>` blocks route to reasoning too; a stream holds an opening
  for at most 256 bytes to resolve it ([decision](decisions/GEMMA-UNOPENED-THOUGHT.md)).
- Assistant history may carry `reasoning` (alias `reasoning_content`); the Qwen3.8 ladder template
  replays it inside a `<think>` block, as its vendor template does by default.

Why the table reads this way: [decision](decisions/REASONING-CONTROL.md).

<a id="embeddings-and-rerank--the-capture-surfaces-laneembed-serve-2026-08-26"></a>

## Embeddings and rerank

`POST /v1/embeddings` returns the last-token hidden state after the final norm, L2-normalized, with
optional Matryoshka truncation (`dimensions`); up to 32 inputs per request. `POST /v1/rerank` judges
each document with the Qwen3-Reranker prompt and scores P("yes") over the yes/no logit pair; up to
64 documents, and a model whose "yes" and "no" are not single tokens is refused. Shapes:
[API surfaces](API-SURFACES.md#embeddings-and-rerank).

No decode step runs. Each input is one admitted worker request with the usual auth, lanes, deadlines
and rate limits, its tokens counted as prompt usage under the metering id `<x-request-id>.<index>`
(one `[meter] admit` line per item). Capture requests bypass every cross-request reuse tier (a hit
would skip the prime they read), never speculate, and prime alone. Prompts at or above the 16-token
prime floor pool from the prime's hidden stack; shorter ones walk the tokenwise path on the same
numeric program.

Serve these on batch-class keys, so they ride the harvest lane and shed under interactive load. A
model card that serves only capture declares `surface = "embedding"` or `"rerank"` in its metadata,
so the catalogs stop advertising chat affordances ([model metadata](serving/model-metadata.md)).

## Multi-GPU

Which shape applies depends on whether the model fits one card. Placement policy:
[decision](decisions/PRO6000-MULTICARD.md); workload guide: [Multi-GPU](workloads/multi-gpu.md).

### Replica fleet

A model that fits one card scales as N independent `memra-server` processes, one per GPU
(`CUDA_VISIBLE_DEVICES` places each), behind a reverse proxy that routes on `/readyz` and restarts
on `/health`. Each replica is a single supervised instance under the
[supervision contract](#the-supervision-contract-systemd). The server accepts bodies up to
192 MiB; give the proxy the same limit, or the proxy becomes the real ceiling.
Replica measurements: [Performance](PERFORMANCE.md#batched-decode-replicas-and-endurance).

| Tool | What it does |
|---|---|
| `tools/serve-fleet.sh start\|stop\|status\|restart` | Replica-fleet supervisor without systemd: starts `REPLICAS_PER_GPU` replicas on each GPU in `GPUS` (ports GPU-major from `BASE_PORT`), fronts them with `serve-proxy.py`, and restarts any replica or proxy whose `/health` fails once `LOAD_GRACE` has passed. Pidfiles and logs go under `$FLEET_RUN`. It starts the proxy without `--max-body-mb`, so a fleet it starts caps bodies at 32 MiB |
| `tools/serve-proxy.py` | Least-outstanding reverse proxy over `--backends`, at most `--cap` (default 8) outstanding requests per backend: the engine's exactness-tier batch width and the anti-thrash bound for two replicas per GPU. Past capacity a bounded FIFO queue (`--queue-max` 256, `--queue-deadline` 30 s) holds requests; overflow or the deadline answers 429 with `Retry-After`. Routes on each backend's `/readyz`; serves its own `GET /health` and `GET /metrics` (JSON). Bodies default to 32 MiB (`--max-body-mb`, 1 to 192): pass `--max-body-mb 192` to match the server. A `/v1/` POST needs a `Content-Length` and exactly one credential header, both checked before the proxy reads the body |
| `tools/load-serve.py` | Concurrent OpenAI-format load harness: `--concurrency` workers on `/v1/chat/completions`, aggregate output tok/s and p50/p95 latency, one JSONL line per load point (`--per-request` adds one per request) |
| `tools/serve-smoke.sh [model.gguf [draft.gguf]]` | Serving smoke battery against one booted server: chat (streaming and not), plain completions, concurrency, greedy determinism and spec-vs-plain greedy identity. Exits nonzero on a failed check |

### Pipeline-parallel (PP-2) serving

For a model that fits only across two cards: one engine process, the layer trunk cut into two
stages, each stage's weights and KV on its own card.

```bash
MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 \
MEMRA_MODELS="big=/path/to/model.gguf" memra-server
```

- The boot prints `[pp] cross-device transport: stage0=dev0 stage1=dev1` when the split is live.
  `MEMRA_PP_SPLITS` sets explicit cuts (default an even split).
- **Exactness.** `decode-batch-gate --mode pp` compares every f32 logit of every row against the
  unsplit walk on the same weights: 0 differing bits on seven configurations
  ([receipt](../research/pp2-batch-20260806/)). The batched split costs 0.5% to 1.5% at B=4 to 16,
  almost all of it the stage seam ([Performance](PERFORMANCE.md#pipeline-parallel-pp-2)).
- **Speculation is off by default here.** Speculative PP-2 is correct, but plain decode wins every
  measured c=1/2/4 cell, so the placement-aware gate admits no request to speculation
  ([receipts](../research/specplace-20260808/)). `MEMRA_SPEC_GATE=0` restores always-spec.
- **What refuses.** The four decode paths without a stage split fail closed under a sharded
  cross-device placement (`pp::refuse_unsplit_if_remote`): an unsplit trunk peer-reads every remote
  stage's weights each step, measured 7.4 against 208.9 tok/s at B=1. `MEMRA_PP_ALLOW_UNSPLIT_BATCH=1`
  re-admits them as a measurement door; `MEMRA_PP_SHARD=0` keeps all weights home (full speed,
  without the capacity PP-2 exists for).
- **Prefill.** Adjacent prompt chunks overlap, stage 0 of chunk N+1 beside stage 1 of chunk N
  (`MEMRA_PRIME_PIPE`, on), over up to eight auto-sized microchunks with a short fill and a
  shrinking tail (`MEMRA_PRIME_CHUNK_SCHED`; an explicit `MEMRA_PRIME_CHUNK` keeps fixed ranges).
  Simultaneous fresh Step-3.7 prompts batch their weight-streaming work while attention and KV
  stay per request (`MEMRA_STEP35_PRIME_BATCH`, on). Each has its byte-identity gate
  ([pipeprime](../research/pipeprime-20260808/), [microchunk](../research/microchunk-20260808/),
  [primebatch](../research/primebatch-20260808/)).
- `MEMRA_PP_HOST_BOUNCE=1` stages the boundary through pinned host memory for hosts whose peer copy
  reports success but does not preserve bytes ([receipt](../research/hostbounce-20260810/RESULTS.md));
  it turns the prefix cache, session affinity and speculation off. A runtime peer probe that finds
  a bad copy moves the boundary to host bounce on its own (`/metrics`
  `peer_probe_degraded_to_host_bounce`). Byte-probe peer paths when you provision a box.
- Gates for any change to a split path: [Testing](TESTING.md#multi-gpu-pp-n-exactness-gates-run-on-the-multi-card-box).

### Experimental PP-3/PP-4 wavefront

`MEMRA_PP_WAVE=1` adds a 3-4 card schedule: up to one exact-width request wave per stage, advanced
in dependency order with one host worker per stage; prompt microchunks use the same shape. It
requires 3 or 4 distinct CUDA ordinals, native P2P, the double-slot boundary policy, ModelPlan
pipeline capability with legal cuts, and a qualified batched decode rewrite; host bounce and
repeated devices refuse. A failed mutating wave taints its caches and aborts those sessions.
`/metrics.pp_wave` exposes ticks, cells and host-walker overlap. It is infrastructure, not a
measured serving route: run the PP-N gate matrix on the exact host before enabling it outside a
research process.

### Automatic placement, tensor and expert parallelism

`MEMRA_PARALLEL=auto` with `MEMRA_PARALLEL_DEVICES=a,b[,c,d]` (2 to 4 distinct devices) plans one
model from its ModelPlan: contiguous PP for a dense model, whole-expert EP for a routed MoE when
every rank keeps its reserve, else PP or a refusal. Step-3.7 has model-specific TP, EP and hybrid
placement (`MEMRA_STEP_TP`, `MEMRA_STEP_EP`); its planner checks the 64/96 query-head geometry, the
eight KV heads and the 128-row E4M3 scale blocks before any allocation, which refuses illegal
layouts early but is not throughput evidence. GLM-5.3-Flash TP-2 (`MEMRA_GLM5_TP`) serves plain
only. DeepSeek-V4 has its own `MEMRA_DSV4_DEVICES` and `MEMRA_DSV4_TOPOLOGY`. A replica fleet or PP
is never TP. Predictive admission enforcement and several per-card defaults stay off on a
multi-device deployment.

## Authentication and tenants

<a id="api-keys--multi-key-tenant-auth-laneapi-keys-2026-08-05"></a>

### API keys

- `MEMRA_API_KEY` sets one static bearer, tenant `default`; an empty value is a boot FATAL.
- `MEMRA_API_KEYS` sets a keyring: a TOML file of `[[keys]]` entries (`sha256` of the key, `tenant`
  matching `[A-Za-z0-9_-]+`, `lane` `interactive` or `batch`, `enabled`, optional `rate_limit`), or
  an inline `tenant:sha256hex[:lane],...` list. Only hashes are stored. A malformed ring is a boot
  FATAL. The file must be a private regular file, not a symlink (mode 0600 or 0640), owned by the
  service user, with one hard link and at most 8 MiB; every reload and key command checks it again. It reloads on mtime change (polled at most every 2 s), and a
  broken rewrite keeps the previous ring and logs loudly.
- With neither, the server is open, tenant `default`. A non-loopback `MEMRA_ADDR` without a key
  source refuses to start; `MEMRA_ALLOW_OPEN_BIND=1` is the explicit development override, and it
  never opens the metrics routes.

```
memra-server --gen-key acme [--lane batch] [--rate-limit 4] [--keys /path/keys.toml]
memra-server --revoke-key <key-prefix> [--keys /path/keys.toml]
```

`--gen-key` prints the plaintext key (`mk-<tenant>-<48 hex>`) once and appends its hash; a new
keyring is created with mode 0640. `--revoke-key` disables the key matching an unambiguous prefix,
writing a temporary sibling and renaming it atomically, so a running server's reload sees the whole
old or new file. `--keys` defaults to `MEMRA_API_KEYS`. Gate: `tools/apikeys-gate.sh`.

**Request law.** `Authorization: Bearer <key>` on every key route (`/v1/messages` and
`/v1/auth/check` also take `x-api-key`). A disabled key is a 403; an unknown key or a missing header
is a 401. `MEMRA_API_KEY` keeps working as tenant `default` with or without a keyring.

**What the tenant drives.** Cache isolation (the `t:<tenant>` namespace, [above](#cache-salt-isolation));
lane class (batch-class keys default to `harvest` and get a 403 on `x-lane: interactive`); rate
limits (a per-key `rate_limit` can only narrow the lane cap, keys of one tenant share its gauge);
and the metering seam (one `[meter] admit id=<x-request-id> tenant=<t> lane=<l> model=<m>` line per
admitted request).

### Metrics access

Metrics need a bearer whenever the bind is non-loopback, any API-key source is configured, or
`MEMRA_METRICS_TOKEN` is set. A configured metrics token never authorizes completion routes.

- With `MEMRA_METRICS_TOKEN`, that token alone reads `/metrics` and `/yield/metrics`, completion
  keys get 403, and the scrape sees every tenant row plus the operator-only gauges and aggregates.
  Use it only for an operator trusted with every tenant's volume.
- Without it, a keyring key reads `/metrics` scoped to its own `tenants` row and
  `adsd_suspect_total` entry, and `/yield/metrics` is a 403.
- The single `MEMRA_API_KEY` keeps its cumulative counters and yield view, but not the
  operator-only prefix, capacity and speculation blocks.
- Loopback with no key source and no token is open with full visibility.

The operator-only fields expose cross-tenant volume and the memory and scheduler state an
attacker could use to time a [Fill-and-Squeeze attack](https://arxiv.org/abs/2602.07878).

<a id="health-readiness-and-fault-handling-serve-hardening-lane-2026-08-06"></a>

## Health, supervision and faults

`/health` and `/livez` report inference liveness, not process liveness: they derive from a heartbeat
the scheduler loop stamps every iteration, the engine's prime odometer, and the panic and GPU fault
latches. Receipts: [research/serve-hardening-20260806/](../research/serve-hardening-20260806/);
gate: `tools/health-fault-gate.sh`.

### Worker phases

| Worker phase | `/health` | Meaning |
|---|---|---|
| `loading` | 503 | weights not resident. On a first boot the port binds after the load, so a probe sees connection refused; over HTTP this phase is a respawn |
| `warming` | 503 | weights resident, the boot calibration probe running (one spec-shaped generation). Skipped when no probe runs (`MEMRA_ADMIT_CALIBRATE=0`, `MEMRA_SERVE_SPEC=0`, `MEMRA_ADMIT_RESERVE_MB`) |
| `idle` | 200 at any beat age | the worker waits for work; a quiet server stamps nothing for hours |
| `busy` | 200 while forward progress advances; 503 past `MEMRA_HEALTH_STALL_S` (120 s) | progress is the fresher of the scheduler heartbeat and the prime odometer (one completed prime chunk), so a long prefill reads busy and healthy; `MEMRA_HEALTH_PROGRESS=0` uses the heartbeat alone |
| `dead`, or a fault latched | 503 at once | worker panic or fatal Xid: a latch, not a timeout |

The body is `{status, models, worker: {phase, beat_age_ms, tick_max_ms, stall_threshold_ms,
forward_progress_age_ms, prime_progress, generation, xid_warnings, gpu_probe, ...}}`, plus a
top-level `detail` on a red, where a quoted panic payload lands. `status` is `ok`, `draining` or
`unhealthy`. `tick_max_ms`, the longest scheduler iteration the process has seen, is the receipt
for revisiting the stall bound. The stall bound does not catch a hang inside one prime chunk, a
livelock that keeps completing chunks, or per-session starvation. A dedicated route (the
DeepSeek-V4 thread) registers its own health record, judged by the same bound
([deployment binaries](serving/deployment-binary.md)).

### Readiness and drain

`/readyz` answers `ready` (200) when a model is loaded, the worker is alive and the server is not
draining, else `not_ready` (503). Not ready is not a restart request: loading and draining are
healthy states that must not take traffic. Queue depth never flips readiness; backpressure belongs
on the request path. Worker-related 503s carry `Retry-After: 2` and `retry-after-ms: 2000`.

SIGTERM starts a graceful drain: `/health` reports `draining` (still 200), `/readyz` 503, new
completions get 503 `draining` with `Retry-After` set to `MEMRA_DRAIN_S` (clamped to 1-60), and
in-flight requests, streams included, finish within `MEMRA_DRAIN_S` (default 30 s). Then the
process joins the GPU worker and exits 0.

### Worker panic ladder

The worker runs inside `catch_unwind`. A panic marks health dead with the quoted payload; after a
2 s backoff one respawn reloads the weights (`MEMRA_WORKER_RESPAWN`, default 1), and work queued
during the dead window is served by the respawned worker. When the budget runs out
(`STATUS=worker unrecoverable; exiting`) or the respawn's weight reload fails
(`STATUS=respawn load failed; exiting`), the process exits 70 so the supervisor restarts it whole.
Startup FATALs exit 1. One attempt, because CUDA errors are sticky per process: a respawn loop on a
poisoned context looks alive and serves nothing. `MEMRA_PANIC_AFTER` is the fault-injection door.
A respawn reloads the weights only: the tokenizers and chat templates read at boot stay, so a
replaced tokenizer or template file takes effect after a full process restart.

### Per-request faults

Every per-session step, prefill call and batched call runs under `request_fault_guard`. A panic
whose payload quotes no driver or library error, on a CUDA context that still answers a
synchronize, an allocation and a readback, is a request fault: one
`[fault] request=<id> route=lane<n>/<model> site=<site> panic=<msg>` line, `request_faults_total`
plus one, and only that request (or that batched wave) ends with `code: worker_fault`; its KV never
reaches a reuse pool. Anything else goes up the ladder above. A rising `request_faults_total` with a
flat `worker_respawns_total` is a request-shaped bug that needs a repro, not a card that needs a
restart. A panic that poisons a mutex is outside this boundary. Gate: `tools/request-fault-gate.py`.

### GPU faults

A watcher thread (`MEMRA_GPU_WATCH`, on) tails Xid lines from `/dev/kmsg`, falling back to
`journalctl -k -f`, and latches unhealthy on the fatal classes 48, 64, 79, 94, 95, 119 and 120,
counting the rest as warnings. Every `MEMRA_GPU_WATCH_S` (60 s) it probes `nvidia-smi` for
uncorrectable ECC and row-remap failures as a child killed at `MEMRA_GPU_PROBE_TIMEOUT_S` (10 s),
because Blackwell's worst wedge (Xid 119/120) hangs the query tools and tells the process nothing.
One probe past its deadline makes the process degraded and still live; `MEMRA_GPU_PROBE_MISSES`
(3) consecutive misses latch the fault. Fatal Xid, ECC and row-remap findings latch on first sight.
At startup the probe retries up to six timeouts to let VRAM teardown finish after a redeploy. A
fault survives a worker respawn. `/health` publishes `worker.gpu_probe.{degraded, miss_streak,
last_ok_age_ms, degraded_reason, latched_reason}`; a guard should restart on `latched_reason`, not
on `degraded`.

### The supervision contract (systemd)

`deploy/systemd/memra-server.service` is an example unit for one supervised instance: copy it and
edit the paths. `Type=notify` with `NotifyAccess=main` sends `READY=1` after the models load and
the socket binds, so `systemctl start` returning means the server can serve. `WATCHDOG=1` is sent
only while inference is live. Exit 70 (unrecoverable worker) reads apart from exit 1 (bad config).
Three of its values are sized against server defaults; changing one side alone produces a unit that
misbehaves only during a failure:

| Directive | Value | Coupling |
|---|---|---|
| `WatchdogSec` | 180 | must exceed `MEMRA_HEALTH_STALL_S` (120); the heartbeat feeds both. Raise both together |
| `TimeoutStopSec` | 60 | must exceed `MEMRA_DRAIN_S` (30). Under `Type=notify` the server also sends `EXTEND_TIMEOUT_USEC` for two drain deadlines plus 5 s; a unit without `Type=notify` needs this above twice `MEMRA_DRAIN_S` |
| `TimeoutStartSec` | 600 | must exceed the slowest cold load (about 120 s for a 27B NVFP4 from page cache) |
| `StartLimitIntervalSec`, `StartLimitBurst` | 3600, 4 | systemd's 10 s / 5 cannot trip when each start takes minutes; four failed loads an hour stops the unit for a human |
| `RestartSec`, `RestartSteps`, `RestartMaxDelaySec` | 10, 4, 160 | a card that threw an Xid needs the driver to settle; the ramp needs systemd 254 or later |
| `OOMPolicy` | `kill` | the kernel OOM killer can take one thread, classically the worker, and leave a process that accepts connections and never serves. Host memory only; CUDA OOM is a 503 |

With `kernel.dmesg_restrict=1` an unprivileged unit sees Xids only through `journalctl`; grant
`CAP_SYSLOG` (ambient and bounding) or rely on the probe-hang and ECC detectors, which need no
kernel log. The watcher logs which source it got. The unit is not `ProtectSystem=strict`: model
paths, `/dev/nvidia*` and the CUDA cache need real filesystem access. Keep stdout and stderr in a
log file: the `[engine-error]` lines are the operator's evidence.

## Errors

The status class comes from the producer, so a client or router can tell a malformed request from
an engine fault. `Retry-After` is always whole seconds, at most 60, with a matching `retry-after-ms`.

| Condition | Status | `type` | `code` | Retry headers |
|---|---|---|---|---|
| malformed field, bad template, bad `response_format`, bad `timeout_ms` | 400 | `invalid_request_error` | | `x-should-retry: false` |
| JSON body type or syntax error | 400 | `invalid_request_error` | | `x-should-retry: false`; `param` names the field when the parser has a path |
| missing or non-JSON `Content-Type` | 415 | `invalid_request_error` | | `x-should-retry: false` |
| prompt at or past the context cap; a request that cannot fit an idle server at its post-load capacity | 400 | `invalid_request_error` | `context_length_exceeded` | `x-should-retry: false` |
| unknown model id | 400 | `invalid_request_error` | `model_not_found` | `x-should-retry: false` |
| non-streaming request that cannot fit its deadline | 400 | `invalid_request_error` | `nonstream_deadline_infeasible` | `x-should-retry: false`; the message names a `max_tokens` that fits |
| unknown `x-lane` | 400 | `invalid_request_error` | `invalid_lane` | `x-should-retry: false` |
| bad, missing or disabled key; batch key on `interactive` | 401 / 403 | `authentication_error` | | `x-should-retry: false` |
| deadline missed before any token (or before the first token, streaming) | 408 | `timeout` | `deadline_exceeded` | none |
| deadline missed mid-generation, OpenAI dialects | 200 | | `deadline_exceeded` in the response `error`, `finish_reason: "error"` | none |
| body larger than 192 MiB | 413 | `invalid_request_error` | `request_too_large` | |
| interactive shed | 429 | `rate_limit_error` | `shed_deadline`, `shed_queue`, `shed_queue_wait` | `Retry-After` = the wait estimate |
| dark-lane shed, per-key cap, memory defer past its budget | 429 | `rate_limit_error` | `rate_limit_exceeded` | `Retry-After` |
| body pool full | 429 | `rate_limit_error` | `body_admission_busy` | `Retry-After: 1` |
| request fault (a panic confined to one request) | 500 | `server_error` | `worker_fault` | none |
| step, prefill, graph or constraint fault | 500 | `server_error` | `engine_error` | none |
| out of VRAM, step OOM past its park budget, worker restarting, an idle server short of the free memory for a request its capacity holds | 503 | `server_error` | `overloaded` | `Retry-After: 5` |
| worker channel dropped | 503 | `server_error` | `overloaded` | `Retry-After: 2` |
| new request during a drain | 503 | `server_error` | `draining` | `Retry-After` = `MEMRA_DRAIN_S`, at most 60 |

Engine-fault bodies carry a stable sentence, never the producer's text. `overloaded` reads "the
model is temporarily at capacity; retry after the Retry-After delay"; `engine_error` reads "the
engine could not complete this request; retry, and report the request id if it persists". The
producer text goes to the `[engine-error] class=... <text>` log line. Refusals the server authors
(sheds, request shape, constraints) keep their exact text. An unknown model is a 400 rather than a
404 by design, because routers count a 404 as a provider failure; clients branch on `code` either
way. A failure after the 200 is committed sends the same error object as a final `data:` chunk and
closes the stream. The Anthropic surface uses its own error shape with the same statuses.

## Observability

### Metrics

`GET /metrics` returns JSON; `Accept: text/plain;version=0.0.4` (or OpenMetrics) selects Prometheus
text under the same [access rules](#metrics-access). Every field and family is in
[metrics](serving/metrics.md). Watch first: the `memra_queue_wait_seconds`, `memra_ttft_seconds`,
`memra_e2e_seconds` and `memra_tpot_seconds` histograms (`{model, route, lane}`),
`memra_requests_total{...,code}` (`499` is a client that left), the active, queued and KV gauges;
`prompt_tokens_in`, `cached_tokens_in` and the operator-only `lcp_histogram` (measure hit depth, not
only hit rate); and `request_faults_total` beside `worker_respawns_total`. Counters publish every
32nd tick and on any tick that admits or retires a request or moves the speculative or host-tier
counters, so a scrape can trail a response by one tick; the log lines are the per-event record. `tools/cache-meter-gate.py` checks the metering closed form.

<a id="per-request-timing-log-memra_request_log-2026-11-09"></a>

### Per-request timing log

`MEMRA_REQUEST_LOG=/path/requests.jsonl` writes one JSON line per request on the four generation
routes, streaming and blocking, refusals included: timing, end reason, HTTP status, token counts,
the reply's tool calls, and a session key that holds across turns. No prompt or reply text is
written; content, identities and arguments appear only as unsalted sha256 digests, so the log is
pseudonymous, not anonymous. Unset, nothing is installed.

The file opens for append at boot (created 0640); a path that cannot be opened stops the boot.
Rotate with `copytruncate`. A writer thread drains a queue bounded at 4,096 lines and 64 MiB, a line
that does not fit is dropped and counted (`memra_request_log_lines_total{outcome="dropped_queue_full"}`),
and serving never waits for it. Generated bytes are identical with the log on and off
([receipt](../research/request-log-20261109/)). Fields, session key and end reasons:
[request log](serving/request-log.md).

### ADSD acceptance-collapse detector

[ADSD](https://arxiv.org/abs/2607.21804) shows that an adversarial suffix can collapse speculative
acceptance, as cost amplification, while outputs stay correct. With keys configured, memra compares
each tenant's last eight speculative requests with the model's rolling 64-request history,
excluding that tenant's own rows where it can. A window is collapse-shaped at 0.20 or more below the
baseline with a pooled two-proportion z-score of -3.0 or lower (baselines need 16 samples and 512
drafted tokens, a window 128). Three in a row emit one `[adsd-suspect]` line and add to the tenant's
`adsd_suspect_total`, latched until acceptance recovers to within 0.10. Detection only: nothing is
throttled or evicted. Read `usage.spec` and the log, then apply the tenant's rate limit by hand.
The strongest known adversary for this signal is [Mistletoe](https://arxiv.org/abs/2605.14005).

## Gateway listing surface

memra keeps four model-list views apart, because listing schemas reject fields shaped for another:

- `GET /models` returns `{"object":"list","data":[{"id":"<alias>","object":"model"}]}` and does not
  change.
- `GET /models?schema=openrouter` renders the router feed (schema version 2.4: typed modality
  objects with nested pricing and capacity, `supported_parameters` per model).
- `GET /models?schema=openmodels` renders the flat feed; it answers 400 unless every model declares
  the fields that feed requires, so it never serves a superficially valid feed.
- `GET /v1/models` is the contract-v2 catalog: one row per model with `context_length`,
  `endpoints`, modalities, `capabilities`, `pricing`, `lifecycle` and `reliability`.
- Any other `schema` value is a 400.

What the process cannot derive comes from the optional TOML file `MEMRA_MODEL_METADATA` names, one
`[models."<alias>"]` table per `MEMRA_MODELS` alias (limits, `surface`, `default_reasoning_effort`,
sampling defaults, pricing and capacity declarations, lifecycle, reliability, modalities, and a
top-level `[provider]` block). It is parsed before the GPU worker starts; an unknown field, a bad
value or an alias missing from `MEMRA_MODELS` is fatal, so roll out a new binary before a config
with new keys. Absent values are omitted, never rendered as zero, and every catalog publishes the
smaller of the trained context and the declared envelope. Schema and examples:
[model metadata](serving/model-metadata.md#schema).

### Vendor sampling defaults

Seven optional `default_*` keys substitute the model vendor's published sampling recommendation
into fields a request omits, identically on all four generation surfaces. An explicit client value
always wins, and an explicit `temperature: 0` is still greedy. `default_temperature = 0.0` is fatal
at boot, because a zero default would make every omitting client greedy. An optional
`non_thinking_sampling` table declares the vendor's second arm for requests whose thinking is off.
Keys and ranges: [model metadata](serving/model-metadata.md#vendor-sampling-defaults); why:
[decision](decisions/VENDOR-SAMPLING-DEFAULTS.md).

**Performance note.** Vendor recommendations usually include truncation filters (`top_p` below 1, a
positive `top_k`), which move the default request out of the pure-temperature regime. Exactness is
unchanged on every route, but the throughput profile of the default path changes; a client opts back
in with explicit `top_p: 1.0` and `top_k: 0`. The test
`vendor_defaults_leave_the_pure_temp_sampled_spec_regime` pins this.

## Embedding memra-server in a deployment binary

The stock binary is the open engine with no accounting. A deployment binary compiles against the
crate and adds its own surfaces through `memra_server::ServerWiring`
([deployment binaries](serving/deployment-binary.md)):

- `with_metering(factory)` compiles in a `memra_server::metering` implementation, which speaks
  tokens and verdicts, never money; `with_job_store` supplies a background-result store;
  `claiming(var)` declares the deployment variables the binary consumes itself.
- `on_ready(RuntimeHandles { .. })` hands over the trim, metadata reload, tenant purge and KV host
  handoff handles and the drain signal. Drop every handle on that signal: the GPU worker exits only
  when every command sender is gone.
- Auth stays in the engine: keys, tenant identity, lane class and rate limits are engine security
  surface.

### Dead-darklane background jobs

`MEMRA_BG_JOB=<command>` runs a background job only while the engine has nothing to do, stops it
(SIGSTOP to its process group) on the first sign of work, and resumes it after `MEMRA_VALLEY_S`
(2 s) of quiet. A stopped job keeps its memory; `MEMRA_BG_VRAM_MB` (default 0, CPU only) is granted
at launch only when every visible GPU has that much free plus `MEMRA_MOE_RESIDENT_HEADROOM_GB` (2). `MEMRA_BG_YIELD_MODE=checkpoint` asks the
job to checkpoint on SIGUSR1 and exit 75, relaunches it next valley, and kills it after
`MEMRA_BG_CKPT_GRACE_MS` (5 s); write the command as `exec <program>` so the exit code reaches the
runner. Reference job: `tools/bg-ckpt-counter.py`. Protocol:
[deployment binaries](serving/deployment-binary.md#dead-darklane-background-jobs-memra_bg_job).

## Exactness guarantees

Laws in full, with the gates and receipts behind each: [exactness](serving/exactness.md). The
project rule behind them is [one numeric program per request](../AGENTS.md#one-numeric-program-per-request).

### The isolation contract

Greedy serving is isolated-identical under concurrent load at the defaults: a request's output is
byte-identical whether it arrives alone or inside a full batch. The serve gate replays the same
prompts at c=1 and c=16 and byte-compares every stream; it runs 16 prompts at 96 `max_tokens`
arriving together, so quote the contract with that scope.

The default keeps B=1 and B>=2 on the one generic batched decode body for every architecture, and
GraphSession replay is off, so a request that gains or loses a peer never changes numeric program.
`decode-batch-gate` config mode enforces B=1 against B=N bit identity. `MEMRA_SERVE_B1FAST=1` is a
fixed-solo measurement door only; use it only where admission guarantees a request can never gain a
peer. It stays off, even when set, on every GatedDeltaNet plan, Gemma 4, the Step program and
SwiGLU-OAI trunks. The one-class default gave up the eager door's solo win, measured at 8.33%
decode on Qwen3.5-9B and 5.19% on a 27B at c=1 on the 82-SM RTX 5090 Laptop (2026-08-05, N=5,
[receipt](../research/servepath-p2-20260805/)) before the door closed on GatedDeltaNet plans; c>1
is unaffected. A continuation chunk of a long cold prompt joins a cross-request prime batch only
when every trunk operation of the plan has a carried-prime implementation; routed-MoE plans prime
those chunks alone. Why: [decision](decisions/ONE-CLASS-DECODE-DEFAULT.md).

### First-token cross-config drift

Serving primes prompts batched while the historical oracle stream is tokenwise, two numeric
programs by design, so on a near-tie prompt the first generated token can differ from the tokenwise
stream. On the 2026-08-02 six-model sweep 10 of 144 first tokens flipped, every flip at a top-two
margin of 0.70 or less ([receipt](../research/prime-gate-coverage-20260802/)). A cache hit compared
with a cold run of another prime program, and any change in which prime calls run concatenated,
inherit this class. `MEMRA_PRIME_TOKENWISE=1` pins the oracle stream at prefill cost. Chunked
prefill itself is split-stable: output is byte-identical across `MEMRA_PRIME_CHUNK` values
([exactness](serving/exactness.md#chunked-prefill-is-split-stable)).

### The prime-grid law

On the GDN hybrids, a prompt primed as two calls split at L is bit-identical to the monolithic
prime if and only if L is a multiple of `gdn_chunk_size()` (32): the chunked WY scan segments per
call, so an off-grid split materializes recurrent state the monolithic program never computes.
Lawful floating-point behavior, not a defect. So the server rounds every boundary it chooses down
to the grid: checkpoint stops, cache captures and seeds, and pool rewinds. A resume re-primes at
most 31 tokens, and a recurrent resume primes at least 16 rows. The
[capture law](#prompt-caching) states the operator consequences; the sites, gates and receipts are
in [exactness](serving/exactness.md#the-prime-grid-law). Why: [decision](decisions/PRIME-GRID-LAW.md).

What the grid law still does not promise: a verbatim-extension resume of an attention-only cache
keeps decode-written rows a cold prefill never reproduces (a bounded perturbation that flips only
near-ties), and the in-batch fanout takes the raw LCP as its capture length
([detail](serving/exactness.md#what-the-grid-law-still-does-not-promise)).

### Guarantees and their receipts

The standing list, each with its receipt, is in
[exactness](serving/exactness.md#serving-contract-guarantees-and-the-receipt-for-each). In short:
serving fails closed on exposure ([sec1](../research/sec1-20260811/RESULTS.md)); the GGUF parser
rejects malformed files without panicking the worker ([ggufhard](../research/ggufhard-20260811/RESULTS.md));
the GPU Gumbel sampler keeps uniforms below 1.0 ([longdepth](../research/longdepth-20260809/RESULTS.md));
explicit `max_tokens` is enforced exactly ([honesty](../research/honesty-20260809/RESULTS.md));
release builds retire prefix pins and bound client salts ([pinfix](../research/pinfix-20260809/RESULTS.md));
Step-3.7-Flash PP-2 decodes in one numeric class at every batch width
([b1fix](../research/b1fix-20260810/RESULTS.md)); the Step SWA ring, on by default for the step37
family, fits 2 to 12 concurrent 262k sessions ([kv256](../research/kv256-20260809/RESULTS.md)); and
graceful shutdown joins the GPU worker before exit. Every receipt directory has its verdict line in
[research/INDEX.md](../research/INDEX.md).

## Further reading

- [API surfaces](API-SURFACES.md): every route's request and response schema.
- [Flags](FLAGS.md): every `MEMRA_*` variable, its default and its rollback seam.
- [docs/serving/](serving/): the reference pages linked throughout this guide.
- [Testing](TESTING.md#serving): the serving, prefix-cache and multi-GPU gates.
- [Performance](PERFORMANCE.md#serving-measurements): serving measurements and open gaps.
- [Decisions](decisions/README.md#serving-admission-and-delivery): why each serving default is what
  it is.
- [Cookbook](COOKBOOK.md), [rig cards](rigs/) and [workload guides](workloads/): what to run where.
