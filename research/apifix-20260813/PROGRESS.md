# cx-apifix progress

Lane: `lane/cx-apifix`  
Base: `main` at `5b6a80bf4fe9c9fe081c69928e563e59aa923a63`  
Started: 2026-08-13 (Asia/Jerusalem)

## Scope

Stage, prove, and document the API metadata/runtime fixes requested for the OpenRouter supplier
surface. Do not touch the live serve box, submitted application, public endpoint, merge state, tags,
or remotes.

## Gates

- [x] Trace every declared value from source/config/env through both model-schema emitters.
- [x] Report the existing mechanism before implementation changes.
- [x] Reconcile advertised prompt/context/output limits with actual runtime validation.
- [x] Make limits, capacity, pricing, and offered model set configurable in one documented place.
- [x] Give every offered model complete input, cached-input, completion, request, and concurrency capacity.
- [x] Preserve honest capacity defaults and measurement TODOs; do not extrapolate unpublished 3-card results.
- [x] Prove clean queueing or HTTP 429 admission under long-context KV pressure; never OOM-crash.
- [x] Run a genuine long-prompt and long-generation end-to-end check locally.
- [x] Pass the local-server 21-check public protocol/accounting gate.
- [x] Record failure text verbatim, raw evidence, and exact maintenance-window cutover commands in `RESULTS.md`.
- [x] Commit the finished lane only; no merge, tag, push, endpoint change, or form update.

## Activity

- 2026-08-13: Confirmed this worktree is clean on `lane/cx-apifix`, exactly at current local `main`
  (`5b6a80bf4fe9c9fe081c69928e563e59aa923a63`). Created this progress ledger before any
  implementation edit.
- 2026-08-13: Pre-change mechanism trace reported to the owner before touching implementation or
  deployment configuration:
  - `MEMRA_MODELS` is the served-model alias/path roster. `/models` iterates those loaded aliases.
  - `MEMRA_MODEL_METADATA` selects the TOML registry. The current values are literal entries in
    `deploy/gateway/q27-models.toml`; they are not Rust constants or per-field environment values.
  - Worker/model caps supply `max_context_length`; the TOML independently supplies
    `max_prompt_length`, causing today's reviewer-visible 262,144 versus 7,680 disagreement.
  - The OpenRouter emitter maps prompt/cached-prompt capacity to input, completion/concurrency to
    output, and request RPM to model level. Q35 has only `concurrency`, so its missing capacity is
    a registry omission.
  - Both completion routes apply the TOML prompt/output limits. Prompt enforcement happens after
    actual render/tokenize and before cache/GPU admission; omitted `max_tokens` becomes the TOML
    output maximum and explicit excess is a clean 400.
  - Found a context-bound defect that must precede the metadata raise: finite-output requests use
    `prompt + max_tokens + 8` without capping that allocation to trained model context. Setting both
    limits to 262,144 unchanged could request a roughly 524k allocation from a 262k model.
  - Descriptive `capacity.concurrency` does not enforce admission. The keyring's per-tenant
    `rate_limit` produces early HTTP 429; interactive session and estimated-VRAM overflow queue
    FIFO in the worker. Runtime OOM survival remains an empirical gate, not a source-only claim.
- 2026-08-13: Staged one validated registry with Q35 active, Q27 removed, and Qwen3.8-27B plus
  Gemma 4 26B-A4B present only as non-emitting planned entries. The registry now carries the three
  requested price schedules, 262,144-token independent prompt/output ceilings, an 8,192-token
  ordinary default, a complete conservative Q35 capacity block, and explicit measurement TODOs.
- 2026-08-13: Capped every request-shaped context allocation to the loaded model's trained context.
  The worker now applies its VRAM headroom test to the first request too: an unattainable request is
  rejected pre-header as retryable HTTP 429, while pressure with active work retains FIFO queueing.
- 2026-08-13: Static gates pass: `cargo test -p memra-server` (256 passed, 0 failed) and
  `python3 -m unittest deploy/gateway/test_probe.py` (9 passed, 0 failed). Raw console logs are in
  `research/apifix-20260813/raw/`. No formatting command was run.
- 2026-08-13: Closed a streaming admission gap found before GPU execution. Interactive streams now
  wait for the worker's existing immediate `PromptUsage` admission acknowledgement; an admission
  error therefore remains a real pre-header HTTP 429, while successful requests still do not wait
  for a first token. Added direct tests for both the error and acknowledgement paths.
- 2026-08-13: Removed quadratic whole-tail detokenization from the serve loop. Each session now
  appends the bytes for one generated token and preserves the existing UTF-8 cursor and stop-string
  semantics, so field-length generation no longer repeatedly decodes its entire history. The
  reproducible long-limit harness is
  `research/apifix-20260813/long_limits_gate.py`.
- 2026-08-13: Under the local 5090 lock, eight simultaneous full-context reservations queued FIFO
  through measured VRAM pressure: all eight returned HTTP 200, the process stayed live, and the
  worker recorded zero OOM parks. The separately rate-limited overload cell returned four HTTP 200
  plus four real pre-header HTTP 429 responses. The local public protocol/accounting gate passed
  all 21 checks with exact usage/metrics equality.
- 2026-08-13: A retained ledger audit found that worker-originated pre-header errors had the right
  client HTTP status but were durably classified as `admission_shed`. Commit `9c11bb9f8` now carries
  the worker error taxonomy into the ledger; the full server suite passes 258 tests with 0 failures,
  including direct 429 and context-length classification tests.
- 2026-08-13: The ordinary unconstrained generation completed exactly 131,072 output tokens with
  `finish_reason=length`, 131,072 token events, HTTP 200, and zero OOM parks. The final runtime
  binary then completed a 524,286-byte genuine UTF-8 prompt as 262,143 prompt tokens plus one output
  token, exactly filling the 262,144-token trained context.
- 2026-08-13: Repeated the metadata/limit and 21-check gates on the exact final runtime binary. The
  limit cells passed 3/3, the public protocol/accounting gate passed 21/21 with exact metrics, the
  overload shape returned four HTTP 200 plus four clean HTTP 429 responses, and the durable ledger
  preserved `context_length_exceeded` versus `rate_limit_exceeded`.
- 2026-08-13: Stopped the localhost-only server and verified no remaining memra process, compute
  app, listener, or 5090 lock holder. `RESULTS.md` contains owner-run cutover and rollback commands;
  this lane performed no remote, public endpoint, form, merge, tag, push, or release operation.
