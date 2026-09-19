# What a serving inference engine must have for real inference usage — and where memra stands

Lane `serving-musthaves-20260919`. Audit at memra `61be8b0d` (origin/main, 2026-09-19).
Tracking issues for the ranked gaps: #521–#533 (one per gap, §7).
Method: (1) the converged 2026 industry baseline (every mainstream engine ships it, so a
customer assumes it); (2) the production incident record of this program's own serving
(private sibling repo, `../darklanes/ops/incidents/` and `research/*incident*`), which is the
only "real usage" evidence we own; (3) a file:line audit of `crates/memra-server` against
both. Every status cell below was checked in the tree, not inferred from docs.

Scope: the **engine process behind the launcher**. Router, keys-as-product, billing ledger,
status page and fleet are control-plane and live in darklanes. Where the engine must expose a
seam for the control plane, the seam is listed here.

Status vocabulary: **HAVE** (implemented and a re-runnable gate exists), **HAVE/unsealed**
(implemented, proof is a dated research receipt or unit test only — no serving-shape gate
re-runs it), **PARTIAL** (exists on some routes / behind a default-OFF door / accepted but
ignored), **MISSING** (grepped, not found). "Serving-shape gate" = a script that boots the real
server on a GPU and asserts the property over HTTP.

---

## 0. The one-sentence answer

A serving engine is fit for real inference when a customer can point an unmodified OpenAI
SDK at it and, under mixed concurrent load, every request gets **the same bytes it would
have gotten alone**, **first token in bounded time regardless of what its neighbours are
prefilling**, **an honest bill** (`usage` incl. `cached_tokens`), **a clean typed error or a
partial stream — never a silent 200 with nothing in it**, while the operator gets
**health that measures serving, metrics in a scrapeable format, a drain that finishes
in-flight work, and a process that a bad request cannot take down**. Everything else is
tuning.

---

## 1. Correctness contract (the part nobody else can do for you)

| # | Must-have | Why it is a must-have (evidence) | memra status | Gate |
|---|---|---|---|---|
| 1.1 | **One numeric program per request**: bytes identical solo / batched / joined mid-batch / graph vs eager / spec vs plain | Two shipped defects with exactly this root cause (`research/eosclass-20260813/`, `research/splitiso-20260813/`); a load-history-dependent early EOS is a correctness bug customers see as "the model got dumber under load" | **HAVE** on Step PP-2 and the batched trunk (`docs/SERVING.md:252-344`, `:2634-2640`); Qwen3.5-MoE eager-excluded as defence in depth (`SERVING.md:296-306`) | serve-smoke c=1 vs c=16 byte compare; `tools/iso-gap-gate.sh`, `tick-invariance-gate.sh`, `prime-grid-gate.sh` (fast-gate cmd cells) |
| 1.2 | **Chunk-split invariance**: prefill chunking must not change greedy output | Found 2026-08-05: changing only `MEMRA_PRIME_CHUNK` changed greedy bytes at 97 tokens (`SERVING.md:2446-2470`) | **HAVE** (fixed by default, lane/chunkinv-flip) | `prime-grid-gate.sh` |
| 1.3 | **Exact `max_tokens` and `usage` on visible token ids**, not speculative rounds | Billing truth; `research/honesty-20260809/` | **HAVE/unsealed** (`SERVING.md:2619-2621`) — receipt only, no named script | none named |
| 1.4 | **Launch-config exactness gate on SHORT prompts** | 2026-08-29: a pinned door corrupted logits at prefill; every launch gate passed because gate prompts were ≥613 tokens or one-word answers; short prompts babbled all day (`darklanes research/step37-degen-incident-20260829/INCIDENT.md`) | **PARTIAL** — gates exist; coverage of the short-prompt margin class was the miss | serve-smoke needs a short-prompt greedy arm per model |
| 1.5 | **Template fidelity oracle** (HF-exact rendering, thinking on/off, tools) | memra's own template oracles disagreed with each other (`LAW:oracle-unification-prereq`, darklanes `gpu/models/glimmer.md`) | **PARTIAL** — Rust per-model renderers, no general Jinja (`lib.rs:5043-5045`); reasoning separation HAVE (`toolcall.rs:45-65`) | gemma4 tools gate (5090) |

## 2. API contract (what an unmodified SDK expects)

| # | Must-have | Evidence | memra status | Gate |
|---|---|---|---|---|
| 2.1 | `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/embeddings`, `/v1/rerank`, streaming SSE, `[DONE]` | table stakes | **HAVE** (`lib.rs:5846-5890`) | `tools/serve-smoke.sh` |
| 2.2 | Second and third dialects: `/v1/responses`, Anthropic `/v1/messages` | agentic CLIs speak these; cutover gate G2 requires tools round-trip from a real CLI | **HAVE** (`lib.rs:5870-5874`; unit-tested SSE/tool grammar `responses_api.rs:1238+`, `anthropic.rs:1204+`) | unit only; no live gate |
| 2.3 | **Unsupported params 400 loudly, never silently ignored** | an ignored `n=3` or `logprobs` returns a 200 the client mis-parses | **HAVE** for `logit_bias`, `logprobs`, `n≠1`, `best_of` (`lib.rs:9251-9270`) | unit |
| 2.4 | `logprobs` / `top_logprobs` | eval harnesses, rerank-by-logprob, classifier use; every mainstream engine has it | **MISSING** (rejected 400, `lib.rs:9258`) | — |
| 2.5 | `n>1` / `best_of` | sampling-based eval, self-consistency | **MISSING** (400) | — |
| 2.6 | `stream_options.include_usage` honoured (final usage chunk) | SDK default in streaming billing; "accept and ignore" means a streaming client never sees its bill | **PARTIAL** — parsed, documented cosmetic (`lib.rs:4755-4756`); terminal usage is emitted on the Responses/Anthropic surfaces | — |
| 2.7 | `response_format` json_object / json_schema, compile isolation + timeout | table stakes; a grammar compile must not HOL-block the box | **HAVE** (llguidance, `constrained.rs:451-547`; 5 s compile timeout, `SERVING.md:1653-1717`) | unit; no live gate |
| 2.8 | Tools: `tool_choice` `auto`/`none`/**named**/`required`, **parallel tool calls** | agent frameworks set `tool_choice={"type":"function",...}` and `parallel_tool_calls` routinely | **PARTIAL** — auto/none HAVE; named/required rejected; `parallel_tool_calls` accepted-and-ignored (`responses_api.rs:16-18`) | gemma4 tools gate |
| 2.9 | Stop strings and stop token ids | table stakes | **PARTIAL** — stop strings HAVE (`lib.rs:3138-3179`); stop-token-id param MISSING | unit |
| 2.10 | `/tokenize`, `/detokenize` | clients size prompts for caps/pricing before sending; the 192 MiB body ceiling and context-feasibility rejection make this a real need | **MISSING** | — |
| 2.11 | Typed OpenAI-shaped error JSON + `x-request-id` echo on every error path, incl. 413/429/503/408 | router marketplace contract v2 (`SERVING.md:2629-2633`) | **HAVE** (`lib.rs:314-330`, `anthropic.rs:52-75`) | unit |
| 2.12 | Image input via URL (not only base64) | most SDK examples pass URLs | **PARTIAL** — base64 data URI only (`lib.rs:4108-4174`) | — |

## 3. Scheduler and memory (where "real usage" actually breaks)

| # | Must-have | Evidence | memra status | Gate |
|---|---|---|---|---|
| 3.1 | **Prefill never starves peers** — chunked prefill interleaved with decode as the DEFAULT, bounded tick time | **2026-09-05, 11.5 h of 408s for three tenants**: one 135k-token prefill ran in ONE worker tick (`tick_max_ms 92542`) during which nothing else on the box was served, every queued request hit the deadline; every monitor stayed green (`darklanes ops/incidents/2026-09-05-glm53-cache-eviction-timeouts.md`). Workload archive: GLM in:out 68.6:1, prompt p50 84k — **prefill IS the product** (`darklanes research/sglang-cutover-20260912/CONTRACT.md §3`) | **PARTIAL** — `MEMRA_PREFILL_TICK` caps concurrent prefill at 1024 tok/session, but "a naked sole fresh interactive request may prime up to 8192 tokens in one call" (`docs/FLAGS.md:357`); the fairness door `MEMRA_PRIME_YIELD` is **default OFF** with a decide-by (`FLAGS.md:426`); GLM5 hyper path is eager-only (`memra-kv/src/lib.rs:185-191`) | `research/prefill-fairness-20260908/` receipts; no default-ON serving gate |
| 3.2 | **Prefix cache that is real, reported, and cannot self-evict** | same incident: SLRU protected cohort trapped every new entry in probation → `cold=1` every turn; `cached_tokens` decides the customer's bill ($0.03 vs $0.15/M) and the 2026-09-01 GLM cache defect billed everything at full price | **HAVE/policy-fragile** — exact LCP cache with tenant/salt isolation (`worker.rs:6415-6499`, `SERVING.md:1724-1815`); SLRU losing shape documented in FLAGS; LRU is a launcher choice, not the default | `tools/cache-meter-gate.py`, `spec-on-cache-hit-gate.sh` |
| 3.3 | **Admission keeps DRIVER headroom, not just pool headroom** | 2026-09-05: admission counted pool-cached blocks as free; a large prime asked the driver (cuBLAS workspace, graph instantiation) → step-OOM at prime, `499 client_disconnected` (`darklanes ops/incidents/2026-09-05-long-prompt-driver-oom.md`). Same class: prefix eviction frees to the pool, not the driver (`TRAP:prefix-eviction-frees-to-the-pool-not-the-driver`) | **HAVE** since v0.126.1 (`MEMRA_ADMIT_DRIVER_HEADROOM_MB`, `worker.rs:5861-5930`, `docs/FLAGS.md:337`) | unit on incident numbers; no serving-shape gate |
| 3.4 | **Deadline-aware admission and backpressure**: 429 + `Retry-After` before work, never after; missed TTFT = zero bill | `SERVING.md:1012-1141`; the alternative is a hung client | **HAVE** (`worker.rs:3153-3157`) | `serve-stress-gate.sh` c=64 |
| 3.5 | **Pre-header refusal is a customer error** — a 408 at the admission budget must count in the error counter | 2026-09-09: probe TTFT pinned at the 10 s pre-header budget for an hour while `customer_errors_1h` read 0; 9/25 short requests took 408 and nothing alerted (`TRAP:sentinel-probe-at-the-deadline-reports-zero-errors`) | **PARTIAL** — 408 emitted; the counter is control-plane wiring | — |
| 3.6 | Running-request preemption under KV pressure (swap or recompute) | vLLM/SGLang baseline; without it, admission must be conservative | **PARTIAL** — pause-boundary demotion for retired/tool-paused sessions (`worker.rs:8239-8245`, `:17841-17957`), step-OOM park/requeue (bounded retries); no generic preemption manager | — |
| 3.7 | Long-context KV tiering (device → pinned host → NVMe) | 1M-context products, 84k-p50 prompts | **PARTIAL** — host prefix tier exists (`worker/host_glm.rs`); host tier empties on every deploy (`TRAP:host-tier-empties-on-deploy`); generic spill lane in flight (PR #518, lanes A–D) | HostPrefix identity gate (rented 5090, dev evidence only) |
| 3.8 | Arena growth must not eat long-context admission | first request on a fresh boot claimed the headroom a later deep request needed (`TRAP:slru-arena-eats-long-ctx-admission`) | **PARTIAL** — pinned by launcher, not by default | — |
| 3.9 | Multi-tenant fairness / QoS lanes with per-tenant concurrency | 2026-08: a one-in-flight-per-tenant metering gate 429'd every parallel call from a paying tenant (`TRAP:tenant-gate-429`) | **HAVE** — x-lane interactive/judge/harvest (`prime_fairness.rs:77-91`), per-key concurrency (`auth.rs:116-118`) | `apikeys-gate.sh` (manual) |

## 4. Operability (what the on-call needs from the process)

| # | Must-have | Evidence | memra status | Gate |
|---|---|---|---|---|
| 4.1 | **Health measures serving, not a proxy of it**; a canary that can hang must not latch a fault | **2026-09-13, 28 min down, card healthy**: `[gpu-watch]` nvidia-smi hung once during CUDA-graph capture, `mark_gpu_fault` latched for the process life, `/readyz` stayed 503, router dropped the model (`darklanes ops/incidents/2026-09-13-glm-b200-gpuwatch-false-wedge.md`). Earlier: beat-age-only health SIGTERMed a progressing worker with 22 requests in flight (`FLAGS.md:358`) | **PARTIAL** — `/health`,`/livez`,`/readyz` with loading/idle/busy/draining/fault states and a forward-progress odometer (`health.rs:459-499`, `SERVING.md:1390-1468`); the latch-forever canary is the incident's mechanism and the fix was a launcher env raise, not a default | no fault-injection serving gate |
| 4.2 | **Request-level fault isolation**: a panic in one request must not take peers or the process | 2026-08-25: a 16-token resumed turn hit an assert inside the GPU worker → 20 panics, crash loop, 502 (`darklanes research/incident-dspark-shortsuffix-20260825/`); 2026-09-02: `u32::MAX` candidate in a restore arm panicked the worker; policy "one respawn then exit" is **fleet-fatal** — every in-flight session on the box dies (`TRAP:glm53:fullcover-restore-panics-the-drafter-walk`) | **PARTIAL** — `catch_unwind` + one respawn + exit 70 (`SERVING.md:1390-1468`, `health.rs:217-352`). Poisoned CUDA context is genuinely unrecoverable; Rust-side asserts are not and should fail the request | none |
| 4.3 | **Graceful drain**: SIGTERM → refuse new, finish in-flight, exit 0 within a bound; readiness flips first | blue/green on a pair that cannot overlap needs it; `SERVING.md:1384-1388` | **HAVE** (`lib.rs:5914-5922`, `MEMRA_DRAIN_S` default 30) | no serving-shape gate; final guarantee bullet names no receipt (`SERVING.md:2667-2668`) |
| 4.4 | **Ready only after warmup** (graph capture, first prime) — readiness must not race the canary | 2026-09-13: the fresh process printed `Engine ready` and latched in the same second while `[glm5-tp-sym-graph] engaged: 12 graph piece(s)` was being recorded | **PARTIAL** — readiness gated on load (`lib.rs:5754-5756`); no explicit warmup phase; spec admission needs a warmup request after boot (`TRAP:sse-bursts-not-tokens`) | — |
| 4.5 | **Prometheus exposition** with TTFT / TPOT / ITL / queue-time histograms, KV utilisation, prefix hit rate, per-model labels | universal scrape target; the JSON `/metrics` needs a bespoke exporter for every dashboard | **PARTIAL** — authenticated JSON `/metrics` with counters, step p50/p99, queue/active, prefix cache, pool/device free, spec telemetry (`lib.rs:6452-6667`); no Prometheus text format, no OpenTelemetry, no TPOT/ITL histograms | `cache-meter-gate.py` checks JSON closed form |
| 4.6 | Structured logs with request id, route, model, tenant, phase timings; never `/dev/null` | `LAW:serving-auditable` (a box with stdout on /dev/null had no record of which numeric program loaded) | **HAVE** (stderr, `ttft.rs:175-177`, request ids everywhere) — line-oriented, not JSON | — |
| 4.7 | **Truncated-200 class is observable**: a stream that dies mid-way must be countable | flag battery read a 2.3× faster median while 72 % of requests never finished, all HTTP 200 (`GATE:glm53:fullcover-cell-needs-a-completion-count`) | **PARTIAL** — SSE error event on worker death (`SERVING.md:1099-1111`); no counter for streams closed without `finish_reason` | — |
| 4.8 | **Boot prints the selected program** and every door it armed; a door that does not engage prints that it did not | darklanes#630 burned four rented boots comparing OFF against OFF because a walker door printed nothing on an unhandled route (`FLAGS.md:427`) | **HAVE** — build fingerprint (`SERVING.md:926-1011`), `[prime-walk]` boot line | — |
| 4.9 | Single pinned config surface (env catalog + launcher bytes banked with the deploy) | a box-only launcher edit is not deployed state; the next install silently changes the numeric program (`TRAP:launcher-is-restart-truth`); door renamed → old binary refused by new launcher → auto-restore could not run (2026-09-11 30-min outage) | **PARTIAL** — env-driven (`MEMRA_MODELS`, `MEMRA_ADDR`, …; `docs/FLAGS.md` census enforced pre-push); no config file; binary↔launcher compatibility is not self-checked | `tools/check-flags.sh` |
| 4.10 | Admin seams for the control plane: keyring hot-reload, `/admin/trim`, metadata reload, per-tenant usage counters | metering and blue/green need them | **HAVE** — keyring reload (`SERVING.md:2052-2128`), trim (releases on the second call, `TRAP:admin-trim-releases-on-second-call`), `RuntimeHandles` (`lib.rs:5650-5680`) | `apikeys-gate.sh` |

## 5. Scale-out and model breadth

| # | Must-have | Evidence | memra status | Gate |
|---|---|---|---|---|
| 5.1 | PP / TP / EP that is **topology-valid** (never a TP degree that violates head/KV/expert partitioning) and byte-identical across widths | `AGENTS.md` Step owner lane; `SERVING.md:432-590` | **HAVE** PP-2 sealed; PP-3/4 wavefront experimental; TP-2 on GLM5 with per-rank calibrated prime bands (`TRAP:glm53:tp-quad-prime-band-is-its-own`) | `decode-batch-gate --mode pp`, `prime-split-gate.sh`, `ppn-gate` (multi-card, not in release battery) |
| 5.2 | Multiple models in one process | co-tenancy on one card | **HAVE** (`MEMRA_MODELS` list, `lib.rs:6041-6046`) — with the measured law that an 8B co-tenant breaks a realtime ASR neighbour (`LAW:no-8b-co-tenant-next-to-realtime-speech`) | — |
| 5.3 | Speculative decoding with acceptance telemetry and a restored-session equality gate | acceptance is the sensitive half; bytes cannot catch drafter-context bugs (`GATE:glm53:restored-drafter-context-by-acceptance`) | **HAVE** — MTP/NextN, EAGLE3, DFlash2 (`eagle.rs`, `lib.rs:3566-3592`) | `glm5_dflash_session_gpu` gate 11, `serve-gemma4-spec-gate.sh` |
| 5.4 | Multi-LoRA | common in fine-tune-serving products; optional for a base-model API | **MISSING** | — |
| 5.5 | New-model onboarding cost bounded (days, not weeks) | owner ruling 2026-09-12: "every bringup of a model is costing hours of gpu, late to market, hours of researching and many bugs" (`LAW:total-cost-is-a-selection-criterion`) | **PARTIAL** — ModelPlan compiler + packs (`crates/memra-gguf/src/model_plan.rs`) is the right structure; the per-model serve arms (`dsv4_serve.rs`, `host_glm.rs`, hyper routes) are where the cost still lands | `model verify` ladder |

## 6. Qualification: what turns a feature into a claim

| # | Must-have | Evidence | memra status |
|---|---|---|---|
| 6.1 | **Serving-shape gates in the release battery** — the battery must boot the server and drive HTTP, not only kernels | `tools/release-battery.sh` runs kernel-check / argmax-margin / run-spec only; CI is GPU-less compile+unit (`.github/workflows/ci.yml`); every one of the 15 "serving-contract guarantees" in `SERVING.md:2600-2668` cites a dated research receipt, none names a re-runnable serving gate, and the graceful-shutdown bullet cites nothing | **GAP** — the gates exist (`serve-smoke`, `serve-stress-gate`, `apikeys-gate`, `cache-meter-gate`, `serve-st-gate`, gemma4 batch/tools/spec, iso-gap, prime-grid, tick-invariance) but are not the release bar |
| 6.2 | **Completion counting** in every stress or flag cell: clean / truncated-200 / refused / rode-a-respawn, timing over clean rows only | `GATE:glm53:fullcover-cell-needs-a-completion-count` | not in `serve-stress-gate.sh` |
| 6.3 | **Fault-injection arms** for health: hung canary, wedged worker, OOM at prime, client disconnect mid-stream, SIGTERM with streams open | three of the six 2026-09 outages were a check failing closed on a signal that was not the thing it claimed to measure (`darklanes research/incident-q38md-catalog-20260911/LANE.md`) | none |
| 6.4 | **Short-prompt greedy arm per model** in the launch gate | 1.4 above | missing |
| 6.5 | Interleaved A/B, both orders, N≥5, same window, for any default flip; TTFT/E2E/TPOT/ITL p50/p95/p99 for serving claims | `AGENTS.md` Active accelerator owners | **HAVE** as doctrine; enforced by review, not tooling |

---

## 7. Ranked gaps, by what they have already cost

1. **Prefill fairness default-OFF (3.1) — #521.** One incident, 11.5 h, three tenants, every monitor green. The industry default is chunked-prefill-interleaved-with-decode; memra has the door (`MEMRA_PRIME_YIELD`) and the receipts, and the decide-by date is the deadline to make it the naked default or delete it.
2. **Health/readiness fault-injection (4.1, 4.4, 6.3) — #524 (mechanism: #516).** Two outages (2026-09-11, 2026-09-13) where the engine was healthy and a check was wrong. A latch-forever canary needs a hysteresis or a re-probe; readiness needs a warmup phase that includes graph capture.
3. **Fleet-fatal panic policy (4.2) — #525.** "One respawn then exit" turns a single bad request into a box outage. Rust-side asserts in per-request paths should fail the request with a typed error; only a poisoned CUDA context justifies exit 70.
4. **Release battery has no serving-shape cell (6.1) — #526.** Every serving guarantee is a research receipt. Promote `serve-smoke` + `serve-stress-gate` (with completion counting) + `cache-meter-gate` into `tools/release-battery.sh`, GPU-gated like the rest.
5. **Prometheus + latency histograms (4.5) — #522.** Not an incident cause, but every incident above was detected by the owner or a customer, not a dashboard; TPOT/ITL/queue-time histograms per model are the alerting substrate.
6. **API breadth (2.4–2.6, 2.8–2.10, 2.12) — #527 logprobs, #528 n>1, #529 include_usage, #530 tool_choice/parallel, #531 tokenize, #532 stop_token_ids, #533 image_url.** `logprobs`, `n>1`, `include_usage`, named `tool_choice`, `parallel_tool_calls`, `/tokenize`. Each is small; together they are the difference between "OpenAI-compatible" and "works with my SDK".
7. **Prefix-cache policy fragility (3.2) — #523.** LRU-by-launcher is a fix, not a design; the cache needs an admission rule that guarantees the newest turn fits, and `cached_tokens` needs to be asserted on turn 2 of every replayed conversation (cutover gate G4).

## 8. What this means under the 2026-09-12 ruling

The owner moved the hosted serving path to the SGLang fork and kept memra as the engine
research lane and byte-exact oracle (`LAW:serving-runtime-is-the-sglang-fork`). The rows
above therefore split:

- **Must stay sharp for the oracle role**: 1.1–1.3 (one program, chunk invariance, exact
  accounting), 5.3 (spec acceptance equality), 6.5 (measurement doctrine). These are what make
  memra's captures worth comparing an SGLang stack against.
- **Would have to close before memra serves customers again**: 3.1, 4.1–4.5, 6.1–6.3. None is
  a kernel problem; all are scheduler, lifecycle and gate work, and every one has a dated
  incident behind it.
- **Are the evaluation criteria for the fork too**: this list is the engine half of cutover
  gates G2–G8. The 2026-09 incidents were not memra-specific mechanisms; a fork stack that
  cannot show prefill fairness under a 135k prefill, a warmup-gated readiness, a drain, a
  completion count and Prometheus histograms will reproduce them.

## Sources

- Code: `crates/memra-server/src/{lib,worker,health,auth,constrained,responses_api,anthropic,embed_api,audio_api,ttft,prime_fairness}.rs`, `crates/memra-kv/src/lib.rs`, `crates/memra-sampling/src/lib.rs` at `61be8b0d`.
- Docs: `docs/SERVING.md`, `docs/FLAGS.md`, `docs/TESTING.md`, `docs/API-SURFACES.md`, `tools/release-battery.sh`, `.github/workflows/ci.yml`.
- Incident record (private sibling repo, read-only): `../darklanes/ops/incidents/2026-09-05-glm53-cache-eviction-timeouts.md`, `2026-09-05-long-prompt-driver-oom.md`, `2026-09-11-glm5b200-v0138-gate-crash.md`, `2026-09-13-glm-b200-gpuwatch-false-wedge.md`, `2026-08-31-key-containment-503.md`, `2026-09-01-router-hang-rollback.md`; `../darklanes/research/{step37-degen-incident-20260829,incident-dspark-shortsuffix-20260825,orn-metering-outage-20260826,incident-q38md-catalog-20260911,glm-outage-deploy-fix-20260911,sglang-cutover-20260912}/`; lesson corpus `../darklanes/agent-knowledge/gpu/{serving-economics,fleet-ops,verdicts-ledger}.md` and `gpu/models/{glm53-flash,gemma4}.md`.
- Industry baseline: 2026 engine comparisons agree the feature floor is continuous batching, paged KV, chunked prefill, prefix caching, speculative decoding, structured output, multi-LoRA, OpenAI-compatible HTTP, Prometheus metrics and k8s probes; used only to define "table stakes", not as a scoreboard (`AGENTS.md` positioning rule).
