# cx-gateway Week 2 / Agent Day 2 results

Date: 2026-08-12
Branch: `lane/cx-gateway`
Pinned base: `main@7f305984b`
Scored implementation tip: `3ff7b536e634c20b59ad7eac37dd4b8caa704a40`

## Verdict

The production-gateway implementation, exact-source local gates, scored eu-west
pair preflight, and two-hour pair soak are **GREEN**. Both independent Q27
replicas ran beyond 7,200 seconds with zero protocol, schema, billing, output,
or unexpected-status errors. This is a technical gateway PASS on the loopback
pair; it does not claim the public edge, legal pages, status service, or durable
production restore path are live.

The OpenRouter application is **NOT READY -- DO NOT SUBMIT**. The technical soak
is green, but the public API currently returns Cloudflare 530, the live privacy
page describes a different trial policy, the live terms page says it is not in
force, and the status/paging plus durable restore drill are not live. Only the
owner/orchestrator submits after every checklist item is green.

## Audit: reused versus built

| Requirement | Already present and reused | Gap closed in this lane |
| --- | --- | --- |
| Stable HTTPS and lifecycle | `deploy/glue` Cloudflare Tunnel, key tooling, trial up/down, and the RunPod provision/systemd pattern | Q27 overlay, public-state capture, runbook, and off-host manifest procedure; public publication remains blocked |
| Authentication and tenant limits | Hashed bearer-key keyring, revoke/issue tooling, and per-key concurrency caps | Paired preflight provisions two isolated c=4 tenants and retains only the content-free keyring hash |
| OpenAI protocol | Streaming/non-streaming chat and text completions, tools, constrained structured output, terminal usage | External scoring for both protocol modes, including streamed tool-delta reconstruction and strict JSON validation |
| Usage and caching | Worker-truth prompt/completion/cached-token usage and tenant-scoped prefix-cache namespaces | Exact client/engine reconciliation, serial 90% hit proof, and same-salt cross-tenant miss/hit proof |
| Overload | Early key-cap admission and OpenAI-shaped 429 responses | Eight-way c=4 bursts require accepted work plus clean 429 body, retry headers, rate-limit headers, and request ids |
| Durable billing | Aggregate fleet meter existed; no durable request-level cost receipt existed | Opt-in fail-closed JSONL ledger with exact decimal ordinary/cached/completion cost, terminal sync, request ids, and no bodies or plaintext credentials |
| Provider catalog | Generated OpenRouter provider document and model registry existed | Q27 v2.4 metadata, live admission enforcement for advertised prompt/output limits, and validation against a freshly downloaded official schema |
| Operations | Trial lifecycle and metric-token lanes existed | Status/incident publication plan, safe external probe, restart/restore procedure, and content-free artifact/config manifest capture |

No model bytes, quantization, kernel defaults, performance-board values, spill
paths, or product/business documents changed in this lane.

## Local verification

The final exact-source local battery is retained under `raw/local/`:

- `cargo test -p memra-server`: 202 passed, zero failed;
- external probe tests: 9 passed, zero failed;
- `kernel-check`: 106 cells, ALL GREEN;
- prime gate: 8/8;
- `run-spec` K=1..8: PASS;
- `run-gen` argmax: MATCH;
- serve smoke: zero failures;
- c=64 stress: ALL GREEN; and
- Q27 acceptance: 1/1.

`check-flags` confirms the new request-ledger flag is documented. Its remaining
two findings are the pre-existing base omissions for `MEMRA_MOESD_CACHE_CAP` and
`MEMRA_MOESD_CACHE_DEPTH`, not gateway additions.

## Provider schema and public-state receipts

The [official OpenRouter provider monitor schema](https://openrouter.ai/docs/assets/provider-monitor-schema-v2.openapi.json)
was downloaded again on 2026-08-12. It declares v2.4, is 64,560 bytes, and has SHA-256
`c5ec05a453e262c9c1fd9041ca2624e48b8681ed48df9e73ab5a3642e00675d0`.
It differs from the prior cached schema, so the probe downloads and validates
the current bytes at run start and every ten minutes during a long soak.

The live application form snapshot has SHA-256
`c1030d86900e7ff0db35f880786c6af981d4e595b5ee1be090be8f08daa4240c`.
All 15 visible fields were required on 2026-08-12. Public probes captured
Cloudflare 530 from both `https://api.tiyuvta.ai/readyz` and
`https://api.tiyuvta.ai/models?schema=openrouter`; this is a public launch
blocker, not a remote-loopback protocol result.

## Eu-west Q27 pair soak

**PASS.** The corrected-runner preflight and scored two-hour soak both passed on
runtime commit `3ff7b536e634c20b59ad7eac37dd4b8caa704a40`:

| Receipt | Pair result |
| --- | ---: |
| Minimum replica elapsed time | 7,203.033 seconds |
| Requests with durable terminal rows | 29,448 |
| Successful responses | 28,536 |
| Expected early-overload 429s | 912 |
| Scored protocol/schema/billing/output errors | **0** |
| Exact 90%-hit cache requests | 27,340 = 24,606 hits + 2,734 misses |
| Cache protocol mix | 13,670 streaming + 13,670 non-streaming |
| Tool-call checks | 119 = 60 streaming + 59 non-streaming |
| Strict structured-output checks | 119 = 60 streaming + 59 non-streaming |
| Overload bursts | 228; every burst was four 200s + four clean 429s |
| Thermal regime | N=2,884 five-second GPU rows; 26--63 C; 29.68--512.62 W |

The staged artifact is the pinned 15,705,920,064-byte Q27 GGUF with SHA-256
`d8d71c7e8a01a1c964fd904a7b496eaef19bdd66827e0949e66c723da742d517`.
The runner uses two independent replicas, one per physical RTX PRO 6000 GPU,
with c=4 per tenant, spec off, context 8,192, and a 4,096 MiB prefix cache per
replica. It runs the same streaming/non-streaming, tools, strict structured
output, exact 90%-hit cache, isolation, overload, schema, output, and billing
checks concurrently against both replicas. A loopback pair result does not
assert that a public load balancer or edge is live.

The ledger replay matched every request id one-to-one: 28,536 completed rows and
912 zero-cost rejected rows. Completed usage totaled 137,425,863 prompt tokens,
split into 123,994,546 cached and 13,431,317 ordinary prompt tokens, plus
1,701,895 completion tokens. At the explicitly non-commercial technical
preflight tariff, exact decimal costs were $4.029395100 ordinary prompt,
$9.299590950 cached prompt, and $3.403790000 completion, totaling
$16.732776050. These are reconciliation receipts, not approved customer prices.

The raw counter records 22,776 concurrent output hashes that differ from the
serial c=1 golden. This is the already-qualified Week-1 batched-prime near-tie
boundary, not a suppressed error: every 60-token output was one of exactly the
three frozen classes already observed by the sell gate (the c=1 class plus the
two c=4 classes), every miss retained the c=1 class, no cache-only or novel
class appeared, and the semantic, serial-cache, `run-gen`, and K=1..8 `run-spec`
gates passed. This result does not claim byte identity across batching
compositions.

All raw request receipts, cost ledgers, unfiltered server logs, five-second GPU
and VM traces, setup provenance, failed preflight attempts, successful
preflight, final soak, nested off-host manifests, and independent local replay
are under `raw/`. The content scan found no plaintext generated key, key-value
assignment, or non-placeholder bearer credential in the 53,264,418 captured
bytes. A second pass over the staged Git blobs verified all 553 references in
21 remote SHA-256 manifests, including the captured HTTP CRLF bytes.

## Exact OpenRouter application readiness state

The status below maps every required field from the captured 2026-08-12 form.
`Technical` means the repository has evidence but the owner still controls the
submitted answer. `Blocked` means the application must not be submitted.

| # | Required field | State | Evidence / remaining decision |
| ---: | --- | --- | --- |
| 1 | Company Name | Owner required | Owner supplies the legal/company name. |
| 2 | Website | Technical | `https://tiyuvta.ai` is live; owner confirms it is the submitting company site. |
| 3 | Your Email | Owner required | Must be a company email and will receive the Slack Connect invitation. |
| 4 | Display Name | Owner required | Owner confirms the user-facing provider name. |
| 5 | Desired Slug | Owner required | Owner confirms a lowercase slug and OpenRouter availability. |
| 6 | Distinguishing Features | Technical / owner required | Evidence can support low latency, high throughput, and unique infrastructure; owner chooses the claims. |
| 7 | Extra Details | Owner required | Owner approves the hardware, caching, architecture, volume, and rate-limit copy. |
| 8 | URL to `/models` API | **Blocked** | Both loopback replicas validated against official schema v2.4 twelve times during the soak, but the intended public URL `https://api.tiyuvta.ai/models?schema=openrouter` still returns Cloudflare 530. |
| 9 | API Base URL | **Blocked** | Both loopback replicas pass the complete protocol soak, but the intended public base `https://api.tiyuvta.ai` still returns Cloudflare 530. |
| 10 | URL to Privacy Policy | **Blocked** | Candidate exists, but the live page describes trial prompt/response retention and training use; owner approval/publication is absent. |
| 11 | URL to Terms of Service | **Blocked** | Candidate exists, but the live terms say draft/not in force and leave operator/governing-law details unresolved. |
| 12 | Data Policy | **Blocked** | Content-free gateway design is documented, but the production policy needs owner approval and publication consistent with field 10. |
| 13 | Supported Output Modalities | Technical | Text only for this Q27 offer; owner confirms the selection. |
| 14 | Inference Location | Technical | Registry and pair evidence identify DE / Frankfurt; owner confirms the form wording. |
| 15 | HQ Location | Owner required | Owner confirms the legal headquarters location. |

## Non-form launch blockers

- `q27-models.toml` deliberately remains `is_ready = false`.
- The checked-in prompt/cache/completion prices are a technical-preflight tariff,
  not an owner-approved commercial offer.
- `status.tiyuvta.ai`, its outside-failure-domain five-minute probe, the paging
  destination, and backup contact are not captured live.
- The research host has local NVMe but no production `/data` durable mount,
  Cloudflare connector unit, or hardened production systemd unit. No restore
  drill from an off-host manifest has therefore passed.
- The production candidate privacy, retention, and terms pages have not been
  owner-approved or deployed.

The application becomes eligible for owner/orchestrator submission only after
all public/legal/operations blockers above are closed, owner-controlled form
answers and pricing are confirmed, `is_ready` is changed deliberately, and the
safe probe is repeated through public HTTPS. The technical pair soak is no
longer a blocker.

## Ledger-integrity addendum: disconnect settlement

The post-soak Hermes steering-2 audit found that a dropped streaming response
still emitted an `abandoned` ledger row with null usage and cost. That merge
blocker is closed. The worker now publishes authoritative prompt and cached
prompt counts at admission, the response consumer advances partial completion
usage once per token delta, and `PendingReceipt::drop` prices the accumulated
usage with the same exact-decimal schedule used by completed rows. Completed and
rejected terminal-row behavior is unchanged.

The append open is also hardened on Unix: the final path component is opened
with `O_NOFOLLOW`, the opened object must be a regular file, and permissions may
not exceed the 0640 class. Focused tests accept 0600/0640, reject 0660, 0644,
and 0610, and reject a final-component symlink.

The server-level disconnect cell dropped the response body immediately after
receiving its first SSE data delta. Its synced row was `outcome=abandoned`,
HTTP 499, with prompt/cached/completion/total usage `1/0/1/2` and exact total
cost `$0.0000022890`. The complete unfiltered test transcript is
`raw/ledgerfix-disconnect-cell.log` (11 lines, SHA-256
`38d8c05c6f43526b272f73354faf63faf7477dfda7be61decb2b47d3fcbb7b1e`).
The focused ledger suite passed 7/7 and the final `cargo test -p memra-server`
gate passed 206/206. Per steering, this focused disconnect cell replaces a
second two-hour run; the original pair-soak result above is unchanged.
