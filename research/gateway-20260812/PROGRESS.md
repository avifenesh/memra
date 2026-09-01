# cx-gateway Week 2 / Agent Day 2 progress

Date: 2026-08-12
Branch: `lane/cx-gateway`
Pinned base: `main@7f305984b`

## Scope and gates

- [x] Audit the existing HTTPS, auth, OpenAI protocol, usage, overload, isolation, and ledger paths.
- [x] Reuse the existing cx-glue/cx-fleet trial and TLS machinery; wire only verified gaps.
- [x] Generate the OpenRouter provider `/models` schema v2.4 and prepare privacy, retention, and terms publication candidates.
- [x] Add an external synthetic probe, status/incident plan, restart/restore runbook, and off-host manifests.
- [x] Run server tests and local CI for every server-code change.
- [x] Run the two-hour mixed tools/cache/overload soak on the flock-coordinated eu-west Q27 pair.
- [x] Commit raw receipts and `RESULTS.md`, including existed-vs-built and the exact application checklist state.

## Constraints held

- No merge, tag, push, performance-board update, broad formatting, or verification bypass.
- Check GPU ownership with `fuser` and queue behind `cx-q35bug` before GPU stages.
- The owner/orchestrator submits the OpenRouter application; this lane only proves readiness.
- Spill performance and model-quality evidence remain separate; this gateway lane changes neither.

## Timeline

- 2026-08-12: Started. Progress ledger created before the repository audit. Current implementation state and remote availability are not yet claimed.
- 2026-08-12: Audit found the OpenAI streaming/non-streaming, tools, constrained
  output, worker-truth ordinary/cached usage, keyring/caps, early 429, tenant cache
  namespace, generated provider-schema, TLS, lifecycle, and aggregate fleet-meter
  paths already present. The material runtime gap was a durable per-request usage
  and cost ledger; the prior external smoke did not score tools, strict JSON,
  cross-tenant cache behavior, overload, or exact billing.
- 2026-08-12: Added an opt-in fail-closed JSONL receipt ledger. It snapshots the
  model registry's decimal prices, records no request/response bodies or plaintext
  credentials, syncs one terminal row before completion publication, and rejects
  startup when an enabled model has incomplete prices. Focused handler, decimal,
  rejection, and provider-registry tests pass.
- 2026-08-12: Downloaded the official OpenRouter provider monitor schema again on
  this date. It is version 2.4, 64,560 bytes, SHA-256
  `c5ec05a453e262c9c1fd9041ca2624e48b8681ed48df9e73ab5a3642e00675d0`;
  it differs from the prior cached copy, so the probe downloads and validates the
  live schema at run start and every ten minutes during the soak.
- 2026-08-12: Added the full external probe: exact streaming/non-streaming output,
  tool call, strict structured output, serial cache equality, two-key tenant/cache
  isolation, exact 90%-hit groups, eight-way cap-four bursts, current schema
  validation, tenant engine-counter reconciliation, and one-to-one decimal cost
  reconciliation. Nine focused Python tests pass, including normalized SSE
  success and streamed tool-call reassembly. Tool calls and strict structured
  output run in both streaming and non-streaming modes. The long soak keeps seven
  hot 4,860-token entries plus one
  rotating cold entry; each cold insert finishes before the c=4 hit batches, so
  LRU ordering cannot silently turn a declared 90% hit group into a miss.
- 2026-08-12: Prepared static privacy/retention/terms candidates, the status and
  incident publication plan, restart/restore runbook, application checklist, and
  content-free off-host manifest capture. These artifacts do not assert owner
  approval or publication. The checked-in provider prices are explicitly a
  technical-preflight tariff and `is_ready` remains false.
- 2026-08-12: Live eu-west preflight found the GPU flock free and both cards at zero
  compute processes. The prior scratch copy of Q27 had been cleaned; the exact
  scored trunk remains locally available at 15,705,920,064 bytes with the pinned
  `d8d71...d517` SHA-256. It was transferred to the eu-west local NVMe, re-hashed
  there, and promoted atomically only after the byte count and full SHA matched.
  No substitute artifact is authorized.
- 2026-08-12: Closed a catalog/runtime drift found during review: declared provider
  prompt and output maxima are now enforced admission limits. Omitted output length
  is bounded to the listed maximum, larger explicit output or allocation requests
  fail before submission, and rendered/tokenized prompt length is checked before
  cache lookup or GPU admission. `cargo test -p memra-server` passes 202/202.
- 2026-08-12: The remote gate launches two independent Q27 replicas concurrently,
  one on each physical GPU, and scores the complete protocol/cache/overload/billing
  workload against both. This follows the Week-1 Q35 rejection's Q27-replica
  direction while keeping the public edge state separate and honestly unqualified.
- 2026-08-12: Shell syntax, ShellCheck, Python syntax, content-free manifest capture,
  and manifest self-verification pass. The public capture remains honestly red:
  `https://api.tiyuvta.ai/readyz` returns Cloudflare 530, and the live privacy/terms
  pages do not yet match the production candidates or carry final owner approval.
- 2026-08-12: Exact-source local gates are green after the final server changes:
  `cargo test -p memra-server` 202/202; kernel-check 106 cells / ALL GREEN;
  prime gate 8/8; run-spec K=1..8 PASS; run-gen argmax MATCH; serve smoke zero
  failures; c=64 stress ALL GREEN; Q27 acceptance 1/1. `check-flags` still reports
  the two pre-existing `MEMRA_MOESD_*` baseline omissions; the gateway ledger flag
  is documented and no longer appears in the new-name list.
- 2026-08-12: The first paired preflight reached both ready replicas and passed
  the live v2.4 schema plus streaming/non-streaming plain requests. It then failed
  symmetrically because the probe treated `max_tokens = 16` as an exact output
  count; both valid deterministic completions stopped on EOS at three tokens.
  Corrected the short cache/isolation checks to require an exact prompt/cache
  split and a non-empty completion within the OpenAI maximum. The sold-shape
  60-token soak checks remain exact, and the focused probe suite passes 8/8.
- 2026-08-12: Preflight attempt 2 passed schema, plain streaming/non-streaming,
  serial 90%-hit cache, two-tenant isolation, tools, and strict structured
  output on both replicas. Each overload burst then produced the required four
  accepted 60-token responses plus four clean early 429s, but the probe rejected
  two concurrent output hash classes. Week-1 sellgate evidence explicitly
  documents this batched-prime near-tie class: serial cold/hit bytes match,
  cached c=4 introduces no class absent from cold c=4, and cross-composition
  hashes are diagnostic. Aligned the probe to that boundary, retained all hashes
  without response bodies, and added a regression that preserves all eight burst
  receipts without turning the numeric class into an output error. Probe tests
  pass 9/9; semantic plain/tools/strict-JSON and exact serial-cache gates remain.
- 2026-08-12: Preflight attempt 3 passed on both replicas: 56 requests total,
  48 successful completions, eight clean 429s, tools and strict structured output
  in both protocol modes, tenant/cache isolation, exact engine usage, two
  per-request cost-ledger reconciliations, and zero scored errors. Manifest
  verification then caught a post-processing defect before the soak: awk's
  trimmed power fields compared lexically, inverting 29.76/455.54 W. Coerced
  thermal fields to numeric values and added a numeric-order regression; bash
  syntax and ShellCheck remain green. The short preflight will be repeated so
  the scored soak and its thermal receipt share one corrected runner commit.
- 2026-08-12: Corrected-runner preflight passed on both replicas: 56 requests,
  48 successful responses, eight expected early 429s, and zero scored errors.
  The two-hour soak then passed concurrently on the pair. Minimum elapsed time
  was 7,203.033 seconds; 29,448 request ids produced 28,536 successful responses
  and 912 expected 429s with zero protocol, schema, billing, output, or
  unexpected-status errors. The exact cache mix was 24,606 hits plus 2,734
  misses (90%) across equal streaming/non-streaming traffic, with 119 tool-call
  and 119 strict structured-output checks.
- 2026-08-12: Independently replayed both 14k-line request ledgers against the
  captured provider prices and client usage. Every request id had exactly one
  terminal row; ordinary, cached, completion, and total decimal costs matched.
  All long completion hashes stayed inside the frozen Week-1 c1/c4 output
  classes, with no novel or cache-only class. Verified the setup, successful-run,
  failed-attempt, driver, and nested off-host manifests locally, and scanned all
  53,264,418 captured bytes without finding a plaintext generated key or
  non-placeholder bearer credential.
- 2026-08-12: Technical gateway gate is GREEN. OpenRouter application state
  remains NOT READY / DO NOT SUBMIT: public API probes still return Cloudflare
  530, legal pages are not approved/live in their production form, status/paging
  and durable restore are not live, prices/form answers require owner decisions,
  and `is_ready` remains false.
- 2026-08-12: Closed the Hermes steering-2 ledger-integrity merge blocker.
  Streaming receipts now retain worker-truth prompt/cache usage and count each
  consumed token delta, so a client-body drop writes priced partial usage rather
  than null accounting. Unix ledger opens now use `O_NOFOLLOW` and refuse modes
  above the 0640 class. The server-level one-delta disconnect cell produced an
  abandoned 499 row with usage `1/0/1/2` and total cost `$0.0000022890`; its raw
  transcript is `raw/ledgerfix-disconnect-cell.log` with SHA-256
  `38d8c05c6f43526b272f73354faf63faf7477dfda7be61decb2b47d3fcbb7b1e`.
  Focused ledger tests pass 7/7 and `cargo test -p memra-server` passes 206/206.
