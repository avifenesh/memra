# Serving metrics

What `memra-server` reports on `GET /metrics` and `GET /yield/metrics`: the JSON fields, the
Prometheus families, which caller sees which, and how the cache and speculative-decoding counters
are defined. For operators who scrape the server; starting and running it is in
[SERVING.md](../SERVING.md), and the per-request log is in [request-log.md](request-log.md).

## Formats and freshness

- `GET /metrics` answers JSON by default. It answers the Prometheus text exposition
  (`text/plain; version=0.0.4`) when the `Accept` header prefers it: the best `q` among
  `text/plain` and `application/openmetrics-text` must be strictly greater than the best `q` of
  `application/json`. Wildcards count for neither side, so a tie, a wildcard-only header, a
  missing header and a header like `application/json, text/plain, */*` all get JSON. A
  Prometheus server's default scrape header gets text.
- `GET /yield/metrics` answers JSON only (see [/yield/metrics](#yieldmetrics)).
- Counters are cumulative since process start. A restart resets them; nothing resets mid-run.
- The GPU worker owns the counters and copies a snapshot for the HTTP side at the end of a
  scheduler tick: every 32nd tick, and on any tick that admitted a request, retired one, moved the
  speculative counters or moved a host-tier counter. A scrape that races a response's last tick
  can read totals one tick old. The worker's log lines are the per-event record.
- A scrape never queries the device. Device and pool readings are the worker's latest sample,
  and the Prometheus device gauges carry its age.

## Access and tenant scoping

Metrics need a bearer token when the bind is not loopback, when an API key source
(`MEMRA_API_KEY` or `MEMRA_API_KEYS`) is configured, or when `MEMRA_METRICS_TOKEN` is set. An
empty `MEMRA_METRICS_TOKEN` stops the boot. The token never authorizes a completion route. Key
setup is in [SERVING.md](../SERVING.md#api-keys); the variables are in
[FLAGS.md](../FLAGS.md#server-and-api).

| caller | scope | `/metrics` | `/yield/metrics` |
| --- | --- | --- | --- |
| loopback bind, no API key, no metrics token | operator | every field and family | every field |
| `MEMRA_METRICS_TOKEN` | operator | every field and family, all tenant rows | every field |
| `MEMRA_API_KEY` alone (no keyring, no metrics token) | completion domain | process-wide fields and families, all `tenants` and `adsd_suspect_total` rows | every field except `batch_size_last` |
| a key, keyring configured, no metrics token | tenant | only its own `tenants` row and `adsd_suspect_total` entry; the Prometheus body is empty | 403 |
| a completion key while `MEMRA_METRICS_TOKEN` is set | none | 403 | 403 |
| no bearer or an unknown one, auth required | none | 401 | 401 |

With `MEMRA_ALLOW_OPEN_BIND=1`, no key and no metrics token, completion routes are open and both
metrics routes answer 401 to everyone. A tenant row is keyed `t:<tenant>` with a keyring, so a
tenant sees one row across all its `cache_salt` values. Operator-only fields expose cross-tenant
volume, latency, prefix shape and the memory and scheduler state that times a
[Fill-and-Squeeze attack](https://arxiv.org/abs/2602.07878); give the metrics token only to an
operator trusted with all tenants' usage.

## `/metrics` JSON fields

### Process-wide fields

Present for the operator and completion-domain scopes.

| field | meaning |
| --- | --- |
| `admitted`, `completed`, `tokens_out` | requests admitted and retired, and output tokens. Dedicated-route requests are folded into these and into the prompt split |
| `step_p50_ms`, `step_p99_ms` | central-worker decode-step latency over the current window (ticks that advanced an interactive session). Dedicated routes are not in it |
| `prompt_tokens_in`, `cached_tokens_in`, `computed_tokens_in` | the prompt split, see [Cache metering](#cache-metering) |
| `admission_session_defers`, `admission_vram_defers` | admission decisions deferred at the session cap or by the VRAM gate. A queued request counts once per tick it is deferred |
| `admission_vram_hol_holds` | requests held behind an earlier VRAM-deferred request in the same admission pass |
| `admission_drafter_refusals` | DFlash requests the VRAM gate refused their drafter and priced again as plain, once per request |
| `admission_settle_waits`, `admission_settle_wait_ms` | admission passes that requeued an arrival because the reading, less the charges of admitted sessions that own no route state yet, did not hold it while the reading without those charges did (a settle wait, once per pass it waits), and the total milliseconds requests spent in settle waits, from a request's first settle wait to the pass that let it past, before any VRAM defer that follows (memra#1255) |
| `prefill_deadline_ticks` | ticks whose interactive prefill used the earliest-deadline order instead of round robin |
| `prefix_headroom_evictions` | prefix-cache entries evicted to give back the transient reserve after an insert |
| `step_oom_parks` | successful step-OOM requeues |
| `continuation_pool_hits`, `continuation_pool_evictions` | plain-session continuation pool |
| `plain_affinity_rewinds` | continuation resumes rewound to a checkpoint on a turn that rewrote history (subset of hits) |
| `continuation_pool_grid_rewinds`, `continuation_pool_grid_declines` | recurrent-cache hits rewound to their grid checkpoint (subset of hits), and parked ends refused (no checkpoint, under the prime floor, or a failed restore) |
| `spec_pool_hits`, `spec_pool_misses`, `spec_pool_evictions`, `spec_pool_affinity_rewinds` | speculative session pool |
| `spec_pool_sampler_refusals` | spec-pool probes refused because the parked session's sampler differs (subset of misses) |
| `spec_pool_grid_rewinds`, `spec_pool_grid_declines`, `spec_pool_grid_park_skips` | spec-pool hits on a recurrent trunk rewound to an on-grid turn checkpoint (subset of hits); hits refused for lack of one (the request primes cold); recurrent retires freed instead of parked |
| `dspark_pool_grid_declines`, `dspark_pool_grid_park_rewinds`, `dspark_pool_grid_park_skips`, `dspark_pool_grid_rewind_resumes`, `dspark_pool_grid_rewind_grows` | DSpark pool under the grid law: hits refused; retires rewound to their prime boundary and parked; retires freed with nothing to rewind to; resumes of a rewound session; resumes that first grew it to the request's cap |
| `served_plain`, `served_spec`, `served_dspark` | requests by decode program at admission, see [Speculative decoding](#speculative-decoding) |

The pools and the grid law are described in [prompt-cache.md](prompt-cache.md) and
[exactness.md](exactness.md).

### Operator fields

Present only for the operator scope.

| field | meaning |
| --- | --- |
| `cache_hit_token_ratio` | `cached_tokens_in / prompt_tokens_in`, 0 before the first prompt |
| `prefix_cache_hits`, `prefix_cache_misses`, `prefix_cache_inserts`, `prefix_cache_evictions` | device prefix-cache probe outcomes and churn |
| `prefix_cache_hit_tokens` | tokens restored by prefix-cache hits (sum of served entry lengths, a prompt-end tail's tokens included when it served) |
| `prefix_cache_entries`, `prefix_cache_bytes` | resident device entries and bytes (gauges) |
| `prefix_cache_skips_budget`, `prefix_cache_skips_pinned` | snapshots refused before insertion: larger than the whole budget, or a pinned entry that cannot fit beside the in-flight leases. Only the first means the budget can never admit that shape |
| `prefix_cache_seed_grid_refusals`, `prefix_cache_restore_grid_refusals` | prompt-end seeds and longer restores refused by the prime-grid law (the `seed REFUSED (grid)` and `restore grid law` log lines, counted) |
| `prefix_cache_tail_attaches`, `prefix_cache_tail_strips`, `prefix_cache_tail_hits`, `prefix_cache_tail_skips` | [prompt-end tails](prompt-cache.md#prompt-end-tails) attached to grid entries, stripped (for room, or replaced by a newer prompt end on the same entry), restored whole by an exact re-send, and not attached for want of free budget room (a tail never evicts) |
| `lcp_histogram` | `{edges, counts}`, see [Cache metering](#cache-metering) |
| `prefix_host_entries`, `prefix_host_bytes` | pinned-host spill tier residency (gauges; the tier is off unless `MEMRA_KV_HOST_MB` is set) |
| `prefix_host_demotions`, `prefix_host_promotions` | device-to-host and host-to-device moves |
| `prefix_host_demote_ms`, `prefix_host_promote_ms` | cumulative copy wall time; `demote_ms / demotions` is the cost per move |
| `prefix_host_rejected_allocs` | host or device allocation and copy failures during a move. The first pinned-host failure also turns the tier off, with a log line |
| `prefix_host_purges`, `prefix_host_purged_entries`, `prefix_host_purged_bytes` | tenant purges through the deployment's purge handle, and the host entries and bytes they removed |
| `prefix_host_tenant_rejects` | demotions dropped at the per-tenant share cap (`MEMRA_KV_HOST_TENANT_PCT`). A rising count with low pool occupancy is one tenant crowding the pool |
| `prefix_host_tenant_reclaims` | host entries evicted from a tenant's own row to admit that tenant's demotion at its cap (subset of the tier's evictions) |
| `prefix_host_tenant_reclaims_wasted` | reclaims whose demotion then failed to insert. Nonzero is a regression to investigate |
| `prefix_host_pause_demotes`, `prefix_host_pause_cancels` | tool-call pause demotions (`MEMRA_KV_PAUSE_DEMOTE`, subset of demotions), and armed ones the session's next request beat |
| `prefix_host_handoff_exports`, `prefix_host_handoff_imported_entries`, `prefix_host_handoff_imported_bytes`, `prefix_host_handoff_skips` | deploy handoff file (`MEMRA_KV_HOST_HANDOFF`): files written, entries and bytes imported, frames refused on import (each logs its reason) |
| `kv_flex_borrowed_bytes`, `kv_flex_sheds`, `kv_flex_shed_ms` | KV budget flex (`MEMRA_KV_FLEX`): device cache bytes above the configured floor, reclaims that evicted, cumulative reclaim wall time |
| `active_sessions`, `queued_requests` | live sessions and queued requests in the central worker |
| `admission_inflight`, `admission_booked_bytes` | per model: in-flight sessions and the sum of their admission charges |
| `continuation_pool_entries`, `spec_pool_entries` | parked sessions (gauges) |
| `cuda_driver_free_bytes` | the driver's free bytes on the worker's device |
| `cuda_pool_reserved_bytes`, `cuda_pool_used_bytes`, `cuda_pool_cached_bytes` | the worker's CUDA memory pool: reserved from the driver, in use, cached for reuse |
| `kv_vmm_mapped_bytes`, `kv_vmm_reserved_bytes`, `kv_vmm_owed_bytes`, `kv_vmm_graveyard_bytes` | on-demand KV planes: backed and reserved bytes, growth owed to active sessions, bytes pending release |
| `kv_vmm_grows_total`, `kv_vmm_mapper_grows_total`, `kv_vmm_grow_waits_total`, `kv_vmm_grow_failures_total`, `kv_vmm_released_bytes_total`, `kv_vmm_quarantined_bytes_total` | on-demand KV plane events. All `kv_vmm_*` fields stay 0 where the planes are off; they are a card default on the full-power RTX PRO 6000 Blackwell Workstation Edition |
| `routes` | one row per dedicated route, see below |
| `request_faults_total` | guarded worker calls that failed one request and kept the worker |
| `worker_respawns_total` | worker thread respawn attempts |
| `dspark_cold_drafter_grid_refusals_total` | cold-drafter conversions refused because the plain hit was off the prime grid or mispositioned |
| `constraint_compiler_fail_closed` | per model, 1 while the structured-output compiler refuses new compiles (4 timed-out compile workers still running), else 0. Absent until a compiler exists |
| `serve_idle_seconds` | seconds the worker has been fully idle: no active session, no queued admission, no pending handoff |
| `bg` | idle-time background job state (`state`, `yield_mode`, `launches`, `yields`, `resumes`, `preempts`, `ckpt_kills`, `last_yield_signal_us`, `job_pid`, `vram_budget_mb`). Present only when `MEMRA_BG_JOB` is set; see [deployment-binary.md](deployment-binary.md) |
| `budget_source_reload_failed`, `budget_source_reload_consecutive`, `budget_source_available` | health of the deployment's limits source. Present only when a deployment binary's metering seam reports one |
| `dual_pp` | dual-active PP-2 decode: `overlaps`, `slot_pairs`, `slot_uses`, `slot_collisions`, `dropped_timing_samples` and `cuda_event_spans` per stage (`samples`, `total_ms`, `mean_ms`). Present once that path has run |
| `pp_wave` | PP-3/PP-4 wavefront: completed `ticks`, entered `cells`, concurrent-cell `overlaps`. Present once it has run |
| `peer_probe_bypassed`, `peer_probe_boundary_copies`, `peer_probe_runtime_reprobes`, `peer_probe_runtime_failures`, `peer_probe_deferred_total` | multi-card boundary-copy probe counters |
| `peer_probe_integrity_degraded`, `peer_probe_degraded_to_host_bounce` | latched flags: a runtime probe found a bad peer copy, and the boundary fell back to a host bounce |
| `spec`, `spec_tau`, `spec_accept_by_position` | see [Speculative decoding](#speculative-decoding) |

A `routes` row describes one dedicated route (today the DSv4 serving thread), named by its served
model: `name`, `capacity` (its concurrency, 1 for DSv4), `waiting` (reserved, not yet dequeued),
`inflight` (submitted, response not finished), `running`, the outcome counters `admitted`,
`completed`, `failed`, `cancelled` and `refused`, `tokens_out`, the prompt split
(`prompt_tokens_in`, `cached_tokens_in`, `computed_tokens_in`), `rounds`, `service_p50_ms` and
`service_p99_ms` (request wall time) and `round_p50_ms` and `round_p99_ms` (decode or speculative
round cadence).

### Tenant rows

Every scope gets the rows it may see; a field is absent while it has none.

| field | meaning |
| --- | --- |
| `tenants` | per row: `prompt_tokens_in`, `cached_tokens_in`, `cache_hit_token_ratio`. Absent until the first admission |
| `adsd_suspect_total` | per row: acceptance-collapse incidents the ADSD detector counted. Detection only: it never throttles or evicts. The detector is described in [SERVING.md](../SERVING.md#adsd-acceptance-collapse-detector) |

## Cache metering

The cache counters count tokens the worker admitted, and they match the response usage:
`usage.prompt_tokens` and `usage.prompt_tokens_details.cached_tokens` (shapes in
[API-SURFACES.md](../API-SURFACES.md)) are the per-request values these totals sum.

- `prompt_tokens_in` counts every admitted prompt token. `cached_tokens_in` is the subset restored
  from any cache: the continuation pool, a speculative-session resume or the prefix cache. A
  request that rides an in-flight prefill of the same prefix is credited when that prefill lands.
  `computed_tokens_in` is `prompt_tokens_in - cached_tokens_in`, the tokens actually prefilled.
- `cache_hit_token_ratio` is token-weighted: `cached_tokens_in / prompt_tokens_in`.
- `lcp_histogram` takes one sample per prefix-cache probe: the served entry's length on a hit,
  the longest common prefix with any entry on a miss. `edges` are lower bounds,
  `[0, 1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]`, and the last bucket is unbounded. Requests
  that never probe the prefix cache (the speculative tier and other non-batched requests) are not
  in it; their cached tokens still count in `cached_tokens_in`.
- `tenants` rows key on the tenant half of the cache namespace. With a keyring that is
  `t:<tenant>`, one row per tenant across its `cache_salt` values. Without one it is the raw
  `cache_salt`, and `""` is the default namespace. Rows are capped at 256; later namespaces add to
  an `(other)` row, so totals stay exact while a client sending many salts cannot grow the map.

How the tiers, budgets and isolation work is in [prompt-cache.md](prompt-cache.md).

The metering costs nothing measurable on the hot path. On the RTX 5090 Laptop (2026-08-07), the
binary before the counters and the one with them, both resident, interleaved five times in
alternating order with N=100 requests per arm at c=1 (greedy, 64-token generation, prefix-hit
steady state): p50 496.3 against 496.1 ms, p95 502.1 against 501.1 ms
([`research/cache-meter-20260807/`](../../research/cache-meter-20260807/)).

## Speculative decoding

Three surfaces, all counting the same way as `MEMRA_SPEC_STATS`: a chain token cut by the
draft's p-min threshold counts in neither drafted nor accepted.

**Lifetime `spec` block** (operator only). Per model, cumulative since the model loaded (models
load once per process). Absent until the first speculative burst, so a server with speculation off
sends no `spec` key. Values below are illustrative:

```json
"spec": {
  "q9": {
    "rounds": 118, "drafted": 354, "accepted": 213,
    "acceptance_rate": 0.602, "tokens_per_round": 2.805,
    "pos_drafted":  [118, 118, 118],
    "pos_accepted": [96, 71, 46],
    "accept_rate_per_pos": [0.814, 0.602, 0.390]
  }
}
```

`acceptance_rate` is `accepted / drafted`. `tokens_per_round` is `(accepted + rounds) / rounds`,
the target's bonus token included. `accept_rate_per_pos[j]` is the share of rounds that offered
draft position `j` and accepted it. Acceptance walks a prefix (position `j` is accepted only if
`0..j-1` were), so a healthy drafter decays from position 0. Positions are tracked up to 8, and
the arrays stop at the deepest position ever drafted; the totals count deeper drafts too.

**Rolling window** (operator only). `spec_tau` and `spec_accept_by_position` cover the last 30
seconds per model. `spec_tau` is `accepted / rounds`, the mean accepted draft prefix without the
bonus token. `spec_accept_by_position` carries the denominator per position, which matters when
K varies by request:

```json
"spec_tau": { "q9": 1.805 },
"spec_accept_by_position": {
  "q9": {
    "window_seconds": 30.0, "rounds": 118,
    "offered":  [118, 118, 118],
    "accepted": [96, 71, 46],
    "accept_rate": [0.814, 0.602, 0.390]
  }
}
```

Both are absent while the window holds no rounds.

**Per request.** A request that ran speculative rounds carries `usage.spec`: `rounds`, `drafted`,
`accepted` and `acceptance_rate` for that request alone (a resumed session does not carry the
previous request's counts). Requests without speculative rounds have no `spec` key; official
OpenAI SDKs ignore the extra field.

`served_plain`, `served_spec` and `served_dspark` classify each request at admission: plain
decode, a speculative program (MTP or a drafter), or the DSpark route. A later demotion to plain
decode does not move a request between them; demotions are in the server log. They let a check
confirm that speculation engaged on a probe instead of trusting a fluent 200.

The counters are relaxed atomic adds at the round accounting the engine already does: no GPU
sync, no per-token allocation, no lock on the hot path. Validation capture:
[`research/accept-telemetry-20260805/`](../../research/accept-telemetry-20260805/). Speculative
serving itself is in [speculative.md](speculative.md).

## Prometheus families

All histograms are cumulative and unsampled, with fixed buckets. Latency buckets are 0.005, 0.01,
0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120 and 300 seconds, then `+Inf`. Token-count
buckets are 0, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536 and 131072,
then `+Inf`. Labels come only from loaded model names, fixed backend and lane names, device
indexes and bounded status codes; request ids, cache salts, tenant names and rejected model names
never become labels. Each family's lines stay together, so promtool and strict parsers accept the
output.

**Process-wide** (operator and completion-domain scopes), unlabelled:

| family | type | observation |
| --- | --- | --- |
| `memra_requests_admitted_total`, `memra_requests_completed_total` | counter | the JSON `admitted` and `completed` |
| `memra_tokens_out_total` | counter | output tokens |
| `memra_prompt_tokens_in_total`, `memra_cached_tokens_in_total`, `memra_computed_tokens_in_total` | counter | the prompt split |
| `memra_step_latency_p50_seconds`, `memra_step_latency_p99_seconds` | gauge | the JSON step percentiles, in seconds |

**HTTP outcomes** (operator), labelled `{model,route,lane}`, on completions, chat completions,
Responses and Messages. `route` is `hybrid` for the central worker or `dsv4` for a dedicated
route. A body that did not parse or named no served model gets `model="unknown",route="unresolved"`.
`lane` is the scheduler lane the request resolved to (a batch key defaults to `harvest`); a
request refused before that keeps its `x-lane` hint or `interactive`. These count logical HTTP
requests; an internal replay does not add one.

| family | observation |
| --- | --- |
| `memra_requests_total{...,code}` | responses by status code. `499` means the client left before a terminal response |
| `memra_requests_refused_total` | errors returned before a stream opened, client cancellations excluded |
| `memra_pre_header_deadline_total` | HTTP 408 answers |
| `memra_response_errors_total` | error statuses, failed responses and truncated streams, including errors carried inside an HTTP 200 |
| `memra_streams_incomplete_total` | streams that ended without their protocol's terminal event, cancellations included |
| `memra_streams_truncated_total` | the same, cancellations excluded |
| `memra_requests_cancelled_total` | client cancellations before a terminal response |

A response that ends at `max_tokens` is a completed response. A typed terminal error is a failed
response, even when the client closes right after reading it.

**Generation timing** (operator), labelled `{model,route,lane}`. One sample set per worker
attempt: an internal replay is a new attempt. Recorded by the central worker (`route="hybrid"`);
capture-only embedding and rerank work is excluded.

| family | type | observation |
| --- | --- | --- |
| `memra_queue_wait_seconds` | histogram | worker submission to admission |
| `memra_ttft_seconds` | histogram | worker submission to the first accepted token event, queue wait included |
| `memra_e2e_seconds` | histogram | worker submission to a successful finish, queue wait included |
| `memra_tpot_seconds` | histogram | intervals between accepted token events; N events give N-1 samples |
| `memra_prefill_tokens` | histogram | prompt tokens, cache reads included, once per admitted attempt |
| `memra_completion_tokens` | histogram | generated tokens of successful attempts, length-limited ones included |
| `memra_cached_tokens_total` | counter | cache-restored prompt tokens, including credit given later to followers of an in-flight prefill |
| `memra_emitted_token_events_total` | counter | accepted token events, including those of requests later cancelled or failed |

The emission clock stops at the worker's successful send of each token event, empty UTF-8
fragments included. It does not time network flushes.

**Capacity** (operator):

| family | labels | observation |
| --- | --- | --- |
| `memra_kv_used_bytes` | `{model,route}` | backed bytes of the canonical KV, recurrent, indexer and draft state planes of live and parked sessions |
| `memra_kv_capacity_bytes` | `{model,route}` | reserved capacity of those planes, VMM allocation granularity included |
| `memra_active_sessions`, `memra_queued_sessions` | `{model,route}` | active and admission-waiting sessions |
| `memra_device_free_bytes` | `{device}` | the driver's free bytes on each device a model owns, at the worker's last sample |
| `memra_device_memory_sample_age_seconds` | `{device}` | age of that sample |

The KV gauges exclude prefix snapshots, rollback copies and graph workspace. They publish with the
worker snapshot and go to 0 when the worker exits. They cover the central worker only, not a
dedicated route's allocations.

**Prefix cache and faults** (operator):

| family | type | observation |
| --- | --- | --- |
| `memra_backend_prefix_cache_hits_total`, `_misses_total`, `_inserts_total`, `_evictions_total` | counter | device prefix-cache probes and churn, labelled `{route="hybrid",tier="device"}`; evictions include superseded entries |
| `memra_backend_prefix_cache_bytes` | gauge | retained bytes, same labels |
| `memra_prefix_cache_hits_total`, `_misses_total`, `_inserts_total`, `_evictions_total`, `_hit_tokens_total` | counter | the same counters without labels, and the hit-token mass |
| `memra_prefix_cache_bytes` | gauge | device prefix-cache bytes |
| `memra_cache_hit_token_ratio` | gauge | the JSON `cache_hit_token_ratio` |
| `memra_request_faults_total`, `memra_worker_respawns_total`, `memra_dspark_cold_drafter_grid_refusals_total` | counter | the JSON fields of the same names |

**Dedicated routes** (operator), labelled `{route}` with the route's served model name:

| family | type | observation |
| --- | --- | --- |
| `memra_route_requests_total{route,code}` | counter | terminal outcomes; `code` is `completed`, `failed`, `cancelled` or `refused` |
| `memra_route_tokens_out_total` | counter | output tokens |
| `memra_route_waiting`, `memra_route_running`, `memra_route_capacity` | gauge | the `routes` row values |
| `memra_route_queue_wait_seconds` | histogram | reservation to dequeue |
| `memra_route_e2e_seconds` | histogram | successful admission to completion |
| `memra_route_round_seconds` | histogram | compute-round latency, not a token-gap metric |

**Central worker per model** (operator), labelled `{model}`. These keep their original
admission-based definitions; the generation timing families above start at submission instead.

| family | observation |
| --- | --- |
| `memra_hybrid_queue_wait_seconds` | queue to admission, once per admitted request |
| `memra_hybrid_ttft_seconds` | admission to the first committed token |
| `memra_hybrid_e2e_seconds` | admission to a successful finish; failed, aborted and OOM-torn-down sessions are excluded |
| `memra_hybrid_token_gap_seconds{model,lane}` | intervals between accepted token events, empty text and EOS ids included; queue and prefill excluded |

A step-OOM park replay that had already emitted a token records a second
`memra_hybrid_ttft_seconds` sample for the same client request.

**Request log** (operator), present only when `MEMRA_REQUEST_LOG` is set:
`memra_request_log_lines_total{outcome}` with `outcome` `written`, `dropped_queue_full` or
`dropped_write_error` ([request-log.md](request-log.md)).

## `/yield/metrics`

Per-lane QoS counters and the interactive step latency, the receipts endpoint of the x-lane QoS
gate (the lanes and shedding are in [SERVING.md](../SERVING.md#lanes-x-lane)). Operator and completion-domain
scopes only.

```json
{
  "lanes": {
    "interactive": { "admitted": 0, "shed": 0, "completed": 0, "tokens_out": 0 },
    "judge":       { "admitted": 0, "shed": 0, "completed": 0, "tokens_out": 0 },
    "harvest":     { "admitted": 0, "shed": 0, "completed": 0, "tokens_out": 0 }
  },
  "interactive_step_ms": { "p50": 0.0, "p99": 0.0 },
  "batch_size_last": 0
}
```

`batch_size_last` (the last decode batch size) is operator only.

## Checking the exposition

`tools/cache-meter-gate.py` checks the cache accounting against closed forms on a fresh server.
It sends N requests that share a K-token `prompt_ids` prefix plus one under another `cache_salt`,
then checks each response's `cached_tokens`, the `/metrics` totals, the LCP histogram buckets and
the tenant rows. Boot the server with `MEMRA_SERVE_SPEC=0` so every request takes the batched tier
the prefix cache serves, and with no prior traffic:

```bash
MEMRA_SERVE_SPEC=0 MEMRA_MODELS=m=/path/model.gguf memra-server &
python3 tools/cache-meter-gate.py http://127.0.0.1:8080 m \
  --prometheus --promtool /path/to/promtool --raw-out cache-meter.jsonl
```

`--api-key-file` and `--metrics-token-file` pass separate completion and operator credentials to
a keyed server. With `--prometheus` the gate also runs `promtool` on the text scrape and checks
the token-volume histograms, capacity cleanup, label bounds and that the JSON body is unchanged.
`tools/metrics-live-gate.py` adds clean, queued, refused and cancelled request checks on the
outcome families. `tools/serve-smoke.sh` runs the cache gate as its cache-metering arm. Receipts:
[`research/cache-meter-20260807/`](../../research/cache-meter-20260807/) and
[`research/cache-metrics-20261002/`](../../research/cache-metrics-20261002/).
