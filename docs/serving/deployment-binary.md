# Embedding memra-server in a deployment binary

This page is for developers who build their own server binary on the `memra-server` crate: the
wiring entry point, the metering seam and its terminal outcomes, the background job store, the
runtime handles, the policy contracts that serve routes register (with the dedicated DSv4 route as
the one example), and the idle-time job runner. The stock `memra-server` binary is this crate with
the stock wiring; the operator guide is [Serving](../SERVING.md).

## Entry points

- `memra_server::serve_main()` is the stock binary. It validates argv against the stock set
  (`argv::validate`; an unknown token exits 2 with the token named), then runs
  `serve_with(ServerWiring::stock())`.
- `memra_server::serve_with(wiring)` is what a deployment binary calls. It does not validate
  argv, so the binary parses its own command line first; it may call `argv::validate` itself for
  the stock refusal. `serve_with` still answers `--version` and the keyring CLI
  (`--gen-key`, `--revoke-key`) and exits before any engine work.

## `ServerWiring`

| Builder | Effect |
|---|---|
| `ServerWiring::stock()` | the open engine: no accounting, no admin listener, no capture |
| `ServerWiring::with_metering(factory)` | admission and accounting come from the factory (below) |
| `.with_job_store(Arc<dyn JobStore>)` | background results live in this store instead of the in-memory one |
| `.claiming("MEMRA_...")` | the binary consumes this deployment-only variable itself (next section) |
| `.on_ready(hook)` | `hook(RuntimeHandles)` runs once, when the worker is live; it is not awaited |

Authentication stays in the engine (`memra_server::auth`, `MEMRA_API_KEY` and `MEMRA_API_KEYS`):
the key file, tenant identity, per-key lane class and rate limits apply under every wiring.

## Deployment-only environment

The stock binary refuses to boot when any of these is set, because it carries no surface that
reads them and set-but-unread configuration must not fail open:

`MEMRA_REQUEST_LEDGER`, `MEMRA_TENANT_BUDGETS`, `MEMRA_ADMIN_ADDR`, `MEMRA_ADMIN_TOKEN_FILE`,
`MEMRA_CAPTURE_DIR`.

The refusal is `FATAL: <name> is a deployment-binary surface; this build ships no
accounting/admin/capture`. A custom wiring disarms it for exactly the names it passes to
`claiming`. The check runs after the metadata file loads and before the worker starts.

## The metering seam

`memra_server::metering` speaks tokens and verdicts, never money: what admission means (budgets,
prices, tenancy policy) belongs to the implementation.

- `MeteringFactory` builds the implementation once the models are loaded, from `MeteringInit {
  models }` (the served aliases). `Ok(None)` means no accounting; `Err` is a boot FATAL, so
  accounting configuration never fails open.
- `Metering`: `enforces_limits()` (false admits every authenticated tenant without a reservation),
  `is_limited(tenant)`, `reserve(tenant, principal, model, prompt_tokens, completion_bound)`
  returning an optional opaque `Permit`, `open(meta, permit)` returning the request's `Receipt`,
  `captures(tenant)`, `limits_health()` for the metrics surface, and `drain_kill()`.
- `RequestMeta` carries the request id, tenant, the key's non-secret prefix (`principal`), model,
  route, lane, stream flag, the effective completion bound and the reserved context.
- `Receipt` records prompt usage at admission (`record_prompt_usage`) and each completion token
  (`record_completion_token`), then settles exactly once through one of the calls in the table
  below. `UsageCounts` (`prompt_tokens`, `cached_prompt_tokens`, `completion_tokens`) is worker
  truth.
- A `Permit` dropped without being consumed must release what it holds.

Admission refusals map to fixed HTTP answers:

| Verdict | HTTP | `code` |
|---|---|---|
| `AdmitError::Insufficient`, `AdmitError::Blocked` | 402 | `insufficient_balance` |
| `AdmitError::Unenrolled`, or `is_limited` false while limits are enforced | 402 | `tenant_not_enrolled` |
| `AdmitError::PrincipalCapped` (the key's own cap) | 402 | `key_spend_cap_reached` |
| `AdmitError::Unavailable` | 503 | `tenant_budget_unavailable` |
| a `Receipt` call returns `Err` | 500 (an error event after stream headers) | `request_ledger_unavailable` |

### Terminal outcomes

Every request settles its receipt through exactly one path. Only the first three carry usage;
an implementation must charge nothing for the rest.

| Terminal path | Seam call | Usage |
|---|---|---|
| The response was delivered in full | `complete(usage, elapsed)` | the full counts |
| `timeout_ms` cut generation after tokens, or a background job was cancelled after tokens; the produced tokens are delivered | `complete_deadline_partial(usage, elapsed)` | the delivered counts |
| The client disconnected mid-generation | none: the receipt is dropped | the tokens recorded so far |
| Refused or failed before or during generation, including a worker that dies mid-stream (`worker_fault`, `worker_channel_closed`) | `reject(status, code)` | none |
| The deadline passed before the first token or before stream headers | `settle_unbilled("deadline_exceeded", 408, ...)` | none |
| Admission shed: the wait cannot fit the deadline, the queue bound is reached, or the opt-in queue-wait ceiling (`MEMRA_QUEUE_WAIT_CEILING_S`) is exceeded | `settle_unbilled("shed_deadline" \| "shed_queue" \| "shed_queue_wait", 429, ...)` | none |
| A background job was cancelled before any token | `settle_unbilled("cancelled", 409, ...)` | none |
| A background result could not be stored | `settle_unbilled("background_storage_failed", 503, ...)` | none |
| The drain deadline passed with requests in flight | `drain_kill()` once, then the receipts drop | none |
| A handler panicked while holding the receipt | the receipt drops while the thread unwinds | none |

A dropped receipt therefore has three causes, and the implementation tells them apart: after
`drain_kill()` every drop is the server's shutdown, a drop with `std::thread::panicking()` true
is a crash, and any other drop is a client walk-away. A mid-stream worker death reaches the
client as the same error object the blocking path returns, never as a stream that just stops.
The deadline rules themselves are in [Serving](../SERVING.md).

## Background job store (`JobStore`)

`background: true` requests ([HTTP API](../API-SURFACES.md#background-jobs-background)) keep
their result in a `JobStore`. The stock store is `job_store::InMemoryJobStore`: process-local, a
terminal result kept for `MEMRA_BACKGROUND_JOB_TTL_SECS` (default 900) after the job ends, under a
resident cap of `MEMRA_BACKGROUND_JOB_MAX_BYTES` (default 64 MiB). The TTL starts at the terminal
state and polls do not consume the result.

- `put`, `get`, `take`, `cancel` and `sweep` hold one record per opaque, tenant-scoped key
  (length-prefixed tenant and id). Keep the whole key. A new key may only enter as `Queued`; a
  write to a terminal record is refused (`AlreadyTerminal`), and so is a write to an unknown or
  evicted key (`NotFound`), so a late worker cannot resurrect a cancelled job.
- `reserve_output(id, bytes)` reserves working output before the server buffers it (every
  record reserves at least 256 bytes), and `publish_terminal(id, record, settle)` stores the
  terminal record, runs the settlement callback and exposes the result only after it succeeds.
  Both default to `PublicationUnsupported`: a store without them makes background delivery fail
  visibly before any usage settles. A result that does not fit settles
  `background_storage_failed` with no usage.
- A shared store does not move a live generation or its cancellation to another process. Route
  cancel requests to the process that owns the generation; restart recovery belongs to the
  deployment.

Design record: [COMPLETE-RESULT-PATH-V1](../decisions/COMPLETE-RESULT-PATH-V1.md); default:
[BACKGROUND-DELIVERY-DEFAULT](../decisions/BACKGROUND-DELIVERY-DEFAULT.md).

## Runtime handles (`on_ready`)

The hook receives `RuntimeHandles` once the models are loaded and the worker accepts commands.
The stock server exposes none of these over HTTP; a deployment's admin surface does.

| Field | Call | Contract |
|---|---|---|
| `trim: TrimHandle` | `trim().await` | the worker trims its pools and returns its report; errors if the worker is down or silent for 30 s |
| `metadata_reload: MetadataReloadHandle` | `reload()` | re-reads the boot `MEMRA_MODEL_METADATA` path with the boot validation; returns `path`, `sha256`, `models`, `has_provider`; a failure keeps the old set ([model metadata](model-metadata.md#loading-the-file)) |
| `purge: PurgeHandle` | `purge_tenant(tenant).await` | drops a tenant's parked KV state; call it on key revocation and tenant deletion with the keyring tenant id; `device_pinned_left > 0` means live sessions still lease entries, so call it again after they drain; 30 s timeout |
| `kv_handoff: HostHandoffHandle` | `export(force).await`, `import().await` | moves the host KV tier across a restart through the file `MEMRA_KV_HOST_HANDOFF` names (inert without it); export runs on a drained process (it refuses under traffic unless `force`; 900 s timeout); import validates the header, answers, then restores entries one per tick (`prefix_host_handoff_*` in `/metrics`) |
| `shutdown: watch::Receiver<bool>` | | turns `true` when the graceful drain completes |

`TrimHandle`, `PurgeHandle` and `HostHandoffHandle` each wrap a worker command sender, and the GPU
worker exits only when every sender is dropped. A deployment surface must end and drop all three
on the shutdown signal, or the process hangs at exit. `MetadataReloadHandle` touches memory only
and needs no drop.

## Serve route contracts

A process has up to two serve routes: the central worker and a dedicated route, today only the
DSv4 thread (`dsv4_serve.rs`). Both share the request and event contract, so auth, metering, tool
parsing and every API dialect behave the same on each. Policies that read a side channel instead
need a writer on every route, and each route declares them in a `RouteContract`
(`route_contract.rs`).

- The nine policy surfaces: `capacity`, `occupancy`, `progress`, `fault-ownership`,
  `shutdown-ownership`, `memory-cost`, `rewrite-qualification`, `prime-fairness`,
  `service-metrics`. A route declares each one implemented (with a call-site token a unit test
  finds in the route's source) or refused by name with the issue that owns the gap.
- An undeclared surface fails the registry check before the worker reports ready
  (`FATAL: worker init failed`). A policy the operator armed that no route honors refuses the
  boot with the route named; today that is `MEMRA_REWRITE_BUNDLE` in a process whose only route
  is DSv4. A mixed process boots, and the bundle governs the central worker.
- Each boot prints one line per route:
  `[route-contract] model=... route=... capacity=... implemented=[...] refused=[...]`. The DSv4
  route refuses `rewrite-qualification` and `prime-fairness` and implements the other seven.

## The dedicated DSv4 route

DeepSeek-V4-Flash serves on its own thread and engine ([model card](../models/deepseek-v4-flash.md)).
The route owns its health, admission and memory charge.

- **Health.** The route registers its own record beside the central worker's and is judged with
  the same stall bound (`MEMRA_HEALTH_STALL_S`). Its phase is `loading` until the thread first
  idles, then `idle` or `busy`, and `dead` once the thread exits. Forward progress is the
  thread's prime odometer plus one stamp per decode step or speculative round; a busy route with
  no fresh signal past the bound is stalled and turns `/health` red with the route named, and
  `/readyz` stays not ready while any route is loading. `/health` reports the process `phase`
  (busy while any route serves), the central worker's `scheduler_phase` and a `routes` array. A
  caught per-request panic counts in `request_faults` and the thread keeps serving.
- **Admission.** The route's requests are admitted against its own book, not the central lane's
  session cap. Its capacity is its serving lanes (`MEMRA_DSV4_SESSIONS`, 1 to 16; unset: 16 on
  the plain TP/EP route, 2 on a plain PP-2 load, 1 with a drafter), which is also its
  `X-RateLimit-Limit`. The queue bound per lane is `MEMRA_MAX_QUEUE_DEPTH`, default four times
  capacity; the wait estimate is one
  request's decode on the route times the waves ahead, without prime time, which once priced
  short requests behind a long prime out of their deadlines
  ([receipt](../../research/dsv4-route-receipt-20260926/)). A reservation ticket rides the
  request and releases at dequeue. `/metrics` adds a `routes` array (`capacity`, `waiting`,
  `inflight`, `running`, the admitted, completed, failed, cancelled and refused counters, token
  counters, `service_p50/p99_ms`, `round_p50/p99_ms`).
- **Memory charge** (`dsv4_admit.rs`). Before any state is allocated or a parked prefix is
  consumed, the route charges each owning card: the planned session cache plus a fixed
  per-session term measured at boot on a 1024-token calibration session, plus the C4 gathers when
  the host C4 tier is on; stages on one card sum, and the host tier charges against
  `MemAvailable`. The decision is the shared rule: device fit first, then the host tier with LRU
  eviction of parked entries, then a defer that re-reads memory every 50 ms within
  `MEMRA_ADMIT_DEFER_BUDGET_MS` (clamped to half the stall bound). Past the budget the answer is
  429 `rate_limit_exceeded` with `Retry-After: 5`; a session no card can ever hold is a 400
  `context_length_exceeded` naming the card, the bytes and the largest session that fits. Each
  decision prints one `[admit-mem] ... route=dsv4-thread verdict=...` line. Forward-time scratch
  is not charged, so a driver out-of-memory there still answers 503.
- The route refuses `response_format` by name and keeps no prefix cache.

## Dead-darklane background jobs (`MEMRA_BG_JOB`)

Idle-time background jobs: an operator command that runs only while the server has no work at all
and yields the moment work arrives. Which jobs run is outside this crate; the seam is
`MEMRA_BG_JOB` plus the checkpoint protocol below. Unset, the runner does not exist.

- **Valley.** The worker is idle when it has no active session, no queued admission and no
  pending HTTP handoff. Resume waits `MEMRA_VALLEY_S` (default 2) seconds of quiet; yield fires on
  the first sign of work, with no debounce. The runner starts only after the models load.
  `/metrics` publishes `serve_idle_seconds` in the operator view (0 whenever there is work).
- **Process.** `MEMRA_BG_JOB` runs under `sh -c` as a child in its own process group, with
  `PR_SET_PDEATHSIG=SIGKILL` so a killed server leaves no job behind. The runner polls every
  `MEMRA_BG_POLL_MS` (default 25 ms). On drain it sends SIGCONT and SIGTERM to the group, then
  SIGKILL to whatever is left after 2 s.
- **`MEMRA_BG_YIELD_MODE=stop`** (default): yield is SIGSTOP to the group and resume is SIGCONT.
  A stopped job keeps its memory, VRAM included, for its whole life. Measured on an RTX 5090
  (2026-08-07, N=5): 19.4 ms median and 23.3 ms max from request to stopped job. With an
  eight-spinner CPU job beside c=8x16 streaming bursts, burst latency p95 read 2.497 s against
  2.478 s without the job, TTFT p95 0.520 s against 0.514 s, and aggregate throughput 410.8
  against 413.1 tok/s ([receipt](../../research/darktrain-20260807/)).
- **`MEMRA_BG_YIELD_MODE=checkpoint`**: yield is SIGUSR1 to the group, meaning "checkpoint now and
  exit 75". The runner relaunches the same command at the next valley and the job resumes from
  its own file. Exit 0 is complete and never relaunched; any other exit is a loud failure, never
  relaunched. A job still running `MEMRA_BG_CKPT_GRACE_MS` (default 5000) after SIGUSR1 is
  killed, so the semantics are at least once: write checkpoints atomically. Start a one-command
  job with `exec` (`MEMRA_BG_JOB="exec python3 train.py"`) so the job, not the shell, receives
  SIGUSR1. `tools/bg-ckpt-counter.py` is a minimal job that follows the protocol.
- **VRAM.** `MEMRA_BG_VRAM_MB` (default 0, a CPU-only job) launches only while the minimum free
  memory across visible GPUs covers the budget plus `MEMRA_MOE_RESIDENT_HEADROOM_GB` (default
  2). An unreadable `nvidia-smi` refuses the launch; a refused job retries at the next valley.
  Staying inside the budget after launch is the job's contract.
- **Metrics.** With `MEMRA_BG_JOB` set, operator `/metrics` adds a `bg` block: `state`,
  `yield_mode`, `launches`, `yields`, `resumes`, `preempts`, `ckpt_kills`,
  `last_yield_signal_us`, `job_pid`, `vram_budget_mb`.
