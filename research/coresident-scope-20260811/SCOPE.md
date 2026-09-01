# Co-resident per-model worker refactor scope

## Scope and evidence freeze

This is a read-only code-cost map for **N startup-resident models, each owned by its own
CUDA-owner worker, inside one `memra-server` process**. It does not select a config syntax,
failure policy, scheduling policy, deployment topology, or implementation.

- Source inspected: `/home/avifenesh/projects/bw24`, branch `main`.
- Source commit: `96afb32e197e973b256ba61a733bb185cf767302`.
- The current source checkout and remote `main` matched that commit at final validation. The lane
  remains rooted at `4e7a4a3343b8d3dffaa2170ee9eea5fca6a4d910`; all cited source paths are
  unchanged between the lane base and the evidence commit.
- Source and config files were not edited; no build, formatter, benchmark, or GPU gate was run.

Size meanings are relative code-surface sizes, not calendar estimates:

- **S** — localized type/parser or glue change with bounded call-site updates.
- **M** — a cross-module contract plus focused unit/integration coverage.
- **L** — an ownership, concurrency, or failure-domain change spanning runtime and gates.

The current topology is:

```text
comma-separated MEMRA_MODELS
  -> Vec<(alias, path, optional draft)>
  -> one worker::spawn call
  -> one command channel + one supervisor/OS thread
  -> one Engine / primary CUDA context
  -> HashMap<alias, LoadedModel>
  -> one active-session queue and one scheduler tick loop
  -> same-model decode chunks within that loop
```

That topology is stated directly in the server module contract
(`crates/memra-server/src/main.rs:1-6`) and worker contract
(`crates/memra-server/src/worker.rs:1-16`).

## 1. Config parser — **S**

### Current state

`MEMRA_MODELS` is already a multi-model spec. The parser returns a tuple vector, not a named
`ModelSpec` struct: `Vec<(String, String, Option<String>)>`
(`crates/memra-server/src/main.rs:1785-1793`). It splits on commas, resolves and validates each
model and optional draft, and attaches the next model at the `out.push(...)` call
(`crates/memra-server/src/main.rs:1794-1843`). If the variable is absent or has no valid entries,
the fallback itself contains `main` and `judge` entries
(`crates/memra-server/src/main.rs:1845-1849`).

The tuple shape currently propagates to:

- OpenRouter metadata alias validation (`crates/memra-server/src/main.rs:362-380`).
- `worker::run` (`crates/memra-server/src/worker.rs:2723-2728`).
- `worker::spawn` (`crates/memra-server/src/worker.rs:7316-7323`).

### Exact touch-list

- `crates/memra-server/src/main.rs`: `parse_models_config`, the model-plan type consumed by
  `load_openrouter_metadata`, and the startup handoff in `main`.
- `crates/memra-server/src/worker.rs`: the matching `run` and `spawn` model-plan parameters.

The parser already supplies a second model; no registry-list feature is missing. The scoped parser
work is the explicit per-worker information that the current three-field tuple cannot carry.
Which fields or spelling represent that information is intentionally unresolved here.

### Naive-extension breakage

Looping over the existing tuples and spawning one worker per tuple would give every worker the
same process-wide placement input: `run` reads `MEMRA_PP_DEVICES` itself and derives one device
before constructing the engine (`crates/memra-server/src/worker.rs:2730-2738`). The current tuple
cannot express different worker placement or PP shape. The parser also has no topology-level
validation point for such data; merely retaining the tuple preserves that ambiguity.

## 2. `worker::spawn` / `worker::run` — **L**

### Current state and CUDA ownership

`main` makes exactly one `worker::spawn(models, health)` call and receives one sender, one model
list/capability map, one metrics handle, and one join handle
(`crates/memra-server/src/main.rs:1601-1633`). `AppState` stores exactly one command sender, one
metrics handle, and one health handle (`crates/memra-server/src/main.rs:393-418`). Both completion
handlers send every model request to that sender
(`crates/memra-server/src/main.rs:2863-2926`,
`crates/memra-server/src/main.rs:2945-3026`).

`spawn` creates one command channel and one named `memra-gpu-worker` supervisor thread
(`crates/memra-server/src/worker.rs:7327-7335`). On each attempt the supervisor passes the entire
model vector into `run` (`crates/memra-server/src/worker.rs:7342-7375`). `run` binds ownership by
deriving the primary ordinal and constructing one local `Engine`
(`crates/memra-server/src/worker.rs:2716-2738`); `Engine::new` creates the `Gpu`, while `Gpu::new`
creates the CUDA context and compute stream (`crates/memra-engine/src/lib.rs:935-982`,
`crates/memra-runtime/src/lib.rs:102-110`). The local `engine` value never leaves `run`.

The thread-affinity contract is explicit: build the Engine and every model on one OS thread and do
not move Engine/Cache/CudaSlice values across thread boundaries
(`crates/memra-server/src/worker.rs:1-8`).

### Exact touch-list

- `crates/memra-server/src/worker.rs`: `spawn`, `run`, their ready/supervision channels, return
  type, respawn loop, and shutdown handle ownership.
- `crates/memra-server/src/main.rs`: the single spawn/join lifecycle, `AppState.cmd_tx`, and model
  dispatch in both completion handlers.
- `crates/memra-server/src/worker.rs`: `PENDING_ADMITS`, because it is one process-global atomic
  around the current single command channel (`crates/memra-server/src/worker.rs:330-339`).

### Naive-extension breakage

- Sharing the current `Engine` between new worker threads violates the stated thread-affinity
  contract and the Engine's single-worker scratch assumption: shared CUTLASS scratch is safe today
  because all compute serializes on one worker/stream
  (`crates/memra-engine/src/lib.rs:591-633`).
- Keeping all models on the current thread and only adding another `Engine` leaves one command
  queue and one scheduler loop, so it does not create independent CUDA-owner workers.
- Calling current `spawn` once per model leaves `AppState` with nowhere to store or select among N
  senders and join handles. Reusing the current supervisor unchanged also gives each supervisor
  its own whole-process exit path after an unrecoverable worker failure
  (`crates/memra-server/src/worker.rs:7399-7416`). Whether one failed model ends the process or only
  removes that model is an unresolved contract, not decided here.
- Startup is currently all-or-nothing: the one spawn returns only after every model in its load
  loop has produced one ready result (`crates/memra-server/src/worker.rs:2760-2763`,
  `crates/memra-server/src/worker.rs:2883-2937`). N ready channels need an aggregate startup
  contract before the listener lifecycle can retain equivalent meaning.

## 3. Engine ownership — **L**

### Current state: Engine is not 1:1 with a model

`LoadedModel` owns a `HybridModel`, tokenizer, EOS id, source-kind bit, and constraint compiler; it
does not own an `Engine` (`crates/memra-server/src/worker.rs:76-89`). `run` creates one Engine, then
loads every model through that same Engine and inserts each into `loaded`
(`crates/memra-server/src/worker.rs:2730-2763`,
`crates/memra-server/src/worker.rs:2766-2809`,
`crates/memra-server/src/worker.rs:2883-2887`).

The ownership split is therefore:

- **Per Engine:** CUDA context/stream/modules and lazy scratch/cache state
  (`crates/memra-engine/src/lib.rs:591-633`).
- **Per model:** weights/config and model-local prime slabs
  (`crates/memra-engine/src/hybrid.rs:1204-1228`).
- **Current relationship:** one Engine to N `LoadedModel` values on one worker.

### Proven process-global blocker

The PP runtime is not Engine-local. `PpNRt` lives in a process-wide `OnceLock`; the first caller
freezes one stage count and device map for the process
(`crates/memra-engine/src/pp.rs:588-615`). It reads the process environment while building that
runtime (`crates/memra-engine/src/pp.rs:617-655`). Its host-bounce storage additionally rejects a
second model width after the first width initializes it, explicitly stating that one PP runtime
supports one model geometry per process (`crates/memra-engine/src/pp.rs:910-947`).

### Exact touch-list

- `crates/memra-server/src/worker.rs`: move the current one-Engine/N-model ownership boundary to
  the per-model worker boundary while retaining CUDA-owned caches and publication on that thread.
- `crates/memra-engine/src/pp.rs`: the process-wide `RTN`, environment-derived placement, stage
  engines/streams, and model-geometry-bound host-bounce runtime.
- `crates/memra-engine/src/lib.rs`: the Engine ownership/scratch invariants touched by concurrent
  Engine instances. The storage is already per Engine; this is an audit and contract surface, not
  evidence that every Engine method needs modification.

`crates/memra-runtime/src/lib.rs:33-41` and `crates/memra-runtime/src/lib.rs:102-154` are supporting
evidence that a `Gpu` context and stream are instance-owned. No required runtime-file change is
proven there by this read-only pass.

### Naive-extension breakage

Two worker-local Engines can still collide through the first-built global `PpNRt`: a later model
would see the first model's devices/stage engines and, under host bounce, may fail on a different
`n_embd`. A PP model and a non-PP model also cannot obtain independent PP settings while PP
selection comes from one process environment. The global runtime's streams and stage Engines would
be shared across worker threads even though the Engine scratch contract assumes one serial owner.

This is the only proven process-global engine blocker included in the size. The engine contains
many process-wide tuning `OnceLock`s, but whether independent workers require per-model tuning
flags is not specified by this task; expanding those flags is therefore not counted or implied.

## 4. Health, `/v1/models`, and `/metrics` — **M**

### Current state

- **Health:** one `WorkerHealth` stores one heartbeat, phase, generation, worker-fault latch, and
  GPU-fault latch (`crates/memra-server/src/health.rs:103-131`). The single worker publishes all
  loaded-model capabilities before marking that health handle ready
  (`crates/memra-server/src/worker.rs:2883-2937`); the ready-state method is singular
  (`crates/memra-server/src/health.rs:194-220`). The HTTP payload lists all models beside one
  singular `worker` object, and both liveness/readiness query that one handle
  (`crates/memra-server/src/main.rs:1852-1927`).
- **`/v1/models`:** this handler is already multi-model. It iterates the cached model list and
  looks up capabilities per alias (`crates/memra-server/src/main.rs:2415-2421`). Its registry data
  arrives as the single worker's combined ready result
  (`crates/memra-server/src/main.rs:1622-1656`).
- **Metrics:** `AppState` holds one `SharedMetrics`; most fields are worker-wide scalar counters,
  percentiles, gauges, and one Engine's CUDA-pool values, while speculative telemetry and
  constraint fail-closed gauges are already keyed by model
  (`crates/memra-server/src/worker.rs:341-425`). The scheduler publishes the scalar and per-model
  fields from one loop and one Engine (`crates/memra-server/src/worker.rs:4637-4697`), and
  `/metrics` reads that one snapshot (`crates/memra-server/src/main.rs:1931-2063`).
- **Dark-lane idleness:** `ValleySignal` reads one health phase plus the one global
  `PENDING_ADMITS` counter (`crates/memra-server/src/darklane.rs:75-108`), and production creates
  one runner from one health handle (`crates/memra-server/src/darklane.rs:306-316`).

### Exact touch-list

- `crates/memra-server/src/main.rs`: `AppState`; startup aggregation; `health_payload`,
  `health_live`, `health_ready`, `get_metrics`; and the model/capability registry feeding
  `list_models_v1`.
- `crates/memra-server/src/health.rs`: singular `WorkerHealth` representation and its
  ready/live/snapshot semantics.
- `crates/memra-server/src/worker.rs`: `Metrics`, per-worker publication, and the values returned
  by `spawn`.
- `crates/memra-server/src/darklane.rs`: valley detection and production runner wiring.

The `/v1/models` iteration itself is not a single-model blocker. The touch is the lifecycle and
availability data behind the cached list, not a missing loop.

### Naive-extension breakage and explicit ambiguity

- Selecting one worker's health hides failures in the others; making any worker fault the one
  process-global fault changes a per-model failure into whole-process unavailability. The source
  has no contract for partial model availability, so that policy remains unresolved.
- Retaining the current cached model list after independent worker failure continues advertising
  an alias whose worker may be dead. Removing it dynamically changes the current startup-resident
  catalog contract. No choice is made here.
- Selecting one metrics handle drops N-1 workers. Blind summation is also wrong for
  `step_p50_ms`/`step_p99_ms`, `batch_size_last`, and same-device CUDA free/pool gauges; those are
  not additive. Counter and gauge aggregation semantics are part of the scoped work.
- Pointing `ValleySignal` at one worker can launch background work while another worker is busy.
  Leaving `PENDING_ADMITS` process-global has the opposite coupling: an admit for one model makes
  every worker appear non-idle and can terminate unrelated speculative bursts
  (`crates/memra-server/src/worker.rs:6866-6883`).

## 5. Scheduler — **L**

### What the tick loop actually iterates

The scheduler does **not** iterate a model set. It owns one global `active: Vec<Session>` and one
global FIFO request queue, plus shared pending-constraint, reuse, prefix-cache, admission, policy,
and metric state (`crates/memra-server/src/worker.rs:2963-3059`). Each outer iteration drains the
one command receiver and admits into that one session vector
(`crates/memra-server/src/worker.rs:3061-3126`). The tick then runs serial phases over sessions:
spec, prefill, and decode (`crates/memra-server/src/worker.rs:3442-3448`).

Model selection happens inside those session passes. Prefill resolves `loaded[&s.model]`
(`crates/memra-server/src/worker.rs:4184-4204`); decode builds a ready-session list, then
`group_chunks` splits it into same-model chunks before dispatch through that model
(`crates/memra-server/src/worker.rs:4219-4288`,
`crates/memra-server/src/worker.rs:6607-6621`). Thus different models cannot share one batched
kernel call, but they still serialize through the same Engine owner and outer tick.

The relevant ownership split today is:

| State | Current scope | Evidence |
|---|---|---|
| `LoadedModel`, caps, chunk cap, eager-only classification, admission cost | per-model values held in worker maps | `crates/memra-server/src/worker.rs:2760-2763`, `crates/memra-server/src/worker.rs:2939-2993` |
| continuation/spec reuse | worker-owned maps keyed by `(model, cache namespace)` | `crates/memra-server/src/worker.rs:864-872`, `crates/memra-server/src/worker.rs:2968-2974` |
| Engine, receiver, active sessions, FIFO queue, pending compiles, prefix cache, lane policy, latency window, aggregate counters | one worker/scheduler | `crates/memra-server/src/worker.rs:2723-2738`, `crates/memra-server/src/worker.rs:2963-3059` |
| HTTP in-flight/tenant limits and graceful drain | process-wide `AppState`/globals | `crates/memra-server/src/main.rs:408-418`, `crates/memra-server/src/main.rs:1686-1733` |

### Exact touch-list

- `crates/memra-server/src/main.rs`: alias-to-command-sender routing before request submission.
- `crates/memra-server/src/worker.rs`: `handle_cmd`; the model map/order assumptions; the worker
  loop's active/queue/cache/admission/metrics ownership; and process-global admission signalling.
- `crates/memra-server/src/darklane.rs`: process idleness composed from scheduler activity.
- `crates/memra-lanes/src/lib.rs` is conditional: it is touched only if current process-wide QoS
  limits must remain shared across per-model loops. The source does not establish that policy, so
  it is not counted as a required change.

### Naive-extension breakage

- Sending all requests through the current `AppState.cmd_tx` still feeds one scheduler; sending by
  alias requires a registry of worker handles before the current worker-side unknown-model check
  (`crates/memra-server/src/worker.rs:4727-4751`).
- Cloning the loop per model silently turns current worker-wide session caps, lane admission
  windows, latency percentiles, caches, and counters into per-model values. Keeping them shared
  introduces synchronization and cross-worker ownership not present today. This is a scoped
  semantic seam, not a choice between the two.
- The current same-model chunking prevents mixed weights in one kernel call, but it provides no
  cross-worker CUDA overlap or isolation proof. Treating it as the requested refactor would retain
  the single-owner serialization.

## 6. Tests and gates — **M**

### Current single-worker/single-model assumptions

| Test/gate | Current hard-coded shape | Missing refactor arm |
|---|---|---|
| `memra-server` handler tests | `fake_worker_state` creates one command channel, one fake worker, one health handle, and model `m` (`crates/memra-server/src/main.rs:4539-4606`) | alias-to-worker dispatch plus N-worker health/metrics behavior |
| health unit tests | each test constructs one `WorkerHealth` and asserts one worker's phase/fault transitions (`crates/memra-server/src/health.rs:667-740`) | aggregate/partial-failure cases once that contract exists |
| model catalog unit test | the OpenAI helper already covers two aliases (`crates/memra-server/src/main.rs:5422-5429`) | no catalog-loop arm is missing; worker availability behind those aliases is untested |
| `tools/serve-smoke.sh` | every normal boot passes one `smoke=...` entry and every request names `smoke` (`tools/serve-smoke.sh:31-46`, `tools/serve-smoke.sh:63-112`) | two resident aliases, requests to both, concurrent overlap, and routing/failure isolation |
| `tools/serve-stress-gate.sh` (`sstress`) | one `stress=...` entry; all clients send model `stress` (`tools/serve-stress-gate.sh:42-59`, `tools/serve-stress-gate.sh:88-103`) | mixed-model concurrent pressure and survival of both owners |
| `tools/accept-gate.sh` (`accept`) | explicitly one boot per model/draft/K group, always alias `m` (`tools/accept-gate.sh:142-166`) | simultaneous second-model arm; the existing per-model acceptance cells remain distinct |
| `tools/step35-b2-geometry-gate.sh` (`b2geo35`) | one `MEMRA_MODELS` spec with process-wide PP-2 and every request names `step35` (`tools/step35-b2-geometry-gate.sh:89-108`) | PP model plus a second independently placed model in the same process |

The gate registry maps any `crates/memra-server/` change to
`sstress,accept,tickinv35,tickinv35c,b2geo35,b2geo35c`, but contains no multi-worker/mixed-model
cell (`tools/fast-gate/map.tsv:121-131`). `local-ci` runs the one-model smoke and stress gates
(`tools/local-ci.sh:269-284`) and explicitly describes its default acceptance arm as one model/one
cell (`tools/local-ci.sh:301-309`).

### Exact touch-list

- Unit-test regions in `crates/memra-server/src/main.rs`,
  `crates/memra-server/src/worker.rs`, and `crates/memra-server/src/health.rs`.
- `tools/serve-smoke.sh`.
- `tools/serve-stress-gate.sh`.
- `tools/accept-gate.sh`.
- `tools/step35-b2-geometry-gate.sh`.
- `tools/fast-gate/map.tsv` and `tools/local-ci.sh` so the resulting multi-worker arm is inside
  the merge battery rather than an unregistered side gate.

The existing `kernel-check`, `run-gen` argmax, and `run-spec` K=1..8 gates remain per-model engine
correctness gates. They do not exercise server worker multiplicity, so they do not replace the
missing arm and are not themselves counted as refactor touch points.

## Consolidated size and risk map

| Surface | Size |
|---|---:|
| 1. Config parser/model-plan plumbing | S |
| 2. `worker::spawn` / `worker::run` lifecycle | L |
| 3. Engine and PP-runtime ownership | L |
| 4. Health, model catalog backing state, metrics, dark-lane idleness | M |
| 5. Scheduler and request dispatch ownership | L |
| 6. Unit/integration gate expansion | M |
| **Total (overlapping surfaces, not additive)** | **L, upper end** |

The total remains one large refactor rather than six independent changes because `main` startup,
worker supervision, Engine/PP ownership, scheduler state, and health/metrics all meet at the same
worker handle. The parser and catalog loops are already multi-model and are not the dominant cost.

### Ranked highest-risk seams

1. **CUDA/PP ownership and first-use global state.** `PpNRt` freezes one process-wide placement and
   model geometry while the requested shape introduces multiple CUDA-owner threads. A naive split
   can bind later work to the first model's stages/streams or reject the later model geometry.
2. **Worker failure domain, startup, and HTTP routing.** The current single supervisor owns one
   sender, one ready verdict, one health object, and a whole-process exit path. N workers require a
   precise aggregate contract; the current code supplies no partial-availability semantics.
3. **Cross-worker scheduler signals and observability.** `PENDING_ADMITS`, dark-lane idleness,
   lane capacity, latency percentiles, and CUDA gauges are singular today. Naive sharing couples
   independent models; naive duplication changes process-level limits and can misreport health or
   capacity. Existing gates do not exercise this topology.

## Deliberately unresolved

This scope does not decide whether to perform the refactor, what config syntax to add, how models
map to devices, whether one failed model fails the process, whether QoS is per model or process,
how non-additive metrics are exposed, or whether another deployment topology is preferable. Those
are owner decisions outside this read-only enumeration.
