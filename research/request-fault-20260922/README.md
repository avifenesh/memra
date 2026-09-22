# Request-fault boundary (memra#525)

Verdict: a Rust panic inside one request's step now fails that request with `code: worker_fault`
and leaves its peers untouched; only a driver-looking panic or a failed post-panic CUDA probe
still reaches the supervisor's respawn/exit ladder. Wire proof on the local RTX 5090, 9B NVFP4,
two runs: the salted stream ended typed, all three peers finished byte-identical to the control
round, `request_faults_total` read 1, `worker_respawns_total` 0, `/health` generation 0.

## The defect

The GPU worker runs every session on one thread inside one `catch_unwind` (the memra#50 ladder).
A panic in ONE request's step (a bad index, an unwrap on a request-shaped edge) unwound the whole
scheduler loop: every in-flight `Session` was dropped, every peer stream ended truncated after a
200, `/health` flipped dead and the worker respawned (or exited 70) for a fault that was never the
card's. The truncated-200 class is the worst shape a streaming client can receive: it reads as a
complete answer.

## The mechanism (worker.rs)

`request_fault_guard(context_healthy, request_id, route, site, f)` runs `f` under `catch_unwind`.
On a panic it classifies once, at the catch site:

- payload quotes `DriverError`, `CUDA_ERROR_*`, `CUBLAS_STATUS_*`, `cuBLAS` or an OOM string, OR
  the probe fails (`stream().synchronize()`, `zeros(16)`, `dtoh`): WORKER fault, `resume_unwind`
  into the existing ladder unchanged (a CUDA error is sticky per process; the respawn is right
  there and only there). One `[fault] ... WORKER FAULT` line names the request before re-raising.
- otherwise: REQUEST fault. `REQUEST_FAULTS_TOTAL += 1`, one
  `[fault] request=<id> route=lane<n>/<model> site=<site> panic=<msg>` line, and an `Err` carrying
  `REQUEST_FAULT_PREFIX` ("request fault:"). `EngineError::engine` classifies that text as the new
  `ErrClass::WorkerFault` (500, `server_error`, `code: worker_fault`, no Retry-After), the same
  text-classification seam `is_cuda_oom` already uses, so the 27 existing
  `EngineError::engine(format!(..{err}))` arms needed no edit and retire exactly their session.

Guarded sites: the async-chain + `step_session` decode step, the spec-stepper chain
(`step_dspark_spec` / `step_glm5_spec` / `step_gemma_spec` / `step_session`), the ready-loop
`step_session`, both `prefill_tick` calls (interactive and dark-lane chunk), constraint-mask
staging (`stage_grammar_mask`), the prefix-fanout leader prime (`prime_cache` in
`dedup_interactive_prefixes`), both batched prime calls (`prime_cache_batch`, interactive and
dark; a request fault retires the wave like a tainted one, never handed to the single-prime or
chunk path) and the batched decode call (retires the wave, ids joined in the route). The counter
counts guarded calls that panicked: a wave counts once while every request in it fails typed.
Review round 1 (revuto) named the three sites that were still bare and the test-global races;
round 2 named the park: a single-session request fault used to retire through `finished` alone,
leaving `aborted` false, so `retire_may_park` would have parked the half-updated KV into the
shared reuse pool for a later prefix match to resume from. `quarantine_request_fault` now marks
every such session aborted at all seven single-session arms (the wave arms already did). All
fixed in the same PR. A
dedicated `fault-inject` step at the top of the batched tick runs only when
`MEMRA_FAULT_INJECT_CACHE_SALT` is set. The supervisor bumps `WORKER_RESPAWNS_TOTAL` at
`attempt += 1`; both counters are on the operator `/metrics` body.

Not promised: a panic while a `std::Mutex` is held poisons it, and the next `lock()` panics
outside the guard and takes the ladder; a panic that leaves a shared structure half-updated is
caught, but its residue is whatever the unwound frames left. `MEMRA_PANIC_AFTER` still panics at
retire time, outside every guard, so it still exercises the worker ladder.

## Gates

- `request_fault_guard_tests` (worker.rs, CPU-only, 5 tests): Ok/Err pass through untouched and
  count nothing; a plain panic with a healthy probe is a request fault (prefix, site, panic text,
  counter, `WorkerFault` class, client sentence); a plain panic with a dead probe re-raises and
  counts nothing; a driver-looking panic re-raises even with a healthy probe; plain and OOM
  messages keep their `Engine` / `Overloaded` classes.
- `tools/request-fault-gate.py` (wired in `tools/local-ci.sh`, `MEMRA_CI_FAULTGATE=0` skips):
  one boot with the door, control round of 3 greedy streams, fault round of the same 3 plus a
  salted stream. V1 typed error, V2 peers byte-identical, V3 counters and generation, V4 exactly
  one `[fault]` line naming the site and no `[worker] PANIC`.

## Receipts (local RTX 5090, Qwen3.5-9B NVFP4 MTP GGUF, plain path, greedy, 48 tokens)

| run | typed | identical | faults | respawns | generation | log |
|---|---|---|---|---|---|---|
| `raw/gate-5090-run1` | yes | 3/3 | 1 | 0 | 0 | one `[fault]` line, site=fault-inject |
| `raw/gate-5090-run2` | yes | 3/3 | 1 | 0 | 0 | one `[fault]` line, site=fault-inject |

Peer text hashes were the same across both runs and both rounds (`888d259f…`, `8c59c164…`,
`b5ed9ff8…`). The salted stream's error object:
`{"type":"server_error","code":"worker_fault","message":"this request hit an internal fault and was retired; other requests were not affected. Report the request id"}`.

Two gate-side misfires before the first PASS, kept here because they are the kind a later reader
re-hits: (1) the first cut injected the fault inside the per-session decode/prefill closures, but
a fresh 4-stream burst on the 9B takes the batched prime and batched decode paths, so the salted
request never met an injection point (that is why injection is its own guarded step now, and why
the batched prime call is guarded); (2) the gate collected only `delta.content`, and the 9B
streams thinking first, so every text hashed empty (it now disables thinking and collects both).

`raw/ladder-control-panic-after/` (if present): the same server with `MEMRA_PANIC_AFTER=1`, the
worker-fault control: `[worker] PANIC`, generation 0 -> 1, `worker_respawns_total` 1,
`request_faults_total` 0.
