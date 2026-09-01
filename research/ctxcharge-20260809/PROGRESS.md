# request-context charge and allocation-OOM reclaim lane — progress

## 2026-08-09 intake

- Branch/worktree: `lane/cx-ctxcharge` at base `b8d6250f8110e70cba1c868cda6c23de7ba49eb6`
  (`v0.73.1`).
- Steering check: `~/.lanectl/inbox/cx-ctxcharge.md` is absent at intake.
- Validation contract read first:
  `../wt-cx-val256/research/val256-20260809/RESULTS.md`, with raw receipts under that lane's
  `research/val256-20260809/raw/`.
- Scope is limited to two admission defects: derive charge from each request's effective context
  cap, and on cache-allocation OOM reclaim the oldest parked continuation then retry exactly once.
- Preserve the existing request-shaped estimator, oldest-across-pools LRU hook, spec shrink
  reserve, admission polling behavior, generation behavior, and all unrelated dirty work.
- No origin push, merge, tag, `rustup`, `nsys`, perf-board edit, or runtime-default change.

## Planned proof

1. Trace request-cap derivation, the cap256k 5090 gate's false-negative shape, cache allocation,
   and the existing parked-session reclaim seam.
2. Add focused server unit tests that fail for the 262k-server/request-cap mismatch and pin a
   bounded reclaim-and-retry-on-allocation-OOM contract.
3. Implement the smallest fixes and run `cargo test -p memra-server` (154-test baseline plus new
   tests).
4. Extend and run `research/cap256k-20260809/run-5090-mixed-ctx.sh` under
   `flock /tmp/memra-gpu.lock`; retain raw logs proving distinct 8k, 128k, and 262k charges.
5. Under the same lock discipline, run the admit-OOM c=64 local-CI cell and q9 `run-gen` argmax
   gate. Commit raw receipts, write `RESULTS.md`, and stop without pushing.

## Status

Complete: both runtime fixes and every requested CPU/GPU gate are green. The audited final verdict
and evidence ledger are in `RESULTS.md`.

## Static diagnosis and implementation

- Defect 1 was in `request_ctx_cap`, before the estimator: explicit `max_ctx` and finite
  `prompt + max_tokens + 8` were both raised with `.max(MEMRA_CTX)`. The estimator correctly
  charged `shape.ctx_cap`; the shape itself had already been inflated to the 262,144-token server
  default. The old cap256k 5090 gate used `MEMRA_CTX=8192`, so its explicit 8k/128k/256k values
  were all at or above that floor and could not expose the inversion.
- Commit `d479ffe0` makes explicit `max_ctx` authoritative and finite requests allocate from
  `prompt + max_tokens + 8`; only an omitted output bound uses the server context default.
  Commit `6b3eb79c` adds the separate 8k-prompt-on-262k-server regression test.
- Defect 2 was the PP-aware plain-cache allocation branch in `admit`: after a failed
  `pp::new_cache`, it yielded only prefix-cache entries. The global continuation-pool LRU hook
  existed only in the pre-admission defer path, so a gate-passed allocation OOM never touched the
  three parked plain sessions from the Box1 receipt.
- Commit `bf924f46` reuses that same global LRU hook only for quoted CUDA allocation OOM, evicts
  exactly one oldest parked plain/spec session, and retries the allocation exactly once. The
  prefix-cache yield remains; spec's existing eviction/right-size ladder is unchanged.

## CPU-side gate

- `cargo test -p memra-server`: **156 passed**, 0 failed (154 baseline + 2 new regression tests).
- Raw output: `raw/cargo-test-memra-server.log`.

## Extended mixed-context 5090 gate

- Gate scripts: `research/cap256k-20260809/run-5090-mixed-ctx.sh` and `run_mixed_ctx.py`,
  extended in `ceb10930`. The server now runs at `MEMRA_CTX=262144`; the client sends an
  8,120-token raw prompt with no `max_ctx` and 64 output tokens (effective 8,192), an explicit
  131,072-token request, two 262,144-token parks, then the existing c=4 131,072-token burst.
  The client fails if any charge class is absent from the server log.
- Raw block: `raw/20260809T161158Z-after-mixed-ctx/`, N=1, one exclusive
  `/tmp/memra-gpu.lock` hold. The server log recorded 8,192 / 131,072 / 262,144 context charges:
  152 MB / 2,536 MB / 4,968 MB on the spec-shaped path (same 18,560 B/token coefficient and
  103 MB learned residual after the first request).
- All 8 requests completed. The c=4 burst completed 4/4 with TTFB service order
  0.043/0.088/0.280/0.299 seconds (0.256-second span), zero admission VRAM defers, zero step-OOM
  parks, and an empty captured failure scan. One oldest parked spec session was reclaimed on
  defer before the burst (effective free 2,251 -> 7,341 MB).
- Thermal regime: seven one-second samples, 51--66 C. The pre-existing Hermes embedding context
  held 394 MiB before and after the block; no other compute process was present at entry.

## Admission pressure and generation gates

- Raw block: `raw/20260809T161809Z-gates/`, one exclusive `/tmp/memra-gpu.lock` hold for all
  three live checks. All commands exited 0; model and draft SHA-256 values are pinned in
  `metadata.txt`.
- Normal admit-OOM c=64: **64/64** streams completed well formed, the worker remained healthy,
  and the captured server failure scan was empty. Informational completion wall p50/p95 was
  24.9/28.0 seconds and TTFB p50/p95 was 3.87/5.00 seconds. The explicit `max_ctx=8192` in this
  gate preserves its original per-session pressure under request-shaped sizing.
- Teeth arm: **TEETH OK**. With `MEMRA_ADMIT_RESERVE_MB=16`, 46/64 completed and 18 client
  receipts quoted `DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")` (16 batch-step and
  2 step errors), proving the gate still detects a broken reserve.
- q9 `run-gen`, prompt token 55, `MEMRA_NGEN=8`: prefill argmax 268 equals decode argmax 268,
  **MATCH**; all eight tokens generated and the process exited 0.
- Thermal regime: 59 one-second samples, 49--78 C. The pre-existing Hermes embedding context
  held 394 MiB before and after the block; no other compute process was present at entry or left
  behind at exit.
