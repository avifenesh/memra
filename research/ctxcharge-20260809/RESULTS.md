# Request-owned context charge and cache-allocation OOM reclaim

Lane `lane/cx-ctxcharge`, based on `b8d6250f` (`v0.73.1`), repaired the two admission defects
reported by the Box1 256k validation. Runtime and GPU validation was performed 2026-08-09 on the
local RTX 5090 under `/tmp/memra-gpu.lock`.

## Verdict

**PASS for the requested repair lane.** Request charge now follows the request-owned context
shape, and a plain-cache allocation OOM can reclaim the global-oldest parked continuation and
retry exactly once. The server unit suite is 156/156, the extended mixed-context gate observes
distinct 8k/128k/256k charges, the normal c=64 admission cell is 64/64 and clean, its forced-bad
reserve still trips the gate, and q9 `run-gen` reports argmax `MATCH`.

| contract | result | primary receipt |
|---|---|---|
| Explicit 128k request on a 256k server charges 128k | **PASS** | `ctx=131072`, 2,536 MB spec charge |
| Finite 8k prompt on a 256k server charges 8k | **PASS** | `ctx=8192`, 152 MB spec charge |
| Full-cap request remains 256k | **PASS** | `ctx=262144`, 4,968 MB spec charge |
| Allocation OOM reclaims one parked LRU and retries once | **PASS (unit)** | two attempts, one reclaim; no-reclaim error gets one attempt |
| admit-OOM c=64 local-CI cell | **PASS** | 64/64 well-formed, worker alive, failure scan empty |
| admission-gate teeth | **PASS** | forced 16 MB reserve loses 18/64 with quoted CUDA OOMs; `TEETH OK` |
| q9 generation argmax | **PASS** | prefill 268 = decode 268, `MATCH`; 8 tokens generated |

## Defect 1 — the server default was acting as a floor

The estimator was not at fault: it charged the `ctx_cap` carried by the prepared request. The
bad value originated in `request_ctx_cap`, where both an explicit `max_ctx` and the finite
`prompt_tokens + max_tokens + 8` shape were raised with `.max(MEMRA_CTX)`. With
`MEMRA_CTX=262144`, every smaller request therefore arrived at admission as a 262k request.

The earlier 5090 cap256k gate missed this because it launched the server with `MEMRA_CTX=8192`
and supplied explicit 8k, 128k, and 256k request caps. Every requested cap was at or above the
incorrect floor, so the faulty `.max()` happened to produce the expected value.

Commit `d479ffe0` gives the three request shapes distinct contracts:

- explicit `max_ctx`: authoritative request cap;
- finite `max_tokens` without `max_ctx`: `prompt_tokens + max_tokens + 8`;
- neither bound supplied: the `MEMRA_CTX` server fallback, retaining the existing oversized-prompt
  growth and model-context clamp.

The unit tests pin both an explicit 131,072 cap and an 8,120-token prompt plus 64 output tokens on
a 262,144-default server. The public flag and serving docs now describe this behavior rather than
calling `MEMRA_CTX` a universal floor.

### Live 5090 charge receipt

The extended gate starts a 262,144-context server and parses its admission log, failing if any of
the three charge classes is absent:

| request shape | logged context | path | coefficient | fixed | charged cost |
|---|---:|---|---:|---:|---:|
| 8,120 raw prompt tokens, no `max_ctx`, `max_tokens=64` | 8,192 | spec | 18,560 B/token | 0 MB | 152 MB |
| explicit `max_ctx=131072` | 131,072 | spec | 18,560 B/token | 103 MB | 2,536 MB |
| explicit `max_ctx=262144` | 262,144 | spec | 18,560 B/token | 103 MB | 4,968 MB |

All 8 requests completed. The c=4 128k burst completed 4/4 with TTFB service order
0.043/0.088/0.280/0.299 seconds, a 0.256-second span, zero final admission VRAM defers, zero
step-OOM parks, and an empty failure scan. Before the burst, the existing defer hook reclaimed
one oldest parked spec session and re-read effective free memory from 2,251 to 7,341 MB.

Raw receipt:
[`raw/20260809T161158Z-after-mixed-ctx/`](raw/20260809T161158Z-after-mixed-ctx/).
This is N=1 under one exclusive lock; seven one-second thermal samples ranged 51--66 C.

## Defect 2 — allocation failure bypassed the defer reclaim hook

In the Box1 OFF arm, admission passed and the PP-aware plain path then failed inside
`pp::new_cache`. That failure path could yield prefix-cache entries, but the global continuation
LRU hook existed only before admission deferral. With no defer, three parked full-cap entries were
never considered.

Commit `bf924f46` wraps that allocation with a bounded helper. After the first failure it preserves
the existing prefix-cache yield; when the captured error is a CUDA OOM, it also calls the existing
global plain/spec LRU eviction hook. If either source released memory, allocation is attempted one
more time. A second failure is returned to the request. There is no loop, and a non-OOM with no
reclaimed state is not retried.

The regression test forces repeated `CUDA_ERROR_OUT_OF_MEMORY`: allocation is called exactly
twice and reclaim exactly once. Its control error reclaims nothing and is attempted exactly once.
The older global-LRU selector tests continue to pin oldest-across-plain-and-spec ordering.

The exact two-GPU Box1 PP-2 failure was not rerun in this local lane. Its reclaim behavior is
unit-pinned here; a fresh Box1 validation is still required before treating the original remote
capacity/affinity block as repaired evidence.

## Gate battery

| gate | N / thermal regime | result |
|---|---|---|
| `cargo test -p memra-server` | 156 tests, CPU | **156 passed**, 0 failed |
| extended `run-5090-mixed-ctx.sh after` | N=1; 7 one-second samples, 51--66 C | **PASS**, 8/8 complete, all three charge contexts present |
| `tools/serve-stress-gate.sh ... 64` | N=1 burst; shared 59-sample block, 49--78 C | **ALL GREEN**, 64/64; wall p50/p95 24.9/28.0 s; TTFB p50/p95 3.87/5.00 s |
| same c=64 gate with `--teeth` | N=1 burst in the same lock | **TEETH OK**, 46/64 complete; 16 batch-step and 2 step OOM receipts |
| `MEMRA_NGEN=8 run-gen <q9> 55` | N=1 in the same lock | **MATCH**, prefill/decode argmax 268; process exit 0 |

The c=64 client now sends explicit `max_ctx=8192`. That preserves the original gate's 8k/session
admission pressure; otherwise short finite requests would correctly right-size to only a few
hundred tokens and make the test lose its intended VRAM pressure.

Raw receipts:

- [`raw/cargo-test-memra-server.log`](raw/cargo-test-memra-server.log)
- [`raw/20260809T161158Z-after-mixed-ctx/`](raw/20260809T161158Z-after-mixed-ctx/)
- [`raw/20260809T161809Z-gates/`](raw/20260809T161809Z-gates/)

The live blocks used q9 model SHA-256
`52c9cceb190055e0591a9a30c21f7200572eaf3ff1c59f6e9a1eda838a8f39de` and draft SHA-256
`b2cec2af426aa3d5e8bdd2bc9cee2b26278358e4efc59598d7f5ab3ec36bc045`. The pre-existing Hermes
embedding context held 394 MiB before and after both blocks; no other compute process was present
at entry or left behind at exit.

## Scope and discipline

- Small implementation, regression-test, gate, raw-data, and documentation commits were kept
  separate. Generated perf boards and raw tuning data were untouched.
- `~/.lanectl/inbox/cx-ctxcharge.md` was absent at intake and every pre-block check.
- No origin push, merge, tag, `rustup`, `nsys`, runtime-default flip, or perf-board edit was made.
