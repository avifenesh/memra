# Cache/spec serving hardening — 2026-08-09

## Owner report

> the response get substantially slower as more request are coming very fast, meaning the cache is not working well or not aligned with the spec. we have a lot of hardening to do.

## Scope

- Lane: `lane/cx-cachespec`
- Remote rig: box1 Sbox, Step-3.7-Flash PP-2
- Serving shape: `MEMRA_CTX=262144`, `MEMRA_PREFIX_CACHE_MB=2048`, default PP2 cross-device speculative placement
- Client shape: growing same-session conversation with rewritten history, followed by a concurrency-4 burst
- Comparison: default speculative policy versus `MEMRA_SERVE_SPEC=0`

## Required evidence

For every request, retain raw logs and report:

- request index and concurrency phase
- input/output token counts
- time to first token and end-to-end latency
- decode throughput
- deltas for cached input tokens, prefix-cache hits/misses/evictions, admission defer/park counters, and pool state

The receipt must locate latency growth in TTFT and/or decode and correlate it with the counter that moves. Failures will be quoted from captured stderr; unknown failures will remain labeled unknown. Every reported median will include N and thermal regime.

## Hypotheses under test

1. The speculative path fails to credit prefix-cache reuse.
2. Session or KV state is not retired between requests.
3. Admission deferrals grow as the pinned pool fills, possibly through a Step-specific gate that differs from the fixed Gemma path.
4. The 2 GiB prefix-cache budget thrashes on large entries.

No hypothesis is considered confirmed until the replay and counters distinguish it. Spill/cache performance is kept separate from model-quality claims.

## Execution log

- 2026-08-09: Initialized write-first evidence contract before source inspection, remote build, or GPU work.
- 2026-08-09: Added per-request replay capture and `/metrics` visibility for admission,
  continuation/spec pools, and CUDA pool state. Local harness tests passed 4/4.
- 2026-08-09: Box1 release build and `cargo test --release -p memra-server` passed using
  the machine's existing cargo. The first preflight used a non-login PATH and captured
  `bash: line 1: cargo: command not found`; no Rust toolchain was installed. The first test
  spelling captured `error: no library targets found in package memra-server`, then the
  package-level command passed.
- 2026-08-09: The first RunPod attempt rejected a box1-built binary with
  ``version `GLIBC_2.39' not found``. Built natively with the pod's existing cargo, added
  owner-service restoration and CUDA-quiescence guards, and completed the live-rig receipt.
  The owner binary was restored and `/readyz` verified afterward.
- 2026-08-09: Completed the controlled box1 default-policy versus
  `MEMRA_SERVE_SPEC=0` replay from one frozen workload. Both arms selected K=0 and produced
  identical cache/admission counters; the sequential slowdown remained.
- 2026-08-09: Root cause localized to the plain path: the reusable prefix frontier freezes
  at 6,148 tokens, while two full-context continuation entries miss every rewrite and amplify
  c=4 admission serialization. Wrote the receipt and the plain-affinity/reclaim design in
  `RESULTS.md`.
- 2026-08-09: Ran an additional forced-spec K=3 control after box1 became free. It did not
  flatten the replay: all 16 later affinity probes declined two tokens before the saved
  prompt-end checkpoint (burst siblings diverged by 21), cached tokens stayed zero, and c4
  serialized to 117.531 s fourth-request TTFT. It also captured five max-token overshoots
  (769/770 returned for a 768 cap) and a spec-only global `tokens_out=0` telemetry omission.
  These are documented separately from the deployed K=0 root cause.
- 2026-08-09: Released the replay lock, then acquired a separate block for a pinned-host
  transfer screen. CUDA bandwidthTest reported 512 MiB H2D in 9.44 ms and 1 GiB in
  18.85--18.87 ms on both box1 GPUs. This is recorded only as a host-tier prototype signal,
  not end-to-end checkpoint performance; the tool's own warning is retained in the raw log.

## Exit criteria

- Reproduction harness and raw default/spec-off logs committed.
- Request-index receipt identifies the first dominant latency term and moving counter.
- If the cause is small, a surgical fix plus regression gate is committed and remeasured.
- If the cause is large, the code-path anatomy and an implementable design are committed.
- GPU lock is released between measurement blocks; nothing is pushed or tagged.
