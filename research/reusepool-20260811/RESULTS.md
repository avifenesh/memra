# cx-reusepool results

Status: complete

## Finding

`MEMRA_REUSE_POOL` was enforced on each `(model, cache namespace)` vector independently. A client
that supplied a fresh `cache_salt` on every turn could therefore retain approximately
`MEMRA_REUSE_POOL × namespaces × parked-entry bytes` in each populated whole-session tier. The
27bab control demonstrated the consequence: 76 speculative entries followed by a captured
`CUDA_ERROR_OUT_OF_MEMORY`.

## Fix

- `MEMRA_REUSE_POOL` remains the per-namespace LRU cap, preserving reuse locality and its existing
  `0`-disables-pooling behavior.
- `MEMRA_REUSE_POOL_GLOBAL_CAP` adds a process-wide entry ceiling across all models, namespaces,
  continuation pools, and speculative pools. The default is 16; `0` disables parking process-wide.
- Every eligible retire makes room before publishing the new entry. If the global ceiling is
  full, the existing age metadata evicts the globally oldest entry across both pool types and
  charges the existing pool-specific eviction counter.
- The default is anchored to `research/27bab-20260810/RESULTS.md`: its bounded two-namespace Q27
  run retained 16 speculative entries with zero OOM parks and 27.34 GB driver-free.

## Verification

- Focused release regression, CPU-capped: 1 passed, 0 failed.
  `nice -n 15 taskset -c 0-7 cargo test -p memra-server --release worker::tests::parked_entry_ceiling_bounds_salt_fanout_and_evicts_global_oldest -- --exact`
- Required package gate, CPU-capped: 183 passed, 0 failed.
  `nice -n 15 taskset -c 0-7 cargo test -p memra-server --release`
- Flag catalog gate: `tools/check-flags.sh` reported 456 runtime literal reads and no new drift
  beyond `research/docsync3-20260811/flags-drift.txt`.
- No GPU runtime, merge, tag, push, formatting, or perf-board operation was performed.
