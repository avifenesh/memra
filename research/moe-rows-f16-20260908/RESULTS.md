Concluded on 2026-09-09: NEGATIVE numeric no-go; F16 door removed.
See [final results](../moe-rows-f16-20260909/RESULTS.md). This document retains the prior interrupted attempt.

# F16 rows component cell, 2026-09-08

Status: BLOCKED before the real-input oracle. No KEEP or NEGATIVE verdict.
PR #294 remains draft; no serving dispatch or default was added.

Source 241c3d6f4, rebased without conflicts onto origin/main 72aa777c3.
The parked extension f1ecd9113 was synthetic. The new bench-only harness captures
native mint layer inputs and exact binary routes, then replays both complete
conversion + routed FFN chains against the f64 numeric-class oracle.
See PLAN.md for the exact capture scope, gate bands and weighting policy.

## Completed validation

| Check | Result |
|---|---|
| Release build, B200 CUDA 13.1.115, sm_100a | PASS |
| Formatting | PASS |
| Clippy, engine lib + component bench + F16 test, -D warnings | PASS |
| Engine unit suite, GPU 0, shared lock | 449 passed, 0 failed, 19 ignored |
| F16 synthetic GPU test | Compiled; execution pending |
| Real-input oracle | Not started at last live readback |

The inherited extension's explicit closure drop failed Clippy's drop_non_drop
lint; removing the unnecessary drop fixed it. An initial build command used a
hyphenated name for an underscore-named binary and failed before compilation.
Both original failures and successful corrected checks remain in the raw archive.

## Requested saving table

| Tokens | Routed layers | Weighted saving, ms/round | Oracle |
|---|---:|---|---|
| 2 | 42 planned | Not measured | Queued at last readback |
| 4 | 42 planned | Not measured | Not run |
| 7 | 42 planned | Not measured | Not run |

The shared qualification host stopped while the t2 gate was queued behind other
lanes. SSH then refused connections and the provider's intended state was stopped.
No failing layer/tensor exists and there is no raw timing archive. The >=0.5 ms
retention rule cannot be evaluated. The default-off door is unchanged with its
2026-09-20 decide-by date. A hardware interruption is not a negative numeric or
performance result, so the door-removal rule has not been triggered.

## Evidence and continuation

Partial validation archive: partial-receipts.tar.gz, SHA256
`e3d70b86607ffad6f3fa459ddc8ac86cf5c43d91f28e5708fe535ceb368dbacc`. This is a build/unit/queue archive, not timing evidence.
The component binary SHA256 was
`1e42409939314f0067a580aed7e54e4b1342a241838da4ce83af2f74a4d2fc42`.
Raw unit output and all completed checks were copied off the host before it stopped.
The source and corrected per-shape driver are retained. Resume the real t2 gate
on an available non-production B200, copy each completed shape's receipts off-box,
then gate t4/t7 before timing. Do not reuse queued/empty logs as a pass.

No rig cargo, GPU gate, bench, smoke server or CI ran. The commit formatting hook
was executed on the B200 host. Push uses MEMRA_SKIP_PERF_CI=1 under the owner rule.
No other lane's process was killed. This lane cancelled only its own queued command
to correct the prompt file path. No serving-pair action was taken.
