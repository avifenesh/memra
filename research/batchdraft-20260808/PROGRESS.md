# Cross-request batched draft/verify — stage 1 progress

Date: 2026-08-08 UTC / 2026-08-09 local

Lane: `lane/cx-batchdraft`

Base: `cbe25b75e95f9aed8863771b625e12c35b016286`

## Scope

This lane stops at anatomy, measured c=4 decode-side headroom, and an implementation seam. It does
not implement cross-request speculative decoding. The concurrent-prefill saturation verdict is a
fixed premise, not a target to revisit: this work concerns decode-side batching of speculative
draft/verify rounds only.

## Plan

1. Trace the current per-session path from `worker.rs::step_session` into the speculative engine:
   identify draft generation, verify-forward construction, KV/cache mutation, acceptance and
   rollback, and the scheduler condition that excludes speculative sessions from shared decode
   batches.
2. Freeze the anatomy as a call/data-flow map with exact source locations and state ownership.
   State precisely why separately stepped sessions cannot share a verify forward today.
3. On box1, acquire `/tmp/memra-gpu.lock` for each measurement block and record entry/exit GPU
   state. Use the pinned Step-3.7-Flash artifacts under `~/step37/models/step-3.7-flash/` and the
   lane runtime with `MEMRA_TICK_TRACE=1`.
4. Run an interleaved c=4 spec-ON decode experiment. Preserve raw build, server, client, tick-trace,
   and GPU-utilization logs. State N and thermal regime; quote failures from captured stderr only.
5. Derive per-request draft and verify GPU/wall-time plus idle gaps from the trace. Estimate the
   counterfactual of fusing four requests' verify passes as one forward, separating the measured
   bound from assumptions about batched-kernel scaling. Do not claim draft-side gains this stage
   does not measure.
6. Write the design seam: scheduler grouping contract, ragged/padded verify representation, engine
   batch API and per-session result demultiplexing, KV/cache transaction boundaries, graph/bucket
   implications, deterministic sampling, and failure isolation.
7. Specify the gate plan: single-session oracle equivalence; c=4 same-length and ragged acceptance;
   staggered-depth/ladder-rung isolation; K=1..8 self-consistency; cancellation/EOS/rejection;
   cache/KV rollback; batched-call evidence; and throughput/idle-gap acceptance criteria.

## Evidence contract

- Raw logs live under `research/batchdraft-20260808/raw/box1/`; summaries never replace raw runs.
- Measurement arms are interleaved within one lock hold where practical, with the exact commit,
  command, environment, model/artifact identity, request shape, N, and thermal state recorded.
- The lock is released after every measurement block and both GPUs must be shown quiescent at exit.
- No origin push, no release/tag, no `rustup`, and no published perf-board movement in this lane.
- `~/.lanectl/inbox/cx-batchdraft.md` is checked at least hourly while work continues.

## Status

- [x] Required instructions and source research read.
- [x] Branch/base/worktree state verified.
- [x] Current spec anatomy mapped in `ANATOMY.md`.
- [x] Box1 c=4 trace and headroom estimate complete in `RESULTS.md`.
- [x] Design seam and gate plan complete in `DESIGN.md`.

## Receipts

- `6ed286e0` — write-first plan, committed before code anatomy.
- `3248e4f9` — diagnostic-only tick coverage plus PP-correct/interleaved verify-width probe; both
  affected crates passed `cargo check`.
- `6d9a7820`, `eca906de`, `7da700d7`, `b93651f7` — frozen c=4 and box1 harness, with native-response
  parsing and independently resumable lock blocks.
- `af7a30a2` — raw box1 serving and m-scale evidence plus the reproducible join/projection analyzer.

The scored serving block is `20260808T210100Z` (N=5 per arm); the m-scale block is
`20260808T210500Z` (N=25 per width). Both valid blocks have explicit flock-release and zero-memory
post-state receipts. The two earlier attempts are retained and marked unscored in `RESULTS.md`.

## Outcome

- Verify is 93.72% of the steady speculative round phase at c=4.
- Serial session handoff is only 0.011 ms median; this is an underfilled weight-stream problem, not
  a scheduler-sleep problem.
- An ideal one-verify-cost live wave projects to 1.84x client throughput for synchronized requests
  and 1.73x for divergent requests, with every non-verify cost held measured and constant.
- Today's contiguous m=16 verify costs 1.038x four serial m=4 calls; naively flattening four rows
  projects a 1.9-2.2% serving regression. Stage 2 therefore needs a true per-cache B x T core and a
  promoted exact M=16 tier, not reuse of the scalar T=16 call.
- `~/.lanectl/inbox/cx-batchdraft.md` was absent at the initial and 2026-08-09T00:12+03:00 checks.

## Final validation

- `cargo check -p memra-server --bin memra-server` — pass.
- `cargo check -p memra-engine --bin verify-mscale` — pass.
- Python byte-compile, shell syntax check and analyzer cardinality/value assertions — pass.
- `python3 tools/update-perf-board.py --check` — `perf board is up to date`.
- Final box1 audit: `/tmp/memra-gpu.lock` immediately acquirable; both GPUs at 0 MiB / 0% with no
  compute applications (2026-08-09T00:23+03:00).
- Work remained on `lane/cx-batchdraft`; nothing was pushed, tagged or released.
