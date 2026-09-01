# specmech-20260810 results — c=2 speculative PP pipeline

## Verdict

**HOLD.** Increment 1 is exact, default-off, and policy-safe, but it misses the
performance win. At c=2 its N=5 median is **53.691 tok/s**, only **+0.27%** over
serial spec and **-53.26%** versus plain. `MEMRA_SPEC_PIPE` must remain an
experimental opt-in; do not promote it to a runtime default.

The phase receipt also closes the proposed increment-2 pricing question. Even an
optimistic schedule that hides every measured draft, verify-readback, and commit
millisecond under the opposite session's verify projects only **55.195 tok/s**,
still **-51.95%** versus plain. The missing gain is in verify issue itself, not in
the small primary-engine tail.

Scored remote source: `4e0a8ce25eb9c1801e7c9c84ac71ad60347925ba`
(local code-equivalent shutdown commit: `6b7acac8`). Scored server SHA-256:
`a2f457d73b099add209a9cb6f7799deff9226c1b1080f8d30e0b683c6dc8ba43`.

## As-built schedule

Admission is deliberately narrow: two same-model, same-K, greedy,
unconstrained, devacc-enabled, warm continuation sessions on cross-device PP-2.
Cold primes, sampled or constrained requests, diagnostics, unequal/odd lanes,
and every shape outside this matrix use the pre-existing serial path.

Each paired call keeps two existing `generate_spec_inner2` call stacks, so all
round-local buffers are naturally per session. `SpecSession` continues to own
cache, MTP scratch, committed tokens, predictions, graph state, pending token,
and telemetry. The primary-engine mutex serializes setup, draft argmax,
verify-column argmax/accept, commit, refresh-fill, and session tail.

The intended interval order is:

```text
draft A -> draft B -> one reverse fence
        -> enqueue A verify (A.s0 -> A.s1/head)
        -> enqueue B verify (B.s0 -> B.s1/head)
        -> accept/commit A -> accept/commit B
```

The two PP stages retain separate stage engines, and alternating persistent
boundary slots protect A-stage1 / B-stage0 overlap. Ordinary PP callers keep the
old fence and slot path. The worker pairs only after the existing cold-first
sort, flushes both returned bursts through the existing accounting path, and
falls back serially when a peer finishes early.

## Method

- Box: `<private-host-redacted>`, 2x NVIDIA RTX PRO 6000 Blackwell Server Edition.
- Artifact: Step-3.7-Flash IQ4_XS target plus Q8_0 MTP draft. Target bytes:
  46,483,327,296; first-shard SHA-256
  `b940497a9cec2f801f07e3a9783f2115fd8bf79cbd453225b4f73d86bcd11259`.
  Draft bytes: 3,707,276,416; SHA-256
  `469a81667a6cd6d87a85d501d57155fd90cee5af7010fd289c5169881763fd57`.
- Shape: `CUDA_VISIBLE_DEVICES=0,1`, `MEMRA_PP_STAGES=2`,
  `MEMRA_PP_DEVICES=0,1`, `MEMRA_CTX=262144`, `MEMRA_MOE_GROUPED=1`,
  `MEMRA_PREFILL_TICK=2048`.
- c=2: concurrency 2, 8 measured requests, 128 greedy completion tokens,
  2 warmups. c=1: 4 requests and 1 warmup. c=4: 8 requests and 4 warmups.
- Each point used a fresh server. The N=5 arms used rotating order under one
  uninterrupted `/tmp/memra-gpu.lock` hold from 15:31:23Z to 16:03:48Z.
- Thermal regime for every published median: N=5 interleaved, P0, no other
  compute apps at the lock boundaries. Across the 37 arms, 74 pre-run GPU
  samples were 29–36 C and 74 post-run samples were 33–35 C; post-run clocks
  were 2317–2415 MHz. These are boundary snapshots, not an in-run throttling
  trace.

Raw method and measurements are in
[`raw/perf/driver.log`](raw/perf/driver.log),
[`raw/perf/points.jsonl`](raw/perf/points.jsonl), and
[`raw/perf/requests.jsonl`](raw/perf/requests.jsonl).

## A/B results

All values below are aggregate completion tok/s. Every median is N=5 under the
thermal regime above.

| Shape | Arm | Median | Min–max | Delta | Requests |
|---|---|---:|---:|---:|---:|
| c=2 | plain | 114.878 | 114.742–114.911 | reference | 40/40 OK |
| c=2 | serial spec | 53.544 | 53.482–53.602 | -53.39% vs plain | 40/40 OK |
| c=2 | pipelined spec | 53.691 | 53.501–53.723 | +0.27% vs serial; -53.26% vs plain | 40/40 OK |
| c=1 | serial spec | 53.475 | 53.392–53.568 | reference | 20/20 OK |
| c=1 | PIPE door open | 53.552 | 53.449–53.596 | +0.14% vs serial | 20/20 OK |
| c=4 | plain | 141.157 | 140.742–141.390 | reference | 40/40 OK |
| c=4 | default policy + PIPE door | 141.037 | 141.014–141.356 | -0.08% vs plain | 40/40 OK |

c=1 emitted zero `[spec-pipe]` lines, proving the door used the serial fallback.
Every c=4 policy arm logged `K=0 source=pp2-placement`, with zero `[spec-acc]`
or `[spec-pipe]` lines. The default PP-2 policy is therefore unchanged.

## Exactness

| Gate | Result | Evidence |
|---|---|---|
| Local `cargo check` | PASS | Final code, exit 0 |
| Local `cargo test --workspace` | PASS | Final code, exit 0; all runnable tests passed |
| `run-spec` K=1..8 | PASS 8/8 | [`raw/exactness-final/run-spec.log`](raw/exactness-final/run-spec.log) |
| b1fix plain | 2/2 golden | [`raw/exactness-final/plain/qos-summary.json`](raw/exactness-final/plain/qos-summary.json) |
| b1fix serial spec | 2/2 golden | [`raw/exactness-final/serial/qos-summary.json`](raw/exactness-final/serial/qos-summary.json) |
| b1fix pipelined spec | 2/2 golden; paired path engaged | [`raw/exactness-final/pipe/qos-summary.json`](raw/exactness-final/pipe/qos-summary.json) |
| Plain vs serial vs pipeline bytes | byte-identical | [`raw/exactness-final/identity-summary.json`](raw/exactness-final/identity-summary.json) |

All six b1fix completions are the same 326-byte response, SHA-256
`21b8293f2298978c74fb89f32d9b14e3ea921f39924cfce88c73b01f445bb6de`.
The final-head exactness block ended `EXACTNESS_PASS`.

The first performance attempt also exposed a shutdown-only lifecycle race:
after `[server] drain complete`, CUDA deinitialized before a paired session's
pending-token retirement finished. The exact failure is retained in
[`raw/perf-attempt1/c2-r01-pipe-server.log`](raw/perf-attempt1/c2-r01-pipe-server.log).
The server now owns and joins the GPU worker during graceful shutdown. The
direct regression receipt is in [`raw/shutdown-smoke`](raw/shutdown-smoke), and
all 37 final performance server logs end with GPU-worker shutdown complete and
contain no failure signature.

## Gap decomposition

The dedicated trace observations were 53.551 tok/s serial and 53.688 tok/s
pipelined. They are excluded from N=5. Raw phase lines are in
[`raw/perf/c2-trace-serial-server.log`](raw/perf/c2-trace-serial-server.log)
and [`raw/perf/c2-trace-pipe-server.log`](raw/perf/c2-trace-pipe-server.log).

Serial continuation phase medians, in milliseconds per session round:

| Burst shape | Phase records | Draft | Verify issue | Verify wait | Commit/host | Total |
|---|---:|---:|---:|---:|---:|---:|
| 19 rounds | 16 | 0.711 | 23.221 | 0.326 | 0.147 | 24.405 |
| 17 rounds | 3 | 0.712 | 23.265 | 0.324 | 0.147 | 24.447 |

The paired trace contains eight 19-round intervals and four 17-round intervals.
Their median walls are respectively **48.662 ms** (N=8) and **48.655 ms**
(N=4) per two-session round pair. That is effectively the sum of two serial
rounds. In the paired call stacks, verify-issue itself expands to 47.097 ms per
round (19-round bursts, 16 phase records), so the intended stage overlap does
not shorten the critical path.

The coordinator releases B verify only after A's complete PP verify body has
returned and marked `verify_done`. On this path, body-level issue does not get
far enough ahead of execution to expose useful A-stage1 / B-stage0 overlap.
This is directly consistent with the pair wall, without requiring an inferred
OOM, clock, or kernel explanation.

For the requested full steady-state price:

- The 12 paired continuation intervals cover 760 of the 1024 measured tokens
  and consume 10,703.695 ms. Cold setup/first bursts cover the other 264 tokens
  and remain serial in this matrix.
- From the serial phase medians, the maximum hideable primary work is 22.5 ms
  per session for each 19-round burst and 20.1 ms for each 17-round burst.
  Across both sessions and all 12 intervals, that is **520.8 ms**.
- Subtracting every one of those milliseconds from the full traced wall is an
  optimistic upper bound: 19.073286 s -> 18.552486 s, or **55.195 tok/s**.
  That is only +2.81% over the measured pipeline trace and remains -51.95%
  versus the N=5 plain median.

Therefore overlapping accept/commit/draft-next alone cannot close the gap. A
future experiment would first need a boundary-level release after A's stage-0
TX, rather than after A's whole PP body, and must re-establish exactness before
claiming any stage overlap. That experiment is not part of increment 1.

## Promotion decision

- Exactness gate: PASS.
- c=1 fallback: PASS.
- c=4 policy preservation: PASS.
- c=2 performance win: FAIL.
- Decision: **HOLD**, leave `MEMRA_SPEC_PIPE` default OFF.

No merge, tag, origin push, or perf-board update was performed.
