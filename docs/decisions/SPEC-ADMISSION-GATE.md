# Speculation is admitted by load, and a live speculating session demotes under load (2026-08-07)

**Status:** amended. Adopted 2026-08-07 (lane/spec-gate) as a LOW/HIGH concurrency band on every
placement. Since 2026-08-08 a sharded cross-device PP-2 placement never admits speculation. Since
2026-11-01 sampled sessions demote too. Since 2026-11-02 a boot-measured cost model decides for
MTP-served models that have an acceptance verdict, and the band decides the rest (see
[Amendments](#amendments)). Code: `spec_gate_defaults` and `resolve_spec_gate_thresholds` in
`crates/memra-server/src/worker.rs`. Flags: `MEMRA_SPEC_GATE`, `MEMRA_SPEC_GATE_LOW`,
`MEMRA_SPEC_GATE_HIGH`, `MEMRA_SPEC_GATE_MODEL` ([docs/FLAGS.md](../FLAGS.md)). Receipts:
[`research/spec-serving-20260801/`](../../research/spec-serving-20260801/),
[`research/spec-scaling-20260806/`](../../research/spec-scaling-20260806/),
[`research/spec-gate-20260806/`](../../research/spec-gate-20260806/RESULTS.md),
[`research/specplace-20260808/`](../../research/specplace-20260808/PROGRESS.md),
[`research/batch-spec-20261101/COST-GATE.md`](../../research/batch-spec-20261101/COST-GATE.md).

The rationale below moved out of the serving doc on 2026-11-12. The old text is kept in
[docs/archive/SERVING-2026-11-12.md](../archive/SERVING-2026-11-12.md#measured-numbers-qwen35-9b-q8_0-receipts-in-research)
("Spec fast lane").

## Question

MTP speculation is a single-stream latency tier. On an H100 80GB with a 27B Qwen MTP cell it ran
1.82x plain serving at c=1 (131.8 against 72.5 tok/s, 2026-08-01,
`research/spec-serving-20260801/batched-spec-design.md`). Plain batching overtakes it between c=2
and c=4, because the spec path is a serial burst queue: each speculating session's whole burst is
stepped in a host loop, and spec rows are excluded from batched decode. The guidance then was to
run spec and bulk traffic as two server processes. Can one server serve both?

## Decision

One server admits speculation per request, by load:

- admit a new request to the spec path only while `active + 1 <= LOW` (default 2, the last
  measured winning rung);
- demote every live speculating session into batched decode at `active >= HIGH` (default 4, the
  first measured losing rung);
- `active == 3` is a hysteresis band: a speculating session keeps bursting, a new arrival is
  admitted batched. `HIGH <= LOW` is clamped to `LOW + 1` with a warning, because no band means
  mode thrash;
- demotion is one way per session. A demoted session stays in batched decode until it ends. New
  arrivals take speculation again once the count falls back to `LOW`, so the policy is per
  request and a session switches mode at most once.

The demotion is a cache handoff: the session's `(cache, next_pred)` moves into the batched row,
and a carried pending token is committed first so the cache is not one row short of the emitted
stream. For greedy it is byte-exact: a session demoted mid-generation writes the same stream as one
batched from the start.

`MEMRA_SPEC_GATE=0` restores always-spec on every placement (the rollback seam). An operator
`MEMRA_SPEC_K` pin owns the policy and bypasses the gate.

## Rejected

- **Two processes, one for spec and one for bulk.** The superseded guidance. One process with the
  gate keeps both curves (below) and needs no client-side routing.
- **Pooling the verify across sessions (August 2026).** Refuted at the time by the exact decode
  kernels' 16-column width ceiling: a pooled verify could carry at most 16 columns across all
  sessions, so it could not lift the crossover far enough
  (`research/spec-scaling-20260806/RESULTS.md`, "The hard width ceiling is T=16"). The 2026-11
  batch-spec lane later built pooled verifies in 16-row chunks; the cost model prices them.
- **Bounding the burst quantum (`MEMRA_SPEC_BURST=4`) to soften the mixed tick.** Measured: it does
  not recover the mixed-tick penalty and costs throughput (`research/spec-gate-20260806/RESULTS.md`).
- **Admission gating without demotion.** A session admitted while the box was quiet keeps its
  whole-burst hold when load arrives. Demotion is what pulls per-stream p95 in the worst mixed state
  from 11.243 s to 7.249 s (same receipt).
- **The single-card band on PP-2.** See the placement measurement below.

## Measurement

**The band, 2026-08-06.** One RTX 5090 Laptop GPU (24 GB), Qwen3.5-9B NVFP4 + its trimmed MTP
drafter, context 4,096, K=3, greedy. c-ladder N=5, rep-major with rotating arm order, 60 load
points, 0 errors. Aggregate tok/s, medians:

| c | gated | never-spec | always-spec |
|---:|---:|---:|---:|
| 1 | 251.2 | 138.5 | 251.9 |
| 2 | 250.0 | 221.5 | 250.8 |
| 4 | 357.4 | 383.5 | 249.7 |
| 8 | 504.7 | 521.2 | 248.4 |

The gate tracks speculation at c=1-2 (1.81x over batched at c=1) and batching at c=4-8 (2.03x over
always-spec at c=8). Per-stream p50 at c=8 is batched's 1.963 s, against always-spec's 3.973 s.
The cost it keeps: a first-wave TTFT p95 of 0.423 s against never-spec's 0.017 s at c=4. It is
confined to the at most `LOW` sessions admitted before a load ramp (requests 0-7 of the c=8 wave,
25% of requests at c=4 and c=8, the harness's first-wave fraction); p50 matches never-spec. An
operator for whom cold-ramp p95 matters more than c=1 throughput sets `MEMRA_SPEC_GATE_LOW=0`.

Exactness: a forced demotion at generated token 129 of 768 (B=1 on both sides, the diagnostic
`MEMRA_SPEC_DEMOTE_AT` door) was byte-identical to the batched-from-start stream, 3,122 bytes. The
accept-gate cells passed 6/6, `run-spec` K=1..8 passed, serve-smoke had 0 failures, serve-stress
c=64 was 64/64. Receipt: [`research/spec-gate-20260806/RESULTS.md`](../../research/spec-gate-20260806/RESULTS.md).

**Placement, 2026-08-08.** Two RTX PRO 6000 Blackwell Server Edition cards, Qwen3.5-9B
NVFP4-MTP, greedy, N=3 or more per cell, decision rule written before the runs. Aggregate tok/s,
forced spec (`MEMRA_SPEC_GATE=0`) against forced plain (`MEMRA_SERVE_SPEC=0`):

| placement | c | spec | plain | spec/plain |
|---|---:|---:|---:|---:|
| PP-2, cross-device | 1 | 112.5 | 223.3 | 0.50x |
| PP-2, cross-device | 2 | 112.3 | 340.3 | 0.33x |
| PP-2, cross-device | 4 | 112.1 | 593.4 | 0.19x |
| single card | 1 | 374.8 | 224.5 | 1.67x |
| single card | 2 | 374.5 | 347.5 | 1.08x |
| single card | 4 | 377.3 | 617.1 | 0.61x |

The `step35` architecture on PP-2 lost too (35.9/36.2/36.7 against 85.7/101.6/121.7 tok/s at
c=1/2/4, N=3). So a
sharded cross-device PP-2 placement defaults to `LOW=0`/`HIGH=1` (never admit), and single card
keeps `LOW=2`/`HIGH=4`: the crossover stayed between c=2 and c=4 on the current train. PP-N,
same-device PP and the other rollback shapes keep the single-card band, because they were not
measured as PP-2 cells. Receipts:
[`research/specplace-20260808/PROGRESS.md`](../../research/specplace-20260808/PROGRESS.md),
[`research/pp2spec-crash-20260807/`](../../research/pp2spec-crash-20260807/).

## Amendments

- **Sampled sessions demote (2026-11-01).** The 2026-08-07 band excluded sampled and constrained
  sessions from demotion. Sampled sessions now demote: the handed token is already a draw, the
  penalty window is the worker sampler's own history, and a demoted sampled row starts its plain
  draws at the spec session's first unused Philox event (`demoted_philox_base`). Constrained
  sessions still do not demote (their `next_pred` is the unmasked verify argmax) and stay bounded
  by `LOW`. GLM5 sessions demote greedy only.
- **The cost model decides where it can (2026-11-02, lane/batch-spec).** At boot every MTP-served
  model measures its verify and plain-step costs on the device (`[spec-cost]` line). Once the
  model has 64 rounds of per-position acceptance, admission K and tick demotion take the draft
  length (0 = plain) that maximizes predicted aggregate tok/s at the projected concurrency, and
  realized tick rates replace the prediction after three samples. On the RTX 5090 Laptop with
  Qwen3.5-9B (three boots per arm, each concurrency measured on the boot's first wave at it) the
  final binary served 0.99x the band at c=1, 1.06x at c=2, 0.90x at c=4, where one exploring wave
  pays for the learning, and 1.02x at c=8. The band still decides for operator pins, the PP-2
  default, explicit `MEMRA_SPEC_GATE_LOW`/`_HIGH`, DFlash2-served models and models without a
  verdict.
  `MEMRA_SPEC_GATE_MODEL=0` restores the band everywhere. Receipt:
  [`research/batch-spec-20261101/COST-GATE.md`](../../research/batch-spec-20261101/COST-GATE.md).
- **The K table inside an admitted request** has its own records:
  [SPEC-K-COLD-LONG-K2.md](SPEC-K-COLD-LONG-K2.md) and
  [SPEC-K-COLD-SHORT-5090.md](SPEC-K-COLD-SHORT-5090.md). The entry-yield bound for a speculating
  session under short newcomers is [SPEC-ENTRY-YIELD-BOUND.md](SPEC-ENTRY-YIELD-BOUND.md).

## What would change it

A placement or card where speculation wins at the concurrencies the band refuses, measured as an
interleaved A/B, moves that placement's default. A cost-model receipt on GB10 and the RTX 5090 that
settles the band-against-model A/B removes `MEMRA_SPEC_GATE_MODEL`.
