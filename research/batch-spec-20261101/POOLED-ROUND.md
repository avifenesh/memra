# Pooled MTP verify in the serving round

Step 2 of the batch-spec lane: the worker's spec phase runs same-model MTP continuation bursts in
lockstep, and each round's verifies share one target forward.

## What runs

- **Worker** (`step_spec_pooled`, worker.rs). Each tick, phase (a) groups the MTP sessions that
  are on a continuation burst by model (`spec_pool_candidate`). A session qualifies when its prompt
  is fully primed, no prime walker is in flight, it is unconstrained, it has no vision input and it
  is not on another spec route. Every group of two or more runs as a cohort.
  - Each member runs the same preamble, round-cadence emission and post-burst accounting that
    `step_session` runs for it alone. These were extracted unchanged into `spec_burst_pre`,
    `SpecBurstEmit` and `spec_burst_post`.
  - Results feed the existing per-session handling: OOM park, errors, finish and telemetry.
  - A fault inside a cohort reaches every member as that member's step error.
- **Engine** (`generate_spec_sessions_pooled`, spec.rs). It begins one burst per member
  (`spec_burst_begin`), then runs rounds in lockstep:
  - every live member runs its own gate and prepare;
  - the round's verifies run as pooled forwards (`verify_pooled_dev`) over consecutive members, at
    most 16 rows per forward;
  - every member runs its own finish.

  A member leaves the lockstep when its own loop condition ends, and its tail runs right away. A
  round that cannot pool (a single member, or a verify-graph round) verifies serially.
- **Who pools** (`spec_pool_capable`):
  - the qwen35 t-parallel verify family with an MTP head;
  - no pipeline-parallel cuts, no ROUND-STREAM;
  - no MTP verify-graph pool. That pool is a model-owned mutex that a burst holds for its whole run,
    so it is on by default only for GDN+MoE models; the dense 27B and the 9B pool.
- **Attention rows below the fa vec floor** now pool as well. They take the serial walk's per-row
  `fa_decode_kvmod_view` arm over their own session's cache.

## Exactness

1. **Pooled verify vs per-session verify** (`decode-batch-bench --pooled`, 5090, q9). Bitwise
   equal on every logit row, and on a follow-up step that reads the written state. Shapes tested:
   4x4, 1..5 and 2x2 at depths 40 (below the vec floor), 200 and 2048, plus the earlier
   receipts.
2. **Served streams, pooled vs serial** (`battery/pool_cell.py`, `battery/compare-pool-cell.txt`).
   - Setup: one fresh boot per arm, `MEMRA_SPEC_GATE=0` so every session speculates.
   - Load: waves of c = 2, 4, 8 concurrent requests, alternating greedy and sampled (T=0.8,
     per-request seeds), 200 tokens each.
   - Arrivals are staggered 0.25 s and `MEMRA_ADMIT_YIELD=0` fixes burst boundaries. Both are
     needed to make the serial reference reproducible at all:
     - two serial runs with simultaneous arrivals differed on one sampled stream, because a
       cross-request batched prime is a different numeric class;
     - two serial runs with arrival-timed burst ends differed on one sampled stream, because a
       sampled burst draws its boundary token from the session's Philox stream at every burst end.
   - Result: serial, serial rerun and pooled all match on **14/14** streams. The server logged
     pooled cohorts of 2 to 8 sessions.
3. **Single-session and mixed paths** (the ROUND-SPLIT battery on the pooled binary, against the
   round-split reference). `battery/compare-pool-serial-paths.txt` is **BYTE-IDENTICAL** on every
   cell. That includes the battery's concurrent greedy pair, which now pools.

## Speed (RTX 5090 Laptop, q9, the pool cell, aggregate tok/s)

| c | serial | serial rerun | pooled |
|---:|---:|---:|---:|
| 2 | 101.78 | 101.74 | 113.53 |
| 4 | 98.00 | 98.23 | 110.48 |
| 8 | 98.18 | 98.28 | 110.88 |

The gain is +11-13%.

- **Why it is small on the 5090:** for q9 on the 5090, verify is about half of a serial round, and
  pooled verify is 1.25-1.65x over that half. Drafting stays serial per session: K MTP steps, each
  with a 4-byte readback. Batching the draft is step 3. Above 4 sessions at K=3, the 16-row tier
  splits the cohort into several forwards.
- **On GB10:** the verify over 1-8 rows costs the same as one row (GB10-CR.md), so the pooled
  share is much larger there.

## Binaries

`memra-server` sha256 prefixes: pooled `13ec038086e7700a`, round-split reference
`559b914819a7ffa4`. The committed tree differs from the measured pooled binary only by two
clippy type aliases (`PooledCommitHook`, `PooledBurstResult`).

## Follow-ups (2026-11-02)

- **Chunked finish.** The lockstep driver finishes each chunk's rounds before the next chunk's
  verify (`pooled_finish_member`). A round never holds more than one chunk's verify outputs and
  checkpoints. On a dense 27B that is about 1 GB per session-round. The DFlash2 port hit the same
  OOM at c=8 and needed the same fix.
- **Prime bursts end at the first round.** A session priming its prompt is not a cohort candidate,
  so its first 32-token burst ran serially. On q9 at c=4 that was four serial bursts before any
  pooled round. When a same-model spec peer is live (`spec_pool_peer`), a prime burst now emits
  one token and the session joins the next tick's cohort.
  - Greedy streams do not change.
  - For a sampled request under concurrency it moves one boundary draw, the same class as an
    admission-yield burst end.
  - A lone session keeps its full first burst.
- **Token accounting.** Cohort members now count their tokens; see COST-GATE.md.
