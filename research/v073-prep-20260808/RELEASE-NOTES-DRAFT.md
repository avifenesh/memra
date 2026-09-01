Changes since v0.72.0:

## Performance

- PP-2 fresh prefill now overlaps adjacent prime chunks with concurrent stage-owned host walkers.
  The final ungrouped N=5 medians are 330.0 / 401.8 / 417.6 tok/s at pp512/2048/4096, with a
  200-prime exactness soak clean; `MEMRA_PRIME_PIPE=0` is the schedule-only serial rollback.
- Naked PP-2 auto geometry now uses a dynamic equal-work microchunk schedule. It is bit-identical
  to fixed ranges and measures +1.4% / +0.3% / flat at pp512/2048/4096 (N=5); use
  `MEMRA_PRIME_CHUNK_SCHED=fixed` for the equal-token rollback. No pp4096 material-win claim.
- Lever C groups Step35 expert prefill inside the legal host-sigmoid routing family. It wins
  53–63% on the rented Step pair, but loses 75.3% on the local resident-KAT transfer gate, so it
  ships **opt-in** as `MEMRA_MOE_GROUPED=1`; the naked default remains the established path.
- Step35 simultaneous complete fresh prompts now batch their weight streams across PP stages while
  retaining per-request attention/KV state. The measured T=520 gain is 2.5% at B=2 and 2.3% at
  B=4 (N=5); `MEMRA_STEP35_PRIME_BATCH=0` restores the serialized fallback.
- TTFT tracing found only 6–10 ms outside prime. Completion-only timing, SSE-keepalive exclusion,
  and one widened outer prefill call for a sole fresh request move 4k TTFT from 7.118 s to
  5.992 s while short TTFT stays 0.589 s.
- Sharded cross-device PP-2 now defaults to plain batched decode because spec loses every measured
  q9 and Step35 c=1/2/4 cell. Single-card keeps the low-concurrency spec lane; forced spec remains
  reachable for rollback and the #87 crash gate.

## Features

- Same-window cold fanout now computes one exact tenant-scoped prefix, deep-copies the snapshot to
  siblings, reports exact cached-token credit, and pins the entry until every participant retires.
  The N=8 Step burst moves from 22.263 s to 3.852 s p50 TTFT; cross-tenant and cross-salt sharing
  remain forbidden. `MEMRA_PREFIX_DEDUP=0` is the rollback.
- Speculative depth is request-owned. With `MEMRA_SPEC_K` unset, losing placement/concurrency cells
  choose K=0, cached-long prompts with at least 1024 resumed tokens choose K=2, and other eligible
  cold prompts choose K=3. A non-negative `MEMRA_SPEC_K`, including 0, pins the operator choice.
- The architecture onboarding kit centralizes migrated per-layer geometry, generates guarded
  chunk/tick/B>1 gate scaffolds plus registry rows, and makes `docs/ONBOARDING.md` the canonical
  artifact-to-green runbook with Qwen 3.8 as the worked example.

## Fixes

- The GPU Gumbel sampler no longer admits `u == 1.0f`: 128 of 2^32 Philox values rounded to one,
  making `-log(-log(u))` infinite so an arbitrary vocabulary id won the sample — roughly one
  injected token per 261 sampled at a 128k vocabulary. The clamp changes no other sample's bits;
  the two live corruption seeds are now a standing `sample-check` gate.
- Release builds actually release prefix pins. The only production unpin call sat inside
  `debug_assert!`, which compiles its whole argument out of release builds, so served/fanout
  cache entries pinned VRAM forever. A release-profile regression test pins the fix.
- Explicit `max_tokens` is exact: the worker clamps emission to the request budget while the
  engine keeps its cache-authoritative surplus, so clients are never billed past their cap.
  Speculative bursts emit one token event per visible id (EOS included), `n_tokens` always
  equals the token-array length, and spec/round-robin paths now count `tokens_out`, per-lane
  tokens, and step timing — the rate limiter and QoS gate previously ran on defaults on a
  spec-only box. Spec and plain decode produce byte-identical output for the same request.
- Plain-decode session affinity checkpoints conversations at the last turn marker and resumes
  rewritten histories by rollback: TTFT slope on growing conversations drops from 0.224 to
  0.062 ms per uncached token, and the cached-token counter advances instead of freezing.
  Checkpoints arm only for nominatable sessions, so anonymous fanout keeps full prefix-cache
  sharing.
- Admission charges each request from its own effective context cap through one shared
  allocation-geometry source instead of a scalar frozen at first admit, and a shortfall now
  reclaims the oldest parked session before deferring — the receipted c=4 serialization behind
  parked state is gone.
- Client `cache_salt` is validated and length-capped at the HTTP boundary, and raw `t:`-prefixed
  salts are rejected when no keyring is configured (pre-auth cache-namespace flooding and tenant
  spoofing are closed).
- `/v1/models` entries advertise `supported_parameters` from probed model capabilities, including
  the reasoning knobs on thinking models.
- Step35 grouped prefill preserves the sequential oracle's ordered arithmetic on both unclamped
  and clamped layers; model-backed comparisons are byte-identical.
- The required 5090 transfer gate now controls the grouped default, preventing the rented-pair win
  from silently promoting a 75.3% local regression.
- TTFT traces cover completion routes only and no longer count SSE keepalive comments as first
  output.
- Dynamic-microchunk batteries stop on the first red and refuse performance handoff unless the rig
  is idle.
- The PP counter-union merge restores the `Drop` implementation's closing brace.

## Documentation

- The request-conditioned K contract, dynamic schedule, fanout dedup/pinning design and results,
  pipeline ordering, and the consolidated architecture-onboarding runbook are documented.
- RunPod serving operations now explicitly prohibit `MEMRA_CONFIDENCE_TRACE` and
  `MEMRA_DEBUG_SPEC`: both expose decodable request/completion token IDs, and the boundary is an
  operator rule rather than a code-enforced safeguard.

## Other

- The explicit Step trial config is serve-ready through the HTTP surface: 0.595 s short TTFT,
  6.052 s 4k TTFT, 12.2 ms 4k cache-hit TTFT, 36.5 tok/s per stream at c=4, and a ten-minute
  124/124 replay with zero 5xx or sheds. The 4k cache entry is about 343 MB, so deployments must
  size `MEMRA_PREFIX_CACHE_MB` above the 256 MB default for that workload.
- Concurrent-prefill anatomy rejects scheduler work as the path to 3K tok/s: four different-prefix
  4k primes saturate the pair at 580.5 tok/s versus the 674 tok/s solo class. No production
  concurrent-prime mechanism lands; `MEMRA_TICK_TRACE=1` is diagnostic only.
- The merged target-rig batteries cover `kernel-check`, `run-gen` argmax, `run-spec` K=1..8,
  PP split/pipeline liveness, Step chunk/tick invariance plus canaries, prefix isolation and
  accounting, `serve-smoke`, and request-policy behavior. Release tagging still requires the
  designated-rig battery on the eventual version-bump commit.

Boards + reproduction artifacts: https://huggingface.co/Avifenesh/memra-bench · full experiment log in research/tune-data/
