# cx-evictchurn progress

## Scope

Measure prefix-cache eviction behavior under a contended, multi-tenant working set on the
thermally capped local RTX 5090. This is a behavior lane: no absolute-throughput claims.

Starting point: `lane/cx-evictchurn` at tag `v0.81.2`
(`18885ec479d897a3e8c42b0d408a71fa3edaa708`).

## Initial rig check — 2026-08-13

- GPU: NVIDIA GeForce RTX 5090 Laptop GPU.
- Observed before taking the card: 157 MiB used, 0% GPU utilization, 210 MHz current graphics
  clock, 57 C.
- `/tmp/battery-*.log` contains no current run; the newest log is from 2026-08-12 and completed.
- The global 210–1200 MHz thermal cap will not be changed.

## Checklist

- [x] Verify clean dedicated worktree and `v0.81.2` base.
- [x] Read the required prefixmoney report/harness/gate, serving docs, engine cache, and worker
  insert/evict path.
- [x] Record the current policy with file-and-line evidence.
- [x] Extend the existing workload shape for round-robin, Zipf/hot-set, and sequential scan.
- [x] Capture hit, hot-hit, eviction, refusal, churn, avoidable-eviction, and hit/miss TTFT data.
- [x] Run prefix-cache byte-exactness.
- [x] Write `RESULTS.md`, retain raw logs, and commit the completed lane.

## Decisions / blockers

- The scored run is `raw/run-20260812T221018Z`. It used the requested 35B IQ4_XS artifact,
  a fixed 782 MiB cache budget (12 of 40 equal-sized entries), four tenant salts, and fresh server
  boots per pattern. All phases passed and the manifest verifies.
- `raw/run-20260812T220712Z` is a retained non-scored attempt. It stopped because the imported
  exactness client rejected valid EOS-only completions; `ATTEMPT.md` records the exact failure and
  the narrow harness correction.
- Verdict: **needs-policy-fix**. Timestamp LRU fully thrashes the 40-prefix round-robin trace and
  allows cold one-hit inserts to displace 13 of 120 hot reuse opportunities in the 80/20 trace.
  The proposed next experiment is a byte-budgeted probation/protected SLRU; no engine policy was
  changed in this lane.
- Runtime code is unchanged, so the engine-change GPU battery was not triggered. The existing
  prefix exactness gate passed, cache-hit byte identity passed 107/107 within the contention run,
  and the targeted host prefix-cache tests passed 8/8.
