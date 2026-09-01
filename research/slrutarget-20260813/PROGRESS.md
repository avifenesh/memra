# SLRU acceptance target progress — 2026-08-13

## Scope and provenance

- Worktree: `/home/avifenesh/projects/wt-cx-slrutarget`
- Branch: `lane/cx-slrutarget`
- Frozen base: `v0.82.0` (`7624b4f5fd914f4909056ae794f5229fbd14b21b`)
- Deliverable: derive an acceptance target from a defensible traffic model, compare SLRU with plain
  LRU at the production 4096 MiB budget and the cache-size lane's recommended budget, identify any
  workload where SLRU loses, and recommend whether the default should change.
- This lane will not re-run the shipped `slrucache` campaign, edit engine code, alter generated perf
  boards, merge, tag, push, or touch the live serve box.

## Constraints carried into the lane

- Treat committed `research/cachesize-20260813/` evidence as the real-traffic anchor when available.
- Keep scored work off box1; any local RTX 5090 behavior cell must retain the thermal cap and acquire
  `flock /tmp/memra-5090.lock`.
- Tee raw output before parsing it. Quote captured failures rather than inferring their causes.
- Keep spill-performance and model-quality claims out of this cache-policy decision.
- Do not flip the default in this lane.

## Checkpoints

### 2026-08-13 — lane initialized

- Confirmed a clean worktree on the requested branch and exact `v0.82.0` base.
- Created this file before reading or running the research campaigns.
- Next: read the required prior-lane evidence, inventory the cache-size lane as committed at start,
  and freeze the traffic-model inputs before calculating results.

### 2026-08-13 — required evidence read and traffic model frozen

- Read `research/slrucache-20260813/{RESULTS.md,PROGRESS.md}` and the complete
  `research/evictchurn-20260813/` report, ledger, Python driver, and local runner. The shipped SLRU
  replay is not being rerun.
- `research/cachesize-20260813/` is absent from `v0.82.0`, so this lane froze the committed sibling
  worktree state available at start: `lane/cx-cachesize` commit
  `f5657142ee586cb1a6ea23c857265d9e597a67e6`. Its tree is clean. Its N=5 reduction and production
  recommendation are explicitly pending; its authoritative completed evidence is therefore the
  sold-shape entry-byte table plus committed per-boot raw receipts, not a final cache-size verdict.
- Sold 4,860-token entries are 301,215,744 bytes (287.2617 MiB) for Q27 and 110,964,480 bytes
  (105.8240 MiB) for Q35. A shared Q27+Q35 pair is 412,180,224 bytes (393.0857 MiB). At 4,096 MiB,
  the global cache holds 14 Q27, 38 Q35, or 10 pairs; at the highest tested 49,152 MiB arm it holds
  171 Q27, 464 Q35, or 125 pairs.
- The cache-size protocol uses a 96-key uniform cycle to measure a capacity curve. It is useful for
  sizing but is not adopted as the policy acceptance distribution: it intentionally makes every
  key return only after a full 96-key cycle, whereas the product premise is a returning
  conversation/agent session amid unrelated one-hit traffic.
- Commercial truth is the sold hit-latency envelope, not raw residency. Existing N=5 target-rig
  evidence puts Q27 mixed-hit TTFT at p95 19.820 ms with a 21.565 ms sold ceiling and a clean c=16
  throughput knee; Q35 is p95 10.260 ms at mixed c=4 and has a clean c=40 throughput knee. The
  cache-size lane defines stricter per-request hit-TTFT ceilings of 22 ms (Q27) and 11 ms (Q35), but
  its multi-budget N=5 concurrency reduction is not committed yet.
- Frozen primary estimate: 1,000 requests; eight returning logical sessions (two waves of the sold
  c=4 cap), 90% returning requests drawn Zipf(alpha=1.0), and 10% unique one-hit scan requests,
  shuffled deterministically after one guaranteed reference per hot key; seeds 3407..3436 (N=30).
  Run Q27-only, Q35-only, and a worker-global paired variant in which all eight logical sessions
  retain both model entries and returning/scan requests split 50:50 across models. The paired hot
  bytes are the conservative global-budget case. Eight sessions, Zipf, and the model split are
  estimates, not observed live traffic. The 90/10 split comes from the frozen sell/cache-size shape;
  hotness and short reuse follow published agent/chat workload evidence, but no external traffic
  has reached this endpoint yet.
- Frozen sensitivity grid: logical-session hot sets 4/8/16/32/64/96, Zipf alpha 0.8/1.0/1.2,
  scan shares 0/10/25/50%, Q27-only/Q35-only/paired variants, both measured entry sizes, and budgets
  4,096/49,152 MiB, N=30 deterministic seeds. Add stationary cyclic controls and phased hot-set
  turnover to search for SLRU losses.
- Acceptance target: on the primary model, SLRU must not reduce total or returning-request hit rate;
  after a hot key's first hit, scan-caused misses must be zero; cold scan requests remain misses;
  no simulated accounting overflow occurs. Sensitivity results and any losing workload are reported,
  not averaged away. A policy result cannot by itself claim additional concurrent sessions or
  dollars/day: that requires the pending N=5 target-rig cache-size reduction at the sold TTFT gates.
- No GPU cell is needed: this is a deterministic policy simulation using measured entry bytes and
  the shipped SLRU semantics. No engine code is touched, so the standard GPU exactness battery is
  not triggered.

### 2026-08-13 — deterministic sweep complete

- Added an exact byte-LRU/byte-SLRU simulator and reducer. Before study rows, its self-check
  reproduced all committed evictchurn/slrucache summary counts and returned `PASS`.
- The CPU-only sweep emitted 26,332 raw JSONL rows, covering the primary N=30 model, the full
  432-scenario sensitivity grid, stationary cycles around capacity, and phased hot-set turnover.
  All stdout/stderr was teed before reduction; the raw manifest verifies.
- Primary 4,096 MiB result: Q27 total hit rate 88.170% -> 89.193% (+1.023 pp), Q35 equal at
  89.200%, and worker-global paired 83.493% -> 88.153% (+4.660 pp). SLRU had zero post-first-hit
  misses in every primary arm. All shapes were equal at the 49,152 MiB highest-tested arm.
- Across 432 stationary sensitivity scenarios, SLRU was better in 239, equal in 193, and worse in
  zero. A separate phased-turnover control found the real loss boundary at every size/variant:
  LRU 75% versus SLRU 0% when a stale protected cohort stays idle and the disjoint new cyclic
  cohort is one logical session larger than residual probation capacity.
- Source/config audit found that v0.82.0 has no LRU rollback setting:
  `MEMRA_PREFIX_CACHE_PROTECTED_PCT` selects 1..99 and defaults/falls back to 80, so enabled caches
  run SLRU. `MEMRA_PREFIX_CACHE_MB=0` disables caching entirely. The report records this mismatch
  without touching runtime code.
- Decision: `NEEDS-REAL-TRAFFIC`. The report specifies the privacy-safe live event stream, shadow
  replay, commercial concurrency gate, and explicit rollback seam required to settle the default.
- No GPU, box1, or live endpoint was touched. No engine, board, README, or product file changed.

### 2026-08-13 — final verification

- Deterministic re-reduction is byte-identical to committed `analysis.json`; the raw row count is
  26,332 and every decision assertion passes. Python compilation, shell syntax, diff whitespace,
  exact v0.82.0 ancestry, branch identity, and research-directory-only scope all pass.
- `raw/SHA256SUMS` verifies every raw artifact, including `raw/final-validation.log`. The frozen
  cache-size source still states that its N=5 reduction and production recommendation are pending.
