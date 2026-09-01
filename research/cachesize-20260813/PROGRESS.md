# cx-cachesize progress ledger

Date: 2026-08-13
Branch: `lane/cx-cachesize`
Base: `v0.81.2` (`18885ec479d897a3e8c42b0d408a71fa3edaa708`)
Rig: box1 only (`ubuntu@<rented-box-ip>`, `/opt/scratch/nvme`), one RTX PRO 6000

## Objective

Measure actual prefix-snapshot device bytes and the cache-budget capacity curve for Q27 and Q35 at the frozen 4,860-token sell-gate prompt shape. Produce a numeric one-card production recommendation for `MEMRA_CTX=8192`, `MEMRA_MAX_SESSIONS=96`, with raw logs and tenant-clean shutdown receipts.

## Campaign contract

- Hold `/tmp/memra-gpu.lock` for the entire scored campaign and also exclude GPU1 through its
  coordination lock, `/tmp/memra-gpu-1.lock`.
- Keep one card and one model resident at a time.
- Reuse the frozen requal2/sell-gate workload; working-set size and cycling order must be explicit.
- Run at least five interleaved repetitions, reversing budget order on even repetitions.
- Tee raw stdout and stderr before parsing.
- Record hit/miss TTFT, output throughput, cache counters, admission/defer counters, VRAM, clocks, power, and temperature.
- Stop after the lane commit; do not merge, tag, push, edit generated boards, or touch `<vast-ssh-host>`.

## Status

- [x] Verified clean dedicated worktree and exact `v0.81.2` base.
- [x] Created this ledger before any campaign code or result artifact.
- [x] Audit existing cache accounting and frozen workload machinery.
- [x] Stage the exact tag on box1, acquire the campaign lock, and capture provenance.
- [x] Measure bytes per entry for both models at 4k and 8k.
- [ ] Run the interleaved cache-budget campaign.
- [ ] Analyze the knee, latency envelope, cost counters, and maximum sessions.
- [ ] Write `RESULTS.md`, manifests, and shutdown receipts.
- [ ] Review and commit the complete lane.

## Notes

- Existing requal logs show that the documentation estimate cannot be reused as evidence for this lane; the campaign will derive bytes from current runtime metrics/log accounting on box1.
- Published medians will state repetition count and thermal regime. Any failed run will retain its raw stderr and will not be assigned a cause unless the captured text states it.

## 2026-08-13 — frozen protocol and box1 preflight

- The runtime's `prefix_cache_bytes` metric is the exact sum of device-resident prefix KV plus recurrent-state bytes. A cold full-prompt insert changes both `prefix_cache_entries` and `prefix_cache_bytes`, so the entry-size probe needs no runtime instrumentation or source change.
- The 343 MB documentation value is explicitly a Step-model 4k example. It is not treated as Q27 or Q35 evidence.
- Capacity uses `N=96`, matching the requested production `MEMRA_MAX_SESSIONS`. The 96 logical prefixes are tenant-isolated cache keys carrying the exact qualified 4,860-token prompt identity; this preserves the frozen prompt content while exercising the real per-tenant residency cost.
- Hot requests consume one deterministic shuffled permutation without replacement and wrap only after all 96 keys have been visited. Ten percent of each frozen cell remains unique cold churn. This retains the sell-gate's 9:1 intended mix while letting the measured hit rate fall honestly when the budget cannot retain the working set.
- The minimum requested c=4, c=16, and model-knee cells are included. Intermediate widths are retained to locate the largest concurrency that still meets the sold hit-p95 class rather than merely bracketing it.
- Box1 preflight at `2026-08-12T21:50:40Z`: both PRO 6000 GPUs reported 0 MiB used, no compute applications or target-port listeners were present, the GPU flock had no holder, `/opt/scratch/nvme` had 3.1 TiB free, and both model hashes matched the frozen receipts.
- Harness static checks and the pure control test pass: each cycle visits all 96 keys exactly once before reshuffle, and every concurrency cell preserves the frozen request-count and 9:1 role rules.

## 2026-08-13 — entry accounting and scored restart

- The exact v0.81.2 source was staged in a dedicated detached remote worktree and rebuilt fresh for CUDA 13.2 / sm_120a. The resulting `memra-server` SHA-256 is `617e25089fdd6b9415b5a35ee27d925dde6534c9c7ec0253789727094eb05ab5`.
- Exact device bytes per entry from the runtime's `prefix_cache_bytes` delta are Q27: 278,528,000 at 4,096 tokens, 301,215,744 at the sold 4,860-token shape, and 400,162,816 at 8,192 tokens; Q35: 103,874,560, 110,964,480, and 141,885,440 respectively. Each probe reconciled one miss, one insert, one retained entry, zero evictions/defers/OOM parks, and passed twice with identical byte counts.
- The documentation's approximately 343 MB 4k Step example is therefore not an entry-size receipt for either scored model. The measured 4k entries are 265.625 MiB for Q27 and 99.0625 MiB for Q35.
- A first partial attempt was excluded at `2026-08-12T22:15:00Z` after a separately locked GPU1 campaign appeared on the rig's shared `PIX` PCIe topology. Only one of 60 budget boots had completed. The owned GPU0 process tree was stopped, the lock and GPU0 were verified clear, and all 53 partial raw files were retained under the remote `attempt1-gpu1-overlap` directory with the observed process/topology receipt.
- A second attempt was excluded after six Q27 boots when Q35 c=32 exposed a harness cycle-boundary defect. Two working keys repeated within the same cell after the 96-key permutation wrapped, so same-window prefix dedup credited both with 1,024 partial cached tokens. The exact counter delta was 4 hits / 36 misses / 11,768 cached tokens while the full-hit-only classifier expected 2 / 38; the harness failed closed and the runner performed a tenant-clean shutdown. The fix deterministically swaps a repeated next-cycle key later in that permutation. An exhaustive control confirms distinct keys in every cell across all 60 planned boots and an exact 0..95 permutation for every complete cycle.
- A third attempt was excluded after a Q27 16 GiB cell returned one HTTP 200 response with only 11 of the requested 60 tokens and `finish_reason=stop`. Its captured counters still reconciled exactly (`20` admitted/completed, `34,020` cached tokens, `7` hits / `13` misses, `1,151` output tokens, zero admission defers and zero OOM parks), and the server log contained no CUDA, OOM, fatal, or error line, so no runtime failure cause is assigned. The harness difference was a batched seed width of eight versus the frozen sell-gate's sequential hot-cache setup; the attempt is retained under `attempt3-batched-seed-eos` and excluded from scoring.
- A focused sequential-seed diagnostic then passed all four Q27 widths at 16 GiB: 80/80 requests produced exactly 60 tokens with no partial completion or server failure line. Seeding is now sequential at concurrency one, matching the frozen sell-gate, and each one-token seed still publishes the snapshot at prefill completion.
- A fourth launch stopped during source preflight, before acquiring the GPU lock or starting a server, because its wrapper pre-created the output directory that the fail-closed harness requires to be absent. The two empty launcher artifacts are retained under `attempt4-launch-wrapper-*`; the corrected wrapper keeps its launcher files beside the scored directory.
- A fifth attempt was excluded after its six Q27 arms and Q35 1 GiB arm passed. Its coarse c=4/8 boundary could only prove that the sold-latency maximum lay in c=4..7; it could not identify the requested exact maximum. The owned in-progress Q35 4 GiB sweep was terminated through its timeout wrapper, which made the fail-closed runner clean up its server and samplers. Both GPUs returned to 0 MiB, ports cleared, and the lock released. The final grid adds c=5/6/7 to both models and restarts from zero so that every capacity arm and the exact latency boundary share one uninterrupted lock hold.
- A sixth partial attempt was excluded before its first expanded-grid boot completed because the working-key permutation seed still included the budget. That made each arm statistically equivalent but not trace-paired, leaving access order as a second changed variable. The owned GPU0 sweep exited through its fail-closed cleanup; GPU0 returned to 0 MiB, ports cleared, and the lock released. At the same boundary an unrelated `cx-lcprestore` job appeared on GPU1 (two captured processes using 15,018 and 14,186 MiB on the shared `PIX` topology); it was observed and left untouched. The corrected cycle seed depends only on model and repetition, and namespace labels are fixed-width, so each budget now sees the identical role/key/concurrency trace. The final runner also holds `/tmp/memra-gpu-1.lock` alongside the required global lock so a conforming GPU1 job cannot enter during the scored regime.
- The full scored campaign restarts from zero only after the paired-trace control passes and both GPUs and both coordination locks are clear. No row from any excluded attempt is eligible for the N=5 reduction.

## 2026-08-13 — recovered scored segment and restored-hit EOS

- The orchestrator checkpoint preserved the lane after two external harness process-group sweeps. A live box1 audit found the remote `raw/scored` segment from `2026-08-13T00:05:06Z`: both entry probes passed again, Q27 repetition 1 at 1,024 and 4,096 MiB completed with `sweep.exit=0` and clean `PASS` summaries, and the 8,192 MiB boot reached a fail-closed summary before the lane process disappeared. The two passing boots remain eligible and will not be repeated; the interrupted-lock segmentation will be reported explicitly rather than mislabeled as one uninterrupted hold.
- Q27 8,192 MiB is excluded. At c=4, paired key 87 was a full 4,860-token hit but emitted the same 11-token EOS sequence and text hash as attempt 3. The identical paired key missed and completed all 60 tokens in both smaller-budget boots. Counters reconcile exactly and the server-failure scan is empty. This is a repeatable exact-length failure on a restored hit, but the paired misses followed a different prime/decode batch class and are not yet a cache-exactness oracle; no lower-level cause is assigned.
- Sequential seeding did not eliminate the divergence, so the earlier batched-seed hypothesis is retracted. Before resuming the long campaign, a focused paired hit-versus-miss diagnostic must determine whether the trigger is stored state, eviction depth, or the decode batch class. The recovered 65-file raw segment is sealed under `raw/attempt7-cache-hit-eos/`; the original remote directory remains untouched.

## 2026-08-13 — restored-hit oracle preflight

- The checkpointed serial oracle passed local Python compilation/import checks; its runner passed
  `bash -n` and `shellcheck`. At 8,192 MiB, exact Q27 byte accounting predicts 28 retained sold-shape
  entries. Target key 87 is inserted ninth from the end, so it remains resident after the 96-key
  fill; both its cold baseline and three restored probes use serial prime/decode configuration.
- Box1 preflight at `2026-08-13T01:49:05Z` found both physical cards at 0 MiB, P8, 26 C, and zero
  utilization, with no compute applications, lock holders, memra servers, or target-port listeners.
  Physical GPU0 is `GPU-54dd2b6f-9311-dd31-672b-60be2ed28a79`; GPU1 is
  `GPU-2b4cf166-fd33-f161-8536-ca04bc72280c`. `nvidia-smi topo -m` reports `PIX` between them, so
  the diagnostic and all resumed scored work continue to require both coordination locks and an
  idle neighbouring card.

## 2026-08-13 — restored-hit oracle verdict

- The focused Q27 8,192 MiB oracle ran under both coordination locks from
  `2026-08-13T01:51:59Z` through `2026-08-13T01:54:37Z`. The cold key-87 baseline and all three
  serial restored hits completed 60/60 tokens with the identical text SHA-256
  `200ec271e8c0eb57fb6b7d42d3ed53e4590c5e72f0303b5ef3c74d363eab88e7`.
- Accounting was exact: 28 retained entries / 8,434,040,832 bytes after 96 inserts, 68 evictions,
  three full 4,860-token hits, zero admission/session/VRAM defers, and zero OOM parks. The server
  failure scan is empty and the raw manifest verifies. This rules out stored-state corruption for
  the focused serial case; it does not assign a lower-level cause to the concurrent 11-token EOS.
- Tenant-clean shutdown passed: both GPUs returned to 0 MiB/P8, the memra process and target ports
  cleared, and both locks were released. A different lane acquired the shared lock only after this
  diagnostic completed; it was observed and left untouched.
- Subsequent orchestrator steering assigns the concurrent EOS reproduction to `cx-eosclass` and
  directs this lane to exclude Q27 repetition 1 / 8,192 MiB, reuse the passing 1,024 and 4,096 MiB
  boots, and score the rest of the grid. The resumed harness must also fail closed on a compute-app
  audit before every scored cell.

## 2026-08-13 — scored resume harness

- `run-resume` enumerates exactly 57 boots in the locked odd/even model and budget order, skipping
  the already-passing Q27 repetition 1 / 1,024 and 4,096 MiB boots and the explicitly excluded
  Q27 repetition 1 / 8,192 MiB EOS boot. That arm will be reported as N=4 plus its excluded
  failure receipt; every other model/budget/concurrency cell remains N=5.
- Every newly emitted sweep row carries physical GPU index, UUID, and name. Before seeding and
  before each scored concurrency cell, the harness records a fail-closed compute-app receipt that
  requires exactly the owned server PID on physical GPU0 and rejects every other process/card.
  An independent 250 ms two-GPU sidecar spans the full resumed lock hold.
- Python compilation/import, `bash -n`, and `shellcheck` pass. The compute-app parser passed both a
  matching-PID control and a foreign-PID negative control on the local development rig. No source
  format pass was run.

## 2026-08-13 — excluded attempt 8 and corrected continuation

- The first continuation misread the EOS steering and tried to replace Q27 repetition 1 / 8,192
  MiB with a passing boot. Its c=4 cell reproduced the same key-87 11-token output and text hash;
  c=5 passed. No new runtime cause was assigned, and no row is eligible for the capacity reduction.
- After verifying process parentage, the lane sent `TERM` only to its owned sweep timeout before
  c=6 completed. The runner recorded exit 143, drained/stopped its server, and released both locks.
  The server failure scan is empty. Across 682 two-GPU sidecar samples, GPU1 stayed at 0 MiB / 0%
  utilization, and every completed pre-seed/pre-cell compute-app audit passed.
- The raw partial is sealed under `raw/attempt8-repeated-eos/`. A later cleanup observation records
  a different lane's `kernel-check` on GPU0 after it acquired the released global lock; the owned
  runner/sweep PIDs, memra server, and target listeners were already absent.

## 2026-08-13 — segmented reducer contract

- The reducer requires 59 valid boots across the recovered and completed continuation segments,
  plus exactly one explicit excluded sweep: Q27 repetition 1 / 8,192 MiB with the captured
  11-token key-87 EOS hash and an empty server failure scan.
- It reports Q27 / 8,192 MiB as N=4 and every other cell as N=5, validates the combined prescribed
  boot order and same-trace budget pairing where an arm exists, and keeps the two lock timings
  separate. It will not label the recovered segment as an uninterrupted completed campaign.
- Every new boot must have one clean pre-seed and one clean per-cell compute-app receipt, physical
  GPU0 identity on every JSONL row, and the completed continuation must have a zero-use GPU1
  sidecar. The two reused pre-steering boots are admitted only by their exact model/budget/rep keys
  and their physical-GPU snapshot UUID.

## 2026-08-13 — scored continuation and documentation checkpoint

- The continuation acquired both coordination locks at `2026-08-13T02:12:27Z` after waiting for
  the prior conforming holder. Q27 repetition 1 / 16,384 MiB cleared the c=4 boundary that exposed
  the excluded 8,192 MiB EOS case; its pre-seed and completed-cell compute-app guards identified
  only the owned server on physical GPU0 while the two-GPU sidecar had kept GPU1 idle through this
  checkpoint.
- `docs/FLAGS.md` and `docs/SERVING.md` now state the measured sold-shape Q27/Q35 entry sizes next
  to the existing Step example. The edit uses the exact byte accounting above, labels binary MiB,
  and does not touch a generated perf-board block.
- Q27 repetition 1 / 16,384 MiB completed PASS at `2026-08-13T02:18:21Z`: 96/96 seeds and
  140/140 scored requests completed at the requested lengths, all eight compute-app guards passed,
  counters reconciled, the server-failure scan was empty, and the immutable boot directory was
  copied into the lane before the next boot completed.
- Source audit confirms that the production cache budget is worker-global across loaded models:
  `(model, namespace)` partitions lookup visibility, but one `PrefixCache.total_bytes` and global
  LRU enforce `MEMRA_PREFIX_CACHE_MB`. The paired Q27+Q35 capacity arithmetic therefore uses one
  shared budget, and the docs now state that ownership explicitly.
- The older six-entry cross-check is reconciled rather than contradicted: it mixed three 4,860-token
  full entries with three 4,374-token partial-exactness entries. Their exact averages are
  293,999,616 bytes for Q27 and 108,709,440 bytes for Q35. The dedicated full-entry probes isolate
  the sold shape and therefore supply the authoritative 301,215,744 / 110,964,480-byte values.
- The requested warning is specified as a boot-time, non-fatal comparison between the shared cache
  budget and the largest loaded model's exact entry geometry at `MEMRA_CTX`. At context 8,192 the
  thresholds are 400,162,816 bytes for Q27 and 141,885,440 bytes for Q35; the existing per-insert
  skip log remains the request-shape backstop. No runtime change is made in this evidence lane.

## 2026-08-13 — repetition 2 stopped on Q27 restored-hit exactness

- The resumed runner was not lost to another process-group sweep. It completed Q27 repetition 2 /
  16,384 MiB, wrote `sweep.exit=1` and a `FAIL` summary, performed its fail-closed cleanup, and
  released both coordination locks. The owned server and sweep processes are absent, ports 18427
  and 18435 are clear, both locks are free, and GPU1 remained at 0 MiB / 0% through the final
  sidecar sample.
- Three working requests were full 4,860-token cache hits but stopped after 11 output tokens:
  prefix 49 at c=4, prefix 95 at c=16, and prefix 88 at c=6. All three have
  `finish_reason=stop` and text hash
  `ddbbcf35ae93821874be64e15f63feecb2471bbe6ea0c23b63fa00b1e98a9b73`, exactly matching the
  earlier excluded Q27 8,192 MiB key-87 failure. The paired positions in the already-passing
  32,768 and 49,152 MiB repetition-2 boots are also full hits, but each completes 60 tokens with
  `finish_reason=length` and hash
  `5790654979cb98bfacf6d3593b6a5d3def7a5f4bd2a1b8b65e4a6fabe1a72f66`.
- The three failed cells each reconcile 20 admitted/completed requests and 1,151 output tokens;
  admission defers, VRAM defers, and OOM parks are zero. Every pre-seed/pre-cell compute-app guard
  passed and the server-failure scan is empty, so no lower-level runtime cause is assigned.
- This new failure invalidates the reducer's assumption that Q27 / 8,192 MiB repetition 1 is the
  only exactness exclusion. Continuing by merely rerunning the 16,384 MiB boot would pass-select a
  nominal N=5 sample. The remaining scored grid is therefore paused at this fail-closed boundary;
  no completed measurement has been repeated or discarded.
- The sibling `lane/cx-eosclass` evidence independently reproduces this exact 11-token hash by
  moving a request from the eager B=1 program to the generic batched program as peers arrive. Its
  same-binary `MEMRA_SERVE_B1FAST=0` control and repaired one-program default produce only the
  normal 60-token hash across the deterministic width gate. The source repair is commit
  `7cd4561a6`; it is not an ancestor of `main` or this lane and is not part of the pinned v0.81.2
  campaign runtime `18885ec4`.
- Nineteen valid boots are sealed across the recovered and resumed segments. A post-fix
  continuation cannot be combined with them: it changes both the floating-point program and the
  measured solo-path performance. The honest choices are to leave the pre-fix campaign incomplete,
  or explicitly authorize a complete restart on one repaired runtime. The latter necessarily
  repeats completed measurements and is not assumed here.
