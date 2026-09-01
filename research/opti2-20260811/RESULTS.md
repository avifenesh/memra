# OPTIPIPE increment 2 — depth-1 controller results

Date: 2026-08-11
Lane: `lane/cx-opti2`
Corrected source head used for the scored box1 block: `80087bbb`
Scored server SHA-256: `33b49fa294b35ab3d1d66ea73c52bb099e11d64c4baa8f045ae4743ee35f0649`
Rig: box1, 2x RTX PRO 6000 Server Edition; every GPU block held
`flock /tmp/memra-gpu.lock`

## Verdict

**NO-GO for promotion. Keep `MEMRA_OPTI_CONTROLLER_Q` absent by default and treat the depth-1
controller as a diagnostic-only research door.**

The corrected controller is exact on the pinned Step target and leaves a real successor stage-0
ticket outstanding while the predecessor's stage 1 resolves. That mechanical result does not
translate into an economic win. At c=2, the best measured controller arm was q*=0.9 at **53.376
tok/s**: **3.70% below serial K=1**, **15.41% below the merged seam**, and **55.93% below plain**.
Every tested q threshold lost to all three controls.

The result is stronger than “the unconditional arm has too many misses.” The online confidence
stub pays one extra shadow draft step at every opportunity merely to obtain its q score. Higher
thresholds select substantially better subsets, but rejected opportunities still pay that tax.
The current gate therefore cannot satisfy the product requirement that low-q traffic remain on
plain generation with zero online shadow cost.

This is an exactness GO for the branch implementation and a throughput **NO-GO**. It is not a
merge, default, tag, release, or perf-board result.

## What landed

- `OptiForkGateMode::Controller` carries one real generation-owned successor ticket: its
  stage-0 boundary, `VerifyCkpt`, verify tokens, q value, draft probability, eager continuation
  seed when needed, optimistic scratch length, and issue timestamp. RAII teardown drains an
  unresolved ticket.
- While round N resolves, the scheduler can prepare and issue exactly one round N+1 stage 0.
  It never issues the successor's stage 1 before N's accept decision. A hit promotes the carried
  boundary; a miss restores stage-local state and reruns the unchanged serial path.
- Validity is exact and device-derived: `n_acc == 1 && bonus == optimistic_pending`. Reconcile is
  driven by the actual accepted count rather than a fabricated forced-harness value.
- Both graph replay and the pinned Step target's eager MTP continuation are supported. The latter
  was required because this artifact intentionally takes the eager draft fallback.
- The q stub is explicitly uncalibrated: it multiplies the first two draft-token top
  probabilities. q*=0 is the unconditional measurement arm. Thresholded policy has a
  three-consecutive-admitted-miss breaker; q*=0 remains unconditional so the safety breaker does
  not truncate its measurement.
- Serving's head-primary PP-2 shape publishes the accept decision before stage 0 peer-reads it
  for conditional reconciliation. The stream record/wait ordering follows CUDA event semantics
  ([runtime event API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html),
  [driver execution API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html)).
- Admission remains fail-closed outside the explicit greedy K=1, device-accept, two-stage PP
  shapes. Recurrent state, pipe/ring/round-stream, replay, sampled or constrained generation,
  host-bounce, and unsupported PP arrangements do not enter the controller.
- The server research door is fresh-process-only, bounded to q in `[0,1]`, and absent by default.
  No runtime default or documented production flag changed.

The pinned Step model exposes no recurrent cache. The implementation refuses recurrent shapes,
so these receipts make no model-backed recurrent-state claim.

## Corrected c=2 performance

The authoritative block used one uninterrupted lock hold from 03:48:02Z through 04:24:44Z. It
cyclically interleaved seven arms for N=5 points per arm. Each point used two warm-up requests,
then eight measured greedy requests at concurrency 2 and 128 maximum generated tokens. All 280
scored requests completed; there were no errors or sheds. Two additional instrumented anatomy
points also completed, giving 296/296 clean measured requests, but those synchronized traces are
excluded from throughput scores.

The remeasured controls were within 0.12% of the frozen floor triple, so the negative controller
result is not explained by a shifted baseline. Source data and exact values are in
[summary.json](raw/box1/perf-c2-3/summary.json); the complete chronological receipt is in
[driver.log](raw/box1/perf-c2-3/driver.log).

| Arm | N | Median tok/s | Min–max tok/s | vs serial | vs seam | vs plain |
|---|---:|---:|---:|---:|---:|---:|
| Plain | 5 | **121.109** | 120.085–121.548 | +118.50% | +91.94% | — |
| Serial K=1 | 5 | **55.427** | 55.258–55.550 | — | -12.16% | -54.23% |
| Merged seam | 5 | **63.098** | 61.894–63.267 | +13.84% | — | -47.90% |
| Controller q*=0.0 | 5 | **40.513** | 39.910–40.699 | -26.91% | -35.79% | -66.55% |
| Controller q*=0.5 | 5 | **51.350** | 50.569–51.558 | -7.36% | -18.62% | -57.60% |
| Controller q*=0.7 | 5 | **52.018** | 51.268–52.303 | -6.15% | -17.56% | -57.05% |
| Controller q*=0.9 | 5 | **53.376** | 52.524–53.568 | **-3.70%** | **-15.41%** | **-55.93%** |

The merged seam still leaves a 47.90% gap to plain. The depth-1 controller closes none of it:
even its highest threshold is slower than serial K=1, which is itself 54.23% behind plain.

## Controller economics

The counts below aggregate the five scored points for each controller arm. “Hit/miss” is validity
among admitted successors. “Shadow tokens” counts every extra draft token spent to label or issue
an opportunity; “wasted tokens” counts rejected probe work plus admitted miss work. The resolution
timer runs from successor issue until N's decision becomes available.

| q* | Checks | Admits (rate) | Hit / miss (rate) | Reconciles | Shadow tokens | Wasted tokens | Resolution median |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 2,970 | 2,970 (100.00%) | 750 / 2,220 (25.25% / 74.75%) | 2,220 | 5,940 | 4,440 | 12.272 ms |
| 0.5 | 2,810 | 620 (22.06%) | 500 / 120 (80.65% / 19.35%) | 120 | 3,430 | 2,430 | 12.313 ms |
| 0.7 | 2,770 | 370 (13.36%) | 290 / 80 (78.38% / 21.62%) | 80 | 3,140 | 2,560 | 12.323 ms |
| 0.9 | 2,690 | 170 (6.32%) | 130 / 40 (76.47% / 23.53%) | 40 | 2,860 | 2,600 | 12.283 ms |

No scored arm tripped the breaker or drained a tail ticket. The different threshold arms do not
have identical check counts because admission changes the generated controller history. Their
admitted hit rates therefore must not be read as four thresholds over one fixed trajectory.

The unconditional q*=0 arm measured the controller's actual validity label at **25.25%**, not the
38.1% aggregate K=2 acceptance used in the pre-build design model. Those figures come from
different retained trajectories and are not interchangeable.

### Fixed-trajectory ranking diagnostic

For a clean ranking check, the summarizer applies every threshold counterfactually to q*=0's
fixed 2,970-observation trajectory. The q proxy spans 0.003583–0.988212 with median 0.187672.

| Counterfactual threshold | Selected | Hits | Selected hit rate |
|---:|---:|---:|---:|
| 0.0 | 2,970 | 750 | 25.25% |
| 0.5 | 580 | 460 | 79.31% |
| 0.7 | 370 | 330 | 89.19% |
| 0.9 | 90 | 90 | 100.00% |

This table is **not throughput** and does not replace the independently run q arms. In particular,
the actual q*=0.9 arm admitted 170 successors with a 76.47% hit rate because its execution history
diverged from q*=0. The fixed trajectory only shows that the proxy has ranking signal worth
studying; it does not show calibration, a stable production segment, or positive expected value.

### Excluded phase anatomy

`MEMRA_SPEC_PP_ANATOMY=1` synchronizes diagnostic boundaries, so these two points were declared
non-scoring. Their weighted per-round timings localize cost but are not uninstrumented steady-state
intervals.

| Trace | Rounds | Draft | PP verify | Verify/accept | Commit/rollback | Other |
|---|---:|---:|---:|---:|---:|---:|
| q*=0.0 | 650 | 0.559 ms | 30.702 ms | 0.025 ms | 0.135 ms | 0.077 ms |
| q*=0.7 | 608 | 0.657 ms | 23.351 ms | 0.025 ms | 0.134 ms | 0.049 ms |

The traces reinforce the end-to-end result: rollback bookkeeping is small, but eliminating
reconciles does not repay the always-on shadow-draft and speculative-verify schedule on this
workload.

## Schedule correction and retained negative control

The first complete N=5 block, [`perf-c2-1`](raw/box1/perf-c2-1/summary.json), is deliberately
retained but excluded from the verdict. Its successor was enqueued after N stage 1, so tickets
resolved around 0.08 ms after issue. It measured the same controller policy without the requested
stage overlap and is only a negative schedule control.

Reordering issue before N stage 1 expanded issue-to-decision time to about 12.3 ms, proving that
the successor ticket now remains outstanding across predecessor resolution. That correction also
exposed a real numeric bug: the Step35 row verifier derived RoPE positions from mutable
`cache.pos`, so the early N+1 stage-0 mutation made N stage 1 use the wrong row. The state
comparator caught the first layer-0 K-cache byte mismatch in
[`debug-hit-n8-1`](raw/box1/debug-hit-n8-1/driver.log).

The fix threads immutable `pos0` through every verify-layer entry point and uses `pos0 + row` for
Step35. The reduced miss-plus-hit reproduction then passed full state and continuation identity
with 12.2 ms ticket lifetimes in
[`debug-hit-n8-2`](raw/box1/debug-hit-n8-2/driver.log). The full battery and the entire N=5 block
were rebuilt and rerun only after that proof.

`perf-c2-2` contains no performance point. It never acquired the GPU lock and its terminated
queue-only waiter retained the exact lines `Terminated` and `FAIL: GPU lock timeout`; it is not a
scored runtime failure.

## Exactness gates on corrected code

| Gate | Result | Retained evidence |
|---|---|---|
| Real-prompt hit/miss state | **PASS** — N=128, 80 opportunities, 20 hits, 60 misses/reconciles, zero refusals/drains; output, every live cache/scratch/hidden byte, and a 17-token controller-off continuation exact | [driver.log](raw/box1/final-hit-state-4/driver.log) |
| `kernel-check` | **ALL GREEN** — 376 `OK` cells | [kernel-check.log](raw/box1/exact-gates-3/kernel-check.log) |
| `run-gen` PP-2 | **MATCH** — both required argmax comparisons | [run-gen.log](raw/box1/exact-gates-3/run-gen.log) |
| `run-spec` | **PASS K=1..8** self-consistency | [run-spec.log](raw/box1/exact-gates-3/run-spec.log) |
| Controller q sweep identity | **PASS** at q*=0.0, 0.5, 0.7, 0.9; q*=0 exercised 30 real misses/reconciles at the corrected schedule | [driver.log](raw/box1/exact-gates-3/driver.log) |
| Serial fresh processes | **PASS 10/10** — every 326-byte completion SHA-256 `21b8293f2298978c74fb89f32d9b14e3ea921f39924cfce88c73b01f445bb6de` | [driver.log](raw/box1/serial-boots-2/driver.log), [hashes.txt](raw/box1/serial-boots-2/hashes.txt) |
| Local engine suite | **59 passed, 0 failed, 1 CUDA-only ignored** | [test-engine-lib.log](raw/local/final-checks-pos0-2/test-engine-lib.log) |
| Local server suite | **162 passed, 0 failed** | [test-server.log](raw/local/final-checks-pos0-2/test-server.log) |

The full real-prompt gate's ticket resolutions span 12.08–12.34 ms. Both hit retention and miss
reconciliation occur in one run, which is stronger than separate forced terminal tests. Every
final GPU receipt ends with both devices at 0 MiB and no compute process.

One earlier local wrapper, `final-checks-pos0-1`, requested a nonexistent `memra-server --lib`
target and lacked fail-fast behavior. Its trailing `PASS` is invalid and is not cited above; the
corrected `final-checks-pos0-2` receipt is authoritative.

## Evidence integrity and thermal regime

- The scored block retained combined stdout/stderr before parsing and printed `PERF_PASS` only
  after all 37 load points, accounting checks, failure-signature checks, and GPU teardown passed.
- The 193-file remote payload and local copy have the same aggregate tree SHA-256
  `214352a6f1224c08eb27710f4babcddee3dc25fe9ed14dcbf2709809403eb4f0`.
  Locally generated `summary.json` is file 194 and has SHA-256
  `c9b77baa4a058e7510de105b68e40170cc614b34df86496a75b2e96d385322ca`.
- Across 140 scored thermal snapshots, temperature was 30–37 C, SM clock 180–2415 MHz, and
  power 31.44–98.84 W. The block began with both GPUs at 0 MiB, 30 C, P8, 180 MHz and ended at
  0 MiB, 35/36 C, P0, 2400/2317 MHz.
- No captured authoritative log contains a run failure, request error, or shed. Instrumented
  traces and invalid controls are named and excluded rather than silently mixed into medians.

## What would change the verdict

The next defensible experiment is not another threshold sweep over this online product proxy. It
is a trained and calibrated prefix-survival selector, following the scheduling idea rather than
the drafter architecture in [DSpark](https://arxiv.org/html/2607.05147), with these hard gates:

1. Train and calibrate from non-public retained `v_N` traces, keyed by exact target, draft,
   prompt-template/workload, runtime, and rig hashes. Freeze the model and thresholds before the
   scored evaluation.
2. Make the low-confidence production path byte- and performance-equivalent to plain generation.
   It must not execute the current extra d2 shadow step merely to decide that it should stay off.
3. Identify a retained traffic segment whose conservative lower confidence bound clears the
   re-solved economic threshold. The 90/90 q*=0 counterfactual tail is a small ranking observation,
   not that proof.
4. Rerun interleaved N=5 auto-vs-plain on that retained segment and on low-q traffic. Promotion
   requires a guarded win over plain on the admitted segment and zero regression on the plain
   path, followed by the full 2x PRO 6000 pre-release battery.

Until those conditions hold, the correct production policy is plain generation. The controller
door remains useful for collecting offline labels and testing scheduler mechanics, but its current
runtime behavior should not be enabled by default.

No push, merge, tag, release, perf-board edit, `cargo fmt`, `rustup`, or `nsys` was performed.
