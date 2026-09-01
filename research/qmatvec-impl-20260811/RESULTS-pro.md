# IQ4_XS qmatvec arm-1 — PRO promotion verdict

Date: 2026-08-11

Candidate: `4f15557c7a24d396966a9b3b34c970518e5d81ce`

Baseline: `49f5002d7a37291c9b551ac2f683ce2edb27d163`

Rig: rented box1, 2x NVIDIA RTX PRO 6000 Blackwell Server Edition; the timed and
profiled harness ran on GPU0 under one exclusive GPU-lock window.

## Verdict

**PROMOTION GO — the arm cuts the unprofiled Step semantic-mix time by 6.638%
on PRO silicon, with exactness frozen.** The candidate is byte-identical to both
the PRO baseline and the RTX 5090 reference dump, and the candidate
`kernel-check` is ALL GREEN. The counter movement supports the intended
wide-load mechanism, but the interleaved wall-time result—not replayed NCU
duration—is the performance verdict.

## Bit-identity and correctness

| Gate | Baseline | Candidate / reference | Result |
|---|---|---|---|
| PRO output dump SHA-256 | `e52f8369ce62dd250f4d203033bc5e8f6e6c1c9ba4e262ce84565d22eb92accf` | `e52f8369ce62dd250f4d203033bc5e8f6e6c1c9ba4e262ce84565d22eb92accf` | byte-identical; `cmp` exit 0 |
| Cross-rig dump | candidate PRO hash above | RTX 5090 reference `e52f8369ce62dd250f4d203033bc5e8f6e6c1c9ba4e262ce84565d22eb92accf` | exact match |
| Candidate `kernel-check` | — | 77 cells passed, 21 skipped | **ALL GREEN** |

The hash receipt is
[`raw/pro-battery/bitid.txt`](raw/pro-battery/bitid.txt); the complete candidate
battery is
[`raw/pro-battery/kernel-check.log`](raw/pro-battery/kernel-check.log).

## Step-mix timing

The harness replays the fixed 315-launch Step semantic mix: 2,872,946,688
weight bytes and 2,879,039,040 logical bytes per synthetic token sweep. Each arm
ran eight 256-repetition samples in an ABBA-interleaved order, in one window on
GPU0. Clocks were stock and unlocked. Battery-boundary telemetry was 28 C / 180
MHz at idle entry and 37 C / 2,392 MHz at exit; the clock snapshots bracket the
whole battery rather than each timed arm.

| Arm | N / order | Timed-row GPU0 thermal | Clock regime | Median time | Median logical throughput | Change vs baseline |
|---|---|---|---|---:|---:|---:|
| Baseline | 8 / ABBA interleaved | 40–45 C | stock, unlocked | 2.863324 ms | 1,005.5 GB/s | — |
| Candidate | 8 / ABBA interleaved | 37–43 C | stock, unlocked | 2.673256 ms | 1,077.0 GB/s | **-6.638% time / +7.110% throughput** |

The logical-throughput value divides the fixed logical byte bill by CUDA-event
time; it is not a DRAM-counter measurement. Raw rows and the reduction are
[`timing-rows.csv`](raw/pro-battery/timing-rows.csv) and
[`timing-summary.txt`](raw/pro-battery/timing-summary.txt); boundary telemetry is
[`thermal-start.csv`](raw/pro-battery/thermal-start.csv) and
[`thermal-end.csv`](raw/pro-battery/thermal-end.csv).

## NCU mechanism receipt

Means below cover four profiled `qmatvec_iq4_XS_dp4a` launches per arm. NCU
replay is mechanism evidence only; its instrumented duration is not a throughput
measurement.

| Metric | Baseline | Candidate | Movement |
|---|---:|---:|---:|
| `lg_throttle` stalls / issue-active | 2.900 | 0.475 | **0.164x (-83.6%)** |
| `long_scoreboard` stalls / issue-active | 12.26 | 27.39 | **2.23x (+123.4%)** |
| L1TEX throughput, % of peak sustained elapsed | 61.3875% | 38.7725% | **0.632x (-36.8%)** |

The named target moved in the intended direction: consolidating the scalar
unpack path into wider loads sharply reduced pressure on the local/global load
instruction queue, and L1TEX pipe utilization also fell. `long_scoreboard` did
**not** fall—it more than doubled. That is consistent with the remaining stall
mix migrating toward exposed memory-dependency latency once unpack/issue work
shrinks: the same model-byte work now spends proportionally more issue-active
cycles waiting for L1TEX results. It is not evidence that the arm is slower.
The unprofiled, ABBA-interleaved Step-mix wall time fell 6.638%, so wall time is
the promotion authority while the counters explain where the stalls moved.

Raw counter reduction:
[`ncu-summary.txt`](raw/pro-battery/ncu-summary.txt). Raw exports:
[`ncu-baseline.csv`](raw/pro-battery/ncu-baseline.csv) and
[`ncu-candidate.csv`](raw/pro-battery/ncu-candidate.csv).

## Scope boundary

This is a PRO-silicon promotion of qmatvec arm 1. It does not move an end-to-end
decode or serving cell, and it does not move the tracked RTX 5090 or H100 boards.
