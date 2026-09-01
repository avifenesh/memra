# Q27 clean-throughput knee — cache + prefill

Date: 2026-08-12

## Verdict

The frozen single-card Q27 mixed90 knee is reproduced at **c=12**. The first
decline is not a VRAM, admission, or prefix-cache-capacity failure. It is a
scheduler shape:

1. At c=16, both cold misses are in the first admitted wave in all five
   repetitions. The sole CUDA worker executes two synchronous 1,024-token
   prefill calls before each decode phase, so cache-hit streams sit behind
   roughly 580 ms prefill phases.
2. Above the exact-16 decode tier, c=20 is split 16+4. The four-row tail is
   inefficient; c=24 rebounds because its 16+8 partition is better filled.

The config-first A/B did **not** move the knee. Explicit
`MEMRA_PREFILL_TICK=2048` versus the naked 1,024 default was interleaved for
five whole-server rounds. Every baseline and candidate boot still had knee
c=12; the candidate's paired median throughput delta was **-0.13% at c=16**
and it materially worsened cache-hit TTFT. There is no winner, so this lane
proposes **no default flip and no sold-cap change**.

The clean c=24 rebound is real, but the frozen definition stops at the first
non-rise and the latency envelope is worse there. Any sellable-concurrency
decision remains an owner call.

## Frozen result

All values are N=5 medians on one RTX PRO 6000 Blackwell Server Edition. The
characterization used one boot and one continuous GPU-lock hold. The A/B used
five alternating whole-server pairs under one continuous lock hold; each arm
has one observation per repetition and width.

| concurrency | characterization baseline tok/s | interleaved baseline tok/s | prefill-2048 tok/s | paired candidate delta |
|---:|---:|---:|---:|---:|
| 8  | 174.802 | 174.984 | 175.091 | +0.010% |
| 12 | **186.216** | **186.048** | **186.261** | -0.068% |
| 16 | 182.998 | 182.883 | 182.645 | **-0.130%** |
| 20 | 181.685 | 181.422 | 181.761 | +0.175% |
| 24 | 186.445 | 186.189 | 186.414 | +0.040% |

The baseline path is c8 -> c12 **+6.53%**, then c12 -> c16 **-1.73%**,
which fixes the formal knee at c=12. The prefill-2048 path falls **1.94%**
from c12 to c16, so its formal knee is also c=12.

Every one of the 75 scored cells across characterization and A/B was clean:

| block | cells | requests OK | output tokens reconciled | expected hits / misses | cached/prompt/hit-token drift | admission / VRAM defers | OOM parks |
|---|---:|---:|---:|---:|---:|---:|---:|
| characterization baseline | 25/25 | 550/550 | 33,000 | 495 / 55 | 0 / 0 / 0 | 0 / 0 | 0 |
| A/B baseline | 25/25 | 550/550 | 33,000 | 495 / 55 | 0 / 0 / 0 | 0 / 0 | 0 |
| A/B prefill-2048 | 25/25 | 550/550 | 33,000 | 495 / 55 | 0 / 0 / 0 | 0 / 0 | 0 |

## What binds

### The formal c12 -> c16 drop is prefill head-of-line work

The frozen role shuffle puts this many cold misses into the first client wave:

| concurrency | first-wave cold misses, repetitions 1..5 |
|---:|---|
| 8  | 0, 2, 0, 1, 0 |
| 12 | **1, 2, 1, 1, 2** |
| 16 | **2, 2, 2, 2, 2** |
| 20 | 2, 2, 2, 2, 2 |
| 24 | 2, 2, 3, 2, 3 |

The diagnostic trace is a single run, used for mechanism attribution rather
than throughput promotion. It captured 12 scheduler iterations with two
single-session prefill calls, 2,048 tokens total, **zero batch calls**, and a
580.2 ms median prefill phase (562.3-600.5 ms). The corresponding one-call,
1,024-token phase was 290.45 ms median. That is the named saturation point:
interactive prefill loops over active sessions synchronously on the sole CUDA
worker before the decode phase.

The latency response matches the mechanism. In the interleaved A/B baseline,
cache-hit TTFT p95 medians were 311.5 ms at c12 and 596.1 ms at c16. Raising the
quantum did not create concurrent prefill compute: prefill-2048 measured 603.8
ms and 1,194.4 ms respectively. Paired median hit-TTFT-p95 regressions were
**+93.7% at c12** and **+265.6% at c16**. Maximum observed scheduler tick also
rose from 1,523 ms to 1,846 ms while throughput remained flat.

### B=16 itself is not the c16 throughput ceiling

Steady decode-only trace rows show B=16 is the most efficient measured wave:

| ready rows | median decode phase | row throughput |
|---:|---:|---:|
| 8  | 23.7 ms | 337.6 tok/s |
| 12 | 35.0 ms | 342.9 tok/s |
| 16 | **43.7 ms** | **366.1 tok/s** |
| 20 | 61.5 ms | 325.2 tok/s |
| 24 | 67.4 ms | 356.1 tok/s |

The server explicitly logged `decode wave cap 16; scheduler tick cap 16
(exact-16 tier)`. The c20 phase is numerically the serial 16+4 partition:
43.7 + 17.7 = 61.4 ms versus 61.5 ms observed. The c24 phase is the 16+8
partition exactly: 43.7 + 23.7 = 67.4 ms. This explains the c20 valley and
c24 rebound without calling either an OOM or a hard card-capacity collapse.

### Cache size, admission, and gross VRAM are excluded

At the end of the continuous baseline, the prefix cache held 4,217,020,416
bytes (3.93 GiB) in 14 entries and the CUDA driver still reported
68,033,249,280 bytes (63.36 GiB) free. The 49 lifetime evictions did not turn
expected hits into misses: all 495/495 expected full hits landed, with exact
cached-token and prefix-hit-token reconciliation. `MEMRA_MAX_SESSIONS=96` was
well above the tested widths, and the runtime reported zero session defers,
zero VRAM defers, and zero OOM parks.

That evidence removes cache MB, max sessions, and admission caps as useful
knee-moving A/Bs on this workload. A fixed narrower decode cap also cannot fix
both relevant widths: cap 12 would turn c20 into the better-filled 12+8 shape
but would simultaneously turn efficient c16 into 12+4; widths above 16 have no
exact kernel class. No such arm was promoted to a scored experiment.

## Config A/B verdict

The one changed setting was `MEMRA_PREFILL_TICK`: unset (1,024 concurrent
interactive default) versus explicit 2,048. Prefix cache MB, max sessions,
decode cap, model bytes, runtime bytes, workload, and all other serve settings
were identical. Arm order was baseline/candidate in odd rounds and reversed in
even rounds. Width order followed the frozen global repetition rotation.

The candidate is **rejected for this serving shape**:

- all five baseline knees: c12;
- all five candidate knees: c12;
- c16 paired throughput deltas: +0.080%, -0.130%, -0.528%, -0.230%, -0.004%;
- c12 paired median throughput: -0.068%;
- c16 paired median throughput: -0.130%;
- c12/c16 hit-TTFT p95 materially regressed;
- max scheduler tick grew 21.2%.

The trace also proves why a code tweak was not appropriate in this lane. A
balanced decode-tail scheduler would improve c20 but not the formal c12 -> c16
drop. The remaining prefill mechanisms are continuation-capable cross-request
batching or a total-prefill-work budget across sessions; those are broader
correctness/QoS changes, not a bounded config-ceiling patch justified by this
negative A/B. They belong in a separately gated follow-up, if prioritized.

## Sellable-concurrency and gross $/day implication

Using the stated published prices ($0.287/M input tokens and $2.751/M output
tokens), and assuming the frozen 4,860:60-token request mix is continuously
saturated and billed at those list prices, the characterization c12 median
corresponds to:

- 186.216 output tok/s = 3.1036 requests/s;
- 15,083.5 billed input tok/s;
- about **$374.02 input + $44.26 output = $418.28 gross usage/day/card**.

This is list-price gross usage math, not demand, margin, or a recommendation.
The tested change produced **zero knee-concurrency gain**, so its
evidence-backed incremental sellable-capacity implication is **$0/day**.
Headroom over the sold cap of 4 stays **200%** (c12), before and after. The
unpaired candidate median at c12 would arithmetically differ by less than
$0.50/day, but the paired throughput verdict is flat/negative and hit latency
regresses, so that is not a revenue win.

## Protocol and provenance

- Runtime source: `b671c3e17035d757944439a5345b66d2f442ebe5`, staged as a clean,
  lane-owned checkout and freshly built as memra v0.79 with CUDA 13.2 / sm_120a.
- Runtime SHA-256: `fa41dcf08d09a2107f769b1fd2696997fc9ce6cd26870ee7d21ae3545833f6f3`.
- Model: `Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf`, SHA-256
  `d8d71c7e8a01a1c964fd904a7b496eaef19bdd66827e0949e66c723da742d517`.
- Frozen workload lock SHA-256:
  `85597a0a28ed874f440b4a966c0b43fd3e31b94fe868266de9e299decc208c34`;
  canonical prompt-id SHA-256:
  `eba9dff66d4ebf3f40d6db80298ab9c884fa1bb2e1a6d4ca2c7dadd4f21513fb`.
- Shape: physical GPU 0 only, `CUDA_VISIBLE_DEVICES=0`; GPU 1 remained idle.
  Spec off, 4,096 MB prefix cache, prefix dedup on, reuse pool off, affinity off,
  max sessions 96, 60 output tokens, 90% full-prefix hits.
- Characterization: one boot, N=5/width, no cooldown; sampled 27-60 C, max
  496.27 W, max 2,422 MHz, 100% median NVML GPU-busy over the captured run.
- Diagnostic: one boot, N=1/width for c12/16/20/24, tick trace on; not a scored
  throughput comparison.
- Config A/B lock hold: 2026-08-12 12:11:59Z-12:29:56Z. Five alternating
  whole-server pairs, no artificial cooldown. Baseline sampled 26-59 C;
  candidate 34-59 C. Both reached 2,422 MHz and 100% median NVML GPU-busy.
- NVML utilization fields are interpreted per the current
  [NVIDIA nvidia-smi documentation](https://docs.nvidia.com/deploy/nvidia-smi/index.html):
  they are sampled activity percentages, not direct FLOP occupancy.

## Receipt index

- Machine-readable reduction: `analysis.json` (generated by `analyze.py`).
- Baseline: `raw/baseline/`; manifest-file SHA-256
  `bdbd1fe44b24757d5aabd7405f4ef12f0f753d253807b7c2b67e12ec5faca3ec`.
- Tick diagnostic: `raw/diagnostic-trace/`; manifest-file SHA-256
  `889b7f67d53aa30fca3be15776766ab5df77ae3e8fe7a13c6db883b27cdbc816`.
- Interleaved A/B: `raw/ab-prefill2048/`; outer manifest-file SHA-256
  `314df482a18f6d04173a89f5087eebb2616aacf2b29a69d40372bc56e3b6aaf9`.
- Fresh build: `raw/build/`.

No run used `nsys`; no source default, board, tag, merge, or remote branch was
changed.
