# MoESD target-efficiency study — box1

Date: 2026-08-11

Measured source: `edbf6827d2c6993b15301c966a898a419aebfd40`

Rig: hyperscaler sbox-2card `<private-host-redacted>`, 2x NVIDIA RTX PRO 6000 Blackwell
Server Edition, stock 600 W limits

## Verdict: NO-GO (CLOSED)

At the frozen pivot `(B=8, gamma=4)`, the target forward took **169.232 ms** versus
**42.801 ms** for `gamma=1`. The packed four-column forward therefore cost **98.849%**
of four serial target steps. The owner-approved gate requires at most 66.667%, expressed as
`serial_amortization > X=1.5`; the measured value was only **1.0116**.

The frozen acceptance proxy projects **52.38 tok/s** at the pivot, only **30.17%** of the
frozen plain-B8 threshold of **173.62 tok/s**. Both decision gates fail. K=0 remains the
correct serving policy, the next dollar goes to DSpark acceptance-only, and this result does
**not** reopen the closed PP-2 speculative-decode verdicts #87/#94.

| decision input | frozen requirement | N=5 result | gate |
|---|---:|---:|---|
| `B` | 8 | 8 | fixed |
| `gamma` | 4 | 4 | fixed |
| `T_T(B,1)` | input | 42.801 ms | measured |
| `T_T(B,4)` | <=114.135 ms for X=1.5 | 169.232 ms | **FAIL** |
| paper target efficiency `T1/T4` | reported, not compared with X | 0.2529 | diagnostic |
| serial amortization `4*T1/T4` | >1.5 | 1.0116 | **FAIL** |
| realistic throughput | >173.62 tok/s | 52.38 tok/s | **FAIL** |
| final decision | both gates pass | NO-GO / CLOSED | **FAIL** |

## Metric normalization

[MoESD v4 §3.1](https://arxiv.org/html/2505.19645v4) defines target efficiency as
`T_T(B,1) / T_T(B,gamma)`. The DESIGN then approves `X=1.5` while describing the gate as
“the packed columns cost at most 67% of gamma serial steps.” Those are different
normalizations, so the reducer retains both without changing the frozen decision:

- `target_eff = T1/Tgamma` is the paper's literal metric.
- `serial_amortization = gamma*T1/Tgamma` is the owner decision input.
- Equivalently, the X gate is `Tgamma <= gamma*T1/X`.

Applying X directly to the paper ratio would not represent the stated serial-cost comparison.
Every raw and summary row therefore exposes both names.

## What the sweep says

The pivot is not an expert-union saturation point. Across the 42 MoE layers, the median layer
activated 26.5 distinct experts at `(8,1)`, 63.5 at `(8,4)`, and 86.0 at `(8,8)` out of 288.
At the pivot the across-layer range was 47–76 experts and the mean was 63.05; every layer's
union was deterministic across all five repetitions. Moving from gamma 4 to 8 still increased
the mean union by 35.4%, so the extra columns were not close to free.

There is also a repeatable total-row latency cliff at `B*gamma=16`: `(2,8)`, `(4,4)`,
`(8,2)`, and `(16,1)` all take about 127–129 ms, while the adjacent 12-row cells take about
60 ms. This study records the cliff but does not attribute it to a specific kernel.

Compute amortization does reappear at other widths: gamma 4 reaches 2.059 at B=16, 1.806 at
B=24, and 1.652 at B=32. That does not change the frozen B=8 decision, and the acceptance gate
still dominates: the best `gamma>1` cell in the entire sweep is only 98.86 realistic tok/s
at `(32,3)`. The target-only projection also omits draft and rejection overhead, so it is an
optimistic upper bound rather than an end-to-end serving claim.

## Full N=5 matrix

`ms` is the median with the five-run range in brackets. `paper eff` is `T1/Tgamma`;
`serial amort` is `gamma*T1/Tgamma`. `effective` accepts every packed row, while `realistic`
uses the frozen per-position Step-3.7 acceptance sums. `union p50 [min,max]` takes the median
of the 42 per-layer median unions, followed by the across-layer range. Per-run per-layer unions
remain in `raw/box1/raw.jsonl`.

| B | gamma | ms, median [range] | paper eff | serial amort | effective tok/s | realistic tok/s | union p50 [min,max] |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 9.273 [9.252, 9.302] | 1.000 | 1.000 | 107.84 | 107.84 | 8.0 [8, 8] |
| 1 | 2 | 14.051 [13.890, 14.087] | 0.660 | 1.320 | 142.34 | 52.45 | 14.0 [11, 16] |
| 1 | 3 | 18.601 [18.438, 18.644] | 0.499 | 1.496 | 161.28 | 56.07 | 17.0 [13, 22] |
| 1 | 4 | 23.214 [22.925, 23.276] | 0.399 | 1.598 | 172.31 | 47.73 | 19.0 [13, 25] |
| 1 | 6 | 32.371 [32.043, 32.599] | 0.286 | 1.719 | 185.35 | 37.19 | 27.0 [17, 36] |
| 1 | 8 | 41.972 [41.522, 41.991] | 0.221 | 1.767 | 190.61 | 30.97 | 31.5 [19, 42] |
| 2 | 1 | 14.178 [13.977, 14.208] | 1.000 | 1.000 | 141.07 | 141.07 | 13.0 [9, 16] |
| 2 | 2 | 23.325 [23.009, 23.467] | 0.608 | 1.216 | 171.49 | 63.19 | 23.0 [15, 28] |
| 2 | 3 | 32.331 [32.161, 32.769] | 0.439 | 1.316 | 185.58 | 64.52 | 28.5 [17, 34] |
| 2 | 4 | 41.969 [41.404, 42.033] | 0.338 | 1.351 | 190.62 | 52.80 | 31.0 [19, 40] |
| 2 | 6 | 60.185 [59.744, 60.254] | 0.236 | 1.413 | 199.38 | 40.01 | 39.0 [26, 53] |
| 2 | 8 | 127.311 [127.304, 127.379] | 0.111 | 0.891 | 125.68 | 20.42 | 45.5 [30, 64] |
| 4 | 1 | 23.723 [23.399, 23.759] | 1.000 | 1.000 | 168.61 | 168.61 | 20.0 [16, 25] |
| 4 | 2 | 42.070 [42.047, 42.326] | 0.564 | 1.128 | 190.16 | 70.07 | 33.0 [25, 39] |
| 4 | 3 | 60.654 [60.098, 60.673] | 0.391 | 1.173 | 197.84 | 68.78 | 42.0 [31, 51] |
| 4 | 4 | 127.442 [127.363, 127.528] | 0.186 | 0.745 | 125.55 | 34.78 | 46.5 [35, 56] |
| 4 | 6 | 148.705 [148.609, 148.904] | 0.160 | 0.957 | 161.39 | 32.39 | 59.0 [44, 72] |
| 4 | 8 | 168.629 [168.550, 168.715] | 0.141 | 1.125 | 189.77 | 30.84 | 65.0 [49, 84] |
| 8 | 1 | 42.801 [42.590, 42.843] | 1.000 | 1.000 | 186.91 | 186.91 | 26.5 [19, 36] |
| 8 | 2 | 128.083 [127.926, 128.213] | 0.334 | 0.668 | 124.92 | 46.03 | 43.0 [33, 54] |
| 8 | 3 | 149.678 [149.533, 149.819] | 0.286 | 0.858 | 160.34 | 55.75 | 54.0 [40, 71] |
| 8 | 4 | 169.232 [168.431, 169.495] | 0.253 | 1.012 | 189.09 | 52.38 | 63.5 [47, 76] |
| 8 | 6 | 210.123 [208.588, 210.796] | 0.204 | 1.222 | 228.44 | 45.84 | 78.5 [58, 94] |
| 8 | 8 | 248.376 [248.072, 250.159] | 0.172 | 1.379 | 257.67 | 41.87 | 86.0 [64, 109] |
| 16 | 1 | 128.796 [128.742, 128.907] | 1.000 | 1.000 | 124.23 | 124.23 | 34.0 [23, 44] |
| 16 | 2 | 170.766 [169.993, 170.959] | 0.754 | 1.508 | 187.39 | 69.05 | 56.0 [39, 67] |
| 16 | 3 | 211.803 [210.579, 212.506] | 0.608 | 1.824 | 226.63 | 78.79 | 64.0 [46, 82] |
| 16 | 4 | 250.266 [249.429, 251.828] | 0.515 | 2.059 | 255.73 | 70.84 | 76.0 [51, 93] |
| 16 | 6 | 335.288 [333.207, 337.770] | 0.384 | 2.305 | 286.32 | 57.46 | 95.0 [65, 112] |
| 16 | 8 | 414.156 [411.223, 417.023] | 0.311 | 2.488 | 309.06 | 50.22 | 104.5 [73, 132] |
| 24 | 1 | 151.163 [151.110, 151.337] | 1.000 | 1.000 | 158.77 | 158.77 | 42.0 [27, 53] |
| 24 | 2 | 213.626 [212.421, 213.713] | 0.708 | 1.415 | 224.69 | 82.80 | 67.0 [47, 79] |
| 24 | 3 | 276.836 [276.288, 278.949] | 0.546 | 1.638 | 260.08 | 90.42 | 77.5 [60, 97] |
| 24 | 4 | 334.711 [334.201, 338.148] | 0.452 | 1.806 | 286.81 | 79.45 | 88.5 [66, 112] |
| 24 | 6 | 457.336 [456.054, 460.392] | 0.331 | 1.983 | 314.87 | 63.18 | 114.0 [81, 137] |
| 24 | 8 | 579.100 [575.937, 581.873] | 0.261 | 2.088 | 331.55 | 53.88 | 126.0 [92, 151] |
| 32 | 1 | 171.611 [170.683, 172.114] | 1.000 | 1.000 | 186.47 | 186.47 | 46.0 [30, 62] |
| 32 | 2 | 254.031 [252.720, 254.835] | 0.676 | 1.351 | 251.94 | 92.84 | 78.5 [56, 91] |
| 32 | 3 | 337.623 [336.358, 340.276] | 0.508 | 1.525 | 284.34 | 98.86 | 91.0 [66, 108] |
| 32 | 4 | 415.529 [414.671, 418.922] | 0.413 | 1.652 | 308.04 | 85.33 | 103.0 [71, 132] |
| 32 | 6 | 577.697 [572.717, 581.523] | 0.297 | 1.782 | 332.35 | 66.69 | 125.5 [88, 155] |
| 32 | 8 | 740.555 [736.717, 747.129] | 0.232 | 1.854 | 345.69 | 56.17 | 137.5 [99, 167] |

## Method and provenance

- One release process loaded the three-part Step-3.7-Flash IQ4_XS target plus the pinned external
  MTP Q8_0 artifact, primed 32 deterministic depth-128 sessions, froze greedy continuations, and
  measured all 42 cells five times. Odd runs were ascending and even runs descending; one boot
  warmup was excluded.
- The timed region contains the target forward and CUDA completion only. Expert-union telemetry
  replays identical rows after cache restore, so its router D2H synchronization is not charged to
  `T_T`. The serving wrapper still uses identity row-to-cache mapping and the diagnostic capture
  remains disabled by default.
- The harness proved B1/gamma1 logits bit-identical to the existing decode path and B1/gamma8
  packed-causal argmax identical with a reported maximum logits delta of exactly zero before
  emitting any row.
- `raw/box1/measurements.jsonl` and `raw/box1/raw.jsonl` each contain exactly 210 rows;
  `RESULTS.jsonl` contains 42 N=5 summaries plus one decision row. The reducer validates the
  complete Cartesian set and the required alternating order before writing output.
- The matrix ran from 17:48:47.637Z to 17:50:07.295Z. The 500 ms sampler retained 242 samples;
  row-associated temperatures were 36–48 °C and sampled power peaked at 289.66 W. This is a
  stock-clock cold-to-warm alternating regime, not a fixed-clock or thermal-steady-state claim.
  The lock-acquired snapshot was 27 °C/P8/0 MiB, and the final snapshot was 40–41 °C/P0/0 MiB.
- Source and toolchain are recorded in `raw/box1/manifest.txt`. `raw/box1/SHA256SUMS` pins all
  three target shards, the MTP artifact, five release binaries, and all three study scripts. The
  staged Git bundle SHA-256 was `42cb57664708b02161c2e1975fc261cac48409d9ca1f0a386dbbcfb4cd5bca82`.

## Exactness and release state

The same exclusive lock hold continued through the required battery:

- `kernel-check`: `ALL GREEN (83 cells, 21 skipped)`, with 380 `OK` rows.
- `run-gen`: prefill/decode and batched-prime/tokenwise argmax both `MATCH`.
- `run-spec`: K=1..8 self-consistency passed, 8/8.
- `decode-batch-gate --mode pp`: B=1,2,4,8 split and unsplit references were bit-identical;
  13 gate arms passed and the final PP-2 verdict was `ALL GREEN`.
- Final compute-apps were empty and both GPUs were at 0 MiB before `MOESD_PASS` released the
  lock.

The scored receipt is `raw/box1/`. `raw/attempt1-build-stopped/` contains the conservatively
stopped build-only queue handoff and no measurement rows. `raw/attempt2-incomplete-battery/`
contains a complete exploratory matrix and green kernel check, but is excluded because the old
summary-string assertion ended the driver before the remaining exactness gates. No failure cause
was inferred and no attempt was deleted.

This lane changed no serving default, performance board, model artifact, routing decision, or
live service. It remains unmerged, untagged, and unpushed for orchestrator review.

## Scope limits

This is a target-model compute study with deterministic synthetic prompts and frozen historical
acceptance proxies. It is not a public evaluation, does not time the draft or rejection stages,
and does not claim end-to-end speculative throughput. Both expert banks were resident; the result
does not price spill misses, prefetch, H2D traffic, or partial residency and therefore must not be
used as a Hy3 spill-capacity claim. Those limits only make the pivot throughput projection more
optimistic; they do not rescue either failed frozen gate.
