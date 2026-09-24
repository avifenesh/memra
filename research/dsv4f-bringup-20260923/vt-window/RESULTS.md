# DSv4 DSpark verify window: the per-slot confidence window becomes the default

Scope: one model, one hardware shape. `tiyuvta/DeepSeek-V4-Flash-0731-NVFP4@bafd09f8cab4f4f4f25e1cdafbcdefc05b90ee38`
on 2x RTX PRO 6000 Blackwell Workstation Edition, 2026-09-24. Served DSpark route (PP-2, matrix
expert program, host sampler, one serving lane). Binary: main `25bbb91f5`, one build for every
row (hash `ae50ee52d98058c4` in `raw/q-pair4.summary`). One boot per row under
`/tmp/memra-gpu.lock`, 250 ms telemetry, cells greedy c1 x8, sampled c1 x8 and greedy c1
ignore-eos x4.

## Arms

- `d6`: the default before this change. Unbounded depth, which is the drafter's 5 drafts plus 1.
- `d5`, `d4`, `d3`: `MEMRA_DSV4_SPEC_DEPTH=5/4/3`, a fixed per-round depth cap.
- `vt`: `MEMRA_DSV4_VT=slot`, the per-slot confidence window at tau 0.5 (dspark vtconf H4).
  Each round forwards drafts while the drafter's own per-slot accept probability stays at or
  above tau.

## Results (decode p50 tok/s, per-arm median)

| arm | N | greedy c1 | sampled c1 | greedy ignore-eos |
|---|---|---|---|---|
| d6 | 2 + 3 | 66.91 / 66.90 | 54.27 / 54.26 | 66.75 |
| d5 | 2 | 68.17 | 56.20 | 68.14 |
| d4 | 2 | 69.07 | 56.08 | 68.98 |
| d3 | 2 | 64.41 | 54.39 | 64.29 |
| **vt** | 3 | **70.86 (+5.9%)** | **59.55 (+9.8%)** | **70.71 (+5.9%)** |

- The depth sweep is `raw/q-pair4.sh`, order d6 d4 d3 d5 then d5 d3 d4 d6, N=2 per depth.
- The window A/B is `raw/q-pair8.sh`, order vt d6 d6 vt vt d6, N=3 each. Rows within an arm
  agree to 0.1 tok/s.
- q-pair4's two vt rows are void. They passed `MEMRA_DSV4_VT=slot@0.5`, which the engine
  refused per request, and the bench then counted empty streams as successes. Both defects are
  fixed: #708 refuses a bad value at boot, and `raw/bench.py` counts a stream error or a missing
  `finish_reason` as an error.

Text: every request's text hash is identical across all six vt and d6 rows. That holds for
greedy and for sampled, where the position-keyed draws make the sampled output independent of
the verify width. The window changes how many drafts each round verifies, never which tokens
are emitted.

## Why the window beats every fixed depth

At d6 a round drafts about 4.8 tokens and accepts about 2.4 of them (the `[dspark-acc]` lines in
`raw/depth/r11-prof/serve.log`: for example 181 of 358 drafted over 74 rounds). A fixed cap
trades verify width against accepted length for every round alike. The per-slot window cuts
each round where the drafter's own confidence drops, so it verifies wide when the draft is good
and narrow when it is not. Sampled traffic accepts less and gains more (+9.8%).

## Decision

`MEMRA_DSV4_VT` unset now resolves to `slot` at tau 0.5, floor 0. `off` stays as the rollback
seam, decide-by 2026-10-08, and is deleted then if unused. `MEMRA_DSV4_VT_TAU` and `_FLOOR` tune the
default window; set together with `off`, they refuse.
