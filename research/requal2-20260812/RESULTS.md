# P0: naked-default sold-envelope requalification

Date: 2026-08-12

Rig: box1 GPU0, one NVIDIA RTX PRO 6000 Blackwell Server Edition, one model resident at a time

Scored runtime source: `e78054f5fec808703d050a5d9545f2ac2cc162cb` (`584ed0af0+`, v0.81.0 naked default)

## Verdict

**P0 REGRESSION: the pre-coldhol pair qualification is stale on the v0.81.0 naked default.**

- Q35-A3B is **not qualified at c=4**. It passed only 34/40 required base cells and 41/80
  scored cells. Across the full N=5 grid, 714/2,300 requests stopped at 26/60 completion tokens.
- Q27 remains clean at c=4 and confirms the predicted clean throughput knee at c=16, but its
  pooled c=4 cache-hit TTFT p95 regressed from 21.565 ms to 269.139 ms. The old latency envelope
  therefore does not survive even though all 70 Q27 cells completed cleanly.
- The pair must not be represented by the old `research/requal-20260812/RESULTS.md` envelope.
  The OpenRouter application has already been submitted with that 40/40 narrative; Surplus and
  Onlist amendments explicitly retained their earlier capacity claims; Q35 is active in the
  OpenModels and BitRouter surfaces. Those are correction/hold follow-ups, not changes made here.

The reducer emits `P0_REGRESSION` with 42 records: 39 failed Q35 cell-integrity records, the Q27
hit-TTFT p95 regression, and the consequential Q35 knee/headroom regressions.

## Sold-cap envelope and explicit old-to-new diff

Each c=4 mixed row pools N=5 cells: 90 full-prefix hits and 10 real misses. Q35's numeric c=4
latency observations remain useful diagnostically, but its throughput is not a publishable
envelope because only 98/100 requests reached 60 tokens.

| Model | Required base | All scored | c=4 hit TTFT p50 | c=4 hit TTFT p95 | c=4 mixed output | Clean knee | Headroom over c=4 | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Q27 | **40/40** | **70/70** | **18.296 ms** | **269.139 ms** | **144.641 tok/s** | **c=16** | **300%** | c=4 clean; **P0 tail regression** |
| Q35-A3B | **34/40** | **41/80** | 7.388 ms | 99.115 ms | 407.072 tok/s (invalid envelope) | none at or above c=4 | **0% qualified** | **P0: NOT at c=4** |

| Model / metric | Old envelope | New default | Delta | Direction |
|---|---:|---:|---:|---|
| Q27 hit TTFT p50 | 18.573 ms | 18.296 ms | -0.277 ms / -1.491% | improved |
| Q27 hit TTFT p95 | 21.565 ms | 269.139 ms | **+247.574 ms / +1,148.033%** | **regressed** |
| Q27 mixed output | 144.462 tok/s | 144.641 tok/s | +0.179 tok/s / +0.124% | improved |
| Q27 clean knee | c=12 | c=16 | +4 / +33.333% | improved |
| Q27 c=4 headroom | 200% | 300% | +100 percentage points | improved |
| Q35 hit TTFT p50 | 7.623 ms | 7.388 ms | -0.235 ms / -3.083% | observed improvement, cell invalid |
| Q35 hit TTFT p95 | 101.203 ms | 99.115 ms | -2.088 ms / -2.063% | observed improvement, cell invalid |
| Q35 mixed output | 394.157 tok/s | 407.072 tok/s | +12.915 tok/s / +3.277% | **not comparable: 68 tokens short** |
| Q35 clean knee | c=32 | none at or above c=4 | reducer floor c=4, -28 / -87.5% | **regressed** |
| Q35 c=4 headroom | 700% | 0% qualified | **-700 percentage points** | **regressed** |

Q27 mixed throughput rose cleanly through c=16 and then declined: 144.641, 178.665, 186.195,
187.839, and 186.290 completion tok/s at c=4/8/12/16/20. That confirms c=16 as the clean
first-decline knee on the sold workload shape.

The Q27 hit-tail regression was absent in repetitions 1 and 2, then reproduced in repetitions 3,
4, and 5. Their per-cell hit p95 values were 18.873, 19.876, 269.139, 299.354, and 296.810 ms;
five of the pooled 90 hit requests exceeded 100 ms. This is a repeated N=5 tail, not a single-run
claim.

## Complete regression surface

Q27 was clean 5/5 in both cold and mixed90 at every measured width c=1,2,4,8,12,16,20:
1,400/1,400 requests reached 60 tokens and no cell failed.

Q35 completed every scheduled cell, but 39/80 cells failed integrity:

| c | Cold clean cells | Cold complete requests | Mixed90 clean cells | Mixed90 complete requests | Short requests |
|---:|---:|---:|---:|---:|---:|
| 1 | 5/5 | 100/100 | 5/5 | 100/100 | 0 |
| 2 | 5/5 | 100/100 | 5/5 | 100/100 | 0 |
| **4 sold cap** | 5/5 | 100/100 | **4/5** | **98/100** | **2** |
| 8 | **0/5** | **60/100** | 5/5 | 100/100 | **40** |
| 16 | **0/5** | **35/100** | **3/5** | **96/100** | **69** |
| 32 | **0/5** | **26/200** | **1/5** | **188/200** | **186** |
| 40 | **0/5** | **27/200** | **1/5** | **186/200** | **187** |
| 48 | **0/5** | **32/250** | **2/5** | **238/250** | **230** |

Every one of the 714 short Q35 responses returned HTTP 200 with `done=true`,
`finish_reason=stop`, and exactly 26 completion tokens. The client therefore marked each one
failed even though the HTTP request completed. Across all 150 cells:

- admission-session defers, admission-VRAM defers, and step-OOM parks were zero;
- prompt, cached-token, and prefix-cache-hit-token drift were zero;
- engine `tokens_out` matched the actually returned token totals; and
- no captured `out of memory`, CUDA error, or server-fatal line explains the truncation.

The captured evidence establishes the symptom, not its implementation cause.

## Correctness gates

The standard gates remained green, which is why the sold-shape replay is load-bearing:

- `kernel-check`: `ALL GREEN (95 cells, 13 skipped)` on GPU0;
- Q27 and Q35 `run-gen`: prefill/decode and batched-prime/tokenwise argmax `MATCH`;
- Q27 and Q35 `run-spec`: K=1..8 self-consistency `PASS`;
- serial prefix-cache exactness: both models reconciled exactly
  `27,702 client usage = 27,702 cached_tokens_in = 27,702 prefix_cache_hit_tokens`; and
- Q27 used 824 and Q35 used 1,288 continuation prime-batch calls across the five scored boots.

## Frozen method and provenance

- The unchanged workload lock is
  `85597a0a28ed874f440b4a966c0b43fd3e31b94fe868266de9e299decc208c34`; the imported frozen
  replay is `91eac7250e0d268ac6be8cfd1ee64e346d405dc412824dab45f224e9563e1e5b`.
- The prompt shape remained 4,860 prompt tokens plus 60 requested completion tokens. Each model
  retained the same 40 base cells at c=1/2/4/8. The extension was Q27 c=12/16/20 and Q35
  c=16/32/40/48, both frozen cold and mixed90 arms, N=5 per cell.
- Odd repetitions booted Q27 then Q35; even repetitions booted Q35 then Q27. Only physical GPU0
  was visible, one model was resident at a time, arm order alternated, and width order rotated.
- One uninterrupted `/tmp/memra-gpu.lock` hold ran from `2026-08-12T16:35:00Z` to
  `2026-08-12T17:26:08Z`. Thermal maxima during this N=5 same-lock regime were 68 C, 525.09 W,
  2,422 MHz, 77,845 MiB, and 100% utilization.
- The checkout and `target/` were fresh. Rust 1.97.1 and CUDA 13.2 produced `memra-server`
  `2ab01ba55d76844419fd4ef2ee7d7094ca21f9f827a2913ba726a17f77ce561a`.
- Scored artifacts remained byte-identical: Q27
  `d8d71c7e8a01a1c964fd904a7b496eaef19bdd66827e0949e66c723da742d517`; Q35
  `df27a780435b7b45c2597536112ea3cb091f8544c3d0c3318d9f4258b31f7adf`.
- The full raw manifest contains 111 verified entries and hashes to
  `bbae0cf4d0861e4534411254db17958717bc9089ac78d84ed99266e41a9ea76b`.
- `origin/main` advanced after the run to `9b43b556b39a17d7d85ffb919e03fb3f5234863f`. The intervening code is
  multi-device peer-probe readiness/spec-admission recovery; this harness is single-card and pins
  `MEMRA_SERVE_SPEC=0`, so the measured naked plain scheduler path did not move.

## Provider-submission impact — follow-up required

Current `origin/main` and the live BitRouter PR were rechecked after reduction:

- `research/connect-20260812/INQUIRIES.md:20-26` preserves the shared outbound claim that both
  models passed 40/40 and quotes Q27 144.462 / 21.565 ms p95 plus Q35 394.157 / 101.203 ms p95.
- `research/connect-20260812/OPENROUTER.md:39-46` is the submitted narrative: it says both models
  passed 40/40 and links the old qualification. `SUBMISSIONS.md:17` records the form confirmation.
- `research/connect-20260812/SUBMISSIONS.md:12,14` says the delivered Surplus and Onlist price
  amendments left capacity unchanged. Its BitRouter/OpenModels rows still treat Q35 as active,
  and line 63 still publishes the old qualification receipt.
- Live BitRouter PR #814 remains open at fork head
  `6e4729e237562e58bc98009639f9b1c5154106f8`. Its current `tiyuvta.yaml` has pricing and
  capability fields but **no capacity, TTFT, or throughput field to refresh**. The required
  follow-up is Q35 qualification/provider state, not a nonexistent numeric manifest field.

No message, form, provider manifest, runtime code, old historical result, generated board, merge,
tag, push, or formatting surface was changed in this evidence lane.

## Coldfix acceptance criteria

Re-run this exact same-lock N=5 workload and require all of the following before restoring the old
provider claims:

1. Q35: 40/40 base cells and 80/80 total cells clean, 2,300/2,300 requests at 60 tokens, including
   5/5 clean for every cold and mixed90 row above.
2. Q27: 70/70 cells clean and no c=4 cache-hit TTFT regression versus the old 21.565 ms p95; the
   269-299 ms repetition tails must not recur.
3. Both models: the full standard exactness battery and serial 27,702-token reconciliation pass,
   with zero accounting drift, admission/VRAM defers, OOM parks, CUDA errors, or server-fatals.

## Receipts

- Machine-readable reduction: [`analysis.json`](analysis.json)
- Sealed full campaign: [`raw/scored/campaign/`](raw/scored/campaign/)
- Correctness and exactness logs: [`raw/scored/gates/`](raw/scored/gates/) and
  [`raw/scored/exactness/`](raw/scored/exactness/)
- Full manifest: [`raw/scored/MANIFEST.sha256`](raw/scored/MANIFEST.sha256)
- Independently sealed first-failure attempt: [`raw/attempt1/`](raw/attempt1/)
