# Full-head MTP depth with native prompt-prefix reuse

The cost learner did not beat the calibrated fixed-depth control on either
artifact. Qwen lost **4.25%** to K=2 and **3.35%** to native adaptation. Gemma lost
**2.96%** to K=3 and was effectively tied with native adaptation (**+0.07%**).
Retain the current runtime policies; these results do not justify promoting the
experimental learner.

## Completed held-out comparison

Each model has ten balanced matched sets, five policies per set and one
eight-turn conversation per run: **100 runs and 800 timed turns** in total.

| Model and full-head policy | Output tokens | Request seconds | Output tok/s |
|---|---:|---:|---:|
| Qwen3.8-27B, calibrated K=2 | 128,817 | 1,138.058 | 113.190 |
| Qwen3.8-27B, original K=3 | 132,112 | 1,172.162 | 112.708 |
| Qwen3.8-27B, native adaptive | 126,396 | 1,127.144 | 112.138 |
| Qwen3.8-27B, instrumented native | 126,396 | 1,128.126 | 112.041 |
| Qwen3.8-27B, learned D | 134,111 | 1,237.437 | 108.378 |
| Gemma 4 12B, calibrated K=3 | 53,029 | 297.045 | 178.521 |
| Gemma 4 12B, original K=5 | 51,651 | 342.871 | 150.642 |
| Gemma 4 12B, native adaptive | 53,840 | 311.001 | 173.119 |
| Gemma 4 12B, instrumented native | 53,840 | 311.127 | 173.048 |
| Gemma 4 12B, learned D | 51,199 | 295.539 | 173.240 |

The metric is **total returned output tokens / total native request seconds**.
Gemma's learner takes slightly less time than calibrated K=3 but returns fewer
tokens, so its throughput is lower. Models and protocols are never pooled.

| Learned D compared with | Qwen pooled change | Qwen median pair / wins | Gemma pooled change | Gemma median pair / wins |
|---|---:|---:|---:|---:|
| Calibrated fixed depth | -4.25% | -4.24% / 1 of 10 | -2.96% | -2.94% / 1 of 10 |
| Original fixed depth | -3.84% | -3.55% / 1 of 10 | +15.00% | +14.69% / 10 of 10 |
| Native adaptive | -3.35% | -3.31% / 0 of 10 | +0.07% | -0.16% / 4 of 10 |
| Instrumented native | -3.27% | -3.19% / 0 of 10 | +0.11% | -0.04% / 5 of 10 |

Gemma's gain over the original K=5 control does not establish a gain from online
learning: native adaptation captures essentially all of it, and calibrated K=3
is faster still.

## What persists across the conversation

All later turns must reuse the exact preceding **render-stable prompt
checkpoint**. The previous reply and new user text are processed as the next
suffix. This follows the native prompt-checkpoint path; it does not claim that
every generated KV row survives a turn boundary.

Missing checkpoints, changed token prefixes, overwritten state and cold fallback
are fatal. Every turn records cached tokens, newly processed tokens and the next
checkpoint. Across all recorded input tokens, including each conversation's cold
first turn, learned D reused **83.81% on Qwen** and **85.84% on Gemma**. The other
policies use the same cache rule.

The complete learner state persists within and across replies, including partial
observation blocks. Final per-depth counters equal the eligible observations
from all eight requests. The recorded exploration schedule replays, and both
models exercise upward and downward depth changes.

## Protocol and identity

- One RTX 5090 32GB, driver 595.84, configured 540W, CUDA 13.1 / sm_120a.
  Watts and clocks are receipt metadata, not explanations for the policy result.
- Qwen uses the pinned NVFP4+Q5_K GGUF and embedded MTP; Gemma uses the pinned
  QAT Q4_0 target and QAT Q8_0 assistant. Full vocabularies are 248,320 and
  262,144 respectively. Artifact revisions and hashes are in the two lock files.
- The cost learner is unchanged: sixteen-round blocks, EWMA 0.5, 1% hysteresis
  and a periodic probe after eight exploitation blocks. Candidates are K=1..7
  for Qwen and K=1..5 for Gemma. Head masking and confidence cuts are off.
- Sampling is temperature 0.7, top-k 20, top-p 0.95, with up to 2,048 returned
  output tokens. These are pinned study settings, not a vendor-default serving
  qualification.
- The held-out initial prompts contain 16,389 / 16,393 Qwen tokens and
  16,390 / 16,385 Gemma tokens. Context reservation is 49,152.
- Calibration uses two disjoint source-code pools and opposite depth orders
  before evaluation. It selects K=2 and K=3. Evaluation alternates two other
  source-code pools, with ten unique seeds and balanced forward/reverse orders.
- Every run is a fresh process with at least ten seconds of unscored warmup.
  There is no response-length or minimum-duration filter on completed runs.
- Request time includes rendering/tokenization, checkpoint restore, suffix
  priming, generation, synchronization and output detokenization. Model startup,
  warmup and receipt I/O are excluded. This is the direct native-session API,
  not an HTTP/WAN or concurrent-serving measurement.

Measured runtime source:
`b0703b0a2bcbfe9e17e8166a79acc73057b4dffb`.

The source archive SHA-256 is
`93a7d4131d0fbe9e503a8152f560d55e4a83fcce152d73032bc773188837b002`.
It contains the entire buildable measurement source and required compile-time
fixtures. The candidate includes the merged continuation fixes and the Gemma
direct restored-session prime adjustment. The publication applies no runtime
change.

## Validation and receipts

Both families pass short and full-length cold/resumed greedy identity at every
tested depth: 352 correctness turns. All 160 paired timed native/instrumented
prompt and output tapes match. The full offline audit verifies 50 runs / 400
turns per model, the calibration winner, fixed-depth engagement, cache reuse,
learner continuity, unique seeds, frozen arm orders, runtime/artifact identity
and token/time arithmetic.

Eleven focused preparation tests and five archive-reader tests pass. Eight
deliberate corruptions of selection, calibration records, evaluation seeds and
throughput are rejected. A separate reset-state red arm rejects lost learner
observations. All three portable archives unpack safely and reproduce both
complete reports exactly.

- [Qwen audit, pair results and round diagnostics](qwen-analysis.json)
- [Gemma audit, pair results and round diagnostics](gemma-analysis.json)
- [Archive manifest](receipts/manifest.json)
- [Exact archive reproduction receipt](receipts/REPRODUCTION.json)
- [Earlier cold-prefill follow-up](../mtp-calibrated-depth-20260920/RESULTS.md)

The cold and warm studies differ in runtime, input/output shape and clock
boundaries. Their absolute rates are not a cache-speedup comparison. Periodic
probe costs are descriptive here; a probes-on/off experiment remains necessary
to establish their causal contribution.
