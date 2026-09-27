# Context-conditioned depth with paired local evidence

The original schedule recorded ten matched sets per model, five policies per set, and one eight-turn conversation per run: **100 runs and 800 held-out timed turns**. The throughput comparison uses **95 runs and 760 turns** after excluding complete matched sets flagged by the exact-repeat screen. The remaining original seeds were completed without replacements.

The metric is total returned output tokens divided by total native request seconds. Rendering/tokenization, checkpoint restore, suffix processing, classification, policy updates, generation, synchronization and detokenization are included. Model loading, token-byte table construction, warmup and receipt I/O are excluded.

All included calibration, development and held-out records passed the exact-repeat screen. The reports reproduced from their token tapes, times, source bindings, cache traces and paired-evidence records.

Excluded matched sets (all five policies are omitted from the throughput comparison):

- gemma, cycle 4: calibrated turn 8: period 3, 258 repeated tokens. Raw records remain archived.

These are comparisons among complete conversations that passed the screen. The excluded response and its matched controls are retained separately from the score; the screening outcome is part of the result.

| Model | Global fixed K | Prose K | Code K | Numeric K |
|---|---:|---:|---:|---:|
| qwen | 3 | 3 | 3 | 4 |
| gemma | 4 | 2 | 4 | 4 |

These are calibration choices. Both contextual policies begin with the same priors; the held-out comparison below measures whether using or updating them improves complete request throughput. K counts drafted tokens.

| Model | Calibrated fixed K | Fixed tok/s | Native tok/s | Frozen context tok/s | Online tok/s |
|---|---:|---:|---:|---:|---:|
| Qwen3.8-27B NVFP4+Q5_K / embedded MTP | 3 | 115.117 | 112.889 | 113.992 | 114.461 |
| Gemma 4 12B QAT Q4_0 / QAT Q8_0 assistant | 4 | 188.940 | 192.184 | 195.484 | 192.671 |

| Model / online compared with | Pooled change | Median paired change | Wins |
|---|---:|---:|---:|
| qwen / fixed | -0.57% | -1.14% | 4/10 |
| qwen / native | +1.39% | +1.79% | 8/10 |
| qwen / frozen context | +0.41% | +0.33% | 7/10 |
| gemma / fixed | +1.97% | +1.59% | 7/9 |
| gemma / native | +0.25% | +0.19% | 5/9 |
| gemma / frozen context | -1.44% | -2.11% | 2/9 |

The frozen-context comparison isolates whether online preference updates add value beyond a calibrated context lookup. Models and protocols are never pooled together.

## What the controller actually did

| Model | Completed local comparisons | Preference adoptions | Recorded policy CPU / request time | Trial round time / request time |
|---|---:|---:|---:|---:|
| qwen | 290 | 7 | 0.0071% | 1.761% |
| gemma | 111 | 3 | 0.0102% | 1.674% |

A preference adoption requires a candidate to beat an 8/4/8-round locally bracketed incumbent comparison by more than 2%, with no more than 10% before/after baseline drift, on two distinct prompts. Context transitions reuse the established preference immediately. Trial time includes useful generated output and is not removable overhead. The CPU figures are recorded classifier/update diagnostics, not counterfactual asynchronous speedups.

## Scope and validation

- One RTX 5090 32GB, CUDA 13.1 / sm_120a, driver 595.84, recorded 525 W limit. The power limit is metadata, not a causal explanation.
- Full vocabulary heads; no head masking or confidence cutoff. Temperature 0.7, top-k 20, top-p 0.95, up to 2,048 returned tokens per turn, 49,152-token reservation.
- Initial prompts are approximately 16K tokens. Render-stable prompt checkpoints persist across turns; the previous reply and new user text are re-primed as a suffix. This does not retain every generated KV row.
- The scenarios are synthetic and share a conversation format. The byte classifier uses fences, numeric density and a short prompt hint; it is not a semantic classifier.
- The paired revision uses fresh held-out topics and seeds. First-iteration development results motivated the update; its partial held-out outcomes were not used.
- Native-session correctness is checked cold/resumed and across all allowed fixed depths and the contextual policies. Ordinary kernel, prefill/decode, plain/spec and Gemma session-boundary runner receipts are retained separately.
- This is a direct native-session study, not an HTTP/concurrency or vendor-default serving qualification. No serving default is changed by this report.

## Source and evidence

Measured source: `cb2f1783a0818705c5528f5b11cae0b0bc176deb`.

Runtime archive SHA-256: `7f5a5d5446e3af717085d0258a6993743e892a32badc79af94366523a6144328`.

- qwen binary SHA-256: `87045f0d1dc5b1fb00667741065506c7d05d8458a25c290c4154aaa1c44570cd`.
- gemma binary SHA-256: `014b3087ac1730a32bce9483e5f3b022d36bc099f5587b29e728f079d45e26c9`.
- The current portable archives contain the raw commands, token tapes, timing, GPU telemetry, calibration, state traces and paired comparisons. `publication/reproduce.py` verifies them before loading their exact audit code.
- `prior-receipts/` preserves the first iteration’s completed development study and its rejected calibration set separately. Its results are not mixed into the final score.
