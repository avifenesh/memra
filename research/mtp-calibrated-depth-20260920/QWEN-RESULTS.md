# Qwen: calibrated fixed depth remains ahead of the cost learner

On ten fresh paired held-out sets, fixed K=2 reached **100.35 native request-E2E output tok/s**. The unchanged cost learner reached **96.24**, losing **4.10%** to calibrated fixed K on all ten pairs. It also lost 3.06% to the original K=3 control and 2.21% to native adaptation. This is a negative result for this online policy on the tested code-review workload.

Calibration selected K=2 before opening evaluation: 102.39 tok/s versus 100.38 at K=3 across two separate source pools. On held-out prompts, calibrated K=2 improves 1.09% over K=3 (8/10 paired wins) and 1.97% over native adaptation. Keep calibration and held-out estimates separate.

## Held-out throughput

| Full-head policy | Output tok/s | Output tokens | Request seconds |
|---|---:|---:|---:|
| Calibrated fixed K=2 | 100.352 | 80,751 | 804.677 |
| Original fixed K=3 | 99.272 | 80,680 | 812.719 |
| Native adaptive | 98.414 | 77,046 | 782.880 |
| Native adaptive with timing fences | 98.362 | 77,046 | 783.291 |
| Learned D | 96.235 | 80,672 | 838.278 |

Rates pool output tokens / request seconds. Each arm has ten eight-turn conversations; there is no cross-model pooling. Startup and at least ten seconds of warmup precede each run. All scored runs last at least sixty measured seconds.

| Learned D compared with | Pooled change | Median paired change, N=10 | Wins |
|---|---:|---:|---:|
| Calibrated fixed K=2 | -4.10% | -3.50% | 0/10 |
| Original fixed K=3 | -3.06% | -3.03% | 0/10 |
| Native adaptive | -2.21% | -2.72% | 2/10 |
| Native adaptive with timing fences | -2.16% | -2.63% | 2/10 |

## Every pair

| Seed | K=3 | Native | Calibrated K=2 | Learned D | D vs calibrated |
|---|---:|---:|---:|---:|---:|
| 20263000 | 101.25 | 98.99 | 102.28 | 99.65 | -2.57% |
| 20263001 | 98.88 | 95.91 | 98.50 | 95.67 | -2.88% |
| 20263002 | 100.54 | 100.48 | 100.97 | 97.67 | -3.27% |
| 20263003 | 97.55 | 97.05 | 97.84 | 95.02 | -2.88% |
| 20263004 | 99.00 | 100.36 | 100.79 | 97.71 | -3.06% |
| 20263005 | 97.52 | 98.30 | 99.26 | 94.66 | -4.63% |
| 20263006 | 100.42 | 99.49 | 102.73 | 95.89 | -6.66% |
| 20263007 | 98.27 | 97.83 | 101.45 | 94.93 | -6.43% |
| 20263008 | 100.14 | 96.79 | 100.76 | 97.01 | -3.72% |
| 20263009 | 99.17 | 98.47 | 99.03 | 94.21 | -4.87% |

## Audit and scope

All 50 held-out runs / 400 turns pass the complete offline audit. All 80 paired native/instrumented-native replies and prompt tapes match exactly. Both held-out pools have five distinct native first-reply tapes across five seeds, confirming observed sampling variation. Calibration receipts, the frozen K=2 choice and actual fixed-depth round traces pass a separate audit. Deliberately corrupted choices, receipt hashes and depth traces are rejected.

Before scoring, short and full-workload greedy gates matched all eleven arms over 176 turns, giving 160 comparisons to native references. These gates establish policy self-consistency; the earlier plain-target oracle remains historical evidence in PR #569.

One RTX 5090 32GB, driver 595.91.07, configured 525W, CUDA 13.1, sm_120a. The pinned Qwen3.8-27B NVFP4+Q5_K GGUF supplies embedded MTP; both heads cover all 248,320 tokens. H and confidence cuts are off. Source/model manifests retain exact revisions, bytes and hashes. No format substitution or runtime-default change.

The sampler is temperature 0.7, top-k 20, top-p 0.95, with 1024 output-token caps. Initial held-out prompts contain 3789 and 3998 tokens; reservation 49152 is capacity, not a claim that inputs have that length. Actual own-answer history carries through eight turns. Every request is primed cold, preserving the previous study's numeric-lineage isolation; learner state alone persists.

The native-study timer begins before prompt rendering/tokenization and includes cold priming, generation, synchronization and policy carryover through completed token IDs. Output text decoding and receipt writes occur afterward. This is the unchanged native request clock from the handover, not an HTTP/WAN serving measurement.

The unchanged cost learner starts at K=3 and explores K=1..7 with 16-round blocks, EWMA 0.5, 1% hysteresis and a periodic probe after eight exploitation blocks. The extra fixed:K driver argument changes only the diagnostic fixed policy. The legacy startup label fixed_depth=3 still names the original control; actual added-arm depths are in the command and independently checked round tapes.

The earlier study's results remain unchanged. These new held-out prompts and fresh controls show that a calibrated static point beats this learner; they do not identify the causal cost of periodic probes. The separate probes-on/off experiment is still pending.

Raw gates, calibration and all held-out records are in receipts/qwen-records.tar.gz with per-file hashes in receipts/qwen-records.sha256. qwen-analysis.json, qwen-selection-audit.json and qwen-audit-rejection-tests.json retain the audit outputs.
