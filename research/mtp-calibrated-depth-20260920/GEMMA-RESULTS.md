# Gemma: calibrated K=3 beats the cost learner

Across ten held-out paired sets, calibrated fixed K=3 reached **206.31 request-E2E
output tok/s**. Learned D reached **199.69**, a **3.21% loss**, with no wins over
the calibrated control. Learned D remained ahead of the original K=5 control but
was effectively flat against native adaptation.

This is the completed **cold-prefill** study. Conversation text and learner state
persist across requests; KV is rebuilt. It does not establish the result for
native prompt-prefix reuse. See `CONTINUING-SESSION-CAVEAT.md`.

## Held-out throughput

| Full-head policy | Output tok/s | Output tokens | Request seconds |
|---|---:|---:|---:|
| Calibrated fixed K=3 | 206.311 | 162,591 | 788.087 |
| Original fixed K=5 | 189.368 | 162,210 | 856.585 |
| Native adaptive | 199.458 | 162,269 | 813.550 |
| Native adaptive with timing fences | 199.309 | 162,269 | 814.157 |
| Learned D | 199.686 | 162,679 | 814.673 |

Each policy has ten runs, with two independent eight-turn conversations per run.
Rates pool returned output tokens over request seconds.

| Learned D compared with | Pooled change | Median paired change | Wins, N=10 |
|---|---:|---:|---:|
| Calibrated K=3 | -3.21% | -3.65% | 0/10 |
| Original K=5 | +5.45% | +5.68% | 10/10 |
| Native adaptive | +0.11% | +0.30% | 6/10 |
| Native adaptive with timing fences | +0.19% | +0.36% | 6/10 |

Calibration selected K=3 before evaluation, using two separate source pools and
opposite depth orders. Its pooled calibration rate was 214.02 tok/s. That number
is separate from the held-out estimate.

## Evidence and scope

The completed remote audit covers **50 runs and 800 turns**, including full-head
engagement, token and time accounting, cold priming, native/instrumented token
identity, and replay of the recorded exploration schedule. Short and full-workload
greedy gates cover all nine policies over 288 turns, with 256 comparisons to
native references. Gate evidence is in `receipts/gemma-gate-audit.json`.

The target is the pinned Gemma 4 12B QAT Q4_0 GGUF with its QAT Q8_0 MTP assistant.
Both heads cover all 262,144 vocabulary entries; head masking and confidence cuts
are off. Sampling is temperature 0.7, top-k 20, top-p 0.95, with 1,024-token output
caps. The initial held-out prompts contain 4,060 and 4,221 tokens. Context
reservation is 49,152 tokens.

The clock includes rendering/tokenization, cold priming, generation,
synchronization and policy carryover. Truncation checks, output detokenization
and receipt writes follow the clock. Startup and at least ten seconds of warmup
precede scoring. All scored runs satisfy the original sixty-second threshold.

Hardware is one RTX 5090 32GB, driver 595.91.07, configured 525W, CUDA 13.1 /
sm_120a. Power metadata is a measurement condition, not an attribution of the
observed policy differences.

The controller starts at K=5, explores K=1..5 in sixteen-round blocks, uses EWMA
0.5 and 1% hysteresis, and probes after eight exploitation blocks. These results
do not isolate the causal cost of probing.

The sealed remote `gemma-audit.json` is preserved as `gemma-analysis.json`.
Calibration, selection, gates, and all held-out records are in
`receipts/gemma-records.tar.gz`, with per-file hashes in
`receipts/gemma-records.sha256`. The archive was verified against the final
remote receipt seal and checked member by member after compression.
