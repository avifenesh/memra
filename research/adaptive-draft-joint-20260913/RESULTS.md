# First execution results — 2026-09-13

This is a native correctness and learning-data pilot on one RTX 5090. It is not a serving performance result and does not establish joint-controller novelty.

## Model and code locks

The exact two targets and Gemma assistant are in `artifacts.lock.json`. The Qwen artifact matches the earlier local qualified file byte for byte. Starting engine source is `3bb21381848067d922dec1320261846f99ceb29a`; this branch adds the diagnostic recorder, native prompt helper and token-limit correction described below. Binary hashes are retained in each pilot manifest. No E4B artifact was used.

## Correctness finding and repair

The first coding prompt exposed a one-shot Gemma speculative generation error: a 96-token request returned 97 tokens. The first 96 matched plain decoding. This affected K=1,2,4,6,8 and the traced K=4 arm. The old native gate compared only the common prefix and returned exit 0; the new harness rejected full-length inequality. Original evidence: `raw-r1/gemma-p0-k1.log` and the other `gemma-p0-*` rows; first-pilot outcome was 15 passing and 6 failing subprocess checks.

Cause: after appending the pending token, the loop checked EOS but did not check whether that token reached the requested limit before appending an accepted draft. The fix checks the length at that boundary. The native Gemma gate now rejects unequal lengths and token-limit overshoot instead of reporting common-prefix agreement as enough.

Rerun `r2`: **21/21 subprocess checks passed**. These comprise Qwen K=1..8 on three prompts, Gemma K=1,2,4,6,8 on three prompts, and three additional traced Gemma K=4 runs. All had nonzero draft acceptance. The three recorder ON/OFF token comparisons also passed. Retained commands, complete token outputs and exit statuses: `raw-r2/raw/runs-r2.jsonl` and `raw-r2/raw/pilot-r2/`. The same pinned public smoke prompts were used before and after the fix; none enters training or final policy selection.

The Gemma width ceiling was explicitly raised to 8 in the qualification cells, avoiding a falsely labelled K8 result that actually ran under the default K7 ceiling. No new serving default is selected by this pilot.

## Learning setup and current evidence

Code-review workload from native engine source, split by source file before generation: 88 training, 24 calibration, 24 held-out prompts. The exact prompt manifest is banked separately from the public smoke set.

Own-generation rank corpora:

| Target | Generated training tokens | Minimum for 4,096 rows | Status |
| --- | ---: | ---: | --- |
| Gemma 4 12B QAT Q4_0 | 22,528 | 16,384 | Corpus floor passed |
| Qwen3.5-9B NVFP4 | 22,525 | 16,384 | Corpus floor passed |

Both rank artifacts and original generated token IDs are checkpointed with hashes. The diagnostic completed 54 Gemma trace runs: full, static-trimmed and append-only adapting heads on six training, six calibration and six held-out prompt groups. All 54 matched their plain target's complete 128-token output. Two small acceptance predictors were fitted using training data only. The calibration and held-out partitions remained separate.

**Measured predictor result:** the simple head-aware fit lost narrowly to the shared baseline on this held-out set. This does not settle the proposed joint head/depth controller, which was not implemented or evaluated in this pilot.

| Held-out metric, lower is better | Shared predictor | Head-aware predictor |
| --- | ---: | ---: |
| Brier score | 0.1793217 | 0.1796074 |
| Log loss | 0.5317165 | 0.5317317 |

Held-out scope: 2,183 labeled proposals nested in **six independent prompt groups**, each exercised under all three head configurations. Across all splits there are 6,539 labeled proposals. A paired prompt-group analysis gives a head-aware-minus-baseline Brier difference of +0.000288, with a 10,000-resample percentile interval [+0.000072, +0.000498]. This is a small, six-group pilot, not a broad generalization result. Optimizer and feature choices are fixed in `learn_pilot.py`; the calibration partition was reported but not used to tune these first fits. Weights and data hash: `checkpoints/acceptance-predictors.json`.

The anticipated confidence issue appears descriptively in these traces. On labeled proposals, static trim averaged 85.85% raw confidence and 60.24% agreement, while the adaptive setup averaged 82.53% confidence and 61.37% agreement. The adaptive head learned up to 133 additional rows in a request. These aggregates have different physical widths and differently selected proposal positions, so they do not isolate a causal effect of learning or establish a throughput improvement.

`audit_receipts.py` verified all 56 learning-stage raw logs (two corpus generations plus 54 trace runs) against their recorded SHA256 values, the learner's data hash, and the absence of duplicate prompt hashes across the 136-prompt pool. Output: `receipt-audit.json`. Raw traces and logs are in `raw-learning/`; ranks, original generation IDs and per-request learned-row sidecars are in `checkpoints/`.

Only accepted-prefix positions and the first rejection receive actual-continuation labels; later positions are censored. The recorded confidence is the physical head's softmax probability, not target acceptance. Recorder I/O and the extra confidence computation make these trace timings unsuitable for uninstrumented speed claims.

This diagnostic uses unequal head widths (full, 4,096 static, and 4,096+512 adaptive). It can investigate confidence calibration under head changes, but cannot substitute for the planned equal-capacity head comparison, adaptive-depth baselines, balanced AB/BA repetitions, or sampled serving qualification.

## Checks

- Native GPU correctness and recorder engagement: `r2` passed as above.
- Rust formatting on the three changed Rust files: passed.
- Repository runtime-flag census: passed; the recorder is documented and default OFF.
- Python scripts compile successfully. The Windows-origin shell checker required an LF-normalized execution copy; `verify_export.py` performs that conversion and removes its temporary file.

The full nine-arm controller study, learned row-replacement policy, sampled gate qualification, and serving measurements remain later stages. No inference speedup is claimed here.
