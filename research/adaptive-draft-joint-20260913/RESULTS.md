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

Both rank artifacts and original generated token IDs are checkpointed with hashes. The current next stage collects confidence/acceptance traces under full, static-trimmed and append-only adapting Gemma heads, then fits a shared acceptance predictor and a head-aware predictor on training data. The calibration and held-out partitions stay separate.

Only accepted-prefix positions and the first rejection receive actual-continuation labels; later positions are censored. The recorded confidence is the physical head's softmax probability, not target acceptance. Recorder I/O and the extra confidence computation make these trace timings unsuitable for uninstrumented speed claims.

This diagnostic uses unequal head widths (full, 4,096 static, and 4,096+512 adaptive). It can investigate confidence calibration under head changes, but cannot substitute for the planned equal-capacity head comparison, adaptive-depth baselines, balanced AB/BA repetitions, or sampled serving qualification.

## Checks

- Native GPU correctness and recorder engagement: `r2` passed as above.
- Rust formatting on the three changed Rust files: passed.
- Repository runtime-flag census: passed; the recorder is documented and default OFF.
- Python scripts compile successfully. The Windows-origin shell checker required an LF-normalized execution copy; `verify_export.py` performs that conversion and removes its temporary file.

The full nine-arm controller study, learned row-replacement policy, sampled gate qualification, and serving measurements remain later stages. No inference speedup is claimed here.
