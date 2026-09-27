# Context-conditioned speculative depth

The calibrated depth depends on content in this study, but the tested online
update rule did not beat the strongest control for either model. Pulling current
Memra fixes and repeating the comparisons did not materially change that result.

| Model | Strongest control | Online result against it |
|---|---|---:|
| Qwen3.8-27B NVFP4+Q5_K with embedded MTP | Global fixed K=3 | -0.57%, 4/10 wins |
| Gemma 4 12B QAT Q4_0 with Q8_0 assistant | Frozen contextual calibration | -1.44%, 2/9 wins |

Gemma's calibrated profile used K=2 for prose and K=4 for code/numeric spans.
That frozen lookup was faster than its native adaptive control. Qwen's profile
used K=3 for prose/code and K=4 for numeric spans. These are measured calibration
choices, not a rule that harder tasks always need a longer draft.

[RESULTS.md](RESULTS.md) contains the complete original comparison, controls,
learning events and scope. [LATEST-ENGINE.md](LATEST-ENGINE.md) records the
interleaved check after updating to Memra main dc192cd9. The exact tested runtime
implementations are preserved in their source archives. No experimental dispatch
path or serving-default change is retained in the active engine.

## Method

The controller observes the latest user instruction and committed output only.
A 128-byte window, Markdown fence state, numeric density and a short prompt hint
classify prose, code and numeric spans. The features are inexpensive and do not
consume workload labels or future output.

Both contextual policies start at the calibrated global K. Each context keeps a
preferred K that can be reused immediately. The online policy tests an adjacent
depth after 128 eligible rounds when its prior is close or current cost worsens.
A comparison contains eight incumbent rounds, four candidate rounds and eight
more incumbent rounds, all in one context and request. Incumbent rates before
and after must agree within 10%; the candidate must improve the pooled local
rate by over 2%. Two distinct requests must confirm a preference change.

Incomplete comparisons stop at context/request/terminal boundaries. Counts,
partial diagnostic blocks, established preferences and confirmations persist
across requests. Terminally truncated rounds do not update the learner, but
their time remains in the complete-request denominator.

The controls are native adaptation, a globally calibrated fixed K, and a frozen
context policy using the same priors and classifier. An instrumented native arm
measures tracing overhead. Every scored conversation uses native prompt-checkpoint
reuse across eight turns. The previous reply/new user suffix is re-primed; the
study does not retain every generated KV row.

The primary metric is returned output tokens divided by complete native request
seconds. Model loading, warmup and receipt I/O are excluded. The setup is one
RTX 5090 32GB, full vocabulary heads, temperature 0.7, top-k 20, top-p 0.95,
approximately 16K initial prompts, and a 2,048-token output cap. This is a native
session study, not HTTP, concurrency, or vendor-default serving qualification.

## Recorded and scored data

The original schedule recorded 100 runs / 800 turns. In Gemma cycle 4, fixed K=4,
turn 8 emitted an exact three-token `*10` cycle at least 86 times. The original
guard stopped the run. After reviewing the flag, the remaining original cycles
5–9 continued without replacement seeds or changes to policy, priors, inputs or
sampler. All five policies in the flagged set were excluded, leaving 95 runs /
760 turns for throughput. Every raw record remains archived. Independent replay
must reproduce every exclusion and prove that no recorded control was omitted.

The initial controller's two development sets per model are banked separately.
Its development results motivated the paired-evidence revision. No outcome from
its interrupted partial held-out run selected that revision. Its rejected
calibration set is also retained.

The latest-engine follow-up used 32 interleaved runs / 256 turns, shared original
priors and workload pools, fresh seeds, and both runtime orders. No group was
excluded. Every paired prompt/output token-ID tape matched across versions.
Online throughput changed by -0.021% on Qwen and +0.002% on Gemma. This is a
bounded diagnostic comparison, not a new calibration or deployment decision.

## Evidence and reproduction

- `prior-receipts/`: first-controller calibration, correctness, development and
  the rejected calibration set; runtime e1cd38a4.
- `receipts/`: complete original paired-evidence study, including its excluded
  matched set; runtime cb2f1783.
- `latest-receipts/`: old/latest interleaved comparison and current-engine gates;
  latest runtime 5450580f, based on upstream dc192cd9.

Each bundle has a pinned manifest, exact member hashes, raw token/timing records,
source bindings and an expanded publication-boundary review. The shared reader
checks all members before extraction. Reproduction loads only recognized,
hash-verified source snapshots and does not overwrite the recorded results.
The CI workflow replays all three bundles and their boundary reviews.

The initial source-export attempt for the latest engine omitted its new
build-time FLAGS registry. It was stopped before scoring and retained privately;
the corrected archive contains the exact registry. Operational records, unused
interrupted outputs and exact executables are banked privately with a complete
custody manifest.
