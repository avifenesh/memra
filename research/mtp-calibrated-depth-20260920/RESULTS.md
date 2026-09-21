# Full-head MTP depth: cold-prefill calibrated controls

The completed cold-prefill follow-up favors calibrated fixed depth over the
unchanged cost learner on both tested artifacts.

| Model | Calibrated K | Calibrated output tok/s | Learned output tok/s | Learned change | Learned wins |
|---|---:|---:|---:|---:|---:|
| Qwen3.8-27B NVFP4+Q5_K, embedded MTP | 2 | 100.352 | 96.235 | -4.10% | 0/10 |
| Gemma 4 12B QAT Q4_0, Q8_0 assistant | 3 | 206.311 | 199.686 | -3.21% | 0/10 |

These are separate model studies on one RTX 5090. No cross-model pooling is
used. Each model has ten matched sets and five policies. Qwen contributes
400 held-out turns; Gemma contributes 800 because its original protocol runs
two independent eight-turn conversations per run.

Full heads, the learner, weights and sampling settings stay fixed. Calibration
uses separate source-code prompts before held-out evaluation. The primary metric
is total returned output tokens divided by total native request-E2E seconds.

KV is rebuilt on every request. The conversation and learner persist; this is
not evidence for a conversation using native prompt-prefix reuse. Initial
prompts are approximately 4K tokens, output caps are 1K, and output text
detokenization follows the measured clock. The continuing-session study has a
separate protocol and runtime identity.

- [Qwen result and every pair](QWEN-RESULTS.md)
- [Gemma result and scope](GEMMA-RESULTS.md)
- [Continuing-session correction](CONTINUING-SESSION-CAVEAT.md)
- [Receipt manifest](receipts/manifest.json)

Keep the existing runtime policies. This follow-up supplies a stronger fixed
control; it does not establish a serving default or explain the causal effect
of periodic probes.
