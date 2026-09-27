# Full-head MTP learned-depth research — 2026-09-20

The learner lost to fixed K=3 on Qwen; Gemma improved over fixed K=5, with little extra benefit over native adaptation.

This study measures **MTP only, with the full vocabulary head held constant in every arm**.
H learning, vocabulary trimming and confidence-based cuts are disabled. The language-model
weights stay fixed; an external online controller chooses the number of MTP draft steps.

| Model / full head in both arms | Fixed depth E2E tok/s | Learned E2E tok/s | Learned vs fixed | Wins | Learned vs native adaptive | Wins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8-27B, 248,320 vocabulary rows; fixed K=3 | 96.750 | 94.377 | −2.45% | 1/8 | −3.42% | 0/8 |
| Gemma 4 12B, 262,144 vocabulary rows; fixed K=5 | 175.325 | 188.889 | +7.74% | 8/8 | +0.43% | 6/8 |

Native adaptive rates were 97.714 tok/s for Qwen and 188.080 for Gemma. These are
within-model paired comparisons. Different model sizes, templates and thinking behavior
make the absolute rates across models unsuitable as a depth-policy comparison.

## What this establishes

The Gemma result is a repeatable improvement over the specified fixed K=5 control.
Its existing reactive policy captures almost all of that benefit. The small additional
+0.43% for learning has two losing pairs and a +0.36% median paired change; it should
not be described as a large or general learning advantage. On Qwen, this learner loses
to both the fixed K=3 control and the existing reactive rule.

Fixed K=3 and K=5 were selected before measurement from the native MTP starting points.
They are not a calibrated best-fixed-depth oracle for this workload. No claim is made
that these results rule out other learned controllers, depths or workloads.

## Configuration and controls

- Qwen: NVFP4+Q5_K GGUF with its own embedded MTP block, full target and MTP projection.
  Learned and native adaptive policies may use K=1..7; the learner starts at K=3.
- Gemma: QAT Q4_0 target with its matching QAT Q8_0 MTP assistant, full vocabulary.
  Learned and native adaptive policies may use K=1..5; the learner starts at K=5.
- Four arms: fixed depth, native adaptive depth, instrumented native adaptive depth,
  and learned depth. The fixed and learned arms include timing instrumentation;
  native versus instrumented-native measures that overhead separately.
- This is the disclosed new Memra-native cost learner, not a faithful port of the
  unavailable original remote H/C/D source. It uses 16 eligible rounds per block,
  pooled emitted tokens / measured round time, EWMA 0.5, 1% switching hysteresis,
  and periodic probes after eight exploitation blocks. All candidates are initially explored.
- Every request is primed cold, and the native KV cache is used within the request.
  Only learner/history state carries across conversation turns. This prevents depth-dependent
  commit overshoot from also changing whether a later prompt is cold-primed or cache-resumed.
- One RTX 5090 32GB, driver 595.71.05, 575W, CUDA 13.1 / sm_120a; standard GPU lock,
  serialized execution and GPU-process monitoring. No new machine was rented.
- One frozen code-analysis workload using real repository source, without repeated padding.
  Initial prompts have 3,703 Qwen tokens and 4,037 Gemma tokens. Context grows over eight turns.
  Qwen uses its reasoning-preserving template; Gemma uses its native default template.
- Temperature 0.7, top-k 20, top-p 0.95, no penalties, max output 1,024, context reservation
  49,152. Qwen: eight continuing turns/run. Gemma: two independent eight-turn conversations/run,
  resetting history and learner and offsetting the second seed by 1,000,003.
- Eight paired sets per model, using two cycles of a balanced four-treatment Williams design.
  Every arm has the same base seed within its set. Warmup is at least 10s; each selected
  scored run has at least 60s of measured request time.

The main metric is total generated tokens / total request E2E seconds, including rendering,
allocation, cold priming, generation and policy work. Model load, warmup and receipt writes
are excluded. Generation counts include reasoning and EOS tokens. The round timer encloses
snapshot, draft, verification and commit work; it is narrower than request E2E. These are
native-session measurements, not HTTP serving throughput.

## Audit and exclusions

**64 selected scored runs and 768 generation turns passed the independent audit.**
The audit checks full-head/MTP engagement, artifact/runner identities, one hardware/binary/workload
per family, token counts, elapsed sums, the 60s minimum, seed offsets, cold priming, and
native/instrumented prompt and output equality. The learner's initial/probe schedule was also
replayed from eligible rounds and checked against the recorded depth choices.

Qwen's unchanged native MTP path passed the plain-target greedy oracle at fixed K=1..8.
Both models passed short and full-workload greedy gates across all four arms. The code-workload
gate uses the same 1K output cap as scored runs. Exact commands and raw outputs are banked.

The initial Qwen short gate mixed cold priming and exact-prefix reuse according to native
commit overshoot. It diverged on turn 3 and was rejected before scoring. Holding priming
constant removed the mismatch; the fresh gates passed. The failed gate remains in the archive.

Three partial timing sets were rejected after other GPU test processes appeared. Entire
interrupted sets were retried with the original seed and order. Completed clean sets were
preserved; selection depended on completion and contamination records, never on speed.
`qwen-selected-sets.json` and `gemma-selected-sets.json` are the authoritative selection ledgers.
All excluded data and the scheduling change remain inspectable.

The earlier DFlash experiment is separate and contributes no measurements to this MTP result.

## Sources and next research

See [Qwen](QWEN-RESULTS.md), [Gemma](GEMMA-RESULTS.md), [the mechanism analysis](EXPLANATION.md),
[the causal follow-up plan](NEXT-EXPERIMENTS.md), artifact locks, selection ledgers and audit JSON.
The exact raw files are stored in 22 checksummed archives under `receipts/archives/`,
with metadata and exclusions alongside them. From this directory, unpack and audit with Python 3.12+:

```sh
python3 unpack_receipts.py --destination receipts-expanded
python3 analyze.py --family qwen --ledger qwen-selected-sets.json --receipts receipts-expanded --output qwen-audit.json
python3 analyze.py --family gemma --ledger gemma-selected-sets.json --receipts receipts-expanded --output gemma-audit.json
```

The next useful experiment is a full-head fixed-depth sweep with separate calibration and
evaluation workloads, followed by a context-aware or less frequently probing learner against
both best-fixed and native adaptive controls. Recover the original remote controller before
claiming a reproduction of it. Keep head size fixed throughout.

This publication archives the experiment without applying it to the runtime.
The exact altered runtime source is published as
[`prototype.patch`](prototype.patch), reconstructing the measured code from public base
`7326f0e176326bb9b445720068cc502ea132ffad`. The original measurement commit
`a37e30a6dd9637ece727c040f12a115dd6d96517` was local; its runtime blob hashes are
listed in [`prototype-manifest.json`](prototype-manifest.json). See [SOURCE.md](SOURCE.md).
All binary and artifact hashes remain in the receipts.
