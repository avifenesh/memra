# Fast per-request depth routing

The continuation in [`forecast/PROTOCOL.md`](forecast/PROTOCOL.md) tests a shared
shallow tree that predicts upcoming output format from a bounded committed
prefix. It selects only K=2,3,4 and requires no LLM or draft-head training.
The first stage measures forecasting, transfer between models, causal inputs
and CPU cost on the archived conversations. Native throughput requires a
subsequent executed policy comparison.

The [forecast pilot and coverage audit](forecast/RESULTS.md) found that a shared
17-node tree costs 1.873 us p99, but the existing prefix rule is stronger overall.
The Qwen evaluation has no annotated code windows and all 12 requested-code
final answers are empty at the 512-token cap. The earlier throughput rows below
describe that capped workload; they do not settle actual prose/code adaptation.

Fast lexical routing passed CPU checks, but request-level K did not beat calibrated fixed K on either model.

| Model | Fixed K=3 | Prompt routing | Change |
|---|---:|---:|---:|
| Qwen3.8-27B NVFP4+Q5_K, embedded MTP | 106.225 tok/s | 105.377 tok/s | -0.798% |
| Gemma 4 12B QAT Q4_0, Q8_0 assistant | 185.064 tok/s | 184.995 tok/s | -0.037% |

These are six clean paired conversations per model on one RTX 5090, eight turns
each, approximately 16K initial input, 512-token output caps, sampled 0.7/20/0.95
and default thinking. The metric includes complete native request time and
prefix reuse. No held-out set was excluded. [NATIVE-RESULTS.md](NATIVE-RESULTS.md)
contains the controls, learned K tables, per-class rates and reconstruction scope.
This establishes no benefit over the best tested fixed depth and promotes no
serving default.

This experiment classifies the requested output once and selects a speculative
depth for the complete request. It starts with a bounded lexical classifier:
ordinary word/phrase rules, no trained model, regex dependency, tokenizer,
allocation in the classifier, or model download.

The [CPU result](CPU-RESULTS.md) records 95/95 expected fixture outcomes and a
0.179 us median / 0.298 us p99 complete routing call on the measured Xeon CPU.
Native decoding is evaluated separately under [NATIVE-PROTOCOL.md](NATIVE-PROTOCOL.md).

Source containers produced by the current builders use deterministic gzip/tar
metadata. A measured archive is verified against a digest supplied by the
separately pinned receipt manifest, never a digest defined inside that archive.
Earlier executed archives keep their exact recorded bytes and artifact identities.

The input is the **latest user instruction**, not the complete chat transcript.
Prose, code and numerical requests get separate labels. Mixed and unsupported
requests use the caller's fixed-depth fallback. Quoted text, Markdown fences,
block quotes and explicitly marked `<reference>`, `<context>` and `<document>`
blocks are input material. The initial vocabulary is English; unsupported
languages fall back. Work is bounded by 256 KiB and 512 visible words; exceeding
either limit also falls back. A long reference followed by a short instruction
can therefore be routed without scanning the reference for intent.

`router.rs` exports `select_depth(instruction, profile, ceiling)`. It returns the
kind and K without touching model state. A caller invokes it once before drafting
and retains that K; runtime admission and legal-depth ceilings still apply.
Explicit operator pins bypass the policy. Classification may run alongside GPU
prefill, but overlap is not assumed in the timing report.

The prior [span-level study](../mtp-context-depth-20260921/README.md) measured
Gemma 4 12B QAT Q4_0/Q8_0 profiles 2/4/4 and Qwen3.8-27B NVFP4+Q5_K profiles
3/3/4 on one RTX 5090. This study recalibrated all controls from the same
fixed-depth runs at its request shape. The request-level tables were Qwen
3/2/3 and Gemma 3/3/4 for prose/code/numeric, each with global K=3 fallback.
The classification rule and lookup were frozen before held-out measurement.

## Interface

The reusable interface is the Rust function; it starts no subprocess. The small
CLI is for inspection and the CPU qualification harness:

```sh
rustc --edition=2024 -O main.rs -o prompt-depth-router
printf 'Explain this Python function.' | ./prompt-depth-router classify 2,4,4,4 8
```

The four profile entries are prose, code, numerical, and fallback depth. The last
argument is the runtime's admitted/legal ceiling; zero keeps speculation off.
The example returns `{"kind":"prose","k":2}`. A profile is supplied explicitly,
so the tool embeds no model-family allowlist or universal depth recommendation.

`run_cpu.py --out <new-directory>` compiles the native executable, checks the
behavioral cases and request interface, then records five timing batches.
The workflow runs this on a hosted CPU and retains raw per-call timings,
separate process-startup measurements, compiler/source/host identity, command
logs and exact executables. Recorded timings include the measurement clock's
overhead and do not assume prefill overlap.

## Evaluation contract

- `cases.tsv` is a fixed, synthetic behavioral suite. It tests requested output,
  quoted source, negation, mixed outputs, case handling and uncertain requests.
  It is not a blind or representative estimate of real-world classification
  accuracy.
- CPU correctness and release-build timing run on hosted CI, with source, binary,
  corpus and host identity retained. No local-rig gates or benchmarks are run.
- Timing covers the complete in-process classification and depth selection.
  Input loading, stdout, and process launch are reported separately from that
  reusable call. Oversized/word-limited fallbacks are not pooled with successful
  classification latency.
- A native model comparison must include fixed K and the frozen span policy,
  equivalent prompts and sampled settings, complete-request time, routing time,
  exclusions and failures. CPU latency alone establishes no decoding speedup.
- A learned NLP classifier is justified only if this simple baseline misses
  useful routing coverage or a measured performance requirement.

Tracking: Memra issue #635. This is a research component; no serving default or
model numerical program changes.

## Native receipt reproduction

`native-receipts/` retains every calibration, correctness and held-out record,
including exact prompt/output tapes, routing decisions, span costs, timing and
binary/source bindings. The archived runtime is the prior tested engine plus
the request-driver changes; the active engine has no new dispatch door.

```sh
python3 reproduce_native.py --receipts native-receipts \
  --manifest-sha256 "$(cut -d ' ' -f 1 native-receipts/manifest.sha256)" \
  --parent-archive ../mtp-context-depth-20260921/latest-receipts/runtime-source.tar.gz \
  --out /tmp/prompt-depth-reproduction --check NATIVE-RESULTS.md
```

The output directory must be new. Replay verifies every member hash, compiles
only the externally pinned classifier source, reconstructs the full registered
matrix, refits both controls from the same calibration data, and reproduces the
report. Expanded public-boundary review is recorded in
`native-publication-review.json`; the two compressed-stream matches are bound to
exact archive hashes, and every expanded data member passed the secret scan.
