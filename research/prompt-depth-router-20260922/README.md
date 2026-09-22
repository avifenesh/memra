# Fast per-request depth routing

This experiment classifies the requested output once and selects a speculative
depth for the complete request. It starts with a bounded lexical classifier:
ordinary word/phrase rules, no trained model, regex dependency, tokenizer,
allocation in the classifier, or model download.

The [CPU result](CPU-RESULTS.md) records 95/95 expected fixture outcomes and a
0.179 us median / 0.298 us p99 complete routing call on the measured Xeon CPU.
Native decoding is evaluated separately under [NATIVE-PROTOCOL.md](NATIVE-PROTOCOL.md).

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
3/3/4 on one RTX 5090. Those are candidate mappings for this experiment.
Their effectiveness as one K per request has not yet been measured.

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
