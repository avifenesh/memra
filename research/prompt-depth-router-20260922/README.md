# Fast per-request depth routing

The [draft-only C/K/D continuation](joint-v8/VERDICT.md)
trained small controllers with the Qwen target and MTP head frozen.
On six fresh eight-turn code conversations, joint learned K/C/D
measured 140.75 tok/s versus 141.29 for fixed draft K=20/D=3/C=0
and 142.29 for the fastest measured fixed draft K=10/D=3/C=0.
It did not establish a throughput win. **In that continuation,
K means MTP draft sampler top-k and D means draft length.**
In the earlier studies below, K denoted draft depth.

On the earlier tested Qwen code corpus, keep draft depth K=3/C=0
as the research control.
The earlier fixed-positive-C sampler censored sampled picks before
target verification, so its measured rates are diagnostics. The
[separate corrected-source live-C experiment](confidence/adaptive-v3/VERDICT.md)
measured a data-derived cutoff at **−13.44%** versus K=3/C=0 on six
heldout warm native code conversations; its same-budget C=0 monitor
was −12.87%. No learning-specific throughput gain or served default
was established.

Bounded prompt-prefix depth routing also did not establish a
universal win. The [fixed-confidence verdict](confidence/VERDICT.md)
covers the earlier K=3 grid: development selected C=0.15 at +6.81%
pooled tok/s, but its code-only heldout comparison measured +1.03%,
with a 4K loss of 4.98%. Its offline replay did not beat calibrated
fixed C. The [sampled exactness audit](confidence/EXACTNESS.md)
explains why those earlier positive-C rates cannot support serving.

For the earlier Qwen3.8 prefix study, keep draft depth K=3
as the research control.
The K=4 code setting in `prefix/prefix_policy.rs` is the frozen experimental
arm from the completed study: its pooled throughput changes versus K=3
were negative at 256, 1K and 4K prompt tokens, with pointwise intervals that
include zero; the 16K comparison was also inconclusive. It is not a Qwen serving
recommendation. The [confidence protocol](confidence/PROTOCOL.md) fixes
the K=3 ceiling for its separate request shape.

The primary continuation is [`prefix/PROTOCOL.md`](prefix/PROTOCOL.md):
forecasters read only the first X user-prompt tokens, with X=64,128,256; compare
fixed K=3 against adaptation on prose and code at 256/1K/4K/16K prompt lengths.
Normal model tokenization is reused. No LLM or draft-head training is required.
The first complex-contract qualification reached no Qwen final code at either
8,192 or 12,288 output tokens. The versioned
[`simple-helper protocol`](prefix/SIMPLE-PROTOCOL.md) retains that attempt
and measures a separate six-scenario corpus. The
[scoped verdict](prefix/VERDICT.md) and [full result](prefix/RESULTS.md) show
Gemma code gains against fixed K=3 at shorter prompts, losses on Gemma prose,
and no transferable win on Qwen. No serving default is promoted.
The earlier decoder-prefix tree in [`forecast/PROTOCOL.md`](forecast/PROTOCOL.md)
is retained as a separate diagnostic.

The [forecast pilot and coverage audit](forecast/RESULTS.md) found that a shared
17-node tree costs 1.873 us p99, but the existing prefix rule is stronger overall.
The Qwen evaluation has no annotated code windows and all 12 requested-code
final answers are empty at the 512-token cap. The earlier throughput rows below
describe that capped workload; they do not settle actual prose/code adaptation.

In the earlier 512-token request-level study, fast lexical routing passed CPU
checks but the selected K did not beat calibrated fixed K on either model.

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
