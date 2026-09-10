# Native ASR modality plan

Status: native CPU mel, encoder, cached decoder, beam-1 policy and clip window program all execute; end-to-end clip parity is measured on 36 of the 71 oracle clips (stopped clean, resumable). See the measured status ladder below. Tracking: [#414](https://github.com/avifenesh/memra/issues/414).
Engine baseline: `1657a5a80`; lane `lane/asr-modality-20260909`.

Build both native speech paths now. Memra owns model math, frontend, state and decoding.
External implementations supply pinned offline oracle data only. There is no external serving
bridge. Engine code is FSL-1.1-ALv2; checkpoint licenses and notices remain separate.
This document covers engine work. Product strategy, economics and customer records stay private.

## Scope and honest size

The newest added pack is `llama_dense`, commit `b02b135fe` (#219). Its pack, shared tensor
contract and gate manifest are the pattern. Whisper needs new modality operations, so a text
`ModelConfig` with an architecture alias is insufficient. This skeleton normalizes the original
speech config directly through `model_packs::whisper::PACK`, compiles a canonical `ModelPlan`
with a typed speech subplan and binds the shared `TensorContract`. The speech entry is not in
the text `PACKS` registry or generic `model inspect` CLI yet. Wiring that CLI and a native
reference executor is follow-up work, not a second engine.

The estimate is **16-30 agent-hours for FP16 native paths**, or **18-34 hours including optional
INT8**. The **1-2 hour Whisper pack** is one component, not an estimate for the complete audio
runtime. Correctness diagnosis and remote qualification dominate uncertainty. These are work
estimates, not latency claims or a delivery guarantee.

## Checkpoint census and provenance

### Whisper large-v3

Source: [ivrit-ai/whisper-large-v3 at 766847c9795b3b5cc0d42f8476199c711d5cee21](https://huggingface.co/ivrit-ai/whisper-large-v3/tree/766847c9795b3b5cc0d42f8476199c711d5cee21).
The checkpoint card declares Apache-2.0. Preserve its notices and base-model provenance.
The measured CT2 artifact, `ivrit-ai/whisper-large-v3-ct2@5896b9f3bc0600fd8fc133e8b161f1f76c597596`,
is a derived artifact. Its revision is not the source-weight revision. Do not transfer a CT2
quality receipt until conversion equivalence has been established tensor by tensor.

Actual range-read headers and index are checked in under
[`crates/memra-gguf/src/model_packs/whisper/fixtures`](crates/memra-gguf/src/model_packs/whisper/fixtures).
`source.lock.json` records metadata hashes, shard sizes and publisher-declared payload hashes.
The initial metadata skeleton did not download payloads. The frontend follow-up downloaded both
source shards and verified their publisher SHA256 values before the CPU encoder oracle capture.

| Census surface | Exact value |
| --- | --- |
| Physical tensors | 1,259, all F32 |
| Elements, including encoder position buffer | 1,543,490,560 |
| Source tensor payload | 6,173,962,240 bytes |
| Two shard files, including headers | 4,993,448,880 + 1,180,663,192 = 6,174,112,072 bytes |
| Prospective FP16 payload | 3,086,981,120 bytes, before cache/workspace; conversion not performed |
| Encoder / decoder | 32 / 32 layers, hidden 1280, 20 heads, head width 64, FFN 5120 |
| Positions / vocabulary | Encoder 1500, decoder 448, vocabulary 51866 |
| Mel / conv stem | 128 mel channels; `[1280,128,3]` then `[1280,1280,3]`; biases `[1280]` |
| Top-level tensors | 11: two conv weight/bias pairs, two position tables, token embedding, two final norm weight/bias pairs |
| Encoder layer | 15 tensors: Q/K/V/O weights, Q/V/O biases, two biased norms, two biased FFN linears |
| Decoder layer | 24 tensors: encoder layer pattern plus cross Q/K/V/O and cross-attention norm |
| Output head | Tied to `model.decoder.embed_tokens.weight`; no `proj_out.weight` physical tensor |

K projections have no bias. The encoder position table is stored and must be bound, even though
it originates as a sinusoid. Decoder positions are learned absolute positions; neither path uses
RoPE. The source config says `use_cache=false`; cached decoding is a planned execution rewrite
that must pass full-prefix versus incremental parity, not a different checkpoint.

Tokenizer metadata pins `<|startoftranscript|>=50258`, `<|he|>=50279`, `<|transcribe|>=50360`,
`<|notimestamps|>=50364`, EOS 50257. Preserve the byte-level BPE tokenizer, added tokens, suppression
lists, timestamp rules and generation config. No chat template is appropriate. Tokenizer IDs are
metadata-tested here; transcript decoding and the generation policy remain unqualified.

### Nemotron 3.5 streaming RNNT

Source: [nvidia/nemotron-3.5-asr-streaming-0.6b at 1c8deaecc64b91f034d73e08dd8b64625eb3395d](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/tree/1c8deaecc64b91f034d73e08dd8b64625eb3395d).
Use the `.nemo` source archive for this census. A current safetensors sibling is a separate
artifact and cannot inherit archive parity. The publisher declares OpenMDW-1.1 for the checkpoint.

The existing local archive was read as metadata, without importing torch or NeMo. Its full
SHA256 was verified as `210214ed94039bf6bfbb9a047c7fa289628db75b103e2bf6381fa78285436a74`, matching
the publisher hash; compressed size 2,368,284,501 bytes. The tar contains:

| Member | Bytes |
| --- | ---: |
| `427ad33c6285472cb01c3eb843d2309d_tokenizer.model` | 406554 |
| `c8ead90c911846569df4738620acfc0f_vocab.txt` | 78294 |
| `model_config.yaml` | 217588 |
| `model_weights.ckpt` | 2552386187 |

The checkpoint is a PyTorch ZIP inside tar. Only its first `model_weights/data.pkl` member was
parsed, with a restricted metadata unpickler accepting OrderedDict, storage descriptors and
shape-only tensor rebuilds. No tensor storage was materialized and no model code executed.
The future importer must preserve storage offsets/strides and reject executable pickle globals.
LSTM weights share flat storage; aliasing must not become duplicate or silently substituted data.

The full name/shape/storage census is
[`research/asr-modality-20260909/nemo-tensor-census.json`](research/asr-modality-20260909/nemo-tensor-census.json).
Architecture config and archive layout are beside it.

| Group | Tensors | Elements | Key shapes |
| --- | ---: | ---: | --- |
| Preprocessor buffers | 2 | 33296 | Hann `[400]`, mel filter `[1,128,257]` |
| Encoder | 636 | 609141760 | 12 subsampling tensors plus 24 blocks of 26 tensors |
| Predictor | 9 | 14940160 | Embedding `[13088,640]`; two LSTM layers, gates `[2560,640]`, two biases `[2560]` per layer |
| Joint | 6 | 9455648 | Encoder `[640,1024]`, predictor `[640,640]`, output `[13088,640]`, each with bias |
| Prompt kernel | 4 | 4459520 | `[2048,1152]`, `[2048]`, `[1024,2048]`, `[1024]` |
| Total | 657 | 638030384 | F32 storage, 2552121536 logical bytes including buffers |

Encoder: 24 layers, width 1024, 8 heads of 128, FFN width 4096, no linear biases except where
the census explicitly has them. `dw_striding` subsampling is causal, factor 8, 256 conv channels;
its projection is `[1024,4352]`. Relative attention has separate `pos_bias_u/v [8,128]` and
`linear_pos [1024,1024]`. Conformer convolution has pointwise expansion `[2048,1024,1]`, causal
depthwise `[1024,1,9]`, pointwise contraction `[1024,1024,1]`. Despite its `batch_norm` tensor name,
the declared conv norm is **LayerNorm**, not BatchNorm. Each block has two half-step FFNs,
self-attention, convolution and final normalization; preserve checkpoint order and scaling.

Prompt conditioning is a required slot: 128-way prompt representation, `he-IL` prompt index 64,
projected with the encoder row via the four prompt tensors. It is not an RNNT token prepended
by guesswork. The SentencePiece vocabulary has 13087 symbols plus blank/padding at the decoder
boundary. Pin blank index, LSTM gate order and prompt scheduling to oracle captures before use.
Fine-tuned student artifacts need their own byte manifest and census; the base archive hash is
not evidence for a successor. The private oracle lane owns that successor selection.

## Operators, reuse and missing work

Audit at this lane's baseline: `docs/MODELS.md`, `docs/KERNELS.md`, `docs/FLAGS.md`,
`docs/RELEASING.md`, the pack registry, canonical plan, tensor contract, reference executor,
`cu/` and engine sources. Existing primitive names describe reuse candidates, not ASR qualification.

| Operation | Existing Memra material | Required native work |
| --- | --- | --- |
| PCM to Whisper log-mel | Scalar/matrix utilities | STFT, periodic Hann, centered reflect padding, power spectrum, Slaney filters, log10 floor 1e-10, max-minus-8 clamp, `(x+4)/4`; 16 kHz, FFT400, hop160, 128 mel, right-pad to 30 s then drop final STFT frame |
| Whisper conv stem | GDN causal/depthwise conv as a reference for owned code | Dense channel-mixing Conv1d kernel3, padding1, strides1/2, biases and exact GELU; GDN conv is not equivalent |
| Encoder norms/FFN | `layer_norm_bias`, dense GEMM, add, activations | Typed biased LayerNorm with epsilon1e-5; **erf GELU**, not existing tanh approximation or gated FFN; ungated FFN wiring |
| Encoder attention | Bidirectional vision attention, row softmax, GEMM | Speech sequence/mask/position geometry, biased Q/V/O and bias-free K, inverse-sqrt64 scaling, full-window parity |
| Decoder | Text causal attention/KV, projections, tokenizer library | Absolute positions, biased norms/linears, self-KV plus independent fixed cross-KV, cross-attention projection/scoring, tied output |
| Whisper decode | Token masks and token selection pieces | Hebrew/task prefix, deterministic beam1, suppression, EOS/token cap, timestamp/alignment policy; no temperature fallback in the deterministic arm |
| RNNT frontend | Owned DSP primitives to be added for Whisper | Separate FFT512/hop160/window400/Hann/128-mel contract; `normalize=NA`, padding and eval dither semantics from pinned NeMo; preserve stored window/filter buffers |
| FastConformer stem | GEMM and depthwise conv patterns | Causal strided Conv2d and depthwise/pointwise subsampling, exact time/frequency padding and boundary carry |
| FastConformer blocks | Dense GEMM, biased norms, SiLU, add, softmax | Half residual FFNs, GLU convolution module, relative-position score/shift, limited-context chunk masks; LayerNorm conv semantics |
| Streaming caches | Existing KV/state allocation patterns | Attention last-channel cache, causal convolution last-time cache, cache lengths, subsampler/frontend carry, position offset; left and right context accounting |
| Predictor and joint | Embedding, GEMM, add, ReLU primitives | Two-layer LSTM hidden/cell state, input/forget/cell/output gate mapping, joint ReLU and blank transition, bounded symbols per acoustic step |
| RNNT decoding/prompt | Token selection and session patterns | Greedy first, beam hypotheses later if required, blank/emit transition state, prompt kernel with `he-IL`, prompt injection cadence, final flush and reset |
| Audio sessions | Native scheduler/admission framework | Sample-indexed audio API, bounded queues, cancellation, provisional/final revisions, two independent model state lifetimes and c1/c4 isolation |

No new runtime flag is introduced by this skeleton. All speech operations block every existing
execution rewrite. `NativeReference` remains unset. The reference executor explicitly refuses
speech plans rather than interpreting empty text layers as a working model. Existing text plan
Debug serialization is preserved so adding `speech=None` does not invalidate their plan hashes.

## Runtime contract

Input is mono 16 kHz PCM in monotonic, sample-indexed **80 ms frames**. RNNT uses the supported
`[56,0]` attention-context arm, not the archive's first `[56,3]` default. Preserve left-context
attention and convolution state across chunks, track valid lengths, and never expose right-context
samples before they arrive. The schema must also express right-context buffering for `[56,3]`,
`[56,6]` and `[56,13]` without admitting them under the 80 ms qualification. Frame cadence is not
an 80 ms word-latency claim. Persist LSTM state across emitted symbols; blanks advance acoustic
time without incorrectly advancing predictor state. Bound maximum symbols per frame.

Whisper receives a completed bounded waveform at an utterance boundary. Recompute its full
bidirectional encoder for each waveform. Cache decoder self-attention within that decode and
encoder cross-attention K/V only for that same waveform. The initial arm uses Hebrew transcription,
no previous-text conditioning and no RNNT text injection. A 12 s owned-utterance cap, received
context and audio ownership remain explicit; no offline future audio or full-clip hindsight rewrite.

Reserve a native causal endpoint interface: 20 ms VAD frames, 300 ms trailing non-speech,
200 ms pre-roll/trailing received context, 400 ms left context at forced boundaries. Resolve the
DSP implementation under Memra's owned-code rule; do not link an external inference library.
The next utterance's RNNT must run while the previous Whisper finalizes. Requests carry stream,
segment, revision and owned half-open sample interval; immutable finals replace only their own
provisional segment. Reset, cancellation, reconnect and errors must not leak or duplicate state.

Deterministic beam1 is the ASR-specific decode contract requested for this lane. Qualify omitted
sampling/decoding parameters against it, including bounded no-speech behavior. This is scoped to
ASR and does not change sampled defaults for text models. Timestamp mode is a separate pinned
oracle arm; text-only no-timestamp output cannot establish word-boundary alignment correctness.

## Oracle plan and qualification ladder

The private oracle lane produces offline captures over all **71 clips**, with PCM hashes,
original sample boundaries, tokenizer/scorer hashes, exact source and derived artifact revisions,
conversion command/version, backend versions, decoding settings and output hashes. This Memra
lane consumes pinned files. No faster-whisper or NeMo dependency enters the runtime.

1. Freeze source mapping first. Compare source F32 tensors, explicit F32-to-FP16 conversion and
   the measured CT2 artifact, including reordered/packed tensors and tied output. Existing CT2
   metrics do not establish this mapping. Freeze NeMo base/student manifests separately.
2. Capture faster-whisper FP16 **beam1**, temperature zero, Hebrew transcription: mel values,
   conv/encoder rows, decoder step logits and token IDs, self/cross KV snapshots, raw UTF-8 text,
   suppression/timestamp state and final output. Freeze the actual oracle versions used by the
   private lane; `semantic-sources.json` is source-reading provenance, not an execution receipt.
3. Capture NeMo greedy streaming at `[56,0]`: per-chunk frontend/subsampler rows, encoder output,
   last-channel/last-time/length caches, prompt rows, predictor hidden/cell, joint logits, blank
   and token transitions, partial transcripts, reset and end-of-stream flush. Include uneven
   final chunks, zero/silence, long continuation and interleaved sessions.
4. Require exact token IDs and output UTF-8 bytes for the matched beam1/greedy program on all
   71 clips. Intermediate floats use a predeclared stage-specific numerical bound, with argmax
   agreement and first-divergence logging. Cross-engine float bit identity is not assumed.
   Optimized Memra rewrites require byte identity within the declared native numeric class.
   A mismatch is a diagnostic/fix gate, never rounded into a pass by WER similarity.

| Ladder stage | Acceptance and persisted evidence |
| --- | --- |
| This skeleton | Config/frontend geometry, 1259 source bindings, shapes/dtypes/extents/index integrity, source hashes, wrong/missing/extra/duplicate rejection; execution unsupported |
| Config / tensor / tokenizer gates | Full payload hashes and archive manifest; no missing/unexpected/ambiguous tensors; Hebrew/task/blank/prompt/timestamp IDs, tokenizer bytes and generation policy |
| Tiny native parity | Owned native DSP, conv, norm, exact GELU, encoder/cross-attention, LSTM/joint primitive fixtures; cache and masking boundary cases |
| NativeReference | Actual complete speech forward/decode executes in Memra's unfused native executor, with persisted tiny/stage parity. Loading metadata alone does not qualify |
| Checkpoint parity | All pinned clip tokens/text match offline oracle; staged floats and caches within fixed bounds; FP16 conversion validated; trained successor independently pinned |
| NativeQualified | Checkpoint plus full audio endpoint gates: ready/model ID, default decoder, chunk pacing, c1 isolation, c4 separately measured admission, concurrency/reset/cancel/reconnect/flush, capacity/context limits and rollback; bind artifact/plan/numeric stream/binary/bundle |
| NativeTuned | Qualified plus current binary-bound rewrite receipts for each admitted device and execution surface; phase profiles and actual end-to-end measurements |

The integrated acceptance battery records first partial p95 <=500 ms, no >1 s continuing-speech
update stalls, finalization p50 <=800 ms and p95 <1500 ms per domain, with failed/missing finals
counted. Timing starts from capture of the last owned speech sample, not decoder submission.
Separate endpoint, frontend, encoder, decoder, queue and delivery time; record cold/warm and
concurrency. The private product lane owns quality thresholds, scoring and release claims.

## Work breakdown and next executable step

| Work item | Agent-hours | Exit |
| --- | ---: | --- |
| Whisper pack, source metadata and tensor/tokenizer contract | 1-2 | This skeleton plus payload provenance and onboarding CLI wiring |
| Native frontend and Whisper encoder | 3-5 | Mel, conv, norm, GELU, position and bidirectional encoder parity |
| Whisper cross-attention decoder and ASR decode | 3-5 | Cached/uncached step parity and deterministic output |
| FastConformer/RNNT pack, prompt and streaming state | 4-8 | Native `[56,0]` chunk/flush parity against NeMo |
| Native audio API, scheduling and session protocol | 2-4 | Concurrent partial/final lifecycle and isolation |
| Remote qualification and one diagnostic/fix cycle | 3-6 | All 71 clips and bound endpoint receipts |
| FP16 subtotal | **16-30** | Both native paths; no INT8 prerequisite |
| Optional CT2-equivalent INT8 | 2-4 | Exact scale/rounding contract, independent parity/quality/performance gates |
| Total with optional INT8 | **18-34** | Estimate, not measured elapsed work |

**Next executable step:** implement the owned Whisper PCM-to-log-mel reference operation and
strided conv stem, then compare their intermediate outputs to the oracle lane's pinned fixtures.
In parallel work scheduling, RNNT metadata/import work is independent; no second owner is claimed
by this bounded assignment. Add the speech inspector CLI before checkpoint execution. Keep FP16
weights with explicit accumulation/norm precision; only introduce INT8 after a separately
receipted quality and latency win. Generic Q8_0 is not CT2 per-row INT8.

No rental is needed for this skeleton. All rig work is CPU-only and niced. No local cargo tests, runtime gates, CI,
smoke servers or inference were run. Required text-only pre-push checks ran under nice. Push uses `MEMRA_SKIP_PERF_CI=1`; hosted CI is required.
Later GPU work must use an explicitly nonproduction lane-owned rental, validate CUDA allocation,
keep useful work queued, and destroy it with inventory readback at lane completion. Existing
production and training hosts are outside this lane. A release follows `docs/RELEASING.md` after
qualification; this draft is not a fleet rollout or a support declaration.

## Sources and evidence limits

- [Pinned Whisper source files](https://huggingface.co/ivrit-ai/whisper-large-v3/tree/766847c9795b3b5cc0d42f8476199c711d5cee21), checked-in metadata and publisher hash declarations.
- [Pinned NeMo archive](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/blob/1c8deaecc64b91f034d73e08dd8b64625eb3395d/nemotron-3.5-asr-streaming-0.6b.nemo), locally verified SHA256 and restricted tensor census.
- [OpenAI Whisper audio/model sources](https://github.com/openai/whisper), [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [NeMo](https://github.com/NVIDIA/NeMo): exact reading revisions in `research/asr-modality-20260909/semantic-sources.json`. Reading does not license vendoring kernels.
- Private requirements read at Darklanes `ae42664bed248cd91124a9cf76829637734b0a2f`, sections 1/2/3/6 of the two-tier program, and its model-onboarding checklist. No private corpus or customer content is copied here.

## Measured status ladder, 2026-09-09

Where the two paths actually stand after the decode stage. Every "done" row below is bound to
a receipt in `research/asr-modality-20260909/`; every "missing" row is work, not a formality.

### Whisper large-v3, Hebrew transcription

| Piece | State |
| --- | --- |
| Log-mel, padded single-window | Done. 5.6743622e-5 vs HF FP32, bound 1e-3 |
| Log-mel, real clip program | Done. 364 of 364 real 30-second windows at 1.1920928955078125e-7, bound 1e-3 |
| Conv stem and 32 encoder blocks | Done in F32 and strict F16. 0.0002231598 vs HF F32; F16 at the measured HF F16 floor |
| Cached decoder, self and cross KV | Done in F32 and strict F16, argmax 12 of 12 against the FP32 truth |
| Window program (seek, extent, advance) | Done. Reproduces all 364 pinned window boundaries across 71 clips |
| Beam-1 policy (suppression, timestamps, forced timestamp, caps) | Done. Replays the oracle's own decodes step for step on its raw logits |
| End-to-end clip parity vs CT2 beam-1 | 36 of 71 clips swept (stopped clean, resumable): 213/259 windows token-exact, 27/36 clips text-exact, `d1` 0.0739 pt and `whatsapp` 0.2237 pt WER vs CT2, both inside the 0.21-0.34 pt band the oracle's own backends disagree by; every divergent window diagnosed as a CT2 fp16 threshold effect, none a native defect |
| Native detokenizer | Done. Byte-level BPE through `Detokenizer` and the RNNT SentencePiece reader; the engine's transcript agrees with the scorer's tokenizer on 36/36 clips |
| Tokenizer and vocabulary bound in the engine | **Encode direction missing.** Ids are pinned as plan constants and gated against the checkpoint's generation config; the BPE decode direction is native (`Detokenizer`), but no byte-level BPE encode lives in Memra yet |
| Timestamp and word-alignment gate | **Missing.** Parity is token-level; a timestamp that lands one unit off is currently only visible as a token difference |
| No-speech and temperature fallback | **Missing by design in this arm.** The deterministic beam-1 program has no fallback; a product arm needs its own pinned policy |
| Speech plans in the reference executor and `model inspect` | **Missing.** The executor still refuses speech plans; the speech pack is not in the text `PACKS` registry |
| F16 end-to-end sweep | **Missing.** The 71-clip sweep runs F32; the F16 arm has stage receipts but no clip parity |

`NativeReference` needs the remaining 35 sweep clips, the tokenizer's encode direction and
vocabulary binding inside the engine (the detokenize direction is done and agrees 36/36), and
the speech plans wired into the reference executor and `model inspect`. Everything above them
is bring-up evidence, not production permission.

`NativeQualified` additionally needs, none of it started:

- A GPU execution path. Every number in this lane is a CPU reference measured on efficiency
  cores. There is no CUDA kernel for the speech operations, no residency plan and no
  performance receipt, and the CPU reference is not evidence for any of them.
- An audio endpoint shape in `memra serve`: readiness, model id, the request/response contract
  for a bounded utterance, streaming partials, concurrency, admission and context limits,
  cancellation, reconnect and rollback, each with a receipt bound to one binary and one plan.
- The integrated acceptance battery in the runtime contract section above, timed from the last
  owned speech sample.
- Optional INT8. CT2 serves per-row INT8; generic Q8_0 is not that. It stays optional until a
  separately receipted quality and latency win exists, and it is not a prerequisite for FP16.

### Nemotron 3.5 streaming RNNT, Hebrew successor

The whole path executes: `rnnt-stage stream` takes PCM and returns token ids for the `[56, 0]`
arm, matching the reference session chunk by chunk. Receipts in `RNNT-ENCODER.md`.

| Piece | State |
| --- | --- |
| Archive layout, census and contract bind | Done on the real `clean-step-21959.nemo`: 657 tensors, 638,030,384 elements, nothing missing or extra, no aliased storage |
| `[56, 0]` streaming state contract | Done: 1,575,424 state elements per session, causal, driver constants measured, other arms expressible and not admitted |
| NeMo oracle captures | Done on the rig: encoder, head and whole-session references for a real 2-second Hebrew clip, CPU FP32, dither off, one thread |
| Frontend | Done. 5.34058e-05 against the reference, bound 1e-3 |
| FastConformer subsampler and 24 blocks, cache-aware | Done. 26 of 26 chunks at 2.533197403e-07, and 2.086162567e-07 when fed the native mel |
| Prompt kernel, predictor, joint | Done. 4.768371582e-07, 7.748603821e-07, 9.155273438e-05 |
| Greedy RNNT | Done. 9 of 9 token ids identical |
| Streaming session | Done. 26 of 26 chunk partials identical, final identical |
| Shared `TensorContract` and `ModelPlan` integration | **Missing on purpose.** Needs a torch-zip `CheckpointDialect`, which touches every text builder that matches on the dialect |
| Tokenizer and detokenizer in the engine | Detokenizer done (archive SentencePiece, stage 10); tokenizer encode missing |
| More than one clip and one arm | Three clips and silence (2 s, 15.8 s, `d1-000` 15 s, digital silence), all exact; still one arm, no corpus, no long continuation, no interleaved sessions |
| Session lifecycle beyond a single run | **Missing.** No revision semantics, cancellation, reset, reconnect, concurrency or admission |
| GPU execution | **Missing.** Every number is a CPU reference |

## Stage 1 receipt, 2026-09-09

The native CPU frontend passed the 2-second synthetic waveform gate, using the normal 30-second
padded window: `[128,3000]`, max absolute delta **5.6743622e-5** against HF transformers 5.16.1 /
torch 2.14.0 FP32, threshold **1e-3**. Receipt: `research/asr-modality-20260909/stage1-mel.json`.
The source is a Memra-owned direct DFT, periodic Hann and Slaney log-mel implementation, with
no external DSP dependency. The synthetic fixture is checked into the CUDA-free reference crate
so hosted CI repeats the numerical assertion. It is not a WER or GPU performance receipt.

The owner explicitly allowed this follow-up's tiny, niced, single-process CPU oracle and stage
gates on the rig. No GPU was opened, and no local workspace CI or general battery was run.
`tools/whisper_cpu_oracle.py` is offline capture tooling only. The stage runner consumes binary
fixtures and source safetensors; it never calls that tool as an execution backend.

## Stage 2 encoder receipt, 2026-09-09

The native reference now executes both strided convolutions and every encoder block in F32
or strict F16 arithmetic (FP32 accumulators, binary16 weights/activation values). The CPU
reference stores half values in f32 carriers; this is not a GPU residency or speed claim.
F32 max abs **0.0002231598** against HF F32 passes 1e-3. Native F16 max abs **0.25** against
HF F16 passes the owner-amended, same-fixture HF F16-vs-F32 floor **0.2771682739**.
See `research/asr-modality-20260909/ENCODER-NUMERICS.md` and its raw JSON receipts.
The original 1e-2 F16-vs-F32 gate and all failed diagnostics remain explicitly recorded.
Native support stays unset until the remaining decoder/policy and qualification gates pass.
