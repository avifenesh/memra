# Speech program

What it takes for memra to serve any speech model, and how a new one gets onboarded.

This file owns the *program*: model families in scope, what "support" means for a speech
model on the same support-state ladder the text models use, the repeatable onboarding path,
the gate set, and the performance thesis with its kill criteria. The lane-level bring-up
record for the first two checkpoints lives in [ASR-MODALITY-PLAN.md](../ASR-MODALITY-PLAN.md)
and its receipts in `research/asr-modality-20260909/`; this file does not restate them.

Engine only. Fleet placement, pricing, customer exposure and published facts are private and
are not decided here.

---

## 1. Where the code actually is, 2026-09-10

Honest starting position, because a program that starts from a flattering framing is worth
less than one that starts from a measured deficit.

| Path | State | Evidence |
| --- | --- | --- |
| Whisper large-v3, Hebrew transcription | Frontend, conv stem, 32 encoder blocks, cached decoder, beam-1 policy and clip window program all execute in the CPU reference. End-to-end clip parity **missed its gate** and landed on a diagnosis | log-mel 5.6743622e-5 (bound 1e-3); encoder F32 0.0002231598; 36 of 71 clips swept, 213/259 windows token-exact, `d1` 0.0739 pt / `whatsapp` 0.2237 pt WER vs CT2 against a 0.05 pt limit |
| Nemotron 3.5 streaming RNNT `[56,0]` | The whole path executes and streams: frontend, cache-aware FastConformer, prompt kernel, LSTM predictor, joint, greedy, session state | 657/657 tensors, frontend 5.34058e-05, encoder 2.533197403e-07 over 26 chunks, head 9.155273438e-05, greedy 9/9, session 26/26 partials plus final identical |
| Audio-LM class (Qwen3-ASR, Voxtral) | Not started | n/a |
| **GPU execution for any speech operation** | **Does not exist.** No CUDA kernel, no residency plan, no performance receipt | Every number above is a CPU reference measured on efficiency cores |
| **Audio endpoint in `memra serve`** | **Exists as of 2026-09-11 and fails CLOSED.** `/v1/audio/transcriptions` plus the `/v1/audio/sessions` lifecycle (open / frames / close / list) carry the session contract, the resident-session cap, the per-lane bounded queue, the typed shed taxonomy and the declared decode; with no speech pack resident every path refuses `engine_unbound` rather than answering with something that is not the model. **The transcript bytes are step 2's work** | `audio_api.rs`, 18 handler + contract tests; scheduler in `memra-lanes::audio_stream`, 20 tests |

So: memra has a correctness spine for two speech families, a serving surface that admits and
sheds honestly, and **no speed**. On the metric this program is about, memra has not entered the
race: there is still no GPU number, and an endpoint that refuses is not a product. What the
spine buys is that entering is cheap and checkable, every GPU kernel written from here has
a banked CPU reference to be byte-identical against, which is the expensive part of a speech
bring-up and it is already paid for.

`NativeReference` is unset for every speech model. Nothing here is production permission.

---

## 2. Families in scope

Three architecture classes, chosen because they cover the field and because each one reuses a
different part of the engine we already have.

### 2.1 Encoder-decoder AED (Whisper family)

Bidirectional encoder over a fixed padded window, autoregressive decoder with cross-attention.
`whisper-large-v3`, `large-v3-turbo`, `distil-*`, and any Hebrew or other adaptation of them.

Engine shape: full encoder recompute per window; decoder holds a self-attention KV cache **and
an independent fixed cross-attention K/V** for that window. The two cache lifetimes are the
risky part and they are why this family is not the streaming answer: the encoder is not causal,
so streaming it means either re-encoding a growing buffer or bolting causality on after the
fact. Post-hoc causalization of an offline AED encoder, even repaired by self-distillation,
went 7.6 to 17.6 WER with p50 word-commit 4.1 s
([qfuxa/qwen3-asr-0.6b-streaming](https://huggingface.co/qfuxa/qwen3-asr-0.6b-streaming)).
Treat this family as the **batch/utterance** path, not the streaming path.

### 2.2 Cache-aware streaming transducer (FastConformer / RNNT / TDT)

Encoder trained with a fixed left/right attention context so activation caches make a
non-autoregressive encoder run autoregressively; every frame is processed exactly once.
`nemotron-3.5-asr-streaming-0.6b`, `nemotron-speech-streaming-en-0.6b`, `parakeet-tdt-*`,
Zipformer transducers.

Engine shape: attention last-channel cache, causal-convolution last-time cache, cache lengths,
subsampler and frontend carry, position offset, plus a two-layer LSTM predictor hypothesis and
a joint with a blank transition. This is the family memra streams natively today.

Licence discipline for this family, learned the expensive way: `nemotron-3.5-asr-streaming-0.6b`
is OpenMDW-1.1 and `nemotron-speech-streaming-en-0.6b` is the NVIDIA Open Model License. **Never
infer a licence across a family**; read the card of the exact artifact.

### 2.3 Audio-LM (Qwen3-ASR, Voxtral Realtime)

An audio encoder plus a projector into an ordinary text decoder. `Qwen3-ASR-1.7B` is
Qwen3-1.7B + projector + a 300M AuT encoder with a dynamic 1 s to 8 s attention window;
`Voxtral-Mini-4B-Realtime` is a 970M causal sliding-window encoder + 25M adapter + a 3.4B
Ministral decoder with Ada-RMSNorm delay conditioning.

Engine shape: the decoder is a first-class native text path already, and the
projector-prefix shape is the one the **vision** path already implements. There is no
cross-attention cache to manage, which removes the Whisper family's worst streaming hazard.
New surface: the audio encoder itself, its window/cache program, and, for Voxtral, a new
norm variant (`NormKind` is RMS only today) plus a sinusoidal-embedding MLP injected into the
decoder FFN.

**Out of scope until something in scope is qualified:** TTS, speech translation, diarization,
speaker ID, and audio understanding beyond transcription. Naming them is not a roadmap.

---

## 3. Support states for a speech model

Same ladder as the text models. Loading and running are not support.

**`NativeReference`**, the complete speech forward and decode execute in memra's unfused
native executor, with persisted stage parity beneath them. Requires: the pack in the registry,
the plan compiled through the shared `TensorContract`, the tokenizer bound in **both**
directions inside the engine, speech plans accepted by the reference executor and by
`model inspect`. Bring-up evidence only. **Not production permission.**

**`NativeQualified`**, `NativeReference` plus checkpoint parity on the full pinned clip set,
plus the serving battery of §5 bound to one artifact, one plan, one numeric stream, one binary
and one bundle hash. Minimum state for production admission.

**`NativeTuned`**, `NativeQualified` plus current binary-bound rewrite receipts for each
admitted device and execution surface, with phase profiles and end-to-end measurements.

A speech model at `NativeReference` on CPU is exactly where both current checkpoints are, and
it is why no speech model may be served.

---

## 4. Onboarding a new speech model

Seven steps, in order. This is a path, not a per-model adventure: each step has a fixed
artifact and a fixed refusal. Skipping one is how a model gets served fluently and wrongly.

**0. Pin and licence.** Exact repository revision (never a mutable family tag), full payload
SHA256 verified locally against the publisher's declaration, archive layout recorded, licence
read from the artifact's own card. A finetuned successor gets its own manifest and census; the
base archive hash is not evidence for it.

**1. Tensor census.** Count, element total, dtypes, shapes, storage offsets and strides against
the real archive. Reject missing, extra, duplicate and silently-aliased storage, LSTM gate
weights share flat storage and aliasing must not become substituted data. Metadata parsing only:
a restricted unpickler for torch archives, no model code executed, no tensor storage
materialized.

**2. Frontend contract.** Sample rate, FFT size, hop, window shape and periodicity, mel bank
flavour (Slaney vs HTK), filter count, log floor and clamp, normalization, dither, and padding
semantics. **Never inherited from a sibling model**: Whisper is FFT400/hop160/128-mel with a
`(x+4)/4` scaling and a 30 s right pad; the Nemotron RNNT is FFT512/hop160/window400 with
`normalize=NA` and its own stored window and filter buffers. Two speech models in the same
engine already disagree on this.

**3. Tokenizer, vocabulary and decode policy.** Special ids pinned as plan constants **and**
gated against the checkpoint's own generation config. Both directions bound in the engine:
encode and detokenize. Suppression lists, timestamp rules, blank index, prompt slots and
language/task ids. No chat template is appropriate for an ASR model; an audio-LM's text decoder
still has one and it must come from the checkpoint, never from a fallback.

**4. Streaming state contract**, if the model streams. Cache shapes and total element count per
session, left and right context accounting, and which `att_context_size`-style arms are
*expressible* versus which are *admitted*. Only the admitted arm is qualified. Right-context
samples are never exposed before they arrive.

**5. Stage parity against a pinned oracle.** Frontend, then encoder, then head or decoder, then
the decode policy, each with its own predeclared numeric bound and a first-divergence log.
Bounds are measured against truth, not typed: the F16 encoder bound is the same-fixture
HF-F16-vs-F32 floor, not a round number. External implementations supply pinned offline fixture
data only; no external inference library enters the runtime or sits behind a served endpoint.

**6. End-to-end and serve.** Checkpoint parity on the full pinned clip set (exact token ids and
output UTF-8 bytes for the matched deterministic program), then the serving battery of §5. A
mismatch is a diagnostic gate; it is never rounded into a pass by WER similarity.

---

## 5. The gate set

A gate that has never failed proves nothing. Every row below names its caller, its trigger, the
input set that makes it non-vacuous, and a red arm that **has actually fired**. Rows whose red
arm has not fired yet are marked, and they are not counted as protection.

| Gate | Caller / trigger | Non-vacuous input set | Red arm that fired |
| --- | --- | --- | --- |
| **G1 census** | `model inspect` on the speech pack; hosted CI on the checked-in fixtures | The real archive: 1,259 Whisper tensors / 657 RNNT tensors | Wrong, missing, extra and duplicate tensors are all rejected paths in the skeleton; the RNNT aliased-storage case is explicitly tested |
| **G2 frontend** | speech stage runner; hosted CI on the checked-in binary fixture | Synthetic 2 s waveform **and** 364 real 30 s windows across 71 clips | Yes. Successive frontend diagnostics are recorded in `research/asr-modality-20260909/`; the 1e-3 bound is measured against HF FP32, not asserted |
| **G3 encoder numerics** | stage runner, F32 and strict F16 arms | Same fixtures, both arithmetic classes | **Yes, this one failed and was re-derived.** The original 1e-2 F16-vs-F32 gate FAILED; the bound is now the same-fixture HF F16-vs-F32 floor 0.2771682739, and every failed diagnostic stays recorded |
| **G4 decoder / head / greedy** | stage runner; session replay | 12 decoder steps argmax; RNNT 9/9 tokens, 26/26 chunk partials | Yes. 46 divergent windows in the 36-clip sweep, each traced to a cause, 31 of them measured CT2 fp16 precision effects |
| **G5 tokenizer** | plan compile; generation-config cross-check | Checkpoint generation config, added tokens, suppression list | Partly. Detokenize agrees with the scorer's tokenizer 36/36. **Encode direction is missing, so this gate is currently half-vacuous and is named as such** |
| **G6 checkpoint parity** | full clip sweep vs pinned oracle | 71 clips, two domains, per-domain WER delta bound 0.05 pt | **Yes, RED today.** 0.0739 `d1` and 0.2237 `whatsapp` against the 0.05 pt limit on the 36-clip partial sweep |
| **G7 serve decode contract** | registry half: the acceptance gate a candidate passes before it gets traffic, on any entry declaring `task = "transcription"`. Live half: post-deploy probe against the served binary | Both registry shapes we would serve (streaming RNNT, batch Whisper), every shipped registry snapshot, and the pinned RNNT beam decoder itself. Live half: the exact default decode shape, submitted with no decoding parameters, twice | **Registry half: yes, and its red arms are the configs a person actually writes**: an entry that declares nothing (and so inherits faster-whisper's `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]` ladder), a text model's vendor-sampling stanza copied across, and beam on a streaming partial path. **Registry half, engine side: yes, and it now refuses at LOAD.** `validate_asr_decode_contract` (memra-server) rejects a silent transcription entry, a text model's sampling stanza copied across, beam on a partial-emitting path, a ladder that starts hot or is non-monotonic, a determinism flag that contradicts its ladder, a beam width that does not match its strategy, and decode keys on a non-ASR row — 10 tests, each named after the stanza a person writes, with the two shapes we would actually serve asserted PASSING so the gate is not a wall. **Live half: ARMABLE but NOT ARMED**: the endpoint exists (§1) and reports the resolved decode on `x-memra-asr-decode` even on its refusal, but two byte-identical no-parameter transcriptions cannot be compared until an engine is bound. It reports `not-armed`. See §5.1 |
| **G8 streams@SLO battery** | darklanes `ops/serving/asr_streams_slo.py` against a served endpoint on a lane-owned box; the scheduler half is `cargo test -p memra-lanes --lib audio_stream` in hosted CI | Readiness, model id, streaming partial/final/revision shape, concurrency ladder c1 → c4 → c16 → c64, admission limit, cancel, reconnect, flush, rollback. Minimum duration per rung, derived not typed: `queue_frames / (1000/frame_ms - served_rate_hz/streams)`, i.e. the PER-LANE bleed (the aggregate form gives 2.56 s for the measured c4 geometry whose real answer is 10.24 s, and would bless exactly the rung length that misses the defect) | **Yes, and it is the best red arm we own — and half of it now runs with no GPU.** On the card: one A100 holding a resident RNNT student plus a resident large-v3 **shed 54 of 71 streams on incoming-queue overflow at c4** while c1 ran fine. In CI: `MEMRA_AUDIO_DRIVE=per_lane_worker` drives the SAME `AudioScheduler` the endpoint drives, at the measured step cost (31.522 ms at batch 1, 35.890 ms at batch 32), and refuses 4 of 4 streams at c4 with every refusal a typed `queue_overflow` pinned at the cap, while `fused` holds 64 of 64 at max queue depth ≤ 3. A THIRD red arm came out of writing it: a 10 s probe of the broken shape refuses NOTHING while its backlog is already 48 of 64, so a short ladder rung passes the defect — which is why the duration is derived above. **The GPU half of G8 is still unrun: the CI half bounds scheduling logic and makes no claim about milliseconds** |
| **G9 non-vacuity** | every numeric gate | n/a | Each numeric gate carries a first-divergence log and refuses an empty input set; a sweep that scores zero clips is a failure, not a pass |

Two rules that bind all of them:

- **Timing starts from capture of the last owned speech sample**, not from decoder submission.
  Endpoint, frontend, encoder, decoder, queue and delivery time are separated, and cold versus
  warm and each concurrency level are recorded separately.
- **Admission must shed with a typed error, not drop.** The c4 collapse was a bounded-queue
  overflow that lost streams. A capacity limit that is reached is a product behaviour; a
  capacity limit that silently eats streams is a defect.

### 5.1 G7, the served decode contract

G7 was recorded NOT BUILT above because a decode default is a product decision. It has
been made and recorded; the decision and its evidence are private, the contract it puts on
this engine is not.

**What G7 asserts.** A served ASR endpoint decodes **deterministically, by written
contract**, in a way that matches the decode its WER was measured on, and it **says which
decode that is** rather than being silent about it.

That is not a departure from the serving discipline the text models are held to. The rule
there is *serve the vendor recommendation and verify the shape a client actually sends*;
the reason it reads as "serve sampled" is that for text generation the vendor
recommendation happens to be sampled. For speech it happens to be deterministic, in every
family in §2:

- `openai/whisper` v20250625: `DecodingOptions.temperature: float = 0.0`. Called as a
  library it is greedy (`beam_size: Optional[int] = None`); the CLI ships
  `--beam_size` default `5`. Both deterministic.
- `faster-whisper` 1.2.1 / CTranslate2 4.8.2: `beam_size: int = 5` at temperature 0, and
  `temperature > 0` is a **different code path** (`sampling_topk: 0`,
  `sampling_temperature: t`), so temperature 0 means no sampling arguments exist at all.
- NeMo: `RNNTDecodingConfig.strategy: str = "greedy_batch"`, and
  `nemotron-3.5-asr-streaming-0.6b`'s own `model_config.yaml` ships
  `decoding.strategy: greedy_batch`. The only `temperature` in the RNNT decode surface is
  `softmax_temperature`, which sharpens beam scoring; it draws no sample.

**The temperature-fallback ladder is quality-failure recovery, not a decode default.** Its
first rung is always 0.0, and a hotter rung is reached only when the previous one failed a
quality test (`compression_ratio > 2.4`, `avg_logprob < -1.0`); a high no-speech
probability *cancels* fallback rather than triggering it. An endpoint may serve the ladder,
but only declared exactly, and only if the response reports the temperature it used.

**Beam is not free for a streaming partial path, and it fails two different ways.** On
`nemo_toolkit 3.1.0+ea1ebf55b` (`rnnt_beam_decoding.py` sha256
`6e5a8b190b338717865db2f23d87e28ba5be9a0b1f1559dab1f9c885b4c27e53`), `tsd` (L744), `alsd`
(L907), `maes` (L1139) and the batched beam wrapper (L1704) each raise
`NotImplementedError` on a partial hypothesis, and `nsc` is not wired. But
`default_beam_search` (L591) **accepts** partials and collapses the beam to do it: L624-628
rebuild `kept_hyps` as a one-element list seeded from `partial_hypotheses.y_sequence[-1]`
and `dec_state`, score reset to `0.0`, no alternative carried. Served width across a chunk
boundary is 1 whatever `beam_size` says, so the offline beam WER does not transfer. A
bring-up that knows only the first failure and reaches for `default` finds that it runs.

**The contract on this engine.** A speech model's registry entry must declare, explicitly:
whether the decode is deterministic; the strategy by name, with a beam width if it is a
beam family; the fallback ladder exactly, or `[]` to disable it, with its trigger
thresholds if non-empty; and whether the entry serves streaming partials. No stochastic
sampling key may resolve into a transcription request. Silence is a violation, not a
default. And a transcription response must report the temperature that produced it:
without that field the contract cannot be checked from the outside at all, which is the
same reason HTTP 200 is not a receipt for the text models.

**Where the gate lives.** The rule set and the red/green suite are in the private
repository, wired into the acceptance gate that a candidate passes before it gets traffic
and into hosted CI. The engine's obligations are the two above: carry the registry keys
through to the resolved decode, and report the temperature actually used per segment.

**Where G7 does not protect yet.** The live half (two identical no-parameter transcriptions of
one clip must return byte-identical text) is written and fixture-tested and **still cannot
fire**. The reason changed on 2026-09-11: the endpoint now exists and resolves the declared
decode — it reports it on an `x-memra-asr-decode` response header even on its own refusal, so an
operator can see WHICH decode a registry row resolved to — but with no engine bound there are no
two transcripts to compare. It reports `not-armed` rather than passing. Same treatment as G5's
missing encode direction, for the same reason.

**What the engine now refuses on its own.** `validate_asr_decode_contract` runs inside
`validate_openrouter_metadata`, i.e. at every metadata load, boot and hot reload. A gate can only
check the file it was pointed at; the binary refuses a silent or self-contradictory transcription
entry whatever produced the file. The refusals, each a stanza a person writes: an entry that
declares nothing (and the refusal quotes the `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]` ladder silence
inherits), any `default_*` sampling key or `non_thinking_sampling` table on a transcription row,
a strategy that is not served (`maes`/`tsd`/`alsd`/`nsc` among them — declaring one would look
configured and fail at runtime), a beam family with no width or a width with no beam family, beam
on a `serves_partials = true` row (with the `kept_hyps` collapse cited in the refusal), a ladder
that starts hot or does not strictly increase, `decode_deterministic = true` under a hotter rung,
`decode_deterministic = false` with no hotter rung, and any decode key on a non-ASR surface.

---

## 6. Performance thesis

### 6.1 The headline metric

**Concurrent real-time streams per GPU at a fixed end-of-speech-to-final p95, on an
accuracy-tier model (>= ~1B parameters).** Short form: **streams@SLO**. The derived number a
buyer actually reads is `box $/hr ÷ streams@SLO` = cost per concurrent stream-hour.

Why this and not the alternatives:

- **Not cost per audio-hour for batch.** That fight is over and nobody won it, because it
  stopped mattering. The Open ASR Leaderboard measures Whisper large-v3 at **RTFx 146** and
  Parakeet TDT 0.6B v3 at **RTFx 3330** on one A100-SXM4-80GB at batch 64
  ([arXiv:2510.06961](https://arxiv.org/html/2510.06961v4)). At RTFx 146 on a rented A100 a
  full audio-hour of compute is under a cent. Making that 2x cheaper moves nothing.
- **Not raw RTF for a single stream.** A single stream needs RTF < 1 and then stops caring;
  everything above that is invisible to the user and worth nothing to the buyer.
- **Not finalization latency alone.** It is a constraint, not the axis. Our own first cell
  delivered p50 714/762 ms and p95 1047/1100 ms with 1702 of 1702 finals present against a
  1500 ms gate, latency passed and the product still failed. Latency belongs *inside* the
  metric as the SLO, which is exactly how streams@SLO is defined.
- **Streams@SLO is where the market actually prices the difference.** Every hosted speech
  vendor publishes a higher rate for streaming than for async on the same audio, and that
  multiple is the concurrency cost; none of them charges a premium for batch throughput. The
  price table and the margin arithmetic are business facts and stay private, not in this
  repository.
- **And it is where the published field is thin**, which is the honest reason it is worth
  attacking (§6.2).

### 6.2 What the incumbents actually deliver

Batch throughput, published and comparable:

| System | Number | Conditions | Source |
| --- | --- | --- | --- |
| Whisper large-v3 (transformers) | RTFx **146**, WER 7.44% | A100-SXM4-80GB, batch 64 | [arXiv:2510.06961](https://arxiv.org/html/2510.06961v4) Table 3 |
| Whisper large-v3-turbo | RTFx **200**, WER 7.83% | same | same |
| Canary-1B-v2 | RTFx **749**, WER 7.15% | same | same |
| Parakeet TDT 0.6B v3 | RTFx **3330**, WER 6.32% | same | same |
| ASR NIM Parakeet-0.6B-CTC, offline | **353.62** RTFX at 1 stream, **3707.4** at 32 | H100 | [NVIDIA ASR NIM performance](https://docs.nvidia.com/nim/speech/26.02.0/reference/performances/asr/performance.html) |
| faster-whisper / CTranslate2 | "up to 4 times faster than openai/whisper for the same accuracy", 50-70% less VRAM, INT8 on GPU and CPU. **No measured RTFx published by the project itself** | n/a | [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) |

Streaming concurrency, published:

| System | Number | Conditions | Source |
| --- | --- | --- | --- |
| ASR NIM Parakeet-0.6B-CTC, high-throughput | **512 streams**, avg 166.85 ms, p95 615.9 ms, 494.12 RTFX | H100 | [NVIDIA ASR NIM performance](https://docs.nvidia.com/nim/speech/26.02.0/reference/performances/asr/performance.html) |
| same, low-latency config | **64 streams**, avg 32.012 ms, p95 37.779 ms | H100 | same |
| same, low-latency config | **64 streams**, avg 53.643 ms, p95 97.948 ms | A100 | same |
| ASR NIM **Nemotron ASR Streaming** (en-US, low-latency, 160 ms chunks) | **128 streams** p95 **92.2** ms / **256** streams not published | A100 | same |
| same | **128** streams p95 **69.0** ms, **256** streams avg 74.7 / p95 **123.9** ms, RTFX 253.8 | H100 | same |
| same | **128** streams p95 **54.6** ms, **256** streams avg 62.1 / p95 **89.4** ms, RTFX 253.9 | **B200** | same |
| ASR NIM **Parakeet-1.1B-CTC** (en-US, low-latency) | "Maximum effective # of streams with n-gram language model: **160**"; 64 streams p95 56.0 ms | B200 | same |
| ASR NIM Parakeet-1.1B-RNNT (multilingual, low-latency) | stream counts to 128 published, **latency columns EMPTY — ABSENT** | B200 | same |
| Qwen3-ASR-1.7B streaming | **48-64 concurrent streams**, p95 finalization **under 0.50 s**, 100 ms client chunks | one RTX PRO 6000 or H100 | [Baseten model library](https://www.baseten.co/library/qwen3-asr-1-7b-streaming/) |
| faster-whisper / CTranslate2 | **No number exists, because no streaming path exists.** CT2 serves bounded utterances | n/a | n/a |
| whisper.cpp | No published streams-per-GPU figure found. Its `stream` example is a single-session sliding window | n/a | n/a |
| vLLM, SGLang | No published streaming-ASR concurrency benchmark found in this search. vLLM exposes Whisper as a batch transcription endpoint | n/a | n/a |
| Whisper-Streaming (LocalAgreement-2) | **3.3 s** average English word-emission latency; no concurrency figure | A40 | [IJCNLP-AACL 2023 demo](https://aclanthology.org/2023.ijcnlp-demo.3.pdf) |
| Deepgram Nova-3 | P50 **337-509 ms** audio-to-final, min 184 ms; architecture and streams-per-GPU undisclosed | hosted | [Deepgram latency docs](https://developers.deepgram.com/docs/measuring-streaming-latency) |

And the one board measured with OUR clock, independently and reproducibly — the Pipecat STT
benchmark, `TTFS = final TranscriptionFrame receipt − speech_end_time`, 1,000 samples of
`pipecat-ai/smart-turn-data-v3.1-train`, semantic WER, every row a hosted API over the network
([pipecat-ai/stt-benchmark](https://github.com/pipecat-ai/stt-benchmark)):

| Vendor / model | TTFS median | TTFS P95 | TTFS P99 | WER mean | Streams per GPU |
| --- | ---: | ---: | ---: | ---: | --- |
| NVIDIA Nemotron 3.0 ASR (en) | **221 ms** | **238 ms** | 252 ms | 1.90% | ABSENT |
| NVIDIA Nemotron 3.5 ASR (multilingual) | 236 ms | 253 ms | 266 ms | 4.54% | ABSENT |
| Deepgram nova-3-general | 247 ms | 298 ms | 326 ms | 1.71% | ABSENT |
| Soniox stt-rt-v4 | 249 ms | 281 ms | 310 ms | 1.25% | ABSENT |
| Soniox stt-rt-v5 | 260 ms | 305 ms | 313 ms | 1.34% | ABSENT |
| AssemblyAI universal-3-5-pro | 282 ms | 354 ms | 393 ms | 1.44% | ABSENT |
| Cartesia ink-2 | 299 ms | 328 ms | 1584 ms | 1.47% | ABSENT |
| ElevenLabs scribe_v2_realtime | 281 ms | 348 ms | 407 ms | 3.16% | ABSENT |
| Meta muse-voice-transcribe-1.0 | 392 ms | 1292 ms | 1922 ms | **0.97%** | ABSENT |
| Google gemini-3.5-transcribe-live | 458 ms | 532 ms | 599 ms | 2.24% | ABSENT |
| Mistral voxtral-mini-transcribe-realtime | 525 ms | 973 ms | 1913 ms | 4.44% | ABSENT |
| OpenAI gpt-4o-transcribe | 637 ms | 965 ms | 1655 ms | 3.24% | ABSENT |
| OpenAI gpt-realtime-whisper | 740 ms | 878 ms | 1080 ms | 2.92% | ABSENT |
| Speechmatics | 495 ms | 676 ms | 736 ms | 1.40% | ABSENT |
| Azure | 1016 ms | 1345 ms | 1791 ms | 1.21% | ABSENT |
| Google latest-long | 878 ms | 1155 ms | 1570 ms | 2.84% | ABSENT |

Every `ABSENT` above is recorded, not estimated: no hosted vendor publishes streams per GPU, and
the two boards are DISJOINT — the hosted board has our clock and no concurrency, the NIM board has
concurrency and a different clock. Nobody publishes the pair.

Read those tables together and the gap is specific, and narrower than the first reading of it.

**The NIM latency column is not our clock.** The harness is `riva_streaming_asr_client
--simulate_realtime --interim_results=false` over one LibriSpeech dev-clean file, three
iterations per stream, and the reported figure is "Overall latency of all responses", measured
from chunk submission to response receipt. That is per-chunk response time. Our SLO starts at
capture of the last owned speech sample and ends at the final transcript. The two are not
comparable and treating them as comparable makes our position look worse than it is
(`TRAP:riva-512-is-not-our-tier`).

**But the stream counts on that board ARE published at our tier, which the earlier reading of it
missed.** Parakeet-1.1B-CTC and Parakeet-1.1B-RNNT are ≥ 1B and both have rows; the B200 section
states a ceiling outright — "Maximum effective # of streams with n-gram language model: 160" for
1.1B-CTC at 160 ms chunks. So "nobody publishes concurrency at the accuracy tier" was wrong.
What nobody publishes is **concurrency AND a finalization SLO AND a WER on a named corpus, for
one model on one card**. Every row in the field has exactly two of those three.

**The accuracy-tier hosted figure with a finalization SLO is still one vendor's.** Baseten reports
Qwen3-ASR-1.7B at 48-64 concurrent streams with p95 finalization under 0.50 s on one RTX PRO 6000,
on unnamed audio in an unstated language, unreproduced by us.

### 6.3 The specific gap we intend to beat, and where we stand

Target, restated 2026-09-11 so it is falsifiable against what the field actually publishes. It is
a PAIR, because the two published boards are disjoint and a claim that wins only one leg is
hollow:

1. **Latency leg.** p95 TTFS (`speech_end` → final segment, the Pipecat harness clock) **< 238 ms**
   on the `pipecat-ai/smart-turn-data-v3.1-train` sample set, i.e. at or under the best hosted row
   in the field (NVIDIA Nemotron 3.0 ASR, 221 ms median / 238 ms p95), at a semantic WER inside
   1 pt of it.
2. **Concurrency leg.** That p95 held at **>= 128 concurrent real-time streams on ONE card**, on a
   named card, at the accuracy tier (>= ~1B).

Leadership is winning both legs at once on one card, and saying which card. Either leg alone is
already published by somebody else.

Where we stand: **streams@SLO = 0, and the latency leg is the harder one.** There is no GPU kernel
for any speech operation. Our own best measured finalization number is the two-tier RNNT student's
p50 714/762 ms and p95 1047/1100 ms at c1 on a different (Hebrew) corpus — **4.4x the p95 leg** —
and the 480 ms of lookahead that bought this lane's only free quality win
(`att_context_size [56,6]`) is ALONE twice the entire 238 ms budget. The honest reading: the
concurrency leg is reachable with scheduler and kernel work that is already scoped; the latency leg
is a chunk/lookahead policy question that no amount of kernel speed answers, because the budget is
spent on audio we deliberately wait for.

**Step 1 ran on 2026-09-10 and the collapse is engineering, not physics.** On one A100 80GB PCIe,
profiled at phase level: of a 26.18 ms streaming step only **12.06 ms is kernel execution**, there
are **2,904 device kernel launches per 80 ms of audio**, and the identical fused step costs
**31.522 ms at batch 1 and 35.890 ms at batch 32 (+13.9%**, reproduced at +14.7% in a second
process). A card that is SM-saturated at c4 scales roughly linearly in batch; this one is flat.
On that same card, in the same session, with the same audio and the same 64-frame bounded queue,
the drive shape alone decides the outcome: one worker at batch 1 sheds **2 of 4** streams at c4
with every lane at the queue cap, while one fused batch-N step consuming a frame from every lane
holds **64 of 64** at max queue depth 3 and 44% step duty.

So the ceiling this program has to beat is not the A100. It is our own scheduler. That is the
result §7 step 1 was bought to produce, and it clears K1. Step 3 has since moved that scheduler
into the engine (`memra-lanes::audio_stream`) with the losing drive shape kept as a named red arm,
so the collapse can be re-run in CI without a card. It does **not** establish a
`streams@SLO`: the batched arm replays one captured chunk across lanes on a shared batched cache
with no finalizer, no endpointing and no latency percentile, so it bounds compute capacity and
nothing else. Turning that bound into a served number is step 3.

Receipts (private): darklanes `research/speech-k1-20260910/RESULTS.md`.

---

## 7. Ranked plan

Sized in GPU-hours. Development time is not the schedule.

**Step 0, this document. 0 GPU-hours, done.** Defines the capability, the ladder, the
onboarding path, the gate set and the kill criteria, so that every step below has a registered
decision rule before it spends anything.

**Step 1, the incumbent baseline board. ~2.5 GPU-hours on one A100-80GB. Decisive and
cheapest; do this before writing a single kernel.** One rented non-production A100-SXM4-80GB
(so the numbers are directly comparable to the leaderboard's own A100 batch-64 rows), CUDA
allocation verified before the first byte is staged. Three arms on our pinned clip set plus one
English control:

- **A**: faster-whisper/CT2 large-v3, batch 1 / 8 / 64: does RTFx 146 reproduce on our audio,
  and what does CT2 actually cost per audio-hour on a card we could rent?
- **B**: CT2 under N simultaneous single-clip requests, N = 1, 4, 16, 64: the concurrency
  behaviour of the incumbent, which nobody publishes.
- **C**: reproduce the c4 collapse, with SM occupancy, VRAM and queue depth sampled
  throughout.

Decides: **is our c4 collapse a capacity wall or an admission defect?** If the card is under
~40% occupancy when it sheds, the collapse is ours to fix, the streams@SLO target is reachable,
and the program is real. If the card is saturated at c4 with an accuracy-tier model resident,
the product does not exist at that tier and K1 fires.

**Step 2, GPU kernels for the RNNT path. ~4-6 GPU-hours. DEPRIORITISED behind step 3 by the
2026-09-10 step-1 result**, which found 54% of the streaming step is gap between launches and
took one card from 4 streams to 64 with a scheduler change and no kernel. Kernels are not where
the loss is; the serving surface is. Take the family that already
streams end to end and put it on the GPU: frontend, FastConformer subsampler and blocks,
prompt kernel, predictor, joint. Exit, and it is deliberately not "RTFx > 1", which any GPU
clears instantly and would prove nothing: **byte identity against the banked CPU reference**,
plus a per-stream RTF measured interleaved against CTranslate2 on the same card in the same
session. That second half is what feeds K2, so the step that produces the number and the
criterion that judges it are the same measurement, not two separate ones taken weeks apart.
The parity half is cheap and safe precisely because the CPU reference already exists.

**Step 3, audio endpoint in `memra serve` plus the G8 battery. ~6-10 GPU-hours. PART ONE LANDED
2026-09-11 with no GPU; the measurement half is unrun.**

Landed: the `/v1/audio/*` surface (§1), per-lane sessions, the fused batch-N drive as the default,
the per-lane batch-1 drive as a named red arm (`MEMRA_AUDIO_DRIVE`), typed admission shedding with
a code / limit / observed on every refusal, the shed counters an operator reads per code, the ASR
decode contract refused at metadata load, and the scheduler's red and green arms running in hosted
CI against the measured step cost. 38 tests. No GPU, no rental.

Not landed, and it is the half that produces the number: a bound engine behind the surface (that
is step 2), the partial/final/revision event stream, cancel/reconnect/flush, and the G8 battery on
a card. **Exit is unchanged and unmet: a measured `streams@SLO` on one named card with the c4 red
arm reproduced and then green.** Nothing in the landed half is a performance receipt; the step
cost it schedules against is a 2026-09-10 A100 measurement, not a claim about any other card.

**Step 4, the accuracy tier (audio-LM class). Gated on steps 1 and 3.** AuT encoder, audio
projector, the streaming window and cache program. ~3-6 GPU-hours of bring-up before any
qualification. **Do not start this before the base-model question resolves elsewhere**, see K4.

Steps 2 and 3 run on the cheapest card that fits, not on an A100; step 1 uses an A100
specifically because that is the card the published rows were measured on.

---

## 8. Kill criteria

Registered here before the spend, so the program can be stopped by evidence rather than by
fatigue.

- **K1, capacity wall. ANSWERED 2026-09-10: does not fire.** The criterion was: if step 1 shows
  the card SM-saturated at c4 with an accuracy-tier model resident, "many concurrent streams on
  one card at the accuracy tier" is not a thing that exists. Step 1 measured the opposite. The
  card is idle between launches, not saturated (§6.3), and the same card goes from shedding 2 of
  4 to holding 64 of 64 on a drive-shape change with no kernel written. A resident large-v3
  finalizer costs the student 30% on the mean step and 2x on p95, which is a real budget item for
  step 3 and still leaves each stream under half of real time. **K1 does not fire; the program
  continues.**
- **K2, no engine advantage. NOT SETTLED, and its decision content has largely expired.** The
  criterion was: if step 2's GPU RNNT path lands within noise of CTranslate2's per-stream RTF on
  the same card after one optimization cycle, memra is not meaningfully faster on throughput and
  the batch axis is closed permanently. That path still has zero lines of code, so there is
  nothing to compare and no verdict may be issued. Step 1 closed the axis anyway, by price rather
  than by speed: batch-64 CTranslate2 measured on the pinned clip set costs about $0.015 per
  audio-hour of compute against a published market price one to two orders of magnitude higher,
  so engine throughput on the batch axis moves the P&L by a couple of points whatever it turns
  out to be. **The "our engine is fast on batch transcription" claim should be dropped from this
  program now rather than after step 2 measures it.** Streams@SLO carries the program alone.
- **K3, no business.** If step 3's streams@SLO on an RTX PRO 6000 lands below the break-even
  count for that card, the engine costs more per stream-hour than the cheapest published
  streaming price and the program is not a business. Sell the model, not the engine. (The
  break-even count and the arithmetic behind it are business facts and live in the private
  repository, not here.)
- **K4, quality decides first.** No amount of streams@SLO matters if the model behind it
  cannot clear the quality bar its market is judged on. The base-model question is open and
  owned by another lane. **Step 4 stays capped until that lane reports**, and speed work must
  not be used as evidence that the model question is settled.

K1 was answered on 2026-09-10 for about $3.58 of rented A100 time, well inside the ~7 GPU-hour
estimate, and it cleared. K2 needs step 2 to exist before it can be answered at all, and §6.3's
result reorders the plan so that step 3 comes first.

---

## 9. What this program does not claim

- No speech model is supported. `NativeReference` is unset for both current checkpoints, and
  loading, running and streaming on a CPU reference are not support.
- No speed claim of any kind exists for memra on speech. There is no GPU number.
- The 3.3 s, 512-stream, 48-64-stream, RTFx 146 and RTFx 3330 figures above are other people's
  measurements under their own conditions, cited so the gap is checkable. None of them was
  reproduced by us, and none of them was measured on Hebrew audio.
- Nothing here is a fleet rollout, a release, or a customer-facing claim.
