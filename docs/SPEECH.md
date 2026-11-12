# Speech

What memra has for speech models (ASR and TTS), the contracts a served speech path must meet, and
the path a new speech model takes into the engine. For contributors working on the speech packs,
the speech reference stages and the `/v1/audio/*` routes.

The program record (starting position, measured TTS artifact facts, incumbent board, targets,
ranked plan and kill criteria) is
[research/speech-program-20260911/](../research/speech-program-20260911/README.md). The ASR lane's
plan and status are [PLAN.md](../research/asr-modality-20260909/PLAN.md) and
[STATUS.md](../research/asr-modality-20260909/STATUS.md).

## 1. Current state

No speech pack holds a support state (§3), and no speech model can be served.

| Path | State | Evidence |
| --- | --- | --- |
| Whisper large-v3 (`ivrit-ai/whisper-large-v3`, Hebrew) | Pack `whisper` binds the checkpoint (1,259 tensors). The CPU reference runs the log-mel frontend, conv stem, 32 encoder blocks, cached decoder, beam-1 policy and clip window program; `whisper-stage transcribe` takes PCM and returns token ids. Clip parity against CT2 beam-1 missed its 0.05 pt per-domain WER gate on a 36-clip partial sweep (`d1` 0.0739 pt, `whatsapp` 0.2237 pt), with every divergent window traced to a cause | [STATUS.md](../research/asr-modality-20260909/STATUS.md), [TRANSCRIBE-PARITY.md](../research/asr-modality-20260909/TRANSCRIBE-PARITY.md) |
| Nemotron 3.5 streaming RNNT (`nvidia/nemotron-3.5-asr-streaming-0.6b` and a Hebrew successor, `[56,0]` arm) | Pack `nemotron_rnnt` binds the archive (657 tensors) and refuses to run it. The CPU reference runs the whole streaming path (`rnnt-stage stream`): frontend, cache-aware FastConformer, prompt kernel, LSTM predictor, joint, greedy decode and session state, token-exact against NeMo on two speech clips and a silence clip | [STATUS.md](../research/asr-modality-20260909/STATUS.md), [RNNT-ENCODER.md](../research/asr-modality-20260909/RNNT-ENCODER.md) |
| Qwen3-TTS (`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice@0c0e3051`) | Pack `qwen3_tts` holds the census (404 talker and 271 codec-decoder tensors), the talker prefill, the declared decode and the emit contract (§2.4.1), with refusal tests for malformed contracts. No native forward, codec decoder or stage-parity execution | `crates/memra-gguf/src/model_packs/qwen3_tts/` |
| MiMo-V2.6 audio input (audio-LM class) | Portable reference oracles cover the PCM log-mel frontend, the codec Conv1D frontend and encoder layers, RVQ, the patch-encoder attention, and code grouping with the embedding sum. Research CUDA kernels cover the codec frontend, codec encoder layers, RVQ selection, the embedding sum and the audio patch encoder; only `mimo_*` probe and gate bins reach them, and no request path takes audio input | `crates/memra-reference/src/mimo_audio*.rs`, the `mimo_audio_*` rows of [KERNELS.md](KERNELS.md) |
| GPU execution of an ASR or TTS pack | None. No Whisper, RNNT or TTS operation has a CUDA kernel, a residency plan or a performance receipt; every speech number above is a CPU reference | n/a |
| `/v1/audio/*` in `memra-server` | `POST /v1/audio/transcriptions`, `POST` and `GET /v1/audio/sessions`, and `POST /v1/audio/sessions/{id}/frames` and `/close` carry the session contract, the resident-session cap, the per-lane bounded queue, the typed shed codes and the declared decode. No speech pack can be resident, so the two opening POSTs (transcriptions, session open) answer 503 `engine_unbound` once the request passes validation; `GET /v1/audio/sessions` answers 200 with the operator view (drive, caps, shed counters); frames and close answer 404 `unknown_session`, since no session can open | `crates/memra-server/src/audio_api.rs`, `memra-lanes::audio_stream`, [API-SURFACES.md](API-SURFACES.md#audio) |

The speech packs sit outside `PACKS`, so automatic pack selection and `memra model inspect` never
reach them, and the reference executor refuses a speech plan (`native speech execution pending`).
Code: packs in `crates/memra-gguf/src/model_packs/{whisper,nemotron_rnnt,qwen3_tts}/`, reference
stages in `crates/memra-reference/src/speech/`, the CPU stage runners `whisper-stage`,
`rnnt-stage` and `speech-text`, the scheduler in `crates/memra-lanes/src/audio_stream.rs`.

The Whisper reference operators (`crates/memra-reference/src/speech/`, numeric programs and bounds
in [ENCODER-NUMERICS.md](../research/asr-modality-20260909/ENCODER-NUMERICS.md)) are native CPU
code, not CUDA support:

| Operator | Program | Numeric class |
| --- | --- | --- |
| `WhisperFrontend::compute` | periodic Hann, direct real DFT, Slaney filters, Whisper log normalization | F32 |
| `WhisperEncoder::encode` | biased strided convolution, positions, LayerNorm, full attention, residual FFNs | F32, or binary16 values with FP32 accumulation |
| `gelu_erf` | owned A&S 7.1.26 evaluation matching the pinned HF CPU program | FP32 fused polynomial |
| matrix product (`matrix.rs`) | four fixed FMA partial sums, AVX2 over time rows, cache-resident row groups, columns split over `MEMRA_SPEECH_THREADS` workers (default 1) | one reduction order at every group size and thread count, bit-identical by construction |
| `WhisperDecoderSession::step`, `decode.rs` | cached self and cross attention, tied logits, the pinned beam-1 suppression, timestamp grammar and forced-timestamp rules | as the encoder |

## 2. Families in scope

Four architecture classes cover the field on both the STT and the TTS side. The order of work
across them is argued in the program record on decision value, never on how much is already built.

| Class | Examples | Engine shape |
| --- | --- | --- |
| Encoder-decoder AED (Whisper) | `whisper-large-v3`, `large-v3-turbo`, `distil-*` | Bidirectional encoder over a fixed padded window; decoder with a self-attention KV cache and an independent fixed cross-attention K/V per window. The batch and utterance path, not the streaming one: post-hoc causalization of an offline AED encoder went from 7.6 to 17.6 WER with a p50 word commit of 4.1 s ([qfuxa/qwen3-asr-0.6b-streaming](https://huggingface.co/qfuxa/qwen3-asr-0.6b-streaming)) |
| Cache-aware streaming transducer (FastConformer, RNNT, TDT) | `nemotron-3.5-asr-streaming-0.6b`, `nemotron-speech-streaming-en-0.6b`, `parakeet-tdt-*`, Zipformer | Encoder trained with fixed left and right context, so caches run it frame by frame: attention last-channel cache, causal-convolution last-time cache, cache lengths, subsampler and frontend carry, position offset, a two-layer LSTM predictor and a joint with a blank transition |
| Audio-LM | `Qwen3-ASR-1.7B` (Qwen3-1.7B decoder, projector and a 300M AuT encoder with a dynamic 1 s to 8 s attention window), `Voxtral-Mini-4B-Realtime` (970M causal sliding-window encoder, 25M adapter, 3.4B Ministral decoder), MiMo-V2.6 audio | Audio encoder plus projector into a native text decoder, the vision path's projector-prefix shape, no cross-attention cache. Voxtral adds Ada-RMSNorm delay conditioning (`NormKind` has `Rms` and `LayerNorm` today) and a sinusoidal-embedding MLP in the decoder FFN. MiMo-V2.6 turns each four RVQ code rows into one 4096-wide trunk token |
| Streaming TTS (codec audio-LM) | `Qwen3-TTS-12Hz-*`, `chatterbox`, `orpheus-*`, `higgs-tts-2`, `csm-1b`, `kyutai/tts-*` | Text decoder emitting discrete codec tokens, a multi-codebook sampling head (one step emits several tokens), a codec decoder and the streaming emit contract (§2.4.1). No audio encoder |

Licences differ inside a family (`nemotron-3.5-asr-streaming-0.6b` is OpenMDW-1.1,
`nemotron-speech-streaming-en-0.6b` the NVIDIA Open Model License): read the card of the exact
artifact. For TTS the licence is a step-0 gate, since several of the best expressive checkpoints are
non-commercial on purpose. Out of scope until something in scope is qualified: speech translation,
diarization, speaker ID, voice conversion and audio understanding beyond transcription.

### 2.4 Streaming TTS: what the artifact fixes

Read the frame rate off the artifact, not the name. The pinned Qwen3-TTS artifact is a 12.5 Hz codec
(`output_sample_rate` 24000, `decode_upsample_rate` 1920): one frame is exactly 80.00 ms, and a
second of audio costs 12.5 talker forwards, each with a nested 15-step code predictor. Three more
facts shape the emit contract, and the pack pins them (`EMIT`, `frames_for_audible`): frame 0 is
silent and structural (peak about 5e-05, about -85 dBFS); the vendor reference emits
`max_new_tokens - 1` frames; and on an EOS-terminated prompt the two vendor entry points disagree by
one frame (17 frames of audio against 16 code rows), so the parity set needs such a prompt.

Parity bounds are measured, not typed. Greedy decoding on the codec head is chaotic under any change
of arithmetic (bf16 against float32 on one pin, fixture and seed flips 37 of 39 talker argmaxes), so
argmax and code identity are reachable only by a path bit-identical to the vendor kernels; any other
native path is compared distributionally or on quality and reported that way. The codec decoder is
the one stage with a real tolerance: the vendor's own bf16-vs-F32 floor, max abs sample delta
0.00738058 at 36.5 dB SNR, is the bound for a native codec decoder. A stage-coherent oracle captures
every stage from one generation and checks that the decode of its own codes is bit-identical to the
audio it returned. A non-autoregressive model such as `Kokoro-82M` (StyleTTS 2 + ISTFTNet, one
forward pass per utterance) has no decode loop, no session state and no cache across chunks, so
none of the decode-loop levers reach it; it may be served, but no speed claim is made on it. The measurements and their conditions
are in the program record (§2.4, §2.4.2); their raw receipts are not in this repository.

### 2.4.1 The streaming emit contract

Fixed on 2026-09-11, before the implementation, so a battery can be written against it. Every
constant is read off the pinned artifact or measured on it; the pack's `EmitContract` holds them.

- **Frame geometry.** One chunk is exactly 1920 samples at 24000 Hz (80.00 ms). A partial frame is
  never emitted; a chunk that is not a whole number of frames is a contract violation.
- **Audibility is declared on the wire.** Every frame carries `audible`, the `peak` behind it and
  the server's threshold (`audible_peak` 1e-3, -60 dBFS), so a client can recompute `audible` from
  the PCM. Silent frames are still delivered, with their duration declared.
- **Two clocks.** The terminal event reports time to first chunk and time to first audible frame;
  the difference, the silent lead-in, measured 76.8 ms at batch 1 and 61.2 ms at 64 streams on the
  vendor reference runtime.
- **Frame count is promised in emitted frames, never in `max_new_tokens`.** A client that asks for
  N frames gets N chunks, or a terminal event with the count and the stop reason (`eos`,
  `frame_limit`, `cancelled`). Which vendor count is authoritative is a pack decision.
- **Admission reuses the ASR shed codes** (`memra_lanes::audio_stream::ShedCode`) with their
  meanings: `engine_unbound` 503 (no TTS pack resident; fails closed), `sessions_exceeded` 503 (the
  session cap is full), `queue_overflow` 429 (the stream's bounded backlog is full: the lane fell
  behind real time; on TTS the queue holds outbound frames the client has not drained),
  `unknown_session` 404. Every refusal carries the code, the limit and the observed value; the
  retryable codes carry `retry-after` and `x-should-retry`.
- **Request validation is not shedding.** An unknown voice, unsupported language or format, or an
  over-length input is a 400 with its own typed reason, so a shed counter never counts client typos.
- **Cancel is a real seam.** A cancelled stream stops, releases its slot and ends with a `cancelled`
  terminal event carrying its frame count. A disconnect without cancel releases the same way.
- **Rollback seam.** The served decode is the declared one (G7, §5): for this artifact the vendor
  samples on both the talker and the code predictor (`SERVED_DECODE`: temperature 0.9 and top-k 50
  on each loop, talker repetition penalty 1.05). Greedy is the instrument for byte-identity gates,
  never the served shape. The shape is a registry entry, and the response reports the one it used.
- **Not in the contract.** No mp3 or opus (an encoded format changes the chunk quantum), no speed or
  pitch parameter, no voice-quality claim.

## 3. Support states for a speech model

The same three states as the text packs ([ARCHITECTURE.md](../ARCHITECTURE.md#10-support-states)).
Loading and running are not support. What each state requires of a speech pack:

- `NativeReference`: the complete speech forward and decode execute in memra's unfused native <!-- support: none; not NativeReference -->
  executor with persisted stage parity: the pack in the registry, the plan compiled through the
  shared `TensorContract`, the tokenizer bound in both directions inside the engine, and speech
  plans accepted by the reference executor and by `model inspect`.
- `NativeQualified`: the first state plus checkpoint parity on the full pinned clip set and the <!-- support: none; not NativeQualified -->
  battery of §5, bound to one artifact, plan, numeric stream, binary and bundle hash.
- `NativeTuned`: the second state plus current binary-bound rewrite receipts for each admitted <!-- support: none; not NativeTuned -->
  device and execution surface, with phase profiles and end-to-end measurements.

Every speech pack is below the first state. What blocks it: the Whisper tokenizer's encode
direction (the detokenizer agrees with the scorer 36/36; no byte-level BPE encode is ported), the
RNNT's move onto the shared `TensorContract` (it needs a torch-zip `CheckpointDialect`), and speech
plans in the reference executor and `model inspect`.

## 4. Onboarding a new speech model

Seven steps, in order, on top of the general procedure in [ONBOARDING.md](ONBOARDING.md). Each has
a fixed artifact and a fixed refusal.

0. **Pin and licence.** Exact revision, full payload SHA256 checked against the publisher, archive
   layout recorded, licence read from the artifact's own card. A finetuned successor gets its own
   manifest and census.
1. **Tensor census.** Count, elements, dtypes, shapes, storage offsets and strides against the real
   archive; reject missing, extra, duplicate and silently aliased storage (LSTM gate weights share
   flat storage). Metadata only: a restricted unpickler, no model code, no tensor storage read.
2. **Frontend contract.** Sample rate, FFT, hop, window, mel flavour (Slaney or HTK), filter count,
   log floor and clamp, normalization, dither, padding. Never inherited: Whisper is
   FFT400/hop160/128-mel with `(x+4)/4` scaling and a 30 s right pad; the Nemotron RNNT is
   FFT512/hop160/window400 with `normalize=NA` and its own stored window and filter buffers.
3. **Tokenizer and decode policy.** Special ids pinned as plan constants and gated against the
   checkpoint's generation config; encode and detokenize both in the engine; suppression, timestamp
   rules, blank index, prompt slots, language and task ids. No chat template for ASR; an audio-LM's
   decoder takes its template from the checkpoint.
4. **Streaming state contract**, if the model streams: cache shapes and elements per session,
   context accounting, and which `att_context_size` arms are expressible against admitted. Only the
   admitted arm is qualified.
5. **Stage parity against a pinned oracle.** Frontend, encoder, head or decoder, decode policy,
   each with a predeclared bound measured against truth (the F16 encoder bound is the same-fixture
   HF-F16-vs-F32 floor) and a first-divergence log. External code supplies offline fixtures only.
6. **End to end, then serve.** Exact token ids and UTF-8 bytes on the full pinned clip set for the
   matched deterministic program, then the battery of §5. A mismatch is never rounded into a pass
   by WER similarity.

## 5. The gate set

A gate that has never failed proves nothing. A row whose red arm has not fired says so and does
not count as protection.

| Gate | Caller | Non-vacuous input | Red arm |
| --- | --- | --- | --- |
| G1 census | pack tests on checked-in header fixtures (`cargo test -p memra-gguf`) | 1,259 Whisper and 657 RNNT tensors from the real archives | wrong, missing, extra and duplicate tensors rejected; RNNT aliased storage tested |
| G2 frontend | stage runner; hosted CI on the checked-in fixture | synthetic 2 s waveform and 364 real 30 s windows across 71 clips | yes; the 1e-3 bound is measured against HF FP32 |
| G3 encoder numerics | stage runner, F32 and strict F16 | the same fixtures in both classes | yes: the 1e-2 F16-vs-F32 gate failed; the bound is the HF F16-vs-F32 floor 0.2771682739 |
| G4 decoder, head, greedy | stage runner; session replay | 12 decoder steps; RNNT 9/9 tokens, 26/26 partials | yes: 46 divergent windows in the 36-clip sweep, each traced |
| G5 tokenizer | plan compile; generation-config check | generation config, added tokens, suppression list | partly: detokenize 36/36; encode missing, so half-vacuous |
| G6 checkpoint parity | clip sweep against the pinned oracle | 71 clips, two domains, WER delta bound 0.05 pt | yes, red: `d1` 0.0739, `whatsapp` 0.2237 (36 clips) |
| G7 served decode | registry half: `validate_asr_decode_contract` at metadata load (§5.1); live half: an out-of-repo post-deploy probe | the streaming RNNT and batch Whisper registry shapes and the pinned RNNT beam decoder; live: the default decode sent twice with no parameters | registry half yes; live half armable, not armed |
| G8 streams@SLO | scheduler half: `cargo test -p memra-lanes --lib audio_stream` in hosted CI; served half: an out-of-repo battery | readiness, model id, partial/final/revision shape, ladder c1, c4, c16, c64, admission limit, cancel, reconnect, flush, rollback; each rung lasts at least `queue_frames / (1000/frame_ms - served_rate_hz/streams)`, the per-lane bleed (the aggregate form gives 2.56 s for the measured c4 geometry, whose real answer is 10.24 s, and would bless exactly the rung length that misses the defect) | yes. Outside this repository, one A100 with a resident RNNT student and large-v3 shed 54 of 71 streams on queue overflow at c4 while c1 ran fine. In CI, `MEMRA_AUDIO_DRIVE=per_lane_worker` at the measured step cost (31.522 ms batch 1, 35.890 ms batch 32) refuses 4 of 4 at c4 with typed `queue_overflow`, `fused` holds 64 of 64 at queue depth 3 or less, and a 10 s probe of the broken shape refuses nothing at backlog 48 of 64 (why rung length is derived). GPU half unrun |
| G10 TTFA (§2.4 only) | clock and chunk half: an out-of-repo self-test; served half: once a TTS pack exists | the pinned artifact's measured chunk shapes and two malformed ladders | half armed: eight red arms refuse (vacuous chunk, measured silent lead-in, a 16 kHz void-run rate, vendor off-by-one, partial frame, silent-row ladder, means instead of p95, served half unbound). Served half not armed |
| G9 non-vacuity | every numeric gate | n/a | a first-divergence log; an empty input set or a zero-clip sweep fails |

Two rules bind every gate. Timing starts from capture of the last owned speech sample, not decoder
submission, with endpoint, frontend, encoder, decoder, queue and delivery time, cold and warm, and
each concurrency level recorded separately. Admission sheds with a typed error, never a drop: a
limit that is reached is defined behaviour, one that silently eats streams is a defect.

### 5.1 G7, the served decode contract

**What G7 asserts.** A served ASR endpoint decodes deterministically, by written contract, in the
way its WER was measured, and it says which decode that is. This is the text models' rule (serve
the vendor recommendation, verify the shape a client sends) applied to speech, where every family
in §2 recommends a deterministic decode:

- `openai/whisper` v20250625: `DecodingOptions.temperature: float = 0.0`. Called as a library it is
  greedy (`beam_size: Optional[int] = None`); the CLI ships `--beam_size` default `5`. Both
  deterministic.
- `faster-whisper` 1.2.1 / CTranslate2 4.8.2: `beam_size: int = 5` at temperature 0, and
  `temperature > 0` is a different code path (`sampling_topk: 0`, `sampling_temperature: t`), so
  temperature 0 means no sampling arguments exist at all.
- NeMo: `RNNTDecodingConfig.strategy: str = "greedy_batch"`, and
  `nemotron-3.5-asr-streaming-0.6b`'s own `model_config.yaml` ships
  `decoding.strategy: greedy_batch`. The only `temperature` in the RNNT decode surface is
  `softmax_temperature`, which sharpens beam scoring; it draws no sample.

**The temperature-fallback ladder is quality-failure recovery, not a decode default.** Its first
rung is 0.0; a hotter rung is reached only after a failed quality test (`compression_ratio > 2.4`,
`avg_logprob < -1.0`), and a high no-speech probability cancels fallback. An endpoint may serve a
ladder only declared exactly, with the temperature it used reported.

**Beam is not free on a streaming partial path, and it fails two different ways.** On
`nemo_toolkit 3.1.0+ea1ebf55b` (`rnnt_beam_decoding.py` sha256
`6e5a8b190b338717865db2f23d87e28ba5be9a0b1f1559dab1f9c885b4c27e53`), `tsd` (L744), `alsd`
(L907), `maes` (L1139) and the batched beam wrapper (L1704) each raise `NotImplementedError` on a
partial hypothesis, and `nsc` is not wired. But `default_beam_search` (L591) accepts partials and
collapses the beam to do it: L624-628 rebuild `kept_hyps` as a one-element list seeded from
`partial_hypotheses.y_sequence[-1]` and `dec_state`, score reset to `0.0`, no alternative carried.
Served width across a chunk boundary is 1 whatever `beam_size` says, so the offline beam WER does
not transfer. A bring-up that knows only the first failure and reaches for `default` finds that it
runs.

**The contract on this engine.** A transcription entry in the model metadata file
(`MEMRA_MODEL_METADATA`, `surface = "transcription"`) declares `decode_deterministic`,
`decode_strategy` (`greedy`, `greedy_batch`, `beam` or `default_beam_search`), `decode_beam_size`
exactly for a beam family, `decode_temperature_ladder` exactly (or `[]`), and `serves_partials`. No
stochastic sampling key may resolve into a transcription request. Silence is a violation, not a
default. A transcription response must report the temperature that produced it: without that field
the contract cannot be checked from the outside at all, which is the same reason HTTP 200 is not a
receipt for the text models.

**What the engine refuses on its own.** `validate_asr_decode_contract` runs inside
`validate_openrouter_metadata`, so every metadata load, boot and hot reload refuses: an entry that
declares nothing (the message quotes the `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]` ladder silence inherits
from a faster-whisper-class call); any `default_*` sampling key or `non_thinking_sampling` table on
a transcription row; an unserved strategy (`maes`, `tsd`, `alsd`, `nsc`); a beam family with no
width or width 0, or a width without a beam family; beam on a `serves_partials = true` row; a rung
outside [0, 1], a ladder that starts hot or does not strictly increase; `decode_deterministic =
true` under a hotter rung or `false` with none; and any decode key off the transcription surface.
At request time `top_p` is a 400, and `temperature` must be 0 or a rung of the declared ladder.

**Where G7 does not protect yet.** The live half (two identical no-parameter transcriptions of one
clip return byte-identical text) cannot fire. The endpoint reports the resolved decode on the
`x-memra-asr-decode` header even on its `engine_unbound` refusal
(`<strategy>;temperature=<rung>;partials=<bool>;language=<code or auto>`), but with no engine bound
there are no two transcripts to compare, so the probe reports `not-armed` rather than passing.

### 5.2 ASR runtime contract

Condensed from the lane plan ([PLAN.md](../research/asr-modality-20260909/PLAN.md#runtime-contract)).

- **Input.** Mono 16 kHz PCM in monotonic, sample-indexed 80 ms frames (`FRAME_MS`). Frame cadence
  is not an 80 ms word-latency claim.
- **RNNT.** The admitted arm is `att_context_size [56,0]`, not the archive's first default
  `[56,3]`; the schema expresses `[56,3]`, `[56,6]` and `[56,13]` without admitting them.
  Attention and convolution state persist across chunks, valid lengths are tracked, right-context
  samples are never exposed early. LSTM state persists across emitted symbols, a blank advances
  acoustic time but not predictor state, and symbols per frame are bounded.
- **Whisper.** A completed bounded waveform at an utterance boundary, capped at 12 s of owned audio.
  The encoder is recomputed per waveform; decoder self-attention and cross-attention K/V are cached
  only within that waveform. The first arm is Hebrew transcription with no previous-text
  conditioning, no RNNT text injection and no hindsight rewrite from future audio.
- **Endpointing.** A native causal endpointer is reserved: 20 ms VAD frames, 300 ms trailing
  non-speech, 200 ms pre-roll and trailing context, 400 ms left context at forced boundaries, in
  memra-owned DSP.
- **Two tiers.** The next utterance's RNNT runs while the previous Whisper finalizes. Requests carry
  stream, segment, revision and the owned half-open sample interval; an immutable final replaces
  only its own provisional segment. Reset, cancel, reconnect and errors never leak or duplicate state.
- **Decode.** Deterministic beam 1 (Whisper) and greedy (RNNT); omitted parameters are qualified
  against it, including bounded no-speech behaviour. Text models keep their sampled defaults.
  Timestamp mode is a separate pinned oracle arm.
- **Acceptance targets** (the plan's battery): first partial p95 at most 500 ms, no update stall over
  1 s in continuing speech, finalization p50 at most 800 ms and p95 under 1500 ms per domain, failed
  and missing finals counted.

## 6. Metrics

- **streams@SLO (ASR):** concurrent real-time streams per GPU at a fixed end-of-speech-to-final
  p95, on an accuracy-tier model (about 1B parameters or more), on a named card at a declared chunk
  profile. A streams figure without its chunk is not comparable.
- **TTFA (TTS):** p95 perceived first-audible latency at a stated concurrency on a named card:
  synthesis start to the first non-empty chunk, plus the leading silence inside the stream.
- ASR and TTS figures are different clocks: never blended into one speech claim, nor with batch RTFx.
- The server's SLO is `MEMRA_AUDIO_FINAL_SLO_MS` (default 800, p95 end of speech to final),
  reported by `GET /v1/audio/sessions` and enforced by the G8 battery, not the engine. The plan's
  acceptance battery set p50 800 ms and p95 1500 ms (§5.2).

No speech metric has a memra measurement on a GPU.

## 7. Program plan, targets and kill criteria

The [program record](../research/speech-program-20260911/README.md) keeps the old section numbers:
§6.2 the incumbent board, §6.3 the target pair (p95 TTFS under 238 ms on the Pipecat harness, held
at 240 or more streams on one card at a declared chunk), §7 the ranked plan, §8 the kill criteria.
Step 1 (2026-09-10) found the c4 collapse was the drive shape, not the card, so K1 does not fire.
Step 3 part one (2026-09-11, no GPU) landed the `/v1/audio/*` surface, typed shedding, the decode
contract at metadata load and the scheduler arms in CI; the bound engine and the G8 battery on a
card are not landed. Step 2 (GPU RNNT kernels) ranks behind step 3, so K2 has no verdict. Step 4,
the audio-LM accuracy tier, waits on K4: quality decides first.

## 8. What memra does not claim

- No speech model is supported. Loading, running and streaming on a CPU reference are not support,
  and the TTS pack binds its artifact and executes nothing.
- No speed claim of any kind exists for memra on speech. No ASR or TTS pack has a GPU number.
- The third-party figures in the program record are other people's measurements under their own
  conditions; none was reproduced here, and none was measured on Hebrew audio.
