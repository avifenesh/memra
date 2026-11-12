# Speech program record

The speech program's record: its starting position, the measured facts and parity bounds behind
the TTS emit contract, the performance thesis with the incumbent board and targets, the ranked
plan and the kill criteria. For contributors planning speech work; the engine contract is
[docs/SPEECH.md](../../docs/SPEECH.md).

> Moved from `docs/SPEECH.md` on 2026-11-12. Against the old text there are four classes of edit
> and nothing else.
>
> 1. Private detail is removed: business arithmetic, prices, private receipt paths and owner
>    quotes. A removed span is marked `[private detail removed]` or `[quote removed: private
>    detail]`, except one owner quote in §7 whose sentence was shortened to "(owner ruling
>    2026-09-11)".
> 2. Deployment and product wording is neutralized without a marker: a "rented" or
>    "non-production" card is now the card, "the rental" the GPU time, "before the box was rented"
>    "before the cell ran", "the product" the cell or the capability, and "worth nothing to the
>    buyer" is cut.
> 3. Sections docs/SPEECH.md keeps in current form are not carried: the §2 introduction, §2.1,
>    §2.2, the engine-shape paragraphs of §2.4, §2.4.1, §3 to §5 and §9 "What this program does not
>    claim" (now SPEECH.md §8).
> 4. Dated notes in italics are added.
>
> §2.3 and the tail of §2.4.2 (Kokoro, licence, out of scope) were restored verbatim on 2026-11-12.
> Section numbers are the old SPEECH.md numbers. References to §2.4.1, §3, §4 and §5 point at
> docs/SPEECH.md, which keeps those numbers; §1 here is the starting position of 2026-09-10, and the
> current state is SPEECH.md §1. `memra serve` in this text is the `memra-server` binary. The GPU
> cells below (step 1 on an A100, the TTS launch-bound cell and the TTS floor on an RTX 5090) ran
> outside this repository and their raw receipts are not committed here, so under AGENTS.md's
> evidence rule they are the program's notes, not evidence.

## 1. Where the code actually is, 2026-09-10

Honest starting position, because a program that starts from a flattering framing is worth
less than one that starts from a measured deficit.

| Path | State | Evidence |
| --- | --- | --- |
| Whisper large-v3, Hebrew transcription | Frontend, conv stem, 32 encoder blocks, cached decoder, beam-1 policy and clip window program all execute in the CPU reference. End-to-end clip parity **missed its gate** and landed on a diagnosis | log-mel 5.6743622e-5 (bound 1e-3); encoder F32 0.0002231598; 36 of 71 clips swept, 213/259 windows token-exact, `d1` 0.0739 pt / `whatsapp` 0.2237 pt WER vs CT2 against a 0.05 pt limit |
| Nemotron 3.5 streaming RNNT `[56,0]` | The whole path executes and streams: frontend, cache-aware FastConformer, prompt kernel, LSTM predictor, joint, greedy, session state | 657/657 tensors, frontend 5.34058e-05, encoder 2.533197403e-07 over 26 chunks, head 9.155273438e-05, greedy 9/9, session 26/26 partials plus final identical |
| Audio-LM class (Qwen3-ASR, Voxtral) | Not started | n/a |
| **GPU execution for any speech operation** | **Does not exist.** No CUDA kernel, no residency plan, no performance receipt | Every number above is a CPU reference measured on efficiency cores |
| Streaming TTS (§2.4) | **A non-executing `qwen3_tts` pack defines the pinned tensor, prefill, decode and emit contracts; `NativeReference` remains unset.** <!-- support: none; not NativeReference --> Header fixtures cover 404 talker tensors and 271 codec-decoder tensors, with refusal tests for malformed contracts. The native forward, codec decoder and stage-parity execution are still unimplemented. The streaming emit contract is fixed in §2.4.1 and measured parity bounds are recorded in §2.4.2 | `crates/memra-gguf/src/model_packs/qwen3_tts/`; `docs/SPEECH.md` §2.4.1, and §2.4.2 below; receipts outside this repository [private detail removed] |
| **Audio endpoint in `memra serve`** | **Exists as of 2026-09-11 and fails CLOSED.** `/v1/audio/transcriptions` plus the `/v1/audio/sessions` lifecycle (open / frames / close / list) carry the session contract, the resident-session cap, the per-lane bounded queue, the typed shed taxonomy and the declared decode; with no speech pack resident every path refuses `engine_unbound` rather than answering with something that is not the model. **The transcript bytes are step 2's work** | `audio_api.rs`, 18 handler + contract tests; scheduler in `memra-lanes::audio_stream`, 20 tests |

So: memra has a correctness spine for two speech families, a serving surface that admits and
sheds honestly, and **no speed**. On the metric this program is about, memra has not entered the
race: there is still no GPU number [private detail removed]. What the
spine buys is an ORACLE: every GPU kernel written from here has a banked CPU reference to be
byte-identical against, so correctness is checkable from the first kernel instead of argued. That
is a statement about what can be VERIFIED, not about what it costs to enter; per the owner ruling
of 2026-09-11 [quote removed: private detail], no model, family or step in this document is ranked by how much
work it is.

`NativeReference` is unset for every speech model. Nothing here is production permission. <!-- support: none; not NativeReference -->

---

### 2.3 Audio-LM (Qwen3-ASR, Voxtral Realtime)

*The engine shape is condensed in the Audio-LM row of docs/SPEECH.md §2. Note, 2026-11-12:
`NormKind` now has `Rms` and `LayerNorm`; the RMS-only remark below is the 2026-09-11 state.*

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

### 2.4 Streaming TTS (codec audio-LM)

*The engine-shape paragraphs of this section are condensed in the Streaming TTS row of
docs/SPEECH.md §2, and the artifact facts in docs/SPEECH.md §2.4; the measured artifact facts
follow verbatim.*

The frame rate is the thing to read off the artifact before anything else, because it sets the
step budget directly: a 12 Hz codec means roughly **twelve full decoder forwards per second of
generated audio**, so a 1.9B model spends its whole session in a short, repeated, launch-bound
step. That is the same shape §6.3's profile measured on the ASR side, and it is why this family
is the best-matched case in the engine for graph capture.

**Read the rate, do not read the name.** `Qwen3-TTS-12Hz-1.7B-CustomVoice` @ `0c0e3051` is a
**12.5 Hz** codec: `output_sample_rate` 24000 and `decode_upsample_rate` 1920, so one frame is
**80.00 ms exactly** and a second of audio costs 12.5 steps, not 12. Neither 83.33 ms nor 2000
samples appears anywhere in the artifact. Three more facts about that artifact belong to the
streaming emit contract this family needs, and none of them can be inferred from a sibling:

- **The first emitted codec frame is silent** (peak 5.72e-05, about -85 dBFS) and it is
  structural, not a truncation artifact: the same 1920 samples sliced out of a 12-frame
  generation peak at 5.01e-05 while the rest of the clip peaks at 0.289. A TTFA clock stopped at
  the first chunk therefore times the arrival of silence, and the emit contract owes a caller
  either the silence with its duration declared or the first audible frame, never one labelled as
  the other. Re-measured independently on 2026-09-11 on a second host: frame 0 peak 4.816e-05
  against a rest-of-clip peak of 0.2451.
- **The vendor reference emits `max_new_tokens - 1` codec frames** (`min_new_tokens` is 2 and
  returns one frame), so a harness or a port that passes a frame count straight through is one
  frame short of what it reports.
- **The frame count also depends on WHY the talker stopped, and the two vendor entry points
  disagree when it stops on EOS.** Measured 2026-09-11 on the pinned artifact at
  `0c0e3051f131929182e2c023b9537f8b1c68adfe`, on a fixture that terminates on `codec_eos` rather
  than on the frame cap: the outer entry point returned **17 frames of audio** while the inner
  `model.generate` returned **16 code rows** for the same prompt and seed, so the codec decode of
  the returned codes is one frame shorter than the end-to-end audio and the two cannot be compared
  sample for sample. A fixture that hits the cap cannot show this at all, which is why the parity
  set needs at least one EOS-terminated prompt. The outer entry point does not post-process: its
  returned audio is **bit-identical** to a re-decode of its own captured codes, so the whole
  difference is in the codes. An emit contract that does not say which count a client is promised
  is ambiguous on every request that ends on EOS.

All three are measured. The first two have red arms in an out-of-repo TTFA gate [private detail
removed]; the EOS entry-point divergence is not yet armed and needs the EOS-terminated
stage-parity fixture. The receipts are outside this repository [private detail removed].

### 2.4.2 Parity bounds for this family are measured, and the first ones written down were unreachable

§4 step 5 says a bound is measured against truth and never typed, and §5's G3 row records the
precedent: an original 1e-2 F16-vs-F32 gate FAILED and the bound became the same-fixture
HF-F16-vs-F32 floor, 0.2771682739. This family needs the same correction and it is bigger.

**Greedy decoding on this model's codec head is chaotic under any change to its arithmetic.** Same
pin, same fixture, same seed, greedy on both loops, ONE thing changed (the vendor's own dtype,
bfloat16 against float32): **37 of 39 talker argmaxes flip, the first at step 2, and 582 of 624
codec ids differ.** The cause is measured, not argued: the head's top-2 margin over those 39 steps
is min **0.125**, p10 0.475, p50 2.000, with **10 of 39 steps below 1.0**, and at the first flip
the logit delta was **2.588** against a margin of **1.25**. Once one code differs the two runs are
generating different audio, so the per-step logit delta grows without bound after that (0.137,
3.31, 2.588, 9.33, 11.2, 16.6, 26.8, 21.4) and the end-to-end waveform comparison goes to a
NEGATIVE SNR. Two different valid utterances are uncorrelated.

Three consequences, and they bind the pack rather than the write-up:

1. **A generic compiler was never going to pass an argmax-identity gate here.** The 2026-09-11
   reference-runtime probe found `torch.compile(mode="default")` diverging at step 2 with 584 of
   624 ids differing, and read the cause as inductor's fused BF16 reduction order. The vendor's own
   F32 arm reproduces the same divergence at the same step with the same magnitude and no compiler
   in the path, so the instability is a property of the MODEL. That probe's conclusion stands and
   its cause is now general.
2. **S2 argmax-identity and S3 code bit-identity are reachable only by a path that is
   bit-identical to the vendor's bf16 kernels.** There is no tolerance to be had, because the
   margin distribution says any perturbation diverges. A native implementation that is not
   bit-identical has to be compared distributionally or on quality, and must be reported that way
   rather than as a passed identity gate.
3. **The codec decoder is the one stage where a real numerical tolerance exists**, because it has
   no autoregression: given pinned codes it is a deterministic conv stack. Measured on identical
   pinned codes with only the codec's arithmetic varying, the vendor's own bf16-vs-F32 floor is
   **max abs sample delta 0.00738058, mean 0.000524672, SNR 36.5 dB**, against a signal RMS of
   0.05071. That is **7.4x the 1e-3 peak and 23.5 dB under the 60 dB SNR** that the qualifying
   lane's pre-registration wrote down, so those two numbers are the vendor's own two arithmetics
   failing them. The bound for a native codec decoder is the measured 0.00738058 / 36.5 dB,
   re-derived exactly as G3's was, and the 1e-3 / 60 dB pair is retired for this family rather
   than quietly not applied.

The same arm establishes that the codec is a usable gate at all: decoding one pinned code list
twice in one process is **bit-identical**, and re-decoding it in a fresh process on a different
host under a different driver a day later is **bit-identical to the banked waveform**. So the
codec is reproducible and its tolerance means something. The autoregressive half is not
reproducible across arithmetic changes, and no tolerance on it means anything.

**One consequence for how an oracle has to be captured.** If S1/S2 are hooked out of one vendor
entry point and S3/S4 out of another, the stages are not from the same generation and no
implementation can pass them jointly. On this artifact the two entry points emit different codes at
the same seed (602 of 624 ids differing, first divergent frame 1), and the first banked oracle had
exactly that shape, which is why its own S4 and S5 disagree at SNR -2.856 dB inside one file. **A
stage-coherent oracle captures every stage from ONE generation, and its internal consistency check
is that S4, the decode of that generation's own codes, is bit-identical to S5, the audio that
generation returned.** That check is cheap and it is the one that catches an incoherent oracle, so
it belongs in the gate set rather than in a lane's write-up.

A NON-autoregressive TTS model (`Kokoro-82M`: StyleTTS 2 + ISTFTNet, one forward pass per
utterance) is a different program in this class's clothing. It has no decode loop, no session
state and no cache across chunks, so none of the levers above reach it. It may be served; it may
never be the model a speed claim is made on.

**Licence is a step-0 gate for this family specifically**, harder than for ASR: the authors of
expressive TTS sell that exact capability, and several of the best checkpoints are
non-commercial on purpose. Read the artifact's own licence file, and read it before the census.

**Out of scope until something in scope is qualified:** speech translation, diarization,
speaker ID, voice conversion, and audio understanding beyond transcription. Naming them is not
a roadmap. (TTS was on this list until 2026-09-11 and is now §2.4.)

---

## 6. Performance thesis

### 6.1 The headline metric

**For ASR:** concurrent real-time streams per GPU at a fixed end-of-speech-to-final p95, on an
accuracy-tier model (>= ~1B parameters). Short form: **streams@SLO**. [Private detail removed.]

**For TTS (§2.4):** the same shape with a different clock, **p95 TTFA at a stated concurrency on
a named card**. TTFA is *perceived first-audible latency*: wall-clock from synthesis start to the
first non-empty audio chunk, **plus the leading silence inside the stream**. The second term is
not a detail — a model that answers instantly with 200 ms of silence has not started speaking,
and only a definition that charges for that silence can tell the two apart.

The ASR and TTS numbers are different metrics on different clocks and **may never be blended into
one "fastest speech" claim**. Neither may be blended with batch RTFx.

Why this and not the alternatives:

- **Not cost per audio-hour for batch.** That fight is over and nobody won it, because it
  stopped mattering. The Open ASR Leaderboard measures Whisper large-v3 at **RTFx 146** and
  Parakeet TDT 0.6B v3 at **RTFx 3330** on one A100-SXM4-80GB at batch 64
  ([arXiv:2510.06961](https://arxiv.org/html/2510.06961v4)). [Private detail removed.]
- **Not raw RTF for a single stream.** A single stream needs RTF < 1 and then stops caring;
  everything above that is invisible to the user.
- **Not finalization latency alone.** It is a constraint, not the axis. Our own first cell
  delivered p50 714/762 ms and p95 1047/1100 ms with 1702 of 1702 finals present against a
  1500 ms gate, latency passed and the cell still failed. Latency belongs *inside* the
  metric as the SLO, which is exactly how streams@SLO is defined.
- [Private detail removed.]
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
| **`nemotron-3.5-asr-streaming-0.6b`, the publisher's own card** | **240 streams at an 80 ms chunk** ("~17x more ... 240 vs. 14" against buffered Parakeet RNNT 1.1B), **2,400 at 1120 ms**, "sustains ~1,000 parallel requests". Clock is **median final-token latency vs number of parallel requests** — **p95 ABSENT** | single H100 | [nvidia/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) |
| Kyutai STT-1B (delayed streams, 500 ms architectural delay) | **400 real-time streams**, **no latency SLO attached at all — ABSENT** | one H100 | [kyutai.org/stt](https://kyutai.org/stt/) |
| same, Rust server | **64 simultaneous connections at RTF 3x**, no percentile | one L40S | same |
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
2. **Concurrency leg.** That p95 held at **>= 240 concurrent real-time streams on ONE card at a
   declared chunk profile**, on a named card, at the accuracy tier (>= ~1B). The profile is part
   of the target and not a detail: per `PATTERN:chunk-size-is-a-served-profile-not-a-constant` one
   chunk knob moves a published ceiling by 4.8x, and the two rows this target is set against are
   240 streams at an **80 ms** chunk and 2,400 at **1120 ms** on the same card. A streams number
   without its chunk is not comparable to anything, including our own earlier one.

   **Raised from 128 on 2026-09-11, against us.** The old figure was set against Baseten's 48-64
   and read as "roughly 2x the only comparable published figure". The publisher of the checkpoint
   this engine already streams end to end now states **240 streams on a single H100 at an 80 ms
   chunk** on its own model card (§6.2), so 128 would have been **0.53x the incumbent on the same
   architecture**, not 2x ahead of it. Entering a race behind the leader while quoting a target
   that says "2x" is how a program congratulates itself. Two caveats keep the new number honest:
   it is measured at the 0.6B tier, and its clock is a MEDIAN, so there is still no published p95
   at any concurrency from anybody — which is the gap, and the reason the pair is stated as p95
   AND streams rather than either alone.

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

Receipts: outside this repository [private detail removed].

---

## 7. Ranked plan

Sized in GPU-hours, which are the scarce axis. **Development time is not the schedule and it is
not a criterion** (owner ruling 2026-09-11). Every step below is ordered by
what it decides and what it is worth, never by what it costs to write; the GPU-hour figures size
the GPU time, not the priority.

**Step 0, this document. 0 GPU-hours, done.** Defines the capability, the ladder, the
onboarding path, the gate set and the kill criteria, so that every step below has a registered
decision rule before it spends anything.

**Step 1, the incumbent baseline board. ~2.5 GPU-hours on one A100-80GB. Decisive, and it
spends the fewest GPU-hours of any step that can still kill the program; do this before writing a
single kernel.** One A100-SXM4-80GB
(so the numbers are directly comparable to the leaderboard's own A100 batch-64 rows), CUDA
allocation verified before the first byte is staged. Three arms on our pinned clip set plus one
English control:

- **A**: faster-whisper/CT2 large-v3, batch 1 / 8 / 64: does RTFx 146 reproduce on our audio
  [private detail removed]?
- **B**: CT2 under N simultaneous single-clip requests, N = 1, 4, 16, 64: the concurrency
  behaviour of the incumbent, which nobody publishes.
- **C**: reproduce the c4 collapse, with SM occupancy, VRAM and queue depth sampled
  throughout.

Decides: **is our c4 collapse a capacity wall or an admission defect?** If the card is under
~40% occupancy when it sheds, the collapse is ours to fix, the streams@SLO target is reachable,
and the program is real. If the card is saturated at c4 with an accuracy-tier model resident,
the capability does not exist at that tier and K1 fires.

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
The parity half is CHECKABLE because the CPU reference already exists: an oracle, not a discount.

**Step 3, audio endpoint in `memra serve` plus the G8 battery. ~6-10 GPU-hours. PART ONE LANDED
2026-09-11 with no GPU; the measurement half is unrun.**

Landed: the `/v1/audio/*` surface (§1), per-lane sessions, the fused batch-N drive as the default,
the per-lane batch-1 drive as a named red arm (`MEMRA_AUDIO_DRIVE`), typed admission shedding with
a code / limit / observed on every refusal, the shed counters an operator reads per code, the ASR
decode contract refused at metadata load, and the scheduler's red and green arms running in hosted
CI against the measured step cost. 38 tests. No GPU.

Not landed, and it is the half that produces the number: a bound engine behind the surface (that
is step 2), the partial/final/revision event stream, cancel/reconnect/flush, and the G8 battery on
a card. **Exit is unchanged and unmet: a measured `streams@SLO` on one named card with the c4 red
arm reproduced and then green.** Nothing in the landed half is a performance receipt; the step
cost it schedules against is a 2026-09-10 A100 measurement, not a claim about any other card.

**Step 4, the accuracy tier (audio-LM class). Gated on steps 1 and 3.** AuT encoder, audio
projector, the streaming window and cache program. ~3-6 GPU-hours of bring-up before any
qualification. **Do not start this before the base-model question resolves elsewhere**, see K4.

**Step 5, the TTS family (§2.4). ~11-16 GPU-hours to a published p95 TTFA row.** Steps 0-5 of §4
on a codec audio-LM are CPU work and cost **zero GPU-hours**: pin and licence, tensor census,
the codec tokenizer's contract (frame rate, codebook count, codebook cardinality — the TTS
analogue of §4's frontend contract, and equally un-inheritable between models), tokenizer and
decode policy, and stage parity against a pinned oracle. The GPU half is the codec DECODER, the
multi-codebook sampling head, and the streaming emit contract, then G10.

Its exit is deliberately not "it produces intelligible speech", which any correct port clears:
**byte identity against the banked CPU reference**, then a p95 TTFA at c1 / c8 / c32 / c64 on one
named card with the leading-silence term measured, not assumed.

This step rested on a claim that was an architectural inference and not a receipt: that a
12 Hz codec decode step is launch-bound the way §6.3's ASR step measured. **That cell ran on
2026-09-11 and came back positive**, on one RTX 5090 against `Qwen3-TTS-12Hz-1.7B-CustomVoice`
on a stock reference path, with its acceptance criterion committed before the cell ran:

| Quantity | §6.3's ASR step (A100) | The TTS decode step (RTX 5090) |
|---|---:|---:|
| Step wall | 26.18 ms | **57.99 ms** |
| Kernel inside the step | 12.06 ms | **13.62 ms** |
| Gap | 54% | **76.5%** |
| Device launches per 80 ms of audio | 2,904 | **7,021.8** |
| Batch 1 -> 32, same step | +13.9% | **+3.4%** wall |

One talker forward emits exactly 80.00 ms of audio and contains a nested 15-step autoregressive
code predictor over the remaining codebooks before the 28-layer talker forward runs, which is why
the launch count is 2.4x the ASR step's in a step with a wider gap. Mean kernel duration is
**1.94 us**, below the per-launch host cost. The step's priority is therefore correct and the
ordering is not re-argued. Receipts are outside this repository [private detail removed]; the numbers above are the
whole of what they say.

**Step 5's FLOOR ran on 2026-09-11 and the qualification lane is open.** On one RTX 5090
against the same pinned artifact, on the vendor reference path with no memra in it, first-audible
p95 TTFA is **241.69 ms at batch 1** and **332.23 ms at 64 streams under batched arrival**, so
streams at the 155 ms bar is **zero** on the reference runtime and the gap to close is about
2.1x at 64 streams. The owner's rule for this program stands: **the first measure is the floor,
not a decision.** Two results from that lane bind engine work here rather than the lane's own
write-up:

- **The parity oracle is bit-reproducible**, all five stages at max abs delta 0.0 under greedy on
  both loops with a fixed seed, so the native port can be checked stage by stage from its first
  kernel instead of argued. Reference-runtime only; `NativeReference` stays unset. <!-- support: none; not NativeReference -->
- **`torch.compile(mode="reduce-overhead")` REFUSES this shape**: CUDA graph trees fail on
  warm-up with `accessing tensor output of CUDAGraphs that has been overwritten by a subsequent
  run`, raised inside the talker's own decoder layer loop, because the nested code predictor
  consumes the talker's output tensor across runs. Generic graph capture does not own this step.
  A capture that works has to cover the whole 80 ms frame (one 28-layer talker forward plus the
  15 code-predictor forwards inside it) and copy out of graph-owned memory at the boundary. That
  is the design brief for the pack's decode step, and it is measured rather than assumed.

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
  than by speed [private detail removed]. **The "our engine is fast on batch transcription" claim should be dropped from this
  program now rather than after step 2 measures it.** Streams@SLO carries the program alone.
- **K3.** [Private detail removed: a business criterion.]
- **K4, quality decides first.** No amount of streams@SLO matters if the model behind it
  cannot clear the quality bar its market is judged on. The base-model question is open and
  owned by another lane. **Step 4 stays capped until that lane reports**, and speed work must
  not be used as evidence that the model question is settled.

K1 was answered on 2026-09-10 [private detail removed], well inside the ~7 GPU-hour estimate,
and it cleared. K2 needs step 2 to exist before it can be answered at all, and §6.3's
result reorders the plan so that step 3 comes first.
