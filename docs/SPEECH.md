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
| **Audio endpoint in `memra serve`** | **Does not exist.** No readiness, model id, streaming contract, concurrency, admission or rollback | n/a |

So: memra has a correctness spine for two speech families and **no speed and no serving
surface**. On the metric this program is about, memra has not entered the race. What the
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
| **G7 serve parity** | post-deploy probe against the served binary | The exact default decode shape, submitted with no decoding parameters | **Not built.** Blocked on an owner record: the deterministic beam-1/greedy contract this lane uses is a measurement instrument, and which decode default a served ASR endpoint carries is a product decision, not an engine one |
| **G8 streams@SLO battery** | serving battery on a lane-owned box | Readiness, model id, streaming partial/final/revision shape, concurrency ladder c1 → c4 → c16 → c64, admission limit, cancel, reconnect, flush, rollback | **Yes, and it is the best red arm we own.** One A100 holding a resident RNNT student plus a resident large-v3 **shed 54 of 71 streams on incoming-queue overflow at c4** while c1 ran fine. The battery must reproduce that shape and the fix must turn it green |
| **G9 non-vacuity** | every numeric gate | n/a | Each numeric gate carries a first-divergence log and refuses an empty input set; a sweep that scores zero clips is a failure, not a pass |

Two rules that bind all of them:

- **Timing starts from capture of the last owned speech sample**, not from decoder submission.
  Endpoint, frontend, encoder, decoder, queue and delivery time are separated, and cold versus
  warm and each concurrency level are recorded separately.
- **Admission must shed with a typed error, not drop.** The c4 collapse was a bounded-queue
  overflow that lost streams. A capacity limit that is reached is a product behaviour; a
  capacity limit that silently eats streams is a defect.

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
| Qwen3-ASR-1.7B streaming | **48-64 concurrent streams**, p95 finalization **under 0.50 s**, 100 ms client chunks | one RTX PRO 6000 or H100 | [Baseten model library](https://www.baseten.co/library/qwen3-asr-1-7b-streaming/) |
| faster-whisper / CTranslate2 | **No number exists, because no streaming path exists.** CT2 serves bounded utterances | n/a | n/a |
| whisper.cpp | No published streams-per-GPU figure found. Its `stream` example is a single-session sliding window | n/a | n/a |
| vLLM, SGLang | No published streaming-ASR concurrency benchmark found in this search. vLLM exposes Whisper as a batch transcription endpoint | n/a | n/a |
| Whisper-Streaming (LocalAgreement-2) | **3.3 s** average English word-emission latency; no concurrency figure | A40 | [IJCNLP-AACL 2023 demo](https://aclanthology.org/2023.ijcnlp-demo.3.pdf) |
| Deepgram Nova-3 | P50 **337-509 ms** audio-to-final, min 184 ms; architecture and streams-per-GPU undisclosed | hosted | [Deepgram latency docs](https://developers.deepgram.com/docs/measuring-streaming-latency) |

Read those two tables together and the gap is specific. **The 512-stream number is a 0.6B
English CTC model, the cheapest tier there is, and its latency column is per-chunk response
time, not end-of-speech-to-final.** It is not our SLO and it is not our accuracy tier. The
only published number in the accuracy tier with a real finalization SLO is Baseten's 48-64
streams at p95 < 0.50 s on one RTX PRO 6000, and it is a vendor figure on unnamed audio.
**Nobody publishes streams@SLO for a >= 1B model on a named corpus in a named language.**
That is the gap.

### 6.3 The specific gap we intend to beat, and where we stand

Target: **>= 128 concurrent real-time streams at p95 end-of-speech-to-final < 800 ms, on one
RTX PRO 6000, at the accuracy tier**, i.e. roughly 2x the only comparable published figure.

Where we stand: **streams@SLO = 0.** There is no GPU kernel and no audio endpoint. Our one hard
local datapoint is a *negative* one: on an A100 with a resident RNNT student plus a resident
large-v3, **c1 ran fine and c4 shed 54 of 71 streams on bounded-queue overflow**. Whether that
is a capacity wall or an admission defect is unmeasured, and it is the first thing §7 buys.

We are not behind by a percentage. We have not started. The program is worth funding only if
the first cheap step says the collapse is engineering rather than physics.

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

**Step 2, GPU kernels for the RNNT path. ~4-6 GPU-hours.** Take the family that already
streams end to end and put it on the GPU: frontend, FastConformer subsampler and blocks,
prompt kernel, predictor, joint. Exit, and it is deliberately not "RTFx > 1", which any GPU
clears instantly and would prove nothing: **byte identity against the banked CPU reference**,
plus a per-stream RTF measured interleaved against CTranslate2 on the same card in the same
session. That second half is what feeds K2, so the step that produces the number and the
criterion that judges it are the same measurement, not two separate ones taken weeks apart.
The parity half is cheap and safe precisely because the CPU reference already exists.

**Step 3, audio endpoint in `memra serve` plus the G8 battery. ~6-10 GPU-hours.** Readiness,
model id, streaming partial/final/revision contract, concurrency ladder, typed admission
shedding, cancel, reconnect, flush, rollback. Exit: a measured streams@SLO on one card with the
c4 red arm reproduced and then green.

**Step 4, the accuracy tier (audio-LM class). Gated on steps 1 and 3.** AuT encoder, audio
projector, the streaming window and cache program. ~3-6 GPU-hours of bring-up before any
qualification. **Do not start this before the base-model question resolves elsewhere**, see K4.

Steps 2 and 3 run on the cheapest card that fits, not on an A100; step 1 uses an A100
specifically because that is the card the published rows were measured on.

---

## 8. Kill criteria

Registered here before the spend, so the program can be stopped by evidence rather than by
fatigue.

- **K1, capacity wall.** If step 1 shows the card SM-saturated at c4 with an accuracy-tier
  model resident, "many concurrent streams on one card at the accuracy tier" is not a thing
  that exists. Drop to the 0.6B tier, where NVIDIA already publishes 512 streams and we would
  be a follower, or stop.
- **K2, no engine advantage.** If step 2's GPU RNNT path lands within noise of CTranslate2's
  per-stream RTF on the same card after one optimization cycle, memra is not meaningfully
  faster on throughput and the batch axis is closed permanently. Batch was never the lever;
  this only confirms it. Streams@SLO can still carry the program, but the "our engine is fast"
  claim does not.
- **K3, no business.** If step 3's streams@SLO on an RTX PRO 6000 lands below the break-even
  count for that card, the engine costs more per stream-hour than the cheapest published
  streaming price and the program is not a business. Sell the model, not the engine. (The
  break-even count and the arithmetic behind it are business facts and live in the private
  repository, not here.)
- **K4, quality decides first.** No amount of streams@SLO matters if the model behind it
  cannot clear the quality bar its market is judged on. The base-model question is open and
  owned by another lane. **Step 4 stays capped until that lane reports**, and speed work must
  not be used as evidence that the model question is settled.

K1 and K2 are answered by ~7 GPU-hours of rented time. That is what it costs to find out
before funding the program instead of after.

---

## 9. What this program does not claim

- No speech model is supported. `NativeReference` is unset for both current checkpoints, and
  loading, running and streaming on a CPU reference are not support.
- No speed claim of any kind exists for memra on speech. There is no GPU number.
- The 3.3 s, 512-stream, 48-64-stream, RTFx 146 and RTFx 3330 figures above are other people's
  measurements under their own conditions, cited so the gap is checkable. None of them was
  reproduced by us, and none of them was measured on Hebrew audio.
- Nothing here is a fleet rollout, a release, or a customer-facing claim.
