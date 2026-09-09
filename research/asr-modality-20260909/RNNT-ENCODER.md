# Native cache-aware FastConformer encoder, `[56, 0]`

Stage 6. Memra executes the streaming encoder of the Hebrew RNNT successor: its own causal
subsampling stem, its own relative-position attention over a 56-frame history, its own causal
convolution with its own history, stepped chunk by chunk. Nothing reads a sample the session
does not own yet.

## Result

26 of 26 chunks pass, worst chunk **2.533197403e-07** against a **1e-3** bound.

| Piece | Value |
| --- | --- |
| Clip | first 2.0 s of `whatsapp-019`, real Hebrew audio, 32,000 samples |
| Arm | `[56, 0]`: 56 chunks of left attention context, no right context |
| Chunks | 26, one encoder frame each, 80 ms of audio per step |
| Chunk input | 1 mel frame on the first step, then 17 (8 new plus 9 pre-encode carry) |
| Dropped rows | 0 on the first step, then 2 pre-encoded rows of overlap |
| Native binary | `rnnt-stage`, 13.2 s for the whole sequence on one niced core |
| Receipt | `stage6-rnnt-encoder.json` |

Reference: `tools/nemo_encoder_oracle.py`, torch 2.14.0, NeMo 3.1.0, CPU FP32, one thread,
eval mode, **dither forced to zero**. The checkpoint's own `model_config.yaml` builds the
modules and the checkpoint's own `encoder.*` tensors load into them with `strict=True`, so no
custom model class and no prompt kernel take part: this pins the encoder alone.

## What the capture corrected

The state contract committed with the loader said the attention cache held a key and a value
tensor per layer and doubled its element count. The reference caches the **normalized layer
input** and recomputes keys and values from it: the captured cache is `[24, 1, 56, 1024]`, one
tensor per layer. `state_elements` is now 1,376,256 + 196,608 + 2,560, not 2,752,512 + ....
A contract written from an architecture description and never measured had this wrong.

Two more facts the capture settles, both of which a formula would have got wrong:

- Three stride-2 causal stages take 128 mel bins to **17**, not 16. Causal padding is
  `kernel-1` before and `stride-1` after on both axes, so 128 -> 65 -> 33 -> 17, which is
  exactly the 4352 = 256 x 17 the projection is shaped for.
- The first streaming step is not a normal step: 1 mel frame in, 1 encoder frame out, no
  pre-encode carry and nothing dropped.

## Red arms

Both fire; neither is injected noise.

1. **Wrong overlap handling.** Running the same binary with the drop count set to 0 makes each
   chunk try to emit 3 frames. The encoder refuses at the attention step
   (`cache-aware attention is gated for one frame per chunk, got 3`) rather than emitting
   something, and the checker then refuses the short run.
2. **Off-by-one chunk alignment.** Comparing each native chunk against the *next* reference
   chunk gives max-abs 0.00735 at its smallest, 0.031 median, 0.0618 worst, against reference
   values whose own magnitude is about 0.94. The gate's 1e-3 bound is 7x below the smallest
   wrong-alignment error, so a pass is not something a misaligned run could reach.

## Scope

Encoder only, CPU only, one arm, at the time this stage landed: the reference's own log-mel fed
the chunks. Stage 7 below adds the native frontend and composes the two. No prompt kernel, no
predictor, no joint, no RNNT decoding, no GPU, no serving surface. The wider
`[56, 3]`, `[56, 6]` and `[56, 13]` arms stay expressible in the state contract and are
refused here: they buy accuracy with future audio.

`NativeReference` for the RNNT path is not claimed by this. It needs the frontend, the
predictor, the joint, the prompt slot and greedy decoding, each against its own pinned capture.

## Stage 7: the frontend, and the two composed

The native log-mel now runs from PCM. It is a different program from the Whisper frontend, not
the same one with different constants: a 512-point transform over a 400-sample window, a 0.97
pre-emphasis filter ahead of it, the checkpoint's own window and filterbank buffers rather than
recomputed ones, an additive `2^-24` log guard instead of a dynamic-range clamp, and no
normalization (`normalize: NA` reaches the reference's no-op branch).

| Gate | Max abs | Bound |
| --- | ---: | ---: |
| Frontend, 2 s clip, [128, 201] | 5.34058e-05 | 1e-3 |
| Encoder on the reference's chunk windows | 2.533197403e-07 | 1e-3 |
| Encoder on chunk windows cut from the **native** mel | 2.086162567e-07 | 1e-3 |

The third row is the frontend and the encoder measured as one path: the streaming windows are
cut out of Memra's own features, not the reference's, and the 26-chunk sequence still lands
2.1e-07 from the reference. Receipts `stage7-rnnt-frontend.json` and
`stage7-rnnt-composed.json`.

The padding convention was the whole difference. Reflect padding, which is torch's default and
what the Whisper frontend uses, left exactly three columns wrong out of 201: column 0 at 0.251,
column 1 at 0.0017, and the last valid column 199 at 1.833, with the interior at 1.1e-05. The
reference asks for constant padding explicitly. A one-word difference in a call, and only the
frames that touch an edge can see it.

Still missing for the RNNT path: the prompt kernel, the predictor, the joint, greedy decoding,
and a streaming session lifecycle. The encoder is fed chunk windows by a checker, not by a
native streaming driver.

## Stage 8: the head, and a Hebrew transcript

The RNNT head runs natively: prompt conditioning, the two-layer LSTM predictor, the joint, and
greedy decoding with the reference's own `max_symbols` of 10.

| Gate | Result | Bound |
| --- | ---: | ---: |
| Prompted encoder, 26 frames | 4.768371582e-07 | 1e-3 |
| Predictor rows, 7-token walk plus the start row | 7.748603821e-07 worst | 1e-3 |
| Joint logits, 26 frames against two predictor rows | 9.155273438e-05 worst | 1e-3 |
| Greedy token ids | **9 of 9 identical** | exact |

The tokens are `2 3225 6 2 1270 3155 3235 1273 3158`, which the checkpoint's tokenizer reads as
`כן, אני ומתן`. Receipt `stage8-rnnt-head.json`.

Three things the reference settled that guessing would have got wrong:

- The prompt is not a token. It is a one-hot language slot, he-IL is 64 of 128, concatenated
  onto **every** encoder row and pushed through a 1152 -> 2048 -> 1024 kernel. A prepended
  token would have been a different model.
- The predictor's start step is a zero row, not an embedded blank. The embedding has 13088
  rows and the blank is 13087, so embedding the blank is available and wrong.
- The joint's own output is device-dependent in the reference: it log-normalizes on CPU and
  returns raw logits on GPU, by an explicit device check. The native joint returns logits
  everywhere and the checker normalizes, rather than making Memra's output depend on where it
  ran. This showed up as a 38.7 gap on the joint rows while the greedy tokens matched exactly,
  which is the signature of a monotone transform rather than a numerical fault.

The greedy loop's symbol counter advances on a blank as well as on an emission, so
`max_symbols` bounds work per frame and not only emissions. That is the reference's loop, and
a version that only counted emissions would run longer on a frame that keeps predicting blank.

Still missing: a native streaming session lifecycle (the head is handed a finished run of
encoder frames, not driven chunk by chunk with partial results), the tokenizer inside the
engine, and everything about serving.

## Stage 9: the session

`rnnt-stage stream` takes PCM and returns token ids. It computes its own log-mel, cuts its own
streaming windows, steps the encoder with its own caches, prompts each frame and runs the
greedy loop with a predictor state that persists across chunks.

26 of 26 chunk partials identical to the reference session, final sequence identical: the same
9 ids, the same `כן, אני ומתן`. 13.1 s for 2 seconds of audio on one niced core. Receipt
`stage9-rnnt-stream.json`.

Comparing partials at every chunk and not only the final sequence is the point of this gate. A
session that carries the wrong state can still land on the right final answer, and one that
resets its predictor per chunk usually does on short audio.

The driver constants are now part of the state contract, measured from the reference's own
streaming configuration rather than derived: the first step is one mel frame with no carry,
later steps are eight new frames on a nine-frame carry, and each later step drops the two
pre-encoded rows the carry already produced.

What a session still does not have: partial and final revision semantics, cancellation, reset,
reconnect, concurrency, admission, a tokenizer inside the engine, and any GPU path. The
transcript here is produced by the checker's tokenizer, not by Memra.

## Stage 10: the text, in the engine

The streaming session now ends in Hebrew rather than in ids. `rnnt-stage stream` reads the
archive's own SentencePiece model out of the same mapped bytes the weights came from and writes
`transcript.txt`: for the pinned clip, `כן, אני ומתן`, equal to the reference session's own
transcript. The stream gate compares it.

`SpmDetokenizer` parses the serialized `ModelProto` with a minimal length-delimited reader:
field 1 is the repeated piece record, field 1 inside it is the text and field 3 is the type.
An unrecognized wire type is an error, not a skip.

Two details that would have been silent if guessed:

- The piece-type enum is NORMAL 1, UNKNOWN 2, CONTROL 3, USER_DEFINED 4, **UNUSED 5, BYTE 6**.
  The obvious reading puts BYTE at 5, which would turn byte fallbacks into dropped pieces and
  unused slots into raw bytes.
- The word boundary is `U+2581`, a character in the vocabulary, and the decode convention drops
  exactly one leading space after converting it. Without that the transcript begins with a
  space and never matches.

The Whisper path got the same treatment from the other direction: byte-level BPE through
`Detokenizer`, which has no `encode` at all, because `Tokenizer::from_hf_dir` rightly refuses a
checkpoint whose pre-tokenizer memra has not ported and that refusal is about encoding. On the
eight HF-oracle clips the engine's transcript agrees with the checker's transformers tokenizer
on 8 of 8.

Neither path now needs offline tooling to produce text.
