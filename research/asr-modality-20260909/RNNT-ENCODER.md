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

Encoder only, CPU only, one arm. No frontend (the reference's own log-mel feeds the chunks; a
native NeMo-geometry frontend is separate work with a separate FFT-512 contract), no prompt
kernel, no predictor, no joint, no RNNT decoding, no GPU, no serving surface. The wider
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
