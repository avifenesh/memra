# End-to-end native transcription against the rental oracle

Stage 5. Memra runs the whole clip: native log-mel, native window program, native encoder,
native cached decoder, native beam-1 policy. Nothing from the oracle enters the run. The
oracle is read afterwards, by `tools/check_whisper_transcribe.py`, to score it.

Binary `.lane-asr-stage2/target/release/whisper-stage`
(sha256 `baf38cb6ada267058cb1f8b776ab72303c4fb0c284417ecbfa3d6e25a7eb590c`), checkpoint
`~/hebrew-asr-data/models/whisper-large-v3-ivrit-766847c9`, oracle
`~/hebrew-asr-data/oracle/whisper-large-v3-ivrit`, numeric class F32,
`MEMRA_SPEECH_THREADS=16`, `nice -n 15`, pinned to the machine's efficiency cores.

## How the policy was derived, before it was written

Every rule in `speech::decode` was first replayed in Python against the oracle's own recorded
logits, with the oracle's own prefixes, and had to reproduce the oracle's decodes step for
step. Two independent sources made that possible:

- The `hf-fp32` backend banks **raw** decoder logits, before any generation processor. Applying
  the full policy to those and taking the argmax reproduces the oracle token ids exactly on the
  windows checked (d1-000 windows 0 and 1, whatsapp-001 windows 0 and 1: 152, 113, 181 and 195
  steps, no divergence).
- The `ct2` backend banks **post-suppression and post-timestamp** logits. The ids it holds at
  negative infinity are therefore observable, and the always-suppressed set is exactly the 89
  ids now pinned as `SUPPRESS_TOKENS`.

The window program was derived the same way and is stronger: `seek_advance` plus the segment
rule reproduce **all 364 pinned window boundaries across all 71 clips** from the token
sequences alone. The first rule attempt used the last timestamp instead of the one before it
and missed d1-000 window 2 by 112 frames, which is why the 364-boundary replay is the gate
rather than a spot check.

## Per-window parity

`d1-000`, 11 windows, first clip of the sweep: **9 of 11 windows byte-identical to CT2 beam-1**,
all 11 window boundaries identical, clip text identical.

Both differing windows differ at exactly one timestamp token and agree everywhere else.

| Window | Step | CT2 took | Native took | CT2 logit gap | Rest of window |
| --- | ---: | ---: | ---: | ---: | --- |
| 5 | 123 | 51185 | 51167 | 0.000000 | identical |
| 8 | 64 | 50956 | 50955 | 0.000000 | identical |

The gap is zero because CT2's recorded values for both candidates are the same fp16 number.
Window 5, step 123: CT2 has 51167 and 51185 both at 17.187500, one representable fp16 step
apart from nothing. The native F32 decoder separates them: 17.1956844 against 17.1861877, a
real margin of 0.0094967, which is larger than the fp16 spacing of 0.0078125 at that magnitude
and so cannot survive the conversion.

Two things follow, and they matter for how this is read:

1. This is not a tie-break bug on the native side. A plain argmax over **CT2's own recorded
   logits** picks 51167, the same id the native decoder picks. CT2 emitted the other member of
   its own tie. Changing the native argmax rule would not change the native output, because the
   native logits are not tied.
2. This is not a text difference. Both windows decode to the same words; only the timestamp
   value moves, by 0.36 s in window 5 and 0.02 s in window 8. The window boundaries still match
   the oracle, because the advance is taken from the last closed segment and that timestamp is
   unchanged.

Diagnostic inputs are the native clip log-mel for d1-000, the native F32 encoder for window 5
and the native decoder forced onto the oracle prefix; the encoder capture and the per-step
logits are under `.lane-asr-stage2/diag/` and are scratch, not receipts.

## Scope

CPU reference only. No GPU execution, no serving endpoint, no timestamp-alignment gate, no
F16 end-to-end arm. Text is scored with the checkpoint tokenizer through offline tooling
because Memra has no native detokenizer yet; that is a `NativeReference` gap, listed in
`ASR-MODALITY-PLAN.md`.
