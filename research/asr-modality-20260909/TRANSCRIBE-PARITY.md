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

## The eight HF-oracle clips, complete

61 windows, 4 `d1` and 4 `whatsapp` clips. Receipt `stage5-transcribe-parity.json`.

| Count | Value |
| --- | ---: |
| Windows token-exact | **49 / 61** |
| Clips with an identical window program | 7 / 8 |
| Clips text-exact after leaderboard canonization | **6 / 8** |
| Clips token-exact end to end | 1 / 8 |
| `d1` WER against CT2 | **0.0000 pt** (4 clips, 2364 words) |
| `whatsapp` WER against CT2 | **0.5656 pt** (4 clips, 884 words, 3 substitutions, 1 deletion, 1 insertion) |

**The stage gate is not met on `whatsapp`.** The rule was every clip's canonized text equal to
CT2's, or a WER delta under 0.05 pt per domain. `d1` passes both ways. `whatsapp` fails both:
two of its four clips differ in text, and 0.5656 pt is ten times the limit.

### Every differing window, classified

| Class | Windows | What it is |
| --- | ---: | --- |
| `fp16_tie` | 8 | CT2's own recorded logits hold both candidates at the same value |
| `fp16_ulp` | 1 | The gap is one fp16 step at that magnitude, the smallest it can express |
| `real` | 1 | Two fp16 steps apart, and both candidates are adjacent timestamps |
| `boundary_cascade` | 2 | The window does not start where the oracle's did, because an earlier tied timestamp moved the seek |

None of the twelve is a disagreement CT2's precision could have expressed clearly. The two
that cost words:

- `whatsapp-002` window 0 step 114: CT2 holds `.` and `,` at 22.171875 apiece, exactly equal.
  It took `.`, the native decode took `,`.
- `whatsapp-001` window 3 step 3: CT2 has 21.625 against 21.609375, one fp16 step apart at
  that magnitude. It took a byte-split fragment, the native decode took the whole token `יך`.

The one labelled `real` is `whatsapp-003` window 4 step 86: timestamps 51297 against 51296,
0.015625 apart where the fp16 step is 0.0078125, so two steps. 20 ms of timestamp, and that
clip's text still matches exactly.

### The two references disagree with each other more than the gate allows

The oracle ships two backends. CT2 is FP16 and covers all 71 clips; `hf-fp32` is FP32 and
covers these eight. Scoring the native run against both, and the two against each other:

| Domain | Native vs CT2 | Native vs HF FP32 | CT2 vs HF FP32 |
| --- | ---: | ---: | ---: |
| `d1` | **0.0000 pt** | 0.2114 pt | **0.2114 pt** |
| `whatsapp` | 0.5656 pt | **0.2262 pt** | 0.3394 pt |

On `d1` the native text is identical to CT2's, and the gap to the FP32 backend is exactly the
gap between the two references: 0.2114 either way. There is no closer place to be.

On `whatsapp` the native text sits between them, and it is **closer to the FP32 reference
(0.2262 pt) than the two references are to each other (0.3394 pt)**.

So the 0.05 pt limit is below the disagreement between the oracle's own two backends on this
corpus. No implementation can pass it against CT2 while also matching FP32, because CT2 and
FP32 do not match. Reporting it as a native failure would be reporting the wrong thing.

That is not a pass. The stage gate as written is **not met**: `whatsapp` is 0.5656 pt against
CT2 where the rule allows 0.05. What the FP32 control changes is the diagnosis, not the
verdict.

### Where the native path is genuinely wrong

The FP32 backend breaks the ties CT2 cannot, and it settles all three text-changing steps:

| Step | CT2 took | Native took | HF FP32 prefers | By |
| --- | ---: | ---: | --- | ---: |
| `whatsapp-002` w0 s114 | `.` (13) | `,` (11) | **native** | 0.001682 |
| `whatsapp-003` w4 s86 | 51297 | 51296 | **native** | 0.000969 |
| `whatsapp-001` w3 s3 | 1842 | 25988 | **CT2** | 0.064337 |

Two of the three are cases where the native F32 decode agrees with the F32 reference and CT2's
FP16 tie fell the other way; the HF decode emitted the native token at both steps.

The third is a real defect and is recorded as one. At `whatsapp-001` window 3 step 3 both
references choose 1842 and the native decode chooses 25988, and the FP32 margin is 0.064337,
which is four fp16 steps: wide enough that CT2 could have stated it too. Something in the
native encoder or decoder moved that step by more than 0.06. That is the one divergence in
these 61 windows that is not a coin-flip, and it is unexplained.

### What would settle the rest

Score the 71-clip sweep against CT2 as the only reference it has, and read the result against
the 0.21 to 0.34 pt band the two backends differ by on the eight clips where both exist. A
delta inside that band is not evidence of a defect; a delta outside it is.
