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

### The one divergence that looked real, and was not

The FP32 backend breaks ties CT2 cannot, and it settles all three text-changing steps. Two of
them go the native way outright:

| Step | CT2 took | Native took | FP32 prefers | By |
| --- | ---: | ---: | --- | ---: |
| `whatsapp-002` w0 s114 | `.` (13) | `,` (11) | **native** | 0.001682 |
| `whatsapp-003` w4 s86 | 51297 | 51296 | **native** | 0.000969 |
| `whatsapp-001` w3 s3 | 1842 | 25988 | see below | |

The third looked like a real defect: the banked FP32 logits prefer CT2's 1842 by 0.064337,
four fp16 steps, wide enough that CT2 could have stated it. It is not a defect. **The banked
FP32 window is a different program.**

`whatsapp-001` window 3 is the clip's last window: 1061 real frames of a 3000-frame field. The
CT2 program zero-fills the rest in feature space. The HF backend pads in waveform space, so its
mel pad region carries the analytic silence floor, between -0.393 and -0.564, where CT2's is
exactly 0. The two references are not looking at the same input, and the encoders show it:

| Comparison, window 3 encoder | Max abs | Mean abs |
| --- | ---: | ---: |
| Banked HF FP32 vs native | 20.896336 | 0.150301 |
| Banked HF FP32 vs CT2 | 20.896292 | 0.150240 |
| Native vs CT2 | 1.995950 | 0.001751 |

Recomputing an FP32 reference on the **CT2 window**, with the checkpoint's own weights through
transformers on this machine, settles it:

| Comparison, matched program | Max abs | Mean abs |
| --- | ---: | ---: |
| Matched FP32 vs native | **0.002001** | **0.00000118** |
| Matched FP32 vs CT2 | 1.994110 | 0.001751 |

And at the step itself the matched FP32 reference has 1842 at 21.639971 against 25988 at
21.691929: it **prefers the native token by 0.051958**, six digits from the native decode's own
21.639973 and 21.691919. CT2's FP16 encoder moves that step by enough to flip it.

Correction: an earlier version of this receipt, and the commit that carried it, called this a
genuine native defect. It is not. The evidence for that claim was a reference computed on a
differently padded window. `WHATSAPP-001-W3-DIAGNOSIS.json` has the numbers.

So there is **no window in these 61 where the native path disagrees with a same-program
reference**. Every one of the twelve differing windows is a CT2 FP16 precision effect: a tie,
a one- or two-step separation, or a boundary cascade from one of those.

### What this does to the FP32 control

The domain numbers above compare native text against the banked FP32 text, and every clip's
last window is padded, so each of those comparisons includes one window where the two
references ran different programs. The `CT2 vs HF FP32` column measures that difference as much
as it measures precision. The direction of the conclusion does not change, because the native
text is inside the band either way, but the band is partly an artefact of the padding
convention and should not be quoted as a pure precision figure.

### 71-clip sweep, stopped clean at 36 clips

The sweep ran from 18:13:36Z on one niced process on the efficiency cores. The dead worker's
22:30 UTC timer never fired, so the takeover worker killed the parent script at 23:08:51Z and
let the in-flight `d1-013` finish its own clip boundary at 23:17:58Z (11 windows, banked).
**36 of 71 clips, 259 of 364 windows.** The remaining 35 clips (`d1-014..016`,
`whatsapp-022..053`) are resumable: the sweep skips clips that already have `windows.tsv`.
Receipt `stage5-transcribe-parity.json`.

| Count | Value |
| --- | ---: |
| Clips scored | 36 of 71 |
| Windows | 259 of 364 |
| Windows token-exact | **213 / 259** |
| Clips text-exact vs CT2 | **27 / 36** |
| Clips with an identical window program | 31 / 36 |
| Clips token-exact end to end | 12 / 36 |
| Engine detokenizer agrees with the checker | **36 / 36** |
| `d1` WER vs CT2 | **0.0739 pt** (14 clips, 8117 words; 3 sub, 3 del) |
| `whatsapp` WER vs CT2 | **0.2237 pt** (22 clips, 5365 words; 10 sub, 1 del, 1 ins) |
| Divergence classes | fp16_tie 29, boundary_cascade 13, fp16_ulp 2, real 2 |

**The stage gate is not met:** both domains sit above the 0.05 pt per-domain limit (`d1`
0.0739, `whatsapp` 0.2237). The `whatsapp` delta moved from 0.2089 at 10 clips to 0.2237 at 22:
coin-flip noise around the FP32 offset, not a trend.

#### The inter-backend band, stated with its scope

Corrected by the self-review of PR #416, 2026-09-10. An earlier version of this section, and
the PR body, said both deltas "sit inside the 0.21 to 0.34 pt band the oracle's own two
backends disagree by". That sentence compares two things that are not measured on the same
material, and it should not be read as an envelope the deltas fall into.

| Quantity | Clips | Reference words | `d1` | `whatsapp` |
| --- | ---: | ---: | ---: | ---: |
| Native vs CT2 (the gate) | 36 | 13,482 | **0.0739 pt** | **0.2237 pt** |
| CT2 vs HF FP32 (the "band") | 8 | 3,249 | 0.2114 pt | 0.3394 pt |
| Native vs HF FP32 | 8 | 3,249 | 0.2114 pt | 0.2262 pt |

Three things follow that the shorter sentence hid:

1. **Different corpora.** The gate deltas are measured over 36 clips; the band is measured over
   the 8 clips that have an FP32 backend at all, roughly a quarter of the words. They are not
   two readings of one set.
2. **`d1` is not "inside" the band; it is below it.** 0.0739 pt against a 0.2114 pt
   reference-to-reference gap. On `d1` the native text is byte-identical to CT2's, so there is
   no closer place to be, and the band is not what is bounding it.
3. **The band is not a pure precision figure.** Every clip's last window is padded, and on
   padded windows the banked `hf-fp32` program pads in waveform space while CT2 zero-fills in
   feature space, so each of these comparisons includes one window where the two references ran
   different programs. The `CT2 vs HF FP32` column measures that padding convention as much as
   it measures precision. This is the same split diagnosed on `whatsapp-001` w3 above.

What the band does support, and all it supports: a 0.05 pt per-domain limit is roughly an order
of magnitude tighter than the disagreement between the oracle's own two backends on the clips
where both exist, so the limit is not a scale at which any implementation could distinguish
itself from the references. That is an argument that **the written limit is the wrong
instrument**, not a demonstration that the native run passes anything. The gate is missed.

### The second `real`, and it is not one either: d1-013 window 6

The new `real` looked worse than whatsapp-003's: CT2 took Hebrew token 44644, the native
decode took 51781, and CT2's recorded candidate margin is 0.671875, 43 fp16 steps wide. It is
not a defect, and the reason is instructive: **the candidate margin is not the decision
variable.** 51781 is a timestamp (28.34 s into the window), and at that step the deciding rule
is the timestamp-forcing branch, `log_sum_exp(timestamps) > best_text`. That threshold sits
inside one fp16 step:

| Reading | best_text | timestamp LSE | mass - best | branch |
| --- | ---: | ---: | ---: | --- |
| Native F32 | 22.913361 (44644) | 22.922112 | **+0.008751** | timestamp |
| Matched FP32 reference | 22.913357 (44644) | 22.922117 | **+0.008760** | timestamp |
| CT2 fp16, post-suppression | 22.921875 (44644) | 22.921015 | **-0.000860** | text |

The fp16 step at that magnitude is 0.015625; the whole spread between the two engines'
decisions is 0.0096. The native decode took the FP32-correct branch. The matched FP32
reference on the CT2 window agrees with the native encoder to 1.40e-06 mean / 0.0032 max,
while CT2's fp16 encoder sits 1.39 max away, the same shape as the whatsapp-001 diagnosis.
The window mel itself is byte-identical between the native regeneration and the oracle
(1.1920929e-07 max), and a forced native decode on the oracle prefix takes 44644, proving the
decoder math right and pinning the split on the threshold. Full receipt:
`D1-013-W6-DIAGNOSIS.json`.

So both `real` windows are now diagnosed. The checker's docstring now carries this caveat.

### What each of the 46 is actually derived from, and what it is not

Added by the self-review of PR #416, 2026-09-10. The one-line claim these classes support is
easy to state too strongly, so here is exactly how far each class is measured.

| Class | Windows | Derivation | Strength |
| --- | ---: | --- | --- |
| `fp16_tie` | 29 | `oracle_logit_margin == 0.0` in CT2's **own recorded logits** | Measured per window. CT2 cannot express a preference here at all. |
| `fp16_ulp` | 2 | `0 < margin <= np.spacing(float16(magnitude))` | Measured per window. Both sit at exactly one step, 0.015625. |
| `real` | 2 | Margin wider than one step; each settled by hand against a **matched-program FP32 reference** | Individually diagnosed, with receipts. |
| `boundary_cascade` | 13 | `boundary_matches == False`, assigned **before** any logit is read | **Not** a measured fp16 effect. See below. |

The 13 `boundary_cascade` windows are the ones to be careful about. The checker files them on
the boundary test alone and never consults their logits, and their recorded CT2 margins are in
fact large: 0.419921875 to 8.6328125, hundreds of fp16 steps. That is not evidence of a
precision effect. It is not evidence of a defect either: a window that starts at a different
seek frame is **not the same audio as the oracle's window**, so a token-by-token comparison
against it is not a comparison of two decodes of one input. The margin printed for such a
window is measured against a row that answers a different question.

What makes them attributable is the root, and the roots were not written down until now. Every
one of the 13 follows a divergent, non-cascade window earlier in the same clip:

| Clip | Root window | Cascaded windows | Count |
| --- | --- | --- | ---: |
| `d1-002` | w8, `fp16_tie` | w9, w10 | 2 |
| `d1-012` | w1, `fp16_tie` | w2 | 1 |
| `d1-013` | w6, `real` (diagnosed, timestamp-forcing threshold) | w7, w8, w9, w10 | 4 |
| `whatsapp-005` | w1, `fp16_tie` | w2, w3 | 2 |
| `whatsapp-014` | w1, `fp16_tie` | w2, w3, w4, w5 | 4 |

13 of 13 traced, and every root is either a measured tie or the one hand-diagnosed `d1-013` w6
threshold. So the honest form of the claim is:

> 31 of the 46 divergent windows are measured CT2 fp16 precision effects, 2 are individually
> diagnosed against a matched-program FP32 reference, and the remaining 13 are seek-shifted
> windows that share no input with the oracle window they are scored against, each traceable to
> one of those 33 as its root. **No window in the 36 clips is a native defect against a
> same-program reference.** That conclusion is not the same statement as "all 46 are measured
> fp16 effects", and this receipt should not be read as making the stronger one.

A negative `oracle_logit_margin` would mean CT2 emitted a token its own recorded logits rate
below the one we took, which is the oracle's policy overriding its argmax rather than any
precision effect. None of the 46 has one. The checker now files that case as
`oracle_policy_override` instead of silently folding it into `fp16_ulp`; since no banked window
takes that branch, the classification of all 46 above is unchanged.

### What would settle the rest

The sweep is closed at 36 clips (owner decision, 2026-09-10): the partial-sweep verdict
stands, and this section is the instrument for any future reopening, not queued work. Run the
remaining 35 clips and read the domain deltas against the same 0.21 to 0.34 pt band.
The sharper instrument for any new flip stays the one this receipt used twice: recompute an
FP32 reference on the CT2 window with the checkpoint's own weights, and compare the decision
variable, not the candidate margin. Each window costs about two minutes.
