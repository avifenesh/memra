# ASR lane status

> Status page of the ASR modality lane (`lane/asr-modality-20260909`) at its 2026-09-10 handoff,
> moved from the repository root on 2026-11-12. On 2026-09-12 the owner stopped serving bring-up
> for this lane, which continued as research and correctness work only (the decision is in
> [ASR-MODALITY-PLAN.md](../../ASR-MODALITY-PLAN.md)). Nothing here is a serving claim or a roster
> commitment; the receipts and gates below stay valid as engine records. Current speech state:
> [docs/SPEECH.md](../../docs/SPEECH.md).

The lane's [PR #416](https://github.com/avifenesh/memra/pull/416) merged on 2026-09-10. Issue
#414 (native GPU ASR and the audio serving contract) is open.

## Committed and gated

Whisper: native log-mel (padded and clip programs), both convolutions, all 32 encoder blocks,
the cached decoder, the beam-1 transcription policy and the clip window program all execute in
the CPU reference path. `whisper-stage transcribe` takes PCM and returns token ids.

Nemotron RNNT: the whole streaming path executes. `rnnt-stage stream` takes PCM and returns
token ids: native log-mel, cache-aware FastConformer over the `[56, 0]` arm, prompt
conditioning, the two-layer LSTM predictor, the joint and greedy decoding, with encoder caches
and the predictor hypothesis carried across chunks.

| Stage | Comparison | Result | Bound |
| --- | --- | ---: | ---: |
| Log-mel, synthetic | Native vs HF FP32 | 0.0000567436 | 0.001 |
| Log-mel, 364 real windows | Native vs CT2 | 0.00000011920929 | 0.001 |
| Encoder FP32 | Native vs HF FP32 | 0.0002231598 | 0.001 |
| Encoder FP16 | Native vs HF FP16 | 0.25 | 0.2771682739 |
| Decoder FP32 | Native vs HF FP32 | 0.0000143051 | 0.001 |
| Decoder FP16 | Native vs HF FP16 | 0.015625 | 0.0345668793 |
| Window program | Native rule vs 364 pinned boundaries | 364 of 364 | exact |
| Clip parity, 8 HF clips | Native ids vs CT2 beam-1 | 49 of 61 windows | see below |
| Clip text, 8 HF clips | Native vs CT2, canonized | 6 of 8 clips | see below |
| RNNT census | Contract vs `clean-step-21959.nemo` | 657 of 657 tensors | exact |
| RNNT frontend | Native log-mel vs NeMo CPU FP32 | 0.0000534058 | 0.001 |
| RNNT encoder, [56,0] | 26 chunks vs NeMo CPU FP32 | 0.0000002533 | 0.001 |
| RNNT encoder on native mel | 26 chunks, frontend composed | 0.0000002086 | 0.001 |
| RNNT head | Prompt, predictor, joint | 0.0000915527 | 0.001 |
| RNNT greedy | Token ids | 9 of 9 identical | exact |
| RNNT session, 2 s | 26 chunk partials plus final | 26 of 26 identical | exact |
| RNNT session, 15.8 s | 199 chunk partials, final, transcript | 199 of 199 identical | exact |
| RNNT session, d1-000 15.0 s | 189 chunk partials, final, transcript | 189 of 189 identical | exact |
| RNNT session, 2 s of silence | 26 chunk partials, both empty | 26 of 26 identical | exact |

Whisper text parity, **stage gate missed numerically**, sweep stopped clean at 36 of 71 clips
and 259 of 364 windows:
**213/259 windows token-exact, 27/36 clips text-exact**, `d1` 0.0739 pt and
`whatsapp` 0.2237 pt against CT2. The stage gate wanted every clip equal or under
0.05 pt per domain, so it is **not met on either domain**. The 0.21 to 0.34 pt band the
oracle's own two backends disagree by is a scale reference measured on the 8 clips that have an
FP32 backend, not an envelope containing these 36-clip deltas: `d1` at 0.0739 pt is below it,
not inside it, and the band is itself partly a padding-convention artefact. What it shows is
that a 0.05 pt limit is an order of magnitude tighter than the references' own disagreement.
No divergence is a native defect against a same-program reference. The classes are fp16_tie 29,
boundary_cascade 13, fp16_ulp 2 and real 2; the 31 tie/ulp windows are measured per window,
both `real` windows are individually diagnosed, and all 13 cascades are traced to one of those
33 as their root. `whatsapp-003` window 4 step 86 is two adjacent timestamps two
fp16 steps apart with the clip's text unchanged. `d1-013` window 6 step 146 looked worst (a
0.671875 candidate margin, 43 fp16 steps) but the decision variable there is the
timestamp-forcing threshold, which sits inside one fp16 step: native F32 +0.008751, matched
FP32 +0.008760, CT2 fp16 -0.000860, against an fp16 step of 0.015625. The native branch is
the FP32-correct one. The engine's own detokenizer agrees with the checker's tokenizer on
36/36 clips. Full analysis: [TRANSCRIBE-PARITY.md](TRANSCRIBE-PARITY.md),
[WHATSAPP-001-W3-DIAGNOSIS.json](WHATSAPP-001-W3-DIAGNOSIS.json) and
[D1-013-W6-DIAGNOSIS.json](D1-013-W6-DIAGNOSIS.json).

The speech matrix product is cache-blocked and optionally threaded; both are bit-identical by
construction and re-proved on the real checkpoint. One encoder window fell 122.950 s to
60.477 s single-threaded, and one clip end to end fell 117.2 s to 60.9 s at 16 threads.

## Sweep closed at 36 clips: the partial verdict stands

The 71-clip sweep is closed and will not be resumed (owner decision, 2026-09-10: the
partial-sweep verdict stands; no further CPU-hours on it). 36 clips are banked (`d1-000..013`, `whatsapp-000..021`); the
unswept 35 are `d1-014..016` and `whatsapp-022..053`. The stop receipt,
[sweep-stop-20260909.json](sweep-stop-20260909.json), records the stop time, the clip lists and
the sweep's CPU cost.

The one binary that stands behind all 36 banked clips is pinned by sha256
`baf38cb6ada267058cb1f8b776ab72303c4fb0c284417ecbfa3d6e25a7eb590c` in that stop receipt and
in [stage5-transcribe-parity.json](stage5-transcribe-parity.json), so the banked clips stay
attributable after the sweep binary was deleted. If the last 35 clips are ever wanted, that is a
new lane: rebuild the binary (Commands below), record its hash beside the old one, and score
only under the new hash's name.

## Next executable stage

1. The sweep stays closed. If the remaining 35 clips are ever reopened as a new lane (above),
   give any flipping window a matched-program FP32 reference before a verdict, comparing the
   decision variable, not the candidate margin (see
   [D1-013-W6-DIAGNOSIS.json](D1-013-W6-DIAGNOSIS.json)): the banked `hf-fp32` windows pad in
   waveform space and the CT2 program zero-fills in feature space, so on any padded window the
   two references are not the same program.
2. Tokenizer encoding and vocabulary binding in the engine for the Whisper path (the detokenizer
   exists and agrees 36/36; `Tokenizer::from_hf_dir` still refuses this checkpoint's
   pre-tokenizer, and no BPE encode is ported).
3. Speech plans in the reference executor and the `model inspect` CLI; the pack is not in the
   text `PACKS` registry.
4. RNNT beyond three clips and one arm: a real corpus, the `[56,3]`/`[56,6]`/`[56,13]` arms if
   they are ever wanted, session revision semantics, cancellation and reset.

No speech pack has a support state. The measured ladder, with what each state still needs, is
the [Measured status ladder](../../ASR-MODALITY-PLAN.md#measured-status-ladder-2026-09-09)
section of ASR-MODALITY-PLAN.md; the speech support rules are in
[docs/SPEECH.md §3](../../docs/SPEECH.md#3-support-states-for-a-speech-model).

## Rules this lane runs under

CPU only: one process, no GPU, no local CI, no battery and no smoke server. Engine licence is
FSL-1.1-ALv2. External implementations only ever produce offline oracle files.

The lane's pushes used `MEMRA_SKIP_PERF_CI=1` and said so in the PR body; hosted CI gated the
merge. The flag is retired, and the pre-push hook now refuses it
([docs/FLAGS.md](../../docs/FLAGS.md#release-qualification-controls)). Pushes follow the
current rules in [AGENTS.md](../../AGENTS.md#branches-and-pushes).

Do not resume the removed mixed-precision or matrix variants, and do not tighten FP16 below its
measured reference floor.

## Artifacts

- Whisper checkpoint: `ivrit-ai/whisper-large-v3` at revision
  `766847c9795b3b5cc0d42f8476199c711d5cee21`, both shard hashes verified.
- RNNT checkpoint: `clean-step-21959.nemo`, sha256
  `585c40f214a53e8b5ba565c6176aa6cb548c7c8ab2f7ec5dc6dd45dbdff46ee4`.
- Oracle captures, kept outside the repository: a CPU oracle with the native encoder and decoder
  finals (FP32 and FP16) and their HF FP32 and FP16 references, and the CT2 oracle (a
  `MANIFEST.json`, which `check_whisper_real_audio.py` and `check_whisper_transcribe.py` read;
  71 CT2 clips with log-mel, encoder output, beam-1 tokens and post-suppression logits; 8 HF
  FP32 clips with raw pre-processor logits).
- Tiny fixtures: [memra-reference speech fixtures](../../crates/memra-reference/src/speech/fixtures/).
  The synthetic `tiny-*.safetensors` files need `git add -f`, because `.gitignore` ignores
  `*.safetensors`.

## Commands

Build both stage runners (CPU only). In the table below, `NEW_OUT_DIR` must not exist yet.

```sh
cargo build -p memra-reference --release --bin whisper-stage --bin rnnt-stage
```

| Stage | Command |
| --- | --- |
| Mel | `whisper-stage mel\|mel-clip CHECKPOINT_DIR PCM.f32 OUT.f32` |
| Encoder | `whisper-stage encoder CHECKPOINT_DIR MEL.f32 NEW_OUT_DIR f32\|f16` |
| Decoder | `whisper-stage decoder CHECKPOINT_DIR ENCODER.f32 TOKENS.txt NEW_OUT_DIR f32\|f16` |
| Transcribe | `whisper-stage transcribe CHECKPOINT_DIR PCM.f32 NEW_OUT_DIR f32\|f16` |
| RNNT frontend | `rnnt-stage frontend ARCHIVE.nemo PCM.f32 OUT.f32` |
| RNNT encoder | `rnnt-stage encoder ARCHIVE.nemo ORACLE_DIR NEW_OUT_DIR DROP_EXTRA_PRE_ENCODED` |
| RNNT head | `rnnt-stage head ARCHIVE.nemo ENCODER_ORACLE_DIR NEW_OUT_DIR PROMPT_SLOT` |
| RNNT stream | `rnnt-stage stream ARCHIVE.nemo PCM.f32 NEW_OUT_DIR PROMPT_SLOT` |

Gates: `tools/check_whisper_stages.py`, `tools/check_whisper_decoder.py`,
`tools/check_whisper_real_audio.py`, `tools/check_whisper_transcribe.py`,
`tools/check_rnnt_frontend.py`, `tools/check_rnnt_encoder.py`, `tools/check_rnnt_head.py`,
`tools/check_rnnt_stream.py`. All but `check_rnnt_stream.py` need numpy, and
`check_whisper_transcribe.py` also needs transformers and jiwer.
Captures: `tools/nemo_encoder_oracle.py`, `tools/nemo_rnnt_oracle.py`,
`tools/nemo_stream_oracle.py`. They need numpy, torch and nemo; `nemo_encoder_oracle.py` also
needs yaml.
Run every gate from the repository root; the receipts hash source paths relative to it.
