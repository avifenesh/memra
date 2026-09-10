# ASR lane status

Updated 2026-09-10 (sweep closed at 36 of 71 clips, partial-sweep verdict stands by owner
decision; both `real` divergences diagnosed as CT2 fp16 threshold effects; engine detokenizer
agreement 36/36; scratch disposed, stop receipt banked).
Worktree `~/projects/memra/wt-asr-modality`, branch `lane/asr-modality-20260909`.
Draft PR https://github.com/avifenesh/memra/pull/416, issue #414 remains claimed.

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
36/36 clips. Full analysis: `research/asr-modality-20260909/TRANSCRIBE-PARITY.md`,
`WHATSAPP-001-W3-DIAGNOSIS.json` and `D1-013-W6-DIAGNOSIS.json`.

The speech matrix product is cache-blocked and optionally threaded; both are bit-identical by
construction and re-proved on the real checkpoint. One encoder window fell 122.950 s to
60.477 s single-threaded, and one clip end to end fell 117.2 s to 60.9 s at 16 threads.

## Sweep closed at 36 clips: the partial verdict stands

The 71-clip sweep is closed and will not be resumed (owner decision, 2026-09-10: the
partial-sweep verdict stands; no further rig CPU-hours on it). History: the previous worker's
session died at ~22:21 UTC 2026-09-09 with its 22:30 timer gone with it; the takeover killed
the parent script at 23:08:51Z and let the in-flight `d1-013` bank itself at 23:17:58Z. 36
clips are banked (`d1-000..013`, `whatsapp-000..021`); the unswept 35 are `d1-014..016` and
`whatsapp-022..053`. Sweep CPU: 1050.8 CPU-minutes over 36 clips (62230 s user + 820 s
system), 18:13:36Z start. Stop receipt:
`research/asr-modality-20260909/sweep-stop-20260909.json`.

The one binary that stands behind all 36 banked clips is pinned by sha256
`baf38cb6ada267058cb1f8b776ab72303c4fb0c284417ecbfa3d6e25a7eb590c` in that stop receipt and
in `stage5-transcribe-parity.json`, so the banked clips stay attributable after the scratch
copies were deleted at handoff. If the last 35 clips are ever wanted, that is a new lane:
rebuild the binary from this branch (Commands below), record its hash beside the old one, and
score only under the new hash's name.

## Next executable stage

1. The sweep is closed at 36 clips and stays closed. If the remaining 35 are ever reopened,
   run them as a new lane with a fresh binary hash recorded beside the old one, and give any
   flipping window a matched-program FP32 reference before a verdict, comparing the decision
   variable, not the candidate margin (see `D1-013-W6-DIAGNOSIS.json`): the banked `hf-fp32`
   windows pad in waveform space and the CT2 program zero-fills in feature space, so on any
   padded window the two references are not the same program.
2. Tokenizer encoding and vocabulary binding in the engine for the Whisper path (the detokenizer
   exists and agrees 36/36; `Tokenizer::from_hf_dir` still refuses this checkpoint's
   pre-tokenizer, and no BPE encode is ported).
3. Speech plans in the reference executor and the `model inspect` CLI; the pack is not in the
   text `PACKS` registry.
4. RNNT beyond three clips and one arm: a real corpus, the `[56,3]`/`[56,6]`/`[56,13]` arms if
   they are ever wanted, session revision semantics, cancellation and reset.

The measured ladder, including everything `NativeReference` and `NativeQualified` still need,
is the "Measured status ladder" section of `ASR-MODALITY-PLAN.md`.

## Rules this lane runs under

CPU only on the rig, one process, `nice -n 15`, efficiency cores, no GPU, no local CI, no
battery, no smoke server. Pushes use `MEMRA_SKIP_PERF_CI=1` and say so in the PR body; hosted
CI still gates the merge. Engine licence is FSL-1.1-ALv2. External implementations only ever
produce offline oracle files.

Do not resume the removed mixed-precision or matrix variants, and do not tighten FP16 below its
measured reference floor.

## Artifact locations

Checkpoint: `~/hebrew-asr-data/models/whisper-large-v3-ivrit-766847c9/`, both shard hashes verified.
RNNT checkpoint: `~/hebrew-asr-data/models/campaign-20260908/clean-step-21959.nemo`,
sha256 `585c40f214a53e8b5ba565c6176aa6cb548c7c8ab2f7ec5dc6dd45dbdff46ee4`.
CPU oracle root: `~/hebrew-asr-data/oracle-cpu/asr-modality-20260909/` (`final-enc-f32/`,
`final-enc-f16/`, `final-dec-f32/`, `final-dec-f16/`; references `hf-f32/`, `hf-f16/`,
`decoder-hf-f32/`, `decoder-hf-f16-encf32/`).
Rental oracle: `~/hebrew-asr-data/oracle/whisper-large-v3-ivrit/` (MANIFEST.json; 71 CT2 clips
with log-mel, encoder output, beam-1 tokens and post-suppression logits; 8 HF FP32 clips with
raw pre-processor logits).
Receipts: `research/asr-modality-20260909/`.
Tiny fixture directory: `crates/memra-reference/src/speech/fixtures/`.
Synthetic `tiny-encoder.safetensors` needs `git add -f`; generic weight files are ignored.

## Commands

Python: `~/hebrew-asr-data/venv-nemo/bin/python` (numpy, torch, transformers, ct2, nemo, jiwer).
Build: `PATH=$HOME/.cargo/bin:$PATH CARGO_TARGET_DIR=.lane-asr-stage2/target nice -n 15 \
cargo build -p memra-reference --bin whisper-stage --release -j 2`.
Mel: `whisper-stage mel|mel-clip CHECKPOINT PCM.f32 OUT.f32`.
Encoder: `whisper-stage encoder CHECKPOINT MEL.f32 NEW_OUT f32|f16` (output dir must not exist).
Decoder: `whisper-stage decoder CHECKPOINT ENCODER.f32 TOKENS.txt NEW_OUT f32|f16`.
Transcribe: `whisper-stage transcribe CHECKPOINT PCM.f32 NEW_OUT f32|f16`.
RNNT: `rnnt-stage frontend|encoder|head|stream ARCHIVE ... ` built into
`.lane-asr-stage3/target` so the sweep binary is never rebuilt under it.
Gates: `tools/check_whisper_stages.py`, `tools/check_whisper_decoder.py`,
`tools/check_whisper_real_audio.py`, `tools/check_whisper_transcribe.py`,
`tools/check_rnnt_frontend.py`, `tools/check_rnnt_encoder.py`, `tools/check_rnnt_head.py`,
`tools/check_rnnt_stream.py`.
Captures: `tools/nemo_encoder_oracle.py`, `tools/nemo_rnnt_oracle.py`,
`tools/nemo_stream_oracle.py`.
Run every gate from the worktree root; the receipts hash source paths relative to it.

Scratch `.lane-asr-stage2/` and `.lane-asr-stage3/` was disposed at handoff (2026-09-10):
`sweep-stop-20260909.json` moved to `research/asr-modality-20260909/`; sweep checkpoints,
both `target/` trees, the sweep binary and all diagnostic scratch deleted. `/tmp` ASR scratch
(`/tmp/asr-real`, `/tmp/asr-wt-backup`, `/tmp/asr-redarm-dir.txt`, `/tmp/asr-redarm-3144124`,
`/tmp/nemo-census-21959.json`) deleted with it; the binary hash stays pinned in the receipts.
Both root checkouts stay on main. Keep this worktree/branch for the open PR.
