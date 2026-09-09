# ASR lane status

Updated 2026-09-09 (decode stage, real-audio sweep in flight, RNNT loader).
Worktree `~/projects/memra/wt-asr-modality`, branch `lane/asr-modality-20260909`.
Draft PR https://github.com/avifenesh/memra/pull/416, issue #414 remains claimed.

## Committed and gated

Whisper: native log-mel (padded and clip programs), both convolutions, all 32 encoder blocks,
the cached decoder, the beam-1 transcription policy and the clip window program all execute in
the CPU reference path. `whisper-stage transcribe` takes PCM and returns token ids.

Nemotron RNNT: the `.nemo` archive reader, tensor census, contract bind and `[56, 0]` streaming
state contract are committed. No RNNT execution exists.

| Stage | Comparison | Result | Bound |
| --- | --- | ---: | ---: |
| Log-mel, synthetic | Native vs HF FP32 | 0.0000567436 | 0.001 |
| Log-mel, 364 real windows | Native vs CT2 | 0.00000011920929 | 0.001 |
| Encoder FP32 | Native vs HF FP32 | 0.0002231598 | 0.001 |
| Encoder FP16 | Native vs HF FP16 | 0.25 | 0.2771682739 |
| Decoder FP32 | Native vs HF FP32 | 0.0000143051 | 0.001 |
| Decoder FP16 | Native vs HF FP16 | 0.015625 | 0.0345668793 |
| Window program | Native rule vs 364 pinned boundaries | 364 of 364 | exact |
| Clip parity, d1-000 | Native ids vs CT2 beam-1 | 9 of 11 windows | see below |
| RNNT census | Contract vs `clean-step-21959.nemo` | 657 of 657 tensors | exact |

The two differing Whisper windows differ at one timestamp token each, at a point where CT2's
own recorded logits hold both candidates at the same fp16 value. A plain argmax over CT2's own
logits picks what the native decoder picked. Text is unchanged. Full numbers:
`research/asr-modality-20260909/TRANSCRIBE-PARITY.md`.

The speech matrix product is cache-blocked and optionally threaded; both are bit-identical by
construction and re-proved on the real checkpoint. One encoder window fell 122.950 s to
60.477 s single-threaded, and one clip end to end fell 117.2 s to 60.9 s at 16 threads.

## In flight

The 71-clip end-to-end sweep is running on the rig, one process, `nice -n 15`, pinned to the
efficiency cores, ~82 s per window, 364 windows:

```
tools/whisper_transcribe_sweep.sh ~/hebrew-asr-data/oracle/whisper-large-v3-ivrit \
  ~/hebrew-asr-data/models/whisper-large-v3-ivrit-766847c9 \
  "$PWD/.lane-asr-stage2/target/release/whisper-stage" "$PWD/.lane-asr-stage2/sweep" f32 CLIPS...
```

Clip order puts the eight HF-oracle clips first. Each clip banks its own directory and the
sweep skips clips that already have `windows.tsv`, so an interrupt resumes. Do not rebuild the
release binary while it runs: the script invokes the binary per clip and a rebuild would put
two binaries behind one receipt. The banked copy is `.lane-asr-stage2/sweep-binary/`,
sha256 `baf38cb6ada267058cb1f8b776ab72303c4fb0c284417ecbfa3d6e25a7eb590c`.

Score it when it finishes:

```
~/hebrew-asr-data/venv-nemo/bin/python tools/check_whisper_transcribe.py \
  --oracle ~/hebrew-asr-data/oracle/whisper-large-v3-ivrit \
  --native .lane-asr-stage2/sweep \
  --checkpoint ~/hebrew-asr-data/models/whisper-large-v3-ivrit-766847c9 \
  --binary .lane-asr-stage2/sweep-binary/whisper-stage --threads 16 \
  --receipt research/asr-modality-20260909/stage5-transcribe-parity.json
```

The checker scores only clips that have finished, so it is safe to run mid-sweep.

## Next executable stage

1. Finish the sweep, bank `stage5-transcribe-parity.json`, and record per-domain WER against
   CT2 in `TRANSCRIBE-PARITY.md`.
2. Native detokenizer and tokenizer binding. Until then `NativeReference` cannot be claimed:
   text numbers depend on offline tooling.
3. RNNT execution: FastConformer frontend and subsampler, then blocks against a NeMo oracle
   the private lane has not captured yet.

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
Gates: `tools/check_whisper_stages.py`, `tools/check_whisper_decoder.py`,
`tools/check_whisper_real_audio.py`, `tools/check_whisper_transcribe.py`.
Run every gate from the worktree root; the receipts hash source paths relative to it.

Clean `.lane-asr-stage2/` scratch at handoff after preserving the binary and receipts.
Both root checkouts stay on main. Keep this worktree/branch for the open PR.
