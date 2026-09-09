# ASR lane status

Updated 2026-09-09 (decoder stage + measured FP16 gates).
Worktree `~/projects/memra/wt-asr-modality`, branch `lane/asr-modality-20260909`.
Draft PR https://github.com/avifenesh/memra/pull/416, issue #414 remains claimed.

## Committed and gated

Native Whisper large-v3 log-mel, both convolutions, all 32 encoder blocks, and the cached
decoder (self/cross KV, absolute positions, tied logits) run in the CPU reference path.
Every receipt below binds ONE binary, `457201d65124eee0`, and one pinned checkpoint.

| Stage | Comparison | Max abs | Bound |
| --- | --- | ---: | ---: |
| Log-mel | Native vs HF FP32 | 0.0000567436 | 0.001 |
| Encoder FP32 | Native vs HF FP32 | 0.0002231598 | 0.001 |
| Encoder FP16 | Native vs HF FP16 | 0.25 | 0.2771682739 |
| Decoder FP32 | Native vs HF FP32 | 0.0000143051 | 0.001 |
| Decoder FP16 | Native vs HF FP16 | 0.015625 | 0.0345668793 |

Decoder argmax: 12/12 match the FP32 truth in both numeric classes.
Native FP16 decoder error against the FP32 truth equals HF FP16's own error to the bit
(0.0345668793). 11 focused speech tests pass. fmt and clippy clean.

## Two gate defects found and fixed this stage

1. The encoder receipts did not record which mel the native run consumed. The archived
   winning binary re-run on `native-mel.f32` gives 0.6796875, not the receipted 0.25.
   Source and binary were never wrong; the receipt was silent about its input. The runner
   now banks `input-mel.f32` and the checker refuses a capture without it, or one whose
   banked mel is not the reference log-mel. Both refusals exercised.
2. The FP16 decoder floor was measured across two different encoder oracles, so it carried
   encoder FP16 error instead of decoder rounding. The checker now requires both precision
   captures to name the same `encoder_manifest_sha256` and to agree on argmax.

Four decoder red arms and two encoder red arms fire. See `DECODER-NUMERICS.md` and the
`Capture input pinning` section of `ENCODER-NUMERICS.md`.

Do not resume the removed mixed-precision or matrix variants, and do not tighten FP16 below
its measured reference floor.

## Next executable stage

1. Real-audio gates on the rental oracle: mel vs CT2 log-mel, encoder vs CT2 encoder (FP16),
   beam-1 token sequence exact match vs CT2. 8 HF clips first, then all 71 if CPU allows.
2. Update `ASR-MODALITY-PLAN.md` with the measured status ladder.
3. Start the nemotron streaming RNNT path: tensor contract and loader for the `.nemo`
   tarball, FastConformer cache-aware encoder skeleton with the `[56,0]` state contract.
   Stop after the loader and census test pass.

Keep work CPU-only and niced, one process/thread, fixed short fixtures. No general local CI,
battery, smoke server or GPU work. Full speech support remains unset: transcription policy,
71-clip parity, RNNT, GPU execution and serving qualification are later gates.

## Artifact locations

Checkpoint: `~/hebrew-asr-data/models/whisper-large-v3-ivrit-766847c9/`, both shard hashes verified.
CPU oracle root: `~/hebrew-asr-data/oracle-cpu/asr-modality-20260909/`.
Current captures: `final-enc-f32/`, `final-enc-f16/`, `final-dec-f32/`, `final-dec-f16/`.
References: `hf-f32/`, `hf-f16/` (encoder; identical pcm and log-mel), `decoder-hf-f32/`,
`decoder-hf-f16-encf32/` (HF FP16 decoder recaptured on the FP32 encoder oracle).
Superseded captures are retained under their original names for diagnosis.
Rental oracle: `~/hebrew-asr-data/oracle/whisper-large-v3-ivrit/` (MANIFEST.json; 71 CT2
clips with log-mel, encoder output, beam-1 tokens and post-suppression logits; 8 HF FP32).
Receipts: `research/asr-modality-20260909/`.
Tiny fixture directory: `crates/memra-reference/src/speech/fixtures/`.
Synthetic `tiny-encoder.safetensors` needs `git add -f`; generic weight files are ignored.

## Commands

Python: `~/hebrew-asr-data/venv-nemo/bin/python` (numpy, torch, transformers, ct2, nemo).
Build: `PATH=$HOME/.cargo/bin:$PATH CARGO_TARGET_DIR=.lane-asr-stage2/target nice -n 15 \
cargo build -p memra-reference --bin whisper-stage --release -j 1`.
Encoder: `whisper-stage encoder CHECKPOINT MEL.f32 NEW_OUT f32|f16` (output dir must not exist).
Decoder: `whisper-stage decoder CHECKPOINT ENCODER.f32 TOKENS.txt NEW_OUT f32|f16`.
Encoder gate: `tools/check_whisper_stages.py --oracle DIR --native DIR --binary BIN
--numeric CLASS --receipt R.json [--fp32-control HF_F32_DIR]` (F16 needs the control).
Decoder gate: `tools/check_whisper_decoder.py` with the same flags.
Run every gate from the worktree root; the receipts hash source paths relative to it.

Clean `.lane-asr-stage2/` scratch at handoff after preserving the binary and receipts.
Both root checkouts stay on main. Keep this worktree/branch for the open PR.
