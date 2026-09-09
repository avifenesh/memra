# ASR lane status

Updated 2026-09-09 13:23 UTC. Branch `lane/asr-modality-20260909`, draft PR
https://github.com/avifenesh/memra/pull/416, issue #414 remains claimed.
Work only in this worktree. No GPU or general rig CI. The owner permits tiny niced,
single-process CPU oracle and stage gates. Push with `MEMRA_SKIP_PERF_CI=1`.

## Completed

- Metadata skeleton: commit `3503a3a88`, all hosted CI green.
- Native mel: commit `e4c314c38`; delta 5.6743622e-5 versus HF FP32 (limit 1e-3).
  Source weights fully downloaded and both publisher SHA256s verified.
  Receipt: `research/asr-modality-20260909/stage1-mel.json`.
- HF CPU FP32 reference: real ivrit encoder, deterministic 2-second synthetic PCM,
  normal 30-second padding, 128x3000 mel, 1500x1280 encoder, all 32 layer captures.
- Native FP32 encoder: completed all stages. First block above 1e-2 is layer 20,
  delta 0.010009765625 at [1482,695]. Final normalized output delta 0.0017070770.
  This is a diagnostic run; numerical investigation remains before FP16.

## Live execution and CI

No native/reference inference process is running at this checkpoint.
Stage 1 CI run 34355079221: 8 jobs passed; Clippy failed solely on
`manual_is_multiple_of` in the frontend. Fix that stage independently, then await green.
Encoder/matrix changes are local and not yet committed. Native compile completed.
No rental oracle directory was present at the last check.

## Durable artifacts and commands

- Checkpoint: `~/hebrew-asr-data/models/whisper-large-v3-ivrit-766847c9/`
- Oracle root: `~/hebrew-asr-data/oracle-cpu/asr-modality-20260909/`
- HF reference: `hf-f32/` with `manifest.json`, `pcm.f32`, `log-mel.f32`,
  `conv1.f32`, `conv2.f32`, `layer-00.f32` through `layer-31.f32`, `encoder.f32`.
- Native baseline: `native-f32/`, same encoder stage filenames.
- Comparison: `native-f32-comparison.json` at the oracle root.
- Rental oracle to watch: `~/hebrew-asr-data/oracle/whisper-large-v3-ivrit/`.
- CPU runner: `.lane-asr-stage2/target/release/whisper-stage`.
- Build only that CUDA-free runner with `CARGO_TARGET_DIR=.lane-asr-stage2/target
  nice -n 15 cargo build -p memra-reference --bin whisper-stage --release -j 1`.
- Runner: `whisper-stage encoder CHECKPOINT_DIR MEL.f32 NEW_OUTPUT_DIR f32|f16`.
  Output dir must not exist. Use nice, one process; retain failed runs separately.
- HF capture: `tools/whisper_cpu_oracle.py --checkpoint CHECKPOINT_DIR --out NEW_DIR
  --stage encoder --dtype f32|f16`, through the existing ASR venv under nice.

## Next

1. Repair stage 1 Clippy, preserving local encoder work, push and pass hosted CI.
2. Diagnose/reduce the layer-20 FP32 outlier and record before/after deltas.
3. Run native FP16, matched HF FP16 control and the final FP16-vs-HF-FP32 gate (1e-2).
4. Add small frozen encoder fixtures and rejection tests to hosted CI; no local broad suite.
5. Commit passing encoder stages, update PR receipts, await green, and refresh this file.

Full ASR decoding, RNNT and serving support remain unqualified. This work adds native
reference operators, not an external runtime bridge. Keep this file current at every stage.
