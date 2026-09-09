# ASR lane status

Updated 2026-09-09 15:16:16 UTC.
Worktree `~/projects/memra/wt-asr-modality`, branch `lane/asr-modality-20260909`.
Draft PR https://github.com/avifenesh/memra/pull/416, issue #414 remains claimed.

## Encoder stage ready for commit and hosted CI

- Native mel PASS: max abs 5.6743622e-5 versus HF FP32, limit 1e-3.
- Native FP32 encoder PASS: max abs 0.0002231597900390625, limit 1e-3.
- Native strict FP16 encoder PASS versus HF FP16: max abs 0.25,
  reference-derived limit 0.27716827392578125.
- F32 and F16 bind one binary: `6573d3a07567261e07287babecad3e0f808f276828f54c9a2683899e5ec068eb`.
- Eight focused speech tests pass. No model process is currently running.
- Stage-1 CI all green at `5259c5415`:
  https://github.com/avifenesh/memra/actions/runs/34356817571.
- Encoder commit/CI is next. Verify current head with `gh pr checks 416 -R avifenesh/memra`.

Corrections: native-owned A&S 7.1.26 GELU matches the pinned HF CPU vector program;
8-lane AVX2 moment grouping; half softmax reduction/tail order and reciprocal multiply.
Mixed precision and unsuccessful matrix variants were removed; their failed receipts remain.
Do not resume those experiments or tighten F16 below its measured reference floor.

## Next executable stage

1. Commit/push the encoder files and receipts with `MEMRA_SKIP_PERF_CI=1`; await CI green.
2. Wire the prepared `speech/decoder.rs` to shared native operators. It is not yet in
   `speech/mod.rs` and must not be claimed executable from the source file alone.
3. Generate/run the prepared tiny cached-decoder oracle; test self/cross KV, positions,
   transactional failed steps, and same-prefix full/cached parity.
4. Capture actual-checkpoint decoder step logits with `tools/whisper_decoder_oracle.py`.
5. Commit the passing decoder stage, update PR/this file, wait for hosted CI, and hand back.

Keep work CPU-only and niced, one process/thread. The owner permits only these fixed
2-second CPU fixtures and focused speech tests, not general local CI, batteries or GPU work.
Full speech support remains unset: transcription policy, 71-clip/rental parity, RNNT,
GPU execution and serving qualification are later gates. No external serving runtime.

## Artifact locations

Checkpoint: `~/hebrew-asr-data/models/whisper-large-v3-ivrit-766847c9/`, both shard hashes verified.
Oracle root: `~/hebrew-asr-data/oracle-cpu/asr-modality-20260909/`.
Reference directories: `hf-f32/`, `hf-f16/`, same 2-second synthetic PCM with normal
30-second padding ([128,3000] mel; [1500,1280] encoder). All 32 layer outputs are saved.
Passing native captures: `native-f32-final/`, `native-f16-softmax-order/`.
Archived winning binary: `bin/whisper-stage-encoder-6573d3a07567` under that oracle root.
Failed native candidates and isolated `gelu-probe/` / `norm-probe/` remain for diagnosis.
Receipts: `research/asr-modality-20260909/ENCODER-NUMERICS.md` and sibling JSON files.
Final receipts: `stage2-encoder-f32-final.json`, `stage2-encoder-f16-softmax-order.json`.

Rental oracle expected at `~/hebrew-asr-data/oracle/whisper-large-v3-ivrit/`; absent at last check.
Never invent those captures. All current numerical gates are explicitly CPU/synthetic scoped.
Tiny fixture directory: `crates/memra-reference/src/speech/fixtures/`.
Synthetic `tiny-encoder.safetensors` needs `git add -f` because generic weight files are ignored.

## Commands

Build: `CARGO_TARGET_DIR=.lane-asr-stage2/target nice -n 15 cargo build -p memra-reference
--bin whisper-stage --release -j 1`.
Runner: `.lane-asr-stage2/target/release/whisper-stage encoder CHECKPOINT_DIR MEL.f32
NEW_OUTPUT_DIR f32|f16` (nice; output dir must not exist).
Comparison: existing ASR venv Python, `tools/check_whisper_stages.py --oracle ORACLE_DIR
--native NATIVE_DIR --binary RUNNER --numeric CLASS --receipt RECEIPT.json
[--fp32-control HF_F32_DIR]`. F16 requires HF F16 plus its matched HF F32 control.

Prepared decoder source and its two oracle tools remain uncommitted and unwired for the next stage.
Clean `.lane-asr-stage2/` scratch at handoff after preserving the winning binary and receipts.
Both root checkouts stay on main. Keep this active worktree/branch for the open PR.
