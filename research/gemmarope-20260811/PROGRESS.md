# Gemma4 RoPE cross-device fix — progress

Lane: `lane/cx-gemmarope`

Date: 2026-08-11

## Scope

- Replicate `GemmaAux::rope_freqs` once per distinct pipeline-parallel device.
- Replace all seven direct Gemma4 RoPE reads with a device-local accessor.
- Add a debug-only tensor/stream device-affinity assertion at Gemma4 and Step35 RoPE reads.
- Audit `GemmaAux::e4b` tensors for cross-stage reads and record the verdict.
- Preserve single-device behavior; do not claim Gemma PP-2 qualification.

## Required gates

- `cargo test -p memra-engine`
- Local RTX 5090, all GPU commands under `flock /tmp/memra-gpu.lock`:
  - `kernel-check` ALL GREEN
  - Gemma4 12B `run-gen` argmax MATCH
  - Gemma4 E4B `run-gen` argmax MATCH
- Commit raw gate logs under `research/gemmarope-20260811/raw/`.

## Status

- [x] Read the lane brief, PP audit, and repository law.
- [x] Confirmed clean isolated branch `lane/cx-gemmarope` at `9ff70064`.
- [x] Inspected all seven Gemma4 and four Step35 RoPE read paths.
- [x] Implemented device-local Gemma4 RoPE copies, accessor, and debug-only stream checks.
- [x] `cargo test -p memra-engine` passed (raw log retained; CUDA-only tests remained ignored).
- [x] Fresh release `run-gen` and `kernel-check` binaries built from lane commit `2e280285`.
- [x] Release binaries contain no debug device-assertion panic string (static zero-cost check).
- [x] Full `kernel-check` passed: `ALL GREEN`.
- [x] Gemma4 12B run-gen passed both MATCH lines; 20 generated ids equal the pinned golden.
- [x] Gemma4 E4B run-gen passed both MATCH lines; 20 generated ids equal the pinned golden.
- [x] Record the final verdict in `RESULTS.md`.

## E4B audit

`Gemma4E4bModel::{tok_tbl_gpu,model_proj,proj_norm}` are loaded or lazily uploaded through the
same caller engine used by `gemma4_e4b_inp_pl_dev` and the unsplit E4B trunk. E4B eager decode
explicitly reports its PP path as unwired, and its trunk never switches to a stage engine. These
tensors therefore have no current cross-stage read and need no replication in this lane. If E4B
PP is wired later, they must be revisited as part of that qualification.
