# Gemma4 RoPE cross-device fix — results

Lane: `lane/cx-gemmarope`

Date: 2026-08-11

## Verdict

**PASS for the lane scope.** `GemmaAux::rope_freqs` now owns one copy per distinct
pipeline-parallel CUDA device, and every Gemma4 read resolves the copy for its caller engine.
Debug builds assert tensor/stream device affinity at all seven Gemma4 RoPE reads and the four
existing Step35 RoPE reads. Unit tests, the full local RTX 5090 kernel battery, and both required
Gemma4 generation gates passed.

This is **not** a Gemma4 PP-2 serving qualification. It closes the audited latent RoPE pointer
class; a PP-2 claim still requires its own multi-GPU correctness battery and resolution of any
other pipeline-parallel gaps. No merge, tag, release, or performance-board move was made.

## Change

- Changed `GemmaAux::rope_freqs` from one CUDA allocation to allocations keyed by device ordinal.
- Uploaded the dequantized RoPE factors once to every distinct engine selected by `pp_cuts`.
- Added a fail-fast `GemmaAux::rope_freqs(&Engine)` accessor and removed all seven direct reads.
- Added a debug-only shared assertion that compares tensor and stream CUDA ordinals at the eleven
  Gemma4 and Step35 RoPE read sites. The assertion and its panic text are absent from release
  binaries.

## E4B tensor audit

`Gemma4E4bModel::{tok_tbl_gpu,model_proj,proj_norm}` have no current cross-stage read. They are
loaded or lazily uploaded through the caller engine used by `gemma4_e4b_inp_pl_dev`, while the
unsplit E4B trunk never switches to a stage engine and eager decode explicitly reports its PP path
as unwired. No replication was added. These tensors must be audited again if E4B pipeline
parallelism is wired later.

## Evidence

| Gate | Result | Raw evidence |
| --- | --- | --- |
| `cargo test -p memra-engine` | PASS: 57 library tests passed, 1 CUDA-only test ignored; run-gen and MLA integration tests passed | [cargo-test.log](raw/cargo-test.log) |
| Fresh release build | PASS: `run-gen` and `kernel-check`, CUDA 13.1, `sm_120a` | [cargo-build-release.log](raw/cargo-build-release.log) |
| Release zero-cost check | PASS: debug device-assertion panic text absent from both release binaries | [release-zero-cost-check.log](raw/release-zero-cost-check.log) |
| Full `kernel-check` | PASS: `ALL GREEN: kernels match CPU reference.` | [kernel-check.log](raw/kernel-check.log) |
| Gemma4 12B `run-gen`, 20 tokens | PASS: prefill/decode MATCH; batched-prime/tokenwise MATCH; generated ids byte-identical to pinned golden | [run-gen-g12.log](raw/run-gen-g12.log) |
| Gemma4 E4B `run-gen`, 20 tokens | PASS: prefill/decode MATCH; batched-prime/tokenwise MATCH; generated ids byte-identical to pinned golden | [run-gen-e4b.log](raw/run-gen-e4b.log) |

The GPU battery ran under the required exclusive `/tmp/memra-gpu.lock`; before/after device state
is retained in [gpu-context-before.log](raw/gpu-context-before.log) and
[gpu-context-after.log](raw/gpu-context-after.log). The compact machine-readable outcome is in
[gate-summary.log](raw/gate-summary.log). These are single-run correctness gates, not throughput
evidence, and support no performance conclusion.
