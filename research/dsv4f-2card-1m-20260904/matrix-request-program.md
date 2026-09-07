# Whole-request matrix candidate

2026-09-05 UTC. Continues the grouped-device-routing work. This is an
experimental numerical realization, not a qualified serving default.

## What changed

`MEMRA_DSV4_MOE_PROGRAM=matrix` selects the existing native grouped matrix
realization for all trunk prefill and verification rows. Plain decode uses a
persistent one-row transaction of that same batched executor, rather than
crossing back to the scalar expert program. The first prompt token also uses
the one-row executor. Both plain and DSpark chunked prime follow this rule.

Weights remain the same split-plane Safetensors NVFP4 artifact. Activation
inputs still undergo the source FP8 quantization, then lossless checked half
transport; this change adds no new CUDA arithmetic. Current source reference:
https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/inference/model.py

Trunk, verifier, DSpark state and opaque host snapshots carry a matrix/reference
program tag. Decode, verification, commit, tap use and restoration reject an
incompatible tag. A failed one-row matrix transaction cannot be reused or
snapshotted. The persistent one-row verifier and tap allocation are separately
reported by `matrix_device_scratch_bytes`, charging taps to their physical GPU.

Admission is RefFp8Round NVFP4 trunk, native device math, mode-2 grouped visitor
and direct loader. The original candidate refused EP. The current experimental
composition uses device routing, reused scratch and rebuilt rank-local pointer
tables; see `matrix-ep-chain.md` for its pending gates. The prefill-only grouped
probe remains incompatible. Monolithic reference APIs and full-layer
capture refuse this mode. The server requires nonzero prefill chunking and does
not take the short-prompt monolithic shortcut in matrix mode. Reference remains
the default; existing state must be rebuilt on a matrix/reference switch.

## Checks completed locally

- Engine library: 401 passed, 0 failed, 7 ignored (`matrix-program-final-cpu.log`).
- Server library: 602 passed, 0 failed (`matrix-program-server.log`).
- Strict engine/server library and phase-gate clippy pass (`matrix-program-final-clippy.log`).
- Optimized phase-gate build passes (`matrix-program-final-build.log`).
- Formatting, whitespace and runtime-flag census pass; `FLAGS.md` and `KERNELS.md` describe the candidate and its holds.

No target numerical or performance result is claimed from those checks.

## Target gate

`dsv4_matrix_program_gate` SHA256:
`c5ff8f022b34ccf74a88518ea1269f898822c63c5c6c2cebf158361cd4638767`.

The one-load gate uses real source tokens and requires:

1. Reference-to-matrix and matrix-to-reference live/host/draft state refusals.
2. Cold width equality at 33 tokens (1/32), 160 (1/32/64) and 1025 (32/128/512),
   including position zero, complete logits, live caches, DSpark rings and
   positive device-routing engagement.
3. Plain versus batched verification equality at widths 1/2/6, full commit,
   accepted-prefix rollback and next-step logits.
4. Fixed-seed vendor-shape sampled plain/DSpark output identity over 32 tokens.

The reference-versus-matrix logit delta is characterization only. Passing
within-program gates would not by itself qualify checkpoint quality or speed.
That initial candidate is historical. The repaired combined phase/storage gate
completed on the development PRO pair at 2026-09-05 23:06:43 UTC, status 0.
Binary SHA256: `7654612ebf1ebde481099096519678f4c6143c4ffa233f019443b7f6b7c3954a`.
All four groups above passed, plus fresh/reused storage at widths 1/32/64.
The 32-token sampled twin engaged 15 DSpark rounds. The maximum reference/matrix
final-row logit delta at the 160-token window was 1.0005016; this is not a
quality admission. Raw target receipts are banked in the companion private lane
as `matrix-widefix-model.log` and `matrix-widefix-controller.log`.

Two integration defects were repaired before that pass: the tiled indexer
launcher retained a 64-row guard despite independent row grids, and last-row
prefill head elision used the intentionally absent legacy StepWs in matrix
mode. The former now admits 512 rows with 6178334 exact component comparisons
on both PRO GPUs and a local memcheck pass; the latter uses the existing
one-row matrix verifier's head buffers and batched head kernel.

## Still on the implementation path

Target phase and fresh/reused workspace gates are complete for the scope above;
the per-call half/contribution allocation is removed. Characterize distribution
and quality drift, collect a clean matched profile, and remove remaining host
validation from the matrix critical path with fail-closed publication.
Continue matrix/dense/attention tuning and integrate
the qualified matrix path with EP or TP2 as measured. Cross-request scheduling,
stable-boundary cache reuse, active-host-C4 capacity, actual 1M prompts and c1-c16
serving/fairness remain part of the unchanged objective.
