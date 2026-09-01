# Gemma4 PP-2 auxiliary tensor results

Lane: `lane/cx-gemmaaux`

Date: 2026-08-11

## Verdict

**PASS for this lane's auxiliary-pointer remediation and single-device regression scope.**
The implementation and full static audit pass, as do the focused compile, release build, unit
gate, full local RTX 5090 `kernel-check`, and pinned Gemma4 12B/E4B generation gates. Both
generation runs matched their prefill/decode and batched-prime/tokenwise argmax checks, and each
20-token output was byte-identical to its checked-in golden.

This lane is **not** a Gemma4 PP-2 serving qualification. It addresses the audited auxiliary
pointer class only, and no two-GPU Gemma4 serving run was performed here. No merge, tag, release,
or performance-board move is part of this result.

## Changes

- `GemmaAux::ones` now stores one immutable copy per distinct PP CUDA ordinal.
- `GemmaAux::ones(&Engine)` resolves by caller ordinal and fails fast when a local copy is absent.
- All seven Gemma attention families resolve and debug-assert the local `ones` allocation before
  their norm kernels. The debug guard is absent from release binaries.
- `gemma4_decode_step_h_pp2` snapshots the host position once, uploads it independently on each
  stage stream, and debug-asserts both stage-local allocations before the layer walks.
- The Gemma4 host-bounce refusal for the former shared position pointer was removed after the
  stage-local invariant replaced that path.
- Door-off and `MEMRA_PP_STREAMS=0` behavior retain one allocation and the original one-engine
  position path.

## Model-level auxiliary sweep

| Candidate | Stage-engine use | Verdict |
| --- | --- | --- |
| `GemmaAux::rope_freqs` | Full-attention layers on either PP stage | **SAFE, prior fix**: per-ordinal copies, fail-fast accessor, and debug tensor/stream checks at all seven Gemma4 attention families. |
| `GemmaAux::ones` | Every Gemma4 full-attention layer; 12 norm-kernel branches across seven attention families | **BUG FIXED HERE**: the former primary-only slice is now per ordinal and every read uses the accessor. |
| Per-step `pos_d` | Both PP-2 layer ranges | **BUG FIXED HERE**: stage 0 and stage 1 each upload and consume their own copy of the same host snapshot. |
| `GemmaAux::suppress_d` | Logits tail only, after the last-stage head | **SAFE for the supported serving topology**: it is primary-owned; serving chooses the last/head device as primary, and host bounce requires the primary on that last stage. It is never read inside a layer range. The `ppn-gate` first-stage-primary test topology can depend on peer mapping and is not evidence of arbitrary placement locality. The invariant is now CHECKED, not just argued: `gemma4_suppress` debug-asserts tensor/stream device affinity on `suppress_d` before `mask_ids_rows`, so any topology violating primary==head-stage trips in debug instead of silently peer-reading (post-review addition, `cargo test` 57 pass). |
| `HybridModel::{output_norm, output}` | Last-stage epilogue | **SAFE**: both load through `layer_engine(..., n_trunk - 1)` and execute through the last stage's engine. |
| `HybridModel::embd` | Stage-0 prologue | **SAFE**: host bytes are gathered on the CPU and the row is uploaded through the stage-0 engine. `embd_gpu` is not used by the PP-2 eager Gemma walk. |
| `GemmaAux::e4b.{tok_tbl_gpu, model_proj, proj_norm}` | No current PP stage walk | **NOT LIVE**: E4B eager decode routes to its unsplit trunk before the Gemma PP-2 branch and reports PP as unwired. Re-audit when E4B PP is implemented. |
| Per-layer Gemma4 tensors, including `router_scale_pre` and `per_expert_scale_d` | Owning layer's stage only | **SAFE**: the loader shadows `e` with `layer_engine` for each layer; the PP walker never carries a next-layer norm across the cut. |
| Layer KV/cache buffers | Owning layer's stage only | **SAFE**: PP cache allocation follows the stage fence; only the residual crosses the boundary. |
| Separate `GemmaDraft` device auxiliaries | Drafter/caller engine only | **NOT A STAGE READ**: drafter `rope_freqs` is consumed only by the primary-engine draft chain; its `ones` field currently has no read site. |
| Host configuration/scalars (`Gemma4Config`, layer scales, `cache.pos`) | Host values only | **SAFE**: no CUDA pointer affinity; `cache.pos` is snapshotted once before the two local uploads. |

The raw source inventory is retained in [audit-inventory.log](raw/audit-inventory.log).

## Evidence

| Gate | Result | Raw evidence |
| --- | --- | --- |
| `cargo check -p memra-engine` | PASS, sm_120a | [cargo-check.log](raw/cargo-check.log) |
| `cargo test -p memra-engine` | PASS: 57 library tests, run-gen unit test, and three MLA fixture tests; two CUDA-only tests ignored | [cargo-test.log](raw/cargo-test.log) |
| Fresh release build | PASS: `run-gen` and `kernel-check` built with CUDA 13.1 for sm_120a | [cargo-build-release.log](raw/cargo-build-release.log) |
| Release zero-cost check | PASS: debug auxiliary-device assertion text absent from both release binaries | [release-zero-cost-check.log](raw/release-zero-cost-check.log) |
| Full `kernel-check` | PASS: `ALL GREEN: kernels match CPU reference.` | [kernel-check.log](raw/kernel-check.log) |
| Gemma4 12B `run-gen`, 20 tokens | PASS: both argmax comparisons MATCH; output byte-identical to pinned golden | [run-gen-g12.log](raw/run-gen-g12.log) |
| Gemma4 E4B `run-gen`, 20 tokens | PASS: both argmax comparisons MATCH; output byte-identical to pinned golden | [run-gen-e4b.log](raw/run-gen-e4b.log) |

GPU commands are serialized under `/tmp/memra-gpu.lock`; pre/post device and concurrent-process
state is retained in [gpu-context-before.log](raw/gpu-context-before.log) and
[gpu-context-after.log](raw/gpu-context-after.log). No competing compute process was reported at
either boundary. The battery began from code commit `7f76837b`; the only later commit present at
exit (`a3b4de11`) added this lane's audit evidence and did not change engine source. The extracted
verdicts and pinned-golden hashes are in [gate-summary.log](raw/gate-summary.log).

These are single-run correctness gates and support no throughput conclusion.
