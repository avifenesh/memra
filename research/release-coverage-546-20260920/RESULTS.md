# Release battery coverage — native validation, 2026-09-20

The release battery passed on the exact source `a0a27c3f2d57108a5e16da3bb60d9c4aa116adf0`, including main `746bba2b4238bfa122695a139b73357fb9be9076`, on one exclusively locked RTX PRO 6000 Blackwell Server Edition card in a designated non-serving four-card host. This validates the coverage enforcement in #546 / PR #553.

| Gate | Result |
| --- | --- |
| Kernel coverage | PASS: 110 cells accounted, 103 executed, all 10 required cells executed, 7 named skips within the fixed ceiling of 11 |
| Ornith 1.5 calibrated argmax | PASS: 1 near-tie flip, 0 bad flips |
| Ornith 1.5 greedy speculation | PASS: K=1 through 8 each exactly once, 32 generated tokens |
| Qwen3.8 calibrated argmax | PASS: 0 flips, 0 bad flips |
| Qwen3.8 greedy speculation | PASS: K=1 through 8 each exactly once, 32 generated tokens |
| Lease and child | Both exit 0; no timeout, interruption, or lingering compute |

The seven effective skips are `dtype5-Q3_K`, `f16g-kq-direct-ornith`, `iq4xs-mmq-real`, `nvfp4-27b-shape`, `q4_0-mmq`, `q4_0-sk-arm`, and `sigrouter-served-replay`. Their unavailable artifact/tensor/capture reasons remain in the raw trace. None is a required manifest cell. The required DUAL-BATCHED-AUX cell executed using the actual named 9B oracle, not a renamed substitute. This is coverage and exactness evidence, not a throughput benchmark or qualification of the skipped cells.

## Reproduction and provenance

`build-candidate.py` built six native ELF x86-64 executables in a fresh target directory using Rust 1.97.1, CUDA 13.1, `MEMRA_CUDA_ARCH=120a`, and `CUDA_VISIBLE_DEVICES=`. It checked the clean source before and after compilation and hashed both built and staged executable bytes. `build.json` records the exact command, source tree, compiler versions, build-log digest and each executable digest. No compile-only stub build was used.

The locked command was `bash -x tools/release-battery.sh`, with `MEMRA_KC_MODELS_DIR=/data/models/kernel-oracles` and the normal roster paths backed by the hash-verified model files in `artifacts.json`. No missing-model allowance, narrowed kernel selector, or skip-ceiling override was supplied. `lease.json` records the physical card UUID, command, exit status and cleanup.

`battery-trace.log.gz` is the complete Bash xtrace. It preserves successful command-substitution output that the normal battery otherwise summarizes. Bash prints some data more than once; the gate parses each captured producer output once, so global counts of repeated strings in the trace are not run counts. The final receipt and raw producer outputs are both retained.

This publication adds only research evidence; it does not rebuild or relabel the tested executables. Future source/build changes require their own qualification. The exact executables and original receipt archive are preserved off the rented host; the public manifests identify their bytes without committing large binaries.

## CPU and review checks

The actual-battery fixture suite passed all 11 tests, including narrowed, duplicate, missing, failed, malformed, sampled and complete outputs. Card-isolation fixtures, formatting and diff checks passed. All nine hosted CI jobs and GitGuardian passed on the tested source (run 35497173574). Five independent domain reviews found no unresolved issue at that source.
