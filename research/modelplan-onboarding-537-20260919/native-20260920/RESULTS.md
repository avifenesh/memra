# Issue 537 native regression results — 2026-09-20

All seven planned native regression phases passed at
`fd7ce385b7861e6b96b98682906b5e8ff6ae31c2`. This is evidence for the changed model-load
semantics on the tested paths, not a promotion of pack support states or a claim that every
kernel/model/serving surface is qualified.

## Conditions and identity

- NVIDIA RTX PRO 6000 Blackwell Server Edition, 600 W class, 97,887 MiB reported memory,
  sm_120 / 188 SMs. One-, two- and three-card phases used exactly their assigned physical
  UUID sets through exclusive per-card leases.
- CUDA 13.1.115, driver 595.58.03, Rust 1.97.1 / LLVM 22.1.6, Linux x86_64.
- XFS on Ceph RBD. No local-NVMe, spill-performance, throughput or hardware-default
  promotion follows from these runs. Timing lines in raw runner output are incidental.
- Artifacts are pinned in [artifacts.lock.json](../artifacts.lock.json); all 45 selected
  files were SHA256-verified before and after staging. No alternate repack marker or
  unlisted model file was accepted.
- Every successful phase used the binaries in the archived `replacement-build-a4/build.json`.
  Exact executable hashes, source commit, runtime environment, argv, GPU state and lease
  metadata are retained. Failed/aborted earlier attempts are not counted as passes.

## Results

| Phase | Cards | Actual gate result |
|---|---:|---|
| Focused Step factor load | 1 | 48 decode rows: HF-derived factors and explicit factor tensor yield bit-identical logits. The full-head 64-value storage case preserves the same consumed prefix. The omitted-factor diagnostic mutation changes logits by max **1.4663358**, proving non-vacuity. |
| Qwen3-1.7B Q8_0 dense | 1 | `kernel-check`: **ALL GREEN (99 cells, 22 skipped)**. Prefill/decode and batched-prime/tokenwise argmax MATCH; 64 tokens generated. |
| Ornith-1.5-35B-A3B NVFP4+MTP | 1 | `run-gen` argmax MATCH; all **K=1..8** speculative self-consistency arms PASS. |
| Ornith HTTP smoke | 1 | Owned loopback listener verified against its process socket. One completion: 16 prompt / 32 completion tokens; speculative path engaged; clean drain and GPU-worker shutdown. |
| Step IQ4_XS + Q8_0 MTP, PP-2 | 2 | Argmax **666** MATCH for prefill/decode and batched-prime/tokenwise. All **K=1..8** self-consistency arms PASS. |
| Step required kernel cells | 1 | Explicit `--require-manifest tools/kernel-check-step35.cells --require-cell iq4xs-mmq-real` against the actual Step artifact: **ALL GREEN (98 cells, 21 skipped)**. No checkpoint renaming or substitution. |
| Official Step FP8, PP-3 | 3 | Verify-class prefill/decode argmax **666** MATCH, max logit difference **3.338e-5**. All **K=1..8** self-consistency arms PASS. The FP8 checkpoint is independent of the GGUF compatibility arm. |

The generation/spec runs use the committed `research/e2e/prompts/board-2048.txt` and 64
generated tokens. It tokenizes to 2,048 tokens for the Qwen/Ornith controls and 1,942 for
Step, crossing the 512-token window. The FP8 comparison uses the standing verify-class
prefill reference; it does not relabel fresh-f32-KV `forward_last` as that numeric class.

All seven final wrapper leases finished with exit code 0, no interruption and no lingering
compute. The first focused attempt failed because the test reused an `Engine` after its
MoE cache had been constructed. A fresh Engine per arm repaired that test-lifecycle error;
no production math or tolerance was changed. Its failed raw phase is retained separately.

## Kernel coverage limits

[kernel-census-dense.json](kernel-census-dense.json) preserves the exact 99-cell census:
77 executed cell names and all 22 skipped names, verbatim reasons and applicability. The
22 comprise one absent served-router capture and 21 missing named weight-oracle cells;
they are **not** architecture/device-count exclusions. All nine required Step-manifest
cells were executed in that run. The additional Step run explicitly closed the relevant
`iq4xs-mmq-real` gap using the actual Step checkpoint.

Remaining named-oracle omissions are still unexecuted. In particular, the separate 27B
manifest's `DUAL-BATCHED-AUX` cell was not established here. NVFP4/GDN and Q5_K oracle gaps
can concern mechanisms used by Ornith; its successful end-to-end runs do not turn those
omitted cells into passes. Gemma Q4_0 and the other absent stored formats are identified
from actual selected-artifact tensor inventories, not inferred from filenames. This feeds
the existing #546/#484 coverage work and does not establish the complete release battery.

Two additional bf16-stage subcomparisons at T=64/100 are outside the 22-cell skip tally.
The active hp path uses f16 P/V, so its executed oracle-band checks apply instead of a
bit-identity claim against the different bf16 program. The exact messages are retained.

## Raw evidence

[raw-receipts.tar.gz](raw-receipts.tar.gz) contains 167 original receipt files, including
raw stdout/stderr, command verdicts, separate final leases, model inventories, staging
receipts and executable identity. [manifest.json](manifest.json) binds the archive and
every uncompressed file by SHA256. Raw contents were scanned against the public-boundary
policy before packaging; provider addresses, allocation identifiers and credentials remain
in private coordination records.

The archive contains `phases/`, `leases/`, `replacement-build-a4/`, metadata/toolchain,
fetch/staging receipts, and the detailed census. The only failed GPU phase in it is
`phases/replacement-focused-a1`; the passing replacement is `replacement-focused-a3`.
To inspect the original files, extract the archive into a new directory and compare file
hashes with the manifest. No receipt authorizes merge, tag, deployment, or broader model
support-state promotion by itself.
