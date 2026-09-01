# FOCUS NVFP4 recon — progress log

Lane: `lane/cx-focusrecon` in `/home/avifenesh/projects/wt-cx-focusrecon`, created from
`main` at `dfc921568b4552122f69ac9d18685def8331206f`.
Deliverable: `focus-recon.md` answering whether FOCUS Coupled-Relaxation Scaling can be
expressed by the existing GGUF NVFP4 block at repack time or belongs only in source-side
quantization.
Scope: research documentation only; no code, model artifacts, scored-arm changes, generated
performance surfaces, push, merge, or tag.

## Work log

- [x] Confirmed the dedicated branch/worktree is clean and exactly based on `main`.
- [x] Read the lane inbox and locked the requested Hy3/NVFP4 scope.
- [x] Verified the FOCUS mechanism and optimization cost from the primary paper.
- [x] Traced the repository's NVFP4 block, writer/repack path, and dequant consumers.
- [x] Wrote the evidence-backed format/repack verdict and future-arm placement.
- [x] Reviewed the documentation-only diff and relevant static checks; ready for the requested
  local commit.

## Research receipts

- arXiv currently exposes only FOCUS v1, published 2026-08-03.
- AngelSlim `main` resolved to `67394ce55f6b6cfae702575fbc3c7a05c13fdd74`; note links pin
  packing, export, and offline-export evidence to that revision.
- Verdict: class (a), meaning quantizer-side and runtime-layout-free. The coefficient is transient;
  the current 36-byte block stores only compliant UE4M3 scales and the E2M1 codes it helped select.
- Verified every repository-relative evidence target and all four external primary-source links.
  `git diff --check` is clean, and the locked-arm file retains SHA-256
  `bda2339bef241ea36bf3800929971587e952cabf0217f683adfb7a6be0ae94f1`.
- Re-read the lane inbox after drafting; it contains no steering beyond the original task.

## Constraints carried forward

- `research/per-expert-quant/arms.lock.json` remains byte-untouched.
- All five scored arms remain locked and continue to quantize the pinned BF16 source.
- No public-eval evidence, calibration choice, artifact generation, or GPU work belongs to this
  mechanism recon.
