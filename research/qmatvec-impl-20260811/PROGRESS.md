# qmatvec arm-1 progress

- Date: 2026-08-11
- Branch: `lane/cx-qmatvec-impl`
- Base: `49f5002d7a37291c9b551ac2f683ce2edb27d163`
- Scope: arm 1 only — dense wide-load plus `byte_perm` dot in
  `qmatvec_iq4_XS_dp4a`, preserving bytes, geometry, integer dot order, and
  floating-point accumulation order.
- Hard gate: byte-for-byte identical kernel output, including the alignment
  fallback, followed by `kernel-check` ALL GREEN, `run-gen` argmax MATCH, and
  `run-spec` K=1..8 self-consistency PASS on the local RTX 5090.
- Promotion boundary: local implementation and directional evidence only;
  Step-shape Nsys/NCU and PRO-pair validation remain orchestrator work.

## Status

1. [x] Read the lane inbox and project instructions.
2. [x] Confirm the dedicated worktree is clean and based at the study commit.
3. [x] Capture the pre-change correctness and timing baseline.
4. [x] Implement and audit arm 1.
5. [x] Run the post-change exactness battery and directional timing.
6. [x] Record the result and prepare the complete lane commit.

## Log

- 2026-08-11: Started from a clean dedicated worktree. No implementation files
  touched before this ledger.
- 2026-08-11: Pre-change `kernel-check` passed `ALL GREEN (101 cells, 1
  skipped)` on the local RTX 5090 Laptop GPU. The deterministic 11-shape
  aligned/fallback dump passed internally and hashed
  `e52f8369ce62dd250f4d203033bc5e8f6e6c1c9ba4e262ce84565d22eb92accf`.
  The first balanced-profile 315-launch synthetic sweep measured 5.727087
  ms/token-equivalent (502.706 logical GB/s); this single run is setup evidence,
  not the interleaved verdict.
- 2026-08-11: Replaced only the dense kernel's scalar per-group body with the
  existing force-inlined `expert_dot_iq4xs_g_v`. SASS contains the intended
  streaming 64-bit loads and `PRMT` dot path; registers moved 42 -> 40 while
  shared memory and launch geometry stayed fixed.
- 2026-08-11: Candidate and baseline raw outputs are byte-identical for all 11
  observed shapes at aligned and `base+4` addresses (`cmp` exit 0). Candidate
  `kernel-check`, `run-gen`, and `run-spec` gates all passed; the full local CI
  battery exited 0, and KAT-Coder's dense-IQ4_XS golden stayed token-identical.
- 2026-08-11: Two balanced-profile interleaved windows reproduced a 28.1% time
  reduction / 39.1% effective-throughput gain. Recorded a local **GO** for
  orchestrator Step Nsys/NCU and PRO-pair promotion validation in `RESULTS.md`.
- 2026-08-11: Final scoped diff and full raw evidence are ready for the required
  lane commit. No merge, tag, push, format, or perf-board operation was run.
