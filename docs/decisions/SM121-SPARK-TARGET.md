# GB10 (DGX Spark) builds sm_121a, a 120a-family target (2026-11-01)

**Status:** decided and landed as a build target. Compile-covered in CI (the `121a` cell of the
arch matrix in `ci.yml` and `ci-public.yml`); not yet hardware-qualified. No model gains a support
state from this record.

## Question

GB10 reports compute capability 12.1. Before this change, `build.rs` auto-detection mapped 12.1 to
`120a`, and the `Engine::new` guard admitted a `120a` build on a 12.1 device. The fatbins carry
arch-specific SASS and no PTX, and `a` SASS loads only on the compute capability it names, so that
build could pass the guard and then fail at the first module load. What should a GB10 build
compile, and which policy should it run?

## Decision

Two names per build, one table (`crates/memra-engine/src/cuda_arch.rs`, shared by `build.rs` and
the engine):

- **Target** (`MEMRA_BUILT_CUDA_TARGET`): the exact `-gencode` value. 12.1 auto-detects `121a`;
  `MEMRA_CUDA_ARCH=121a` forces it. The startup guard keys on the target: `120a` runs on 12.0
  only, `121a` on 12.1 only. A mismatch refuses with the target that device needs.
- **Family** (`MEMRA_BUILT_CUDA_ARCH`, `memra_engine::BUILT_CUDA_ARCH`): the policy key. `121a`
  is in the `120a` family, so it compiles the same source set (no stubs: the NVFP4 W4A4, W4A8 and
  block-FP8 MMQ translation units are the real ones) and takes every 120a-keyed default. The
  ~30 runtime sites that compare against `"120a"` did not change.

The two CUTLASS archives (`MEMRA_CUTLASS`, `MEMRA_DSV4_CUTLASS`) compile with a fixed sm_120a
gencode, so they refuse every target other than `120a`. memra-probe follows the same target rule
for its own fatbin.

## Rejected: one `120f` family binary for 12.0 and 12.1

CUDA 13.1 assembles the block-scale MMA translation unit (`cu/mmq_nvfp4_w4a8.cu`) for both
`compute_120f` and `compute_121a`, so a family build is technically possible. It was rejected:

- As the default, `120f` would change the SASS target of the primary 12.0 build (RTX 50-series,
  RTX PRO 6000). That is a compiler/build-default change, which needs the full GPU battery, and
  the naked `120a` build is meant to stay byte-identical.
- As a GB10-only option, it buys nothing over `121a`. GB10 hosts are aarch64, while the release
  prebuilt and every 12.0 build host so far are x86_64, so one binary would not cover both
  anyway. A GB10 runs a binary built from source on a GB10 host, and `121a` is exactly what that
  build auto-detects.

## What the change proves, and what it does not

Proved on the build host (CUDA 13.1, x86_64, no GPU used): `MEMRA_CUDA_ARCH=121a` builds
`memra-server`; the `120a` build's fatbins and MMQ archive are byte-identical to `origin/main`'s.
Unit tests pin the capability-to-target map, the family map and the guard
(`cuda_arch::tests`).

Not yet proved, owed by the first GB10 run:

- The `121a` binary loads every module on a GB10 and passes `kernel-check`, `run-gen` argmax and
  `run-spec` K=1..8 there, built on the GB10 host's own toolkit (aarch64).
- The 120a-keyed defaults (`gdn_mma_default_on`, `serve_devpenalty`, the NVFP4/FP8 MMQ paths)
  were measured on RTX 5090 and RTX PRO 6000 only. GB10 has 48 SMs, unified LPDDR5x memory and
  far lower memory bandwidth; each default it inherits is a one-rig extrapolation until a GB10
  A/B confirms or re-keys it on the device.
- Memory sizing on unified memory: VRAM budgets read `cuMemGetInfo`, which on GB10 reports the
  shared system pool.
