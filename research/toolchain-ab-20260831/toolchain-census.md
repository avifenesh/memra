# Toolchain census, container (fleet build path) vs on-box native

Same memra commit both arms: 3999a92a6e18a231ce8e18fb2b6f37997b00e882
(fleet-convergence engine pin; `git log -1` receipts in each build log,
baked `system_fingerprint` memra-3999a92a6e18 verified in every streamed response).

| axis | CONTAINER arm (C) | ON-BOX arm (N) |
|---|---|---|
| build host | the rig, cached fleet image `darklanes-serve-build:cuda131-u2204` (image id 92edbe46b7d9, created 2026-08-30) | the bench box, native toolchain |
| base OS | Ubuntu 22.04.5 | Ubuntu 24.04 |
| nvcc | CUDA 13.1, V13.1.115 (built 2025-12-16) | CUDA 13.2, V13.2.51 (built 2026-03-02) |
| ptxas | V13.1.115 | V13.2.51 |
| rustc | 1.98.0 (88d9e12ae 2026-08-18) | 1.98.0 (88d9e12ae 2026-08-18), IDENTICAL |
| cargo | 1.98.0 (797e8a9bc 2026-08-05) | 1.98.0, IDENTICAL |
| host gcc | 11.4.0 (Ubuntu 11.4.0-1ubuntu1~22.04.3) | 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04.1) |
| glibc (link-time) | 2.35 | 2.39 |
| MEMRA_NVCC | /usr/local/cuda-13.1/bin/nvcc | /usr/local/cuda-13.2/bin/nvcc (build log: "nvcc from MEMRA_NVCC" confirmed) |
| MEMRA_CUDA_ARCH | 120a (explicit) | 120a (explicit) |
| cargo profile | --release -p memra-server, CARGO_BUILD_JOBS=10 | --release -p memra-server, CARGO_BUILD_JOBS=24 |
| crates compiled | 137, 3m00s | 137, 3m59s |
| binary md5 | dc58e8c52f8d3bce20941fb69736579b | 93db82e0599933ff1af05a201ae3a5c3 |
| binary sha256 | b80a7b2867f8ab55c35eca1b65ff5597de3963a81b9b980040b9bfbca218e553 | caff68db857323952167a31521dc3d9d7b6b730436640c8f6945822ee63f43d4 |
| markers | ring-restore, suffix-door, step-vision: all present | all present |

Runtime note: at run time BOTH binaries resolve libcudart.so.13 / libcublas.so.13 /
libcublasLt.so.13 to the bench box's CUDA 13.2 lib64 (ldd receipts), so this A/B
isolates the compiled-in device code (nvcc/ptxas 13.1-vs-13.2 kernels for sm_120a)
plus host codegen (gcc 11.4-vs-13.3, glibc 2.35-vs-2.39). rustc is identical by
measurement, not by assumption.

Provenance of the two production reference points this cell speaks to:
- The 140.6 tok/s seal (2026-08-29) ran a binary built ON the serving box; its build
  log records "nvcc auto-detected CUDA 13.2" and "MEMRA_CUDA_ARCH auto-detected 120a",
  i.e. the same 13.2-class toolchain as arm N.
- Every fleet binary since is built by the container path with CUDA 13.1, the same
  image generation as arm C.
