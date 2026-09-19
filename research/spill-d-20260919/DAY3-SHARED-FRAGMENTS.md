# Day-3 lead-owned registry amendments

No shared files edited by D. No new environment variable, runtime dispatch door,
CUDA kernel entry point, FFI shim, numeric program, or performance default added.
The two MEMRA reads below reuse existing build controls. They are build configuration,
not default-OFF experiments; a 14-day experiment expiry does not apply.

## FLAGS.md (same landing change as bootstrap)

Update the existing §build rows, not duplicate them:

- `MEMRA_NVCC`: actual engine resolution is explicit override, then existing
  `CUDA_HOME`/`CUDA_PATH`/`CUDA_ROOT` toolkit, otherwise newest runnable release in
  PATH and `/usr/local/cuda*`. The current table's bare `nvcc on PATH` default is
  incomplete. `tools/tier-rig-bootstrap.sh` mirrors this precedence, requires CUDA
  >=13 with `compute_120a`, and pins the accepted absolute path for its builds.
- `MEMRA_CUDA_ARCH`: keep all existing generic behavior. Add that the 5090-only
  bootstrap refuses an explicit value other than `120a` and pins `120a` for its
  builds. This does not change any runtime or other-device default.

## TESTING.md (new runbook pointer only)

Generic spill development bootstrap/first-hour ordering:
`research/spill-d-20260919/RIG-DAY1.md`. Minimum source `914229ae` plus the reviewed
bootstrap/collector. The first native engine/server compile is blocking. Offline
CPU/Linux-target checks and stub bootstrap are **not** native build, GPU, serving,
P2P, four-tier or release qualification.

## Already applied — do not re-apply

Lead `914229ae3f706c0581f5d7f7bb415a14795b1640` adds the engine's `memra-tier`
dependency and explicit `storage-bench` binary target. D merged it at `ca974cef`.
The server needs no new direct memra-kv dependency: `memra_engine::cache` already
re-exports it. D made no further Cargo/lib amendments.

KERNELS/board/decisions fragments: none (no engine kernels or measured/default change).
