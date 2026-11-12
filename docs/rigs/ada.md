# Ada

NVIDIA Ada GPUs (compute capability 8.9): a portable compatibility lane for correctness bring-up
and hardware reports, not a performance target.

| | |
|---|---|
| **Build target** | `sm_89`, auto-detected on an 8.9 card (`MEMRA_CUDA_ARCH=89` forces it); source build only |
| **Prebuilt** | no (retired 2026-09-02) |
| **Push gate** | no (`tools/change-impact.toml`); no CI cell compiles `89` |
| **Maintained** | no: not push-gated, and `tools/change-impact.toml` declares no Ada device; compatibility reports only |

## Run it

Build from source ([INSTALLATION.md](../INSTALLATION.md)), then run `kernel-check`. A clean
compatibility report is a useful contribution even with no optimization proposed
([contributing](../../CONTRIBUTING.md)).

## Limits

- No Blackwell performance or default claim transfers to Ada.
- No pack record names an Ada topology, and no model has a measured route here.
