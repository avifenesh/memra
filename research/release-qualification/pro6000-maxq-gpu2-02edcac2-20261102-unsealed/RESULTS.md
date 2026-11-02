# Unsealed release battery on one RTX PRO 6000 (physical GPU 2)

Status: **UNQUALIFIED**. There is no `record.json`, and `current.json` is unchanged. This is evidence for PR #1025, not a release qualification.

Tested source: `02edcac2a97f929d50bbe3f0d55a4709dee05e0f` (tree `c7c249db6123d0b769f35ff5e62b91986f05dd48`).
Source inputs: `fb8ed9ab1f01df5ea05c028b8f781ea965a88797347553c17f5eddaf9f876e28` (201968 entries). They were identical before and after the run.

## Why it cannot seal

`qualify-release.py capture` accepts only physical NVML index 0. `observe_hardware` refused this card with this message, recorded in `run.unsealed.json`:

> battery headroom queries NVML GPU0; leased CUDA UUID differs

The validator has the same rule: `devices[0].index == "0"` and lease index 0. GPU 0 on this host was busy with other work and could not be used. The battery itself selects the card correctly: `release-battery.sh` resolves CUDA device 0 to its physical UUID through `resolve-physical-gpu.py`. The capture tool has not caught up yet (#264 follow-up). No qualification rule was changed for this run.

The host also has no `memra-gpu-run` wrapper, so capture's lease and ancestry checks could not be satisfied either.

## What ran

- **Build.** `qualify-release.py build`, controlled v3 recipe, inside bubblewrap 0.11.1. Rust 1.97.1, CUDA 13.4.92, `MEMRA_CUDA_ARCH=120a`, 8 jobs. Source-view identities matched before and after the build. `build.json` binds all six ELF digests.
- **Battery.** `tools/release-battery.sh --evidence-dir cells` on `GPU-9f8ecd14-35a9-f104-1dd9-c4757538bb34`, an RTX PRO 6000 Blackwell Max-Q Workstation Edition (CC 12.0, driver 615.71.09, PCI C1:00.0). The host runs Ubuntu 26.04 with glibc 2.43.
- **Runner.** A runner reused capture's own helpers: clean source, `validate_build`, binary staging and identity, `environment()` with no ambient `MEMRA_*` variables, and model/oracle hashing. It also held the canonical exclusive per-UUID FLOCK and captured 250 ms telemetry.

| Cell | Result |
|---|---|
| Kernel | PASS: ALL GREEN (110 cells, 104 executed, 10 required, 6 skipped; budget 11) |
| ornith-ai/ornith-1.5-35b-a3b argmax | PASS: 1 flip, 0 bad under the unchanged calibrated gate |
| ornith-ai/ornith-1.5-35b-a3b run-spec | PASS: K=1..8 self-consistency, identical to plain target (8/8) |
| qwen/qwen3.8-27b argmax | PASS: 0 flips, 0 bad |
| qwen/qwen3.8-27b run-spec | PASS: K=1..8 self-consistency, identical to plain target (8/8) |

The router lines are in `cells/1-kernel.log` at lines 516-517: real router batch-twin bit-identity over 32 m-points from 1 to 2048, and m-invariance. Both had zero mismatches.

The six kernel skips are the same missing oracles as the banked Ubuntu24 records: ornith-1.0 Q4_K_M, KAT-Coder/Step-3.7 IQ4_XS, Qwen3.6-27B NVFP4, both gemma-4 q4_0, and sigrouter replay.

## Run checks

- Source, binary and model identities matched before and after the battery.
- No compute process was left on the card afterwards.
- Peak usage: 77784 MiB, 88 C, 281 W.
- Wall time: 3.1 min.

The Qwen3.6-35B-A3B-UD-IQ4_XS oracle used here is 18209036576 bytes, `df27a780…`. The banked Ubuntu24 records used a different revision: 17730509792 bytes, `649d7508…`. The other three files match those records byte for byte.

## What this does not cover

- **No sealed record.** Sealing needs capture on physical GPU0 under the coordinator wrapper.
- **No second OS profile.** There is no Ubuntu22 run, and no record for the Ubuntu24 CI profile.
- **Not on the PRO 6000 Server Edition.** The banked records ran on the Server Edition; this card is a Max-Q.
- **No HTTP serving, Step or multi-card claims.**
- **No throughput or timing claims.**

The tested executables are archived off this host by digest.
