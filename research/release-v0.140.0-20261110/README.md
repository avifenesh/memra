# v0.140.0 release qualification (2026-11-10)

Tested commit `fc43c21e0781dfa9489e84ae3f86f259b44477e6` (the `release: v0.140.0` commit on main `fcd66f8963`: the version bump, the `Cargo.lock` allowlist pin and the `docs/RELEASING.md` row). The banked publication adds only research and qualification metadata on top of it.

## Records

| record | profile | build host | card | battery |
|---|---|---|---|---|
| `research/release-qualification/release-v0-140-0-ubuntu-24-04/` | `ubuntu-24.04` (glibc 2.39) | stock `ubuntu:24.04` image | one RTX PRO 6000 Blackwell Max-Q Workstation Edition | kernel-check ALL GREEN (122 cells, 115 executed, 10 required, 7 skipped; budget 11); Ornith-1.5-35B-A3B and Qwen3.8-27B: calibrated argmax, served-batch argmax and K=1..8 PASS |
| `research/release-qualification/release-v0-140-0-ubuntu-22-04/` | `ubuntu-22.04` (glibc 2.35) | stock `ubuntu:22.04` image | a second RTX PRO 6000 Max-Q of the same box | kernel-check ALL GREEN (122 cells, 115 executed, 10 required, 7 skipped; budget 11); Ornith-1.5-35B-A3B and Qwen3.8-27B: calibrated argmax, served-batch argmax and K=1..8 PASS |

Each record is one capture under the tested commit's own `tools/memra-gpu-run` (the canonical per-UUID flock plus the box's per-card lock as `--box-lock`, with an explicit `--timeout 900`), sealed after the wrapper exited (`tools/qualify-release.py seal`). Both leases finished clean: exit 0, no timeout, no signal, no process left in the group, no compute left on the card. The captures ran back to back, one per card: `ubuntu-22.04` took 5.8 minutes, `ubuntu-24.04` 5.6 minutes. The kernel oracles are the same three files as for v0.139.0 (`Qwen3.6-35B-A3B-UD-IQ4_XS` at 649d7508..., `Qwen3.5-9B-NVFP4-MTP-GGUF` at 52c9cceb..., `Qwen3.8-27B-NVFP4-Q5K-mtp.gguf` at 1facf36c...), named by `MEMRA_KC_MODELS_DIR`.

The build images carry what `release.yml` installs on GitHub's runners: `build-essential`, `git`, `python3`, `bubblewrap`, the CUDA 13.1 packages from `tools/install-cuda-ci.sh` and Rust 1.97.1. The builds ran as root in the image (bwrap needs it under the host's userns restriction) with `HOME`, `USER` and `LOGNAME` set to the runner's values.

## Byte reproducibility (what `release.yml` checks before packaging)

- GitHub's controlled `ubuntu-24.04` build of the tested commit (gpu-ci `build_reuse_only` run 42722858661; its source manifest names the commit) equals the container build of that profile byte for byte, six of six binaries.
- `tools/release_qualification.py verify --head HEAD --binaries <GitHub's six ELF files> --profile ubuntu-24.04` passes against the banked record. That is the check `release.yml` runs before packaging. The same check passes for each profile's own container binaries.
- No workflow builds the `ubuntu-22.04` profile before a tag. That record comes from the same kit and image recipe on `ubuntu:22.04`, and `release.yml` is its first byte check, as for v0.139.0.

## Superseded records: a toolchain component changed the bytes

A first pair of captures of this same commit passed and is not banked. Their builds could not be reproduced by GitHub.

- What the byte check found: GitHub's controlled `ubuntu-24.04` build matched the first container build in 0 of 6 binaries. `.text`, the linker and the gcc were the same. Only `.rodata` differed, by 456 bytes in `tok-parity`.
- Cause: the build host's Rust 1.97.1 had been copied from another machine and carried the `rust-src` component. With it, rustc writes the standard library's panic locations as `/toolchain/lib/rustlib/src/rust/library/...`. A rustup minimal toolchain, as on GitHub's runners, has no such directory and writes `/rustc/<hash>/library/...`.
- Fix: the component was removed (`rustup component remove rust-src`), both profiles were rebuilt, and the `ubuntu-24.04` rebuild equals GitHub's build six of six. The records above are captures of those rebuilds.
- The check has teeth: `verify --binaries` on the first build's six files answers `rebuilt binaries match no qualified build`.
- The capture kit now refuses before a capture when the toolchain lists `rust-src`, when its `ubuntu-24.04` build differs from GitHub's `build.json`, or when one of the profile's six binaries embeds a sysroot source path. The order is: byte check first, captures second.

The superseded captures were clean runs on the same two cards (no throttle sample, both leases clean). They qualified bytes nobody else can rebuild, so they prove nothing about what `release.yml` packages.

## Box conditions

The box has four RTX PRO 6000 Max-Q cards on one PCIe complex. Other work on the box was stopped for the captures: the two capture cards held no compute process and 2 MiB, a third card's job was paused with its memory held, and the fourth was idle. The capture cards ran at their 250 W limit, which was not changed.

A 1 Hz sampler read each capture card's throttle flags, slowdown counters, temperature, fan and power around its capture:

| capture | samples | thermal or hardware slowdown active | slowdown counters before and after | temperature |
|---|---|---|---|---|
| `ubuntu-22.04` | 345 | 0 | unchanged | 67 C at start, peak 88 C, 64 C at end |
| `ubuntu-24.04` | 336 | 0 | unchanged | 38 C at start, peak 78 C, 56 C at end |

## Cell records for the push gate

| record | device | kinds | result |
|---|---|---|---|
| `research/release-qualification/cells/fc43c21e0781-rtx5090-battery-serving/` | one RTX 5090 Laptop GPU (sm_120a) | battery, serve-smoke | pass |
| `research/release-qualification/cells/fc43c21e0781-rtx5090-perf/` | the same card | perf A/B against `fcd66f8963` (Ornith +0.67%, Qwen3.8-27B -2.31%, no separation) | pass |
| `research/release-qualification/cells/fc43c21e0781-gb10-battery-perf-serving/` | one DGX Spark GB10 (sm_121a) | battery, perf A/B against `fcd66f8963` (Ornith +0.29%, Qwen3.8-27B +0.07%), serve-smoke | pass |

The 5090's perf cell ran as its own record from a cool card (59 C) on a quiet host. The software thermal slowdown flag was active in 14 of 291 one-second samples, all in Qwen3.8-27B runs and spread over both arms (A 5, B 9). That is the laptop's steady state for this load; the samples and the reading are in `cells/fc43c21e0781-rtx5090-perf-conditions/`, and memra#1160 asks the perf cell to record throttle flags itself.

## Other checks on the tested commit

- CI on the release PR (memra#1161) is green, including the complete CPU inventory.
- The `publish.yml` dry run on the release branch (run 42722863609) completed with success: the workspace packages and verifies, nothing is uploaded.
- `tools/release-guard.sh v0.140.0` passes: the workspace version and the eleven internal pins match, and the claim branch is on origin.
- `tools/changelog.sh v0.139.0` drafts 9,977 bytes, under the 120,000-byte cap.
