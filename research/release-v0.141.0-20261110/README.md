# v0.141.0 release qualification (2026-11-10)

Tested commit `24b5f3e3f20dedc18170b469dac1a4d3939e91e3` (the `release: v0.141.0` commit: the version bump, the `Cargo.lock` allowlist pin and the `docs/RELEASING.md` row) on #1205 (`802744a805`, `c3bd4a5d6f`) and #1195 (`c39411ecc2`, `1567b4e474`), on main `2628ece940`. The banked publication adds only research and qualification metadata on top of it.

## Records

| record | profile | build host | card | battery |
|---|---|---|---|---|
| `research/release-qualification/release-v0-141-0-ubuntu-24-04/` | `ubuntu-24.04` (glibc 2.39) | stock `ubuntu:24.04` image | one RTX PRO 6000 Blackwell Max-Q Workstation Edition | kernel-check ALL GREEN (126 cells, 119 executed, 10 required, 7 skipped; budget 11); Ornith-1.5-35B-A3B and Qwen3.8-27B: calibrated argmax, served-batch argmax and K=1..8 PASS |
| `research/release-qualification/release-v0-141-0-ubuntu-22-04/` | `ubuntu-22.04` (glibc 2.35) | stock `ubuntu:22.04` image | a second RTX PRO 6000 Max-Q of the same box | kernel-check ALL GREEN (126 cells, 119 executed, 10 required, 7 skipped; budget 11); Ornith-1.5-35B-A3B and Qwen3.8-27B: calibrated argmax, served-batch argmax and K=1..8 PASS |

Each record is one capture under the tested commit's own `tools/memra-gpu-run` (the canonical per-UUID flock plus the box's per-card lock as `--box-lock`, with an explicit `--timeout 900`), sealed after the wrapper exited (`tools/qualify-release.py seal`). Both leases finished clean: exit 0, no timeout, no signal, no process left in the group, no compute left on the card. The captures ran back to back, one per card: `ubuntu-22.04` from 20:40:47 to 20:46:46 (6.0 minutes), `ubuntu-24.04` from 20:47:27 to 20:53:37 (6.2 minutes), local time. The kernel oracles are the same three files as for v0.140.0 (`Qwen3.6-35B-A3B-UD-IQ4_XS`, `Qwen3.5-9B-NVFP4-MTP-GGUF`, `Qwen3.8-27B-NVFP4-Q5K-mtp.gguf`), named by `MEMRA_KC_MODELS_DIR`.

The build images carry what `release.yml` installs on GitHub's runners: `build-essential`, `git`, `python3`, `bubblewrap`, the CUDA 13.1 packages from `tools/install-cuda-ci.sh` and Rust 1.97.1, installed by rustup with the minimal profile (no `rust-src`). The builds ran as root in the image (bwrap needs it under the host's userns restriction) with `HOME`, `USER` and `LOGNAME` set to the runner's values.

## Byte reproducibility (what `release.yml` checks before packaging)

- Before the captures: GitHub's controlled `ubuntu-24.04` build of the tested commit (gpu-ci `build_reuse_only` run 42859332668; its source manifest names the commit) equals the container build of that profile byte for byte, six of six binaries. The capture kit's guard and `qualify-release.py capture --reference-build` both checked it again before each capture.
- `tools/release_qualification.py verify --head HEAD --binaries <GitHub's six ELF files> --profile ubuntu-24.04` passes against the banked record, with the six files from that run's build capsule. That is the check `release.yml` runs before packaging. `verify --profile` passes for each profile.
- No workflow builds the `ubuntu-22.04` profile before a tag. That record comes from the same kit and image recipe on `ubuntu:22.04`, and `release.yml` is its first byte check, as for v0.139.0 and v0.140.0.

## Box conditions

The box has four RTX PRO 6000 Max-Q cards on one PCIe complex. Other work on the box was stopped for the captures: right before the first capture all four cards held no compute process and 2 MiB each. The capture cards ran at their 250 W limit, which was not changed. The campaign on the other two cards restarted at 20:54:49, after the second capture had ended (20:53:37) and been sealed (20:54:18), so no other work ran on the box during either capture.

A 1 Hz sampler read each capture card's throttle flags, slowdown counters, temperature, fan and power around its capture:

| capture | samples | thermal or hardware slowdown active | slowdown counters before and after | temperature | peak power |
|---|---|---|---|---|---|
| `ubuntu-22.04` | 359 | 0 | unchanged | 38 C at start, peak 83 C, 60 C at end | 255 W |
| `ubuntu-24.04` | 369 | 0 | unchanged | 32 C at start, peak 77 C, 54 C at end | 260 W |

## Cell records for the push gate

| record | device | kinds | result |
|---|---|---|---|
| `research/release-qualification/cells/24b5f3e3f20d-rtx5090-battery-serving/` | one RTX 5090 Laptop GPU (sm_120a) | battery (kernel-check ALL GREEN, 128 cells, 6 skipped), serve-smoke | pass |
| `research/release-qualification/cells/24b5f3e3f20d-rtx5090-perf/` | the same card | perf A/B against `2628ece940` (Ornith +0.49%, Qwen3.8-27B +0.00%, no separation) | pass |
| `research/release-qualification/cells/24b5f3e3f20d-gb10-battery-perf-serving/` | one DGX Spark GB10 (sm_121a) | battery (kernel-check ALL GREEN, 131 cells, 5 skipped), perf A/B against `2628ece940` (Ornith +0.24%, Qwen3.8-27B +0.29%), serve-smoke | pass |

The 5090's perf cell ran as its own record from a cool card (59 C) on a quiet host, under the laptop's 95 W default power limit as for the three landings before it: no thermal slowdown in any of 333 one-second samples, the software power cap active throughout. The samples and the reading are in `cells/24b5f3e3f20d-rtx5090-perf-conditions/`.

## Other checks on the tested commit

- CI on the release PR (memra#1206) is green, including the complete CPU inventory.
- The `publish.yml` dry run on the release branch (run 42859802629) completed with success: the workspace packages and verifies, nothing is uploaded.
- `tools/release-guard.sh v0.141.0` passes: the workspace version and the eleven internal pins match, and the claim branch is on origin.
- `cargo fmt --check`, `git diff --check`, whole-workspace clippy with `-D warnings`, the memra-server suite (1,191 passed, 0 failed) and the 13 CPU contracts `tools/validation_plan.py` selects for the range pass.
- `tools/changelog.sh v0.140.0` drafts 4,403 bytes, under the 120,000-byte cap.
