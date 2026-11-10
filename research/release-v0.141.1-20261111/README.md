# v0.141.1 release qualification (2026-11-11)

Tested commit `40414f5f82f930b6a4f0c580b0d775eb996a874c`, the `release: v0.141.1` commit (the version bump, the `Cargo.lock` allowlist pin and the `docs/RELEASING.md` row). It sits on #1213 (`8f926af60b` to `bb9f7fd7a6`), #1214 (`a7f3c098da`) and #1210 (`dca7646fb7` to `fd254e64ab`, replayed with its tree unchanged), on main `3de59bbf80` (v0.141.0). The banked publication adds only research and qualification metadata on top of it.

## Records

| record | profile | build host | card | battery |
|---|---|---|---|---|
| `research/release-qualification/release-v0-141-1-ubuntu-24-04/` | `ubuntu-24.04` (glibc 2.39) | stock `ubuntu:24.04` image | one RTX PRO 6000 Blackwell Max-Q Workstation Edition | kernel-check ALL GREEN (126 cells, 119 executed, 10 required, 7 skipped; budget 11); Ornith-1.5-35B-A3B and Qwen3.8-27B: calibrated argmax, served-batch argmax and K=1..8 PASS |
| `research/release-qualification/release-v0-141-1-ubuntu-22-04/` | `ubuntu-22.04` (glibc 2.35) | stock `ubuntu:22.04` image | a second RTX PRO 6000 Max-Q of the same box | kernel-check ALL GREEN (126 cells, 119 executed, 10 required, 7 skipped; budget 11); Ornith-1.5-35B-A3B and Qwen3.8-27B: calibrated argmax, served-batch argmax and K=1..8 PASS |

Each record is one capture under the tested commit's own `tools/memra-gpu-run` (the canonical per-UUID flock plus the box's per-card lock as `--box-lock`, with an explicit `--timeout 900`), sealed after the wrapper exited (`tools/qualify-release.py seal`). Both leases finished clean: exit 0, no timeout, no signal, no process left in the group, no compute left on the card. The captures ran back to back, one per card: `ubuntu-22.04` from 00:58:37 to 01:04:43 (6.1 minutes), `ubuntu-24.04` from 01:05:24 to 01:11:23 (6.0 minutes), local time. The kernel oracles are the same three files as for v0.141.0 (`Qwen3.6-35B-A3B-UD-IQ4_XS`, `Qwen3.5-9B-NVFP4-MTP-GGUF`, `Qwen3.8-27B-NVFP4-Q5K-mtp.gguf`), named by `MEMRA_KC_MODELS_DIR`.

The build images and toolchain are those of v0.141.0: `build-essential`, `git`, `python3`, `bubblewrap`, the CUDA 13.1 packages from `tools/install-cuda-ci.sh` and Rust 1.97.1, installed by rustup with the minimal profile (no `rust-src`). The builds ran as root in the image (bwrap needs it under the host's userns restriction) with `HOME`, `USER` and `LOGNAME` set to the runner's values.

## Byte reproducibility (what `release.yml` checks before packaging)

- Before the captures: GitHub's controlled `ubuntu-24.04` build of the tested commit (gpu-ci `build_reuse_only` run 42912845832; its source manifest names the commit) equals the container build of that profile byte for byte, six of six binaries. The capture kit's guard and `qualify-release.py capture --reference-build` both checked it again before each capture.
- `tools/release_qualification.py verify --head HEAD --binaries <GitHub's six ELF files> --profile ubuntu-24.04` passes against the banked record, with the six files from that run's build capsule. `verify --profile` passes for each profile.
- No workflow builds the `ubuntu-22.04` profile before a tag. That record comes from the same kit and image recipe on `ubuntu:22.04`, and `release.yml` is its first byte check, as for the releases before it.

## Box conditions

The box has four RTX PRO 6000 Max-Q cards on one PCIe complex. Other work on the box was stopped for the captures: right before the first capture all four cards held no compute process and 2 MiB each, and they still did after the second capture. The capture cards ran at their 250 W limit, which was not changed. The first capture's card had come off other work shortly before, so it started warm (61 C).

A 1 Hz sampler read each capture card's throttle flags, slowdown counters, temperature, fan and power around its capture:

| capture | samples | thermal or hardware slowdown active | slowdown counters before and after | temperature | peak power |
|---|---|---|---|---|---|
| `ubuntu-22.04` | 365 | 0 | unchanged | 61 C at start, peak 87 C, 63 C at end | 258 W |
| `ubuntu-24.04` | 358 | 0 | unchanged | 37 C at start, peak 78 C, 56 C at end | 255 W |

## Cell records for the push gate

| record | device | kinds | result |
|---|---|---|---|
| `research/release-qualification/cells/40414f5f82f9-rtx5090-battery-serving/` | one RTX 5090 Laptop GPU (sm_120a) | battery (kernel-check ALL GREEN, 128 cells, 6 skipped), serve-smoke | pass |
| `research/release-qualification/cells/40414f5f82f9-rtx5090-perf/` | the same card | perf A/B against `3de59bbf80` (Ornith +0.27%, Qwen3.8-27B -0.26%, no separation) | pass |
| `research/release-qualification/cells/40414f5f82f9-gb10-battery-perf-serving/` | one DGX Spark GB10 (sm_121a) | battery (kernel-check ALL GREEN, 131 cells, 5 skipped), perf A/B against `3de59bbf80` (Ornith -0.15%, Qwen3.8-27B +0.07%), serve-smoke | pass |

The 5090's perf cell ran as its own record from a cool card (59 C) on a quiet host, under the laptop's 95 W default power limit as for the landings before it: no thermal slowdown in any of 334 one-second samples, the software power cap active throughout. The samples and the reading are in `cells/40414f5f82f9-rtx5090-perf-conditions/`.

## Scoped extras on GB10

#1210 and #1213 both change the speculative burst's hooks in `crates/memra-server/src/worker.rs`, and each lane ran its device gate on its own branch. Two scoped extras ran the merged binary: `research/release-qualification/cells/40414f5f82f9-gb10-burst-hooks/` (not a cell record; the gate does not read it).

| extra | what ran | result |
|---|---|---|
| memra#1210's stop-string gate (`tools/spec-stop-string-gate.py`) | a plain boot and the MTP boot at the defaults, greedy and temperature 1 | stop rows equal to plain decode 13/13 per sampler, completion counts equal to the lane's GB10 rows, close and coverage pass: `STOP-STRING GATE: PASS` |
| memra#1213's stall cell (`research/spec-entry-yield-bound-20261110/tools/stall_cell.py`) | greedy, default gate, 2 clients at 0 ms and 4 clients at 0 and 200 ms | the long request advances in every cell (74, 57 and 50 tokens in the 30 s window, against 1 on v0.141.0), longest gap 0.96, 1.79 and 1.56 s, texts prefix-equal to the lane's and to v0.141.0's |

Both ran inside the GB10 landing lease, after the record and with its own `memra-server`. Rows, logs and commands are in that directory's README.

## Other checks on the tested commit

- CI on the release PR (memra#1218) is green on the tested commit, including the complete CPU inventory.
- The `publish.yml` dry run on the release branch (run 42912850822) completed with success: the workspace packages and verifies, nothing is uploaded.
- `tools/release-guard.sh v0.141.1` passes: the workspace version and the eleven internal pins match, and the claim branch is on origin.
- `cargo fmt --check`, `git diff --check`, whole-workspace clippy with `-D warnings`, the memra-server suite (1,199 passed, 0 failed), the memra-engine `--lib` suite (714 passed, 0 skipped) and the 13 CPU contracts `tools/validation_plan.py` selects for the range pass. They ran on `3612045f7e`, whose crates, tools and Cargo files equal the tested commit's; the two differ only in the `docs/RELEASING.md` row, which the bump's review rewrote.
- `tools/changelog.sh v0.141.0` drafts 1,611 bytes, under the 120,000-byte cap.
