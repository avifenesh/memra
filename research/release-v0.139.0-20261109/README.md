# v0.139.0 release qualification (2026-11-09)

Tested commit `abac82d04347391d76fb8103f6a1b46b9fcf06b1` (the `release: v0.139.0` commit on main `958ae32fed`: the
nvcc temp-name build fix, `tools/memra-gpu-run` as revised in review, the version bump). The banked publication adds
only research and qualification metadata on top of it.

## Records

| record | profile | build host | card | battery |
|---|---|---|---|---|
| `research/release-qualification/release-v0-139-0-ubuntu-24-04/` | `ubuntu-24.04` (glibc 2.39) | stock `ubuntu:24.04` image | one RTX PRO 6000 Blackwell Max-Q Workstation Edition | kernel-check ALL GREEN (119 cells, 112 executed, 10 required, 7 skipped; budget 11); Ornith-1.5-35B-A3B and Qwen3.8-27B: calibrated argmax, served-batch argmax and K=1..8 PASS |
| `research/release-qualification/release-v0-139-0-ubuntu-22-04/` | `ubuntu-22.04` (glibc 2.35) | stock `ubuntu:22.04` image | a second RTX PRO 6000 Max-Q of the same box | kernel-check ALL GREEN (119 cells, 112 executed, 10 required, 7 skipped; budget 11); Ornith-1.5-35B-A3B and Qwen3.8-27B: calibrated argmax, served-batch argmax and K=1..8 PASS |

Each record is one capture under the tested commit's own `tools/memra-gpu-run` (the canonical per-UUID flock plus
the box's per-card lock as `--box-lock`), sealed after the wrapper exited (`tools/qualify-release.py seal`). Both
leases finished clean: exit 0, no signal, no process left in the group, no compute left on the card. The record
format qualifies one physical card per capture, so the battery on two of the box's four cards is the two captures,
one per card, run back to back. Both cards ran at their 280 W limit with no thermal throttle. The kernel oracles are
the banked record's (`Qwen3.6-35B-A3B-UD-IQ4_XS` at 649d7508..., `Qwen3.5-9B-NVFP4-MTP-GGUF` at 52c9cceb...) plus
the roster's `Qwen3.8-27B-NVFP4-Q5K-mtp.gguf` (1facf36c...), named by `MEMRA_KC_MODELS_DIR`.

The build images carry what `release.yml` installs on GitHub's runners: `build-essential`, `git`, `python3`,
`bubblewrap`, the CUDA 13.1 packages from `tools/install-cuda-ci.sh` and Rust 1.97.1. The builds ran as root in the
image (bwrap needs it under the host's userns restriction) with `HOME`, `USER` and `LOGNAME` set to the runner's
values.

## Byte reproducibility (what `release.yml` checks before packaging)

- Before the fix: GitHub's controlled `ubuntu-24.04` build of `2288c4bf82` (gpu-ci `build_reuse_only` run
  42506325847) and a container build of the same tree had identical sizes and differed in two of 37 ELF sections
  only, `.strtab` and `.note.gnu.build-id`: nvcc's `tmpxft_<pid>_...` intermediate names, carried by each kernel
  object's FILE symbol. No two builds could match, on GitHub or anywhere.
- After the fix (`crates/memra-engine/build.rs` `normalize_nvcc_temp_symbols`): two consecutive container builds of
  `b260f75ecb` were byte-identical under `cmp` (all six binaries, equal build-ids, 33 zeroed tmpxft symbols in
  kernel-check, none with a pid); GitHub's controlled `ubuntu-24.04` build of the same commit (run 42511403826)
  matched the container build 6/6.
- The tested commit: GitHub's controlled `ubuntu-24.04` build of `abac82d043` (gpu-ci `build_reuse_only` run
  42538550700; its source manifest names the commit) passes `tools/release_qualification.py verify --binaries <its
  six ELF files> --profile ubuntu-24.04` against the banked record, the check `release.yml` runs before packaging. No
  workflow builds the `ubuntu-22.04` profile before a tag; that record comes from the same image recipe on
  `ubuntu:22.04`, and `release.yml` is its first byte check.

## Dry run

The kit ran first on main `2288c4bf82`: both profile builds captured and sealed on the same two cards
(kernel-check ALL GREEN, 119 cells, 111 executed, 10 required, 8 skipped; both roster models pass calibrated argmax
and K=1..8). Nothing from it is banked.

## Superseded records

Two earlier complete sets passed and are not banked. Every tracked file is a source input, so each change below
made the whole set run again.

- `c22694ad0b` (2026-11-08): both captures, both cell records and GitHub's controlled build. The review of the
  release PR then changed `tools/memra-gpu-run`, its test and `tools/qualify-release.py`'s `live_lease` (a
  lock-dir seam for the test). A restart at `268f85177d` stopped during its builds, before any GPU work.
- `d154d64aac` (2026-11-09): both captures, both cell records, and GitHub's controlled `ubuntu-24.04` build (run
  42534660253) matching the container build 6/6. CI then refused the commit: `tools/ci_merge_validation.json` still
  pinned the old `ci.yml` bytes and contract list. Rebinding it is the only difference between `d154d64aac` and
  the tested commit.

## Box conditions

The box has four RTX PRO 6000 Max-Q cards on one PCIe complex. The box's quiet flag was set from the start of the
builds through the end of the rtxpro6000 cell record, and other work on the box had stopped before the captures:
no card held a compute process when either capture or the record started. The rtxpro6000 record's neighbouring
cards stayed at or below 48 C. On the evening before, timed work on this box overlapped other campaigns (the fix
stack's rtxpro6000 perf record and the #1090 prefix-budget A/B cells). Those comparisons interleave their arms on
one card, so they stay internally fair, but their absolute timings are not quiet-box numbers.

## Cell records for the push gate

| record | device | kinds | result |
|---|---|---|---|
| `research/release-qualification/cells/abac82d04347-rtxpro6000-battery-perf-serving/` | one RTX PRO 6000 Max-Q (sm_120a) | battery, perf A/B against `958ae32fed`, serve-smoke | pass |
| `research/release-qualification/cells/abac82d04347-gb10-battery-perf-serving/` | one DGX Spark GB10 (sm_121a) | battery, perf A/B against `958ae32fed`, serve-smoke | pass |

serve-smoke's gemma and Qwen3.6 cold arms skip on the RTX PRO 6000 box (no models), which the cell judge allows.
