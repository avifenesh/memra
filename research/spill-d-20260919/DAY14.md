# WP-D day 14: one entry point for the tier, KV and onboarding-CLI suites (#592 and #590 folded)

Repository: **memra**, branch `lane/spill-d-20260919`. Start: tip `75fac84c9` (= origin, merged to
main by #592). First action: `git merge --no-ff --no-edit origin/main` at `34ed99dfc` (#590), merge
commit `dc106da6c`, pushed (every pre-push arm passed: perf board, flags census 867 reads,
releasability censuses, docs-registry census, public boundary 0 matches). No GPU work: every cell
below is CPU, on the local rig under `systemd-run --user --scope -p CPUQuota=1200% -p MemoryMax=28G`,
and nothing here is hardware qualification.

## 1. Census: two entry points for the same suites (no code changed for this section)

Main at `34ed99dfc` carries two executors of the memra-tier, memra-kv and memra-cli suites. #592
(this lane, day 13, merged 05:24Z) landed `tools/portable-suites.sh`. #590 (codex lane, merged
09:47Z by another session) landed `tools/ci-portable.sh`. Both PRs added a ci.yml job named
`portable-suites` and both added a call to `tools/local-ci.sh`'s `cpu_chain()`.

### 1a. What each entry point runs

| | `tools/portable-suites.sh` (#592) | `tools/ci-portable.sh` (#590) |
|---|---|---|
| cargo command | `cargo test -p memra-tier -p memra-kv -p memra-cli --offline --no-fail-fast` (debug profile, no `--lib`: the six memra-tier integration suites and the four compile-fail doctests run) | `cargo test --release --locked -p memra-tier -p memra-kv -p memra-cli` (release profile, no `--lib`, default fail-fast: the first red binary stops the run) |
| lockfile | `--offline` against the committed `Cargo.lock`; ci.yml warms with `cargo fetch --locked` first | `--locked` (refuses a stale lockfile), network allowed |
| static skip census | `skip-census.py verify --crate memra-tier --crate memra-kv --crate memra-cli` over `src/` and `tests/`: an artifact-gated `#[test]` must be declared in `tools/skip-census.tsv` or the wrapper refuses before cargo runs | none |
| run-side skip census | `skip-census.py run --budget-var MEMRA_PORTABLE_SKIP_BUDGET` at budget 0: every printed SKIP counts and one reds | none (a skipping test reads as a pass) |
| non-vacuity floor | `--min-passed 300` (334 measured 2026-09-21 across 14 binaries) | none (`Ran 0 tests` would be green) |
| raw evidence | the whole cargo output banked at `target/portable-suites.log` | stdout only |
| teeth | `tools/test_portable_suites.sh`, 18 arms: three planted failing tests, one per crate, must red the wrapper with all three targets named (`--no-fail-fast` proven); an undeclared SKIP under `tests/` (2a) and `src/` (2b) must red the static census before cargo; wiring (ci.yml runs the wrapper and the teeth, local-ci runs the wrapper). Builds the planted copy in `target/portable-suites-teeth`, never the tree's target | none |
| last line | `portable-suites: PASS (tier, KV and onboarding CLI suites executed on CPU; skips 0 of budget 0; NOT GPU qualification)` | `portable CI: tier, KV and onboarding CLI suites; GPU qualification is separate` (printed before cargo, so it prints on a red too) |

### 1b. Where each is invoked

| site | `tools/portable-suites.sh` (#592) | `tools/ci-portable.sh` (#590) |
|---|---|---|
| `.github/workflows/ci.yml` | job `portable-suites` at line 401: `needs: changes`, `if: !cancelled() && needs.changes.outputs.code != 'false'` (docs-only PRs skip it like every compile job), ubuntu-24.04, toolchain 1.97.1, rust-cache key `ci-portable-suites`, timeout 45. Steps: `cargo fetch --locked`; `tools/portable-suites.sh`; `tools/test_portable_suites.sh`; the collector suite (`crates/memra-tier/tests/battery/`) with BOTH rig lock paths held for the whole run through `tools/unittest-floor.sh ... 80` (87 measured); `tools/test_unittest_floor.sh` | job `portable-suites` at line 206: no `needs`, no `if` (runs on docs-only PRs too), ubuntu-24.04, toolchain 1.97.1, the SAME rust-cache key `ci-portable-suites`, timeout 30. Steps: `bash tools/ci-portable.sh`; `python3 -m unittest discover -s tools -p test_gpu_ci.py -v` (no floor) |
| `.github/workflows/gpu-ci.yml` | not invoked | not invoked. The dispatch runs no CPU suite: its `build` job validates the candidate, tries a reuse, preflights the input manifest, installs the pinned CUDA compiler and the build sandbox, builds through `tools/qualify-release.py` and packs a capsule; its `gpu` job unpacks, captures and seals; `qualification-result` posts the check run |
| `tools/local-ci.sh` `cpu_chain()` | lines 168-181, after the gguf census: `CARGO_BUILD_JOBS=8 tools/portable-suites.sh`, fatal, no door | line 141, after the memra-server suite: `CARGO_BUILD_JOBS=8 RUST_TEST_THREADS=8 bash tools/ci-portable.sh || return 1`, fatal |
| docs | `docs/TESTING.md` (Generic spill section, "Standing execution"; Collector "Locks"), `docs/FLAGS.md` (`MEMRA_TIER_BATTERY_LOCK_DIR` row), `tools/skip-census.tsv` header | `docs/CI.md` (first paragraph names it "shared with the native local-CI entrypoint"), `docs/ROUTER.md` (the `docs/CI.md` route) |

Net effect on main at `34ed99dfc`: the three crates' suites are listed to run TWICE per CI run
(once in each job, release and debug, about 20 min of runner time doubled) and twice per
`local-ci.sh` run (the release build at line 141, then the debug build at line 179). In practice
they run ZERO times in hosted CI today, see 1d.

### 1c. What one has that the other does not

#590 adds, and #592 does not have:

- `.github/workflows/gpu-ci.yml`: a `workflow_dispatch` that takes one full 40-character candidate
  SHA and, in order: checks the checkout is that SHA and that the content-bound qualification
  tooling exists (`tools/gpu-ci.py candidate()`: `qualify-release.py`, `release_qualification.py`,
  `release_inputs.py`, `release_input_view.py`); tries `release_qualification.py verify` to reuse a
  committed content-equivalent record; otherwise requires a pinned HTTPS input manifest
  (`GPU_CI_INPUTS_URL`, `GPU_CI_INPUTS_SHA256`; schema `memra-gpu-ci-inputs-v1`, lease wrapper
  `memra-gpu-run`, unique `.gguf` oracle names, sizes and SHA-256 digests, a 200 GiB inventory
  ceiling), installs the pinned CUDA compiler and the bwrap build sandbox, builds once on a CPU
  runner through `tools/qualify-release.py build`, packs six ELF binaries plus five metadata files
  into a capsule bound by SHA-256, hands the capsule to a `gpu` job on a run-labelled runner
  (`memra-ci-pro-ubuntu24-<run-id>`) that verifies the digest, requires exactly one RTX PRO 6000
  as GPU0, proves a CUDA allocation, downloads the exact input bytes, runs capture under the
  coordinator's lease wrapper, seals after the wrapper exits, and a third job posts the check run
  `gpu-qualification/pro-ubuntu24` on the candidate only when the sealed source descriptor names
  that same commit (a reuse passes without the GPU job; a CPU build alone never passes).
- `tools/gpu-ci.py` (the adapter above) and `tools/test_gpu_ci.py` (10 CPU controls: manifest
  field, name, URL, digest, size and ceiling refusals; download size and hash checks with no
  partial artifact left; manifest digest mismatch; sealed-commit extraction only from a hash-bound
  `source.json`; capsule members (missing, extra, duplicate, symlink) refuse and leave no output;
  roundtrip keeps 0755; candidate SHA, checkout and prerequisite refusals).
- `docs/CI.md` (the dispatch, its input manifest, receipt custody) and the `docs/ROUTER.md` route.
- Prerequisites not on main today: the four tools above and `tools/install-release-sandbox-ci.sh`
  are in draft #566, so every dispatch refuses as an unconfigured request before any GPU time
  (`docs/CI.md` §1 says so). Nothing in this lane touches that.

#592 adds, and #590 does not have: the static and run-side skip census at budget 0, the
`--min-passed 300` floor, `--no-fail-fast`, the banked raw log, the 18-arm planted-failure
teeth, the `changes` gating and the `cargo fetch --locked` warm step, the collector suite executed
with both rig lock paths held under a unittest floor (lead ruling 12), `tools/unittest-floor.sh`
and its teeth (revuto on #592), and the `MEMRA_TIER_BATTERY_LOCK_DIR` seam with its
`--private-lock-dir-for-tests` guard.

### 1d. Finding: main's ci.yml is not a valid workflow since #590 merged

Two jobs keyed `portable-suites` in one `jobs:` map. GitHub's parser refuses duplicate mapping
keys, so the push run for `34ed99dfc` (run 35585228365) has zero jobs and reads
`This run likely failed because of a workflow file issue.` Every push to main and every PR
against it gets the same result until the duplicate is gone: no build, clippy, server, engine,
boundary or publish job runs. The two previous main runs (`dcaa5bdf5`, `70038ed01`) were green.
PyYAML's `safe_load` keeps the LAST definition silently, so the day-13 battery's
`python3 -c "import yaml; yaml.safe_load(...)"` and the `jobs:` listing it printed would have
passed on this tree; that check is not a duplicate-key check. #590's own CI was green at
`5bf84be1a` because its merge ref predates #592 (05:24Z); the combination existed only on main.
Today's teeth gain a duplicate-job-key arm (section 2).
