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

## 2. The fold: one executor, one job, one call per battery

Coordinator heads-up mid-day: the lead lands a minimal hotfix on main that removes #590's
duplicate job block; this lane's fold continues on top and merges main again once it lands.
The fold, per file:

- `.github/workflows/ci.yml`: #590's `portable-suites` job (the `bash tools/ci-portable.sh` step
  and the bare `python3 -m unittest discover -s tools -p test_gpu_ci.py -v` step) is gone. The
  `gates` job (the CPU-only fixtures, not gated on the change class) gains the workflow-key
  census plus its teeth (section 3). #590's ten GPU-adapter controls move to the end of the
  surviving `portable-suites` job through `tools/unittest-floor.sh tools test_gpu_ci.py 9` (the
  bare discovery was green over zero tests; the floor sits one below the ten measured), with the
  exact step text and position of the lead's hotfix PR #600, so the merge once #600 lands is one
  identical addition, not two copies (first draft of this day had the step in `gates`; moved
  before the first push of the fold). The rest of the `portable-suites` job from #592 is
  unchanged: `needs: changes`, `cargo fetch --locked`, the wrapper, its teeth, the collector
  suite with both lock paths held under floor 80, the floor tool's teeth.
  `check-workflow-keys.py` on the tree lists ten jobs in ci.yml, one of them `portable-suites`.
- `.github/workflows/gpu-ci.yml`: untouched. It runs no CPU suite (section 1b), so there is no
  step to point at the executor; the dispatch, its candidate and prerequisite checks, the
  manifest preflight, the capsule digest, the lease-wrapped capture and the check run are as
  #590 landed them.
- `tools/portable-suites.sh`: the one executor. `--locked` folded in from #590's command
  (`--offline --locked --no-fail-fast`: a lockfile that would need to change is a refusal, not
  an offline re-resolution). Census, budget, floor, banked log and last line unchanged.
- `tools/ci-portable.sh`: no longer an executor. Twelve lines of comment and
  `exec "$(dirname -- "$0")/portable-suites.sh" "$@"`, so anything still holding #590's name
  (its PR body, its docs, a branch cut from it) lands in the wrapper. Nothing tracked calls it.
- `tools/local-ci.sh`: #590's second call (line 141, `bash tools/ci-portable.sh`, a release
  build of the three crates before the memra-engine suite) removed; the day-13 call keeps its
  place after the gguf census and gains `RUST_TEST_THREADS=8` next to `CARGO_BUILD_JOBS=8`
  (#590's test-thread cap, folded). The three crates build and run once per local-ci.
- `tools/test_portable_suites.sh` arm 3 (wiring) gains four assertions, comment lines stripped:
  exactly one `  portable-suites:` job in ci.yml; no live `cargo test` line in ci.yml or
  local-ci.sh names memra-tier, memra-kv or memra-cli (the suites run through the wrapper
  only); `tools/ci-portable.sh`, if present, names `portable-suites.sh` and contains no
  `cargo`; neither ci.yml nor local-ci.sh calls `ci-portable.sh`. 22 arms now.
- `tools/test_gpu_ci.py` gains `test_ci_runs_these_controls_once_under_a_floor`: exactly one
  live ci.yml line runs `tools/unittest-floor.sh tools test_gpu_ci.py <floor>`, and no live line
  runs the file bare. 11 controls now.
- `docs/CI.md` first section rewritten around the one entry point (the wrapper's command,
  census, floor, banked log, teeth; the local caller's two caps; the forward; the `gates` steps).
  `docs/TESTING.md` "Standing execution": the command line shows `--locked`, and a new "One
  entry point (day 14)" paragraph records the fold and points here. `docs/ROUTER.md` route
  unchanged (`docs/CI.md` still owns the hosted CI map and the GPU dispatch).

## 3. Workflow-key census (the coordinator's ask): a loader that raises

`tools/check-workflow-keys.py` loads every `.github/workflows/*.yml` with a `SafeLoader` whose
`construct_mapping` raises on the first duplicate key at any depth (jobs, steps, `with:`,
`env:`, `inputs:`), refuses a file without a `jobs:` mapping, and refuses an empty workflow
directory (zero files is not green). Exit 0 prints each file's job names; exit 1 names the
file, the key, its line and column and the line of the first occurrence. Run against main's
own `ci.yml` at `34ed99dfc` (`day14/guard-on-main-34ed99dfc.log`, verbatim):

```
check-workflow-keys: FAIL: /tmp/spill-d-day14/main-34ed99dfc/ci.yml: duplicate mapping key 'portable-suites' at line 401 column 3 (first at line 206)
check-workflow-keys: FAIL: 1 of 1 workflow files refused
rc=1
safe_load jobs: ['changes', 'gates', 'boundary', 'build', 'clippy', 'server-tests', 'portable-suites', 'engine-tests', 'arch-coverage', 'publish-dryrun']
rc=0
```

Two wiring sites, because the failure is self-blinding. A step inside ci.yml can never catch a
duplicate key in ci.yml (GitHub refuses the file before any job starts); it protects the other
four workflow files. The place that can catch ci.yml is the push: `tools/hooks/pre-push` gains
a "workflow-file census" arm after the docs-registry census, unconditional and with no skip
switch (the same reasoning as the releasability censuses: a tree whose workflow GitHub cannot
parse has no emergency in which pushing it is right), scoped to the memra workspace like the
arms above it, failing closed when the script is missing. This lane's own push of `dc106da6c`
this morning carried the inherited duplicate and the hook let it through; with this arm it
refuses. The third fence is not code: a PR run on a merge ref that predates a base move is
what let #590 land green, and only the branch-protection setting that requires a current
branch closes that (an owner decision, noted for the lead, not changed here).

Teeth, `tools/test_workflow_keys.sh` (nine arms): the control (`yaml.safe_load` accepts the
duplicate-job file, exit 0), a duplicate job key reds and the refusal names the key and its
line, a duplicate nested key (two `run:` in one step) reds, a clean file greens and the green
names its jobs, an empty directory reds, the tree's own workflows green.

## 4. Local battery (CPU only, `systemd-run --user --scope -p CPUQuota=1200% -p MemoryMax=28G`)

One scope, sequential (`day14/battery.sh`, `battery.log`; per cell `<name>.log` and `<name>.exit`).
Every cell exit 0. After the battery the `test_gpu_ci.py` step moved to #600's position, so the
cells that read ci.yml ran again (`day14/rerun/`, all exit 0; the teeth and floors below are
quoted from the rerun where they differ in nothing but timing).

- `cargo fmt --all -- --check`: exit 0 (no Rust touched).
- `tools/portable-suites.sh` (`wrapper.log`, raw cargo output `wrapper-raw.log`):
  `skip-census: VERIFY OK` (tier 0, kv 0, cli 0 artifact-gated skips, all declared),
  `skip-census: running: cargo test -p memra-tier -p memra-kv -p memra-cli --offline --locked --no-fail-fast -- --test-threads=1 --nocapture`,
  `skip-census: 335 passed, 0 skipped (budget 0), 0 failed, 0 filtered out across 14 binaries`,
  `portable-suites: PASS (tier, KV and onboarding CLI suites executed on CPU; skips 0 of budget 0; NOT GPU qualification)`.
  334 became 335 through this morning's main merge. 16 s warm.
- `tools/test_portable_suites.sh` (`rerun/teeth.log`, verbatim):

```
ok   arm1 planted failures red the wrapper (exit 1)
ok   arm1 wrapper banked the raw cargo log
ok   arm1 failures list names planted_retirement_keeps_owner_red_arm
ok   arm1 failures list names planted_red_arm::planted_kv_hierarchy_red_arm
ok   arm1 failures list names planted_red_arm::planted_onboarding_receipt_red_arm
ok   arm1 cargo names the failed target -p memra-tier --test contracts (every crate ran)
ok   arm1 cargo names the failed target -p memra-kv --lib (every crate ran)
ok   arm1 cargo names the failed target -p memra-cli --lib (every crate ran)
ok   arm1 no PASS line
ok   arm2a undeclared SKIP reds the wrapper (exit 1)
ok   arm2a the static census names the planted test (planted_artifact_gated_integration_test)
ok   arm2a refused before cargo ran
ok   arm2b undeclared SKIP reds the wrapper (exit 1)
ok   arm2b the static census names the planted test (planted_skip::planted_artifact_gated_test)
ok   arm2b refused before cargo ran
ok   arm3 ci.yml runs tools/portable-suites.sh
ok   arm3 ci.yml runs this fixture
ok   arm3 tools/local-ci.sh runs tools/portable-suites.sh
ok   arm3 ci.yml has exactly one portable-suites job
ok   arm3 no live cargo test on memra-tier/-kv/-cli outside the wrapper (one executor)
ok   arm3 tools/ci-portable.sh forwards to the wrapper and runs no cargo of its own
ok   arm3 neither ci.yml nor tools/local-ci.sh calls tools/ci-portable.sh
test_portable_suites: 22 ok, 0 FAIL
```

- `python3 tools/test_gpu_ci.py`: `Ran 11 tests in 0.005s`, `OK` (the ten #590 controls plus the
  wiring arm). The ci.yml step form: `unittest-floor: OK: ran 11 tests (floor 9) for tools (test_gpu_ci.py)`.
  The step text keeps #600's "10 measured 2026-09-21" (the count when the floor was set; 11 run
  now, the floor stands).
- `python3 -m pytest -q crates/memra-tier/tests/battery/`: `87 passed, 32 subtests passed in 17.69s`.
- The ci.yml lock-held step, inside `bwrap --tmpfs /tmp` so the rig's real locks were never
  touched (`unittest-lock-held.log`): `holding /tmp/memra-5090.lock and /tmp/memra-gpu.lock in a private /tmp (bwrap) for the whole suite`,
  `OK`, `unittest-floor: OK: ran 87 tests (floor 80) for crates/memra-tier/tests/battery (test_*.py)`.
- `tools/test_workflow_keys.sh`: `test_workflow_keys: 9 ok, 0 FAIL`; `tools/check-workflow-keys.py`:
  `check-workflow-keys: OK: 5 workflow files, no duplicate mapping keys` (ci.yml jobs: changes,
  gates, boundary, build, clippy, server-tests, portable-suites, engine-tests, arch-coverage,
  publish-dryrun).
- `python3 -c "import yaml; yaml.safe_load(...)"` on ci.yml and gpu-ci.yml: `safe_load ok` (kept as
  the control it is, not as the guard).
- `tools/test_ci_change_class.sh`: `test_ci_change_class: 14 arms PASS`. `tools/test_unittest_floor.sh`:
  `test_unittest_floor: 5 ok, 0 FAIL`. `tools/check-action-pins.sh` exit 0 and
  `action-pin census fixture: PASS` (gpu-ci.yml's four pinned actions pass the census; main
  never ran it on that file).
- `bash tools/check-flags.sh`: `check-flags: no uncovered runtime names` (no new `MEMRA_*` read
  today). `bash tools/docs-registry-census.sh`: `ROUTER.md lines=46 (cap 60)`, `flags-table-census:
  docs/FLAGS.md tables=58 rows=903`. `git diff --check` exit 0. `shellcheck -S warning` on the
  wrapper, the forward, both teeth scripts and the hook: exit 0; `bash -n tools/local-ci.sh` exit 0.
- `DOCS_RS` never set; no GPU cell, no engine file, no `.cu`, no flag default, no numeric program.

## 5. Findings for the lead

1. Main was red at the workflow level from 09:47Z (`34ed99dfc`) until #600 lands: zero jobs per
   run, so no build, clippy, boundary or publish check ran on main or on any PR opened since.
   This lane's fold carries the same removal hunk as #600 plus #600's exact `test_gpu_ci.py` step
   text and position, so the merge after #600 is clean and leaves one copy of each.
2. The class of failure (a PR green on a stale merge ref, base moved, combination invalid) is
   closed only by the branch-protection setting that requires the branch to be current before
   merging. The hook arm catches a lane that pushes the duplicate; nothing in this repo can
   catch a merge-button combination. Owner decision, not changed here.
3. `gpu-ci.yml` dispatch prerequisites are still in draft #566 (`qualify-release.py`,
   `release_qualification.py`, `release_inputs.py`, `release_input_view.py`,
   `install-release-sandbox-ci.sh`), so every dispatch refuses as unconfigured today, as
   `docs/CI.md` says. Untouched.
4. Day-13 finding 1 stands: `ci-change-class.sh` treats `research/**` as docs-only while ten
   test-time `research/` reads exist across lanes A/B/C/D.
5. `research/spill-d-20260919/pro-single-day11/build/` is an untracked leftover of a day-11
   build receipt (a source SHA, a driver probe line, an exit code, build/clippy/test logs), not
   ignored and not mentioned in DAY11.md. Left in place; nothing in it is a receipt this lane
   cites, and it is the owner's call whether it is banked or removed.

## 6. Scope and receipts

CPU-only gate plumbing: one executor for three crates' suites, one job, one call per battery, a
workflow-file census with teeth in the hook and the `gates` job, docs aligned. No GPU cell, no
timing claim, no default, flag default, numeric program or support state changed. Receipts under
`research/spill-d-20260919/day14/` (`battery.sh`, `battery.log`, per-cell logs and exit codes,
`wrapper-raw.log` banked verbatim and pinned in the boundary allowlist like day 13's raw logs,
`guard-on-main-34ed99dfc.log`, `rerun/`). Scratch `/tmp/spill-d-day14/` removed at close.
About 3.5 agent-hours against the 4-hour budget.
