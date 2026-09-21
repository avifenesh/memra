# WP-D day 13: #545 (the tier, KV and onboarding-CLI suites get a standing executor) and lead ruling 12 (the collector suite takes a private lock path)

Repository: **memra**, branch `lane/spill-d-20260919`. Start: tip `c63aff586` (= origin, merged to
main by #588). First action: `git merge --no-ff --no-edit origin/main` at `b013885ba`, merge commit
`7fdc9a013`. Owner order: the open pre-DeepSeek issues first; #545 today. No GPU work: every cell
below is CPU, on the local rig under `systemd-run --user --scope -p CPUQuota=<=1200% -p MemoryMax=28G`,
and nothing here is hardware qualification.

## 1. #545: standing CI executes the tier, KV and onboarding-CLI suites

The finding (audit at `61be8b0d9`, unchanged through `b3487a03b`): `ci.yml` compiled memra-tier,
memra-kv and memra-cli (build, clippy) and executed other crates; `tools/local-ci.sh` executed
server, engine and gguf. The three crates' tests (213 `#[test]` in memra-tier plus four
compile-fail doctests, 69 in memra-kv, 12 in memra-cli) ran only when a lane ran them by hand.

Delivered:

- `tools/portable-suites.sh`: the one wrapper. `python3 tools/skip-census.py verify --crate
  memra-tier --crate memra-kv --crate memra-cli` (static census: an artifact-gated `#[test]` must be
  declared; today none exists in the three crates), then `MEMRA_PORTABLE_SKIP_BUDGET=0
  python3 tools/skip-census.py run --budget-var MEMRA_PORTABLE_SKIP_BUDGET --min-passed 300 --log
  target/portable-suites.log -- cargo test -p memra-tier -p memra-kv -p memra-cli --offline
  --no-fail-fast`. No `--lib` (the six integration suites and the doctests are most of the tests),
  `--no-fail-fast` (every binary runs), `--offline` against the committed lockfile, the raw cargo
  output banked (never summary-only). Its last line names what it is:
  `portable-suites: PASS (tier, KV and onboarding CLI suites executed on CPU; skips 0 of budget 0; NOT GPU qualification)`.
- `.github/workflows/ci.yml` job **`portable-suites`** (`needs: changes`, gated like the other
  compile jobs, ubuntu-24.04, toolchain 1.97.1, rust-cache key `ci-portable-suites`, no CUDA
  install: memra-tier has no CUDA dependency, memra-kv links cudarc `dynamic-loading` and no test
  opens the driver, memra-cli depends on gguf/reference/tokenizer). Steps: `cargo fetch --locked`
  (the suites run offline), `tools/portable-suites.sh`, `tools/test_portable_suites.sh`, and the
  collector test suite with BOTH rig lock paths held (section 2). Every step name says CPU and not
  qualification.
- `tools/test_portable_suites.sh`: the teeth (acceptance criterion 3). Arm 1 plants three failing
  tests in a copy of the tree, one per crate (`planted_retirement_keeps_owner_red_arm` appended to
  memra-tier's `tests/contracts/mod.rs`, `planted_red_arm::planted_kv_hierarchy_red_arm` in
  memra-kv, `planted_red_arm::planted_onboarding_receipt_red_arm` in memra-cli) and requires the
  wrapper to exit non-zero, the banked raw log to list all three under `failures:`, and cargo to
  name all three targets (`--no-fail-fast` held, every crate ran). Arm 2 plants a `#[test]` that
  prints `SKIP` and returns; the wrapper must exit non-zero from the static census, naming the
  test, before cargo runs. Arm 3 asserts the wiring (ci.yml runs the wrapper and the fixture;
  local-ci runs the wrapper). Scratch under `mktemp`, removed on exit (`leftover scratch dirs: 0`).
- `tools/local-ci.sh`: `cpu_chain()` runs `CARGO_BUILD_JOBS=8 tools/portable-suites.sh` after the
  gguf census (fatal). No skip door: CPU-only, about 20 s warm. First run after a clean pays a debug
  build of gguf/reference/tokenizer/tier/kv/cli.
- `crates/memra-tier/tests/reclaim/fault.rs`: the two committed-receipt replays no longer guard on
  `receipt(..).exists()`. The day-11 and day-12 receipts are tracked (210 and 261 files); their
  absence is now a failure, never a silent pass (criterion 4: a test that skips for a missing
  artifact must say so, and these have no reason to skip). `cargo test -p memra-tier --test reclaim`:
  `test result: ok. 32 passed`.
- `tools/skip-census.tsv` header names the new budget (0) and where it lives.

### Teeth: the first run found a poisoning path

The first fixture run shared the tree's own target dir (`CARGO_TARGET_DIR=$here/target`) with the
copy. All 15 arms passed, and the next run of the real wrapper on the real tree went RED:
`-p memra-cli --lib` failed with `planted_red_arm::planted_onboarding_receipt_red_arm`
(`day13/wrapper-green.log` is the rerun; the red run's raw log is quoted here). Cause: cargo's
metadata hash for a workspace member excludes its path (target dirs are relocatable), so the copy's
`memra_cli` test binary landed under the real tree's hash `150b7fb610a22835`; `cp -a` preserved
mtimes, the real `lib.rs` was older than that binary, and cargo reused it. Only memra-cli was hit
(the kv and tier test units were rebuilt at 06:54:30; the fingerprint listing is in this record's
tool output). Fix: the copy builds into `target/portable-suites-teeth`, never the tree's target
(under `target/` so rust-cache keeps the registry artifacts between CI runs). The poisoned unit
was rebuilt by touching `crates/memra-cli/src/lib.rs`. Proof after the fix, in order:
`tools/test_portable_suites.sh` (own dir, cold): `test_portable_suites: 15 ok, 0 FAIL`, 27 s; then
the real wrapper: `skip-census: 325 passed, 0 skipped (budget 0), 0 failed, 0 filtered out across
14 binaries`, `planted names in raw log: 0` (`day13/wrapper-after-teeth.log`,
`wrapper-after-teeth-raw.log`).

### Teeth output, verbatim (`day13/teeth.log`, the fixed script, cold teeth target dir)

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
ok   arm2 undeclared SKIP reds the wrapper (exit 1)
ok   arm2 the static census names the planted test
ok   arm2 refused before cargo ran
ok   arm3 ci.yml runs tools/portable-suites.sh
ok   arm3 ci.yml runs this fixture
ok   arm3 tools/local-ci.sh runs tools/portable-suites.sh
test_portable_suites: 15 ok, 0 FAIL
rc=0 elapsed=27s
leftover scratch dirs: 0
```

### The suites themselves (`day13/baseline-cargo.log`, `baseline-cargo-raw.log`)

`skip-census: 325 passed, 0 skipped (budget 0), 0 failed, 0 filtered out across 14 binaries`, 22 s
warm. Per binary: memra-cli lib 12, memra-cli bin 0, memra-kv lib 67, `contracts_v12` 2, memra-tier
lib 2, `bank` 61, `contracts` 68, `peer` 18, `placement` 6, `reclaim` 32, `storage` 53, doc-tests
0 / 0 / 4 (the four compile-fail doctests in memra-tier). Floor 300 leaves room for honest
single-test removals only.

## 2. Ruling 12: the collector suite takes a private lock path

Ruling (INTEGRATION-DAY12.md, verbatim): "Collector pytest lock path. The tier battery's Python
tests take the real rig lock path; they must take a private lock path under test (lane F or D,
small), so a serving job on the rig cannot redden a CPU suite." The integ10 record shows the shape:
attempt 1 lost 20 collector tests to `REFUSED: [Errno 11] Resource temporarily unavailable` while
the lead's serve-smoke held `/tmp/memra-5090.lock`.

Control first (`day13/pytest-control-lock-held.log`): the pre-change suite with both real lock paths
held for the whole run, inside `bwrap --tmpfs /tmp` (a private `/tmp`, so the rig's real locks were
never touched and no lead job could be refused by this proof):
`holding /tmp/memra-5090.lock and /tmp/memra-gpu.lock in a private /tmp (bwrap)` then
`24 failed, 64 passed, 29 subtests passed in 18.90s`, `rc=1`. The 24 span test_collector, test_day10,
test_day4, test_day5, test_day6, test_external_lock and test_native_runner (the bootstrap tests
included: `tier-rig-bootstrap.sh --dry-run` holds the rig lock on purpose).

The seam, `MEMRA_TIER_BATTERY_LOCK_DIR=<existing dir>`, in two production tools and nowhere else:

- `tools/tier-battery.py`: `CANONICAL_LOCKS` (the two-name table, unchanged), `lock_table(dir)` (the
  same two NAMES under the directory: `<dir>/memra-5090.lock`, `<dir>/memra-gpu.lock`; a missing
  directory raises), `LOCKS = lock_table(os.environ.get("MEMRA_TIER_BATTERY_LOCK_DIR"))`. Every
  caller still reads `LOCKS`. A receipt written under the seam records the private path, so
  `--validate` in a process without the seam refuses it: `REFUSED: missing/noncanonical collector
  lock`, exit 2 (pinned by the new
  `test_private_lock_seam_receipts_never_validate_against_the_canonical_table`).
- `tools/tier-rig-bootstrap.sh`: after `RIG = RIGS[a.rig]`, the same re-root of `RIG['lock']`; the
  report, the generated `locked-run.sh` wrapper and the flock all carry the same path.
- NOT given a seam, on purpose: `tools/tier-lock-proof.py` (its docstring: "No third lock path and
  no environment switch are accepted") and the two legacy `kv-host-spill-*-gate.sh` prologues
  (`case "$GPU_LOCK" in /tmp/memra-5090.lock|/tmp/memra-gpu.lock)`, inside the hunks of the
  authorized `LEGACY-EXTERNAL-LOCK.diff` that `test_external_lock.py` reverse-applies and
  re-applies). Those tests already run FIXTURE COPIES of the three files; the copies get the
  literal substituted (`PrivateLockMixin.substitute`, `proof_copy`) and the tracked literals are
  asserted (`LOCKS = ('/tmp/memra-5090.lock', '/tmp/memra-gpu.lock')`,
  `/tmp/memra-5090.lock|/tmp/memra-gpu.lock) ;;`, `GPU_LOCK=${MEMRA_GPU_LOCK:-/tmp/memra-5090.lock}`).

`crates/memra-tier/tests/battery/private_lock.py`: `PrivateLockMixin` (setUp: a fresh directory per
test, `mock.patch.dict` on `os.environ` and on the loaded module's `LOCKS`; `private()`,
`substitute()`, `proof_copy()`, `canonical_env()` for a child that validates committed receipts,
`canonical_locks()` for the same in-process). Mixed into CollectorTests (test_collector,
test_day6), ProSingleTests, Day4Tests, Day5Tests, NativeRunnerTests, BootstrapTests,
ExternalLockTests. Committed receipts keep validating against the canonical table:
`test_real_first_hour_integrity_...` runs under `canonical_locks()` and then asserts the same
receipts REFUSE under the seam; `test_plan_and_schema` asserts the canonical plan without the seam
and the private plan with it. The canonical table is pinned as a literal in
`test_canonical_lock_exclusion_no_third_name` (`B.CANONICAL_LOCKS == {...}`, `lock_table() ==
CANONICAL_LOCKS`, basenames exactly `{memra-5090.lock, memra-gpu.lock}`).

Proof (`day13/pytest-lock-held.log`, verbatim first and last lines):

```
holding /tmp/memra-5090.lock and /tmp/memra-gpu.lock in a private /tmp (bwrap) for the whole suite
86 passed, 32 subtests passed in 16.91s
rc=0 elapsed=18s
```

Without the locks held (`pytest-normal.log`): `86 passed, 32 subtests passed in 16.95s`. 85 tests
at HEAD, 86 now (the seam red-arm test). The exact ci.yml step text, locks held
(`unittest-lock-held.log`): `Ran 86 tests in 16.964s` / `OK`. That step is the standing executor
of the ruling: the hosted job holds both paths for the whole suite on every run.

## Checks (all exit 0)

`cargo fmt --all -- --check`; `tools/portable-suites.sh` (325 passed, 14 binaries; three times,
including after the fixed teeth); `python3 -m pytest -q crates/memra-tier/tests/battery/` (86
passed, with and without the locks held); `tools/test_portable_suites.sh` (15 ok); `bash
tools/check-flags.sh` (`no uncovered runtime names`); `bash tools/docs-registry-census.sh` (tables
58, rows 902); `tools/test_action_pins.sh` (`PASS`, the new job reuses the pinned actions); `git
diff --check`; `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"` (jobs:
changes, gates, boundary, build, clippy, server-tests, portable-suites, engine-tests,
arch-coverage, publish-dryrun); `shellcheck` on both new scripts; `bash -n` and the embedded Python
of `tier-rig-bootstrap.sh` compiled. No engine file, `.cu`, FFI, flag default or numeric program
touched; `DOCS_RS` never set.

## Findings for the lead

1. `tools/ci-change-class.sh` still says "nothing under crates/ or tools/ reads research/ at
   compile or test time (checked 2026-09-02)". That is no longer true: ten test-time reads of
   `research/` exist (`memra-tier/tests/reclaim/fault.rs`, `day11.rs`, `day12.rs`, `bank/day4.rs`,
   `bank/day8.rs`, `bank/main.rs`, `memra-kv/src/tiered/tests.rs` via `include_str!`, lanes A/B/C/D).
   A research-only PR that removes or edits one of those receipts skips every compile job and lands;
   the next code PR reds. Out of today's scope (the classifier has 14-arm teeth); it wants a lane.
2. The new job runs the collector suite with `python3 -m unittest discover` (the tests are
   `unittest.TestCase`; same 86 tests) because pytest is not on a hosted runner and a pip step is
   one more thing to rot. The documented local command stays pytest.
3. `target/portable-suites-teeth` (1.4 GB here) is the teeth's own target dir: a build cache under
   `target/`, not scratch. It exists only where the fixture ran.
4. The wrapper is debug-profile; `local-ci.sh` is otherwise `--release`. The first local-ci after
   this lands pays one debug build of the six crates.

## Push

`git push origin lane/spill-d-20260919` was refused by the pre-push perf-ci freshness arm, verbatim
`pre-push: engine files touched after the last perf-ci battery.`, `base (merge-base with
refs/remotes/origin/lane/spill-d-20260919): c63aff58603f6e9763213927059b860a143db375`, engine
files `crates/memra-engine/src/bin/run_lockstep.rs` and `crates/memra-engine/src/hybrid_forward.rs`
(both inherited through the `origin/main` merge; this lane touches no engine file). Every other
hook arm passed (perf board, flags census 866 reads, releasability censuses, docs-registry census).
No `--no-verify`, no skip variable: the lane stays local and the lead pushes.

## Scope

CPU-only gate plumbing and test-seam work; no GPU cell, no timing claim, no default, flag default
or support state changed. About 4 agent-hours against the 5-hour budget.
