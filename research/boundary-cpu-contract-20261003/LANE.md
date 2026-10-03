# Boundary CPU contract

Boundary-only inputs select their real CPU tests and omit unrelated Cargo jobs while owner/include obligations and unconditional security scans remain intact.

The baseline planner at8b1a0c75f0 selected all9 Cargo jobs for the checker, tests, policy and allowlist. Four actual Git event snapshots at2202ca9cab and the real `ci-change-class.sh` caller now emit `code=false`, all9 jobs false, empty packages and `contracts=public-boundary`. Cargo workspace and both-tree includes are checked first. A tools-owned crate collision retains lanes and its server consumer; valid offline Cargo metadata verifies both paths. An actual standalone CPU Rust reader changes output when the policy changes, and removing its include still retains the old server/native obligation. This fixture is not a Memra binary or model qualification.

The required contract executes all60 real public-boundary controls with a60-test floor, zero skips and zero expected failures. A planted synthetic policy violation fails the actual checker. Missing inputs, directory/FIFO/symlink types, populated staged or committed submodule ancestors and unknown inputs expand or refuse. Nine coherent source mutations fail assertions rather than tool errors. All12 selection methods pass;55 independently named CPU assertions are admitted. Missing assertions/results and skipped mandatory results refuse admission. The framework ran152 tests at floor152; the unchanged cache55, fastgate39 and registry14 checks passed.

The full repository check reported627 grandfathered findings and0 new findings; all627 allowlist entries still pin live tracked files. These whole-tree checks ran at8a523db5b2. The subsequent2202 change strengthens only the Cargo/include fixture and removes an unreachable duplicate continuation. Current-head CI repeats the full scan and drift check. The entire boundary job remains unconditional; historical-ref and pre-push behavior is unchanged. Policy or allowlist edits do not exempt themselves from those checks.

The first6e runner accepted60 false assertions decorated as expected failures. The current actual runner rejects that suite with exit1 and reports expected_failures=60. The first standalone-reader harness accidentally committed its ELF and correctly received full unknown-input expansion; the corrected harness keeps it outside the Git fixture. Those attempts do not establish admission. Exact private source/event bundles and raw failures are retained.

Reproduce the executable regressions and strict suite:

```sh
tools/unittest-floor.sh tools 'test_validation_*.py' 152
python3 tools/validation_plan.py contracts --selected public-boundary
python3 tools/check-public-boundary.py check
python3 tools/check-public-boundary.py verify-allowlist
```

`PROOF.json` binds source/readers, actual plans, shell output, named assertions, mutation outcomes and the CPU reader. Final integration will compose the support source/docs and execution-type inputs on their merged base, refresh source/baseline binding and run the actual combined test count. This record is CPU tooling evidence. Native math, emitted native programs, compiler/build defaults, model artifacts/defaults, tolerances and required native gates remain unchanged. GPU selection stays shadow-only.

publicity: skipped: maintenance release
