# Changed-input validation and edge coverage

This lane changes validation tooling and CI dispatch. It changes no model math, runtime
default, qualification tolerance or release/tag battery.

## Evidence

- `skip-census-before.txt`: the old scanner fails the planted Rust-string/comment scope
  regressions. The patched scanner keeps the actual O_DIRECT test at its correct module path.
- `skip-census-api-candidate.txt`: the patched scanner checks the API candidate against the
  unchanged manifest. No skip budget or manifest exemption was added.
- `cargo-feature-scope.json`: a tiny real Cargo workspace proves that package selection can
  remove a dependency feature. Keeping `--workspace` and filtering binary targets preserves
  the feature union without building the other binary. The package-only control fails its
  compile-time feature assertion.
- `composite-serving-coverage.json`: one captured Qwen3.5-9B collector invocation covers five
  behavioral assertions and four failure controls. Its 60 distinct raw files were hash-checked.
  The run used a cache-enabled boot and a separate cache-disabled red boot. It completed 18
  requests and deliberately aborted one. These are bounded greedy serving observations.
- `composite-harness-drift.json`: the later collector revision does not match the captured
  coverage binding, so the reduced plan expands instead of borrowing that pass.

The composite producer belongs to PR #915. The receipt binds the captured harness revision
and runtime binary separately. It is not evidence for a different collector revision, sampled
quality, maximum model context, GPU batch width, another hardware class or model admission.
The recorded elapsed time is one execution receipt, not a comparison against an unmeasured
alternative schedule.

## Validation

The Python suites cover dependency propagation, old/new fixture inputs, unknown refs and
inputs, renamed dependencies, generated inputs, symlinks, local untracked/masked source,
tested merge-tree dependencies, Cargo feature preservation, mandatory controls, stale source
and execution context, per-edge result admission, and missing contract files.

The existing check names and complete CPU suite groups remain. Missing or failed classifier
outputs expand rather than skip. Native diagnostic selection does not gain model/release
qualification authority. See `docs/VALIDATION-SELECTION.md` for commands and boundaries.
