# Select validation from changed inputs

The planner separates three questions:

1. Which declared inputs and dependency edges can this change affect?
2. Which checks cover those edges, and why can another check be omitted?
3. Which actual assertions passed for the candidate, model, hardware and request?

An edge is affected or unaffected within the declared input graph. Missing information
means affected. A successful build is not a numerical or serving result.

## Plan the current worktree

```sh
python3 tools/validation_plan.py local --base origin/main > validation-plan.json
tools/fast-gate/fast-gate.sh --plan --diff origin/main
```

The JSON records changed paths, direct packages, dependency witnesses, binary edge decisions,
selected CI components, reasons for omissions, CPU contracts and separate native obligations.
Untracked source is included. Index-masked or ignored build inputs refuse a narrow plan.

Cargo normal, development, build and target-specific dependencies all participate. An edit to
`memra-server` selects its checks without rerunning its unchanged dependencies' unit suites.
An edit to `memra-gguf` reaches its consumers. Cargo manifests, toolchains, workflow changes,
unresolved input expressions and unregistered inputs select the complete CI plan.

Inputs outside a crate are real dependencies. The census reads both sides of the change,
including literal/raw/multiline/concatenated Rust includes and runtime fixture paths. The
reviewed build-script and dynamic-include contracts in `tools/validation_inputs.json` are
source-hash bound. For example, `docs/FLAGS.md` generates engine code and is not merely prose.
A producer edit invalidates its declaration until the changed dependency is reviewed.
Unreferenced research receipts do not rebuild the engine. A standalone research probe retains
its own experiment requirement; it does not imply that every engine program changed.

The graph is an explicit dependency contract, not whole-program static analysis. New input
mechanisms must be declared or remain expanded. File names alone do not prove native
independence. Execution traces help find missing edges but do not by themselves prove an
unobserved edge cannot run.

## CI behavior

The existing check names remain stable. Each job uses its own affected-component decision.
Missing outputs or a failed classifier select the full job and clear partial package lists.
An explicit `contracts=none` skips unaffected Python contracts; a missing contract output runs
all available contracts. Thus a documentation-only change does not install sampled-test dependencies.
`validation-plan.json` is retained with the run, including the omission explanations.

Builds filter binary targets while retaining workspace feature unification. Clippy targets
affected packages only when Cargo reports the same resolved dependency feature programs as
the workspace; ambiguity or drift retains the full workspace check. Package verification also
includes forward dependencies and the same feature guard, so it does not silently use an older
published dependency or a different feature program. CUDA setup remains available for that
conservative Clippy/package fallback.
Architecture-conditioned native consumers retain the other-architecture compile check.
Existing CPU suite groups and their test/skip floors remain intact; selection omits a whole
unaffected group rather than hiding tests with name filters or weakening a floor.

Global integrity checks still run, including the public boundary, support/flag registries,
workflow validity and static skip census. The latter must remain global even when an unrelated
compiled suite is omitted. A malformed skip declaration cannot hide behind a skipped job.

Known Python contract changes run their actual tests without Cargo or GPU loading. Fast-gate
reports that CPU result separately from any live-evidence campaign the issue still owes.
The serving collector also selects its sampled-MTP consumer when that collector is present.
Sampled collector dependencies install from its tracked requirements into a temporary private
Python environment, which is removed on success or failure. The network-guard crate belongs to
the CPU core component and keeps its own zero-skip test floor when present in the workspace.
Tier 2 and release/tag qualification are unchanged. Native diagnostic selection remains
shadow-only until its own selected/full and failure-control evidence admits a narrower class.

## Cover several edges with one test

`tools/validation_coverage.py` selects a deterministic, cost-weighted set of tests from a
reviewed manifest. It favors additional required edges per unit cost. It is a greedy set-cover
heuristic, not a claim of a globally optimal schedule.

Each test declares:

- A stable ID and the independently asserted edges it covers.
- Hashes for every harness, oracle and helper input supporting that coverage claim.
- Its model, artifact, hardware and numerical-program scope for native execution.
- A positive cost, whose unit is stated by the manifest author.
- Mandatory regressions and required red-control test IDs.

The manifest also supplies the candidate context and required edges. A context mismatch,
changed harness, missing control or uncovered edge refuses the reduced plan. The optimizer
does not merge server boots or state. Fresh-process and reset requirements remain part of
each test's contract.

```sh
python3 tools/validation_coverage.py coverage-manifest.json --root . > selected-tests.json
python3 tools/validation_coverage.py coverage-manifest.json --root . --results results.json
```

`results.json` is keyed by selected test ID. Each entry names the returned `contract_id` and
needs `status: "passed"`, the exact
context, positive integer `executed`, zero integer `skipped`, and an `edges` map with explicit
`"passed"` assertions. Exit success alone does not cover an edge. The contract digest binds required edges, selected test definitions, controls, source pins and
execution context. Source hashes are checked again when results are admitted. Every asserted
edge of every selected regression and control must pass, including edges beyond the requested
subset. A no-change plan says no validation ran; it cannot produce a pass.
This validation is for the declared behavior assertions, not model or release admission.

A composite serving collector can, for example, assert streaming termination, cold/warm cache
accounting, offered concurrency, disconnect/recovery and a declared context envelope in one
cache-enabled boot, followed by its required cache-disabled red boot. Those are distinct
properties. Offered HTTP overlap does not prove a particular GPU batch width, and an 8k
context observation does not cover a model's maximum context. Keep each assertion and its
raw evidence separately visible, even when one invocation covers several edges.

The scoped-validation research receipt demonstrates selection and result checking against a
real composite run. It records the exact candidate context and the remaining qualification
limits; it does not infer savings against an unmeasured alternative run schedule.

## Deterministic scheduling contracts

`python3 tools/check-coalescer-contract.py --out target/coalescer-contract` extracts the exact
server Coalescer implementation, records its hash, and compiles CPU-only fixtures with `rustc`.
Channel handshakes assert pending membership and one-workspace serialization. A virtual clock
asserts adaptive timeout and publication-origin behavior without assuming an OS wake-up time.
Both positive fixtures and all three independently mutated controls must match their exact
test counts and expected failing assertions. CI and local-ci run the same entry point.

These replace the jittered full-batch-rate unit verdict, whose requested 1.5 ms sleep did not
bound actual scheduling delay. They do not establish a batching performance guarantee under
arbitrary host load. The observed failed batch widths remain in the research receipt.
