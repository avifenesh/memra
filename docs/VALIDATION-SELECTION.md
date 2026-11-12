# Validation selection

How `tools/validation_plan.py` decides which CPU checks a change needs, and what
`tools/validation_coverage.py` admits when one composite test covers several behavior edges. It is for
contributors reading a plan and for maintainers of the planner. Device evidence is a separate
question ([Testing](TESTING.md)); the hosted jobs are in [CI](CI.md).

The planner keeps three questions apart: which declared inputs and dependency edges the change can
affect; which checks cover those edges, and why another check can be omitted; and which assertions
actually passed for the candidate, model, hardware and request. Missing information means affected.
A successful build is not a numerical or serving result, and the planner never turns a CPU result or
a missing model into GPU evidence.

## Run it

```sh
python3 tools/validation_plan.py local --base origin/main > validation-plan.json
tools/fast-gate/fast-gate.sh --plan --diff origin/main
python3 tools/validation_plan.py contracts --selected <names>   # run named CPU contracts
python3 tools/validation_plan.py check-registry                 # every workspace package has a component
```

Other subcommands: `ci` (the event form CI uses), `census <ref>` (the input census of a commit) and
`cargo build|clippy|publish` (run Cargo for the selected packages). The plan JSON records the changed
paths, the directly owning packages, dependency witnesses, each binary edge decision, the selected CI
components (`build`, `clippy`, `server`, `engine`, `portable`, `core`, `lanes`, `arch`, `publish`),
the reason for every omission, the CPU contracts and the native obligations, which it reports and
never discharges. Untracked source is included; index-masked or ignored build inputs refuse a narrow
plan.

## Selection rules

- Cargo normal, development, build and target-specific dependencies all count. An edit to
  `memra-server` selects its checks without rerunning its unchanged dependencies' suites; an edit to
  `memra-gguf` reaches its consumers. Package groups: core (`memra-gguf`, `-reference`, `-tokenizer`,
  `-validate`, `-sampling`, `-net-guard`), portable (`memra-tier`, `-kv`, `-cli`), native
  (`memra-engine`, `-server`, `-probe`).
- Cargo manifests, toolchains, workflow changes, unresolved input expressions and unregistered
  inputs select the complete plan.
- Inputs outside a crate are dependencies. The census reads both sides of the change, including
  literal, raw, multiline and concatenated Rust includes and runtime fixture paths. Reviewed
  build-script and dynamic-include contracts in `tools/validation_inputs.json` are source-hash bound:
  `docs/FLAGS.md`, for example, generates engine code. A producer edit invalidates its declaration
  until the changed dependency is reviewed.
- Symlinks, unproven `..` traversal, missing directories, `cfg_attr` module paths and Rust source
  outside crates expand validation until their graph has an explicit contract.
- Unreferenced research receipts do not rebuild the engine. Research and oracle inputs keep
  fast-gate's native expansion even when no Cargo package rebuilds.
- Known Python contracts in the planner's `TOOL_CONTRACTS` (for example `public-boundary`,
  `support-records`, `q35-cache`, `sampled-mtp`) run their real tests without Cargo or GPU loading.
  A shared Python input selects every declared consumer that is present; a selected but incomplete or
  deleted input refuses execution instead of skipping the contract.
- Boundary checker, test, policy and allowlist inputs select the public-boundary CPU contract, whose
  runner requires at least 60 tests and refuses any skip or expected failure.
- The support-records contract derives its required evidence from the `docs/support-records.toml`
  of both trees; a referenced evidence path must exist, and unknown record shapes, noncanonical paths
  or symlink ambiguity under the copied docs and model-pack trees expand planning.
- Builds filter binary targets but keep workspace feature unification. Clippy narrows to affected
  packages only when Cargo reports the same resolved feature programs as the workspace; otherwise it
  checks the whole workspace. Architecture-conditioned native consumers keep the other-architecture
  compile check.
- Selection omits a whole unaffected suite group. It never hides tests with a name filter or lowers
  a test or skip floor.

The graph is an explicit dependency contract, not whole-program analysis. A new input mechanism is
declared or stays expanded; file names alone never prove native independence.

## Where the plan is used

In `public-ci` thin mode the trusted entry (`tools/trusted_public_ci.py`, run from the event base)
uses the plan to pick the build, clippy and arch packages, the scoped merge guards and the
changed-input CPU contracts. A failed or missing plan selects full mode. Full mode (`ci.yml`) runs the
complete inventory whatever the plan says, so global checks (public boundary, support and flag
registries, workflow validity, the static skip census) always run there ([CI](CI.md)). `fast-gate`
reports the same CPU contracts separately from any live-evidence campaign a change still owes. Native
diagnostic selection stays shadow-only until a selected/full comparison and failure controls admit a
narrower class ([fast-gate](../tools/fast-gate/README.md)).

## Cover several edges with one test

`tools/validation_coverage.py` picks a deterministic, cost-weighted set of tests from a reviewed
manifest, favoring the most uncovered required edges per unit cost. It is a greedy set cover, not an
optimal schedule.

```sh
python3 tools/validation_coverage.py coverage-manifest.json --root . > selected-tests.json
python3 tools/validation_coverage.py coverage-manifest.json --root . --results results.json
```

Each test in the manifest declares a stable id and the edges it independently asserts; hashes of
every harness, oracle and helper input behind that claim; its model, artifact, hardware and
numerical-program scope (each a nonempty string); a positive cost; and its mandatory regressions and
red-control test ids. The manifest also gives the candidate context and the required edges. A context
mismatch, a changed harness, a missing control or an uncovered edge refuses the reduced plan. The
optimizer never merges server boots or state.

`results.json` is keyed by selected test id. Each entry names its `contract_id` and needs
`status: "passed"`, the exact context, a positive integer `executed`, `skipped: 0` and an `edges` map
of explicit `"passed"` assertions; exit success alone covers nothing. Source hashes are checked again
at admission, and every asserted edge of every selected regression and control must pass. Context
matching preserves JSON types recursively. A no-change plan says no validation ran and cannot pass.

One composite boot can assert several properties (streaming termination, cache accounting, offered
concurrency, disconnect and recovery, a context envelope). Keep each assertion and its raw evidence
visible on its own: offered HTTP overlap does not prove a GPU batch width, and an 8k context
observation does not cover a model's maximum. The worked example, with its stated limits, is
`research/scoped-validation-20261002/`.

The earlier text of this page, including the full input-census hardening rules, is kept in
[history/TESTING.md](history/TESTING.md#select-validation-from-changed-inputs).
