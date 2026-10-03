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
including literal/raw/multiline/concatenated Rust includes and runtime fixture paths. Textual
Rust includes across crates inherit all source and external inputs from the source-owning
crate, transitively. Rust source outside crates and referenced symlinks expand validation
until their module-resolution graph has an explicit contract. Whitespace and comments between
macro tokens do not hide an include. The
reviewed build-script and dynamic-include contracts in `tools/validation_inputs.json` are
source-hash bound. For example, `docs/FLAGS.md` generates engine code and is not merely prose.
A producer edit invalidates its declaration until the changed dependency is reviewed.
Unreferenced research receipts do not rebuild the engine. A standalone research probe retains
its own experiment requirement; it does not imply that every engine program changed.

Runtime fixture paths under `research` and `docs` retain their canonical targets across
`.` and proven `..` traversal. Removing a parent component requires a known directory in
the inspected tree and no symlink along the traversed spelling. Symlinks, missing
directories, escaping traversal and pattern-dependent parents expand validation.
Compiled include literals use the same physical traversal check after their complete
`concat!` or registered environment expression is resolved. A symlink cannot disappear
through `..` cancellation, even when its path is split across string fragments.
Conditional `cfg_attr` module path attributes expand validation until their transitive
module graph has a contract; the selector does not evaluate cfg truth. Raw identifiers
for path and cfg_attr retain the same input reach as their ordinary spellings. Comments and
quoted example code cannot create those attribute readers.

Format and glob suffixes keep their conservative reach. Traversal to the repository root
covers the whole subtree, and the census inspects both sides of deletions and renames.

The graph is an explicit dependency contract, not whole-program static analysis. New input
mechanisms must be declared or remain expanded. File names alone do not prove native
independence. Execution traces help find missing edges but do not by themselves prove an
unobserved edge cannot run.

## CI behavior

The existing check names remain stable. Each job uses its own affected-component decision.
Missing outputs or a failed classifier select the full job and clear partial package lists.
An explicit `contracts=none` skips unaffected Python contracts; a missing contract output runs
all available contracts. Thus a documentation-only change does not install sampled-test dependencies.

Shared Python inputs select every declared present consumer. For example,
`cache_qualification.py` reaches Q35 consistency tests, the background and serving
collectors directly, and sampled-MTP through the serving collector. Each keeps its existing CPU command and
native obligations. An absent collector is not invented; a selected incomplete or
deleted input refuses execution instead of silently skipping the contract.
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
The Q35 cache contract also includes its consumed `research/sellgate-20260812/workload.lock.json`
and the two banked logs under `research/spill-lead-20260919/integration-day12/integ68-q35ab/`:
`main-q35-cold-mixed.log` and `integ68-q35-cold-mixed.log`. Prompt IDs are generated from the
workload fields; replay goldens come from seed rows in those logs. This exact input inventory
does not infer all repository fixture dependencies. The historical inputs and Q35 native
obligation remain unchanged. Its always-run CI replay is retained independently of selection.
The support-records contract derives persisted gate evidence from the before and after
`docs/support-records.toml` trees. Each referenced evidence path is required for execution;
both exact sibling `artifact.lock` and `tiny-gate.tsv` paths affect selection even when absent.
The census chooses `artifact.lock` first and otherwise reads `tiny-gate.tsv`, so optional-file
creation and deletion must retain the consumer. Neither alternative is made unconditionally
required. Missing required evidence retains a named refusal. Changed census/test reader bytes,
unknown record shapes or CI tokens, noncanonical paths and symlink ambiguity expand planning.
Unknown-reader execution expands to all available CPU names, including explicit none and stale
subsets, and runs the actual census without trusting derived required-evidence assumptions.
Static inputs, metadata syntax, canonical paths and symlink checks remain mandatory. Execution admission validates every census reader and
record as a regular file before reading any of their content. Reads anchor each path
component to directory descriptors with nofollow opens. A nonblocking leaf open and
fstat refuse a FIFO replacement between metadata inspection and opening. Hashes still
use the exact raw reader bytes; type or symlink failures remain hard refusals even
when unknown-reader expansion is enabled. The checkout root remains the trusted base.
 Known
reader evidence keeps its named missing-file refusal.
The same pinned readers select exact current and potential one-level pack modules, root pack
lists, CLI source, root README/STATUS/AGENTS and non-archive docs Markdown. Rust package and probe
handling runs before the content shortcut, retaining every native obligation. Both event trees
contribute inputs, including creation and deletion. No repository-wide fixture inference.
The CPU test copies whole docs and model_packs trees, so symlink, gitlink and unsupported type
ambiguity under those roots or their ancestors expands planning even for excluded content.
Planning checks these types before reader or compiler-input content I/O. The local no-follow
inventory includes ignored copy inputs; ignored actual reader content prevents scoped planning.
Unreadable local regular files also expand because the fixture cannot copy them.
Exact local ancestor queries retain staged gitlink types when physical directories exist.
Readable excluded archive/non-mod content does not gain a blanket census requirement. The
independent always-run support checks and tested execution fallback remain unchanged. Broader
cited-receipt-parent transport is a separate input-closure question.
Its CPU shortcut requires every changed input to be a declared CPU contract or plain docs.
Research/oracle inputs retain fast-gate's native expansion even when they do not rebuild a
Cargo package; an absent compiler dependency does not prove an absent native probe dependency.
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
  Each native identity must be a nonempty string in both the contract and execution context.
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
An empty requested-edge list produces no-change only when no mandatory test is requested.
Explicit mandatory requests still select their complete control bundles and validate their
scope and source pins. Unavailable mandatory tests expand selection. With zero requested
edges, execution admission still requires every selected guard/control assertion, a real
execution count and no skips; it grants no native qualification.
Scope keys must be present in the execution context, including explicitly declared nulls.
Context matching preserves JSON types recursively: booleans, integers and floating-point
values cannot substitute for one another, even inside arrays or objects. Object key order
does not change identity. Signed floating-point zero retains the distinction already bound
by the contract digest. The same comparison protects the selected, bound and per-result
contexts; a typed mismatch expands selection or refuses result admission.
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
A real callback advances virtual time, publishes through production code, and drives a pending
row. Removing either duration or publication writes must fail that assertion.
Both positive fixtures and all five independently mutated controls must match their exact
test counts and expected failing assertions. CI and local-ci run the same entry point.

These replace the jittered full-batch-rate unit verdict, whose requested 1.5 ms sleep did not
bound actual scheduling delay. They do not establish a batching performance guarantee under
arbitrary host load. The observed failed batch widths remain in the research receipt.
