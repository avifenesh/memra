# Development feedback

Fast-gate explains the next development check. CPU-only tooling uses the contract
and failure-injection checks in docs/TESTING.md. Native runtime/model changes and
publication keep their qualification gates. A tooling merge grants no GPU or model
qualification and does not promote narrower GPU selection out of shadow mode.

For a CPU-only tooling change, run its CPU contract and failure-injection tests.
The native diagnostic plan below does not turn an unrelated tooling change into
a requirement to execute GPU probes.

## Plan before building

```sh
tools/fast-gate/fast-gate.sh --plan --diff origin/main
python3 tools/fast-gate/plan.py --changed crates/memra-engine/src/spec.rs --json
python3 tools/fast-gate/plan.py --context-changed numeric --json
python3 tools/fast-gate/plan.py --cache "$HOME/.cache/memra/components" --json
```

`map.tsv` seeds the existing probes. `dependencies.json` declares transitive
dependencies for build inputs, generated/oracle inputs, harnesses, shared CUDA,
runtime and model plans. Every changed path carries its matching map rows and
direct graph nodes. Every propagated node names the dependencies that reached it.
Compiler, build, runtime, numeric, harness, oracle, model, binary, hardware,
topology and request context changes expand native validation.

Unknown paths, index-masked inputs, ignored build inputs and broad legacy no-gate
rows cannot establish independence. They expand the plan. Changed research data
can be an `include_str!` fixture; a changed gate is also an input. MiMo text,
vision, audio, MTP and serving requirements stay explicit even though the legacy
probe registry does not cover all of them.

Native selection is **shadow-only**. An expanded plan stops before compilation
unless `--probes` explicitly requests named development diagnostics. At the
qualification checkpoint run tier 2 and the affected model's own gates. Explicit
probes prove only those probes. Missing models, missing goldens, self-SKIP and
zero stream agreement fail. An invalid Git ref refuses instead of becoming an
empty diff. A no-change response says that no validation ran.
Explicit probes also work in a source-only tree without Git, with change coverage
reported as unknown. Golden refresh requires Git provenance. A default refresh
pins available goldens and reports missing ones; self-gating probes have no
goldens and are excluded. An explicitly requested missing golden or a refresh
that writes nothing fails.

Before admitting a narrower GPU class, retain a selected/full comparison on the
same candidate, artifact, request shape and target hardware. Replay its historical
failure and a mechanism-changing red arm. A miss expands that class and blocks
adoption. Coverage observations are supporting data, not independence proof.

## Reuse an isolated CPU component

```sh
tools/fast-gate/fast-gate.sh --component qualification-contracts \
  --cache "$HOME/.cache/memra/components"
tools/fast-gate/fast-gate.sh --component release-inputs \
  --cache "$HOME/.cache/memra/components"
```

The command reports `run` or `reuse`, the exact key, receipt path, input
fingerprinting, execution and total elapsed seconds. `--fresh` executes a shadow
check even if a matching result exists. The registry currently admits two CPU
contract suites. They test qualification admission using temporary fabricated
records. They do not run native kernels, compile an engine or qualify a model.

Only the listed source files are mounted in a fresh, read-only input view. Python,
Git, required command binaries, their shared libraries and the visible standard
library are fingerprinted. The subprocess has a fixed environment, fresh temporary
state, no host home, no network and no GPU devices. Unlisted source and ignored
configuration are invisible. A newly consumed file therefore fails until the
contract declares it. This OS boundary is what permits an unrelated source edit
to reuse this component's result. Filename matching alone would not suffice.

Source bytes, executable modes, harness, fixture, contract and runtime identities
are part of the key. Inputs are checked again after execution. Skips, empty suites,
duplicate test IDs, missing results and failed tests cannot publish a reusable
record. Reuse rechecks the input identity, raw log and result hashes. Publication
is atomic; concurrent writers use private temporary directories and a nonblocking
lock. Cache corruption refuses; it does not silently replace the evidence.
Fresh shadow runs retain separate raw records. A fresh failure for an unchanged
key quarantines its older pass and refuses further reuse until the dependency
contract is repaired. Failed raw output remains discoverable beside the cache.

This first runtime contract requires Linux, system Python under `/usr/bin` and
working bubblewrap namespaces. There is no ambient execution fallback. Contracts
and receipt directories are trusted local developer inputs, not signatures or a
defense against a malicious cache owner. Keep the cache outside the source tree.
The returned CPU result has no authority over whole-binary/model qualification.

## Build and model setup

[BUILD-REUSE.md](BUILD-REUSE.md) describes exact controlled-build capsule reuse.
Restores create private files and never share a mutable Cargo target. CUDA archive
hashing remains in `build.rs`; the cache cannot revive an older archive under a
changed input identity.

Resident model setup is not implemented here. Some probes already share a sharded
load (`ppsplit` runs reference, serial split and dynamic split in one process).
Extending that pattern requires verified reset of KV, RNG, graph, allocator and
request state, plus reversed-order controls. Cold-start, isolation, allocation and
fresh-process performance tests retain their original workload.
