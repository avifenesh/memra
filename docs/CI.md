# CPU checks and native GPU qualification

Standard public runners execute builds, static gates, and CPU tests in `ci.yml`.
The `portable-suites` job runs the memra-tier, memra-kv and memra-cli suites, including
integration tests and compile-fail doctests, through **one entry point**,
`tools/portable-suites.sh`: `cargo test -p memra-tier -p memra-kv -p memra-cli --offline
--locked --no-fail-fast` behind `tools/skip-census.py` (static census over `src/` and
`tests/`, run-side budget 0, `--min-passed 300`), the raw cargo output banked at
`target/portable-suites.log`. The same job runs the wrapper's teeth
(`tools/test_portable_suites.sh`: planted failures must red it, an undeclared skip must red
the census, the wiring holds), the collector suite with both rig lock paths held under a
unittest floor, and the floor tool's own teeth. `tools/local-ci.sh`'s CPU chain runs the same
wrapper with `CARGO_BUILD_JOBS=8 RUST_TEST_THREADS=8` while it overlaps the chain with device
checks; hosted execution keeps its runner defaults. `tools/ci-portable.sh` (the name PR #590
introduced for a second runner of the same suites) only forwards to the wrapper. Details and
the measured counts: `docs/TESTING.md`, "Standing execution". Nothing in that job runs GPU
tests or grants GPU qualification.

The same job ends with the CPU failure controls of the GPU adapter below
(`tools/unittest-floor.sh tools test_gpu_ci.py 9`; 10 measured 2026-09-21; a bare
`unittest discover` is green over zero tests). The `gates` job runs
`tools/check-workflow-keys.py`, which loads every workflow file with a loader that refuses a
duplicate mapping key. GitHub refuses a workflow file with a duplicate
key and runs zero jobs; main carried two `portable-suites` jobs on 2026-09-21 (#592 and #590)
and every run was red at the workflow level until one was removed. The `gates` step protects
the other workflow files; `ci.yml` itself is protected by the same census in
`tools/hooks/pre-push` (no skip switch) and by a PR run on a current merge ref.

## Requesting GPU qualification

`gpu-ci.yml` accepts an explicit full candidate commit through `workflow_dispatch`.
Its initial supported scope is the existing generic release battery on one physical
RTX PRO 6000 card, physical GPU0, Ubuntu 24.04. It does not replace additional
hardware, topology, model-support, HTTP, performance, or release-publication gates.

1. The CPU job checks out that exact candidate and requires the content-bound
   qualification tooling tracked in [PR #566](https://github.com/avifenesh/memra/pull/566).
   Until that dependency and the pinned input configuration are present, dispatch
   fails as an unconfigured request before GPU queueing and publishes no candidate
   check run.
2. A valid committed qualification for the same source inputs and OS profile may
   be reused through `release_qualification.py verify`. This reuses the original
   tested evidence, not a new binary or an mtime.
3. Otherwise, the CPU job requires a pinned coordinator input manifest, compiles
   through the controlled native builder, and packages its six ELF files and build
   evidence. The GPU job receives that immutable Actions artifact ID and SHA-256.
4. The coordinator provisions an isolated worker with the run-specific label
   `memra-ci-pro-ubuntu24-<run-id>`. The worker checks the capsule, proves a real CUDA
   allocation, downloads the exact input bytes, and invokes the coordinator's
   physical-card lease wrapper around the native capture.
5. Seal runs only after the wrapper exits and writes its final cleanup record.
   Failed or interrupted work cannot seal. Raw native output and lease records are
   retained as Actions artifacts, including failures.
6. A separate standard-runner job attaches `gpu-qualification/pro-ubuntu24` to the
   requested candidate, comparing it with the commit read from the validated sealed
   source descriptor for a new capture. A skipped GPU job passes only when the CPU
   verifier accepted reusable native evidence. CPU build success alone never passes
   this check.

Provisioning and cleanup are coordinator responsibilities. The engine workflow
contains no provider credentials or fleet policy. The GPU job has a read-only
repository token; only the final reporting job can write checks. Keep public PR
code isolated from deployment credentials, and run the workflow only for candidates
the maintainer has selected for native qualification.

## Input manifest

Set repository variables `GPU_CI_INPUTS_URL` and `GPU_CI_INPUTS_SHA256` to an immutable
HTTPS manifest and its SHA-256. The manifest identifies the existing coordinator
wrapper and the actual named kernel-oracle GGUF files. Do not rename or substitute
an oracle to satisfy the inventory. The qualifier retains the full required-cell
manifests, actual model hashes and skip budgets.

```json
{
  "schema": "memra-gpu-ci-inputs-v1",
  "lease_wrapper": {
    "name": "memra-gpu-run",
    "url": "https://example.invalid/immutable-wrapper",
    "sha256": "<64 lowercase hex characters>",
    "size": 12345
  },
  "oracles": [
    {
      "name": "<actual-named-oracle>.gguf",
      "url": "https://example.invalid/immutable-model",
      "sha256": "<64 lowercase hex characters>",
      "size": 123456789
    }
  ]
}
```

The example is a schema illustration, not a usable model manifest. The coordinator
supplies real URLs, byte counts, hashes, and the established wrapper. The adapter
does not invent another lock namespace or fabricate lease evidence.

Downloads enforce HTTPS, exact byte counts and SHA-256, unique plain filenames,
disk headroom and a total input ceiling. Artifacts cannot redirect outside HTTPS.
Build capsules permit only the named metadata and six executables, with no links,
duplicate members or traversal. A foreign source/build or changed ELF refuses.
Python imports use a private cache outside the checkout and disable bytecode writes,
so adapter verification cannot contaminate the native producer's clean-source gate.

## Receipt custody and release

Download both the controlled-build and native-evidence artifacts before their
retention expires. Preserve the exact executable archive separately from the
evidence publication. Bank reviewed evidence using the native qualifier's `bank`
command in a clean candidate checkout; the workflow never pushes a publication
commit or bypasses review automatically.

The release workflow still validates committed evidence and the exact bytes being
packaged. A CI result for this one OS/card scope does not qualify another shipped
OS profile or another hardware path. The input manifest and runner must be ready
before dispatch; an unconfigured workflow is explicitly unqualified.
