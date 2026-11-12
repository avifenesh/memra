# CI

What the hosted GitHub workflows run, how a change is routed to thin or full CPU validation, and how
to request a GPU qualification run. It is for contributors reading a CI result. Hosted runners have
no GPU: device evidence comes from the batteries in [Testing](TESTING.md); CPU job selection is in
[Validation selection](VALIDATION-SELECTION.md). The earlier text of this page is in
[history/CI.md](history/CI.md).

## Workflows

| workflow | trigger | runs |
|---|---|---|
| `ci-public.yml` (`public-ci`) | PR to main; push to main | `route`, then the thin jobs or `full`; always `ci/merge-result` |
| `ci.yml` (`ci`) | `workflow_call` (from `public-ci` full mode, `release.yml`, `publish.yml`); daily at 02:19 UTC; manual | the complete CPU inventory |
| `gpu-ci.yml` | manual, with a full 40-character candidate commit | controlled build, a GPU capture on a provisioned worker, the `gpu-qualification/pro-ubuntu24` check |
| `boundary-refs.yml` | daily at 06:10 UTC; manual | the public-boundary scan over published refs |
| `release.yml` | tag `v*` | `ci.yml`, the release guard, the build matrix, the GitHub release ([Releasing](RELEASING.md)) |
| `publish.yml` | tag `v*` (publishes); manual (a dry run unless the `publish` input is true) | `ci.yml`, then the crates.io publish |
| `mtp-context-receipts.yml`, `mtp-depth-receipts.yml`, `prompt-depth-router.yml` | PRs that touch their research directories; manual | reproduction of those research receipts |
| `prompt-prefix-depth.yml` | `workflow_call` from `prompt-depth-router.yml`; manual | the prefix-depth receipt and its native compile check |

`ci.yml` has no PR or push trigger of its own: a PR or a push to main runs `ci-public.yml`, which
calls `ci.yml` only in full mode.

## Thin or full

`route` picks thin mode only when the event is a PR to main opened by the repository owner from a
repository the owner owns (checked by the PR user's login and immutable account id and by the head
repository's owner), or a push to main, **and** `tools/trusted_public_ci.py route-plan`, run with
isolated Python from the event base checkout (never the candidate), plans thin. Everything else is
full: external and fork PRs, unknown events, a missing trusted entry, and any change the trusted
planner expands (routing or policy source, unknown dependency ownership, missing inputs). The actor,
labels and commit authors never select thin mode.

Thin mode runs, for the affected packages only:

- `merge-validation`, each command once: the static merge guards (action pins, workflow keys,
  conflict markers, docs registry, publish members, stub ABI, arch matrix), the scoped guards the
  plan reaches (PDL chain, flags coverage, gate OFF-arm lint, support-state census, coalescer
  contract), the changed-input CPU contracts and the base's link checker
  ([`tools/check-doc-links.py`](../tools/check-doc-links.py)) over the candidate's markdown.
- `boundary`: a public-boundary scan of every blob introduced in the pinned commit range. It does
  not prove whole-tree allowlist drift; full mode does.
- `build` and `clippy` through `tools/validation_plan.py cargo`, with the build-identity check, the
  DSv4 native control admission and `cargo fmt --all -- --check`.
- `arch` at `sm_100a` and `sm_121a` with the fatbin-vs-lookup census; the `100a` cell also runs the
  B200 kernel-twin ABI, SASS, layout and admission gates.

Functional tests for an owner change run locally before the push.

Full mode is `ci.yml`. Its job selection is a workflow constant, never candidate output, and
`release.yml` and `publish.yml` wait for it before any effect. This is the merge-time check
inventory:

| job | what it runs |
|---|---|
| `changes` | records the complete selection as `validation-plan.json` |
| `gates` | the action-pin, workflow-key, conflict-marker, docs-registry, support-state and flags censuses with their self-tests; the releasability censuses (publish members, stub ABI, arch matrix), their teeth and the crates.io publish-order fixture; the gate-integrity, release-guard, release-battery, argmax-floor, changelog, pre-push, Q35 cold-mixed, canonical-config and MLA/KDA memory fixtures; the fatbin-lookup and skip-census teeth; the `remove_var` door lint and the PDL-chain census; the validation-selection, release-qualification, compiler-input, build-reuse, cache-qualification, per-card runner and physical-GPU-resolution controls; the CPU admission contracts and the coalescer contract; the public CI integration test; the CUDA-free unit suites through the skip census (memra-gguf, -reference, -tokenizer, -validate and -sampling at budget 20 and at least 400 passed; memra-net-guard at budget 0) and memra-lanes |
| `boundary` | `tools/test_public_boundary.py`, then `tools/check-public-boundary.py check` over the whole tree and `verify-allowlist` (allowlist drift) |
| `build` | the release binaries at `sm_120a`, the build-identity check, the parity-geometry and argmax instrument tests, the DSv4 CPU contract tests |
| `clippy` | `-D warnings` over the workspace |
| `server-tests` | `cargo test --release -p memra-server` |
| `portable-suites` | `tools/portable-suites.sh` and its teeth, the tier collector suite with both rig lock paths held, the GPU CI adapter's CPU controls |
| `engine-tests` | the memra-native CPU expert backend, the DSv4 door-reach gate, `memra-engine --lib` unfiltered through the skip census (budget 0, at least 300 passed) |
| `arch-coverage` | builds at `sm_100a` and `sm_121a` with the fatbin-vs-lookup census; the `100a` cell also runs the B200 static teeth |
| `publish-dryrun` | crate package verification on the newest stable toolchain |
| `complete` | requires every job above and reports the validated head to the caller |

The commands and pass conditions of the CPU contracts and fixtures are in
[Testing](TESTING.md#cpu-contracts-and-fixtures).

## `ci/merge-result`

`ci/merge-result` reports one source-bound result per `public-ci` event. In thin mode
`trusted_public_ci.py result` checks every selected job and the contract execution receipt against
the same commit; in full mode it requires `route` and `full` to succeed and `ci.yml` to report the
exact head as validated. A skipped, failed or cancelled selected job, a stale receipt or a wrong
commit fails it. It is not a required check until a branch protection rule requires it.

`tools/ci_merge_validation.json` binds the full command inventory, both workflows' bytes and the
static merge guard inputs, so a changed command or a missing mapping cannot silently drop coverage;
a deliberate workflow edit updates its hashes in the same change.
`python3 tools/test_public_ci_integration.py` drives routing and execution against real Git events.
`tools/check-workflow-keys.py` (CI `gates` job and the pre-push hook, no skip switch) refuses a
duplicate mapping key at any depth, since GitHub runs zero jobs for such a file; it reports a
construct outside its scope as "cannot answer" (exit 2), never as green.

## Requesting GPU qualification

`gpu-ci.yml` runs the generic release battery for an explicit candidate on one RTX PRO 6000,
physical GPU0, Ubuntu 24.04. That battery covers the qwen35 class only, the two
`tools/release-roster.tsv` checkpoints
([what a record covers](RELEASE-QUALIFICATION.md#what-a-record-covers)); it replaces no other
hardware, topology, model, HTTP, performance or publication gate. The `build_reuse_only` input
measures the controlled build and a verified restore on CPU, with no GPU job.

1. The CPU job checks out the exact candidate and requires `tools/qualify-release.py`,
   `release_qualification.py`, `release_inputs.py` and `release_input_view.py`.
2. A committed qualification for the same source inputs and OS profile is reused through
   `release_qualification.py verify`: the original evidence, not a new binary.
3. Otherwise the input manifest (below) must be set before any GPU time; the job then builds through
   the controlled native builder and packages the six binaries and their build evidence.
4. The coordinator provisions an isolated worker labelled `memra-ci-pro-ubuntu24-<run-id>`, which
   checks the capsule, proves a real CUDA allocation, downloads the exact inputs and runs the capture
   inside the coordinator's physical-card lease wrapper.
5. The seal runs only after the wrapper exits and writes its cleanup record. Raw output and lease
   records are kept as artifacts, failures included.
6. A standard-runner job attaches `gpu-qualification/pro-ubuntu24` to the candidate. A skipped GPU
   job passes only when the CPU verifier accepted reusable evidence; a CPU build never passes it.

Provisioning and cleanup belong to the coordinator: the workflow holds no provider credentials. The
GPU job has a read-only token; only the reporting job writes checks. Run it only for candidates the
maintainer selected. `tools/unittest-floor.sh tools test_gpu_ci.py 9` runs the adapter's CPU
failure controls in CI.

### Input manifest

Set the repository variables `GPU_CI_INPUTS_URL` and `GPU_CI_INPUTS_SHA256` to an immutable HTTPS
manifest and its SHA-256. It names the lease wrapper and the named kernel-oracle GGUF files; never
rename or substitute an oracle to satisfy the inventory. The example shows the schema only:

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

Downloads enforce HTTPS, exact sizes and SHA-256, unique plain file names, disk headroom and a total
ceiling. Capsules admit only the named metadata and the six executables, with no links, duplicates or
traversal. Python imports use a private cache outside the checkout and write no bytecode.

### Receipt custody

Download the controlled-build and native-evidence artifacts before retention expires, and keep the
executable archive apart from the evidence. Bank reviewed evidence with the native qualifier's
`bank` command in a clean candidate checkout; the workflow never pushes a publication commit. The
release workflow still validates committed evidence and the exact bytes it packages, and a result for
this OS and card covers no other OS profile or hardware path.
