# CPU checks and native GPU qualification

Standard public runners execute builds, static gates, and CPU tests in `ci.yml`.
`portable-suites` also runs `memra-tier`, `memra-kv`, and `memra-cli`, including
integration tests and doctests. `tools/ci-portable.sh` is shared with the native
local-CI entrypoint; it does not run GPU tests or grant GPU qualification.

## Requesting GPU qualification

`gpu-ci.yml` accepts an explicit full candidate commit through `workflow_dispatch`.
Its initial supported scope is the existing generic release battery on one physical
RTX PRO 6000 card, physical GPU0, Ubuntu 24.04. It does not replace additional
hardware, topology, model-support, HTTP, performance, or release-publication gates.

1. The CPU job checks out that exact candidate and requires the content-bound
   qualification tools described in `RELEASE-QUALIFICATION.md`. Missing prerequisites
   refuse before a GPU job can be queued.
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
   requested commit. A skipped GPU job passes only when the CPU verifier accepted
   reusable native evidence. CPU build success alone never passes this check.

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
