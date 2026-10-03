# Coverage input admission

Selection and result admission checked a source file, then reopened its path for
hashing. A replacement alias could escape the source root between those operations.
The CLI manifest and result readers also opened file paths without type checks.

All four readers now use one inline stdlib helper. It resolves supported aliases,
checks the source-root boundary when applicable, anchors the resolved components
with directory descriptors, refuses known nonregular leaves, opens without following
replacement links and without blocking on FIFOs, and checks the opened file type.
Descriptors close on success and refusal. The root and explicitly chosen CLI document
parent are trusted bases. This is an opened-object guarantee, not an arbitrary
filesystem or code sandbox.

Contained leaf/directory aliases and absolute paths inside the source root remain
valid. Explicit CLI documents outside `--root` remain valid, including aliases.
Raw hashes and text normalization, typed contexts, schemas, deterministic weighted
cover, mandatory/control bundles, CPU/GPU identity axes and qualification policy
are unchanged. Selection makes unavailable inputs ineligible and expands when it
cannot cover the request. Admission refuses changed or unsafe sources before pass.

The actual old/new source calls reproduce two owned escaping-alias swaps. The CLI
observers show both original document readers reaching real owned FIFOs while
preventing content I/O. Result rows in these probes are synthetic CPU metadata.
They do not establish real test, model, runtime or native execution.

Twelve actual unittest methods cover regular bytes/text, supported aliases, missing
and escaping inputs, real FIFO/directory/socket leaves, alias and FIFO replacements,
resolved ancestor replacement, original-parent descriptor retention, descriptor
cleanup and normal/optimized CLI behavior. Eight guard-removal controls fail their
named assertions with zero fixture errors. The first two mutation attempts exposed
fixture cleanup/setup errors and remain explicitly non-admitted under `receipts/`.

The original source is exact to `567bad56ab`. Public controls reproduce at the
source tuple named by the receipt:

```sh
python3 research/coverage-input-io-984-20261003/tools/controls.py "$PWD" /path/to/owned-output
```

The producer refuses optimized Python before work and rejects skipped or expected
failure outcomes. This historical mutation proof is not a permanent CI gate.
CI retains the twelve behavioral methods in the framework suite, with floor 206.
`PROOF.json` binds the actual working-tree bytes and raw job; it does not relabel
that run as a clean commit. Existing immutable capture copies remain unchanged.
The helper adds no dependency beyond the standard library.

No native math/program, model artifact/default, compiler/build default, tolerance,
required native gate or GPU shadow policy changed. No native qualification.

Publicity: skipped: maintenance release.

The clean source at `0549d98a53` replays 12 IO methods, four before/after
consumers and eight coherent negatives with zero fixture errors, plus 40 coverage
and 16 live caller tests. Workflow and registry checks pass. `CLEAN-COMMIT.json`
retains this distinct tuple. The 206-framework result remains the actual earlier
working-tree run; its source/test bytes match this commit and it is not relabelled.
