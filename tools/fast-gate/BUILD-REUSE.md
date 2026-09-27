# Reuse an exact controlled build

`tools/local_build_cache.py` retains an existing controlled build capsule across
sessions and worktrees. A hit restores its six binaries and provenance files to
a new directory. It avoids rerunning a build when that exact candidate and build
are still the intended input to the next step.

This is opt-in development tooling. It does not change fast-gate selection,
release qualification, pre-push policy, or the CUDA static archive fingerprint in
`crates/memra-engine/build.rs`. A cache hit proves build identity. Model, numeric,
request, hardware, topology, and serving qualification still need their original
gates. Loading these binaries on another machine does not qualify that machine.

## Use

Start with a completed `qualify-release.py build` directory. An oracle manifest
is not needed to retain and restore this build. Use a clean checkout at the
build's exact commit. The current producer admits `controlled-cargo-v3`,
Linux x86-64 and CUDA `120a`; this adapter inherits those limits.

The descriptor records one of two scopes. `controlled-build-only` contains
`source.json`, `build.json`, `build.log`, `fetch.log` and the six build binaries.
Its `oracle_manifest` is explicitly null. `gpu-ci-capsule` additionally contains
the pinned `oracle-manifest.json` required by `gpu-ci.py pack-build`. Attaching or
removing that manifest changes the descriptor and cache key. No oracle descriptor
is fabricated to make a build-only bundle look ready for capture.

Keep the descriptor independently of the cache. It is the caller's expected
source, compiler, platform, configuration, optional oracle and binary identity. Export it
only from a build whose provenance you already trust. The cache provides content
verification, not authenticity against a writer who can replace that descriptor.

```bash
# These paths are outside the source checkout. Each restore has a new output path.
python3 tools/local_build_cache.py describe \
  --repo "$PWD" --build "$BUILD_DIR" --out "$LANE_DIR/build.expected.json"

python3 tools/local_build_cache.py store \
  --repo "$PWD" --build "$BUILD_DIR" --expect "$LANE_DIR/build.expected.json"

# Later, or from another clean worktree at the identical commit:
python3 tools/local_build_cache.py load \
  --repo "$PWD" --expect "$LANE_DIR/build.expected.json" \
  --out "$LANE_DIR/restored-build"
```

`--cache` defaults to `~/.cache/memra/controlled-builds-v2`. All successful commands
emit a JSON record with the scope, content key and measured elapsed seconds. Missing,
stale, malformed or corrupt inputs fail with a nonzero exit status and a reason.
A miss is a reason to build or investigate the mismatch; it never runs a compiler
or silently changes a build specification. Repeated `store` validates an existing
entry and leaves its inode unchanged. `load` refuses an existing output path.

The restored directory contains a copy of each executable, not links to mutable cache
files. It contains no incremental Cargo target, dependency cache, model weights,
GPU receipts or resident process state. A build-only restore is not a valid
`gpu-ci.py` capture input: first attach real pinned oracle inputs through the
existing GPU CI flow. A full GPU capsule can be passed as `--build` to that flow.
Qualification still verifies the runtime platform and all other capture inputs.

## Admission and storage

The v2 expectation binds the complete `source.json`, `build.json`, optional parsed
oracle manifest, scope, and size/SHA-256 of every permitted capsule member. Thus it includes the
exact commit, tracked modes, source input inventory, build command, compiler
executable identities, compiler environment, numeric settings, build platform,
logs, and binary hashes recorded by the controlled producer. Full-record matching
is deliberately conservative: a rebuild with new log bytes receives another key.
No cross-commit or semantic build equivalence is claimed. V1 expectations are
rejected; export a v2 expectation from the trusted build to use this schema.

At export, store and restore, `release_inputs.verify_checkout` verifies actual
tracked bytes and modes, symlink closure, untracked and ignored build/gate inputs,
and effective ancestor/repository Cargo configuration. `release_qualification`
checks source provenance; the adapter additionally requires the exact candidate
and snapshot. Git index hints cannot conceal changed bytes. These checks inherit
the producer's admitted input namespaces and controlled compiler filesystem
boundary. They do not make an arbitrary Cargo invocation hermetic.

The existing `gpu-ci.py` functions verify, pack and unpack full GPU capsules.
Before oracle attachment, the adapter uses the same
`release_qualification.validate_build` authority and exact binary identity
checks. A narrow local pack/unpack path admits only the ten build files, with
the existing capsule size ceiling and the same regular-file, duplicate-member,
path and inventory restrictions. Unexpected files, links and missing members
are rejected. The producer's working directory can contain other build outputs;
only the declared capsule inventory is packed. Store
unpacks its private archive and verifies the resulting payload before publishing
with an atomic no-replace hard link. Each writer uses its own temporary directory
on the cache filesystem. Published archives are read-only. A competing publisher
can validate the existing entry but cannot replace it through this API. Restore
unpacks privately, verifies every payload against the independently held
expectation, runs the existing build verifier, and checks the checkout again.
Temporary directories are removed on success and failure. Use a local filesystem
supporting hard links and rename; filesystem errors fail closed.

The entire capsule is rehashed on admission. This costs CPU and I/O in proportion
to source and binary size. Measure it against the actual build before selecting
it for a lane. The cache does not skip compilation after edits and does not reduce
model loading or gate execution time. It preserves the exact build context that
the current producer records, not a claim about unrecorded inputs in other build
systems. The descriptor and cache should receive the same custody as build logs.

## CPU checks and native acceptance

Run the bounded admission suite with a non-vacuity floor:

```bash
PYTHONDONTWRITEBYTECODE=1 bash tools/unittest-floor.sh tools test_local_build_cache.py 46 -v
```

The suite uses the real source and build verifiers with temporary, clearly marked
synthetic capsule payloads. The same controls run against both inventories.
They exercise actual-byte changes hidden by Git flags,
mode changes, included fixtures, untracked/ignored inputs, ancestor Cargo config,
exact-commit rejection, compiler/platform/numeric/oracle changes, binary and
record tampering, missing/truncated/duplicate/unexpected payloads, immutable
publication, independent restore files, and the complete CLI path. Build-only
controls also refuse missing native provenance and oracle attachments hidden in
an archive. CLI timing rows are synthetic admission
cost, not native compilation savings or GPU evidence.

Before claiming saved native latency, record one controlled native build's full
elapsed time and capsule size, then record store/load elapsed time in the same
environment. Compare the restored binary hashes with that build and run its next
required gate unchanged on the designated non-production hardware. A cache hit
alone cannot satisfy that gate. CPU admission tests are sufficient to exercise
this adapter's refusal behavior, not to qualify any model or hardware.
