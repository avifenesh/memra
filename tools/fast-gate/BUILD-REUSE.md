# Reuse an exact controlled build

`tools/local_build_cache.py` keeps a completed controlled build capsule across sessions and worktrees
and restores its six binaries and provenance files into a new directory. It is for contributors who
would otherwise rebuild the same candidate for the next step. It is opt-in development tooling: it
changes no fast-gate selection, release qualification, pre-push policy or the CUDA archive
fingerprint in `crates/memra-engine/build.rs`. A cache hit proves build identity and nothing else;
model, numeric, request, hardware, topology and serving gates still run, and loading the binaries on
another machine proves nothing about that machine.

## Use

Start from a completed `tools/qualify-release.py build` directory and a clean checkout at the
build's exact commit. The producer admits `controlled-cargo-v3`, Linux x86-64 and CUDA `120a`; this
tool inherits those limits.

```bash
# Paths outside the source checkout. Each restore needs a new output path.
python3 tools/local_build_cache.py describe \
  --repo "$PWD" --build "$BUILD_DIR" --out "$LANE_DIR/build.expected.json"

python3 tools/local_build_cache.py store \
  --repo "$PWD" --build "$BUILD_DIR" --expect "$LANE_DIR/build.expected.json"

# Later, or from another clean worktree at the identical commit:
python3 tools/local_build_cache.py load \
  --repo "$PWD" --expect "$LANE_DIR/build.expected.json" \
  --out "$LANE_DIR/restored-build"
```

`--cache` defaults to `~/.cache/memra/controlled-builds-v2`. Every success prints a JSON record with
the scope, content key and elapsed seconds. Missing, stale, malformed or corrupt inputs exit nonzero
with a reason; a miss never runs a compiler. A repeated `store` validates the existing entry and
leaves it untouched, and `load` refuses an existing output path.

The descriptor (`describe`) has one of two scopes. `controlled-build-only` holds `source.json`,
`build.json`, `build.log`, `fetch.log` and the six binaries, with `oracle_manifest` null.
`gpu-ci-capsule` adds the pinned `oracle-manifest.json` that `gpu-ci.py pack-build` needs; adding or
removing it changes the key. A build-only restore is not a `gpu-ci.py` capture input until real
pinned oracle inputs are attached through the GPU CI flow ([CI](../../docs/CI.md)).

Keep the descriptor apart from the cache, and export it only from a build you already trust: the
cache verifies content, not authenticity against someone who can replace the descriptor.

## What is checked

The v2 descriptor binds all of `source.json` and `build.json`, the optional oracle manifest, the
scope and the size and SHA-256 of every capsule member: the exact commit, tracked modes, source
inventory, build command, compiler identities and environment, numeric settings, platform, logs and
binary hashes. A rebuild with new log bytes gets a new key; no cross-commit equivalence is claimed,
and v1 descriptors are rejected. Export, store and restore re-verify the checkout's tracked bytes,
modes, symlinks, untracked and ignored inputs and effective Cargo configuration. Restores unpack
privately, verify every payload against the descriptor, run the build verifier and write plain file
copies: no links to cache files, no Cargo target, no model weights. Publication is an atomic
no-replace hard link with read-only archives, so use a local filesystem with hard links and rename.
The whole capsule is rehashed on admission; measure that cost against the build before relying on it.

## Tests and native acceptance

```bash
PYTHONDONTWRITEBYTECODE=1 bash tools/unittest-floor.sh tools test_local_build_cache.py 48 -v
```

48 is the floor `.github/workflows/ci.yml` enforces for this suite. It runs the real source and build
verifiers against synthetic capsules: hidden byte and mode changes, untracked and ignored inputs,
ancestor Cargo config, wrong commits, compiler, platform, numeric and oracle changes, tampered
binaries and records, missing or duplicate members and the whole CLI. Its timings are admission cost,
not native build savings.

Before claiming saved native time, record one controlled build's full elapsed time and capsule size,
then store and load times in the same environment, compare the restored binary hashes with the build
and run the next required gate unchanged on the target hardware. A cache hit never satisfies that gate.

The earlier text of this page is kept in
[docs/history/TESTING.md](../../docs/history/TESTING.md#reuse-an-exact-controlled-build).
