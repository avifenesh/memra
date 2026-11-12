# The build fingerprint is a content id of the compiled source (2026-09-01)

**Status:** adopted 2026-09-01 (lane/real-system-fingerprint). No flag. Code:
`crates/memra-server/build.rs` and `crates/memra-server/src/build_id.rs` (the build script
`include!`s the module, so one implementation serves the build and the tests). Gates:
`mod build_identity_tests` in `crates/memra-server/src/lib.rs`.

The rationale below moved out of the serving doc on 2026-11-12. The old text is kept in
[docs/archive/SERVING-2026-11-12.md](../archive/SERVING-2026-11-12.md#the-build-fingerprint-lanereal-system-fingerprint-2026-09-01).

## Chosen

`system_fingerprint` is `memra-<crate version>-<content id>`, for example
`memra-0.123.0-6371ca8a0af4` (the lane's head). The content id is 12 lowercase hex digits: the top
48 bits of an FNV-1a-128 digest over the workspace's compiled inputs, the root `Cargo.toml` and
`Cargo.lock` plus every `*.rs`, `*.toml`, `*.cu`, `*.cuh` and `*.h` under `crates/`, each keyed by
its workspace-relative path and sorted. It does not depend on `read_dir` order or on where the
checkout lives. It is a build identity, not a tamper seal.

- The git sha is still baked, as an extra field (`MEMRA_BUILD_SHA`, `memra_server::BUILD_GIT_SHA`),
  never as the identity. It may read `unknown` there.
- When the workspace tree is unreadable (a vendored or packaged crate), the id falls back to a
  digest of the package name and version: still shaped, never `unknown`, marked
  `build_id_src degraded`. The build prints `cargo:warning=memra build identity DEGRADED: <reason>`,
  and every boot prints `[server] WARNING: build identity is DEGRADED: <reason>`.
- Every boot prints the identity as its first line, and `memra-server --version` (`-V`) prints it
  and exits before any engine, GPU or model work, so an artifact can be identified on a box with no
  GPU.

## Rejected

- **The git sha as the identity.** The field used to be `concat!("memra-", <git rev-parse>)`, with
  any git failure swallowed into the literal `unknown`. Two failures were observed:
  - It degraded silently. A release container compiled as root over a uid-1000 read-only mount,
    git refused the tree with "detected dubious ownership" until a `safe.directory` line landed on
    2026-08-30, and the build baked `unknown`. Binaries from that build answered
    `system_fingerprint: memra-unknown` on every response. The envelope tests asserted only
    `starts_with("memra-")`, which `memra-unknown` passes, so the defect sat inside a tested
    surface.
  - It does not survive a history rewrite. A rewrite changes every commit sha while the tree's
    bytes stay put, so a sha baked into a shipped binary, quoted in a published claim or a research
    receipt, stops naming anything. Receipt from the lane (`lane/real-system-fingerprint`; neither
    the receipt nor this repository's git history names its PR number): `git commit --amend` over
    the lane's tree moved the commit sha from `2bfd89fb74d2` to `07b909892f43` while the content id
    stayed `6371ca8a0af4`.
- **A build timestamp in the value.** Two builds of the same source must give the same
  fingerprint, because a performance figure's published build pin is compared for equality against
  a live binary's fingerprint; a per-rebuild component would churn every pin and make the
  comparison meaningless. Build time is an artifact fact (file name, mtime), not an identity.
- **A flag.** There is no env var that turns the fingerprint off, downgrades it or overrides it. A
  knob here would be a knob for publishing an unverifiable identity.

## Evidence

The gates in `build_identity_tests`:

- the baked value is non-empty, is not `memra-unknown`, contains no `unknown`, matches
  `memra-<version>-<12 lowercase hex>` and names this crate's version;
- the shape checker rejects both the shipped `memra-unknown` and the old `memra-<sha>` form;
- the baked id is re-derived from the working tree in a separate process and must match, which
  proves it is a function of the source and not of the build environment;
- two scans of one tree agree;
- the id differs from `BUILD_GIT_SHA`, so history cannot become the identity by accident;
- the two OpenAI envelope tests assert the full shape instead of the prefix.

## What would change it

A compiled input outside the hashed set (a new source extension under `crates/`, or a build input
outside `crates/` other than the root manifest and lock file) has to join `BUILD_ID_EXTS` or
`BUILD_ID_ROOT_FILES`; otherwise two different binaries can share an id. A consumer that needs
tamper evidence needs a signed digest of the artifact, not this field.
