# Content-bound release qualification

A successful build is not GPU qualification. `tools/release_qualification.py` validates
native release evidence for the exact source inputs, built executable bytes, roster and
kernel-oracle artifacts, launch configuration, hardware/topology, raw cells and verdicts.
It reuses `release-coverage.py`: every greedy K=1..8 must pass once, both required-cell
manifests remain mandatory, and the kernel skip ceiling remains 11. Calibrated argmax
uses the existing gate; no tolerance, predicate or skip budget is widened.

## Development and publication

The pre-push hook defaults to qualification enforcement for every pushed ref. To publish
unfinished work on a topic branch, explicitly use:

```sh
MEMRA_RELEASE_QUALIFICATION_MODE=development git push origin HEAD
```

The hook prints **UNQUALIFIED DEVELOPMENT** and records the ref/commit in the local Git
skip ledger. This does not grant qualification, even if model files happen to exist.
Development mode refuses main/master and every tag. `MEMRA_SKIP_PERF_CI=1` is retired
and refuses rather than bypassing the gate. `step-pro` remains an additional Step model
and topology gate; its older partial source manifest cannot substitute for this record.
That additional checker now requires hashes for every changed crate/build/tool input,
including KV, tier, tokenizer and server. A deleted input requires a `null` tombstone and
absence from the immutable candidate Git tree as well as the checkout. An uncommitted
local deletion cannot cover a file still present in the candidate; a non-Git checkout
cannot prove a deletion. Adding a hash for another file cannot conceal the deletion.

Both real tag workflows validate the committed record before building/publishing. Manual
crates.io recovery also validates it; a package dry run remains explicitly a build check.
The binary release workflow selects a native-qualified build for each OS profile, checks
out its tested commit and refuses packaging unless all six rebuilt ELF digests match.
A non-reproducible rebuild requires qualification of those actual bytes, not a freshness
waiver. One OS build's record cannot qualify the other OS's different executable bytes.

## Source identity and receipt publication

The source snapshot records the tested commit/tree and a SHA-256 over a canonical inventory
of Git blob IDs and modes for **all tracked inputs**, not just engine files. Changes to KV,
tier, tokenizer, server, build scripts, dependency locks, tools, workflows, test fixtures,
or existing research data invalidate the record. New research-only sidecars are reported
as explicit publication additions; they cannot replace or delete an existing input. There is no file-mtime comparison and
no missing-model waiver. A new checkout verifies the same content identities.

`research/release-qualification/` is reserved exclusively for qualification metadata, never
runtime/build inputs. That namespace is excluded from the source-input inventory so that
recording the result does not create a commit-hash cycle. `research/INDEX.md` may only gain
an append after the tested prefix. Every existing tracked input must remain identical, and no new file may enter a runtime,
tool or build namespace without qualification. A publication review must confirm added
research sidecars are evidence, not newly selected external runtime inputs.
The verifier reports both tested and candidate commits, the input digest and every added
research sidecar path: publication
is an audited content-equivalence claim, not qualification of a newly rebuilt binary.

`current.json` inside that namespace pins one or more `record.json` files by SHA-256.
Each record seals the complete evidence manifest; paths must remain within its directory
and cannot be symlinks. Banked records are read from immutable Git objects. A missing tested
commit is unqualified until fetched; CI uses `--fetch-source` to retrieve that exact SHA
from origin before checking its snapshot. Source inventory
and raw evidence remain usable on a clean checkout without requiring local model files,
CUDA or the original absolute build directory. Their absence does not create new evidence.

## Actual build inputs

Native production compares actual tracked file bytes, executable modes and symlink targets
against the expected Git objects before and after build/capture. `git status` is only an
additional diagnostic: assume-unchanged and skip-worktree cannot hide a different compiler
input. File timestamps only detect concurrent changes while hashing; they never grant
freshness or qualification. Ignored/untracked files under crate, tool and Cargo source
roots refuse. Project Python caches are not consumed: the producer and its children use
fresh cache locations.
File and directory symlinks resolve transitively within the tracked Git closure, including
links through research directories. External, missing or cyclic terminals refuse; a research
link cannot introduce an unrecorded compiler input. Valid historical links remain intact.
Every link outside the metadata-only namespace is conservatively a source input. Such links
must not traverse or terminate in `research/release-qualification/` or `research/INDEX.md`,
including an excluded intermediate link between two otherwise bound files. Directory aliases
that expose those metadata subtrees also refuse. Unreferenced metadata-only additions and
index appends remain eligible for publication equivalence; they cannot change linked compiler
inputs under an unchanged source identity.

The controlled v2 build uses a fresh, detached Git-defined source checkout outside the
caller checkout and its own config-free Cargo home. The actual staged source is checked
before and after compilation, so ignored caller files are never compiler inputs. It may
reuse downloaded archive and index caches, but never inherited configuration, credentials
or extracted source trees.
Compiler-wrapper/target/rustflag environment controls are removed, cargo/rustc are resolved
from the pinned toolchain and their executable hashes are recorded along with nvcc. Ancestor
Cargo configuration is refused; the only admitted project Cargo configuration is the tracked
`[build] jobs` setting. Environment forcing, alternate targets and wrapper config refuse
rather than overwrite the recorded native/stub/architecture claims.
The `controlled-cargo-v2` recipe admits only basic process/network environment settings,
then adds the pinned build settings. Ambient nvcc prepend/append flags, host-compiler
selection, and compiler include/library search overrides are excluded. Earlier controlled
recipe versions cannot be relabelled as passing this boundary.

The generic capture contract explicitly supports **single-file GGUF inputs**. It parses
`split.count` and refuses multi-file models before any GPU operation, even if an entry shard
was renamed. A complete loaded-shard closure is required before split qualification can be
added. This is a qualification-tool limit, not a statement about engine format support.
The separate Step gate retains its full artifact and topology checks.
Oracle directories normalize once (absolute, relative or symlinked directory), while
named file aliases remain under that directory for the kernel resolver and verifier.
Capture never renames or substitutes a model: the coordinator supplies the actual named
weight oracles, whose complete file bytes are hashed before and after the run.

## Produce a record

Use a clean, isolated checkout of the final committed candidate. Do not include credentials
in command arguments or receipt paths. Numeric launch values are recorded as SHA-256 values,
not raw environment dumps. Build is GPU-free and creates a fresh target/source tree; `$BUILD` must be outside the
input checkout:

```sh
python3 tools/qualify-release.py build --expected-head "$COMMIT" --out "$BUILD" \
  --nvcc /usr/local/cuda-13.1/bin/nvcc --jobs 8
```

The coordinator supplies the existing non-serving rig and physical-card wrapper. Capture
requires the live wrapper/child ancestry, canonical exclusive per-UUID FLOCK and exact
CUDA-visible UUID. It never acquires locks, rents a rig or evicts another process.

**Current generic battery scope is physical GPU0 on RTX PRO 6000 Blackwell.** The battery's
headroom query still uses NVML `-i 0`. Capture/validation compare that observed UUID with
CUDA visibility and the lease, refusing any disagreement. Other-card selection is the
separate #264 follow-up, not support inferred from a CUDA ordinal. Step/multi-card topology
qualification stays with its existing gates.

```sh
memra-gpu-run --gpus "$GPU0_UUID" --receipt "$LEASE" -- \
  python3 tools/qualify-release.py capture --expected-head "$COMMIT" --build "$BUILD" --out "$RUN" \
  --oracles /data/models/kernel-oracles
```

Capture stages the six hash-checked build outputs in this isolated checkout's ignored
`target/release`, hashes the roster and all named GGUF oracle files before/after execution,
and stores each actual producer output before parsing, with 250 ms telemetry for the leased UUID. The fixed release battery receives
no ambient `MEMRA_*` overrides; its existing greedy/full-depth sanitation still applies.
The record binds the actual launch environment and hardware observations. It does not
replace the loaded numerical-program rewrite admission policy in #542.

Capture exits **executed, not sealed**. After the wrapper has exited and finalized its
cleanup record, seal the evidence (CPU/file checks only):

```sh
python3 tools/qualify-release.py seal --out "$RUN" --lease "$LEASE/lease.json" \
  --oracles /data/models/kernel-oracles
python3 tools/qualify-release.py bank --out "$RUN" --name candidate-pro-ubuntu-24-04
# For another independently qualified build profile at identical inputs, use --append.
git add research/release-qualification
# Commit the publication, then verify its immutable contents:
python3 tools/release_qualification.py verify --head HEAD
```

Seal refuses failed/missing cells, changed files, interrupted/timed-out leases, lingering
compute and stale identities. Preserve failed captures; do not overwrite or relabel them.
Bank copies only manifested evidence, not executables or unrelated files beside it. Keep
the tested executable archive off the ephemeral rig; its six digests remain in the record.

These records are auditable native-run evidence, not signed third-party attestations. A
reviewer must verify provenance; hashing cannot turn invented observations into execution.
CPU fixtures in `tools/test_release_qualification.py` are temporary fabricated records for
checker tests only. They do not qualify a model, binary, driver, GPU or release. Full HTTP,
model support, performance and other-hardware claims require their own applicable gates.
