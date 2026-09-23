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

The controlled v3 build retains a detached full Git checkout as provenance, then materializes
a separate compiler view from exactly the fingerprinted files. Publication payloads and
`research/INDEX.md` are absent from that view. It carries only a verified Git HEAD/config
descriptor with empty object/ref directories, so build metadata can name the pinned commit
without exposing repository objects or an alternate path to excluded data. Source bytes,
executable modes, the complete view inventory and that identity descriptor are checked before
and after compilation. Metadata additions are not new binary qualification: publication
retains the tested commit/view and its qualified executable hashes.

Linux bubblewrap runs compilation and build scripts in a read-only `/source` view with a
private root, `/proc`, `/dev` and `/tmp`. The full provenance/caller checkouts and host home
directories are not available there; no GPU device or network is exposed. System libraries,
the pinned toolchain and CUDA toolkit are read-only inputs. The private Cargo cache and target
directories are writable. Dependencies are fetched with `--locked` before compilation, which
uses `--offline`; dependency fetching does not execute workspace build scripts. This enforces
the declared filesystem boundary, rather than assuming a direct include or script never reads
publication data. It is not a claim to have audited every host library or arbitrary compiler.

On Ubuntu with restricted unprivileged user namespaces, installing bubblewrap alone is
insufficient: AppArmor can deny loopback setup before the compiler starts. The disposable
GitHub-hosted jobs use `tools/install-release-sandbox-ci.sh` to install a dedicated,
root-owned bwrap copy and grant that exact executable the `userns` permission. The host
restriction remains enabled; the producer retains its private network/filesystem namespaces,
read-only source and dropped capabilities. Linux controls verify an actual successful
sandbox launch, absence of compiler capabilities, and isolation from a live host loopback
listener, in addition to the source/provenance checks. Other build hosts must supply a
working, appropriately permitted bwrap executable; there is no unisolated retry.

The build has its own config-free Cargo home. It may
reuse downloaded archive and index caches, but never inherited configuration, credentials
or extracted source trees.
Compiler-wrapper/target/rustflag environment controls are removed, cargo/rustc are resolved
from the pinned toolchain and their executable hashes are recorded along with nvcc. Ancestor
Cargo configuration is refused; the only admitted project Cargo configuration is the tracked
`[build] jobs` setting. Environment forcing, alternate targets and wrapper config refuse
rather than overwrite the recorded native/stub/architecture claims.
The `controlled-cargo-v3` recipe admits basic process/network settings for dependency fetch,
then sets the compiler namespace's fixed paths and build settings. Ambient nvcc prepend/append
flags, host-compiler selection, compiler include/library search overrides and network credentials
are excluded from compilation. The record binds the sandbox executable, compiler environment,
and verified input-view identities. Earlier recipes cannot be relabelled as passing this boundary.

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
  --nvcc /usr/local/cuda-13.1/bin/nvcc --bwrap /path/to/bwrap --jobs 8
```

Use the distribution's bubblewrap package (an owner-local package extraction is supported).
Missing or unavailable Linux namespace support refuses the build; there is no unisolated
fallback. The release workflow uses this same controlled builder before its ELF identity check.

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
  --oracles /data/models/kernel-oracles --serving "$SERVING_STAGE"
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
checker tests only. They do not qualify a model, binary, driver, GPU or release. The required serving stage below covers only its reviewed model/route programs.
Performance, additional model support and other hardware still require their applicable gates.


## Required serving stage and full release v2

Full publication requires `memra-release-qualification-v2`: the unchanged generic
kernel/argmax/K=1..8 evidence **and** every required serving cell. A generic v1 raw
record remains historical generic-only evidence. It is not rewritten or promoted;
`validate_historical_record` can inspect it, while push/tag/release verification
accepts only full v2. Topic development push remains explicitly UNQUALIFIED as
shown above. There is no protected-ref or missing-serving waiver.

The source-owned input is `tools/serving-release.programs.json`, schema
`memra-serving-policy-v1`. It must be reviewed and committed before building.
There is deliberately no generated default policy: absence refuses qualification.
Each scope names a roster ID, `{id,model,route,profile}`, all consumed model/draft/
metadata artifact paths and byte/SHA-256 identities, the ordered physical device
classes `{name,compute_cap}`, and all eleven cells. Every roster model must appear;
a run cannot select fewer models, routes, requests or scenarios. Policy changes are
build-input changes. The source-owned scenario manifest remains
`tools/serving-release.cells.json`.

Each cell contains `scenario`, its exact adapter `program`, and a `server` object
with explicit `env`, `timeouts` (`startup,overall,drain,kill`) and `http` limits.
The program fixes payloads, ordering, token bounds and independent answer oracles.
It omits only `server_identity`, `endpoint`, `identities` and `client_trace_key`:
these are supplied by the owned process and fixed collector rules. The policy's
`MEMRA_MODELS` must map its sole served alias to the roster artifact. The collector
supplies the loopback address and ordered leased UUIDs; no ambient environment is
merged into the server. Additional artifacts and numerical environment values
must be explicit in the reviewed policy. This does not declare an otherwise
unsupported model, topology or execution route supported.

Use the already validated v3 build. On each reviewed physical set, the coordinator
runs one **complete** scope through the existing per-card wrapper:

```sh
memra-gpu-run --gpus "$SCOPE_UUIDS" --receipt "$SCOPE_LEASE" -- \
  python3 -B tools/serving-run.py capture --expected-head "$COMMIT" \
  --build "$BUILD" --scope "$SCOPE_ID" --out "$SCOPE_RUN"
```

The runner does not build, rent, acquire GPU locks or evict processes. It verifies
live wrapper ancestry and selected exclusive FLOCK rows, source/controller files,
ELFs and artifact bytes before/after. It uses the existing group, overload,
phase-driven cancellation, owned drain and worker-respawn collectors. Failed
captures and unattempted cells remain in the output; inference is never retried
until a desired result appears. Startup probes and all raw capture bytes are kept.
A completed serving scope does not seal itself while its lease is still running.

After **every** scope's wrapper has closed, repeat paired `--scope-run` and
`--lease` options for the full required set:

```sh
python3 -B tools/serving-run.py seal --expected-head "$COMMIT" --build "$BUILD" \
  --scope-run "$SCOPE_RUN_A" --lease "$SCOPE_LEASE_A/lease.json" \
  --scope-run "$SCOPE_RUN_B" --lease "$SCOPE_LEASE_B/lease.json" --out "$SERVING_STAGE"
python3 -B tools/serving-run.py verify --expected-head "$COMMIT" --out "$SERVING_STAGE"
```

The serving stage's physical-card validation uses the route's reviewed set;
it does **not** apply the generic battery's GPU0 rule to every model. Topology,
actual UUID order, worker/process birth and completed lease identities stay bound.
Generic numerical capture remains a separate component on its existing GPU0/PRO
profile. Direct full battery invocation now requires
`tools/release-battery.sh --serving-record "$SERVING_STAGE"`; `--generic-only`
prints its restricted scope and cannot authorize publication. The controlled
`qualify-release.py capture` invokes those generic components; full `seal` requires
`--serving "$SERVING_STAGE"` and binds both into v2. Existing tag workflows and
pre-push hooks use that same strict verifier.

Raw replay validates the complete capture and policy denominators, exact sent
payloads and launch, controller/build/model identities, lifecycle and physical
lease receipts. It reruns each existing scenario predicate instead of trusting
stored `passed` booleans. Full v2 also rejects different generic/serving bytes for
the same model path. All artifact hashing occurs at capture/verification
boundaries; no server/token path is changed.

These are source and evidence checks, not signed attestations or a latency/SLO
claim. Before native qualification, the actual Linux collector must pass on the
selected source and the complete policy must execute under fresh leases.
Component CPU fixtures, synthetic epochs and compiler-only checks remain their
recorded scope. A host-return event or retirement site does not prove GPU
quiescence, and unobserved generation intervals are not described as unchanged.
