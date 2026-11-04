# Content-bound release qualification

A successful build is not GPU qualification. Two gates use native evidence:

- **Branch pushes, main included**, need evidence only for the cells the change reaches
  ([Change impact](#change-impact-and-per-cell-evidence)): a correctness battery on a device
  of each affected, push-gated build target for the exact source inputs, plus performance
  evidence when the change is performance-relevant. Docs, tests, research data, CPU tooling
  and changes provably gated to other targets need none.
- **Tags** need the full content-bound record below. `tools/release_qualification.py`
  validates native release evidence for the exact source inputs, built executable bytes,
  roster and kernel-oracle artifacts, launch configuration, hardware/topology, raw cells
  and verdicts.

Both reuse `release-coverage.py`: every greedy K=1..8 must pass once, both required-cell
manifests remain mandatory, and the kernel skip ceiling remains 11. Calibrated argmax
uses the existing gate; no tolerance, predicate or skip budget is widened. Regressions on
targets a push does not gate are caught by the [daily regression](#daily-regression) of
each device and fixed per hardware.

## Change impact and per-cell evidence

`tools/change_impact.py` classifies a diff by what it can affect. A **cell** is a
(build target, scope) pair. The build targets are not a list in the policy file: the
classifier parses the accepted `MEMRA_CUDA_ARCH` set and the compute-capability map from
`crates/memra-engine/build.rs` (and `src/cuda_arch.rs` when present). A target build.rs
accepts with no row in `tools/change-impact.toml` refuses classification, so a new target
lands with its push-gate decision.

Scopes: `kernel` (`.cu`/`.cuh`, FFI shims), `build` (build scripts, the arch table, Cargo
manifests and lock, toolchain), `numeric` (engine, plans, KV format), `serving`
(scheduler, admission, routes), `defaults` (a code line that names a device key such as
`BUILT_CUDA_ARCH`, a compute capability or a card name), `gate` (the battery, its
manifests and roster, the hook and the impact rules) and `perf`. Path rules in
`tools/change-impact.toml` decide each file; the first match wins. They carry the
structure paths alone cannot show: the `*_stub.cu` units compile only for the `90a` and
`89` families, the block-scale MMA units only for `120a` and `100a`, the CUTLASS units
only for the `120a` target, `wgmma_common.cuh` only for `90a`. Rules that need no device:

- docs, tests, research receipts, the qualification metadata namespace;
- CPU-only crates (CLI, SSRF guard) and CPU tooling. Host crates linked into the served
  binaries (tokenizer, sampling, reference) are `numeric`: `memra_sampling::spec_stop` is
  the speculative-decode stop rule;
- diagnostic binaries outside the release battery;
- comment-only edits of `.rs`/`.cu` files: every changed line is a `//` comment. Blank
  lines and a `//` line ending in a backslash count as code;
- a non-crate path counts as a crate input when the CI include census
  (`tools/validation_plan.py`) says a crate reads it, so `docs/FLAGS.md` is `numeric`.

A change inside `model_packs/<pack>/` reaches only that pack. When the device battery runs
none of the pack's models, the gate names it **UNCOVERED**: the pack's own gate is the
evidence, as for the scoped battery in `AGENTS.md`. Kernels and engine files are never
pack-local, even when named for one family: shared paths call into `glm5_*` and others.

The push base of main is its old value. The push base of any other branch is its merge
base with `origin/main`, so a rebased topic answers only for its own commits, not for what
main gained under it.

The rules in force are the push base's, so a change cannot relax its own requirements
(`push_gate`, path rules, package classes). A build target that is new in the head takes
the head's row with `push_gate` forced on. A change to the rules is itself a `gate` cell.

Where paths cannot tell, a commit can narrow the targets with trailers:

```text
Impact-Targets: 121a
Impact-Reason: the GB10 unified-memory arm; the 120a build is byte-identical
```

The narrowing applies only when every commit in the pushed range declares targets (evidence
commits are ignored), and each one needs a reason. It never narrows `build`, `gate` or
unknown files. The gate prints the reasons. The daily regression still runs every
device, so a wrong declaration fails there.

Evidence per target (`tools/cell_evidence.py`):

| cell scope | evidence |
|---|---|
| `kernel`, `build`, `numeric`, `defaults` | device battery: `kernel-check`, calibrated `run-gen` argmax and `run-spec` K=1..8 on the device roster |
| `serving` | the battery plus `tools/serve-smoke.sh` |
| `perf` | `run-gen` decode A/B against the push base, A B B A A B for every roster model |
| `gate` | all three: a changed gate has to run |
| `defaults` on a line naming a card | battery and perf from that card's device class |

A device counts for the target its compute capability maps to: the RTX 5090 and the RTX
PRO 6000 both count for `sm_120a`. A default keyed on a card name (`RTX PRO 6000`,
`RTX 5090`, `GB10`) is the exception: that card has to run it. The perf verdict is REGRESSED when every candidate run
is slower than every base run and the candidate median is at least 2% below the base median.
With three runs per arm, complete separation alone has a one-sided exact probability of 1/20
under no difference, so it failed about one two-model head in ten on identical code. The 2%
floor comes from the run-to-run noise of the banked records and leaves 0.6% of heads on the
RTX 5090 laptop and none on GB10 (`research/perf-verdict-noise-20261106/`). A reviewed, intended regression
is recorded with `--accept-regression '<why>'`. The baseline must be the push base
itself, not an older ancestor or a commit inside the pushed range. A target the base cannot
build has no baseline; its perf record carries absolute rates.

Targets with `push_gate = false` (today `sm_100a`, `sm_90a`, `sm_89`: no owned device in
reach on every push) are reported as DEFERRED to their device's daily run and the pre-tag
battery. Flip the column when a target gets an owned device and a support claim.

Produce and bank a record for the exact final commit:

```sh
python3 tools/cell_evidence.py need --base origin/main --head HEAD     # which cells
python3 tools/cell_evidence.py run --device rtx5090 --commit HEAD \
    --kinds battery,perf --base origin/main --out ~/.cache/memra-cells/run-1
python3 tools/cell_evidence.py bank --out ~/.cache/memra-cells/run-1
git add research/release-qualification/cells && git commit -m "data: bank cell evidence"
python3 tools/cell_evidence.py check --base origin/main --head HEAD    # the hook's decision
```

`run` builds the six battery binaries for the device's target under
`flock ~/.cache/memra-rig-build.lock nice -n 15` with `CARGO_BUILD_JOBS=2`, one codegen unit
and fat LTO, then runs the battery from a detached checkout of the tested commit under the
device's lock (`gpulock` on the 5090, the unit's `sparklock` lease on a GB10, the rig flock
elsewhere). The record binds the tested
commit, the source-input digest, the device class and observation (GPU UUID hashed), the
binary digests and every raw log. The gate re-judges the raw logs; it never trusts a
verdict field. Records live under `research/release-qualification/cells/`, which is
excluded from the source-input digest, so committing one does not change what it
qualifies. A rebase changes the inputs and needs a new record.

## What a record covers

A record qualifies the tag battery, `tools/release-battery.sh`, and that battery covers the
qwen35 class only, and within it two checkpoints (owner, 2026-11-03). The verifier expects
exactly one kernel cell plus two calibrated argmax cells (the eager cell with its batched-prime
arm and the served batch cell, see [Argmax cells](#argmax-cells)) and one greedy K=1..8 spec
cell per `tools/release-roster.tsv` row. The roster has two rows, both
NVFP4+Q5_K MTP GGUFs: Ornith-1.5-35B-A3B (`own`, `qwen35_moe`) and Qwen3.8-27B (`vendor`,
`qwen35`). The kernel cell takes the first roster file present as its model argument; its
required Step manifest holds synthetic kernel and PP-schedule cells, not a Step model run.
kernel-check also runs real-weight oracle cells on the GGUFs in the bound `oracle_directory`
(`MEMRA_KC_MODELS_DIR`). Those are kernel-level evidence only; apart from the required
`DUAL-BATCHED-AUX` cell (Qwen3.5-9B NVFP4 oracle) they may skip within the 11-skip ceiling.

A record is therefore no model evidence for any other qwen35-class checkpoint, for any other
pack (`gemma4_dense`, `gemma4_moe`, `step35`, `glm5_next`, `glm_dsa`, `deepseek_v4`, `hy3` and
the rest) or for a weight format the two roster files do not carry. Those families get their
evidence from their own family gates (the pack's gate manifest, the scoped battery's cells on
each family a change reaches, `step-pro` for Step-3.7) and from each device's
[daily regression](#daily-regression).

A record carries no served-path evidence either: the battery never starts `memra-server`, and
the serving gates (serve-smoke, spec-on-cache-hit, accept-gate, health-fault and the rest) run
only in `tools/local-ci.sh` and in scoped-battery cells.
[Releasing](RELEASING.md#the-gate-on-the-rig-before-tagging) states the same scope for the tag.

## Argmax cells

Each roster model has two calibrated argmax cells. Both run `tools/argmax-margin-gate.sh` with
the same prompt, window, prefill reference (`forward_last` at each window position), per-model
flip budget and margin rule: every prefill/decode argmax flip must sit at a top-2 margin the
config spread covers. They differ in the program on the decode side.

| Cell (`runs.tsv` kind / receipt) | Decode side | Extra requirement |
|---|---|---|
| `argmax` / `argmax-margin` | `decode_step`, the eager program (`decode_step_h`) | the batched serving prime: only `prime=MATCH`, `FLIP-NEARTIE` or `SKIP:tokenwise-served` |
| `argmax-batch` / `argmax-margin-batch` | the served decode program (`--decode batch`): the batched program at B=1, or `decode_step` for an eager-routed model | every window step's device-sampled token equals the host argmax of its row: only `served=MATCH`, or `served=EAGER-ROUTED` |

The second cell exists because the server's batched scheduler decodes every non-speculative row
of a batch-served model through `decode_step_batch_sampled_lean_masked` at every width, B=1
included (the B=1 fast path is an opt-in seam), and that body carries a decode-config
FP-composition gap against `decode_step_h`. The eager cell alone can therefore pass while a
change to the batched body or its epilogue moves served tokens. Speculative rows are not this
body: with `MEMRA_SERVE_SPEC` on (the default) and a model that ships a draft head, spec-eligible
requests decode through the speculative path, which `run-spec` gates. The batched body serves the
rest: the spec-off posture (the shared-prefix serving posture), requests spec admission declines
(constrained sampled, penalized greedy, vision) and models without a draft head. The batch cell
makes the server's own call for one unconstrained greedy row: one token, a planned cache, the
greedy device sample, no grammar mask, lean logits read back from the cache's device park. The
probe installs no `MEMRA_REWRITE_BUNDLE`, so the cell gates the default bundle-less served
program. The battery unsets `MEMRA_SERVE_B1FAST` for it so it runs the default served program. A
gate run that ignored `--decode` prints no `SERVED-SUMMARY` line and cannot pass.

Both cells are required for every roster model: the checker refuses a record that lacks either
or reports anything but the accepted classes. Which program serves a model's decode is the
engine's `HybridModel::served_decode_is_eager`, the predicate the worker's decode scheduling
sites read. For a model it marks eager-routed (the eager-only class: gemma4 E4B, gemma4 with
`MEMRA_GEMMA4_BATCH=0`, plans without a batched decode arm such as glm_dsa MLA or MiniMax-M3, and
HyperConnections trunks with `MEMRA_HYPER_BATCH=0` or under glm5 TP) the scheduler never calls the
batched body. The probe then steps `decode_step`, the program that does serve it, under the same
margin rules and prints `SERVED-ARGMAX: n/a reason=eager-routed`; the gate reports
`served=EAGER-ROUTED`, and the battery and checker accept that value in place of `MATCH`. The
current roster is batch-served, so its cells report `MATCH`.

## Development and publication

To publish unfinished work on a topic branch without evidence, explicitly use:

```sh
MEMRA_RELEASE_QUALIFICATION_MODE=development git push origin HEAD
```

The hook prints **UNQUALIFIED DEVELOPMENT** and records the ref/commit in the local Git
skip ledger. This does not grant qualification, even if model files happen to exist.
Development mode refuses main/master and every tag. A branch push that fast-forwards the
remote ref and changes only Markdown files needs no record: the hook prints **DOCS ONLY**
and claims no qualification. Every other branch push goes through the per-cell gate above.
Tags always need the full record. `MEMRA_SKIP_PERF_CI=1` is retired
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

**The generic battery qualifies one leased RTX PRO 6000 Blackwell at any NVML index.**
`release-battery.sh` resolves CUDA device 0 to its physical UUID
(`tools/resolve-physical-gpu.py`) and keys every headroom and tenant query on it. Capture
resolves the card the same way and refuses unless that UUID equals the leased CUDA-visible
UUID and its NVML index equals the lease's; telemetry samples the same UUID. The record
names the exact card model it ran on (`card_models`, for example Max-Q Workstation Edition
vs Server Edition, whose power limits differ), along with the PCI device ids and the power
limit. Step/multi-card topology qualification stays with its existing gates.

A sealed full record also counts as the device battery for its card's target in the
per-cell push gate. It does not supply serving or perf evidence.

DSv4-Flash two-card cells run on a 2x RTX PRO 6000 pair, and each names its placement:
`Dsv4Gpu::load` has no default, and every load logs `[load] placing <name>`. Serving
qualification runs the served placement first: `tp_ep`, TP/EP with exact attention TP2, which
memra-server loads unless `MEMRA_DSV4_TOPOLOGY=pp`. A PP-2 cell (`pp`) qualifies only that
rollback. The placement of every DSv4 gate binary is in
[TESTING.md](TESTING.md#dsv4-gate-placement).

```sh
memra-gpu-run --gpus "$GPU_UUID" --receipt "$LEASE" -- \
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

## Daily regression

`tools/daily-regression.py` runs once a day on each device that has it installed. It tests
main's head with the device's full cell set (battery, serve-smoke, perf A/B against the last
head that passed on that device), commits the raw record and a `summary.json` to the
device's results branch `regression/<device>` under `research/daily-regression/<device>/`,
and keeps one GitHub issue per device, labelled `daily-regression`: opened or updated with
the failing cells while the device fails, closed with a comment when it passes again. A
head that was already tested is skipped. A run that cannot take the device lock is not a
failure and opens nothing.

Install it per device; the timer is installed disabled:

```sh
python3 tools/daily-regression.py install --device rtx5090 --at 05:30
systemctl --user enable --now memra-daily-regression@rtx5090.timer   # owner's decision
python3 tools/daily-regression.py run --device rtx5090 --no-publish  # a local dry run
```

The unit templates are `tools/daily-regression/memra-daily-regression@.{service,timer}`.
The instance name is the device class (`rtx5090`, `rtxpro6000`, `gb10`), so a Spark or a
PRO 6000 box installs the same templates with its own class. Per-box overrides
(`MEMRA_REPO`, `MEMRA_DAILY_WORK`) go in `~/.config/memra-daily/<class>.env`. One `gb10` row
serves both Sparks, so a Spark also names its own sparklock unit there (`MEMRA_SPARK_UNIT`);
without it the lock refuses and nothing runs. A refused lease is a busy device, not a failure.
The work dir may be shared with manual `cell_evidence.py run` invocations to reuse their build
trees; the daily run caches its binaries in its own `daily-bins/` and prunes only there. The service
runs at `Nice=15`, `CPUWeight=20`, `IOWeight=20`, and the build keeps the rig build policy
above. The timer is not persistent: a laptop asleep at the hour skips that day.
