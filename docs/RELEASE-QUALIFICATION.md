# Release qualification

The evidence rules for pushes and tags: which GPU cells a change needs, what a tag's sealed
record covers, and how to produce, seal and bank one. For maintainers producing evidence. The
release procedure that consumes these records is [Releasing](RELEASING.md); the gate catalog and
how to choose what to run is [Testing](TESTING.md).

A successful build is not GPU qualification. Two gates use native evidence:

- **Branch pushes, main included**, need evidence only for the cells the change reaches
  ([Change impact](#change-impact-and-per-cell-evidence)): a correctness battery on a device of
  each affected, push-gated build target for the exact source inputs, plus performance evidence
  when the change is performance-relevant. Docs, tests, research data, CPU tooling and changes
  provably gated to other targets need none.
- **Tags** need the full content-bound record below. `tools/release_qualification.py verify`
  checks the exact source inputs, the built executable bytes, the roster and kernel-oracle
  artifacts, the launch configuration, the hardware and topology, and the raw cells and verdicts.

Both reuse `tools/release-coverage.py`: every greedy K=1..8 must pass once, after one
`STEP-IDENTITY` line ending `-> MATCH` or its named `n/a`
([Testing](TESTING.md#the-oracle-a-speculative-route-is-exact-against-memra1146),
[memra#1146](https://github.com/avifenesh/memra/issues/1146)). Both required-cell manifests stay
mandatory and the kernel skip ceiling is 11. Calibrated argmax uses the existing gate; no
tolerance, predicate or skip budget is widened. Regressions on targets a push does not gate are
caught by each device's [daily regression](#daily-regression) and fixed per hardware.

## Change impact and per-cell evidence

`tools/change_impact.py` classifies a diff by what it can affect. A **cell** is a
(build target, scope) pair. The build targets are not a list in the policy file: the classifier
parses the accepted `MEMRA_CUDA_ARCH` set and the compute-capability map from
`crates/memra-engine/build.rs` (and `src/cuda_arch.rs`). A target build.rs accepts with no row in
`tools/change-impact.toml` refuses classification, so a new target lands with its push-gate
decision.

Scopes: `kernel` (`.cu`/`.cuh`, FFI shims), `build` (build scripts, the arch table, Cargo
manifests and lock, toolchain), `numeric` (engine, plans, KV format), `serving` (scheduler,
admission, routes), `defaults` (a code line that names a device key such as `BUILT_CUDA_ARCH`, a
compute capability or a card name), `gate` (the battery, its manifests and roster, the hook and
the impact rules) and `perf`. Path rules in `tools/change-impact.toml` decide each file; the first
match wins. They carry the structure paths alone cannot show: the `*_stub.cu` units compile only
for the `90a` and `89` families, the block-scale MMA units only for `120a` and `100a`, the CUTLASS
units only for the `120a` target, `wgmma_common.cuh` only for `90a`. Rules that need no device:

- docs, tests, research receipts, the qualification metadata namespace;
- CPU-only crates (CLI, SSRF guard) and CPU tooling. Host crates linked into the served binaries
  (tokenizer, sampling, reference) are `numeric`: `memra_sampling::spec_stop` is the
  speculative-decode stop rule;
- diagnostic binaries outside the release battery;
- comment-only edits of `.rs`/`.cu` files: every changed line is a `//` comment. Blank lines and
  a `//` line ending in a backslash count as code;
- a non-crate path counts as a crate input when the CI include census
  (`tools/validation_plan.py`) says a crate reads it, so `docs/FLAGS.md` is `numeric`.

A change inside `model_packs/<pack>/` reaches only that pack. When the device battery runs none
of the pack's models, the gate names it `UNCOVERED`: the pack's own gate is the evidence, as
for the scoped battery in [AGENTS.md](../AGENTS.md#match-validation-to-the-changed-behavior).
Kernels and engine files are never pack-local, even when named for one family: shared paths call
into `glm5_*` and others.

The push base of main is its old value. The push base of any other branch is its merge base with
`origin/main`, so a rebased topic answers only for its own commits. The rules in force are the
push base's, so a change cannot relax its own requirements (`push_gate`, path rules, package
classes). A build target that is new in the head takes the head's row with `push_gate` forced on.
A change to the rules is itself a `gate` cell.

Where paths cannot tell, a commit can narrow the targets with trailers:

```text
Impact-Targets: 121a
Impact-Reason: the GB10 unified-memory arm; the 120a build is byte-identical
```

The narrowing applies only when every commit in the pushed range declares targets (evidence
commits are ignored), and each one needs a reason. It never narrows `build`, `gate` or unknown
files. The gate prints the reasons. The daily regression still runs every device, so a wrong
declaration fails there.

Evidence per target (`tools/cell_evidence.py`):

| cell scope | evidence |
|---|---|
| `kernel`, `build`, `numeric`, `defaults` | device battery: `kernel-check`, calibrated `run-gen` argmax (eager and served-batch) and `run-spec` K=1..8 on the device roster |
| `serving` | the battery plus `tools/serve-smoke.sh` |
| `perf` | `run-gen` decode A/B against the push base, A B B A A B for every roster model |
| `gate` | all three: a changed gate has to run |
| `defaults` on a line naming a card | battery and perf from that card's device class |

A device counts for the target its compute capability maps to: the RTX 5090 and the RTX PRO 6000
both count for `sm_120a`. A default keyed on a card name (`RTX PRO 6000`, `RTX 5090`, `GB10`) is
the exception: that card has to run it.

The perf verdict is REGRESSED when every candidate run is slower than every base run and the
candidate median is at least 2% below the base median. Separation alone failed about one
two-model head in ten on identical code; the 2% floor comes from the banked records' run-to-run
noise ([research/perf-verdict-noise-20261106/](../research/perf-verdict-noise-20261106/)). Record
a reviewed, intended regression with `--accept-regression '<why>'`. The baseline is the push base
itself, not an older ancestor or a commit inside the pushed range. A target the base cannot build
has no baseline; its perf record carries absolute rates.

Targets with `push_gate = false` (today `sm_100a`, `sm_90a`, `sm_89`: no device in the push
gate's reach) are reported as DEFERRED to their device's daily run; the tag battery builds `sm_120a`
only, so it does not cover them. Flip
the column when the target gets a device in reach and a support claim.

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
`flock ~/.cache/memra-rig-build.lock nice -n 15` with `CARGO_BUILD_JOBS=2`, one codegen unit and
fat LTO. It runs the battery from a sparse, detached checkout of the tested commit under the
device's lock (`gpulock` on the 5090, the unit's `sparklock` lease on a GB10, the rig flock
elsewhere); the checkout leaves out research receipts nothing reads
(`cell_evidence.research_kept`). The record binds the tested commit, the source-input digest, the
device class and observation (GPU UUID hashed), the binary digests and every raw log. The gate
re-judges the raw logs; it never trusts a verdict field. Records live under
`research/release-qualification/cells/`, which the source-input digest excludes.

A rebase changes the inputs and needs a new record, with one exception
(`cell_evidence.research_drift`): when every input that differs between the record's tested
commit and the candidate is a research or docs path that no crate and no gate tool reads, the
record still proves its cells and `check` names the drift (`.gitattributes` never qualifies). The
same rule admits a perf A/B whose base differs from the push base only that way. The tested
commit must be present to compare. A tag's full record keeps the stricter source check in the
[verifier contract](#verifier-contract).

## What a record covers

A record qualifies the tag battery, `tools/release-battery.sh`. That battery covers the qwen35
class only, and within it two checkpoints (owner, 2026-11-03). The verifier expects exactly one
kernel cell plus, per `tools/release-roster.tsv` row, two calibrated argmax cells (the eager cell
with its batched-prime arm and the served-batch cell, see [Argmax cells](#argmax-cells)) and one
greedy K=1..8 spec cell. The roster has two rows, both NVFP4+Q5_K MTP GGUFs:

| class | id | pack | battery rule |
|---|---|---|---|
| `own` | Ornith-1.5-35B-A3B | `qwen35_moe` | required: a missing file refuses the release |
| `vendor` | Qwen3.8-27B | `qwen35` | required for a tag (the record admits only the plain battery command); `--allow-missing-vendor` is for development runs |

The kernel cell runs `kernel-check` once, with the first roster file present as its model
argument, and requires the cells in `tools/kernel-check-27b.cells` and
`tools/kernel-check-step35.cells`. The Step manifest holds synthetic kernel and PP-schedule cells,
not a Step model run. kernel-check also runs real-weight oracle cells on the GGUFs in the bound
`oracle_directory` (`MEMRA_KC_MODELS_DIR`). Those are kernel-level evidence only; apart from the
required `DUAL-BATCHED-AUX` cell (Qwen3.5-9B NVFP4 oracle) they may skip within the 11-skip
ceiling.

The battery fixes its own environment:

- kernel-check runs with `MEMRA_KC_FAST` and `MEMRA_KC_ONLY` cleared. The skip ceiling is 11, as
  in `tools/local-ci.sh`, and `MEMRA_CI_KC_SKIP_BUDGET` cannot raise it. A required cell cannot be
  skipped. The receipt reports executed and required counts and every effective skip.
- `run-spec` runs with `MEMRA_SPEC_K`, `MEMRA_PROMPT_DIR` and `MEMRA_GEN_ONLY` cleared and
  `MEMRA_SPEC_TEMP=0 MEMRA_NGEN=32`. It needs exactly one greedy PASS for each K=1..8, the final
  success summary and exit 0. Sampled reproducibility, duplicate depths and a success marker
  without the per-depth results are refused (`tools/test_release_battery_coverage.py`).
- A missing probe binary is a refusal, never a skip: the battery requires
  `argmax-margin-probe` up front and renders any `SKIP` from the gate as a refusal.

A record is therefore no model evidence for any other qwen35-class checkpoint, for any other pack
(`gemma4_dense`, `gemma4_moe`, `step35`, `glm5_next`, `glm_dsa`, `deepseek_v4`, `hy3` and the
rest) or for a weight format the two roster files do not carry (block FP8, Q4_0 and IQ4_XS among
them). Those families get their evidence from their own family gates (the pack's gate manifest,
the scoped battery's cells on each family a change reaches, `step-pro` for Step-3.7 on RTX PRO
6000, [Testing](TESTING.md#target-aware-release-evidence)) and from each device's
[daily regression](#daily-regression).

A record carries no served-path evidence either: the battery never starts `memra-server`, and the
serving gates (serve-smoke, spec-on-cache-hit, accept-gate, health-fault and the rest) run only in
`tools/local-ci.sh` and in scoped-battery cells.

## Argmax cells

Each roster model has two calibrated argmax cells. Both run `tools/argmax-margin-gate.sh` with
`--require-floor` on `research/e2e/prompts/board-2048.txt`, with the same window, prefill
reference (`forward_last` at each window position), per-model flip budget and rules. A cell passes
only when both checks pass:

- **The floor rule** ([memra#1199](https://github.com/avifenesh/memra/issues/1199),
  [decision](decisions/ARGMAX-GATE-GAP-SHIFT-FLOOR.md)) judges the candidate's prefill against its
  own decode. A flip whose gap shift (the sum of the two programs' top-2 margins) is inside the
  cell's calibrated floor is a near tie counted against the budget. A flip or an agreeing row
  above the floor fails. A row the floor row lists as known is named and fails only when the
  candidate's gap shift there exceeds main's recorded one. Each cell has its own floor row in
  `tools/argmax-margin-floors.tsv`, keyed by model, prompt and the sha256 of its bytes, decode
  program and the exact device name.
- **The stored rows** ([memra#1204](https://github.com/avifenesh/memra/issues/1204),
  [decision](decisions/ARGMAX-GATE-STORED-ROWS.md)) judge the candidate against main: its window
  table must equal main's stored table in `tools/argmax-margin-rows/` byte for byte, keyed by
  model and its sha256, prompt and its sha256, decode program, exact device name, window and
  expert placement. A passing cell prints exactly one `STORED-ROWS: identical` line, which the
  checkers require.

A cell with no floor row or no stored table for the device it runs on is refused (`REFUSED`, or
`NO-STORED-ROWS` for the table) on a device of a maintained class (`rtx5090`, `rtxpro6000`,
`gb10`) or an unknown name, never passed as uncalibrated. On a device of a class nobody maintains
(H100, B200) it is `NOT-JUDGED`, which neither passes nor refuses the battery. A floor-rule
failure is a FAIL whether or not main's rows are stored. Calibrating floors, storing main's rows
and recalibrating after an accepted numeric change are in
[Testing](TESTING.md#argmax-margin-gate-floors-stored-rows-recalibration).

The two cells differ in the program on the decode side:

| Cell (`runs.tsv` kind / receipt) | Decode side | Extra requirement |
|---|---|---|
| `argmax` / `argmax-margin` | `decode_step`, the eager program (`decode_step_h`) | the batched serving prime: only `prime=MATCH`, `FLIP-NEARTIE` or `SKIP:tokenwise-served` |
| `argmax-batch` / `argmax-margin-batch` | the served decode program (`--decode batch`): the batched program at B=1, or `decode_step` for an eager-routed model | every window step's device-sampled token equals the host argmax of its row: only `served=MATCH`, or `served=EAGER-ROUTED` |

Why two cells: the server's batched scheduler decodes every non-speculative row of a batch-served
model through `decode_step_batch_sampled_lean_masked` at every width, B=1 included, and that body
carries a decode-config FP-composition gap against `decode_step_h`. The eager cell alone can pass
while a change to the batched body moves served tokens. (Spec-eligible requests decode through the
speculative path, which `run-spec` gates.) The batch cell makes the server's own call for one
unconstrained greedy row, with no `MEMRA_REWRITE_BUNDLE` and `MEMRA_SERVE_B1FAST` unset, so it
gates the default served program. A gate run that ignored `--decode` prints no `SERVED-SUMMARY`
line and cannot pass.

Both cells are required for every roster model. For a model that
`HybridModel::served_decode_is_eager` marks eager-routed (gemma4 E4B, plans without a batched
decode arm such as glm_dsa MLA or MiniMax-M3, and the other cases that predicate lists), the probe
steps `decode_step`, the program that serves it, prints `SERVED-ARGMAX: n/a reason=eager-routed`,
and the gate reports `served=EAGER-ROUTED`, which the checker accepts in place of `MATCH`. The
current roster is batch-served, so its cells report `MATCH`.

## Development pushes

To publish unfinished work on a topic branch without evidence, say so explicitly:

```sh
MEMRA_RELEASE_QUALIFICATION_MODE=development git push origin HEAD
```

The hook prints `UNQUALIFIED DEVELOPMENT` and records the ref and commit in the local Git skip
ledger. This grants no qualification, even if model files happen to exist. Development mode
refuses main, master and every tag.

A branch push that fast-forwards the remote ref and changes only Markdown files outside `crates/`
needs no record: the hook prints `DOCS ONLY` and claims no qualification. A Markdown file that a
crate names (`docs/FLAGS.md`, `docs/EXECUTION-SURFACES.md`) does not count as docs. Every other
branch push goes through the per-cell gate above. Tags always need the full record.
`MEMRA_SKIP_PERF_CI=1` is retired and refuses rather than bypassing the gate. `step-pro` remains
an additional Step model and topology gate ([Testing](TESTING.md#target-aware-release-evidence));
it cannot substitute for this record.

Both tag workflows validate the committed record before building or publishing. Manual crates.io
recovery also validates it; a package dry run stays a build check. The binary release workflow
selects a native-qualified build for each OS profile, checks out its tested commit and refuses
packaging unless all six rebuilt ELF digests match. A non-reproducible rebuild needs
qualification of those actual bytes, not a freshness waiver. One OS build's record cannot qualify
the other OS's different executable bytes.

## Produce a record

Use a clean, isolated checkout of the final committed candidate. Do not put credentials in
command arguments or receipt paths. Numeric launch values are recorded as SHA-256 values, not raw
environment dumps.

### Build

The build is GPU-free and creates a fresh target and source tree; `$BUILD` must be outside the
input checkout:

```sh
python3 tools/qualify-release.py build --expected-head "$COMMIT" --out "$BUILD" \
  --nvcc /usr/local/cuda-13.1/bin/nvcc --bwrap /path/to/bwrap --jobs 8
```

Use the distribution's bubblewrap package (an extracted package also works). Missing or
unavailable Linux namespace support refuses the build; there is no unisolated fallback. The
release workflow uses the same controlled builder before its ELF identity check.

One record per release OS profile, built where that profile's bytes come from: `release.yml`
rebuilds on GitHub's `ubuntu-22.04` and `ubuntu-24.04` runners and refuses packaging unless each
rebuild's six ELF digests equal a qualified build of the same profile (`build.json` names it from
`/etc/os-release`). So a tag needs a sealed record for each profile, the second banked with
`--append`. A build host on another release builds each profile in a stock image of that Ubuntu
with the runner's toolchain: `build-essential`, `git`, `python3`, `bubblewrap`, the CUDA 13.1
packages from `tools/install-cuda-ci.sh` and the pinned Rust, installed with rustup's minimal
profile. Run it as root there with `HOME`, `USER` and `LOGNAME` set to the runner's values (the
sandbox passes those three to the compiler namespace).

### Byte check

Check byte reproducibility before any capture. Dispatch `gpu-ci.yml` with
`build_reuse_only=true` on the candidate and compare its `build.json` digests (the
`build-feedback-diagnostics-<run>` artifact) with the host's `ubuntu-24.04` build. Then give capture the result: `qualify-release.py capture
--reference-build <the run's build.json>` refuses unless the build's six digests equal the
reference's (same schema, source inputs and platform profile; not the build's own file). No
workflow builds `ubuntu-22.04` before a tag, so that profile rests on the same host, image recipe
and toolchain as the checked `ubuntu-24.04` build; `release.yml` is its first byte check.

Two causes of differing bytes are closed in the build: `crates/memra-engine/build.rs` zeroes the
nvcc pid in object FILE symbols (`normalize_nvcc_temp_symbols`), and the build refuses a toolchain
that carries the `rust-src` component, which changes the standard library's panic location
strings. Do not copy a toolchain from a development machine. The incidents behind both are in the
[history](history/RELEASE-QUALIFICATION.md#produce-a-record) and in
[research/release-v0.140.0-20261110/](../research/release-v0.140.0-20261110/README.md).

### Lease

The coordinator supplies a card with no other work on it, leased through
`tools/memra-gpu-run`. Capture requires the live wrapper and child ancestry, the canonical
exclusive per-UUID flock and the exact CUDA-visible UUID; it never takes locks or evicts another
process itself. The wrapper takes `/tmp/memra-gpu-locks/<UUID>.lock` and, with `--box-lock`, the
box's campaign lock (`/tmp/memra-gpu.lock` on a 2x RTX PRO 6000 box), so a capture and a campaign
exclude each other; `validate_lease` does not check this, so pass `--box-lock`. It refuses a card
that already runs compute, runs the command in its own process group holding the lock
descriptors, stops the whole group on a timeout or signal, and finishes the lease only when the
group is empty. Seal refuses a lease that is still `running` or whose group left processes
(`process_group_lingered`). Capture refuses a lease of more than one card. The CPU contract is
`tools/test_memra_gpu_run.py`; the full semantics are in the
[history](history/RELEASE-QUALIFICATION.md#produce-a-record).

The generic battery qualifies one leased RTX PRO 6000 Blackwell at any NVML index.
`release-battery.sh` resolves CUDA device 0 to its physical UUID
(`tools/resolve-physical-gpu.py`) and keys every headroom and tenant query on it. Capture resolves
the card the same way and refuses unless that UUID equals the leased CUDA-visible UUID and its
NVML index equals the lease's; telemetry samples the same UUID. Capture refuses any card that is
not an RTX PRO 6000 Blackwell with compute capability 12.0. The record names the exact card model
(`card_models`, for example Max-Q Workstation Edition vs Server Edition, whose power limits
differ), the PCI device ids and the power limit. Step and multi-card topology qualification stay
with their own gates ([DSv4 placement](TESTING.md#dsv4-gate-placement)).

A sealed full record also counts as the device battery for its card's target in the per-cell push
gate. It does not supply serving or perf evidence.

### Capture

```sh
tools/memra-gpu-run --gpus "$GPU_UUID" --receipt "$LEASE" --box-lock /tmp/memra-gpu.lock -- \
  python3 tools/qualify-release.py capture --expected-head "$COMMIT" --build "$BUILD" --out "$RUN" \
  --oracles "$ORACLES" --reference-build "$GITHUB_BUILD_JSON"
```

Capture stages the six hash-checked build outputs in this checkout's ignored `target/release`,
hashes the roster and every named GGUF oracle file before and after execution, and stores each
producer output before parsing, with 250 ms telemetry for the leased UUID. The release battery
receives no ambient `MEMRA_*` overrides; its own greedy and full-depth sanitation still applies.
The record binds the actual launch environment and hardware observations. It does not replace the
loaded numerical-program rewrite admission policy
([memra#542](https://github.com/avifenesh/memra/issues/542)).

### Seal, bank and verify

Capture exits executed, not sealed. After the wrapper has exited and
finalized its cleanup record, seal the evidence (CPU and file checks only):

```sh
python3 tools/qualify-release.py seal --out "$RUN" --lease "$LEASE/lease.json" --oracles "$ORACLES"
python3 tools/qualify-release.py bank --out "$RUN" --name candidate-pro-ubuntu-24-04
# Bank another independently qualified build profile at identical inputs with --append.
git add research/release-qualification
# Commit the publication, then verify its immutable contents:
python3 tools/release_qualification.py verify --head HEAD
```

Seal refuses failed or missing cells, changed files, interrupted or timed-out leases, lingering
compute and stale identities. Keep failed captures; do not overwrite or relabel them. Bank copies
only manifested evidence, not executables or unrelated files. Keep the tested executable archive
off the ephemeral machine; its six digests stay in the record.

These records are auditable native-run evidence, not signed third-party attestations. A reviewer
must verify provenance; hashing cannot turn invented observations into execution. The CPU fixtures
in `tools/test_release_qualification.py` are fabricated records for checker tests only. They
qualify no model, binary, driver, GPU or release. HTTP, model support, performance and
other-hardware claims need their own gates.

## Daily regression

`tools/daily-regression.py` runs once a day on each device that has it installed. It tests main's
head with the device's full cell set (battery, serve-smoke, perf A/B against the last head that
passed on that device) and commits the raw record and a `summary.json` to the device's results
branch `regression/<device>` under `research/daily-regression/<device>/`. It keeps one GitHub
issue per device, labelled `daily-regression`: opened or updated with the failing cells while the
device fails, closed with a comment when it passes again. A head that was already tested is
skipped. A run that cannot take the device lock is not a failure and opens nothing.

Install it per device; the timer is installed disabled:

```sh
python3 tools/daily-regression.py install --device rtx5090 --at 05:30
systemctl --user enable --now memra-daily-regression@rtx5090.timer   # owner's decision
python3 tools/daily-regression.py run --device rtx5090 --no-publish  # a local dry run
```

The unit templates are `tools/daily-regression/memra-daily-regression@.{service,timer}`. The
instance name is the device class (`rtx5090`, `rtxpro6000`, `gb10`), so a Spark or a PRO 6000 box
installs the same templates with its own class. Per-box overrides (`MEMRA_REPO`,
`MEMRA_DAILY_WORK`) go in `~/.config/memra-daily/<class>.env`. One `gb10` row can serve several
boxes, so each Spark also names its own sparklock unit there (`MEMRA_SPARK_UNIT`); without it the
lock refuses and nothing runs. A refused lease is a busy device, not a failure. The work dir may be
shared with manual `cell_evidence.py run` invocations to reuse their build trees; the daily run
caches its binaries in its own `daily-bins/` and prunes only there. The service runs at
`Nice=15`, `CPUWeight=20`, `IOWeight=20`, and the build keeps the build policy above. The timer is
not persistent: a laptop asleep at the hour skips that day.

## Verifier contract

What `tools/release_qualification.py` and `tools/qualify-release.py` hold a tag record to. The
original long-form text is in the [history](history/RELEASE-QUALIFICATION.md#source-identity-and-receipt-publication).

### Source identity

- The source snapshot records the tested commit and tree and a SHA-256 over a canonical inventory
  of Git blob ids and modes for all tracked inputs, not just engine files. Changes to KV, tier,
  tokenizer, server, build scripts, dependency locks, tools, workflows, test fixtures or existing
  research data invalidate the record. Docs are tracked inputs too: a docs commit between the
  tested commit and the tag voids the tag record.
- There is no file-mtime comparison and no missing-model waiver. A new checkout verifies the same
  content identities.
- `research/release-qualification/` holds qualification metadata only, never runtime or build
  inputs, and is excluded from the inventory so recording a result creates no hash cycle.
  `research/INDEX.md` may only gain an append after the tested prefix.
- New research-only sidecars are reported as publication additions; they cannot replace or delete
  an existing input. Publication is an audited content-equivalence claim, not qualification of a
  newly rebuilt binary.
- `current.json` pins one or more `record.json` files by SHA-256. Each record seals its complete
  evidence manifest; its paths stay within its directory and cannot be symlinks. Banked records are
  read from immutable Git objects. CI passes `--fetch-source` to fetch a missing tested commit.

### Build inputs

- Producing a record compares tracked file bytes, executable modes and symlink targets against the
  expected Git objects before and after build and capture; `git status` flags such as
  assume-unchanged cannot hide a different input. Ignored or untracked files under crate, tool and
  Cargo source roots refuse. Symlinks resolve within the tracked closure and must not reach the
  metadata namespace or `research/INDEX.md`.
- The controlled v3 build compiles from a separate view of exactly the fingerprinted files, checked
  before and after compilation, under Linux bubblewrap: read-only `/source`, private root, `/proc`,
  `/dev` and `/tmp`, no GPU, no network, no host home. Dependencies are fetched with `--locked`
  before compilation, which runs `--offline`. Where AppArmor restricts user namespaces, the hosted
  jobs install a root-owned bwrap with the `userns` permission
  (`tools/install-release-sandbox-ci.sh`); other hosts supply a working bwrap. There is no
  unisolated retry.
- The build has its own config-free Cargo home: cached archives and index only, no inherited
  configuration or credentials. Compiler-wrapper, target and rustflag variables are removed; cargo,
  rustc and nvcc are hashed into the record; the only admitted project Cargo setting is the tracked
  `[build] jobs`. The record binds the sandbox executable, compiler environment and view identities
  (`controlled-cargo-v3`).

### Model inputs

Capture supports single-file GGUF inputs only. It parses `split.count` and
refuses a multi-file model before any GPU operation, even when an entry shard was renamed. This is
a qualification-tool limit, not a statement about engine format support; the Step gate keeps its
own artifact and topology checks. Oracle directories normalize once (absolute, relative or
symlinked); named file aliases stay under that directory. Capture never renames or substitutes a
model: the coordinator supplies the named weight oracles, whose complete bytes are hashed before
and after the run.
