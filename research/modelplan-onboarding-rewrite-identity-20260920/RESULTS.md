# Rewrite qualification identity — #542

Status: review corrections after `662774b4` change the runtime and require fresh native
qualification. The historical single-card admission gate passed at
`62113383d63a0bac510279b954080c592c156630`; that evidence does not qualify the changed
binary. Whole-model, serving, MTP/pipeline, and performance qualification are not promoted.

Base: `b3487a03b0ee3f833c1157e7b7d68f2cb35a3843`.
Branch: `codex/542-trusted-rewrite-identity`.
Environment: macOS arm64, no CUDA toolkit or GPU. No serving machine was accessed.

## Retained re-entry correction (native rerun pending)

SEC-557-2 / PERF-557-2 is fixed in the shared activation primitive: every outermost
`enter_rewrite_execution` validates current libraries/environment before activating
retained permissions. Nested same-generation calls reuse validation. Cached emitters
hold this guard; original model/generation bindings are never replaced.

The exact frozen reviewer environment test failed as reported. The same environment
and injected file-inventory assertions fail against the frozen activation body and
pass against the corrected body; `later-drift-20260920/sealed/comparison.json` records
the source and raw log hashes and the minimal old-signature adapter. There are 18
identity/snapshot host tests. The continuous 10,000-token CPU control validates once;
a standalone re-entry after scope exit validates again. No whole-token latency claim
is inferred from that nested-call result.

Previous-head CI failed because isolated repack children exported filtered libtest
summaries into the outer unfiltered skip census. Child output is now captured, with
an explicit one-test/non-vacuity assertion and complete stdout/stderr on failure.
The unchanged census passes both host suites with zero skips and zero filtered tests.
Its original failure log is retained alongside the correction.

A separate native `library-drift` mode is implemented and scheduled by the runner,
with a real never-executed executable mapping and cache-immutability assertions.
It has not run on native hardware. Qualified graph/prime/worker cases and safe native
later-environment mutation remain explicit planned cases in `NATIVE-REFUSAL-PLAN.md`.
All earlier native evidence remains historical.

This correction passes engine/server library, binary and test cross-target clippy.
The zero-budget skip census reports 18 identity/snapshot and nine repack host tests,
with zero failed, skipped or filtered tests. Runner validation passed 27 controls
in its full run; its case-count assertion caught the newly scheduled twelfth case.
After updating that expectation and asserting the library-drift case is present,
the remaining control passes on a focused rerun. This does not execute native cases.

## Review corrections (native rerun pending)

SEC-557-1: both native NVFP4 disk loaders regenerate from the opened source and
verify/repair the named cache in identity mode. Live bytes use a separate unlinked
read-only backing. Nine CPU regressions pass, including cold, hit, same-size corruption,
after-load mutation/truncation and opened-source replacement for both layouts and flags.
Strict loads pay additional regeneration I/O and private disk capacity, with bounded RAM.

PERF-557-1: private tracked model ownership revokes identity on mutable access. Every
reinstall revokes retained snapshots. Request/session snapshots validate external state
at new/resumed request boundaries; protected token checks use a generation and surface
mask, including nested and pipeline calls. The 15-test host harness passes, including
10,000 repeated protected admissions with one inventory callback and no program
serialization. This is control-flow evidence, not a latency or throughput result.

The native gate adds protected eager replay plus retained-snapshot reinstall and
same-shape mutation refusals. These new native controls have not run yet. Existing
native records below are historical and unchanged.

TC557-01: qualification now requires a successful record emitted by the owned native
builder, with a fresh build directory, clean source contents, toolchain/config inputs,
Cargo completion events and actual executable hashes. Preflight and case boundaries
reject stale source, stale binaries and missing/incomplete records. The builder is an
auditable local provenance record, not a signature or hermetic toolchain attestation.
Historical manually captured build evidence is preserved; it is not retroactively
converted into an owned build record.

Previous candidate CPU preparation: the 15 identity/snapshot tests, nine native-repack helper
tests and 28 runner/build-provenance tests pass. `cargo test -p memra-gguf -p memra-cli`
passes. The combined tree passes engine/server library, binary and test type checking
with `DOCS_RS=1 MEMRA_MMQ_ARCHIVE_HASH=host-check-placeholder cargo clippy --target
x86_64-unknown-linux-gnu -p memra-engine --lib --bins --tests -p memra-server -- -D warnings`.
That command is compile-only on macOS, not native CUDA execution. Formatting, whitespace,
flags census and generated-board checks pass. Independent follow-up review closed
public interior-cache mutation and retained prime/MTP/GLM provenance gaps. Those retained
GPU objects still need native stale-object/replay controls on the rebuilt binary.

## Historical native result

On one exclusively locked RTX PRO6000 Blackwell Server Edition (driver595.58.03,
CUDA13.1, XFS/Ceph-RBD-backed scratch), attempt007 passed all11 operational cases with a
fresh private CUDA cache. Independent quantized-cache verify-prefill and tokenwise eager
outputs passed unchanged tolerances on3 prompts. Installed eager output and two fresh-process
replays were bit-identical, including replay after the separate fresh-KV diagnostic.

A matching bundle installed; missing bundle, changed checkpoint bytes at identical geometry,
changed executable bytes, and changed numerical settings refused for their named reasons.
Failed reinstall revoked eager access without changing any cache plane. Eager-only receipts
refused graph execution and fresh-KV `forward`/`forward_last`. Fresh-KV outputs were produced
only in a separate unqualified process, with no qualification receipt emitted. Standing
`run-gen` and `decode-batch-gate --mode config --batch 2 --steps 16` also passed.

The apparent lazy-library failure was caused by mixing the unqualified fresh-KV diagnostic
into cached-eager capture. Phase-boundary probes established that cached execution did not
invalidate identity. That diagnostic was isolated; the driver-link/head/scratch initialization
experiments were removed from production. No model mathematics or tolerance changed, and
all later library/plan/environment/tensor-program drift checks remain active.

Authoritative final records: `native-server-20260920/server-admission-007/`, its finished
`server-lease-007/lease.json`, and `server-preflight-009/`. Wrapper/child exit0, no timeout,
no interruption, no lingering compute. Exact tested ELF bytes are retained locally under
`local-artifacts/`; `native-server-20260920/tested-binary-preservation.json` seals the archive
and all four executable hashes. The gate executable SHA-256 is
`5eef5b6324578f455ff9b8139d832518e372f6fc88100fada0ce6cbf1a1bef79`.

Historical attempts below are preserved chronologically; their failures are not relabeled
as passes. No full-power/Max-Q performance transfer, NVMe claim, or model/serving promotion
is inferred from this admission test. No merge, tag, or deployment was performed.

## Change

V2 receipts bind the actual opened checkpoint bytes, running executable, loaded numerical
program, and compiled plan. Numerical identity includes selected loaded tensor layouts,
environment, device/driver details, and the bytes of file-backed executable library mappings.
Mapped library replacement or loaded program mutation invalidates the captured identity. Bundle lock/index hashes remain integrity checks. Strict mode is
installed by the common HybridModel loader; missing eager qualification or a failed installation
leaves admission closed. Unbundled execution is explicitly legacy and carries no qualification.
GGUF and safetensors have opened-byte identities; incomplete composite artifact sources refuse
strict admission. DSv4 strict admission explicitly refuses pending #449/#504 coverage.

The CLI shares strict receipt parsing with runtime installation. Imported evidence keeps
`RewriteParity=pending` until independent runtime admission; internally consistent supplied
identities do not promote qualification. Synthetic ModelPlan fixture
parity remains diagnostic evidence and cannot issue checkpoint qualification receipts.

## CPU evidence

- `raw/fail-before.log`: the new regression fails on the baseline because an internally
  consistent bundle with arbitrary implementation and unrelated artifact label is accepted.
- `raw/cpu-review-fixes.log`: 288 gguf tests pass (one pre-existing ignored), 12 CLI tests pass. `cargo test -p memra-gguf -p memra-cli --lib`, including correct/missing/
  stale/mismatched identities, duplicate fields, malformed index, and real opened-weight
  replacement with identical geometry. Source tests cover all shards and retained mappings.
- `raw/runtime-host-tests-final.log`: 11 CPU tests exercise the actual runtime identity/admission
  module through a small host-only harness, including failed-reinstall revocation and direct
  output admission. Engine library tests include these same cases on Linux/CUDA build hosts.
- `raw/runtime-typecheck-final.log`: documentation-only Linux cross-target engine/server
  type check passes; placeholder fatbins are used, no CUDA compile/execution is claimed.
  Command: `DOCS_RS=1 MEMRA_MMQ_ARCHIVE_HASH=host-check-placeholder cargo check
  --target x86_64-unknown-linux-gnu -p memra-engine --lib --bins -p memra-server`.
- `raw/host-harness-reproduced.log`: reproduction via `python3
  research/modelplan-onboarding-rewrite-identity-20260920/run-host-tests.py`.
- `raw/runtime-clippy.log`: documentation-only Linux engine/server clippy with warnings denied
  passes at the published patch. This uses the same placeholder settings as the type check.
- `raw/cpu-clippy.log`: Linux cross-target gguf/CLI clippy with warnings denied passes.
- `raw/cli-typecheck.log`: cross-target Linux CLI all-target type check passes.
- `raw/cpu-suites.log`: broader CPU sweep includes the reference executor: 65 pass, one existing
  Qwen3.5 bitwise golden mismatch. `raw/reference-baseline-fresh.log` reproduces the identical
  values from pristine baseline source in a fresh target directory. This is #548, outside #542.
  The earlier `raw/reference-baseline.log` reused a build cache; the fresh control supersedes it.
- `raw/fmt-check.log` records the intermediate formatting diff; `raw/fmt-final.log` is the
  final formatting check. Diff, flags, docs registry, and generated perf-board checks are run
  before push; no published performance number changes.

## Outstanding qualification

Before integration, run native engine/server suites and the required kernel-check, affected
run-gen argmax, run-spec K=1..8, and serving-shape gates on a designated non-serving rig under
its existing lock. Record exact source, executable, loaded artifact, numerical program and
raw logs. CPU tests and documentation-only cross-target type checks are not CUDA qualification.
No model support state or hardware default is promoted. No merge, tag, or deployment is made.

Strict numerical-program revalidation deliberately checks mutable loaded program state.
Its overhead and supported serving shapes require native qualification before integration.

The tested Rust source content is recorded in `source-sha256.json`.

## Native runner preparation (2026-09-20)

The owner authorized centrally provisioned non-serving GPUs with per-card exclusive locks.
`NATIVE-RUNNER.md` records the exact one-card resource request, immutable source pin, build
command, and central-wrapper launch. Max-Q 96 GB/sm_120 is suitable for this correctness and
memory stage; measurements remain specific to the measured hardware.

`rewrite_identity_gate` typechecks and passes clippy with warnings denied using the documented
Linux/DOCS_RS placeholders; this is preparation only, not GPU qualification. The wrapper
validation/output-completeness suite passes five CPU tests. The runtime identity suite passes
11 CPU tests after excluding the lease receipt location from the numerical digest. Raw logs
are `raw/rewrite-identity-gate-{typecheck,clippy}.log`, `raw/lease-runner-tests.log`, and
`raw/lease-metadata-host-tests.log`.

The runner refuses absent/foreign/partial GPU ownership, revalidates the ancestor FLOCKs while
each child runs, compares fresh-process output hashes, retains named negative-control errors,
and records 250 ms telemetry. GPU execution waits for the coordinator's supplied host/wrapper.
Broader q9 MTP, paired pipeline, and full serving/performance rows remain pending their exact
artifacts and separately assigned lock sets. No separate machine has been rented by this lane.

## Interruption controls

CPU preflight attempt001 and recovered staging evidence are retained under
`native-maxq-20260920/`. The initial build was interrupted; no GPU result was produced.
A local regression found that the runner accepted a signal-killed negative control when
its partial log already contained the expected refusal. The runner now requires orderly
exit1 plus the native gate failure marker and exact identity-refusal phrase, and revalidates
the lease after child exit. `raw/negative-interruption-fail-before.log` reproduces the old
false pass; `raw/negative-interruption-pass-after.log` records six passing CPU tests.

## Replacement CPU preparation

The replacement checkout was first verified at73324210 with GPU visibility empty. Its
Python interpreters are3.10, so the runner's Python3.11-only file_digest helper was replaced
with equivalent streaming SHA256. Empty and binary/chunked payload controls confirm the
same hashes without that API. Seven CPU controls pass; fail-before/pass-after logs are
`raw/python310-digest-{fail-before,pass-after}.log`. No system libraries were changed,
and CUDA compilation/execution waits for coordinator acceptance. The replacement storage
is XFS on Ceph RBD, not proven physical local NVMe; this lane's admission checks remain
eligible, with no NVMe or spill-performance claim.

## Native numerical-class discovery

`native-server-20260920/` contains the first real single-card attempt and standing control.
The initial forward_last/tokenwise comparison failed atmax_abs3.1681318. The standing
quantized-cache verify-prefill control passed at1.907e-6. Fresh-F32 KV and cached KV are
different programs, so the gate was corrected without changing mathematics or tolerances:
`forward-fresh-kv` has a distinct manifest/receipt, and eager-only admission refuses fresh-KV
execution. The cached-KV gate compares independent verify-prefill and tokenwise executions.
CPU compiler/CLI suites pass289+12 tests, runtime host suite passes11, and documentation-only
engine/server cross-target clippy/type checks pass. These are preparation for a fresh native
attempt, not a promotion of model or serving support.

Native attempt002 confirms the corrected numeric-class comparison on all3 prompts, but
receipt binding fails closed on stale identity. Complete vectors and logs are preserved.
Diagnostic validation now identifies the stale component without disclosing environment
values or relaxing any identity comparison. Host tests and Linux cross-target test typechecks
pass; next native run will localize the failure.

Attempt003 localized the stale identity to lazily loaded NVIDIA PTX/compiler libraries.
The loader now materializes the driver JIT path before freezing identity, without a kernel
launch, model arithmetic change, environment override, or removal of library validation.
The initialization module has explicit model ownership. This correction is pending a fresh
native attempt; the failing row is retained.
