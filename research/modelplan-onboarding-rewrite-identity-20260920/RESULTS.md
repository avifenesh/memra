# Rewrite qualification identity — #542

Status: implementation and CPU regression evidence; native GPU qualification pending. Draft PR only.

Base: `b3487a03b0ee3f833c1157e7b7d68f2cb35a3843`.
Branch: `codex/542-trusted-rewrite-identity`.
Environment: macOS arm64, no CUDA toolkit or GPU. No serving machine was accessed.

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
