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
